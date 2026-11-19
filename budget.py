"""budget.py - make a fixed monthly request quota last the whole month, jackpot first.

Provider-agnostic: works for any odds API with a monthly quota (OddsPapi free 250, The Odds API free 500).
Rules:
  1. A safety reserve is never touched.
  2. The Saturday jackpot gets its calls held back for every round still to come this month.
  3. Daily matches share what is left evenly over the remaining days; unused allowance rolls forward.
  4. Every call is logged in the api_usage table, so the budget survives restarts.
Our own count is the floor; if the provider's dashboard figure is entered, the tighter of the two wins.
"""
import calendar
import math
from datetime import timedelta

import engine as E
import store


class BudgetError(Exception):
    pass


def month_used(db, provider, now):
    r = db.q("SELECT COALESCE(SUM(cost),0) c FROM api_usage WHERE provider=? AND substr(ts,1,7)=?",
             (provider, now.strftime("%Y-%m")))
    return int(r["c"][0])


def last_remaining(db, provider, now):
    """Latest 'remaining' figure the provider (or the user, from the dashboard) reported this month."""
    r = db.q("SELECT remaining FROM api_usage WHERE provider=? AND remaining IS NOT NULL AND substr(ts,1,7)=? "
             "ORDER BY id DESC LIMIT 1", (provider, now.strftime("%Y-%m")))
    return int(r["remaining"][0]) if len(r) else None


def saturdays_left(now):
    """Jackpot rounds whose deadline (Saturday 17:00) falls in this month and has not passed."""
    last = calendar.monthrange(now.year, now.month)[1]
    n = 0
    for d in range(now.day, last + 1):
        if now.replace(day=d).weekday() == 5:
            if d == now.day and now.hour >= 17:
                continue
            n += 1
    return n


def _used_by(db, provider, now, since_iso, like):
    r = db.q("SELECT COALESCE(SUM(cost),0) c FROM api_usage WHERE provider=? AND ts>=? AND note LIKE ?",
             (provider, since_iso, like))
    return int(r["c"][0])


def plan(db, provider, total, now=None, reserve=30, jackpot_per_round=3):
    now = now or store.now_local()
    used = month_used(db, provider, now)
    usable = max(total - reserve, 0)
    left = max(usable - used, 0)
    rem = last_remaining(db, provider, now)
    if rem is not None:
        left = min(left, max(rem - reserve, 0))
    sats = saturdays_left(now)
    round_start = E.next_deadline(now) - timedelta(days=7)
    jp_used_round = _used_by(db, provider, now, store.iso(round_start), "jackpot:%")
    hold = 0
    if sats:
        hold = (sats - 1) * jackpot_per_round + max(jackpot_per_round - jp_used_round, 0)
    hold = min(hold, left)
    pool = left - hold
    days_left = max(1, calendar.monthrange(now.year, now.month)[1] - now.day + 1)
    day_start = store.iso(now.replace(hour=0, minute=0, second=0, microsecond=0))
    daily_today = _used_by(db, provider, now, day_start, "daily:%")
    today_limit = math.ceil(pool / days_left) if pool > 0 else 0
    return {"total": total, "reserve": reserve, "used": used, "left": left, "jackpot_hold": hold,
            "daily_pool": pool, "days_left": days_left, "daily_allowance": pool / days_left,
            "daily_today_used": daily_today, "daily_today_left": max(today_limit - daily_today, 0),
            "rounds_left": sats, "provider_remaining": rem}


def allow(db, provider, purpose, cost, total, now=None, reserve=30, jackpot_per_round=3):
    """Raise BudgetError unless this call fits. purpose is 'jackpot' or 'daily'."""
    p = plan(db, provider, total, now, reserve, jackpot_per_round)
    if purpose == "jackpot":
        if cost > p["left"]:
            raise BudgetError(f"No requests left this month ({p['left']} usable of {total}, reserve {reserve}).")
    else:
        if cost > p["daily_pool"]:
            raise BudgetError(f"Daily pool is empty: {p['left']} left, {p['jackpot_hold']} held for the jackpot.")
        if cost > p["daily_today_left"]:
            raise BudgetError(f"Today's allowance is used (about {p['daily_allowance']:.1f} a day). Try tomorrow, "
                              "or type reference odds into the Ref columns for free.")
    return p


def log(db, provider, purpose, key, cost, now, remaining=None):
    db.insert("api_usage", {"ts": store.iso(now), "provider": provider, "cost": int(cost),
                            "remaining": remaining, "note": f"{purpose}:{key}"})


def true_up(db, provider, remaining, now=None):
    """Record the figure shown on the provider's dashboard (cost 0). Keeps our count honest."""
    now = now or store.now_local()
    db.insert("api_usage", {"ts": store.iso(now), "provider": provider, "cost": 0,
                            "remaining": int(remaining), "note": "true-up:dashboard"})
