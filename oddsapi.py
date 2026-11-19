"""oddsapi.py - optional sharp-odds reference from The Odds API (free key, 500 credits per month).

Verified from third-party documentation (NOT from a live call in the build environment):
  GET https://api.the-odds-api.com/v4/sports                      (free, no quota cost)
  GET https://api.the-odds-api.com/v4/sports/{sport}/odds?apiKey=&regions=eu&markets=h2h&oddsFormat=decimal
  cost = markets x regions per call; quota in headers x-requests-remaining / -used / -last.
A hard monthly budget is kept in the database so the free limit cannot be exceeded by accident.
"""
import difflib
import json
import re
from datetime import datetime, timedelta

import numpy as np
import requests

import engine as E
import store
import budget

BASE = "https://api.the-odds-api.com/v4"
DEFAULT_MONTHLY_CAP = 450      # leaves a safety margin under the 500 free credits
RESERVE = 10                   # never go below this many credits remaining
CACHE_TTL_HOURS = 6            # fetched odds are reused (free) for this long

# football-data.co.uk league code -> The Odds API sport key. These are best guesses from memory:
# they are only ever used if the key appears in the live (free) /sports list, otherwise ignored.
SPORT_KEY_GUESS = {
    "E0": "soccer_epl", "E1": "soccer_efl_champ", "E2": "soccer_england_league1", "E3": "soccer_england_league2",
    "SC0": "soccer_spl", "D1": "soccer_germany_bundesliga", "D2": "soccer_germany_bundesliga2",
    "I1": "soccer_italy_serie_a", "I2": "soccer_italy_serie_b", "SP1": "soccer_spain_la_liga",
    "SP2": "soccer_spain_segunda_division", "F1": "soccer_france_ligue_one", "F2": "soccer_france_ligue_two",
    "N1": "soccer_netherlands_eredivisie", "B1": "soccer_belgium_first_div", "P1": "soccer_portugal_primeira_liga",
    "T1": "soccer_turkey_super_league", "G1": "soccer_greece_super_league",
}


BudgetError = budget.BudgetError   # one error type for every provider
PROVIDER = "odds-api"
QUOTA_FREE = 500


def credits_used(db, now=None):
    now = now or store.now_local()
    r = db.q("SELECT COALESCE(SUM(cost),0) c FROM api_usage WHERE provider='odds-api' AND substr(ts,1,7)=?",
             (now.strftime("%Y-%m"),))
    return int(r["c"][0])


def check_budget(db, cost, cap, now=None):
    used = credits_used(db, now)
    if used + cost > cap:
        raise BudgetError(f"Monthly budget would be exceeded: {used} used + {cost} > cap {cap}.")
    last = db.q("SELECT remaining FROM api_usage WHERE provider='odds-api' AND remaining IS NOT NULL "
                "ORDER BY id DESC LIMIT 1")
    if len(last) and int(last["remaining"][0]) - cost < RESERVE:
        raise BudgetError("API reports too few credits remaining.")


def redact(text):
    """Hide API keys that requests puts inside error messages (it prints the full URL, query string included)."""
    return re.sub(r"(?i)(api[_-]?key=)[^&\s'\")]+", r"\1***", str(text))


def call(http_get, url, params, timeout):
    """GET with errors that never contain the API key."""
    try:
        r = http_get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        raise RuntimeError(redact(e)) from None


def list_soccer(api_key, http_get=requests.get, db=None, now=None):
    """Free call. Returns [(key, title)] for in-season soccer competitions.
    If the reply carries the provider's own 'requests remaining' figure, it is recorded (cost 0), so our month counter
    stays honest even after the local database was reset (for example by a cloud restart)."""
    r = call(http_get, f"{BASE}/sports", {"apiKey": api_key}, 20)
    rem = (getattr(r, "headers", None) or {}).get("x-requests-remaining")
    if db is not None and rem is not None:
        try:
            budget.true_up(db, PROVIDER, int(rem), now)
        except (TypeError, ValueError):
            pass
    return [(s["key"], s.get("title", s["key"])) for s in r.json() if str(s.get("key", "")).startswith("soccer")]


def fetch_h2h(db, api_key, sport_key, cap=DEFAULT_MONTHLY_CAP, regions="eu", now=None, http_get=requests.get,
              purpose=None, total=QUOTA_FREE, reserve=50):
    """purpose='daily'|'jackpot' uses the month planner (jackpot calls are ring-fenced); None keeps the plain cap."""
    now = now or store.now_local()
    cost = len(regions.split(","))
    if purpose:
        budget.allow(db, PROVIDER, purpose, cost, total, now, reserve)
    else:
        check_budget(db, cost, cap, now)
    r = call(http_get, f"{BASE}/sports/{sport_key}/odds",
             {"apiKey": api_key, "regions": regions, "markets": "h2h", "oddsFormat": "decimal"}, 20)
    h = r.headers
    last = int(h.get("x-requests-last", cost))
    rem = h.get("x-requests-remaining")
    db.insert("api_usage", {"ts": store.iso(now), "provider": "odds-api", "cost": last,
                            "remaining": int(rem) if rem is not None else None,
                            "note": f"{purpose}:{sport_key}" if purpose else sport_key})
    return r.json()


def cache_get(db, sport_key, now, ttl_hours=CACHE_TTL_HOURS):
    """Fresh cached events for a competition, or None. Costs nothing."""
    r = db.q("SELECT fetched, payload FROM odds_cache WHERE sport_key=? ORDER BY id DESC LIMIT 1", (sport_key,))
    if not len(r):
        return None
    age = (now - datetime.fromisoformat(r["fetched"][0])).total_seconds() / 3600
    if age > ttl_hours:
        return None
    return json.loads(r["payload"][0]), age


def fetch_h2h_cached(db, api_key, sport_key, cap=DEFAULT_MONTHLY_CAP, ttl_hours=CACHE_TTL_HOURS, regions="eu",
                     now=None, http_get=requests.get, force=False, purpose=None, total=QUOTA_FREE, reserve=50):
    """Returns (events, 'cache' or 'api', age_hours). Only the 'api' path spends credits."""
    now = now or store.now_local()
    if not force:
        hit = cache_get(db, sport_key, now, ttl_hours)
        if hit:
            return hit[0], "cache", hit[1]
    events = fetch_h2h(db, api_key, sport_key, cap=cap, regions=regions, now=now, http_get=http_get,
                       purpose=purpose, total=total, reserve=reserve)
    db.insert("odds_cache", {"sport_key": sport_key, "fetched": store.iso(now), "payload": json.dumps(events)})
    return events, "api", 0.0


def _upcoming(ev, now):
    try:
        t = datetime.fromisoformat(str(ev["commence_time"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return True
    return t >= now - timedelta(hours=3)


def cached_events(db, now=None, ttl_hours=CACHE_TTL_HOURS):
    """All fresh cached events that have not started yet. Costs nothing."""
    now = now or store.now_local()
    out = []
    for k in db.q("SELECT DISTINCT sport_key FROM odds_cache")["sport_key"].tolist():
        hit = cache_get(db, k, now, ttl_hours)
        if hit:
            out += [e for e in hit[0] if _upcoming(e, now)]
    return out


def cache_summary(db, now=None, ttl_hours=CACHE_TTL_HOURS):
    now = now or store.now_local()
    rows = []
    for k in db.q("SELECT DISTINCT sport_key FROM odds_cache")["sport_key"].tolist():
        hit = cache_get(db, k, now, ttl_hours)
        if hit:
            rows.append({"Competition": k, "Age (hours)": round(hit[1], 1), "Upcoming events": len(hit[0])})
    return rows


def competitions_for(results, available_keys):
    """Odds API competitions covering the given analysed matches (only keys the live list confirms)."""
    keys = []
    for r in results:
        k = SPORT_KEY_GUESS.get(r.get("league"))
        if k and k in available_keys and k not in keys:
            keys.append(k)
    return keys


def event_anchor(event, prefer=("pinnacle",)):
    """De-vigged home/draw/away probabilities for one event: Pinnacle if present, else median over books."""
    per_book = {}
    for b in event.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") != "h2h":
                continue
            price = {o["name"]: o["price"] for o in m.get("outcomes", [])}
            try:
                inv = np.array([1 / price[event["home_team"]], 1 / price["Draw"], 1 / price[event["away_team"]]])
            except (KeyError, ZeroDivisionError):
                continue
            per_book[b["key"]] = inv / inv.sum()
    if not per_book:
        return None
    for k in prefer:
        if k in per_book:
            return {"probs": per_book[k], "source": k, "n_books": len(per_book)}
    arr = np.array(list(per_book.values()))
    med = np.median(arr, axis=0)
    return {"probs": med / med.sum(), "source": "median of books", "n_books": len(per_book)}


def _names(name):
    n = E.norm(name)
    alias = E.ALIASES.get(n)
    return {n} | ({E.norm(alias)} if alias else set())


def _sim(a, b):
    return max(difflib.SequenceMatcher(None, x, y).ratio() for x in _names(a) for y in _names(b))


def find_event(home, away, events, cutoff=0.78):
    best, best_s = None, 0.0
    for ev in events:
        s = min(_sim(home, ev["home_team"]), _sim(away, ev["away_team"]))
        if s > best_s:
            best, best_s = ev, s
    return best if best_s >= cutoff else None


def build_anchors(rows, events):
    """rows: iterable of (Home, Away). Returns {(norm home, norm away): anchor dict}."""
    out = {}
    for h, a in rows:
        ev = find_event(h, a, events)
        if ev:
            an = event_anchor(ev)
            if an:
                an["event"] = f"{ev['home_team']} v {ev['away_team']}"
                out[(E.norm(h), E.norm(a))] = an
    return out
