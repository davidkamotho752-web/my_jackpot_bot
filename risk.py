"""risk.py - bankroll protection: limits, stops, PAPER/LIVE gate, ruin simulator, proof gate.

Everything here exists to keep the account alive. Money units are wallet KES (what SportyBet shows).
"""
import math
from datetime import timedelta

import numpy as np
import pandas as pd

import store

DEFAULT_LIMITS = {
    "kelly_frac": 0.125,        # fraction of full Kelly staked
    "max_stake_pct": 0.01,      # max stake per bet, share of bankroll
    "max_bets_day": 3,
    "max_risk_day_pct": 0.03,   # total staked per day, share of bankroll
    "daily_loss_pct": 0.03,
    "weekly_loss_pct": 0.08,
    "drawdown_pct": 0.20,       # from peak: LIVE switches back to PAPER
    "jackpot_month_pct": 0.05,  # jackpot spend per month, share of bankroll
    "min_edge": 0.03,           # minimum net edge on the LOWER-bound probability
    "max_odds": 4.0,            # skip long shots (model error is largest there)
    "min_stake": 10.0,
    "z": 1.0,                   # how many uncertainty-sds to subtract from the probability
}
# +1: a larger value is LOOSER (riskier). -1: a smaller value is looser. 0: not a risk limit.
LOOSER = {"kelly_frac": 1, "max_stake_pct": 1, "max_bets_day": 1, "max_risk_day_pct": 1,
          "daily_loss_pct": 1, "weekly_loss_pct": 1, "drawdown_pct": 1, "jackpot_month_pct": 1,
          "min_edge": -1, "max_odds": 1, "min_stake": 0, "z": -1}
LOOSEN_WAIT_DAYS = 7

DEFAULT_TAX = {"deposit": 0.05, "withdrawal": 0.05, "jackpot_withholding": 0.0, "per_bet": False}

GATE_MIN_PREDICTIONS = 300
GATE_MIN_BETS = 300


# ----------------------------------------------------------------------------- limits and settings
def get_limits(db, now=None):
    now = now or store.now_local()
    cur = {**DEFAULT_LIMITS, **db.get("limits", {})}
    pending = db.get("pending_limits", {})
    changed = False
    for k, p in list(pending.items()):
        if now >= store.datetime.fromisoformat(p["effective_at"]):
            cur[k] = p["value"]
            del pending[k]
            changed = True
            db.event("limit_loosened", f"{k} -> {p['value']} (7-day wait over)", now)
    if changed:
        db.set("limits", cur)
        db.set("pending_limits", pending)
    return cur


def propose_limits(db, new, now=None):
    """Tighter values apply immediately. Looser values wait 7 days. Returns (applied, pending) dicts."""
    now = now or store.now_local()
    cur = get_limits(db, now)
    pending = db.get("pending_limits", {})
    applied, queued = {}, {}
    for k, v in new.items():
        if k not in DEFAULT_LIMITS or v == cur[k]:
            pending.pop(k, None) if v == cur.get(k) else None
            continue
        direction = LOOSER[k]
        looser = (v > cur[k] and direction > 0) or (v < cur[k] and direction < 0)
        if looser:
            eff = now + timedelta(days=LOOSEN_WAIT_DAYS)
            pending[k] = {"value": v, "effective_at": store.iso(eff)}
            queued[k] = pending[k]
            db.event("limit_loosen_requested", f"{k}: {cur[k]} -> {v}, effective {store.iso(eff)}", now)
        else:
            cur[k] = v
            pending.pop(k, None)
            applied[k] = v
            db.event("limit_tightened", f"{k} -> {v}", now)
    db.set("limits", cur)
    db.set("pending_limits", pending)
    return applied, queued


def get_tax(db):
    return {**DEFAULT_TAX, **db.get("tax", {})}


def get_mode(db):
    return db.get("mode", "PAPER")


def set_mode(db, mode, gate=None, now=None):
    """Switch mode. LIVE needs a passing proof gate."""
    if mode == "LIVE" and not (gate and gate.get("passed")):
        raise ValueError("LIVE mode is locked: the proof gate has not passed.")
    old = get_mode(db)
    db.set("mode", mode)
    if old != mode:
        db.event("mode_change", f"{old} -> {mode}", now)


# ----------------------------------------------------------------------------- bankroll
def add_ledger(db, mode, kind, amount, note="", now=None, bet_id=None):
    return db.insert("bankroll", {"ts": store.iso(now or store.now_local()), "mode": mode, "kind": kind,
                                  "amount": float(amount), "note": note, "bet_id": bet_id})


def bankroll_state(db, mode, now=None):
    now = now or store.now_local()
    led = db.q("SELECT * FROM bankroll WHERE mode=? ORDER BY id", (mode,))
    bal = peak = 0.0
    curve = []
    for r in led.itertuples():
        bal += r.amount
        if r.kind in ("deposit", "withdrawal"):
            peak += r.amount          # money moved in or out is not performance
        else:
            peak = max(peak, bal)
        curve.append((r.ts, bal))
    dd = (peak - bal) / peak if peak > 0 else 0.0
    today = now.date().isoformat()
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    if len(led):
        day_rows = led[led["ts"].str[:10] == today]
        wk_rows = led[pd.to_datetime(led["ts"], utc=True) >= week_start.astimezone(store.ZoneInfo("UTC"))]
    else:
        day_rows = wk_rows = led
    # jackpot tickets and prizes have their own monthly budget, so they do not trigger the daily/weekly stops
    perf = lambda d: (float(d[~d["kind"].isin(["deposit", "withdrawal", "ticket_cost", "prize"])]["amount"].sum())
                      if len(d) else 0.0)
    day_pnl, week_pnl = perf(day_rows), perf(wk_rows)
    day_start = bal - (float(day_rows["amount"].sum()) if len(day_rows) else 0.0)
    week_start_bal = bal - (float(wk_rows["amount"].sum()) if len(wk_rows) else 0.0)
    pend = db.q("SELECT COALESCE(SUM(stake),0) s FROM bets WHERE mode=? AND decision='BET' AND status='Pending'", (mode,))
    today_b = db.q("SELECT COUNT(*) n, COALESCE(SUM(stake),0) s FROM bets WHERE mode=? AND decision='BET' "
                   "AND substr(created,1,10)=?", (mode, today))
    month = now.strftime("%Y-%m")
    jp = db.q("SELECT COALESCE(SUM(cost),0) c FROM tickets WHERE mode=? AND substr(created,1,7)=?", (mode, month))
    pending = float(pend["s"][0])
    return {"balance": bal, "peak": peak, "drawdown": dd, "pending": pending, "available": bal - pending,
            "day_pnl": day_pnl, "week_pnl": week_pnl, "day_start": day_start, "week_start": week_start_bal,
            "bets_today": int(today_b["n"][0]), "risk_today": float(today_b["s"][0]),
            "jackpot_month": float(jp["c"][0]), "curve": curve}


def guard_status(state, lim):
    """Which stop rules are currently triggered."""
    reasons = []
    if state["balance"] <= 0:
        reasons.append("No bankroll recorded yet (log a deposit in the Bankroll tab).")
    if state["day_start"] > 0 and -state["day_pnl"] >= lim["daily_loss_pct"] * state["day_start"]:
        reasons.append("Daily loss stop reached.")
    if state["week_start"] > 0 and -state["week_pnl"] >= lim["weekly_loss_pct"] * state["week_start"]:
        reasons.append("Weekly loss stop reached.")
    if state["drawdown"] >= lim["drawdown_pct"]:
        reasons.append("Drawdown stop reached.")
    return {"stopped": bool(reasons), "reasons": reasons}


def auto_drawdown_switch(db, now=None):
    """If LIVE and the drawdown stop is hit, drop back to PAPER and log it."""
    now = now or store.now_local()
    if get_mode(db) != "LIVE":
        return False
    st = bankroll_state(db, "LIVE", now)
    lim = get_limits(db, now)
    if st["drawdown"] >= lim["drawdown_pct"]:
        set_mode(db, "PAPER", now=now)
        db.event("drawdown_stop", f"drawdown {st['drawdown']:.1%} >= {lim['drawdown_pct']:.0%}: switched to PAPER", now)
        return True
    return False


def apply_guards(results, state, lim):
    """Enforce stops and daily caps on the list of analysed matches (mutates and returns it)."""
    gs = guard_status(state, lim)
    bets = sorted([r for r in results if r["decision"] == "BET"], key=lambda r: -r["edge_low"])
    bets_left = lim["max_bets_day"] - state["bets_today"]
    risk_left = lim["max_risk_day_pct"] * state["balance"] - state["risk_today"]
    avail = state["available"]
    for r in bets:
        if gs["stopped"]:
            r["decision"], r["reason"], r["stake"] = "SKIP", "Stop rule active: " + gs["reasons"][0], 0.0
            continue
        if bets_left <= 0:
            r["decision"], r["reason"], r["stake"] = "SKIP", "Daily bet limit reached.", 0.0
            continue
        stake = min(r["stake"], max(risk_left, 0.0), max(avail, 0.0))
        stake = math.floor(stake / 10) * 10
        if stake < lim["min_stake"]:
            r["decision"], r["reason"], r["stake"] = "SKIP", "Daily risk cap reached.", 0.0
            continue
        r["stake"] = float(stake)
        bets_left -= 1
        risk_left -= stake
        avail -= stake
    return results


# ----------------------------------------------------------------------------- simulation
def ruin_simulation(edge, odds, stake_pct, n_bets, dd_stop=0.20, n_sims=4000, seed=0):
    """Monte Carlo: bankroll paths with a fixed stake fraction and a TRUE net edge per bet.
    Paths that hit the drawdown stop stop betting (the tool would switch to PAPER)."""
    rng = np.random.default_rng(seed)
    p = min(max((1 + edge) / odds, 0.0), 1.0)
    bal = np.ones(n_sims)
    peak = bal.copy()
    stopped = np.zeros(n_sims, bool)
    for _ in range(n_bets):
        stake = bal * stake_pct * (~stopped)
        win = rng.random(n_sims) < p
        bal = bal + np.where(win, stake * (odds - 1), -stake)
        peak = np.maximum(peak, bal)
        stopped |= (peak - bal) / peak >= dd_stop
    return {"p_stop": float(stopped.mean()), "median_final": float(np.median(bal)),
            "p_below_start": float((bal < 1).mean())}


# ----------------------------------------------------------------------------- proof gate
LABEL_IDX = {"1": 0, "X": 1, "2": 2}


def proof_gate(db, now=None, n_boot=2000, seed=0):
    """Checks that must ALL pass before LIVE mode unlocks. Returns {'passed', 'checks'}."""
    preds = db.q("SELECT * FROM predictions WHERE source='daily' AND result IN ('1','X','2') AND pD1 IS NOT NULL")
    checks = []
    n = len(preds)
    checks.append({"name": f"At least {GATE_MIN_PREDICTIONS} settled model predictions",
                   "ok": n >= GATE_MIN_PREDICTIONS, "detail": f"{n} so far"})
    ci_ok, detail = False, "not enough data"
    cal_ok, cal_detail = False, "not enough data"
    if n >= 30:
        y = preds["result"].map(LABEL_IDX).to_numpy()
        PF = preds[["pF1", "pFX", "pF2"]].to_numpy(float)
        PM = preds[["pM1", "pMX", "pM2"]].to_numpy(float)
        rows = np.arange(n)
        d = np.log(np.clip(PF[rows, y], 1e-9, 1)) - np.log(np.clip(PM[rows, y], 1e-9, 1))
        rng = np.random.default_rng(seed)
        boots = rng.choice(d, size=(n_boot, n)).mean(axis=1)
        lo = float(np.percentile(boots, 2.5))
        ci_ok = lo > 0 and n >= GATE_MIN_PREDICTIONS
        detail = f"mean log-score gain vs baseline {d.mean():+.4f}, 95% CI lower bound {lo:+.4f}"
        probs = PF.ravel()
        hits = np.eye(3)[y].ravel()
        worst, used = 0.0, 0
        for lo_b in np.arange(0, 1, 0.1):
            m = (probs >= lo_b) & (probs < lo_b + 0.1 + (1e-9 if lo_b > 0.85 else 0))
            if m.sum() >= 30:
                used += 1
                worst = max(worst, abs(hits[m].mean() - probs[m].mean()))
        cal_ok = used > 0 and worst <= 0.08
        cal_detail = f"worst gap in {used} well-populated bins: {worst:.3f} (limit 0.08)"
    checks.append({"name": "Model beats the baseline odds (95% bootstrap CI above zero)", "ok": ci_ok, "detail": detail})
    checks.append({"name": "Probabilities are calibrated", "ok": cal_ok, "detail": cal_detail})
    paper = db.q("SELECT * FROM bets WHERE mode='PAPER' AND decision='BET' AND status IN ('Won','Lost')")
    nb = len(paper)
    roi = float(paper["pnl"].sum() / paper["stake"].sum()) if nb and paper["stake"].sum() > 0 else 0.0
    checks.append({"name": f"At least {GATE_MIN_BETS} settled paper bets", "ok": nb >= GATE_MIN_BETS,
                   "detail": f"{nb} so far"})
    checks.append({"name": "Paper bets are profitable (return on stakes above zero)", "ok": nb >= GATE_MIN_BETS and roi > 0,
                   "detail": f"return on stakes {roi:+.2%}"})
    return {"passed": all(c["ok"] for c in checks), "checks": checks}
