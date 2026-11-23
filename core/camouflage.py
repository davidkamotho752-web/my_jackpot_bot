"""Explicit action generator for every bet."""
import random
import core.db as db
import core.model as model

CAMO_RATE = 0.12
CASHOUT_MIN = 0.55
CASHOUT_MAX = 0.75

def _camo_count_today():
    today = db.now_local().strftime("%Y-%m-%d")
    rows = db.q("SELECT COUNT(*) AS n FROM camouflage_log WHERE created LIKE ?", (f"{today}%",))
    return rows[0]["n"] if rows else 0

def _mainstream_leagues():
    return model.MAINSTREAM_LEAGUES

def pick_camouflage(real_bet_count_today):
    if _camo_count_today() > 0:
        return None
    target = max(1, round(real_bet_count_today * CAMO_RATE))
    if real_bet_count_today < 5:
        target = 1 if random.random() < 0.5 else 0
    if target == 0:
        return None
    kind = random.choice(["mainstream_single", "small_accumulator"])
    stake = round(random.uniform(43, 73), 0)
    return {
        "type": kind,
        "stake": stake,
        "market": "mainstream 1X2",
        "reason": "break sharp pattern; look recreational",
    }

def cashout_instruction(bet_id, league, potential_profit):
    if league not in _mainstream_leagues():
        return "Do not cash out. Let it settle."
    if not potential_profit or potential_profit <= 0:
        return "Do not cash out. Let it settle."
    roll = random.random()
    if roll > 0.15:
        return "Do not cash out. Let it settle."
    threshold = round(random.uniform(CASHOUT_MIN, CASHOUT_MAX) * 100, 0)
    return f"Cash out when the cash-out offer reaches ~{threshold:.0f}% of potential profit. Do not wait for settlement."

def required_action(bet, real_today, camo_count_today):
    if bet.get("bet_type") == "CAMOUFLAGE":
        return f"CAMOUFLAGE · stake KES {bet.get('stake', 0):.0f} · Do not track · This is a cost, not a value bet"
    potential = bet["stake"] * (bet["odds"] - 1.0) if bet.get("odds") and bet.get("stake") else 0
    cashout = cashout_instruction(bet.get("id"), bet.get("league", ""), potential)
    return f"REAL BET · stake KES {bet['stake']:.0f} · Take if odds >= {bet.get('min_odds', 0):.2f} · {cashout}"

def log_camouflage(instruction, match_text, odds):
    db.insert("camouflage_log", {
        "created": db.iso(db.now_local()),
        "match": match_text,
        "market": instruction["market"],
        "camo_type": instruction["type"],
        "stake": instruction["stake"],
        "odds": odds,
        "reason": instruction["reason"],
    })
