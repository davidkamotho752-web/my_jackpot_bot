"""Probability calibration. Shrinks model confidence before staking."""
import math
import core.db as db

DEFAULT_SHRINKAGE = 0.85
MIN_SHRINKAGE = 0.60
MAX_SHRINKAGE = 1.00

def current_factor():
    v = db.get("calibration_shrinkage")
    return float(v) if v else DEFAULT_SHRINKAGE

def shrink(probs):
    f = current_factor()
    if len(probs) == 0:
        return probs
    mean = 1.0 / len(probs)
    adjusted = [mean + (p - mean) * f for p in probs]
    total = sum(adjusted)
    return [p / total for p in adjusted]

def record_outcome(predicted_p, actual_index, n_so_far):
    error = abs(predicted_p - (1.0 if actual_index == 0 else 0.0))
    if n_so_far < 50:
        return
    f = current_factor()
    if error > 0.15:
        f = max(MIN_SHRINKAGE, f - 0.01)
    elif error < 0.05:
        f = min(MAX_SHRINKAGE, f + 0.005)
    if abs(f - current_factor()) > 1e-9:
        db.set_key("calibration_shrinkage", f)
        db.log_event("calibration", f"shrinkage -> {f:.3f} after error {error:.3f}")

def kelly_stake(p_low, odds, bankroll, max_stake_pct):
    if bankroll <= 0 or odds <= 1 or p_low <= 0:
        return 0.0
    b = odds - 1.0
    f = (p_low * b - (1.0 - p_low)) / b
    if f <= 0:
        return 0.0
    stake = bankroll * min(f * 0.25, max_stake_pct)
    return math.floor(stake / 10) * 10
