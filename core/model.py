"""Margin-free baseline. Uses sharp bookmaker implied probability."""
import math
import core.calibration as cal

MAINSTREAM_LEAGUES = {"E0", "D1", "SP1", "I1", "F1", "UCL"}

def implied(odds_triple):
    inv = [1.0 / o for o in odds_triple]
    s = sum(inv)
    return [i / s for i in inv], s - 1.0

def predict(sharp_odds=None, soft_odds=None, league=None, matches_in_league=0):
    if sharp_odds and all(o > 1 for o in sharp_odds):
        probs, _ = implied(sharp_odds)
        return cal.shrink(probs), "sharp"
    if soft_odds and all(o > 1 for o in soft_odds):
        probs, _ = implied(soft_odds)
        if matches_in_league < 30:
            probs = [0.9 * p + 0.1 * (1.0 / 3) for p in probs]
        return cal.shrink(probs), "soft"
    return [1.0 / 3, 1.0 / 3, 1.0 / 3], "uniform"

def logloss(probs_list, outcomes):
    total = 0.0
    for p, y in zip(probs_list, outcomes):
        total -= math.log(max(p[y], 1e-9))
    return total / max(len(outcomes), 1)
