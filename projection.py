"""projection.py - what could the daily system earn per month? An honest range, not a promise.

Inputs are the real limits (stake cap, bets per day) and the real costs (5% excise on money in and out). The unknown is the
edge per bet (ROI). Nobody can know it yet, so three cases are shown: no edge, a modest edge, and the edge the tool has
actually measured so far (closing-line value, if enough bets exist). Growth is simulated, because luck dominates a month.
"""
import numpy as np


def simulate(balance, roi, stake_pct, bets_per_month, avg_odds, months=12, paths=3000, seed=1):
    """Monte Carlo of compounding at a fixed % stake. Returns percentiles of the balance after each month."""
    rng = np.random.default_rng(seed)
    p = min(max((1.0 + roi) / avg_odds, 0.01), 0.99)                  # win chance that gives this ROI at these odds
    bal = np.full(paths, float(balance))
    out = []
    for _ in range(months):
        wins = rng.binomial(bets_per_month, p, size=paths)
        # each bet stakes stake_pct of the month-start balance (a close approximation of per-bet compounding)
        gain = wins * (avg_odds - 1.0) - (bets_per_month - wins)
        bal = bal * (1.0 + stake_pct * gain)
        bal = np.maximum(bal, 0.0)
        out.append(np.percentile(bal, [10, 50, 90]))
    return np.array(out)


def scenarios(balance, stake_pct, bets_per_day, avg_odds, measured_roi=None, excise=0.05, months=12):
    n = int(round(bets_per_day * 30))
    cases = [("No edge (ROI 0%)", 0.0), ("Modest edge (ROI +2%)", 0.02), ("Good edge (ROI +4%)", 0.04)]
    if measured_roi is not None:
        cases.append((f"Measured so far (CLV {measured_roi:+.1%})", float(measured_roi)))
    rows = []
    for name, roi in cases:
        sim = simulate(balance, roi, stake_pct, n, avg_odds, months)
        exp_month = balance * stake_pct * n * roi                       # expected profit in month 1, before excise
        # excise: 5% on the deposit already paid, and 5% on whatever profit is withdrawn
        net_exp = exp_month * (1 - excise) if exp_month > 0 else exp_month
        one = simulate(balance, roi, stake_pct, n, avg_odds, 1)[0]
        rows.append({"Case": name, "Bets a month": n, "Expected profit, month 1": float(net_exp),
                     "Month 1 bad luck (10%)": float(one[0] - balance), "Month 1 good luck (90%)": float(one[2] - balance),
                     "After 12 months (median)": float(sim[-1, 1]), "After 12 months (bad luck, 10%)": float(sim[-1, 0]),
                     "After 12 months (good luck, 90%)": float(sim[-1, 2])})
    return rows
