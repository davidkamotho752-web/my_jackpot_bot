"""audit.py - the SportyBet price check: how do the prices you actually see compare with Pinnacle's closing price?

This is the one thing history cannot tell us, because SportyBet's real prices are not in any free file. It is measured
from YOUR logged matches: every analysis you save keeps the 1 X 2 odds you typed from SportyBet, and the free results
file carries Pinnacle's closing odds (the sharpest widely available view of the true chances).

For each logged match and each outcome:
    value = SportyBet odds x Pinnacle closing fair probability - 1
Average value is negative by about the margin SportyBet charges. What matters is how often, where and when it is
positive: that is real, model-free room to profit (stale prices, soft markets, favourable odds bands, early vs late).
It does not use our forecast at all, so it also shows whether the tool's job is even possible on SportyBet.
Log EVERY match you look at, not only the bets: the more matches, the sooner the answer.
"""
import numpy as np
import pandas as pd

import postmortem

SEL = ["1", "X", "2"]
BANDS = [(1.0, 1.8, "1.01-1.79"), (1.8, 3.0, "1.80-2.99"), (3.0, 5.0, "3.00-4.99"), (5.0, 1e9, "5.00+")]
NEEDED = 300


def _closing_raw(row):
    for cols in postmortem.CLOSE_TRIPLES:
        if all(c in row.index for c in cols):
            v = [pd.to_numeric(row[c], errors="coerce") for c in cols]
            if all(pd.notna(x) and float(x) > 1.0 for x in v):
                return np.array([float(x) for x in v])
    return None


def _band(o):
    for lo, hi, name in BANDS:
        if lo <= o < hi:
            return name
    return BANDS[-1][2]


def build(db, data):
    """One row per logged match with closing prices available. Returns (DataFrame, number without closing prices)."""
    p = db.q("SELECT * FROM predictions WHERE home<>'' AND o1>1 AND ox>1 AND o2>1 ORDER BY id")
    if p.empty or data is None or len(data) == 0:
        return pd.DataFrame(), 0
    rows, seen, missing = [], set(), 0
    for r in p.itertuples():
        row = postmortem._find(data, r.home, r.away, r.created)
        if row is None:
            continue
        match_day = pd.Timestamp(row["Date"]).normalize()
        key = (r.home, r.away, match_day)
        if key in seen:
            continue
        seen.add(key)
        raw = _closing_raw(row)
        if raw is None:
            missing += 1
            continue
        pc = (1 / raw) / (1 / raw).sum()
        o = np.array([r.o1, r.ox, r.o2], dtype=float)
        val = o * pc - 1.0
        lead = (match_day - pd.Timestamp(r.created[:10])).days
        rows.append({"Date": match_day, "Match": f"{r.home_in} v {r.away_in}", "League": r.league or "", "Source": r.source,
                     "Lead days": max(int(lead), 0), "o1": o[0], "ox": o[1], "o2": o[2],
                     "Sporty margin": float((1 / o).sum() - 1), "Pinnacle margin": float((1 / raw).sum() - 1),
                     "v1": val[0], "vX": val[1], "v2": val[2], "Best value": float(val.max()), "Best pick": SEL[int(val.argmax())]})
    return pd.DataFrame(rows), missing


def long_form(df):
    """One row per (match, outcome) for grouped views."""
    if df.empty:
        return pd.DataFrame()
    parts = []
    for i, s in enumerate(SEL):
        parts.append(pd.DataFrame({"Outcome": s, "Odds": df[["o1", "ox", "o2"][i]], "Value": df[["v1", "vX", "v2"][i]],
                                   "League": df["League"], "Lead days": df["Lead days"]}))
    out = pd.concat(parts, ignore_index=True)
    out["Odds band"] = out["Odds"].map(_band)
    return out


def _boot_mean(v, n=2000, seed=0):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def summary(df, min_edge=0.03):
    if df.empty:
        return {"n": 0}
    lf = long_form(df)
    s = {"n": int(len(df)), "sporty_margin": float(df["Sporty margin"].mean()), "pinnacle_margin": float(df["Pinnacle margin"].mean()),
         "mean_value": float(lf["Value"].mean()), "share_positive": float((lf["Value"] > 0).mean()),
         "share_edge": float((df["Best value"] >= min_edge).mean()), "mean_best": float(df["Best value"].mean()),
         "best_ci": _boot_mean(df["Best value"]), "min_edge": min_edge}
    s["verdict"] = verdict(s)
    return s


def verdict(s):
    n = s["n"]
    if n < 50:
        return f"Only {n} matches with closing prices so far. Keep logging every match you look at ({NEEDED} for a first reading)."
    gap = s["sporty_margin"] - s["pinnacle_margin"]
    parts = [f"SportyBet's margin on your logged matches averages {s['sporty_margin']:.1%}, against Pinnacle's {s['pinnacle_margin']:.1%} "
             f"(about {gap:.1%} more to overcome)."]
    if s["share_edge"] >= 0.10:
        parts.append(f"In {s['share_edge']:.0%} of matches at least one SportyBet price beat Pinnacle's closing fair price by "
                     f"{s['min_edge']:.0%} or more. That is real room, before any forecast.")
    elif s["share_edge"] > 0:
        parts.append(f"Only {s['share_edge']:.0%} of matches had a price {s['min_edge']:.0%} or more above Pinnacle's closing fair price: "
                     "room exists but is thin, so selectivity and a good forecast matter.")
    else:
        parts.append("No logged match had a price clearly above Pinnacle's closing fair price. Without that room, no forecast helps.")
    if n < NEEDED:
        parts.append(f"{n} of {NEEDED} matches: treat this as an early reading.")
    return " ".join(parts)


def group(df, by):
    """Mean value, share of positive prices and count, by 'Outcome', 'Odds band', 'League' or 'Lead days'."""
    lf = long_form(df)
    if lf.empty or by not in lf.columns:
        return pd.DataFrame()
    g = lf.groupby(by).agg(Prices=("Value", "count"), Mean_value=("Value", "mean"),
                           Share_positive=("Value", lambda v: float((v > 0).mean()))).reset_index()
    return g[g["Prices"] >= 10].sort_values("Mean_value", ascending=False)
