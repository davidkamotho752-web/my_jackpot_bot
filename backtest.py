"""backtest.py - run the past: would the real BET/SKIP rules have made money on matches that already happened?

Walk-forward and honest about what it can and cannot show:
  * For each block of the test window, the model is fitted ONLY on matches before that block (no peeking).
  * The same analyse_match / staking rules as the live app are used (cautious probability, min edge, Kelly cap,
    max odds, bets per day, bankroll that moves).
  * Price taken = a "soft bookmaker" proxy (Bet365 column) made worse by `extra_margin`, because SportyBet Kenya's
    margins are not known and are probably wider than Bet365's. Raise extra_margin to be harsher.
  * Reference ("sharp") = Pinnacle odds columns (opening if present, else closing), exactly what the live tool would
    pull from an odds API. use_sharp_ref=False removes it, to show what the model alone would do.
  * CLV (closing line value) = taken odds x Pinnacle CLOSING fair probability - 1. Positive on average means the tool
    gets prices better than the sharpest market's final view - the best early evidence of a real edge, far less noisy
    than profit.
NOT covered: SportyBet's real prices and limits, double chance / over-under markets (no historical odds in the free
files), the Sporty 13 jackpot (no free history of its pools), and the 5% excise on deposits/withdrawals (reported as a
separate hurdle). A good backtest is necessary, not sufficient: paper trading on live prices is the final test.
"""
import numpy as np
import pandas as pd

import daily
import engine as E
import risk

SOFT_COLS = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]
SHARP_COLS = [("PSH", "PSD", "PSA"), ("PSCH", "PSCD", "PSCA")]
CLOSE_COLS = ("PSCH", "PSCD", "PSCA")


def _triple(row, options):
    for cols in options:
        if all(c in row.index for c in cols):
            v = [pd.to_numeric(row[c], errors="coerce") for c in cols]
            if all(pd.notna(x) and float(x) > 1.0 for x in v):
                return [float(x) for x in v], cols[0]
    return None, None


def _fair(odds):
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


def run(data, test_days=365, refit_days=90, xi=0.002, w=0.6, lim=None, tax=None, extra_margin=0.02,
        use_sharp_ref=True, start_bankroll=10000.0, progress=None):
    """Returns {'bets': DataFrame, 'summary': dict}."""
    lim = {**risk.DEFAULT_LIMITS, **(lim or {})}
    tax = {**risk.DEFAULT_TAX, **(tax or {})}
    data = data.sort_values("Date").reset_index(drop=True)
    end = data["Date"].max()
    start = end - pd.Timedelta(days=test_days)
    blocks, s = [], start
    while s < end:
        blocks.append((s, min(s + pd.Timedelta(days=refit_days), end + pd.Timedelta(days=1))))
        s += pd.Timedelta(days=refit_days)

    bankroll, peak, max_dd = float(start_bankroll), float(start_bankroll), 0.0
    out, n_eval, ref_cols = [], 0, {}
    for bi, (b0, b1) in enumerate(blocks):
        if progress:
            progress(bi / max(len(blocks), 1), f"Block {bi + 1} of {len(blocks)}: fitting on matches before {b0.date()}")
        train = data[data["Date"] < b0]
        test = data[(data["Date"] >= b0) & (data["Date"] < b1)]
        if len(train) < 500 or test.empty:
            continue
        models = E.fit_all(train, xi)
        if not models:
            continue
        tl = E.team_league_map(train)
        day_rows = {}
        for _, r in test.iterrows():
            soft, _ = _triple(r, SOFT_COLS)
            if soft is None:
                continue
            n_eval += 1
            taken = [o / (1 + extra_margin) for o in soft]
            taken = [max(o, 1.01) for o in taken]
            anchors = {}
            if use_sharp_ref:
                sharp, col = _triple(r, SHARP_COLS)
                if sharp is not None:
                    ref_cols[col] = ref_cols.get(col, 0) + 1
                    anchors[(E.norm(r["HomeTeam"]), E.norm(r["AwayTeam"]))] = {"probs": _fair(sharp), "source": "pinnacle"}
            ctx = daily.Context(models, tl, w, tax, lim, bankroll, anchors)
            row = {"Home": r["HomeTeam"], "Away": r["AwayTeam"], "Odds 1": taken[0], "Odds X": taken[1], "Odds 2": taken[2]}
            res = daily.analyse_match(row, ctx)
            if res["decision"] != "BET":
                continue
            close, _ = _triple(r, [CLOSE_COLS])
            day_rows.setdefault(r["Date"], []).append((res, r, close))
        for day in sorted(day_rows):
            picks = sorted(day_rows[day], key=lambda t: -t[0]["edge_low"])[: int(lim["max_bets_day"])]
            risk_left = lim["max_risk_day_pct"] * bankroll
            day_pnl = 0.0
            for res, r, close in picks:
                stake = res["stake"]
                if stake > risk_left or stake <= 0:
                    continue
                risk_left -= stake
                sel = res["selection"]
                won = {"1": "H", "X": "D", "2": "A"}[sel] == r["FTR"]
                pnl = stake * (res["odds"] - 1) if won else -stake
                clv = None
                if close is not None:
                    clv = res["odds"] * float(_fair(close)[{"1": 0, "X": 1, "2": 2}[sel]]) - 1.0
                day_pnl += pnl
                out.append({"Date": day, "League": r["League"], "Match": f"{r['HomeTeam']} v {r['AwayTeam']}",
                            "Pick": sel, "Odds": round(res["odds"], 2), "Model p": round(res["p"], 3),
                            "Net edge": round(res["edge_low"], 3), "Stake": stake, "Won": bool(won),
                            "PnL": round(pnl, 2), "CLV": clv})
            bankroll += day_pnl
            peak = max(peak, bankroll)
            max_dd = max(max_dd, 1 - bankroll / peak)
    if progress:
        progress(1.0, "Done")
    bets = pd.DataFrame(out)
    return {"bets": bets, "summary": summarise(bets, n_eval, start_bankroll, bankroll, max_dd, ref_cols,
                                                use_sharp_ref, extra_margin, test_days, tax)}


def _boot(vals_num, vals_den=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(vals_num, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    d = np.asarray(vals_den, dtype=float) if vals_den is not None else np.ones(len(a))
    idx = rng.integers(0, len(a), size=(n, len(a)))
    est = a[idx].sum(axis=1) / d[idx].sum(axis=1)
    return (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))


def summarise(bets, n_eval, start_bankroll, final_bankroll, max_dd, ref_cols, use_sharp, extra_margin, test_days, tax):
    s = {"matches_evaluated": int(n_eval), "bets": int(len(bets)), "use_sharp_ref": bool(use_sharp),
         "extra_margin": extra_margin, "test_days": test_days, "ref_columns_used": ref_cols,
         "start_bankroll": start_bankroll, "final_bankroll": round(float(final_bankroll), 2),
         "max_drawdown": round(float(max_dd), 4), "flow_hurdle": daily.flow_hurdle(tax)}
    if bets.empty:
        s.update({"roi": None, "roi_ci": None, "clv": None, "clv_ci": None, "hit_rate": None, "staked": 0.0,
                  "profit": 0.0, "verdict": "No bets qualified in this window. The rules were too strict to act, which "
                                           "also means there was nothing to lose."})
        return s
    staked, profit = float(bets["Stake"].sum()), float(bets["PnL"].sum())
    roi = profit / staked
    lo, hi = _boot(bets["PnL"], bets["Stake"])
    clv = bets["CLV"].dropna()
    clo, chi = _boot(clv) if len(clv) else (float("nan"), float("nan"))
    s.update({"staked": staked, "profit": round(profit, 2), "roi": roi, "roi_ci": (lo, hi),
              "hit_rate": float(bets["Won"].mean()), "bet_rate": len(bets) / max(n_eval, 1),
              "clv": float(clv.mean()) if len(clv) else None, "clv_ci": (clo, chi) if len(clv) else None})
    s["verdict"] = verdict(s)
    return s


def verdict(s):
    """Plain-language reading. Deliberately hard to pass: profit alone is mostly luck over a few hundred bets."""
    n = s["bets"]
    clv, clv_ci, roi_ci = s.get("clv"), s.get("clv_ci"), s.get("roi_ci")
    if n < 100:
        return f"Only {n} bets: far too few to say anything. Extend the window or loosen nothing."
    parts = []
    if clv is not None and clv_ci and clv_ci[0] > 0:
        parts.append(f"Closing-line value is positive ({clv:+.1%}, 95% range {clv_ci[0]:+.1%} to {clv_ci[1]:+.1%}): "
                     "the prices taken beat Pinnacle's final view. This is the strongest sign of a real edge.")
    elif clv is not None:
        parts.append(f"Closing-line value is {clv:+.1%} (95% range {clv_ci[0]:+.1%} to {clv_ci[1]:+.1%}): "
                     "not clearly above zero, so there is no proven edge in this window.")
    if roi_ci and roi_ci[0] > 0:
        parts.append(f"ROI {s['roi']:+.1%} with the whole 95% range above zero.")
    else:
        parts.append(f"ROI {s['roi']:+.1%} but the 95% range ({roi_ci[0]:+.1%} to {roi_ci[1]:+.1%}) includes zero or below: "
                     "profit alone cannot be told apart from luck.")
    parts.append("A backtest uses Bet365-style prices, not SportyBet's, and cannot see the jackpot. Treat it as a filter, "
                 "not a green light.")
    return " ".join(parts)
