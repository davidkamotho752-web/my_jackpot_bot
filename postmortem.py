"""postmortem.py - after the matches: what happened, why was the call right or wrong, and what is the tool biased on?

Everything is computed from the append-only prediction log plus the free results file (which also carries Pinnacle's
CLOSING odds, so closing-line value costs no API credits). It only DIAGNOSES; it never changes a setting by itself.
Changing weights stays with the guarded recalibration (walk-forward, halfway moves, logged), because tuning on a few
dozen noisy matches is how a tool talks itself into a bias.

Plain meanings:
  surprise      the result had a low probability under our own forecast (nothing to learn from one such match)
  unlucky       the result was plausible but not our pick
  model misread the market gave the result clearly more chance than our model did
  market misread our model gave it clearly more chance than the market did
CLV (closing-line value) = odds we took x Pinnacle's closing fair probability - 1. Positive on average over many bets is
the best early sign of a real edge: it needs far fewer bets than profit does.
"""
import numpy as np
import pandas as pd

import engine as E
import store

CLOSE_TRIPLES = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA")]
IDX = {"1": 0, "X": 1, "2": 2}


def _fair(v):
    inv = 1.0 / np.asarray(v, dtype=float)
    return inv / inv.sum()


def _closing(row):
    for cols in CLOSE_TRIPLES:
        if all(c in row.index for c in cols):
            v = [pd.to_numeric(row[c], errors="coerce") for c in cols]
            if all(pd.notna(x) and float(x) > 1 for x in v):
                return _fair(v)
    return None


def settled_matches(db):
    return db.q("SELECT * FROM predictions WHERE result<>'Pending' AND home<>'' AND pF1 IS NOT NULL AND " + store.AUTO_DEDUP)


def _find(data, home, away, created):
    start = pd.Timestamp(created[:10]) - pd.Timedelta(days=1)
    m = data[(data["HomeTeam"] == home) & (data["AwayTeam"] == away) &
             (data["Date"] >= start) & (data["Date"] <= start + pd.Timedelta(days=22))].sort_values("Date")
    return None if m.empty else m.iloc[0]


def diagnose(p_final, p_model, p_ref, picked=None, won=None):
    """One plain sentence about a settled match. Inputs are the probabilities given to the result that happened."""
    if p_final < 0.15:
        return f"Surprise: the result had only {p_final:.0%}. Nothing to fix from one match."
    if p_model is not None and p_ref is not None:
        if p_ref - p_model > 0.08:
            return f"Model misread: market gave it {p_ref:.0%}, our model only {p_model:.0%}."
        if p_model - p_ref > 0.08:
            return f"Market misread: our model gave it {p_model:.0%}, market {p_ref:.0%}."
    if picked and won is False:
        return f"Unlucky: the result was plausible ({p_final:.0%}) but was not the pick."
    return "Normal variance." if not won else "Call was right."


def match_table(db, data=None):
    sm = settled_matches(db)
    rows = []
    bets = db.q("SELECT * FROM bets WHERE decision='BET'")
    bet_by_pred = {int(r.pred_id): r for r in bets.itertuples()} if len(bets) else {}
    for r in sm.itertuples():
        k = IDX[r.result]
        pf, pm = [r.pF1, r.pFX, r.pF2][k], [r.pM1, r.pMX, r.pM2][k]
        pd_ = None if pd.isna(r.pD1) else [r.pD1, r.pDX, r.pD2][k]
        pc = None
        if data is not None and len(data):
            row = _find(data, r.home, r.away, r.created)
            cl = None if row is None else _closing(row)
            pc = None if cl is None else float(cl[k])
        b = bet_by_pred.get(int(r.id))
        picked, won, clv = None, None, None
        if b is not None:
            picked, won = b.selection, (b.status == "Won")
            if pc is not None and b.selection in IDX:
                full = _closing(row)
                clv = float(b.odds * full[IDX[b.selection]] - 1.0)
        rows.append({"Date": r.created[:10], "Match": f"{r.home_in} v {r.away_in}", "Source": r.source,
                     "Result": f"{r.fthg}-{r.ftag}", "Outcome": r.result, "P final": pf, "P model": pd_, "P market": pm,
                     "P close": pc, "Pick": picked, "Won": won, "CLV": clv,
                     "Why": diagnose(pf, pd_, pm, picked, won)})
    return pd.DataFrame(rows)


def _ll(p):
    return -np.log(np.clip(np.asarray(p, dtype=float), 1e-9, 1))


def bias_tables(db):
    """Aggregate checks. Returns dict of DataFrames plus a list of plain-language findings."""
    sm = settled_matches(db)
    out = {"findings": []}
    if sm.empty:
        return out
    k = sm["result"].map(IDX).to_numpy()
    P = sm[["pF1", "pFX", "pF2"]].to_numpy()
    rows = []
    for lab, j in IDX.items():
        pred, act, n = P[:, j].mean(), float((k == j).mean()), len(sm)
        se = np.sqrt(max(pred * (1 - pred), 1e-9) / n)
        rows.append({"Outcome": lab, "Predicted": pred, "Happened": act, "Matches": n, "z": (act - pred) / se})
        if n >= 50 and abs(act - pred) > 0.04 and abs((act - pred) / se) > 2:
            out["findings"].append(f"Outcome {lab}: predicted {pred:.1%}, happened {act:.1%} over {n} matches "
                                   f"({'under' if act > pred else 'over'}-predicted).")
    out["outcomes"] = pd.DataFrame(rows)
    sm = sm.assign(ll_final=_ll(P[np.arange(len(sm)), k]),
                   ll_market=_ll(sm[["pM1", "pMX", "pM2"]].to_numpy()[np.arange(len(sm)), k]))
    m = sm.dropna(subset=["pD1"])
    if len(m) >= 30:
        km = m["result"].map(IDX).to_numpy()
        ll_model = _ll(m[["pD1", "pDX", "pD2"]].to_numpy()[np.arange(len(m)), km]).mean()
        ll_mkt = _ll(m[["pM1", "pMX", "pM2"]].to_numpy()[np.arange(len(m)), km]).mean()
        ll_fin = _ll(m[["pF1", "pFX", "pF2"]].to_numpy()[np.arange(len(m)), km]).mean()
        out["scores"] = {"matches": len(m), "model": ll_model, "market": ll_mkt, "final": ll_fin}
        if ll_fin < ll_mkt:
            out["findings"].append(f"The blend scored better than the market alone ({ll_fin:.4f} vs {ll_mkt:.4f} log loss) "
                                   f"over {len(m)} matches.")
        else:
            out["findings"].append(f"The blend is NOT beating the market yet ({ll_fin:.4f} vs {ll_mkt:.4f}, lower is better) "
                                   f"over {len(m)} matches. Raising the market weight is the safe response.")
    lg = (sm[sm["league"] != ""].groupby("league")
          .agg(Matches=("id", "count"), Final=("ll_final", "mean"), Market=("ll_market", "mean")).reset_index())
    lg["Gain vs market"] = lg["Market"] - lg["Final"]                       # positive = blend better than market
    out["leagues"] = lg[lg["Matches"] >= 10].sort_values("Gain vs market")
    if not out["findings"]:
        out["findings"].append("No statistically clear bias yet. That is expected until a few hundred matches are settled.")
    return out


def bet_report(tbl):
    """Overconfidence check on the bets only: do the win probabilities we quoted match how often they won?"""
    if tbl.empty or "Won" not in tbl.columns:
        return None
    b = tbl.dropna(subset=["Won"])
    if b.empty:
        return None
    n = len(b)
    r = {"bets": n, "hit_rate": float(b["Won"].mean())}
    if b["CLV"].notna().sum():
        c = b["CLV"].dropna()
        rng = np.random.default_rng(0)
        boot = rng.choice(c.to_numpy(), size=(2000, len(c))).mean(axis=1)
        r.update({"clv": float(c.mean()), "clv_n": len(c), "clv_ci": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))})
    return r


def bets_needed(effect, sd):
    """Rough number of bets to see `effect` per bet (e.g. 0.01 = 1%) at 95% confidence, given per-bet spread sd."""
    return int(np.ceil((1.96 * sd / effect) ** 2))
