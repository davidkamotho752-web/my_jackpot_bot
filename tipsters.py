"""tipsters.py - a measured, zero-credit ledger for tipsters' picks.

Why a ledger and not a feed: there is no verified free source of tipster picks, and booking codes cannot be read
without scraping SportyBet, which this tool does not do. So picks are typed in by you, BEFORE the match, and the tool
does the one thing that matters: it scores each tipster with the same tests as our own calls.

  * Settled automatically from the free results file (no API credits).
  * A pick entered on or after the match date is voided as a late entry, so back-filled "winners" cannot sneak in.
  * Flat 1-unit stakes, ROI with a bootstrap range, and closing-line value when Pinnacle's closing odds are in the file.
  * A verdict that is hard to pass: a record needs many picks AND a positive closing-line range. With many tipsters
    some look good by luck alone, so even a pass is a lead to keep tracking, not a signal to copy blindly.
Booking codes are stored only as a note. They are never decoded or fetched.
"""
import numpy as np
import pandas as pd

import daily
import engine as E
import postmortem
import store

SELECTIONS = ["1", "X", "2", "1X", "X2", "12", "O2.5", "U2.5"]
MIN_PICKS = 100


def add_pick(db, tipster, home, away, selection, odds, known=None, known_norm=None, note="", now=None):
    """Logs a pick. Returns (id, message). Team names are matched to the results file's names when possible."""
    now = now or store.now_local()
    tipster, home, away = str(tipster).strip(), str(home).strip(), str(away).strip()
    if not tipster or not home or not away:
        return None, "Tipster, home and away are all needed."
    if selection not in SELECTIONS:
        return None, "Unknown pick type."
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None, "Odds must be a number."
    if not odds > 1.0:
        return None, "Odds must be above 1.00."
    h = E.resolve_team(home, known, known_norm) if known else None
    a = E.resolve_team(away, known, known_norm) if known else None
    dup = db.q("SELECT 1 FROM tipster_picks WHERE tipster=? AND home_in=? AND away_in=? AND selection=? "
               "AND substr(created,1,10)=?", (tipster, home, away, selection, store.iso(now)[:10]))
    if len(dup):
        return None, "This exact pick is already logged today."
    pid = db.insert("tipster_picks", {"created": store.iso(now), "tipster": tipster, "home_in": home, "away_in": away,
                                      "home": h or "", "away": a or "", "selection": selection, "odds": odds,
                                      "note": str(note or "")[:200], "status": "Pending"})
    msg = "Logged." if (h and a) else "Logged, but the team names were not recognised in the results file, so it may never settle."
    return pid, msg


def settle(db, data, now=None):
    """Settles pending picks from the results file. Returns how many were settled."""
    now = now or store.now_local()
    pend = db.q("SELECT * FROM tipster_picks WHERE status='Pending' AND home<>'' AND away<>''")
    n = 0
    for r in pend.itertuples():
        row = postmortem._find(data, r.home, r.away, r.created)
        if row is None:
            continue
        match_day = pd.Timestamp(row["Date"]).normalize()
        if pd.Timestamp(r.created[:10]) >= match_day:
            db.x("UPDATE tipster_picks SET status='Void-late', unit_pnl=0, match_date=? WHERE id=?",
                 (match_day.strftime("%Y-%m-%d"), int(r.id)))
            n += 1
            continue
        won = daily.selection_won(r.selection, int(row["FTHG"]), int(row["FTAG"]))
        clv = None
        close = postmortem._closing(row)
        if close is not None and r.selection in postmortem.IDX:
            clv = float(r.odds * close[postmortem.IDX[r.selection]] - 1.0)
        db.x("UPDATE tipster_picks SET status=?, unit_pnl=?, clv=?, match_date=? WHERE id=?",
             ("Won" if won else "Lost", (r.odds - 1.0) if won else -1.0, clv, match_day.strftime("%Y-%m-%d"), int(r.id)))
        n += 1
    return n


def _boot(vals, n=2000, seed=0):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def verdict(n, roi_ci, clv, clv_ci, clv_n):
    if n < MIN_PICKS:
        return f"Too few settled picks ({n} of {MIN_PICKS} needed)."
    if clv is not None and clv_n >= MIN_PICKS and clv_ci[0] > 0:
        return "Positive closing-line value over many picks: worth tracking further, not proof."
    if roi_ci[0] > 0:
        return "Profit range above zero, but closing-line value is not confirmed: could be luck."
    return "No evidence of skill."


def report(db):
    """One row per tipster, from settled picks only (voided late entries are left out)."""
    d = db.q("SELECT * FROM tipster_picks WHERE status IN ('Won','Lost')")
    rows = []
    for name, g in d.groupby("tipster") if len(d) else []:
        n = len(g)
        lo, hi = _boot(g["unit_pnl"])
        clv = g["clv"].dropna()
        clo, chi = _boot(clv) if len(clv) else (float("nan"), float("nan"))
        rows.append({"Tipster": name, "Settled picks": n, "Hit rate": float((g["status"] == "Won").mean()),
                     "Average odds": float(g["odds"].mean()), "ROI (flat 1 unit)": float(g["unit_pnl"].mean()),
                     "ROI low": lo, "ROI high": hi, "CLV": float(clv.mean()) if len(clv) else None, "CLV picks": len(clv),
                     "Verdict": verdict(n, (lo, hi), float(clv.mean()) if len(clv) else None, (clo, chi), len(clv))})
    return pd.DataFrame(rows)
