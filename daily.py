"""daily.py - daily match analysis: BET or SKIP, stake, minimum odds, plus settlement.

Principles:
  * the edge is judged on a LOWER-bound probability (model probability minus an uncertainty margin);
  * singles only (no accumulators);
  * with no model for a match, or no sharp/clean reference, the answer is SKIP.
"""
import math

import numpy as np
import pandas as pd

import engine as E
import risk
import store

SEL_LABEL = {"1": "Home win (1)", "X": "Draw (X)", "2": "Away win (2)",
             "1X": "Double chance: Home or Draw (1X)", "X2": "Double chance: Draw or Away (X2)",
             "12": "Double chance: Home or Away (12)", "O2.5": "Over 2.5 goals", "U2.5": "Under 2.5 goals"}
MODEL_VER = "dc-v1"


def tax_factor(tax):
    """Per-bet haircut. Only used if the 'tax on every bet' switch is on (very conservative)."""
    return (1 - tax["deposit"]) * (1 - tax["withdrawal"]) if tax.get("per_bet") else 1.0


def flow_hurdle(tax):
    """Growth the wallet needs, in total, just to break even after depositing and withdrawing."""
    return 1.0 / ((1 - tax["deposit"]) * (1 - tax["withdrawal"])) - 1.0


def prob_sd(neff, base=0.03, cap=0.10):
    """Heuristic uncertainty in a model probability: larger when a team has little recent data."""
    if neff is None:
        return cap
    return float(min(cap, base * math.sqrt(max(1.0, 20.0 / max(neff, 1.0)))))


def net_edge(p, odds, tax):
    return p * odds * tax_factor(tax) - 1.0


def min_odds_for(p_low, min_edge, tax):
    return (1.0 + min_edge) / (max(p_low, 1e-9) * tax_factor(tax))


def kelly_stake(p_low, odds, tax, bankroll, lim):
    g = odds * tax_factor(tax)
    if bankroll <= 0 or g <= 1 or p_low * g <= 1:
        return 0.0
    f = (p_low * g - 1.0) / (g - 1.0)
    stake = bankroll * min(f * lim["kelly_frac"], lim["max_stake_pct"])
    stake = math.floor(stake / 10) * 10
    return float(stake) if stake >= lim["min_stake"] else 0.0


def selection_probs(pf, pov=None):
    d = {"1": pf[0], "X": pf[1], "2": pf[2], "1X": pf[0] + pf[1], "X2": pf[1] + pf[2], "12": pf[0] + pf[2]}
    if pov is not None:
        d["O2.5"], d["U2.5"] = pov, 1.0 - pov
    return d


def _num(v):
    try:
        v = float(v)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


class Context:
    """Everything the analyser needs (kept in one place so tests can build it easily)."""

    def __init__(self, models, team_league, w, tax, lim, bankroll, anchors=None):
        self.models, self.team_league, self.w = models, team_league, w
        self.tax, self.lim, self.bankroll = tax, lim, bankroll
        self.anchors = anchors or {}
        self.known = set(team_league)
        self.known_norm = {E.norm(t): t for t in self.known}


def analyse_match(r, ctx):
    """r: mapping with Home, Away, Odds 1, Odds X, Odds 2 and optional extra markets / Shift (pts)."""
    o1, ox, o2 = _num(r["Odds 1"]), _num(r["Odds X"]), _num(r["Odds 2"])
    h = E.resolve_team(r["Home"], ctx.known, ctx.known_norm) if ctx.known else None
    a = E.resolve_team(r["Away"], ctx.known, ctx.known_norm) if ctx.known else None
    entered_mkt, overround = E.devig(o1, ox, o2)
    anchor = ctx.anchors.get((E.norm(r["Home"]), E.norm(r["Away"])))
    ref = np.asarray(anchor["probs"]) if anchor else entered_mkt
    model_p = M = lg = min_neff = None
    if h and a and ctx.team_league.get(h) == ctx.team_league.get(a) and ctx.team_league[h] in ctx.models:
        lg = ctx.team_league[h]
        m = ctx.models[lg]
        if h in m["idx"] and a in m["idx"]:
            M = E.score_matrix(m, h, a)
            model_p = E.outcome_probs(M)
            min_neff = min(m["neff"][h], m["neff"][a])
    shift = r.get("Shift (pts)", 0.0) if hasattr(r, "get") else 0.0
    shift = 0.0 if shift is None or (isinstance(shift, float) and np.isnan(shift)) else float(shift)
    final = E.apply_shift(E.blend(ref, model_p, ctx.w, min_neff), shift)

    pover_model = E.p_over(M) if M is not None else None
    ov, un = _num(r.get("Over 2.5")), _num(r.get("Under 2.5"))
    pover = None
    if pover_model is not None and ov and un:
        mk = (1 / ov) / (1 / ov + 1 / un)
        pover = ctx.w * mk + (1 - ctx.w) * pover_model
    probs = selection_probs(final, pover)
    odds_map = {"1": o1, "X": ox, "2": o2, "1X": _num(r.get("Odds 1X")), "X2": _num(r.get("Odds X2")),
                "12": _num(r.get("Odds 12")), "O2.5": ov, "U2.5": un}
    # Optional second bookmaker (1X2 only): the best price for each pick wins, and we remember where it is.
    alt_name = getattr(ctx, "alt_name", "Other bookmaker") or "Other bookmaker"
    book_of = {sel: "SportyBet" for sel in odds_map}
    for sel, key in (("1", "Alt 1"), ("X", "Alt X"), ("2", "Alt 2")):
        alt_price = _num(r.get(key))
        if alt_price and alt_price > 1.0 and (odds_map.get(sel) is None or alt_price > odds_map[sel]):
            odds_map[sel], book_of[sel] = alt_price, alt_name

    out = {"home_in": r["Home"], "away_in": r["Away"], "home": h, "away": a, "league": lg,
           "entered": (o1, ox, o2), "mkt": entered_mkt, "ref": ref, "ref_source": f"reference: {anchor['source']}" if anchor else "your odds (margin removed)",
           "model": model_p, "final": final, "pover": pover, "overround": overround, "min_neff": min_neff,
           "cands": [], "decision": "SKIP", "reason": "", "selection": None, "odds": None, "p": None,
           "p_low": None, "edge_low": -1.0, "stake": 0.0, "min_odds": None, "shift": shift, "book": "SportyBet"}
    if model_p is None:
        out["reason"] = "No model for this match (team not found, or different leagues). Market only, so no edge to claim."
        return out
    sd = prob_sd(min_neff)
    for sel, o in odds_map.items():
        if o is None or sel not in probs:
            continue
        p = float(probs[sel])
        p_low = max(p - ctx.lim["z"] * sd, 0.0)
        out["cands"].append({"selection": sel, "odds": o, "p": p, "p_low": p_low, "book": book_of.get(sel, "SportyBet"),
                             "edge_mid": net_edge(p, o, ctx.tax), "edge_low": net_edge(p_low, o, ctx.tax)})
    if not out["cands"]:
        out["reason"] = "No odds entered."
        return out
    best = max(out["cands"], key=lambda c: c["edge_low"])
    out.update({"selection": best["selection"], "odds": best["odds"], "p": best["p"], "p_low": best["p_low"],
                "edge_low": best["edge_low"], "book": best["book"],
                "min_odds": min_odds_for(best["p_low"], ctx.lim["min_edge"], ctx.tax)})
    if best["edge_low"] < ctx.lim["min_edge"]:
        out["reason"] = (f"Best option ({best['selection']}) has net edge {best['edge_low']:+.1%} on the cautious "
                         f"probability; need at least {ctx.lim['min_edge']:+.0%}.")
    elif best["odds"] > ctx.lim["max_odds"]:
        out["reason"] = f"Odds {best['odds']:.2f} are above the long-shot limit of {ctx.lim['max_odds']:.1f}."
    else:
        stake = kelly_stake(best["p_low"], best["odds"], ctx.tax, ctx.bankroll, ctx.lim)
        if stake <= 0:
            out["reason"] = "Edge is positive but the stake rounds below the minimum."
        else:
            out.update({"decision": "BET", "stake": stake,
                        "reason": f"Net edge {best['edge_low']:+.1%} on the cautious probability."
                                  + (f" Best price is at {best['book']}." if best["book"] != "SportyBet" else "")})
    return out


def worth_confirming(r, threshold=-0.02):
    """True if a match is close enough to value that a sharp price is worth spending a credit on.
    Fair-looking matches (edge near minus the bookmaker margin) are not."""
    if r["model"] is None or not r["cands"] or r["ref_source"].startswith("reference"):
        return False
    return max(c["edge_mid"] for c in r["cands"]) >= threshold


# ----------------------------------------------------------------------------- saving and settling
def save_analysis(db, results, mode, group_id, param_ver, now=None, source="daily"):
    """Writes every analysed match to the append-only prediction log, plus its BET/SKIP decision."""
    now = now or store.now_local()
    ids = []
    for slot, r in enumerate(results, 1):
        if source == "daily":      # one logged call per match per day, even across sessions / reboots (no double counting)
            dup = db.q("SELECT 1 FROM predictions WHERE source='daily' AND home_in=? AND away_in=? AND substr(created,1,10)=?",
                       (r["home_in"], r["away_in"], store.iso(now)[:10]))
            if len(dup):
                continue
        o1, ox, o2 = r["entered"]
        mo = r["model"]
        pid = db.insert("predictions", {
            "created": store.iso(now), "source": source, "group_id": group_id, "slot": slot,
            "home_in": r["home_in"], "away_in": r["away_in"], "home": r["home"] or "", "away": r["away"] or "",
            "league": r["league"] or "", "o1": o1, "ox": ox, "o2": o2,
            "pM1": float(r["ref"][0]), "pMX": float(r["ref"][1]), "pM2": float(r["ref"][2]),
            "pD1": None if mo is None else float(mo[0]), "pDX": None if mo is None else float(mo[1]),
            "pD2": None if mo is None else float(mo[2]),
            "pF1": float(r["final"][0]), "pFX": float(r["final"][1]), "pF2": float(r["final"][2]),
            "pover": r["pover"], "model_ver": MODEL_VER, "param_ver": param_ver})
        ids.append(pid)
        if source == "daily" and r["selection"]:
            db.insert("bets", {"created": store.iso(now), "pred_id": pid, "mode": mode, "decision": r["decision"],
                               "reason": r["reason"], "home": r["home"] or r["home_in"], "away": r["away"] or r["away_in"],
                               "selection": r["selection"], "odds": r["odds"], "p": r["p"], "p_low": r["p_low"],
                               "edge_low": r["edge_low"], "stake": r["stake"] if r["decision"] == "BET" else 0.0,
                               "min_odds": r["min_odds"], "status": "Pending"})
    return ids


def selection_won(sel, hg, ag):
    res = "1" if hg > ag else ("X" if hg == ag else "2")
    if sel in ("1", "X", "2"):
        return sel == res
    if sel == "1X":
        return res in ("1", "X")
    if sel == "X2":
        return res in ("X", "2")
    if sel == "12":
        return res in ("1", "2")
    if sel == "O2.5":
        return hg + ag >= 3
    if sel == "U2.5":
        return hg + ag <= 2
    raise ValueError(sel)


def settle_all(db, data, now=None):
    """Fills in results for pending predictions from downloaded match data, then settles bets and bankroll."""
    now = now or store.now_local()
    n_pred = n_bet = 0
    pend = db.q("SELECT id, created, home, away FROM predictions WHERE result='Pending' AND home<>''")
    for r in pend.itertuples():
        start = pd.Timestamp(r.created[:10]) - pd.Timedelta(days=1)
        m = data[(data["HomeTeam"] == r.home) & (data["AwayTeam"] == r.away) &
                 (data["Date"] >= start) & (data["Date"] <= start + pd.Timedelta(days=22))].sort_values("Date")
        if m.empty:
            continue
        x = m.iloc[0]
        db.x("UPDATE predictions SET result=?, fthg=?, ftag=? WHERE id=?",
             (E.RESULT_TO_LABEL[x["FTR"]], int(x["FTHG"]), int(x["FTAG"]), int(r.id)))
        n_pred += 1
    bets = db.q("SELECT b.*, p.fthg, p.ftag FROM bets b JOIN predictions p ON p.id=b.pred_id "
                "WHERE b.status='Pending' AND p.result<>'Pending'")
    for b in bets.itertuples():
        won = selection_won(b.selection, int(b.fthg), int(b.ftag))
        unit = (b.odds - 1.0) if won else -1.0
        pnl = b.stake * unit
        db.x("UPDATE bets SET status=?, pnl=?, unit_pnl=?, settled=? WHERE id=?",
             ("Won" if won else "Lost", pnl, unit, store.iso(now), int(b.id)))
        if b.decision == "BET" and b.stake > 0:
            risk.add_ledger(db, b.mode, "bet_pnl", pnl, f"{b.home} v {b.away} {b.selection}", now, int(b.id))
        n_bet += 1
    return n_pred, n_bet


# ----------------------------------------------------------------------------- jackpot rounds
def estimate_tickets(odds_rows, outcomes, winners, gamma=1.25, draw_factor=0.85):
    """Rough tickets-sold estimate from one past round: winners13 / (crowd probability of the actual outcomes).
    odds_rows: list of (o1, ox, o2); outcomes: string like '1X2...'. Depends on the placeholder crowd model."""
    q = 1.0
    for (o1, ox, o2), res in zip(odds_rows, outcomes):
        c = E.crowd_shares(E.devig(o1, ox, o2)[0], gamma, draw_factor)
        q *= c["1X2".index(res)]
    est = winners / q if q > 0 else float("nan")
    return {"q": q, "estimate": est}
