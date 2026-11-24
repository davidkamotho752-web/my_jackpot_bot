"""Generate daily predictions. Applies model, tax hurdle, and writes to SQLite."""
import numpy as np
import pandas as pd
import core.calibration as cal
import core.db as db
import core.model as model
import core.tax as tax_mod

SEL_LABEL = {"1": "Home Win", "X": "Draw", "2": "Away Win"}

def _choose(probs, odds):
    edges = []
    for i, p in enumerate(probs):
        e = tax_mod.net_edge(p, odds[i])
        edges.append(e)
    best = int(np.argmax(edges))
    return ["1", "X", "2"][best], edges[best], edges

def analyse_row(row, sharp=None, league_count=0):
    # We handle missing odds gracefully now
    o = [row.get("B365H"), row.get("B365D"), row.get("B365A")]
    if any(pd.isna(v) or v <= 1 for v in o):
        return None
        
    sharp_odds = sharp if sharp else None
    probs, src = model.predict(sharp_odds=sharp_odds, soft_odds=o, league=row.get("League"), matches_in_league=league_count)
    selection, edge, all_edges = _choose(probs, o)
    odds = o[["1", "X", "2"].index(selection)]
    p_low = probs[["1", "X", "2"].index(selection)] * cal.current_factor()
    min_odds = tax_mod.min_odds_for(p_low, 0.03)
    decision = "BET" if edge >= 0.03 and odds >= min_odds else "SKIP"
    
    bankroll = float(db.get("bankroll", 10000))
    stake = round(bankroll * 0.02) if decision == "BET" else 0.0
    
    return {
        "league": row.get("League"),
        "home": row["HomeTeam"],
        "home_in": row["HomeTeam"],
        "away": row["AwayTeam"],
        "away_in": row["AwayTeam"],
        "o1": o[0], "ox": o[1], "o2": o[2],
        "pF1": probs[0], "pFX": probs[1], "pF2": probs[2],
        "ref_source": src,
        "selection": selection,
        "decision": decision,
        "odds": odds,
        "min_odds": min_odds,
        "edge_low": edge,
        "stake": stake,
    }

def run(df, source="manual", limit=50):
    if df.empty:
        db.log_event("predict", "no fixtures to analyse")
        return []
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    now = db.now_local()
    created_rows = []
    league_counts = df.groupby("League").size().to_dict() if "League" in df.columns else {}
    
    for _, row in df.head(limit).iterrows():
        res = analyse_row(row, league_count=league_counts.get(row.get("League"), 0))
        if res is None:
            continue
            
        payload = {
            "created": db.iso(now),
            "source": source,
            "match_date": pd.Timestamp(row["Date"]).strftime("%d/%m/%Y %H:%M"),
            "bet_type": "REAL",
            "action": "Place bet" if res["decision"] == "BET" else "SKIP",
            **res,
        }
        pid = db.insert("predictions", payload)
        if res["decision"] == "BET":
            db.insert("bets", {
                "pred_id": pid,
                "created": db.iso(now),
                "mode": "PAPER",
                "selection": res["selection"],
                "odds": res["odds"],
                "stake": res["stake"],
                "home": res["home"],
                "away": res["away"],
                "bet_type": "REAL",
                "decision": "BET",
            })
        created_rows.append(pid)
    db.log_event("predict", f"wrote {len(created_rows)} predictions from {source}")
    return created_rows