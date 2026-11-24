"""Generate daily predictions."""
import numpy as np
import pandas as pd
import core.calibration as cal
import core.db as db
import core.model as model
import core.tax as tax_mod

SEL_LABEL = {"1": "Home Win", "X": "Draw", "2": "Away Win"}

def analyse_row(row, sharp=None, league_count=0):
    o = [row.get("B365H"), row.get("B365D"), row.get("B365A")]
    
    # If odds are missing, we still save the match so you can see it!
    if any(pd.isna(v) or v <= 1 for v in o):
        return {
            "league": row.get("League"), "home": row["HomeTeam"], "home_in": row["HomeTeam"],
            "away": row["AwayTeam"], "away_in": row["AwayTeam"],
            "o1": None, "ox": None, "o2": None, "pF1": None, "pFX": None, "pF2": None,
            "ref_source": "No odds available", "selection": None, "decision": "SKIP",
            "odds": None, "min_odds": None, "edge_low": None, "stake": 0.0
        }

    sharp_odds = sharp if sharp else None
    probs, src = model.predict(sharp_odds=sharp_odds, soft_odds=o, league=row.get("League"), matches_in_league=league_count)
    
    edges = [tax_mod.net_edge(p, o[i]) for i, p in enumerate(probs)]
    best = int(np.argmax(edges))
    selection = ["1", "X", "2"][best]
    edge = edges[best]
    
    odds = o[best]
    p_low = probs[best] * cal.current_factor()
    min_odds = tax_mod.min_odds_for(p_low, 0.03)
    decision = "BET" if edge >= 0.03 and odds >= min_odds else "SKIP"
    
    bankroll = float(db.get("bankroll", 10000))
    stake = round(bankroll * 0.02) if decision == "BET" else 0.0
    
    return {
        "league": row.get("League"), "home": row["HomeTeam"], "home_in": row["HomeTeam"],
        "away": row["AwayTeam"], "away_in": row["AwayTeam"],
        "o1": o[0], "ox": o[1], "o2": o[2],
        "pF1": probs[0], "pFX": probs[1], "pF2": probs[2],
        "ref_source": src, "selection": selection, "decision": decision,
        "odds": odds, "min_odds": min_odds, "edge_low": edge, "stake": stake,
    }

def run(df, source="manual", limit=50):
    if df.empty: return []
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    now = db.now_local()
    created_rows = []
    league_counts = df.groupby("League").size().to_dict() if "League" in df.columns else {}
    
    for _, row in df.head(limit).iterrows():
        res = analyse_row(row, league_count=league_counts.get(row.get("League"), 0))
        if res is None: continue
            
        payload = {
            "created": db.iso(now), "source": source,
            "match_date": pd.Timestamp(row["Date"]).strftime("%d/%m/%Y %H:%M"),
            "bet_type": "REAL", "action": "Place bet" if res["decision"] == "BET" else "SKIP",
            **res,
        }
        pid = db.insert("predictions", payload)
        if res["decision"] == "BET":
            db.insert("bets", {
                "pred_id": pid, "created": db.iso(now), "mode": "PAPER",
                "selection": res["selection"], "odds": res["odds"], "stake": res["stake"],
                "home": res["home"], "away": res["away"], "bet_type": "REAL", "decision": "BET",
            })
        created_rows.append(pid)
    return created_rows