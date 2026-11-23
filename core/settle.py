"""Settle predictions using ESPN public JSON endpoints."""
import requests
import pandas as pd
from datetime import datetime
import core.db as db
from core import team_resolve

ESPN_LEAGUES = {
    "E0": "eng.1",
    "SP1": "esp.1",
    "D1": "ger.1",
    "I1": "ita.1",
    "F1": "fra.1",
    "UCL": "uefa.champions"
}

def fetch_espn_scores(league, date_str):
    """Fetch scores for a specific league and YYYY-MM-DD date from ESPN."""
    espn_league = ESPN_LEAGUES.get(league, "eng.1") # fallback to EPL
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_league}/scoreboard?dates={date_str.replace('-', '')}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("events", [])
    except Exception:
        pass
    return []

def settle_all():
    """Find pending predictions, fetch their scores, and update them."""
    pending = db.q("SELECT * FROM predictions WHERE result='Pending' AND match_date <= date('now')")
    if not pending:
        return 0
        
    settled_count = 0
    # Group by league and date to minimize API calls
    grouped = {}
    for p in pending:
        key = (p['league'], p['match_date'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(p)
        
    for (league, date_str), matches in grouped.items():
        events = fetch_espn_scores(league, date_str)
        if not events:
            continue
            
        for p in matches:
            home_norm = team_resolve.normalize(p['home'])
            away_norm = team_resolve.normalize(p['away'])
            
            for event in events:
                comps = event.get("competitions", [{}])[0]
                teams = comps.get("competitors", [])
                if len(teams) != 2:
                    continue
                    
                # ESPN structure: teams[0] is usually home, teams[1] is away
                h_team = teams[0].get("team", {}).get("displayName", "")
                a_team = teams[1].get("team", {}).get("displayName", "")
                
                if team_resolve.normalize(h_team) == home_norm and team_resolve.normalize(a_team) == away_norm:
                    status = comps.get("status", {}).get("type", {}).get("completed", False)
                    if not status:
                        continue
                        
                    h_score = int(teams[0].get("score", 0))
                    a_score = int(teams[1].get("score", 0))
                    
                    if h_score > a_score: result = "1"
                    elif h_score == a_score: result = "X"
                    else: result = "2"
                    
                    clv = None # CLV calculation would require closing odds comparison
                    
                    db.x("UPDATE predictions SET result=?, fthg=?, ftag=?, clv=?, settled=? WHERE id=?", 
                         (result, h_score, a_score, clv, db.iso(db.now_local()), p['id']))
                    
                    # Update bets
                    bets = db.q("SELECT * FROM bets WHERE pred_id=?", (p['id'],))
                    for b in bets:
                        won = (b['selection'] == result)
                        pnl = (b['stake'] * (b['odds'] - 1)) if won else -b['stake']
                        db.x("UPDATE bets SET status=?, pnl=?, settled=? WHERE id=?", 
                             ("Won" if won else "Lost", pnl, db.iso(db.now_local()), b['id']))
                             
                    settled_count += 1
                    break # move to next prediction
                    
    db.log_event("settle", f"Settled {settled_count} matches")
    return settled_count