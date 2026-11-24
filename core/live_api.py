import streamlit as st
import pandas as pd

def fetch_live_fixtures():
    """Fetch today's fixtures and odds for the top-5 leagues."""
    # Get the key from Streamlit secrets
    api_key = st.secrets.get("FOOTBALL_API_KEY", "")
    if not api_key:
        return pd.DataFrame()
    
    rows = []
    
    try:
        import fivedollarfootball
        client = fivedollarfootball.Client(api_key)
        
        # Get today's fixtures
        fixtures = client.fixtures()
        
        for match in fixtures:
            # Only include matches that haven't started or are live
            if match.get("status") not in ["scheduled", "live"]:
                continue
                
            fixture_id = match["id"]
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]
            
            # Get the 1X2 odds from Bet365
            odds_data = client.fixture_odds(fixture_id, bookmakers=["bet365"], market="1x2")
            
            # Extract the current odds
            odds = odds_data.get("1x2", {})
            current = odds.get("inplay") or odds.get("opening")
            
            if not current:
                continue
                
            rows.append({
                "HomeTeam": home,
                "AwayTeam": away,
                "B365H": current.get("home"),
                "B365D": current.get("draw"),
                "B365A": current.get("away"),
                "League": match["league"]["name"],
                "Date": match["kickoff_utc"],
                "Source": "live_api"
            })
            
    except Exception as e:
        # If the API fails, return empty so the app doesn't crash
        print(f"Live API error: {e}")
        return pd.DataFrame()
        
    return pd.DataFrame(rows)