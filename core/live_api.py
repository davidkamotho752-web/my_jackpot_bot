import streamlit as st
import pandas as pd
import requests
from datetime import datetime

def fetch_live_fixtures():
    """Try multiple free APIs to find matches. Fallback to ESPN (no key needed)."""
    
    # 1. Try 5DollarFootballAPI (Top 5 Leagues)
    df_5dollar = _fetch_5dollar()
    if not df_5dollar.empty:
        return df_5dollar
        
    # 2. Try API-Football (Broad Coverage)
    df_apifootball = _fetch_api_football()
    if not df_apifootball.empty:
        return df_apifootball

    # 3. Fallback to ESPN (No Key, ~228 Leagues)
    df_espn = _fetch_espn()
    if not df_espn.empty:
        return df_espn

    return pd.DataFrame()

def _fetch_5dollar():
    api_key = st.secrets.get("FOOTBALL_API_KEY", "")
    if not api_key: return pd.DataFrame()
    try:
        import fivedollarfootball
        client = fivedollarfootball.Client(api_key)
        fixtures = client.fixtures()
        rows = []
        for match in fixtures:
            if match.get("status") not in ["scheduled", "live"]: continue
            odds_data = client.fixture_odds(match["id"], bookmakers=["bet365"], market="1x2")
            odds = odds_data.get("1x2", {})
            current = odds.get("inplay") or odds.get("opening")
            if not current: continue
            rows.append({
                "HomeTeam": match["teams"]["home"]["name"],
                "AwayTeam": match["teams"]["away"]["name"],
                "B365H": current.get("home"), "B365D": current.get("draw"), "B365A": current.get("away"),
                "League": match["league"]["name"], "Date": match["kickoff_utc"]
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def _fetch_api_football():
    api_key = st.secrets.get("API_FOOTBALL_KEY", "")
    if not api_key: return pd.DataFrame()
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    params = {"next": 3}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        data = response.json()
        rows = []
        for item in data.get("response", []):
            fixture = item["fixture"]
            teams = item["teams"]
            rows.append({
                "HomeTeam": teams["home"]["name"],
                "AwayTeam": teams["away"]["name"],
                "League": item["league"]["name"],
                "Date": fixture["date"],
                "B365H": None, "B365D": None, "B365A": None
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def _fetch_espn():
    """Fetch today's fixtures from ESPN's public API. No key required."""
    today = datetime.utcnow().strftime("%Y%m%d")
    # We query a few major ESPN league slugs to catch games today
    leagues = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1", "usa.1", "bra.1"]
    rows = []
    for league in leagues:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?dates={today}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            for event in data.get("events", []):
                comps = event.get("competitions", [{}])[0]
                teams = comps.get("competitors", [])
                if len(teams) != 2:
                    continue
                # ESPN structure: team[0] is home, team[1] is away
                h_team = teams[0].get("team", {}).get("displayName", "")
                a_team = teams[1].get("team", {}).get("displayName", "")
                rows.append({
                    "HomeTeam": h_team,
                    "AwayTeam": a_team,
                    "League": league,
                    "Date": event.get("date"),
                    "B365H": None, "B365D": None, "B365A": None
                })
        except Exception:
            continue
    return pd.DataFrame(rows)