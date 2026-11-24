"""Fetch and parse fixtures from football-data.co.uk CSVs."""
import io
from datetime import datetime
import pandas as pd
import requests
import core.db as db

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = ["E0", "D1", "SP1", "I1", "F1", "UCL"]

def _season_codes():
    today = datetime.utcnow()
    y = today.year if today.month >= 7 else today.year - 1
    out = []
    for offset in (0, 1):
        start = y - offset
        end = start + 1
        out.append(f"{str(start)[-2:]}{str(end)[-2:]}")
    return out

def _download_one(url):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or len(r.content) < 500:
            return None
        return pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
    except Exception:
        return None

def download(days_back=0, max_seasons=1):
    frames = []
    seasons = _season_codes()[:max_seasons]
    for league in LEAGUES:
        for season in seasons:
            df = _download_one(BASE.format(season=season, league=league))
            if df is None or len(df) == 0:
                continue
            # Use copy to avoid pandas fragmentation warnings
            df = df.copy()
            df["League"] = league
            frames.append(df)
            
    if not frames:
        return pd.DataFrame()
        
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = all_df.copy()
    
    all_df["Date"] = pd.to_datetime(all_df["Date"], dayfirst=True, errors="coerce")
    all_df["Date"] = all_df["Date"].dt.tz_localize('UTC')
    
    # ONLY GET MATCHES FROM TODAY ONWARDS
    today = pd.Timestamp.now(tz='UTC').normalize()
    all_df = all_df[all_df["Date"] >= today].dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    
    db.log_event("fixtures", f"loaded {len(all_df)} upcoming matches")
    return all_df.sort_values("Date").reset_index(drop=True)