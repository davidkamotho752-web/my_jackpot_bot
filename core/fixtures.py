"""Fetch and parse fixtures from football-data.co.uk CSVs."""
import io
from datetime import datetime, timezone
import pandas as pd
import requests
import core.db as db

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = ["E0", "E1", "E2", "E3", "EC", "D1", "D2", "SP1", "SP2", "I1", "I2", "F1", "F2", "N1", "B1", "P1", "T1", "G1", "SC0", "SC1", "UCL", "EL"]

def _season_codes():
    today = datetime.utcnow()
    y = today.year if today.month >= 7 else today.year - 1
    out = []
    for offset in (0, 1, 2):
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
            df["League"] = league
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    
    # Parse dates
    all_df["Date"] = pd.to_datetime(all_df["Date"], dayfirst=True, errors="coerce")
    all_df["Date"] = all_df["Date"].dt.tz_localize('UTC')
    
    # ONLY GET MATCHES FROM TODAY ONWARDS
    today = pd.Timestamp.now(tz='UTC').normalize()
    all_df = all_df[all_df["Date"] >= today].dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    
    db.log_event("fixtures", f"loaded {len(all_df)} upcoming matches across {all_df['League'].nunique()} leagues")
    return all_df.sort_values("Date").reset_index(drop=True)

def normalize_columns(df):
    for c in ["B365H", "B365D", "B365A", "PSH", "PSD", "PSA", "FTHG", "FTAG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def results_only(df):
    return df.dropna(subset=["FTHG", "FTAG"]).copy()