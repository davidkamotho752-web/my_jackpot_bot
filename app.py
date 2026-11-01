import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
import io
import sqlite3
import os

# 1. HARDEN APPLICATION INFRASTRUCTURE
st.set_page_config(page_title="SportyBet Quant Machine v3.2", layout="wide")
st.title("🤖 Autonomous SportyBet Reinforcement Quant Bot v3.2")
st.caption("System Status: Production Operational | Mode: Dampened Stochastic Gradient Feedback")
st.markdown("---")

DB_FILE = "data/engine.db"
os.makedirs("data", exist_ok=True)

def init_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            live_balance REAL,
            paper_balance REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_name TEXT, home_team TEXT, away_team TEXT, selection TEXT, 
            outcome TEXT, profit REAL, mode TEXT, league_code TEXT, resolved INTEGER DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dixon_coles_weight REAL,
            ewm_span INTEGER,
            realised_log_loss REAL
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM bankroll")
    if cursor.fetchone() == 0:
        cursor.execute("INSERT INTO bankroll (live_balance, paper_balance) VALUES (1000.00, 1000.00)")
        cursor.execute("INSERT INTO learning_weights (dixon_coles_weight, ewm_span, realised_log_loss) VALUES (1.0, 8, 0.693)")
    conn.commit()
    conn.close()

init_database()

if 'live_balance' not in st.session_state:
    conn = sqlite3.connect(DB_FILE)
    res = conn.execute("SELECT live_balance, paper_balance FROM bankroll ORDER BY id DESC LIMIT 1").fetchone()
    st.session_state.live_balance = res if res else 1000.00
    st.session_state.paper_balance = res if res else 1000.00
    conn.close()

# 2. DUAL-SOURCE HYBRID DATA PIPELINE WITH NORMALIZER
@st.cache_data(ttl=3600)
def download_football_data():
    leagues = ['E0', 'E1', 'E2', 'E3', 'EC', 'SP1', 'SP2', 'D1', 'D2', 'I1', 'I2', 'F1', 'F2', 'N1', 'B1', 'P1', 'T1', 'G1', 'SC0']
    combined_dfs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    def clean_team_string(name):
        if not isinstance(name, str): return name
        name = name.strip().lower()
        replacements = {
            'ath madrid': 'atletico madrid', 'ath bilbao': 'athletic_bilbao',
            'sociedad': 'real sociedad', 'celta vigo': 'celta',
            'espanol': 'espanyol', 'vallecano': 'rayo vallecano',
            'un berlin': 'union berlin', 'koln': 'cologne', 'paris sg': 'paris sg',
            'psg': 'paris sg', 'monaco': 'monaco', 'torino': 'torino', 'milan': 'ac milan', 'inter': 'inter milan'
        }
        for k, v in replacements.items():
            if k in name: return v.title()
        return name.title()
    
    for lg in leagues:
        url = f"https://football-data.co.uk{lg}.csv"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                needed = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                df = df[[c for c in needed if c in df.columns]].dropna()
                df['HomeTeam'] = df['HomeTeam'].apply(clean_team_string)
                df['AwayTeam'] = df['AwayTeam'].apply(clean_team_string)
                df['League'] = lg
                combined_dfs.append(df)
        except Exception:
            continue
            
    try:
        realtime_url = "https://githubusercontent.com"
        rt_res = requests.get(realtime_url, timeout=5)
        if rt_res.status_code == 200:
            rt_data = rt_res.json()
            rt_rows = []
            for round_data in rt_data.get('rounds', []):
                for match in round_data.get('matches', []):
                    if match.get('score'):
                        rt_rows.append({
                            'HomeTeam': clean_team_string(match['team1']),
                            'AwayTeam': clean_team_string(match['team2']),
                            'FTHG': int(match['score']['ft']),
                            'FTAG': int(match['score']['ft']),
                            'FTR': 'H' if match['score']['ft'] > match['score']['ft'] else ('A' if match['score']['ft'] > match['score']['ft'] else 'D'),
                            'League': 'E0'
                        })
            if rt_rows: combined_dfs.append(pd.DataFrame(rt_rows))
    except Exception:
        pass

    return pd.concat(combined_dfs, ignore_index=True) if combined_dfs else None

live_data = download_football_data()
team_names = sorted(list(set(live_data['HomeTeam'].unique()).union(set(live_data['AwayTeam'].unique())))) if live_data is not None else []

def analyze_row_probability(df, home, away, odds_1, odds_x, odds_2):
    conn = sqlite3.connect(DB_FILE)
    weights = conn.execute("SELECT dixon_coles_weight, ewm_span FROM learning_weights ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    
    dc_w = weights[0] if weights else 1.0
    active_span = int(weights[1]) if weights else 8
    
    match_row = df[(df['HomeTeam'] == home) & (df['AwayTeam'] == away)] if df is not None else pd.DataFrame()
    lg_context = match_row['League'].iloc[0] if not match_row.empty else 'E0'
    league_df = df[df['League'] == lg_context] if df is not None else pd.DataFrame()
    
    avg_h_goals = max(league_df['FTHG'].mean(), 0.1) if not league_df.empty else 1.2
    avg_a_goals = max(league_df['FTAG'].mean(), 0.1) if not league_df.empty else 1.0
    
    h_games = league_df[league_df['HomeTeam'] == home] if not league_df.empty else pd.DataFrame()
    a_games = league_df[league_df['AwayTeam'] == away] if not league_df.empty else pd.DataFrame()
    
    if h_games.empty or a_games.empty:
        return 33.3, 33.3, 33.3, "X", 0.0, "E0"
        
    h_atk = h_games['FTHG'].ewm(span=min(len(h_games), active_span)).mean().iloc[-1]
    h_def = h_games['FTAG'].ewm(span=min(len(h_games), active_span)).mean().iloc[-1]
    a_atk = a_games['FTAG'].ewm(span=min(len(a_games), active_span)).mean().iloc[-1]
    a_def = a_games['FTHG'].ewm(span=min(len(a_games), active_span)).mean().iloc[-1]
    
    exp_h = (h_atk / avg_h_goals) * (a_def / avg_h_goals) * avg_h_goals * dc_w
    exp_a = (a_atk / avg_a_goals) * (h_def / avg_a_goals) * avg_a_goals * dc_w
    
    max_g = 10
    h_poisson = stats.poisson.pmf(range(max_g), exp_h)
    a_poisson = stats.poisson.pmf(range(max_g), exp_a)
    matrix = np.outer(h_poisson, a_poisson)
    
    rho = -0.10
    if exp_h > 0 and exp_a > 0:
        matrix *= (1 - exp_h * exp_a * rho)
        matrix *= (1 + exp_h * rho)
        matrix *= (1 + exp_a * rho)
        matrix *= (1 - rho)
    matrix = np.clip(matrix, 0, None)
    matrix /= np.sum(matrix)
    
    hw = np.sum(np.tril(matrix, -1))
    dr = np.sum(np.diag(matrix))
    aw = np.sum(np.triu(matrix, 1))
    
    inv_sum = (1/odds_1) + (1/odds_x) + (1/odds_2)
    true_p1 = (1/odds_1) / inv_sum
    true_px = (1/odds_x) / inv_sum
    true_p2 = (1/odds_2) / inv_sum
    
    ev_1 = (hw / true_p1) - 1 if true_p1 > 0 else -1
    ev_x = (dr / true_px) - 1 if true_px > 0 else -1
    ev_2 = (aw / true_p2) - 1 if true_p2 > 0 else -1
    
    ev_map = {"1": ev_1, "X": ev_x, "2": ev_2}
    pick = max(ev_map, key=ev_map.get)
    return hw*100, dr*100, aw*100, pick, ev_map[pick], lg_context

# =====================================================================
# 3. INTERFACE FRAMEWORK & TABS
# =====================================================================
st.sidebar.markdown("### 🛡️ Dual-Wallet Isolation Guard")
st.sidebar.metric("🟢 Live Sandbox Balance", f"KES {st.session_state.live_balance:,.2f}")
st.sidebar.metric("📓 Paper Control Balance", f"KES {st.session_state.paper_balance:,.2f}")

conn = sqlite3.connect(DB_FILE)
total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
pending_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE resolved = 0").fetchone()[0]
active_weights = conn.execute("SELECT dixon_coles_weight, ewm_span, realised_log_loss FROM learning_weights ORDER BY id DESC LIMIT 1").fetchone()
conn.close()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Self-Correction Tuning Parameters")
if active_weights:
    st.sidebar.write(f"Dynamic Model Multiplier: **{active_weights[0]:.4f}**")
    st.sidebar.write(f"Recency Weight Horizon: **{active_weights[1]} Matches**")
    st.sidebar.write(f"System Cross-Entropy Loss: **{active_weights[2]:.4f}**")
else:
    st.sidebar.write("Dynamic Model Multiplier: **1.0000**")
    st.sidebar.write("Recency Weight Horizon: **8 Matches**")
    st.sidebar.write("System Cross-Entropy Loss: **0.6931**")

tab_daily, tab_jack, tab_risk, tab_self_correct = st.tabs(["⚽ TODAY (Daily Finder)", "🏆 JACKPOT (13-Slot)", "📊 BANKROLL AND RISK", "⚡ AUTONOMOUS SELF-CORRECTION CORE"])

with tab_daily:
    st.markdown("### Live Sandbox Single Bet Finder")
    wht_win = st.slider("Withholding Tax on Profit (%)", 0, 30, 20)
    
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
    d_home = col_d1.selectbox("Home Team", options=team_names, index=0) if team_names else col_d1.text_input("Home")
    d_away = col_d2.selectbox("Away Team", options=[t for t in team_names if t != d_home], index=1) if team_names else col_d2.text_input("Away")
    d_o1 = col_d3.number_input("Odds 1", min_value=1.01, value=2.20)
    d_ox = col_d4.number_input("Odds X", min_value=1.01, value=3.10)
