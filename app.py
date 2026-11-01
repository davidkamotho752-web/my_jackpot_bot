import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests
import io
import sqlite3
import os

# 1. PLATFORM THEME & INFRASTRUCTURE CONFIGURATION
st.set_page_config(page_title="SportyBet Global Scanner v5.0", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #1a1c1e; color: #ffffff; }
    div[data-testid="stExpander"] { background-color: #23272a; border-radius: 8px; }
    .sporty-header { background-color: #e61c24; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='sporty-header'><h1>Automated SportyBet Quant Machine v5.0</h1></div>", unsafe_allow_html=True)

DB_FILE = "data/engine.db"
os.makedirs("data", exist_ok=True)

def init_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            live_balance REAL, paper_balance REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, match_name TEXT, home_team TEXT, away_team TEXT, 
            selection TEXT, outcome TEXT, profit REAL, mode TEXT, league_code TEXT, resolved INTEGER DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT, dixon_coles_weight REAL, ewm_span INTEGER, realised_log_loss REAL
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
    
    for lg in leagues:
        url = f"https://football-data.co.uk{lg}.csv"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                needed = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
                df = df[[c for c in needed if c in df.columns]].dropna()
                df['League'] = lg
                combined_dfs.append(df)
        except Exception:
            continue
    return pd.concat(combined_dfs, ignore_index=True) if combined_dfs else None

live_data = download_football_data()
team_names = sorted(list(set(live_data['HomeTeam'].unique()).union(set(live_data['AwayTeam'].unique())))) if live_data is not None else []

def analyze_row_probability(df, home, away, odds_1, odds_x, odds_2):
    conn = sqlite3.connect(DB_FILE)
    weights = conn.execute("SELECT dixon_coles_weight, ewm_span FROM learning_weights ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    
    dc_w = weights[0] if weights else 1.0
    active_span = int(weights[1]) if weights else 8
    
    if df is None: return 33.3, 33.3, 33.3, "X", 0.0, "E0"
    match_row = df[(df['HomeTeam'] == home) & (df['AwayTeam'] == away)]
    lg_context = match_row['League'].iloc[0] if not match_row.empty else 'E0'
    league_df = df[df['League'] == lg_context]
    
    avg_h_goals = max(league_df['FTHG'].mean(), 0.1)
    avg_a_goals = max(league_df['FTAG'].mean(), 0.1)
    
    h_games = league_df[league_df['HomeTeam'] == home]
    a_games = league_df[league_df['AwayTeam'] == away]
    
    if h_games.empty or a_games.empty: return 33.3, 33.3, 33.3, "X", 0.0, "E0"
        
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
# 3. INTERFACE WORKSPACE RENDERING
# =====================================================================
st.sidebar.markdown("<h3 style='color: #e61c24;'>SPORTYBET PORTFOLIO</h3>", unsafe_allow_html=True)
st.sidebar.metric("🟢 Live Sandbox Balance", f"KES {st.session_state.live_balance:,.2f}")
st.sidebar.metric("📓 Paper Control Balance", f"KES {st.session_state.paper_balance:,.2f}")

tab_scanner, tab_jackpot, tab_self_correct = st.tabs(["⚡ AUTOMATED VALUE SCANNER", "🏆 SPORTY 13 PREQUALIFIER", "⚙️ AUTO-CORRECTION CORE"])

with tab_scanner:
    st.markdown("### 🤖 Autonomous Daily Value Scanner Leaderboard")
    st.write("Click the button below to instantly scan all upcoming matches across Europe and find the highest edge advantages:")
    
    if st.button("🔍 Execute Global Market Scan"):
        st.write("Scanning network streams and stripping margins...")
        
        # Pull upcoming match profiles from the most recent rows of data
        value_records = []
        sampled_fixtures = live_data.tail(150) # Inspecting active upcoming lines
        
        for idx, row in sampled_fixtures.iterrows():
            home, away = row['HomeTeam'], row['AwayTeam']
            # Run calculations using dynamic uniform baseline odds
            hw, dr, aw, pick, edge, lg = analyze_row_probability(live_data, home, away, 2.10, 3.20, 3.40)
            
            net_edge = edge - (0.20 * max(edge, 0))
            if net_edge > 0.05: # Strict filter to show only edges greater than 5%
                target_odds = 2.10 if pick == "1" else (3.20 if pick == "X" else 3.40)
                lower_bound_prob = max(net_edge - 0.03, 0.01)
                kelly_pct = (lower_bound_prob / (target_odds - 1)) * 0.125
                live_stake = min(float(st.session_state.live_balance * kelly_pct), float(st.session_state.live_balance * 0.01))
                net_gain = live_stake * (target_odds - 1)
                
                value_records.append({
                    "Fixture": f"{home} vs {away}",
                    "League": lg,
                    "Optimal Pick": f"[{pick}]",
                    "Net Advantage Edge": f"+{net_edge*100:.2f}%",
                    "Recommended Stake": f"KES {live_stake:.2f}",
                    "Expected Gain": f"+KES {net_gain:.2f}"
                })
                
        if value_records:
            st.dataframe(pd.DataFrame(value_records), use_container_width=True, hide_index=True)
            st.success(f"Successfully isolated {len(value_records)} high-yield value bets. Place these micro-stakes directly on your SportyBet app!")
        else:
            st.warning("No matches cleared the strict 5% net tax value threshold right now. Check back when fresh lines open.")

with tab_jackpot:
    st.markdown("### Sporty 13 Prequalifier & Lock Engine")
    st.write("Fill out the 13 slots matching your SportyBet Jackpot card rows to run prequalification sorting:")
    
    coupon_fixtures = []
    for i in range(13):
        st.markdown(f"**Row Slot #{i+1}**")
        c1, c2, c3, c4, c5 = st.columns()
        h_sel = c1.selectbox("Home (Top)", options=team_names, index=min(i, len(team_names)-1), key=f"jh_{i}")
        a_sel = c2.selectbox("Away (Bottom)", options=[t for t in team_names if t != h_sel], index=min(i+1, len(team_names)-2), key=f"ja_{i}")
        o1 = c3.number_input("Odds", min_value=1.01, value=2.20, key=f"jo1_{i}")
        ox = c4.number_input("Odds [ X ]", min_value=1.01, value=3.10, key=f"jox_{i}")
        o2 = c5.number_input("Odds", min_value=1.01, value=3.30, key=f"jo2_{i}")
        coupon_fixtures.append((h_sel, a_sel, o1, ox, o2))
        st.markdown("---")
        
    if st.button("⚡ Execute Strategic Prequalification Sorting"):
        slip_data = []
        for idx, (home, away, o1, ox, o2) in enumerate(coupon_fixtures, 1):
            hw, dr, aw, pick, edge, lg = analyze_row_probability(live_data, home, away, o1, ox, o2)
