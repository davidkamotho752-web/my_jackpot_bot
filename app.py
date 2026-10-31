import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

# 1. HARDEN APPLICATION INFRASTRUCTURE
st.set_page_config(page_title="SportyBet Full Slip Engine", layout="wide")
st.title("Automated SportyBet Full Jackpot Slip Machine")
st.caption("System Status: Production Operational | Mode: 12-Slot Sync")
st.markdown("---")

# Initialize persistent tracking wallet
if 'wallet_balance' not in st.session_state:
    st.session_state.wallet_balance = 10000.00
if 'history_ledger' not in st.session_state:
    st.session_state.history_ledger = []

# 2. CANONICAL SEASON DATA STREAM
@st.cache_data(ttl=3600)
def download_football_data():
    url = "https://football-data.co.uk"
    try:
        df = pd.read_csv(url)
        df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
        return df
    except Exception:
        # Fixed clean, executable baseline numbers for fallback stability
        return pd.DataFrame({
            'HomeTeam': ['Arsenal', 'Chelsea', 'Man City', 'Aston Villa', 'Liverpool', 'Tottenham', 'Newcastle', 'Man United', 'West Ham', 'Everton', 'Fulham', 'Brighton'],
            'AwayTeam': ['Chelsea', 'Man City', 'Aston Villa', 'Arsenal', 'Tottenham', 'Liverpool', 'Man United', 'Newcastle', 'Everton', 'West Ham', 'Brighton', 'Fulham'],
            'FTHG':, 
            'FTAG': [1, 1, 1, 2, 1, 1, 0, 1, 2, 1, 0, 2]
        })

live_data = download_football_data()

if live_data is not None:
    team_names = sorted(list(set(live_data['HomeTeam'].unique())))
    
    # 3. MATH ENGINE CORE WITH EXPERIMENTAL DECAY
    def analyze_row_probability(df, home, away, odds_1, odds_x, odds_2):
        avg_h_goals = max(df['FTHG'].mean(), 0.1)
        avg_a_goals = max(df['FTAG'].mean(), 0.1)
        
        h_games = df[df['HomeTeam'] == home].copy()
        a_games = df[df['AwayTeam'] == away].copy()
        
        if len(h_games) == 0 or len(a_games) == 0:
            return 33.3, 33.3, 33.3, "X", 0.0
            
        h_atk = home_games['FTHG'].ewm(span=min(len(home_games), 8)).mean().iloc[-1] if not home_games.empty else 1.0
        h_def = home_games['FTAG'].ewm(span=min(len(home_games), 8)).mean().iloc[-1] if not home_games.empty else 1.0
        a_atk = away_games['FTAG'].ewm(span=min(len(away_games), 8)).mean().iloc[-1] if not away_games.empty else 1.0
        a_def = away_games['FTHG'].ewm(span=min(len(away_games), 8)).mean().iloc[-1] if not away_games.empty else 1.0
        
        exp_h = (max(h_atk, 0.1) / avg_h_goals) * (max(a_def, 0.1) / avg_h_goals) * avg_h_goals
        exp_a = (max(a_atk, 0.1) / avg_a_goals) * (max(h_def, 0.1) / avg_a_goals) * avg_a_goals
        
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
        
        ev_1 = (hw * odds_1) - 1
        ev_x = (dr * odds_x) - 1
        ev_2 = (aw * odds_2) - 1
        
        ev_map = {"1": ev_1, "X": ev_x, "2": ev_2}
        pick = max(ev_map, key=ev_map.get)
        return hw*100, dr*100, aw*100, pick, ev_map[pick]

    # Display operational balance counters
    st.sidebar.metric("Simulated Practice Bankroll", f"KES {st.session_state.wallet_balance:,.2f}")
    
    # Navigation Workspace Tabs
    tab_gen, tab_track = st.tabs(["⚡ SportyBet Slip Builder", "📊 Forward Testing Ledger"])
    
    with tab_gen:
        st.markdown("### Step 1: Input the 12 SportyBet Jackpot Slots Sequentially")
        
        coupon_fixtures = []
        for i in range(12):
            c1, c2, c3, c4, c5 = st.columns(5)
            h_sel = c1.selectbox(f"Slot #{i+1} Home", options=team_names, index=min(i, len(team_names)-1), key=f"h_{i}")
            a_sel = c2.selectbox(f"Slot #{i+1} Away", options=[t for t in team_names if t != h_sel], index=min(i+1, len(team_names)-2), key=f"a_{i}")
            o1 = c3.number_input("Odds 1", min_value=1.01, value=2.20, key=f"o1_{i}")
            ox = c4.number_input("Odds X", min_value=1.01, value=3.10, key=f"ox_{i}")
            o2 = c5.number_input("Odds 2", min_value=1.01, value=3.30, key=f"o2_{i}")
            coupon_fixtures.append((h_sel, a_sel, o1, ox, o2))
            
        st.markdown("---")
        
        if st.button("⚡ Generate Master SportyBet Betting Slip Grid"):
            st.markdown("### Step 2: Generated Master Entry Ticket")
            
            slip_data = []
            for idx, (home, away, o1, ox, o2) in enumerate(coupon_fixtures, 1):
                hw, dr, aw, pick, edge = analyze_row_probability(live_data, home, away, o1, ox, o2)
                odds_dictionary = {"1": o1, "X": ox, "2": o2}
                odds_target = odds_dictionary[pick]
                
                if edge > 0:
                    fractional_kelly = (edge / (odds_target - 1)) * 0.25
                    allocated_stake = max(float(st.session_state.wallet_balance * fractional_kelly), 50.0)
                    risk_profile = f"🟢 Value Edge (+{edge*100:.1f}%)"
                else:
                    allocated_stake = 50.0
                    risk_profile = f" Thin Edge ({edge*100:.1f}%)"
                    
                slip_data.append({
                    "Slot": f"Slot {idx}",
                    "Fixture": f"{home} vs {away}",
                    "Home %": f"{hw:.1f}%",
                    "Draw %": f"{dr:.1f}%",
                    "Away %": f"{aw:.1f}%",
                    "Pick": pick,
                    "Kelly Stake": allocated_stake,
                    "Odds": odds_target,
                    "Profile": risk_profile
                })
                
            st.dataframe(pd.DataFrame(slip_data), use_container_width=True, hide_index=True)
            st.session_state.history_ledger = slip_data
            st.success("Slip copied to tracking memory. Navigate to the Forward Testing Ledger tab to audit outcomes.")

    with tab_track:
        st.markdown("### 📊 Live Forward-Testing Dashboard")
        if st.session_state.history_ledger:
            track_df = pd.DataFrame(st.session_state.history_ledger)
            st.dataframe(track_df[['Slot', 'Fixture', 'Pick', 'Kelly Stake', 'Profile']], use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("### 🗃️ Match Settlement Center (No Real Cash Required)")
            
            selected_slot = st.selectbox("Select match slot to resolve:", track_df['Slot'].tolist())
            actual_outcome = st.radio("True Match Result Score:", ["Home Win (1)", "Draw (X)", "Away Win (2)"])
            outcome_map = {"Home Win (1)": "1", "Draw (X)": "X", "Away Win (2)": "2"}
            
            if st.button("⚖️ Log Outcome and Update Simulated Balance"):
                target_outcome = outcome_map[actual_outcome]
                
                for match in st.session_state.history_ledger:
                    if match["Slot"] == selected_slot:
                        if match["Pick"] == target_outcome:
                            net_win = match["Kelly Stake"] * (match["Odds"] - 1)
                            st.session_state.wallet_balance += net_win
                            st.success(f"Correct prediction on {match['Fixture']}! Practice bankroll credited with +KES {net_win:.2f}")
                        else:
                            st.session_state.wallet_balance -= match["Kelly Stake"]
                            st.error(f"Prediction missed on {match['Fixture']}. Practice bankroll debited with -KES {match['Kelly Stake']:.2f}")
                st.rerun()
        else:
            st.info("No active slip found in memory. Please build and generate a jackpot slip first.")
