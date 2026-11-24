import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import core.db as db
import core.fixtures as fixtures
import core.predict as predict
import core.tax as tax
import core.camouflage as camo
import core.settle as settle
import core.team_resolve as team_resolve

try:
    from db_init import main as init_db
    init_db()
except Exception:
    pass

st.set_page_config(page_title="Bet Engine", layout="wide", initial_sidebar_state="collapsed")

st.title("⚽ Bet Engine")
st.markdown("A free, self-hosted football betting analytics system.")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["PAPER", "LIVE"], index=0)
    st.markdown("---")
    
    if st.button("Refresh Upcoming Fixtures"):
        with st.spinner("Downloading latest matches..."):
            df = pd.DataFrame()
            
            # 1. Try Live API first
            try:
                import core.live_api as live_api
                df = live_api.fetch_live_fixtures()
            except Exception:
                pass
            
            # 2. Fallback to CSV
            if df.empty:
                try:
                    df = fixtures.download(days_back=7, max_seasons=1)
                except Exception:
                    pass
            
            if not df.empty:
                predict.run(df, source="manual")
                st.success(f"Loaded {len(df)} matches and generated predictions!")
            else:
                st.error("No upcoming fixtures found.")

    if st.button("Settle Results"):
        with st.spinner("Checking ESPN for final scores..."):
            n = settle.settle_all()
            st.success(f"Settled {n} matches.")

    st.markdown("---")
    with st.expander("➕ Add a Team not in the list"):
        new_team = st.text_input("Team Name (exactly as SportyBet shows it)")
        if st.button("Save Team"):
            if new_team:
                team_resolve.add_custom(new_team)
                st.success(f"Added {new_team}!")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Daily Bets", "Jackpot", "Learning", "Bankroll", "Tipsters"])

with tab1:
    st.header("Daily Recommendations")
    preds = db.q("SELECT * FROM predictions WHERE result='Pending' ORDER BY match_date ASC")
    if not preds:
        st.info("No upcoming matches loaded. Click 'Refresh Upcoming Fixtures'.")
    else:
        for p in preds:
            if p['decision'] == 'BET':
                st.info(f"**BET** | {p['match_date']} | {p['home']} vs {p['away']} | Pick: {p['selection']} | Odds: {p['odds']:.2f} | Stake: KES {p['stake']}")
            else:
                st.text(f"SKIP | {p['match_date']} | {p['home']} vs {p['away']} (No edge/No odds)")

with tab2:
    st.header("Sporty 13 Jackpot")
    jackpot_input = st.text_area("Paste 13 matches here (Format: Home vs Away 1 X 2):", height=250)
    if jackpot_input:
        st.success("Matches loaded! Ticket builder logic coming in V2.")

with tab3:
    st.header("Learning & Post-Mortem")

with tab4:
    st.header("Bankroll")
    st.metric("Current Paper Balance", f"KES {float(db.get('bankroll', 5000)):,.0f}")

with tab5:
    st.header("Tipsters")
    st.text("Tipster tracking coming soon.")