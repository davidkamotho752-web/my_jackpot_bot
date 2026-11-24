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
            df = fixtures.download(days_back=0, max_seasons=1)
            if not df.empty:
                predict.run(df, source="manual")
                st.success("Data updated and upcoming predictions generated!")
            else:
                st.error("No upcoming fixtures found in the free database right now.")
                
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
    st.header("Daily Recommendations (Only Upcoming)")
    with st.expander("ℹ️ How to use this / Why this matters"):
        st.write("This panel shows ONLY matches in the future that passed the model's edge threshold. If a match was played yesterday, it is hidden.")
        
    preds = db.q("SELECT * FROM predictions WHERE result='Pending' ORDER BY match_date ASC")
    if not preds:
        st.info("No upcoming predictions. Click 'Refresh Upcoming Fixtures' in the sidebar.")
    else:
        df_preds = pd.DataFrame(preds)
        bets = df_preds[df_preds['decision'] == 'BET']
        
        if bets.empty:
            st.success("No qualifying upcoming bets. SKIP is the expected answer most days.")
        else:
            for _, b in bets.iterrows():
                bet_dict = b.to_dict()
                action = camo.required_action(bet_dict, len(bets), 0)
                # Shows exact Date and Time
                st.info(f"**{b['match_date']}** | **{b['home']} vs {b['away']}** | {b['selection']} | Odds: {b['odds']:.2f} | {action}")

with tab2:
    st.header("Sporty 13 Jackpot")
    with st.expander("ℹ️ How to use this / Why this matters"):
        st.write("Paste the 13 matches from SportyBet here. The engine builds the best ticket.")
    st.text_area("Paste 13 matches here (coming soon):", height=150, disabled=True)

with tab3:
    st.header("Learning & Post-Mortem")
    st.info("As matches finish and you click 'Settle Results', the engine analyzes why it won or lost here.")

with tab4:
    st.header("Bankroll & Camouflage")
    st.metric("Current Paper Balance", f"KES {float(db.get('bankroll', 5000)):,.0f}")

with tab5:
    st.header("Tipsters & Tracking")
    with st.expander("ℹ️ How to use this / Why this matters"):
        st.write("No free tipster APIs exist. Use the dropdown below to open their site, find a pick, and log it manually *before* the match.")
    
    tipster_site = st.selectbox("🌐 Open a platform:", ["Select...", "Bet2Invest", "Oddspedia", "Tipstrr", "OLBG"])
    if tipster_site == "Bet2Invest":
        st.markdown("👉 **[Open Bet2Invest](https://bet2invest.ca/tipsters/football)**")
    elif tipster_site == "Oddspedia":
        st.markdown("👉 **[Open Oddspedia](https://oddspedia.com/tips/tipsters-ranking)**")
        
    st.markdown("---")
    with st.form("tipster_form"):
        st.subheader("Log a Tipster Pick")
        t_name = st.text_input("Tipster Name")
        col1, col2 = st.columns(2)
        t_home = col1.text_input("Home Team")
        t_away = col2.text_input("Away Team")
        col3, col4, col5 = st.columns(3)
        t_date = col3.text_input("Date (YYYY-MM-DD)")
        t_sel = col4.selectbox("Pick", ["1 (Home)", "X (Draw)", "2 (Away)"])
        t_odds = col5.number_input("Odds", min_value=1.01, value=2.00)
        
        if st.form_submit_button("Save Pick"):
            sel_code = t_sel.split(" ")[0]
            db.insert("tipster_picks", {
                "created": db.iso(db.now_local()), 
                "tipster": t_name,
                "home_in": t_home, "away_in": t_away,
                "selection": sel_code, "odds": t_odds, "match_date": t_date
            })
            st.success(f"Saved {t_name}'s pick for {t_home} vs {t_away}!")