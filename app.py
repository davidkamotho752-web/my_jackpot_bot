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
            try:
                import core.live_api as live_api
                df = live_api.fetch_live_fixtures()
            except Exception:
                pass
            
            if df.empty:
                try:
                    df = fixtures.download(days_back=7, max_seasons=1)
                except Exception:
                    pass
            
            if not df.empty:
                predict.run(df, source="manual")
                st.success(f"Loaded {len(df)} matches and generated predictions!")
            else:
                st.error("No upcoming fixtures found in either source.")

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
    with st.expander("ℹ️ How to use this / Why this matters"):
        st.write("This panel shows matches in the future that passed the model's edge threshold.")
        
    preds = db.q("SELECT * FROM predictions WHERE result='Pending' ORDER BY match_date ASC")
    if not preds:
        st.info("No upcoming predictions. Click 'Refresh Upcoming Fixtures' in the sidebar.")
    else:
        bets = [p for p in preds if p['decision'] == 'BET']
        skips = [p for p in preds if p['decision'] == 'SKIP']
        
        if not bets:
            st.success("No qualifying upcoming bets. SKIP is the expected answer most days.")
        else:
            for b in bets:
                action = camo.required_action(b, len(bets), 0)
                st.info(f"**{b['match_date']}** | **{b['home']} vs {b['away']}** | {b['selection']} | Odds: {b['odds']:.2f} | {action}")
        
        if skips:
            with st.expander("Show matches checked but skipped (no edge or no odds)"):
                for s in skips:
                    st.text(f"{s['match_date']} | {s['home']} vs {s['away']} - SKIP")

with tab2:
    st.header("Sporty 13 Jackpot")
    with st.expander("ℹ️ How to use this / Why this matters"):
        st.write("Fill the 13 slots in SportyBet's order, or paste all 13 at once.")
    
    if "sj_df" not in st.session_state:
        st.session_state.sj_df = pd.DataFrame({
            "Home": [""] * 13, "Away": [""] * 13, 
            "Odds 1": [np.nan] * 13, "Odds X": [np.nan] * 13, "Odds 2": [np.nan] * 13
        }, index=range(1, 14))
    
    with st.expander("Faster: paste all 13 at once"):
        txt = st.text_area("Home vs Away 1 X 2", height=100, key="sj_paste")
        if st.button("Fill the 13 slots from this"):
            if txt:
                lines = [l.strip() for l in txt.split('\n') if l.strip()]
                rows = []
                for line in lines[:13]:
                    parts = line.replace(" vs ", " ").split()
                    if len(parts) >= 5:
                        try:
                            rows.append({
                                "Home": parts[0], "Away": parts[1], 
                                "Odds 1": float(parts[2]), "Odds X": float(parts[3]), "Odds 2": float(parts[4])
                            })
                        except ValueError:
                            pass
                
                if rows:
                    df = pd.DataFrame(rows)
                    for _ in range(13 - len(df)):
                        df.loc[len(df)] = {"Home": "", "Away": "", "Odds 1": np.nan, "Odds X": np.nan, "Odds 2": np.nan}
                    df.index = range(1, 14)
                    st.session_state.sj_df = df
                    st.rerun()

    custom_ = list(db.get("custom_teams", []) or [])
    known_ = set([r['home'] for r in db.q("SELECT DISTINCT home FROM predictions")] + [r['away'] for r in db.q("SELECT DISTINCT away FROM predictions")])
    opts_ = [""] + sorted(list(known_) + custom_)
    
    cfg_ = {c: st.column_config.NumberColumn(c, min_value=1.01, format="%.2f") for c in ("Odds 1", "Odds X", "Odds 2")}
    for c in ("Home", "Away"):
        cfg_[c] = st.column_config.SelectboxColumn(c, options=opts_, required=False, width="medium") if len(opts_) > 1 else st.column_config.TextColumn(c)

    ed = st.data_editor(st.session_state.sj_df, key="sj_editor", num_rows="fixed", hide_index=False, column_config=cfg_, height=35 * 14 + 3)
    
    odds_ok = ed[["Odds 1", "Odds X", "Odds 2"]].apply(pd.to_numeric, errors="coerce").fillna(0) > 1
    complete = (ed["Home"].astype(str).str.strip() != "") & (ed["Away"].astype(str).str.strip() != "") & odds_ok.all(axis=1)
    todo = [int(i) for i in ed.index[~complete]]
    
    if todo:
        st.info(f"{13 - len(todo)} of 13 slots complete. Still to fill: slot {', '.join(map(str, todo))}.")
    else:
        st.success("All 13 slots complete! Ticket builder logic coming in V2.")

with tab3:
    st.header("Learning & Post-Mortem")
    st.info("As matches finish and you click 'Settle Results', the engine analyzes why it won or lost here.")

with tab4:
    st.header("Bankroll & Camouflage")
    st.metric("Current Paper Balance", f"KES {float(db.get('bankroll', 5000)):,.0f}")
    camo_log = db.q("SELECT * FROM camouflage_log ORDER BY id DESC LIMIT 5")
    if camo_log:
        st.dataframe(pd.DataFrame(camo_log)[['created', 'match', 'stake', 'reason']])

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