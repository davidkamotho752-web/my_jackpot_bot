"""Bet Engine - daily match analysis + Sporty 13 jackpot, with bankroll protection.
Run:  streamlit run app.py
"""
import hashlib
import json
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from core import daily
from core import engine as E
from core import oddsapi
from core import risk
from core import store

st.set_page_config(page_title="Bet Engine", layout="centered")


# ----------------------------------------------------------------------------- resources
@st.cache_resource
def get_db():
    path = os.environ.get("ENGINE_DB", "data/engine.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return store.DB(path)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_data():
    return E.download_football_data(3)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_models(data, xi):
    return E.fit_all(data, xi)


def kes(x):
    return f"KES {x:,.0f}"


db = get_db()
now = store.now_local()
if db.get("paper_init") is None:
    risk.add_ledger(db, "PAPER", "deposit", 10000.0, "starting paper bankroll", now)
    db.set("paper_init", True)
risk.auto_drawdown_switch(db, now)
mode = risk.get_mode(db)
lim = risk.get_limits(db, now)
tax = risk.get_tax(db)
params = db.get("params", {"xi": 0.002, "w": 0.6})
state = risk.bankroll_state(db, mode, now)
gstat = risk.guard_status(state, lim)

with st.spinner("Loading results (cached for 6 hours)..."):
    data, failed = load_data()
if data.empty:
    models, team_league = {}, {}
else:
    with st.spinner("Fitting team-strength models (first run takes a minute or two)..."):
        models = load_models(data, params["xi"])
    team_league = E.team_league_map(data)

# ----------------------------------------------------------------------------- header
st.title("Bet Engine")
c1, c2, c3 = st.columns(3)
c1.metric("Mode", mode)
c2.metric("Balance", kes(state["balance"]))
c3.metric("Drawdown", f"{state['drawdown']:.1%}", f"stop at {lim['drawdown_pct']:.0%}", delta_color="off")
if mode == "PAPER":
    st.info("PAPER mode: recommendations are recorded but no real money is involved. "
            "LIVE unlocks only after the proof gate passes (Learning tab).")
for reason in gstat["reasons"]:
    st.error(reason + " No new bets are recommended.")
if data.empty:
    st.warning("Results could not be downloaded, so there is no model: every match will be SKIP. "
               "Check your internet connection.")

tab_today, tab_jp, tab_bank, tab_learn = st.tabs(["Today", "Jackpot", "Bankroll", "Learning"])

# ============================================================================= TODAY
with tab_today:
    st.subheader("1. Paste today's matches")
    st.caption("From SportyBet, one match per line:  Girona vs Getafe 2.30 3.00 3.40   (home, away, then 1 X 2 odds)")
    pasted = st.text_area("Matches and odds", height=130, key="d_paste", label_visibility="collapsed")
    ALL_COLS = ["Home", "Away", "Odds 1", "Odds X", "Odds 2", "Odds 1X", "Odds X2", "Odds 12",
                "Over 2.5", "Under 2.5", "Shift (pts)"]
    if "d_df" not in st.session_state:
        st.session_state.d_df = pd.DataFrame({c: ([""] * 8 if c in ("Home", "Away") else [np.nan] * 8) for c in ALL_COLS})
        st.session_state.d_ver = 0
    if st.button("Load matches"):
        rows, bad = E.parse_fixtures(pasted)
        if rows:
            df = pd.DataFrame(rows)
            for c in ALL_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            st.session_state.d_df = df[ALL_COLS]
            st.session_state.d_ver += 1
        if bad:
            st.warning("Could not read these lines:\n\n" + "\n".join(f"- {b}" for b in bad))
        if not rows:
            st.error("No valid lines. Format: Home vs Away 2.30 3.00 3.40")
    more = st.checkbox("Also enter double chance and over/under 2.5 odds")
    order = ALL_COLS if more else ["Home", "Away", "Odds 1", "Odds X", "Odds 2"]
    cfg = {c: st.column_config.NumberColumn(c, min_value=1.01, format="%.2f")
           for c in ("Odds 1", "Odds X", "Odds 2", "Odds 1X", "Odds X2", "Odds 12", "Over 2.5", "Under 2.5")}
    cfg["Shift (pts)"] = st.column_config.NumberColumn("Shift (pts)", min_value=-20.0, max_value=20.0, step=1.0)
    edited = st.data_editor(st.session_state.d_df, key=f"d_ed_{st.session_state.d_ver}", num_rows="dynamic",
                            column_order=order, column_config=cfg, hide_index=True)
    st.session_state.d_df = edited

    valid = edited[(edited["Home"].astype(str).str.strip() != "") & (edited["Away"].astype(str).str.strip() != "") &
                   (edited[["Odds 1", "Odds X", "Odds 2"]].apply(pd.to_numeric, errors="coerce").fillna(0) > 1).all(axis=1)]

    with st.expander("Optional: sharp reference odds (free API key, hard monthly budget)"):
        st.caption("A sharp bookmaker's price is a better yardstick than SportyBet's own. The Odds API free plan gives "
                   "500 credits a month; this tool refuses to use more than the cap below. Get a key at the-odds-api.com. "
                   "It is held only for this session and never saved.")
        key = st.text_input("The Odds API key", type="password", value=os.environ.get("ODDS_API_KEY", ""))
        cap = st.number_input("Monthly credit cap", 10, 500, oddsapi.DEFAULT_MONTHLY_CAP)
        st.write(f"Credits used this month by this tool: **{oddsapi.credits_used(db, now)}** of {cap}")
        if key and st.button("Load competitions (free)"):
            try:
                st.session_state.comps = oddsapi.list_soccer(key)
            except Exception as e:
                st.error(f"Could not load competitions: {e}")
        comps = st.session_state.get("comps", [])
        chosen = st.multiselect("Competitions to fetch (1 credit each)", [c[0] for c in comps],
                                format_func=lambda k: dict(comps).get(k, k))
        if key and chosen and st.button(f"Fetch odds ({len(chosen)} credits)"):
            evs = []
            for sk in chosen:
                try:
                    evs += oddsapi.fetch_h2h(db, key, sk, cap=int(cap), now=now)
                except oddsapi.BudgetError as e:
                    st.warning(str(e))
                    break
                except Exception as e:
                    st.error(f"{sk}: {e}")
            st.session_state.events = evs
            st.success(f"Fetched {len(evs)} upcoming events.")

    if valid.empty:
        st.info("Add at least one match with 1 X 2 odds to see recommendations.")
    else:
        anchors = oddsapi.build_anchors(zip(valid["Home"], valid["Away"]), st.session_state.get("events", []))
        ctx = daily.Context(models, team_league, params["w"], tax, lim, state["balance"], anchors)
        results = [daily.analyse_match(r, ctx) for _, r in valid.iterrows()]
        risk.apply_guards(results, state, lim)

        st.subheader("2. Recommendations")
        st.caption(f"{'Paper' if mode == 'PAPER' else 'Live'} stakes. Singles only. "
                   f"{state['bets_today']} of {lim['max_bets_day']} bets used today.")
        table = []
        for x in results:
            table.append({
                "Match": f"{x['home_in']} v {x['away_in']}", "Call": x["decision"],
                "Pick": daily.SEL_LABEL.get(x["selection"], "-") if x["decision"] == "BET" else
                        (f"best was {x['selection']}" if x["selection"] else "-"),
                "Take if odds at least": f"{x['min_odds']:.2f}" if x["min_odds"] and x["decision"] == "BET" else "-",
                "Stake": kes(x["stake"]) if x["decision"] == "BET" else "-",
                "Net edge": f"{x['edge_low']:+.1%}" if x["selection"] else "-",
                "Why": x["reason"],
            })
        st.dataframe(pd.DataFrame(table), hide_index=True)
        bets = [x for x in results if x["decision"] == "BET"]
        if bets:
            lines = [f"{x['home_in']} vs {x['away_in']} | {daily.SEL_LABEL[x['selection']]} | "
                     f"take only if odds >= {x['min_odds']:.2f} | stake KES {x['stake']:.0f}" for x in bets]
            st.markdown("**Slip to enter in SportyBet by hand**" + (" (paper: do not stake)" if mode == "PAPER" else ""))
            st.code("\n".join(lines), language=None)
        else:
            st.success("No bets today. SKIP is the expected answer most days.")

        probs = pd.DataFrame([{"Match": f"{x['home_in']} v {x['away_in']}",
                               "1 %": round(x["final"][0] * 100, 1), "X %": round(x["final"][1] * 100, 1),
                               "2 %": round(x["final"][2] * 100, 1),
                               "Reference": x["ref_source"],
                               "Model": "yes" if x["model"] is not None else "no"} for x in results])
        with st.expander("Probabilities and data used"):
            st.dataframe(probs, hide_index=True)
        with st.expander("Why so many SKIPs?"):
            st.markdown(
                f"- A price is only worth taking if the model's cautious probability beats it by at least "
                f"**{lim['min_edge']:.0%}** after allowing for uncertainty. SportyBet's margin (shown in the "
                f"probabilities table's source odds) means fair-looking prices fail this test.\n"
                f"- Deposit and withdrawal excise are charged on money moving in and out. With {tax['deposit']:.0%} each "
                f"way your wallet must grow about **{daily.flow_hurdle(tax):.1%}** in total just to break even after "
                f"cashing out. Bet-by-bet this is not charged unless you switch on 'tax on every bet' in Bankroll settings.\n"
                f"- With no sharp reference, the model competes against the same bookmaker whose price you take. "
                f"A strong edge from that alone is rare.")

        sig = hashlib.md5(json.dumps([[x["home_in"], x["away_in"], x["entered"]] for x in results],
