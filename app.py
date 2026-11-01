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

from core import budget
from core import daily
from core import engine as E
from core import oddsapi
from core import oddspapi
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

# ----------------------------------------------------------------------------- sharp odds (free tiers)
def _secret(name):
    v = os.environ.get(name, "")
    if not v:
        try:
            v = st.secrets.get(name, "")
        except Exception:
            v = ""
    return v


PROVIDERS = {
    "OddsPapi - Pinnacle prices (free: 250 requests a month)":
        {"id": oddspapi.PROVIDER, "total": oddspapi.QUOTA_FREE, "reserve": oddspapi.RESERVE, "env": "ODDSPAPI_KEY"},
    "The Odds API (free: 500 credits a month, no Pinnacle)":
        {"id": oddsapi.PROVIDER, "total": oddsapi.QUOTA_FREE, "reserve": 50, "env": "ODDS_API_KEY"},
}

with st.sidebar:
    st.header("Sharp odds (free)")
    pname = st.selectbox("Provider", list(PROVIDERS))
    PV = PROVIDERS[pname]
    okey = st.text_input("API key (this session only)", type="password", value=_secret(PV["env"]))
    jpr = int(st.number_input("Requests held back per jackpot round", 1, 10, 3,
                              help="Kept in reserve for each Saturday still to come this month, so daily use can never starve the jackpot."))
    pl = budget.plan(db, PV["id"], PV["total"], now, PV["reserve"], jpr)
    m1, m2 = st.columns(2)
    m1.metric("Used", f"{pl['used']} / {PV['total']}")
    m2.metric("Usable left", pl["left"])
    m3, m4 = st.columns(2)
    m3.metric("Held for jackpot", pl["jackpot_hold"], f"{pl['rounds_left']} round(s) left", delta_color="off")
    m4.metric("Daily allowance", f"{pl['daily_allowance']:.1f}", f"{pl['daily_today_left']} left today", delta_color="off")
    st.caption(f"{PV['reserve']} requests are never touched. Each call is counted as 1 here until you enter the provider dashboard's own figure below. Odds are cached for 6 hours, so reloading is free.")
    with st.expander("Match my count to the provider dashboard"):
        td = st.number_input("Requests remaining (dashboard)", 0, PV["total"], max(PV["total"] - pl["used"], 0))
        if st.button("Save dashboard figure"):
            budget.true_up(db, PV["id"], td, now)
            st.rerun()
    if PV["id"] == oddspapi.PROVIDER:
        saved_t = db.get("oddspapi_tournaments")
        tl = [tuple(x) for x in saved_t["items"]] if saved_t else []
        if okey and not tl and st.button("Load tournament list (1 request, kept 30 days)"):
            try:
                oddspapi.list_tournaments(db, okey, now)
            except Exception as e:
                st.error(f"Could not load tournaments: {e}")
            st.rerun()
        sharp_choice = st.multiselect("Tournaments to fetch together (1 request for the batch)", [t for t in tl],
                                      format_func=lambda k: dict(tl).get(k, k), key="op_choice")
        dbg = db.get("oddspapi_debug")
        if dbg:
            with st.expander("What the API returned (send this if something shows 0 fixtures)"):
                st.caption(dbg["note"])
                st.code(dbg["head"], language="json")
    else:
        if okey and st.button("Load competitions (free)"):
            try:
                st.session_state.comps = oddsapi.list_soccer(okey)
            except Exception as e:
                st.error(f"Could not load competitions: {e}")
        comps = st.session_state.get("comps", [])
        sharp_choice = st.multiselect("Competitions (1 credit each)", [c for c in comps],
                                      format_func=lambda k: dict(comps).get(k, k), key="oa_choice")
    if okey and sharp_choice and st.button("Fetch chosen now"):
        st.session_state.run_fetch = ("daily", [])
    cs = oddsapi.cache_summary(db, now)
    if cs:
        st.write("Cached (used automatically):")
        st.dataframe(pd.DataFrame(cs), hide_index=True)


def fetch_sharp(purpose, extra=()):
    """Fetch the chosen (plus suggested) tournaments/competitions. Returns messages to show."""
    if not okey:
        return ["Add an API key in the sidebar, or type reference odds into the Ref columns (free)."]
    msgs = []
    try:
        if PV["id"] == oddspapi.PROVIDER:
            ids = list(dict.fromkeys(list(st.session_state.get("op_choice", [])) + list(extra)))
            if not ids:
                return ["Pick tournaments in the sidebar first (load the list once)."]
            ev, src, age = oddspapi.fetch_tournaments(db, okey, ids, purpose, now, total=PV["total"],
                                                      reserve=PV["reserve"], jackpot_per_round=jpr)
            msgs.append(f"{len(ev)} fixtures with Pinnacle 1X2 - " +
                        ("1 request used." if src == "api" else f"cached {age:.1f}h ago, no request used."))
            if not ev:
                msgs.append("Parsed 0 fixtures. Open 'What the API returned' in the sidebar.")
        else:
            keys = list(dict.fromkeys(list(st.session_state.get("oa_choice", [])) + list(extra)))
            if not keys:
                return ["Load and pick competitions in the sidebar first."]
            for sk in keys:
                _, src, age = oddsapi.fetch_h2h_cached(db, okey, sk, now=now, purpose=purpose,
                                                       total=PV["total"], reserve=PV["reserve"])
                msgs.append(f"{sk}: " + ("1 credit used." if src == "api" else f"cached ({age:.1f}h), free."))
    except budget.BudgetError as e:
        msgs.append(str(e))
    except Exception as e:
        msgs.append(f"Request failed: {e}")
    return msgs


def suggested(results):
    """Provider-specific ids/keys that cover the analysed matches (best effort)."""
    if PV["id"] == oddspapi.PROVIDER:
        saved_t = db.get("oddspapi_tournaments")
        tl = [tuple(x) for x in saved_t["items"]] if saved_t else []
        names = {E.LEAGUES[r["league"]] for r in results if r.get("league") in E.LEAGUES}
        return oddspapi.suggest_tournaments(names, tl)
    return oddsapi.competitions_for(results, [k for k, _ in st.session_state.get("comps", [])])


def make_anchors(frame):
    """Sharp references for the rows: cached API odds (free) plus any Ref 1/X/2 typed in by hand."""
    an = oddsapi.build_anchors(zip(frame["Home"], frame["Away"]), oddsapi.cached_events(db, now))
    for _, rr in frame.iterrows():
        ref = np.array([pd.to_numeric(rr.get(k), errors="coerce") for k in ("Ref 1", "Ref X", "Ref 2")], float)
        if np.isfinite(ref).all() and (ref > 1).all():
            inv = 1.0 / ref
            an[(E.norm(rr["Home"]), E.norm(rr["Away"]))] = {"probs": inv / inv.sum(), "source": "manual reference", "n_books": 1}
    return an


def run_pending_fetch():
    job = st.session_state.pop("run_fetch", None)
    if job:
        st.session_state.flash = fetch_sharp(job, job)
        st.rerun()


run_pending_fetch()

# ----------------------------------------------------------------------------- header
st.title("Bet Engine")
c1, c2, c3 = st.columns(3)
c1.metric("Mode", mode)
c2.metric("Balance", kes(state["balance"]))
c3.metric("Drawdown", f"{state['drawdown']:.1%}", f"stop at {lim['drawdown_pct']:.0%}", delta_color="off")
if mode == "PAPER":
    st.info("PAPER mode: recommendations are recorded but no real money is involved. LIVE unlocks only after the proof gate passes (Learning tab).")
for reason in gstat["reasons"]:
    st.error(reason + " No new bets are recommended.")
if data.empty:
    st.warning("Results could not be downloaded, so there is no model: every match will be SKIP. Check your internet connection.")

tab_today, tab_jp, tab_bank, tab_learn = st.tabs(["Today", "Jackpot", "Bankroll", "Learning"])

# ============================================================================= TODAY
with tab_today:
    st.subheader("1. Paste today's matches")
    st.caption("From SportyBet, one match per line:  Girona vs Getafe 2.30 3.00 3.40   (home, away, then 1 X 2 odds)")
    pasted = st.text_area("Matches and odds", height=130, key="d_paste", label_visibility="collapsed")
    ALL_COLS = ["Home", "Away", "Odds 1", "Odds X", "Odds 2", "Odds 1X", "Odds X2", "Odds 12", "Ref 1", "Ref X", "Ref 2", "Shift (pts)"]
    
    # FIXED: Re-enforced exact 4-space indentation alignment inside this layout branch
    if "d_df" not in st.session_state:
