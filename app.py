"""Bet Engine - daily match analysis + Sporty 13 jackpot, with bankroll protection.
Run:  streamlit run app.py
"""
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta

# Works with the engine files at the top level OR moved into a folder called "core".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

import numpy as np
import pandas as pd
import streamlit as st

import audit
import backtest
import budget
import daily
import engine as E
import gaps
import oddsapi
import oddspapi
import postmortem
import risk
import scan
import store
import tipsters

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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_long():
    return E.download_football_data(6)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_models(data, xi):
    return E.fit_all(data, xi)


def kes(x):
    return f"KES {x:,.0f}"


db = get_db()
now = store.now_local()
gaps.load_aliases(db)
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
    load_data.clear()                                   # never keep an empty download in the cache: the next run retries
if not data.empty and not st.session_state.get("auto_settled"):
    st.session_state.auto_settled = True              # once per session, so no rerun loop
    try:
        _np, _nb = daily.settle_all(db, data, now)
        risk.auto_drawdown_switch(db, now)
        _nt = tipsters.settle(db, data, now)
        if _np or _nb or _nt:
            st.toast(f"Auto-settled {_np} matches, {_nb} bets and {_nt} tipster picks from the latest results.")
            st.rerun()
    except Exception as _e:                            # settling must never stop the app from opening
        st.warning(f"Could not auto-settle results: {_e}")
if data.empty:
    models, team_league = {}, {}
else:
    with st.spinner("Fitting team-strength models (first run takes a minute or two)..."):
        models = load_models(data, params["xi"])
    team_league = E.team_league_map(data)

# ----------------------------------------------------------------------------- sharp odds (free tiers)
def _secret(name):
    v = os.environ.get(name, "")
    has_file = any(os.path.exists(p) for p in (os.path.join(".streamlit", "secrets.toml"),
                                                 os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml")))
    if not v and has_file:                     # touching st.secrets with no file shows a red error box on older Streamlit
        try:
            v = st.secrets.get(name, "")
        except Exception:
            v = ""
    return v


PROVIDERS = {
    "The Odds API (free: 500 credits a month; Pinnacle is listed in its EU region)":
        {"id": oddsapi.PROVIDER, "total": oddsapi.QUOTA_FREE, "reserve": 50, "env": "ODDS_API_KEY"},
    "OddsPapi (free: 250 requests a month; Pinnacle prices, newer and less tested)":
        {"id": oddspapi.PROVIDER, "total": oddspapi.QUOTA_FREE, "reserve": oddspapi.RESERVE, "env": "ODDSPAPI_KEY"},
}

with st.sidebar:
    st.header("Sharp odds (free)")
    pname = st.selectbox("Provider", list(PROVIDERS))
    PV = PROVIDERS[pname]
    okey = st.text_input("API key (this session only)", type="password", value=_secret(PV["env"]))
    jpr = int(st.number_input("Requests held back per jackpot round", 1, 15, 3 if PV["id"] == oddspapi.PROVIDER else 8,
                              key=f"jpr_{PV['id']}",
                              help="Kept in reserve for each Saturday still to come this month, so daily use can never starve the jackpot. "
                                   "One request covers 3 tournaments on OddsPapi but only 1 competition on The Odds API, "
                                   "so the Odds API needs a bigger reserve (a jackpot round spans many competitions)."))
    pl = budget.plan(db, PV["id"], PV["total"], now, PV["reserve"], jpr)
    if pl["left"] <= 0:
        st.error("Out of requests for this month. No odds will be fetched. Use the other provider in the list above, "
                 "type reference odds into the Ref columns (free), or wait for the monthly reset. The tool cannot create credits.")
    elif pl["daily_pool"] <= 0:
        st.warning("Only the jackpot reserve is left, so daily matches will not fetch odds. Use the Ref columns (free) or the other provider.")
    st.caption("Running on Streamlit Cloud? A restart clears this counter. Press Load competitions (free) or type the dashboard figure below to correct it.")
    m1, m2 = st.columns(2)
    m1.metric("Used", f"{pl['used']} / {PV['total']}")
    m2.metric("Usable left", pl["left"])
    m3, m4 = st.columns(2)
    m3.metric("Held for jackpot", pl["jackpot_hold"], f"{pl['rounds_left']} round(s) left", delta_color="off")
    m4.metric("Daily allowance", f"{pl['daily_allowance']:.1f}", f"{pl['daily_today_left']} left today", delta_color="off")
    st.caption(f"{PV['reserve']} requests are never touched. Each call is counted as 1 here until you enter the "
               "provider dashboard's own figure below. Odds are cached for 6 hours, so reloading is free.")
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
        sharp_choice = st.multiselect("Tournaments to fetch together (1 request for the batch)", [t[0] for t in tl],
                                      format_func=lambda k: dict(tl).get(k, k), key="op_choice")
        dbg = db.get("oddspapi_debug")
        if dbg:
            with st.expander("What the API returned (send this if something shows 0 fixtures)"):
                st.caption(dbg["note"])
                st.code(dbg["head"], language="json")
    else:
        if okey and st.button("Load competitions (free)"):
            try:
                st.session_state.comps = oddsapi.list_soccer(okey, db=db, now=now)
            except Exception as e:
                st.error(f"Could not load competitions: {e}")
        comps = st.session_state.get("comps", [])
        sharp_choice = st.multiselect("Competitions (1 credit each)", [c[0] for c in comps],
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
            an[(E.norm(rr["Home"]), E.norm(rr["Away"]))] = {"probs": inv / inv.sum(), "source": "manual reference",
                                                           "n_books": 1}
    return an


def run_pending_fetch():
    job = st.session_state.pop("run_fetch", None)
    if job:
        st.session_state.flash = fetch_sharp(job[0], job[1])
        st.rerun()


run_pending_fetch()

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

if failed:
    st.warning(f"{len(failed)} result file(s) could not be downloaded (for example {', '.join(failed[:3])}). "
               + ("The tool still works with the rest." if not data.empty else "Nothing loaded, so every match will be SKIP."))
    if st.button("Try downloading again", key="retry_dl"):
        load_data.clear()
        load_models.clear()
        st.rerun()

for m_ in st.session_state.pop("flash", None) or []:
    st.info(m_)

tab_today, tab_jp, tab_bank, tab_learn, tab_tip = st.tabs(["Today", "Jackpot", "Bankroll", "Learning", "Tipsters"])

# ============================================================================= TODAY
with tab_today:
    st.subheader("0. Shortlist: which upcoming matches qualify? (free, no API credits)")
    sc1, sc2 = st.columns(2)
    scan_days = sc1.selectbox("Look ahead", [1, 2, 3], index=1, format_func=lambda d: f"{d} day(s)")
    scan_margin = sc2.slider("Assume SportyBet is worse than Bet365 by", 0.0, 0.08, 0.03, 0.005, format="%.3f")
    if st.button("Find qualifying matches"):
        fx, fx_err = scan.download_fixtures()
        if fx_err or fx.empty:
            st.session_state.scan_res = None
            st.error(f"Could not read the free fixtures file ({fx_err}). Use the paste box below instead.")
        else:
            rows_, n_ = scan.scan(fx, models, team_league, params, lim, tax, state["balance"], now,
                                  days=int(scan_days), extra_margin=float(scan_margin))
            d0_ = pd.Timestamp(now.date())
            win_ = int(((fx["Date"] >= d0_) & (fx["Date"] < d0_ + pd.Timedelta(days=int(scan_days)))).sum())
            st.session_state.scan_res = {"rows": rows_, "n": n_, "when": now.strftime("%H:%M"), "window": win_}
    sr = st.session_state.get("scan_res")
    if sr:
        if sr.get("window", sr["n"]) > sr["n"]:
            st.caption(f"{sr['window'] - sr['n']} of {sr['window']} fixtures in the window had no Bet365 odds in the free file, "
                       "so they could not be checked. Add them by hand below if you care about them.")
        if not sr["rows"]:
            st.success(f"{sr['n']} matches checked, none qualify. Nothing to look for today.")
        else:
            st.write(f"**{len(sr['rows'])} of {sr['n']} matches qualify.** Open only these on SportyBet and take a pick "
                     "only if its odds are at least the number shown.")
            st.dataframe(pd.DataFrame([{"Date": r["Date"].strftime("%a %d %b"), "Time": r["Time"],
                                        "Match": f"{r['Home']} v {r['Away']}",
                                        "Pick": daily.SEL_LABEL.get(r["Pick"], r["Pick"]),
                                        "Take if SportyBet odds at least": f"{r['Min odds']:.2f}",
                                        "Net edge": f"{r['Net edge']:+.1%}", "Paper stake": kes(r["Stake"])}
                                       for r in sr["rows"]]), hide_index=True)
            if st.button("Send these to the table below (then type SportyBet's odds)"):
                tbl = pd.DataFrame([{"Home": r["Home"], "Away": r["Away"],
                                     **({"Ref 1": r["Ref"][0], "Ref X": r["Ref"][1], "Ref 2": r["Ref"][2]} if r["Ref"] else {})}
                                    for r in sr["rows"]])
                for c in ["Home", "Away", "Odds 1", "Odds X", "Odds 2", "Odds 1X", "Odds X2", "Odds 12", "Over 2.5", "Under 2.5",
                          "Ref 1", "Ref X", "Ref 2", "Shift (pts)"]:
                    if c not in tbl.columns:
                        tbl[c] = np.nan
                st.session_state.d_df = tbl[["Home", "Away", "Odds 1", "Odds X", "Odds 2", "Odds 1X", "Odds X2", "Odds 12",
                                             "Over 2.5", "Under 2.5", "Ref 1", "Ref X", "Ref 2", "Shift (pts)"]]
                st.session_state.d_ver = st.session_state.get("d_ver", 0) + 1
                st.rerun()
        st.caption(f"Scanned at {sr['when']}. Uses Bet365 prices (worsened by the margin above) and Pinnacle as the reference, "
                   "from football-data.co.uk. Its prices are not live, so a shortlisted match is a candidate, not a bet, until "
                   "SportyBet's own odds meet the minimum.")
    st.divider()
    st.subheader("1. Paste today's matches")
    st.caption("From SportyBet, one match per line:  Girona vs Getafe 2.30 3.00 3.40   (home, away, then 1 X 2 odds)")
    pasted = st.text_area("Matches and odds", height=130, key="d_paste", label_visibility="collapsed")
    ALL_COLS = ["Home", "Away", "Odds 1", "Odds X", "Odds 2", "Odds 1X", "Odds X2", "Odds 12",
                "Over 2.5", "Under 2.5", "Ref 1", "Ref X", "Ref 2", "Alt 1", "Alt X", "Alt 2", "Shift (pts)"]
    if "d_df" not in st.session_state:
        st.session_state.d_df = pd.DataFrame({c: ([""] * 8 if c in ("Home", "Away") else [np.nan] * 8) for c in ALL_COLS})
        st.session_state.d_ver = 0
    for c in ALL_COLS:                       # keep older sessions working
        if c not in st.session_state.d_df.columns:
            st.session_state.d_df[c] = np.nan
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
    more = st.checkbox("More columns: double chance, over/under 2.5, reference (sharp) odds, second bookmaker, manual shift")
    alt_name = "Other bookmaker"
    if more:
        alt_name = st.text_input("Name of the second bookmaker (columns Alt 1 / Alt X / Alt 2)", value="Other bookmaker", key="alt_name",
                                 help="Optional. If its price for a pick is higher than SportyBet's, the tool uses it and tells you where.")
    order = ALL_COLS if more else ["Home", "Away", "Odds 1", "Odds X", "Odds 2"]
    cfg = {c: st.column_config.NumberColumn(c, min_value=1.01, format="%.2f")
           for c in ("Odds 1", "Odds X", "Odds 2", "Odds 1X", "Odds X2", "Odds 12", "Over 2.5", "Under 2.5",
                     "Ref 1", "Ref X", "Ref 2", "Alt 1", "Alt X", "Alt 2")}
    cfg["Shift (pts)"] = st.column_config.NumberColumn("Shift (pts)", min_value=-20.0, max_value=20.0, step=1.0)
    edited = st.data_editor(st.session_state.d_df, key=f"d_ed_{st.session_state.d_ver}", num_rows="dynamic",
                            column_order=order, column_config=cfg, hide_index=True)
    st.session_state.d_df = edited

    valid = edited[(edited["Home"].astype(str).str.strip() != "") & (edited["Away"].astype(str).str.strip() != "") &
                   (edited[["Odds 1", "Odds X", "Odds 2"]].apply(pd.to_numeric, errors="coerce").fillna(0) > 1).all(axis=1)]

    st.caption("Sharp reference odds (Pinnacle) and the request budget are in the sidebar. Free alternative: "
               "type a sharp bookmaker's prices into the Ref columns (More columns).")

    if valid.empty:
        st.info("Add at least one match with 1 X 2 odds to see recommendations.")
    else:
        anchors = make_anchors(valid)
        ctx = daily.Context(models, team_league, params["w"], tax, lim, state["balance"], anchors)
        ctx.alt_name = alt_name
        results = [daily.analyse_match(r, ctx) for _, r in valid.iterrows()]
        risk.apply_guards(results, state, lim)

        st.subheader("2. Recommendations")
        st.caption(f"{'Paper' if mode == 'PAPER' else 'Live'} stakes. Singles only. "
                   f"{state['bets_today']} of {lim['max_bets_day']} bets used today.")
        only_bets = st.checkbox("Show only matches that qualify (hide SKIPs)", value=True, key="only_bets")
        table = []
        for x in results:
            if only_bets and x["decision"] != "BET":
                continue
            table.append({
                "Match": f"{x['home_in']} v {x['away_in']}", "Call": x["decision"],
                "Pick": daily.SEL_LABEL.get(x["selection"], "-") if x["decision"] == "BET" else
                        (f"best was {x['selection']}" if x["selection"] else "-"),
                "Where": x.get("book", "SportyBet") if x["decision"] == "BET" else "-",
                "Take if odds at least": f"{x['min_odds']:.2f}" if x["min_odds"] and x["decision"] == "BET" else "-",
                "Stake": kes(x["stake"]) if x["decision"] == "BET" else "-",
                "Net edge": f"{x['edge_low']:+.1%}" if x["selection"] else "-",
                "Why": x["reason"],
            })
        if table:
            st.dataframe(pd.DataFrame(table), hide_index=True)
        elif only_bets:
            st.caption(f"{len(results)} matches checked, none qualify.")
        bets = [x for x in results if x["decision"] == "BET"]
        if bets:
            lines = [f"{x['home_in']} vs {x['away_in']} | {daily.SEL_LABEL[x['selection']]} | "
                     f"take only if odds >= {x['min_odds']:.2f} | stake KES {x['stake']:.0f}"
                     + ("" if x.get("book", "SportyBet") == "SportyBet" else f" | AT {x['book']}") for x in bets]
            st.markdown("**Slip to enter by hand (SportyBet unless a line says otherwise)**" + (" (paper: do not stake)" if mode == "PAPER" else ""))
            st.code("\n".join(lines), language=None)
        else:
            st.success("No bets today. SKIP is the expected answer most days.")

        need = [x for x in results if daily.worth_confirming(x)]
        if need:
            names = ", ".join(f"{x['home_in']} v {x['away_in']}" for x in need[:6])
            ids = suggested(need)
            st.info(f"Worth confirming against a sharp price: {names}.")
            if okey and (ids or sharp_choice):
                if st.button("Confirm now (1 request unless already cached)", key="confirm_daily"):
                    st.session_state.run_fetch = ("daily", ids)
                    st.rerun()
            elif not okey:
                st.caption("To confirm for free: type the sharp prices into the Ref columns, or add a free key in the sidebar.")
            else:
                st.caption("Load the tournament/competition list in the sidebar so matches can be linked to one.")

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
                                     default=str).encode()).hexdigest() + now.strftime("%Y%m%d")
        if st.button("Save this analysis to the log"):
            if sig in st.session_state.get("saved_sigs", set()):
                st.warning("This exact analysis was already saved today.")
            else:
                gid = "D" + now.strftime("%Y%m%d-%H%M%S")
                new_ids = daily.save_analysis(db, results, mode, gid, db.param_version(), now)
                st.session_state.setdefault("saved_sigs", set()).add(sig)
                skipped = len(results) - len(new_ids)
                st.success(f"Saved {len(new_ids)} new matches. They settle automatically in the Learning tab."
                           + (f" {skipped} were already in today's log and were skipped." if skipped else ""))

# ============================================================================= JACKPOT
with tab_jp:
    st.subheader("Sporty 13 jackpot")
    st.caption("KES 50 per combination, closes Saturday 17:00 Kenya time. The 13/13 prize is split between winners.")
    dl = E.next_deadline(now)
    left = dl - now
    st.write(f"Next deadline: **{dl:%A %d %b %H:%M}** ({left.days}d {left.seconds // 3600}h left)")
    jp_paste = st.text_area("Paste the 13 matches in SportyBet order (Home vs Away 1 X 2 odds)", height=130, key="j_paste")
    JC = ["Home", "Away", "Odds 1", "Odds X", "Odds 2", "Crowd 1 %", "Crowd X %", "Crowd 2 %", "Ref 1", "Ref X", "Ref 2", "Shift (pts)"]
    if "j_df" not in st.session_state:
        st.session_state.j_df = pd.DataFrame({c: ([""] * 13 if c in ("Home", "Away") else [np.nan] * 13) for c in JC})
        st.session_state.j_ver = 0
    for c in JC:                             # keep older sessions working
        if c not in st.session_state.j_df.columns:
            st.session_state.j_df[c] = np.nan
    if st.button("Load jackpot matches"):
        rows, bad = E.parse_fixtures(jp_paste)
        if rows:
            df = pd.DataFrame(rows)
            for c in JC:
                if c not in df.columns:
                    df[c] = np.nan
            st.session_state.j_df = df[JC]
            st.session_state.j_ver += 1
        if bad:
            st.warning("Could not read these lines:\n\n" + "\n".join(f"- {b}" for b in bad))
    jcfg = {c: st.column_config.NumberColumn(c, min_value=1.01, format="%.2f") for c in ("Odds 1", "Odds X", "Odds 2", "Ref 1", "Ref X", "Ref 2")}
    jed = st.data_editor(st.session_state.j_df, key=f"j_ed_{st.session_state.j_ver}", num_rows="fixed",
                         column_config=jcfg, hide_index=True)
    st.session_state.j_df = jed
    jv = jed[(jed["Home"].astype(str).str.strip() != "") & (jed["Away"].astype(str).str.strip() != "") &
             (jed[["Odds 1", "Odds X", "Odds 2"]].apply(pd.to_numeric, errors="coerce").fillna(0) > 1).all(axis=1)]
    rounds = db.q("SELECT * FROM rounds ORDER BY id DESC")
    est_hist = rounds["tickets_est"].dropna()
    default_t = int(est_hist.median()) if len(est_hist) else 200_000

    if jv.empty:
        st.info("Paste or type the 13 matches to build a ticket.")
    else:
        if len(jv) != 13:
            st.warning(f"{len(jv)} valid rows (a full jackpot has 13).")
        a1, a2, a3 = st.columns(3)
        prize = a1.number_input("Super Jackpot now (KES)", 0.0, value=15_000_000.0, step=1_000_000.0,
                                help="Enter the figure shown on SportyBet's Jackpot page. 15M is a placeholder.")
        n_t = a2.number_input("Tickets sold (combinations)", 1000, value=default_t, step=50_000,
                              help="Unknown. " + (f"Default is the median of {len(est_hist)} past-round estimates."
                                                  if len(est_hist) else "Placeholder until you enter past rounds below."))
        max_cols = a3.select_slider("Max combinations", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128], value=8)
        b1, b2, b3, b4 = st.columns(4)
        gamma = b1.slider("Crowd favourite bias", 1.0, 2.0, 1.25, 0.05)
        dfac = b2.slider("Crowd draw under-picking", 0.5, 1.2, 0.85, 0.05)
        c12 = b3.number_input("12-correct prize per column", 0.0, value=0.0, step=100.0)
        c11 = b4.number_input("11-correct prize per column", 0.0, value=0.0, step=50.0)

        ctx = daily.Context(models, team_league, params["w"], tax, lim, state["balance"], make_anchors(jv))
        recs, P, C = [], [], []
        for _, r in jv.iterrows():
            x = daily.analyse_match(r, ctx)
            cr = np.array([pd.to_numeric(r.get(k), errors="coerce") for k in ("Crowd 1 %", "Crowd X %", "Crowd 2 %")], float)
            crowd = cr / cr.sum() if np.isfinite(cr).all() and cr.sum() > 0 else E.crowd_shares(x["mkt"], gamma, dfac)
            recs.append(x)
            P.append(x["final"])
            C.append(crowd)
        P, C = np.array(P), np.array(C)
        st.dataframe(pd.DataFrame([{"Slot": i + 1, "Match": f"{x['home_in']} v {x['away_in']}",
                                    "Model": "yes" if x["model"] is not None else "market only",
                                    "1 %": round(x["final"][0] * 100, 1), "X %": round(x["final"][1] * 100, 1),
                                    "2 %": round(x["final"][2] * 100, 1)} for i, x in enumerate(recs)]),
                     hide_index=True)

        need_j = [x for x in recs if not x["ref_source"].startswith("reference")]
        if need_j:
            ids_j = suggested(need_j)
            if okey and (ids_j or sharp_choice):
                if st.button(f"Get sharp prices for the jackpot ({len(need_j)} matches without one)", key="confirm_jp"):
                    st.session_state.run_fetch = ("jackpot", ids_j)
                    st.rerun()
            elif not okey:
                st.caption("Jackpot matches have no sharp reference. Add a key in the sidebar, or type prices into the Ref columns (free).")
            else:
                st.caption("Load the tournament/competition list in the sidebar so jackpot matches can be linked to one.")

        kw = dict(prize=prize, n_tickets=n_t, cons12=c12, cons11=c11)
        with st.spinner("Optimising the ticket..."):
            sel_p, _ = E.build_slip(P, max_cols)
            sel_e, _ = E.optimize_slip_ev(P, C, max_cols, **kw)
        ev_p, ev_e = E.ticket_ev(P, C, sel_p, **kw), E.ticket_ev(P, C, sel_e, **kw)

        def net_after_tax(ev, wh):
            return ev["ev"] * (1 - tax["withdrawal"]) * (1 - wh) - ev["cost"] / (1 - tax["deposit"])

        st.dataframe(pd.DataFrame([
            {"Ticket": "Most likely outcomes", "Combos": ev_p["ncols"], "Cost": kes(ev_p["cost"]),
             "Chance of 13/13": f"{ev_p['p_hit'] * 100:.4f}%", "Net EV after excise": kes(net_after_tax(ev_p, 0)),
             "If 20% withholding": kes(net_after_tax(ev_p, 0.20))},
            {"Ticket": "Pool-aware (best expected return)", "Combos": ev_e["ncols"], "Cost": kes(ev_e["cost"]),
             "Chance of 13/13": f"{ev_e['p_hit'] * 100:.4f}%", "Net EV after excise": kes(net_after_tax(ev_e, 0)),
             "If 20% withholding": kes(net_after_tax(ev_e, 0.20))}]), hide_index=True)
        choice = st.radio("Ticket to use", ["Pool-aware (best expected return)", "Most likely outcomes"], horizontal=True)
        sel, ev = (sel_e, ev_e) if choice.startswith("Pool") else (sel_p, ev_p)
        stats = E.slip_stats(P, sel)
        def _role(i):
            n_pick = len(E.picks_str(sel[i]).replace("/", ""))
            if n_pick > 1:
                return "Covered (more than one pick)"
            return "Strong single" if stats["cover"][i] >= 0.55 else "Risky single"

        def _vs_market(i):
            """Points by which our probability for the chosen outcome(s) beats the market's. None if no model for the match."""
            x = recs[i]
            if x["model"] is None:
                return None
            j = sorted(sel[i])
            return 100.0 * (float(sum(x["final"][k] for k in j)) - float(sum(x["ref"][k] for k in j)))

        vs = [_vs_market(i) for i in range(len(recs))]
        best = {i for i, _ in sorted(((i, v) for i, v in enumerate(vs) if v is not None and v > 0), key=lambda t: -t[1])[:3]}
        st.dataframe(pd.DataFrame([{"Slot": i + 1, "Match": f"{x['home_in']} v {x['away_in']}",
                                    "Pick(s)": E.picks_str(sel[i]), "Covers": f"{stats['cover'][i] * 100:.0f}%",
                                    "Role": _role(i), "Model vs market": "-" if vs[i] is None else f"{vs[i]:+.1f} pts",
                                    "Best fit": "BEST FIT" if i in best else ""} for i, x in enumerate(recs)]), hide_index=True)
        st.caption("Best fit = the three matches where our forecast for your pick beats the market's by the most. "
                   "Matches with no model are never marked. A positive gap is a lead, not proof.")
        st.caption("All 13 matches must be filled in on SportyBet, so a jackpot ticket cannot skip matches. "
                   "The tool instead decides whether the whole ticket is worth buying this week.")
        st.code("\n".join(f"{i + 1:>2}. {x['home_in']} v {x['away_in']}: {E.picks_str(sel[i])}"
                          for i, x in enumerate(recs)), language=None)
        net0 = net_after_tax(ev, 0)
        st.markdown(f"### Jackpot verdict: {'BUY (paper)' if net0 >= 0 and mode == 'PAPER' else ('BUY' if net0 >= 0 else 'SKIP THIS WEEK')}")
        if net0 < 0:
            st.warning("Negative expected value under these assumptions. Tickets sold and crowd picks are the big unknowns.")
        else:
            st.success("Positive expected value under these assumptions only. It depends on the inputs above being right.")
        budget = lim["jackpot_month_pct"] * state["balance"]
        st.write(f"Monthly jackpot budget: **{kes(budget)}**. Spent this month: {kes(state['jackpot_month'])}. "
                 f"This ticket: {kes(ev['cost'])}.")
        over = state["jackpot_month"] + ev["cost"] > budget
        if over:
            st.error("This ticket would exceed the monthly jackpot budget, so it cannot be saved.")
        picks_json = json.dumps([E.picks_str(s_) for s_ in sel])
        round_start = store.iso(E.next_deadline(now) - timedelta(days=7))
        already = len(db.q("SELECT 1 FROM tickets WHERE mode=? AND picks=? AND created>=?", (mode, picks_json, round_start))) > 0
        if already:
            st.info("This exact ticket is already saved for this round, so saving again is switched off (it would charge the cost twice).")
        if st.button("Save this ticket to the log", disabled=over or already):
            gid = "J" + now.strftime("%Y%m%d-%H%M%S")
            daily.save_analysis(db, recs, mode, gid, db.param_version(), now, source="jackpot")
            db.insert("tickets", {"created": store.iso(now), "group_id": gid, "mode": mode, "ncols": ev["ncols"],
                                  "cost": ev["cost"], "p_hit": ev["p_hit"], "ev": ev["ev"],
                                  "picks": picks_json,
                                  "assumptions": json.dumps({"prize": prize, "tickets": int(n_t), "gamma": gamma,
                                                             "draw_factor": dfac, "c12": c12, "c11": c11})})
            risk.add_ledger(db, mode, "ticket_cost", -ev["cost"], gid, now)
            st.success(f"Saved ticket {gid} and recorded the cost in the {mode} bankroll.")

    with st.expander("Past rounds (teaches the tool how many tickets are sold)"):
        st.caption("Enter results from SportyBet's 'Previous Results' page. Tickets-sold estimates use the crowd placeholder "
                   "above, so they blend two unknowns. Treat them as rough.")
        with st.form("round_form"):
            rd = st.text_input("Round date (YYYY-MM-DD)")
            fc = st.columns(3)
            pool = fc[0].number_input("Super Jackpot (KES)", 0.0, step=1_000_000.0)
            w13 = fc[1].number_input("Winners with 13", 0, step=1)
            p13 = fc[2].number_input("Payout per 13 winner", 0.0, step=100_000.0)
            gc = st.columns(4)
            w12 = gc[0].number_input("Winners with 12", 0, step=1)
            p12 = gc[1].number_input("Payout per 12 winner", 0.0, step=100.0)
            w11 = gc[2].number_input("Winners with 11", 0, step=1)
            p11 = gc[3].number_input("Payout per 11 winner", 0.0, step=50.0)
            rp = st.text_area("The round's 13 matches with odds (same format as above), optional", height=90)
            ro = st.text_input("Actual results in order, e.g. 1X2112X21X1X2 (13 characters), optional")
            submitted = st.form_submit_button("Save round")
        if submitted:
            est = None
            rows_, _ = E.parse_fixtures(rp) if rp.strip() else ([], [])
            if rows_ and len(ro.strip()) == len(rows_) and set(ro.strip()) <= set("1X2") and w13 > 0:
                o = [(r["Odds 1"], r["Odds X"], r["Odds 2"]) for r in rows_]
                est = daily.estimate_tickets(o, ro.strip(), int(w13))["estimate"]
            db.insert("rounds", {"round_date": rd, "super_pool": pool, "prize2": None, "prize3": None,
                                 "winners13": int(w13), "winners12": int(w12), "winners11": int(w11),
                                 "payout13": p13, "payout12": p12, "payout11": p11, "tickets_est": est,
                                 "notes": "tickets estimate from crowd placeholder" if est else ""})
            st.success("Round saved." + (f" Estimated tickets sold: {est:,.0f}." if est else
                                         " No estimate: needs the 13 matches, odds, results and at least one 13-winner."))
        if len(rounds):
            st.dataframe(rounds.drop(columns=["id", "prize2", "prize3"]), hide_index=True)

# ============================================================================= BANKROLL
with tab_bank:
    st.subheader("Bankroll and risk")
    st.caption(f"Showing the **{mode}** bankroll. Amounts are wallet KES (what SportyBet shows).")
    if state["curve"]:
        cdf = pd.DataFrame(state["curve"], columns=["time", "balance"])
        st.line_chart(cdf["balance"], height=160)
    m1, m2, m3 = st.columns(3)
    m1.metric("Peak", kes(state["peak"]))
    m2.metric("Staked and pending", kes(state["pending"]))
    m3.metric("Jackpot spend this month", kes(state["jackpot_month"]))

    st.markdown("**Mode**")
    if mode == "PAPER":
        if st.button("Request LIVE mode"):
            gate = risk.proof_gate(db, now)
            try:
                risk.set_mode(db, "LIVE", gate, now)
                st.success("LIVE mode unlocked.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
                for c in gate["checks"]:
                    st.write(("PASS: " if c["ok"] else "FAIL: ") + c["name"] + f" ({c['detail']})")
    else:
        if st.button("Switch to PAPER"):
            risk.set_mode(db, "PAPER", now=now)
            st.rerun()

    st.markdown("**Record money moving**")
    with st.form("flow_form"):
        kind = st.selectbox("Type", ["Deposit (amount you paid)", "Withdrawal (wallet amount)",
                                     "Jackpot or other prize received", "Correction (+ or -)"])
        amt = st.number_input("Amount (KES)", step=100.0)
        note = st.text_input("Note", "")
        ok = st.form_submit_button("Record")
    if ok and amt != 0:
        if kind.startswith("Deposit"):
            credit = amt * (1 - tax["deposit"]) if mode == "LIVE" else amt
            risk.add_ledger(db, mode, "deposit", credit, note or "deposit", now)
            st.success(f"Wallet credited {kes(credit)}" + (f" after {tax['deposit']:.0%} excise." if mode == "LIVE" else "."))
        elif kind.startswith("Withdrawal"):
            risk.add_ledger(db, mode, "withdrawal", -abs(amt), note or "withdrawal", now)
            st.success(f"Wallet debited {kes(abs(amt))}. You receive about {kes(abs(amt) * (1 - tax['withdrawal']))} after excise.")
        elif kind.startswith("Jackpot"):
            risk.add_ledger(db, mode, "prize", abs(amt), note or "prize", now)
            st.success("Prize recorded.")
        else:
            risk.add_ledger(db, mode, "adjust", amt, note or "correction", now)
            st.success("Correction recorded.")
        st.rerun()

    st.markdown("**Limits** (tighter applies now, looser waits 7 days)")
    with st.form("limits_form"):
        L = {}
        lc = st.columns(2)
        L["max_stake_pct"] = lc[0].number_input("Max stake per bet (% of bankroll)", 0.1, 10.0, lim["max_stake_pct"] * 100, 0.1) / 100
        L["kelly_frac"] = lc[1].number_input("Kelly fraction", 0.01, 1.0, lim["kelly_frac"], 0.025)
        L["max_bets_day"] = lc[0].number_input("Max bets per day", 1, 20, int(lim["max_bets_day"]))
        L["max_risk_day_pct"] = lc[1].number_input("Max staked per day (%)", 0.5, 20.0, lim["max_risk_day_pct"] * 100, 0.5) / 100
        L["daily_loss_pct"] = lc[0].number_input("Daily loss stop (%)", 0.5, 20.0, lim["daily_loss_pct"] * 100, 0.5) / 100
        L["weekly_loss_pct"] = lc[1].number_input("Weekly loss stop (%)", 1.0, 40.0, lim["weekly_loss_pct"] * 100, 0.5) / 100
        L["drawdown_pct"] = lc[0].number_input("Drawdown stop (%)", 5.0, 60.0, lim["drawdown_pct"] * 100, 1.0) / 100
        L["jackpot_month_pct"] = lc[1].number_input("Jackpot budget per month (%)", 0.0, 20.0, lim["jackpot_month_pct"] * 100, 0.5) / 100
        L["min_edge"] = lc[0].number_input("Minimum net edge (%)", 0.0, 20.0, lim["min_edge"] * 100, 0.5) / 100
        L["max_odds"] = lc[1].number_input("Maximum odds to bet", 1.2, 20.0, float(lim["max_odds"]), 0.1)
        okl = st.form_submit_button("Apply limits")
    if okl:
        L = {k: (int(v) if k == "max_bets_day" else round(float(v), 6)) for k, v in L.items()}
        applied, queued = risk.propose_limits(db, L, now)
        if applied:
            st.success("Applied now: " + ", ".join(f"{k}={v}" for k, v in applied.items()))
        if queued:
            st.warning("Waiting 7 days: " + ", ".join(f"{k}={v['value']} (from {v['effective_at'][:10]})" for k, v in queued.items()))
        st.rerun()
    pend = db.get("pending_limits", {})
    if pend:
        st.write("Pending loosening:", {k: f"{v['value']} on {v['effective_at'][:10]}" for k, v in pend.items()})

    st.markdown("**Tax settings** (confirm with SportyBet or a tax adviser)")
    with st.form("tax_form"):
        tc = st.columns(2)
        t_dep = tc[0].number_input("Excise on deposits (%)", 0.0, 30.0, tax["deposit"] * 100, 0.5) / 100
        t_wd = tc[1].number_input("Excise on withdrawals (%)", 0.0, 30.0, tax["withdrawal"] * 100, 0.5) / 100
        t_wh = tc[0].number_input("Withholding on jackpot winnings (%)", 0.0, 40.0, tax["jackpot_withholding"] * 100, 1.0) / 100
        t_pb = tc[1].checkbox("Charge tax on every bet (very conservative)", value=bool(tax["per_bet"]))
        okt = st.form_submit_button("Save tax settings")
    if okt:
        db.set("tax", {"deposit": t_dep, "withdrawal": t_wd, "jackpot_withholding": t_wh, "per_bet": t_pb})
        db.event("tax_change", f"deposit {t_dep:.1%}, withdrawal {t_wd:.1%}, withholding {t_wh:.1%}, per_bet {t_pb}", now)
        st.rerun()

    st.markdown("**Ruin simulator** (what happens to a bankroll if the model's true edge is smaller than hoped)")
    sc = st.columns(3)
    s_odds = sc[0].number_input("Typical odds", 1.2, 10.0, 2.2, 0.1)
    s_n = sc[1].number_input("Bets per month", 5, 200, 30)
    s_m = sc[2].number_input("Months", 1, 24, 6)
    if st.button("Run simulation"):
        out = []
        for e in (-0.08, -0.04, -0.02, 0.0, 0.02, 0.04):
            r = risk.ruin_simulation(e, s_odds, lim["max_stake_pct"], int(s_n * s_m), lim["drawdown_pct"])
            out.append({"True net edge per bet": f"{e:+.0%}", "Chance of hitting the drawdown stop": f"{r['p_stop']:.0%}",
                        "Median final bankroll (start = 1.00)": f"{r['median_final']:.2f}",
                        "Chance of ending below start": f"{r['p_below_start']:.0%}"})
        st.dataframe(pd.DataFrame(out), hide_index=True)
        st.caption(f"Stake {lim['max_stake_pct']:.1%} of bankroll per bet (the cap). Real stakes are usually smaller, "
                   "so this is a pessimistic case. Paths that hit the stop stop betting.")

    st.markdown("**Backup**")
    with tempfile.TemporaryDirectory() as td:
        bp = os.path.join(td, "backup.db")
        db.backup(bp)
        with open(bp, "rb") as fh:
            st.download_button("Download a backup of all your data", fh.read(),
                               file_name=f"engine-backup-{now:%Y%m%d}.db")

# ============================================================================= LEARNING
with tab_learn:
    st.subheader("Learning loop")
    if st.button("Update results and settle bets"):
        load_data.clear()
        fresh, _ = load_data()
        if fresh.empty:
            st.error("Could not download results.")
        else:
            npred, nbet = daily.settle_all(db, fresh, now)
            risk.auto_drawdown_switch(db, now)
            st.success(f"{npred} matches and {nbet} bets settled. (The free results file updates a few times a week, "
                       "so recent matches can take a day or two.)")
            st.rerun()

    au_df, au_missing = audit.build(db, data if not data.empty else None)
    with st.expander("Post-mortem: why were the calls right or wrong?", expanded=True):
        pm_tbl = postmortem.match_table(db, data if not data.empty else None)
        if pm_tbl.empty:
            st.info("No settled matches yet. Save an analysis, wait for the results file to include the matches, "
                    "and they appear here automatically when you open the app.")
        else:
            rep_ = postmortem.bet_report(pm_tbl)
            q1, q2, q3 = st.columns(3)
            q1.metric("Matches settled", len(pm_tbl))
            q2.metric("Bets settled", 0 if rep_ is None else rep_["bets"],
                      "" if rep_ is None else f"hit rate {rep_['hit_rate']:.0%}", delta_color="off")
            q3.metric("Closing-line value", "-" if not rep_ or "clv" not in rep_ else f"{rep_['clv']:+.1%}",
                      "" if not rep_ or "clv" not in rep_ else f"95%: {rep_['clv_ci'][0]:+.1%} to {rep_['clv_ci'][1]:+.1%}",
                      delta_color="off")
            bt_ = postmortem.bias_tables(db)
            for f_ in bt_["findings"]:
                st.write("- " + f_)
            st.caption(f"How many bets are enough? To see a 1% closing-line edge you need roughly {postmortem.bets_needed(0.01, 0.07)} "
                       f"bets; to see a 3% profit edge you need roughly {postmortem.bets_needed(0.03, 1.3):,}. That is why closing-line "
                       "value is the first test, and profit comes second.")
            show_ = pm_tbl.tail(40).iloc[::-1].copy()
            for c_ in ("P final", "P model", "P market", "P close"):
                show_[c_] = show_[c_].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
            show_["CLV"] = show_["CLV"].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}")
            st.dataframe(show_[["Date", "Match", "Source", "Result", "Pick", "Won", "P final", "P market", "P close", "CLV", "Why"]],
                         hide_index=True)
            if "outcomes" in bt_:
                if st.checkbox("Show bias checks: outcomes and leagues", key="pm_bias"):
                    o_ = bt_["outcomes"].copy()
                    o_["Predicted"], o_["Happened"] = o_["Predicted"].map("{:.1%}".format), o_["Happened"].map("{:.1%}".format)
                    st.dataframe(o_.drop(columns=["z"]), hide_index=True)
                    if "leagues" in bt_ and len(bt_["leagues"]):
                        st.dataframe(bt_["leagues"].round(4), hide_index=True)
                        st.caption("Gain vs market above zero means the blend beat the market's own forecast in that league.")
            st.caption("The post-mortem only diagnoses. Settings change only through the guarded recalibration below, "
                       "because tuning on a few dozen matches builds a bias.")
        lb = db.get("backtest_last")
        st.markdown("**Live-readiness check**")
        ready = []
        ready.append(("Backtest closing-line value clearly above zero",
                      bool(lb and lb.get("clv_ci") and lb["clv_ci"][0] > 0 and lb.get("bets", 0) >= 100),
                      "run the backtest" if not lb else f"{lb.get('bets', 0)} bets, CLV {lb['clv']:+.1%}" if lb.get("clv") is not None else "no bets"))
        pb = postmortem.bet_report(pm_tbl) if not pm_tbl.empty else None
        ready.append(("At least 150 settled paper bets with closing-line value above zero",
                      bool(pb and pb.get("clv_n", 0) >= 150 and pb["clv_ci"][0] > 0),
                      "none yet" if not pb else f"{pb.get('clv_n', 0)} bets with a closing price"))
        ready.append(("SportyBet price check has at least 300 matches", len(au_df) >= audit.NEEDED,
                      f"{len(au_df)} of {audit.NEEDED} logged matches with closing prices"))
        ready.append(("Proof gate passed", bool(risk.proof_gate(db, now).get("passed")), "see the gate below"))
        for name_, ok_, det_ in ready:
            st.write(("PASS  " if ok_ else "NOT YET  ") + name_ + f"  ({det_})")
        st.caption("Only when all three pass is a small LIVE start reasonable. The proof gate stays the only unlock.")

    with st.expander("SportyBet price check: are the prices you see ever better than fair?", expanded=True):
        st.caption("Uses the SportyBet odds you typed in, compared with Pinnacle's closing odds from the free results file. It does not use "
                   "our forecast. Log EVERY match you look at (press 'Save this analysis'), not only the bets: more matches means a "
                   "faster answer.")
        if au_df.empty:
            st.info("No logged matches with closing prices yet. Closing prices appear in the results file a few days after each match."
                    + (f" ({au_missing} matches are waiting for closing prices.)" if au_missing else ""))
        else:
            au = audit.summary(au_df, lim["min_edge"])
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Matches logged", au["n"], f"{audit.NEEDED} for a first reading", delta_color="off")
            a2.metric("SportyBet margin", f"{au['sporty_margin']:.1%}", f"Pinnacle {au['pinnacle_margin']:.1%}", delta_color="off")
            a3.metric("Prices above fair", f"{au['share_positive']:.0%}", "of all outcomes", delta_color="off")
            a4.metric(f"Matches with a {au['min_edge']:.0%}+ price", f"{au['share_edge']:.0%}")
            st.progress(min(au["n"] / audit.NEEDED, 1.0), f"{min(au['n'], audit.NEEDED)} of {audit.NEEDED} matches for a first reading")
            st.write(au["verdict"])
            if au_missing:
                st.caption(f"{au_missing} more logged matches are waiting for closing prices to appear in the results file.")
            if st.checkbox("Show breakdowns: outcome, odds band, league, days before the match", key="au_groups"):
                for by_, title_ in (("Outcome", "By outcome"), ("Odds band", "By odds band"), ("League", "By league"),
                                    ("Lead days", "By days before the match")):
                    g_ = audit.group(au_df, by_)
                    if len(g_):
                        st.markdown(f"**{title_}**")
                        g_ = g_.assign(Mean_value=g_["Mean_value"].map("{:+.1%}".format), Share_positive=g_["Share_positive"].map("{:.0%}".format))
                        st.dataframe(g_.rename(columns={"Mean_value": "Average value", "Share_positive": "Share above fair"}), hide_index=True)
                st.caption("Average value is negative by about the margin. Look for groups where it is closest to zero or positive: "
                           "that is where SportyBet's prices are cheapest to beat. Small groups mislead, so wait for 300 matches.")

    with st.expander("Data check: is the results file current, and are all team names understood?"):
        fr_ = gaps.freshness(db, data if not data.empty else None, now)
        f1, f2, f3 = st.columns(3)
        f1.metric("Latest result in file", fr_["latest"] or "-",
                  "" if fr_["stale_days"] is None else f"{fr_['stale_days']} day(s) ago", delta_color="off")
        f2.metric("Predictions pending", fr_["pending_total"])
        f3.metric("Names not understood", fr_["pending_unresolved"], "cannot settle", delta_color="off")
        if fr_["pending_waiting"]:
            st.caption(f"{fr_['pending_waiting']} pending are waiting for the free file to add their result"
                       + (f"; {fr_['pending_old']} of them are more than 4 days old, which usually means the team name does not "
                          "match or the match was postponed." if fr_["pending_old"] else "."))
        um_ = gaps.unmatched_names(db)
        if um_.empty:
            st.success("Every team name you entered was matched.")
        else:
            st.write("These names were not found in the results file. Choose the official name for each and save: "
                     "they are remembered and used from now on.")
            st.dataframe(um_, hide_index=True)
            known_ = sorted(team_league)
            al1, al2 = st.columns(2)
            al_typed = al1.selectbox("Name you typed", um_["Name"].tolist(), key="al_typed")
            al_off = al2.selectbox("Official name in the results file", known_, key="al_off") if known_ else \
                al2.text_input("Official name in the results file", key="al_off")
            if st.button("Save this name match", key="al_save"):
                if gaps.add_alias(db, al_typed, al_off):
                    st.session_state.flash = [f"Saved: {al_typed} is now understood as {al_off}. New analyses will use it."]
                    st.rerun()

    with st.expander("Run the past (backtest): how would the rules have done on matches that already happened?"):
        st.caption("Walk-forward: every block is predicted using only earlier matches. Uses the same BET/SKIP and staking rules "
                   "as the live tool, Bet365-style prices made worse by the margin below, and Pinnacle as the sharp reference. "
                   "It cannot see SportyBet's real prices or the jackpot. Takes a few minutes.")
        bc1, bc2, bc3 = st.columns(3)
        bt_days = bc1.selectbox("Period to test", [180, 365, 540, 730, 1095], index=1, format_func=lambda d: f"last {d} days")
        bt_long = st.checkbox("Use 6 seasons of history (needed for periods over 540 days; much slower)")
        bt_margin = bc2.slider("Extra margin (SportyBet is probably worse than Bet365)", 0.0, 0.08, 0.03, 0.005, format="%.3f")
        bt_sharp = bc3.checkbox("Use Pinnacle as reference", value=True)
        if st.button("Run backtest"):
            bt_data = data
            if bt_long or int(bt_days) > 540:
                with st.spinner("Downloading 6 seasons of results (about a minute)..."):
                    bt_data, _bt_failed = load_long()
            if bt_data.empty:
                st.error("No match data loaded.")
            else:
                bar = st.progress(0.0, "Starting...")
                res = backtest.run(bt_data, test_days=int(bt_days), xi=params["xi"], w=params["w"], lim=lim, tax=tax,
                                   extra_margin=float(bt_margin), use_sharp_ref=bt_sharp,
                                   progress=lambda f, m: bar.progress(min(f, 1.0), m))
                bar.empty()
                st.session_state.bt = res
                db.set("backtest_last", {"when": store.iso(now), **{k: v for k, v in res["summary"].items()}})
        bt = st.session_state.get("bt")
        if bt:
            sm, bb = bt["summary"], bt["bets"]
            st.write(sm["verdict"])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Bets", sm["bets"], f"of {sm['matches_evaluated']} matches")
            k2.metric("ROI", "-" if sm["roi"] is None else f"{sm['roi']:+.1%}")
            k3.metric("Closing-line value", "-" if sm["clv"] is None else f"{sm['clv']:+.1%}")
            k4.metric("Worst drawdown", f"{sm['max_drawdown']:.1%}")
            if len(bb):
                st.line_chart(bb.assign(Cumulative=bb["PnL"].cumsum()).set_index("Date")["Cumulative"])
                st.dataframe(bb.drop(columns=["CLV"]).tail(50), hide_index=True)
            st.caption(f"Break-even also needs the wallet to grow {sm['flow_hurdle']:.1%} to cover deposit and withdrawal excise.")

    gate = risk.proof_gate(db, now)
    st.markdown("**Proof gate** (all must pass before LIVE unlocks)")
    for c in gate["checks"]:
        st.write(("PASS  " if c["ok"] else "FAIL  ") + c["name"] + f"  ({c['detail']})")
    st.caption("This gate is necessary, not sufficient. Even 300 bets cannot prove a small edge; "
               "the model-versus-baseline test over many matches carries the statistical weight.")

    sp = db.q("SELECT * FROM predictions WHERE source='daily' AND result IN ('1','X','2') AND pD1 IS NOT NULL")
    if len(sp):
        y = sp["result"].map({"1": 0, "X": 1, "2": 2}).to_numpy()
        PF = sp[["pF1", "pFX", "pF2"]].to_numpy(float)
        PM = sp[["pM1", "pMX", "pM2"]].to_numpy(float)
        PD = sp[["pD1", "pDX", "pD2"]].to_numpy(float)
        st.markdown("**Scorecard (lower is better)**")
        st.dataframe(pd.DataFrame([
            {"Forecast": "Final (used for bets)", "Log loss": round(E.logloss(PF, y), 4), "RPS": round(E.rps(PF, y), 4)},
            {"Forecast": "Baseline (reference odds)", "Log loss": round(E.logloss(PM, y), 4), "RPS": round(E.rps(PM, y), 4)},
            {"Forecast": "Model alone", "Log loss": round(E.logloss(PD, y), 4), "RPS": round(E.rps(PD, y), 4)}]),
            hide_index=True)
        st.caption(f"{len(sp)} settled matches.")
        probs, hits = PF.ravel(), np.eye(3)[y].ravel()
        rows = []
        for lo in np.arange(0, 1, 0.1):
            m = (probs >= lo) & (probs < lo + 0.1 + (1e-9 if lo > 0.85 else 0))
            if m.sum():
                rows.append({"We said": f"{lo:.0%}-{lo + 0.1:.0%}", "Predictions": int(m.sum()),
                             "Average said": f"{probs[m].mean():.1%}", "Actually happened": f"{hits[m].mean():.1%}"})
        st.markdown("**Calibration: when we say X%, does it happen X% of the time?**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("No settled predictions yet. Save analyses on the Today tab, then press 'Update results' after the matches.")

    sb = db.q("SELECT * FROM bets WHERE status IN ('Won','Lost')")
    if len(sb):
        bet = sb[sb["decision"] == "BET"]
        skp = sb[sb["decision"] == "SKIP"]
        roi_b = bet["pnl"].sum() / bet["stake"].sum() if len(bet) and bet["stake"].sum() > 0 else float("nan")
        st.markdown("**Did the rules help?**")
        st.write(f"Bets taken: {len(bet)}, return on stakes {roi_b:+.1%}" if len(bet) else "No settled BET calls yet.")
        if len(skp):
            st.write(f"Skipped best-options: {len(skp)}; if staked at one unit each they would have returned "
                     f"{skp['unit_pnl'].mean():+.1%} per bet. If this is clearly positive, the minimum edge is too strict.")

    tk = db.q("SELECT * FROM tickets ORDER BY id DESC")
    if len(tk):
        trows = []
        for t in tk.itertuples():
            pr = db.q("SELECT slot, result FROM predictions WHERE group_id=? ORDER BY slot", (t.group_id,))
            picks = json.loads(t.picks)
            done = pr[pr["result"].isin(["1", "X", "2"])]
            miss = sum(1 for s in done.itertuples() if s.result not in picks[int(s.slot) - 1])
            trows.append({"Ticket": t.group_id, "Mode": t.mode, "Combos": t.ncols, "Cost": kes(t.cost),
                          "Settled matches": len(done), "Missed so far": miss})
        st.markdown("**Jackpot tickets** (0 missed with all 13 settled = a 13/13 winner; 1 to 2 missed = consolation range)")
        st.dataframe(pd.DataFrame(trows), hide_index=True)

    st.markdown("**Recalibrate the model/market blend** (guarded: moves halfway, bounded 0.4 to 1.0)")
    tune = st.checkbox("Also test other decay rates (slower)")
    if st.button("Run calibration") and not data.empty:
        bar = st.progress(0.0)
        grid = [0.001, 0.002, 0.004] if tune else [params["xi"]]
        res = E.calibrate(data, grid, progress=bar.progress)
        if res.empty:
            st.error("Not enough recent matches with odds to calibrate.")
        else:
            best = res.sort_values("logloss").iloc[0]
            cur = res[res["xi"] == params["xi"]]["logloss"].min()
            xi_new = float(best["xi"]) if (pd.notna(cur) and cur - best["logloss"] > 0.0005) else params["xi"]
            w_new = round(min(1.0, max(0.4, params["w"] + 0.5 * (float(best["w"]) - params["w"]))), 3)
            ev_txt = (f"{int(best['matches'])} held-out matches; log loss blend {best['logloss']:.4f}, model "
                      f"{best['model_only']:.4f}, market {best['market_only']:.4f}; best w {best['w']}")
            db.set("params", {"xi": xi_new, "w": w_new})
            db.log_param("w", w_new, ev_txt, now)
            db.log_param("xi", xi_new, ev_txt, now)
            db.event("calibration", ev_txt, now)
            st.cache_data.clear()
            st.success(f"Market weight {params['w']} -> {w_new}; decay {params['xi']} -> {xi_new}. {ev_txt}")

    with st.expander("Change log"):
        st.dataframe(db.q("SELECT ts, kind, detail FROM events ORDER BY id DESC LIMIT 30"), hide_index=True)
        st.dataframe(db.q("SELECT ts, name, value, evidence FROM param_versions ORDER BY id DESC LIMIT 20"), hide_index=True)


# ============================================================================= TIPSTERS
with tab_tip:
    st.subheader("Tipster ledger (uses no API credits)")
    st.caption("Log a tipster's pick BEFORE the match, exactly as given, with the odds at the time. The tool settles it from the free "
               "results file and scores each tipster with the same tests as our own calls. Picks entered on or after the match date "
               "are voided. Booking codes are kept as plain notes only, never decoded.")
    with st.form("tip_form", clear_on_submit=True):
        tc1, tc2 = st.columns(2)
        t_name = tc1.text_input("Tipster or channel name")
        t_note = tc2.text_input("Note or booking code (optional, text only)")
        tc3, tc4 = st.columns(2)
        t_home = tc3.text_input("Home team")
        t_away = tc4.text_input("Away team")
        tc5, tc6 = st.columns(2)
        t_sel = tc5.selectbox("Pick", tipsters.SELECTIONS)
        t_odds = tc6.number_input("Odds at the time", 1.01, 100.0, 2.00, 0.01)
        if st.form_submit_button("Log this pick (before the match)"):
            _known = set(team_league)
            pid, msg = tipsters.add_pick(db, t_name, t_home, t_away, t_sel, t_odds, _known,
                                         {E.norm(t): t for t in _known}, t_note, now)
            (st.success if pid else st.warning)(msg)
    if st.button("Settle tipster picks now") and not data.empty:
        st.success(f"{tipsters.settle(db, data, now)} tipster picks settled (late entries are voided).")
    rep_t = tipsters.report(db)
    if rep_t.empty:
        st.info("No settled picks yet. Results appear here automatically when you open the app after the matches.")
    else:
        show_t = rep_t.copy()
        for c_ in ("Hit rate", "ROI (flat 1 unit)", "ROI low", "ROI high"):
            show_t[c_] = show_t[c_].map(lambda v: f"{v:+.1%}" if c_.startswith("ROI") else f"{v:.0%}")
        show_t["CLV"] = show_t["CLV"].map(lambda v: "-" if v is None or pd.isna(v) else f"{v:+.1%}")
        show_t["Average odds"] = show_t["Average odds"].map("{:.2f}".format)
        st.dataframe(show_t, hide_index=True)
        st.caption("A tipster needs at least 100 settled picks and a closing-line range above zero to earn even a 'worth tracking' verdict. "
                   "With many tipsters, a few always look good by luck, so never copy one on a short record.")
    recent_t = db.q("SELECT created, tipster, home_in, away_in, selection, odds, status, note FROM tipster_picks ORDER BY id DESC LIMIT 30")
    if len(recent_t):
        with st.expander("Recent picks"):
            st.dataframe(recent_t, hide_index=True)
