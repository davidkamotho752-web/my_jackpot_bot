"""gaps.py - data-gap helpers: team names the tool could not match, user aliases, and data freshness."""
import pandas as pd

import engine as E
import store


# ----------------------------------------------------------------------------- data gaps
def unmatched_names(db):
    """Team names typed in that the tool could not map to a results-file team (so: no model, no settlement)."""
    frames = []
    for sql in ("SELECT home_in AS name FROM predictions WHERE home=''", "SELECT away_in AS name FROM predictions WHERE away=''",
                "SELECT home_in AS name FROM tipster_picks WHERE home=''", "SELECT away_in AS name FROM tipster_picks WHERE away=''"):
        frames.append(db.q(sql))
    allx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"name": []})
    if allx.empty:
        return pd.DataFrame({"Name": [], "Times": []})
    vc = allx["name"].astype(str).str.strip().value_counts()
    saved = {E.norm(k) for k in (db.get("aliases", {}) or {})}
    out = [(n, int(c)) for n, c in vc.items() if n and E.norm(n) not in saved]
    return pd.DataFrame(out, columns=["Name", "Times"])


def load_aliases(db):
    """Apply the names you mapped by hand. Call once at start-up."""
    for k, v in (db.get("aliases", {}) or {}).items():
        E.ALIASES[E.norm(k)] = v


def add_alias(db, typed, official):
    typed, official = str(typed).strip(), str(official).strip()
    if not typed or not official:
        return False
    al = dict(db.get("aliases", {}) or {})
    al[typed] = official
    db.set("aliases", al)
    E.ALIASES[E.norm(typed)] = official
    db.event("alias_added", f"{typed} -> {official}")
    return True


def freshness(db, data, now=None):
    """How current the results file is, and why logged matches are still pending."""
    now = now or store.now_local()
    out = {"latest": None, "stale_days": None, "pending_total": 0, "pending_unresolved": 0, "pending_waiting": 0, "pending_old": 0}
    if data is not None and len(data):
        out["latest"] = pd.Timestamp(data["Date"].max()).strftime("%Y-%m-%d")
        out["stale_days"] = (pd.Timestamp(now.date()) - pd.Timestamp(data["Date"].max()).normalize()).days
    pend = db.q("SELECT created, home, away FROM predictions WHERE result='Pending'")
    out["pending_total"] = len(pend)
    if len(pend):
        unres = (pend["home"] == "") | (pend["away"] == "")
        out["pending_unresolved"] = int(unres.sum())
        out["pending_waiting"] = int((~unres).sum())
        old = pend[~unres & (pd.to_datetime(pend["created"].str[:10]) < pd.Timestamp(now.date()) - pd.Timedelta(days=4))]
        out["pending_old"] = len(old)
    return out
