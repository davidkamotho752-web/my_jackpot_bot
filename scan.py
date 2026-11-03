"""scan.py - shortlist the upcoming matches that qualify, at zero API cost.

Source of fixtures + prices: football-data.co.uk's upcoming-fixtures file (free, no key). It carries Bet365 and
Pinnacle prices for the next matches of the leagues the tool already models.
NOT VERIFIED from the build environment (the site was unreachable there): the file's exact URL and columns, and how often it
is refreshed (the site updates weekly, not live). If it cannot be read the app says so and nothing is spent or lost.

How the shortlist is made: the SAME rules as the live tool (cautious probability, minimum edge, odds cap, Kelly cap).
Price taken = Bet365 made worse by `extra_margin` (SportyBet is probably dearer). Reference = Pinnacle if present.
The shortlist is a FILTER, not a bet slip: it tells you which few matches to open on SportyBet and the lowest odds worth
taking. Always check the real SportyBet price before betting.
"""
import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import daily
import engine as E

LONDON = ZoneInfo("Europe/London")        # football-data kick-off times are assumed to be UK time
NAIROBI = ZoneInfo("Africa/Nairobi")
FIXTURES_URLS = ["https://www.football-data.co.uk/fixtures.csv"]
SOFT = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]
SHARP = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA")]


def parse_fixtures_csv(content):
    content = content[3:] if content[:3] == b"\xef\xbb\xbf" else content           # the real file starts with a hidden BOM
    df = pd.read_csv(io.BytesIO(content), encoding="latin-1", on_bad_lines="skip")
    df.columns = [str(c).replace("\xef\xbb\xbf", "").replace("\u00ef\u00bb\u00bf", "").strip() for c in df.columns]
    need = ["Div", "Date", "HomeTeam", "AwayTeam"]
    if not all(c in df.columns for c in need):
        raise ValueError("unexpected columns: " + ", ".join(map(str, df.columns[:8])))
    df = df.dropna(subset=need).copy()
    for c in E.ODDS_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = E._parse_dates(df["Date"])
    df = df.dropna(subset=["Date"])
    df["League"] = df["Div"]
    return df


def download_fixtures(http_get=requests.get):
    """Returns (DataFrame, error message or None)."""
    err = "not tried"
    for url in FIXTURES_URLS:
        try:
            r = http_get(url, headers=E.HEADERS, timeout=20)
            if r.status_code != 200 or len(r.content) < 100:
                err = f"HTTP {r.status_code}"
                continue
            return parse_fixtures_csv(r.content), None
        except Exception as e:                       # network, parsing, anything: report, never crash
            err = str(e)
    return pd.DataFrame(), err


def _triple(row, options):
    for cols in options:
        if all(c in row.index for c in cols):
            v = [row[c] for c in cols]
            if all(pd.notna(x) and float(x) > 1.0 for x in v):
                return [float(x) for x in v]
    return None


def kickoff(f):
    """Kick-off as an aware datetime (UK time assumed), or None if the file has no usable time."""
    try:
        hh, mm = str(f.get("Time", "") or "").strip().split(":")[:2]
        d = pd.Timestamp(f["Date"])
        return datetime(d.year, d.month, d.day, int(hh), int(mm), tzinfo=LONDON)
    except Exception:
        return None


def kickoff_text(row):
    """'Mon 02 Nov 15:00 UK / 18:00 Nairobi', so nobody misreads the time by 3 hours."""
    ko = row.get("Kickoff")
    if ko is None:
        return f"{row['Date']:%a %d %b} {row.get('Time', '')}".strip()
    return f"{ko.astimezone(LONDON):%a %d %b %H:%M} UK / {ko.astimezone(NAIROBI):%H:%M} Nairobi"


def scan(fixtures, models, team_league, params, lim, tax, bankroll, now, days=2, extra_margin=0.03, collect=None, min_lead_min=20):
    """Returns (qualifying rows, n_scanned). Each row has what the app needs to show it and to fill the table.
    Matches that have started (or start within min_lead_min minutes) are left out: a pre-match bet is no longer possible.
    If `collect` is a list, the full analysis of EVERY judged match is appended to it (for the automatic learning log)."""
    if fixtures.empty or not models:
        return [], 0
    d0 = pd.Timestamp(now.date())
    win = fixtures[(fixtures["Date"] >= d0 - pd.Timedelta(days=1)) & (fixtures["Date"] < d0 + pd.Timedelta(days=days))]
    cutoff = now + timedelta(minutes=min_lead_min)
    rows, n = [], 0
    for _, f in win.iterrows():
        ko = kickoff(f)
        if ko is not None and ko <= cutoff:
            continue
        if ko is None and f["Date"] < d0:                     # no time and in the past
            continue
        soft = _triple(f, SOFT)
        if soft is None:
            continue
        n += 1
        taken = [max(o / (1 + extra_margin), 1.01) for o in soft]
        sharp = _triple(f, SHARP)
        anchors = {}
        if sharp is not None:
            inv = 1.0 / np.asarray(sharp)
            anchors[(E.norm(f["HomeTeam"]), E.norm(f["AwayTeam"]))] = {"probs": inv / inv.sum(), "source": "pinnacle", "n_books": 1}
        ctx = daily.Context(models, team_league, params["w"], tax, lim, bankroll, anchors)
        res = daily.analyse_match({"Home": f["HomeTeam"], "Away": f["AwayTeam"], "Odds 1": taken[0], "Odds X": taken[1],
                                   "Odds 2": taken[2]}, ctx)
        if collect is not None:
            collect.append(res)
        if res["decision"] != "BET":
            continue
        rows.append({"Date": f["Date"], "Time": str(f.get("Time", "") or ""), "Kickoff": ko, "Home": f["HomeTeam"], "Away": f["AwayTeam"],
                     "Pick": res["selection"], "Min odds": res["min_odds"], "Net edge": res["edge_low"],
                     "Stake": res["stake"], "Ref": sharp, "Priced at": res["odds"]})
    rows.sort(key=lambda r: -r["Net edge"])
    cap, kept, per_day = int(lim["max_bets_day"]), [], {}
    for r in rows:                                    # best first, at most max_bets_day per calendar day
        d = r["Date"]
        if per_day.get(d, 0) < cap:
            per_day[d] = per_day.get(d, 0) + 1
            kept.append(r)
    return kept, n


def coverage(fixtures, team_league, now, days=4):
    """Per-league table of what was looked at: fixtures in the window, how many had Bet365 odds, how many had both teams known."""
    cols = ["League", "Fixtures", "With odds", "Both teams known"]
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=cols)
    d0 = pd.Timestamp(now.date())
    win = fixtures[(fixtures["Date"] >= d0) & (fixtures["Date"] < d0 + pd.Timedelta(days=days))].copy()
    if "League" not in win.columns:
        win["League"] = win["Div"] if "Div" in win.columns else "all"
    known_norm = {E.norm(t): t for t in team_league}
    known = set(team_league)
    rows = []
    for lg, g in win.groupby("League"):
        odds_ok = sum(_triple(f, SOFT) is not None for _, f in g.iterrows())
        both = sum(bool(E.resolve_team(f["HomeTeam"], known, known_norm)) and bool(E.resolve_team(f["AwayTeam"], known, known_norm))
                   for _, f in g.iterrows()) if known else 0
        rows.append({"League": E.LEAGUES.get(lg, lg), "Fixtures": len(g), "With odds": odds_ok, "Both teams known": both})
    return pd.DataFrame(rows, columns=cols).sort_values("Fixtures", ascending=False).reset_index(drop=True)
