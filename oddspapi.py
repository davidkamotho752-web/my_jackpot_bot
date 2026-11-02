"""oddspapi.py - sharp reference odds (Pinnacle) from OddsPapi's free tier (250 requests a month, no card).

WHAT IS AND IS NOT VERIFIED (from OddsPapi's own public pages, not from a live call in the build environment):
  base            https://api.oddspapi.io/v4        auth: ?apiKey=
  GET /tournaments?sportId=10                       soccer = sportId 10
  GET /odds-by-tournaments?bookmaker=pinnacle&tournamentIds=1,2,3&oddsFormat=decimal
  market 101 = Full Time Result, outcomes 101 = home, 102 = draw, 103 = away
  odds nested: bookmakerOdds -> slug -> markets -> marketId -> outcomes -> outcomeId -> players -> 0 -> price
Confirmed from OddsPapi's endpoint pages (still not from a live call): odds-by-tournaments takes `bookmaker`
(singular, one slug) and at most 3 `tournamentIds`; its fixtures carry participant1Id/participant2Id, NOT team names,
so names come from GET /participants?sportId=10 (one request, cached 30 days). Cooldown is 1 second per endpoint.
participant1 is treated as home (conventional, to be checked on a real fixture).
NOT confirmed: how a request is counted on the quota page. So: every call is counted as 1 in our own log, the parser accepts several field spellings, and the
app shows what it parsed. If a call parses 0 fixtures, the raw start of the reply is saved (settings key
'oddspapi_debug') so it can be fixed from the real shape. Enter the dashboard's remaining figure to true up the count.

Fixtures are converted to the same event shape the rest of the tool already uses (home_team, away_team,
commence_time, bookmakers[h2h]), so caching, matching and de-vigging are shared with oddsapi.py.
"""
import json
from datetime import datetime

import requests

import budget
import oddsapi
import store

BASE = "https://api.oddspapi.io/v4"
PROVIDER = "oddspapi"
QUOTA_FREE = 250
RESERVE = 30
SOCCER = 10
HOME_ID, DRAW_ID, AWAY_ID = "101", "102", "103"
FULL_TIME_RESULT = "101"
MAX_TOURNAMENTS = 3          # per odds-by-tournaments call, per OddsPapi docs


def _pick(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return default


def _items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "fixtureId" in payload:
            return [payload]
        vals = list(payload.values())
        if vals and all(isinstance(v, dict) and "fixtureId" in v for v in vals):
            return vals
        for k in ("data", "fixtures", "results", "items"):
            if isinstance(payload.get(k), list):
                return payload[k]
    return []


def _price(outcome):
    if not isinstance(outcome, dict):
        return None
    pl = outcome.get("players")
    if isinstance(pl, dict) and pl:
        first = pl.get("0") or next(iter(pl.values()))
        p = _pick(first, "price", "odds")
    elif isinstance(pl, list) and pl:
        p = _pick(pl[0], "price", "odds")
    else:
        p = _pick(outcome, "price", "odds")
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return p if p > 1.0 else None


def normalise(payload, names=None):
    """OddsPapi reply -> list of events in the tool's common shape. Fixtures without a 3-way market are skipped.
    names: {participantId: team name} from list_participants (the odds reply carries ids only)."""
    names = names or {}
    events = []
    for f in _items(payload):
        part = f.get("participants") or {}
        home = (_pick(part.get("home") or {}, "name") or _pick(f, "homeTeam", "participant1Name", "home")
                or names.get(str(f.get("participant1Id"))))
        away = (_pick(part.get("away") or {}, "name") or _pick(f, "awayTeam", "participant2Name", "away")
                or names.get(str(f.get("participant2Id"))))
        if not home or not away:
            continue
        books = []
        for slug, bk in (f.get("bookmakerOdds") or {}).items():
            mk = (bk.get("markets") or {}).get(FULL_TIME_RESULT) if isinstance(bk, dict) else None
            outs = (mk or {}).get("outcomes") or {}
            p1, px, p2 = (_price(outs.get(i)) for i in (HOME_ID, DRAW_ID, AWAY_ID))
            if p1 and px and p2:
                books.append({"key": slug, "markets": [{"key": "h2h", "outcomes": [
                    {"name": home, "price": p1}, {"name": "Draw", "price": px}, {"name": away, "price": p2}]}]})
        if books:
            events.append({"id": _pick(f, "fixtureId", "id"), "home_team": home, "away_team": away,
                           "commence_time": _pick(f, "startTime", "startDate", "trueStartTime", "commence_time"),
                           "bookmakers": books})
    return events


def _get(path, key, params, http_get):
    r = oddsapi.call(http_get, f"{BASE}{path}", {"apiKey": key, **params}, 30)
    return r.json()


def _save_debug(db, payload, note):
    try:
        db.set("oddspapi_debug", {"note": note, "head": json.dumps(payload)[:2500]})
    except Exception:
        pass


def list_tournaments(db, key, now=None, http_get=requests.get, ttl_days=30):
    """[(id, name)] for soccer. Cached for 30 days, so it costs at most one request a month."""
    now = now or store.now_local()
    saved = db.get("oddspapi_tournaments", None)
    if saved and (now - datetime.fromisoformat(saved["fetched"])).days < ttl_days:
        return [tuple(x) for x in saved["items"]]
    budget.allow(db, PROVIDER, "daily", 1, QUOTA_FREE, now, RESERVE)
    payload = _get("/tournaments", key, {"sportId": SOCCER}, http_get)
    budget.log(db, PROVIDER, "daily", "tournaments", 1, now)
    items = []
    for t in _items(payload):
        tid = _pick(t, "tournamentId", "id")
        name = _pick(t, "tournamentName", "name", default=str(tid))
        cat = _pick(t, "categoryName", "country", "category")
        if tid is not None:
            items.append((str(tid), f"{cat} - {name}" if cat else str(name)))
    if not items:
        _save_debug(db, payload, "tournaments parsed 0 items")
    else:
        db.set("oddspapi_tournaments", {"fetched": store.iso(now), "items": items})
    return items


def list_participants(db, key, now=None, http_get=requests.get, ttl_days=30):
    """{participantId: name} for soccer. One request, cached 30 days."""
    now = now or store.now_local()
    saved = db.get("oddspapi_participants", None)
    if saved and (now - datetime.fromisoformat(saved["fetched"])).days < ttl_days:
        return saved["names"]
    budget.allow(db, PROVIDER, "daily", 1, QUOTA_FREE, now, RESERVE)
    payload = _get("/participants", key, {"sportId": SOCCER}, http_get)
    budget.log(db, PROVIDER, "daily", "participants", 1, now)
    names = {str(k): str(v) for k, v in payload.items()} if isinstance(payload, dict) else {}
    if names:
        db.set("oddspapi_participants", {"fetched": store.iso(now), "names": names})
    else:
        _save_debug(db, payload, "participants parsed 0 names")
    return names


def suggest_tournaments(league_names, tournaments):
    """Best-effort: tournaments whose name contains the league's words (e.g. 'England Premier League')."""
    out = []
    for ln in league_names:
        words = [w for w in ln.lower().split() if len(w) > 2]
        for tid, name in tournaments:
            if words and all(w in name.lower() for w in words) and tid not in out:
                out.append(tid)
    return out


def fetch_tournaments(db, key, tournament_ids, purpose, now=None, http_get=requests.get, force=False,
                      total=QUOTA_FREE, reserve=RESERVE, jackpot_per_round=3):
    """One request for the whole batch. Returns (events, 'cache'|'api', age_hours). Only 'api' spends quota."""
    now = now or store.now_local()
    ids = sorted({str(i) for i in tournament_ids})
    if not ids:
        return [], "cache", 0.0
    ck = "oddspapi:" + ",".join(ids)
    if not force:
        hit = oddsapi.cache_get(db, ck, now)
        if hit:
            return hit[0], "cache", hit[1]
    names = list_participants(db, key, now, http_get)
    chunks = [ids[i:i + MAX_TOURNAMENTS] for i in range(0, len(ids), MAX_TOURNAMENTS)]
    budget.allow(db, PROVIDER, purpose, len(chunks), total, now, reserve, jackpot_per_round)
    events = []
    for ch in chunks:
        payload = _get("/odds-by-tournaments", key, {"bookmaker": "pinnacle", "tournamentIds": ",".join(ch)}, http_get)
        budget.log(db, PROVIDER, purpose, "oddspapi:" + ",".join(ch), 1, now)   # counted as 1 until the dashboard says otherwise
        got = normalise(payload, names)
        if not got:
            _save_debug(db, payload, f"odds-by-tournaments parsed 0 fixtures for {ch}")
        events += got
    if events:                       # never cache an empty parse: it would block the retry for 6 hours
        db.insert("odds_cache", {"sport_key": ck, "fetched": store.iso(now), "payload": json.dumps(events)})
    return events, "api", 0.0
