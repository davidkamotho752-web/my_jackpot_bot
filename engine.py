"""engine.py - pure-logic core for the jackpot tool (no Streamlit imports).

Data source : football-data.co.uk (results + Pinnacle/Avg/Bet365 odds, free, widely used in research)
Model       : time-decayed Dixon-Coles (attack/defence/home advantage, low-score correction)
Market      : de-vigged bookmaker odds, blended with the model (weight is calibrated from data)
"""
import difflib
import io
import itertools
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

LEAGUES = {
    "E0": "England Premier League", "E1": "England Championship", "E2": "England League One",
    "E3": "England League Two", "EC": "England National League",
    "SC0": "Scotland Premiership", "SC1": "Scotland Championship",
    "SC2": "Scotland League One", "SC3": "Scotland League Two",
    "D1": "Germany Bundesliga", "D2": "Germany 2. Bundesliga",
    "I1": "Italy Serie A", "I2": "Italy Serie B",
    "SP1": "Spain La Liga", "SP2": "Spain Segunda",
    "F1": "France Ligue 1", "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie", "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga", "T1": "Turkey Super Lig", "G1": "Greece Super League",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{lg}.csv"

# Odds columns in priority order: Pinnacle closing > Pinnacle > market average > Bet365
ODDS_TRIPLES = [
    ("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA"),
    ("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
]
ODDS_COLS = sorted({c for t in ODDS_TRIPLES for c in t})
RES = {"H": 0, "D": 1, "A": 2}
LABELS = ["1", "X", "2"]
RESULT_TO_LABEL = {"H": "1", "D": "X", "A": "2"}

VERSION = "2.1"

LEDGER_COLS = [
    "ticket_id", "created", "slot", "home_in", "away_in", "home", "away",
    "o1", "oX", "o2", "pM1", "pMX", "pM2", "pD1", "pDX", "pD2",
    "pF1", "pFX", "pF2", "pick", "result",
    # added in v2.1: closing-odds probabilities (filled at settlement) and the settings used
    "pC1", "pCX", "pC2", "xi", "w", "model_version",
]


# ----------------------------------------------------------------------------- data
def season_codes(n=3, today=None):
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{s % 100:02d}{(s + 1) % 100:02d}" for s in range(start, start - n, -1)]


def _parse_dates(series):
    try:
        return pd.to_datetime(series, dayfirst=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _get_with_retry(url, tries=3):
    """GET that retries transient failures (network errors, 429, 5xx). 200/403/404 are answers, not failures."""
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code in (200, 403, 404):
                return r
            last = r
        except requests.RequestException:
            last = None
        if k < tries - 1:
            time.sleep(0.4 * (k + 1))
    return last


def _fetch(season, lg):
    url = BASE_URL.format(season=season, lg=lg)
    try:
        r = _get_with_retry(url)
        if r is None or r.status_code != 200 or len(r.content) < 200:
            return None, f"{season}/{lg}"
        df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
    except Exception:
        return None, f"{season}/{lg}"
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if not all(c in df.columns for c in need):
        return None, f"{season}/{lg}"
    keep = need + [c for c in ODDS_COLS if c in df.columns]
    df = df[keep].copy()
    for c in ["FTHG", "FTAG"] + [c for c in ODDS_COLS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")           # blanks / stray text become NaN, never a crash later
    df["Date"] = _parse_dates(df["Date"])
    df = df.dropna(subset=need)
    df["League"] = lg
    df["Season"] = season
    return df, None


def download_football_data(n_seasons=3):
    """Returns (DataFrame of all matches, list of failed 'season/league' downloads)."""
    jobs = [(s, lg) for s in season_codes(n_seasons) for lg in LEAGUES]
    frames, failed = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for df, err in pool.map(lambda j: _fetch(*j), jobs):
            if df is not None and len(df):
                frames.append(df)
            elif err:
                failed.append(err)
    if not frames:
        return pd.DataFrame(), failed
    data = pd.concat(frames, ignore_index=True)
    data["FTHG"] = data["FTHG"].astype(int)
    data["FTAG"] = data["FTAG"].astype(int)
    for c in ODDS_COLS:
        if c not in data.columns:
            data[c] = np.nan
    data = data.sort_values("Date").reset_index(drop=True)
    return data, failed


def add_market_probs(df):
    """Adds mH/mD/mA = de-vigged bookmaker probabilities (best available odds source per row)."""
    df = df.copy()
    out = {k: pd.Series(np.nan, index=df.index) for k in ("mH", "mD", "mA")}
    for h, d, a in ODDS_TRIPLES:
        if not all(c in df.columns for c in (h, d, a)):
            continue
        inv = 1.0 / df[[h, d, a]].astype(float)
        tot = inv.sum(axis=1, skipna=False)
        ph, pdr, pa = inv[h] / tot, inv[d] / tot, inv[a] / tot
        fill = out["mH"].isna() & ph.notna()
        out["mH"][fill], out["mD"][fill], out["mA"][fill] = ph[fill], pdr[fill], pa[fill]
    for k, v in out.items():
        df[k] = v
    return df


def team_league_map(data):
    long = pd.concat([
        data[["Date", "HomeTeam", "League"]].rename(columns={"HomeTeam": "Team"}),
        data[["Date", "AwayTeam", "League"]].rename(columns={"AwayTeam": "Team"}),
    ])
    return long.sort_values("Date").groupby("Team")["League"].last().to_dict()


# ----------------------------------------------------------------------------- model
def fit_dc(df, xi, ref_date, reg=0.5):
    """Time-decayed Dixon-Coles fit on one league. xi = decay per day (0.002 ~ half-life 1 year)."""
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df["HomeTeam"].map(idx).to_numpy()
    ai = df["AwayTeam"].map(idx).to_numpy()
    x = df["FTHG"].to_numpy(float)
    y = df["FTAG"].to_numpy(float)
    days = (ref_date - df["Date"]).dt.days.clip(lower=0).to_numpy(float)
    w = np.exp(-xi * days)
    gl = gammaln(x + 1) + gammaln(y + 1)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    def nll(p):
        att = p[:n] - p[:n].mean()
        dfn = p[n:2 * n] - p[n:2 * n].mean()
        mu, hadv, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        llh = mu + hadv + att[hi] + dfn[ai]
        lla = mu + att[ai] + dfn[hi]
        lh, la = np.exp(llh), np.exp(lla)
        tau = np.ones_like(lh)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        tau = np.maximum(tau, 1e-6)
        ll = x * llh - lh + y * lla - la - gl + np.log(tau)
        return -(w * ll).sum() + reg * (att @ att + dfn @ dfn)

    p0 = np.zeros(2 * n + 3)
    p0[2 * n], p0[2 * n + 1] = 0.2, 0.25
    bounds = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    p = res.x
    neff = np.bincount(hi, weights=w, minlength=n) + np.bincount(ai, weights=w, minlength=n)
    return {
        "idx": idx,
        "att": p[:n] - p[:n].mean(),
        "dfn": p[n:2 * n] - p[n:2 * n].mean(),
        "mu": float(p[2 * n]), "h": float(p[2 * n + 1]), "rho": float(p[2 * n + 2]),
        "neff": {t: float(neff[i]) for t, i in idx.items()},
    }


def score_matrix(model, home, away, max_goals=10):
    """Full scoreline probabilities: M[i, j] = P(home scores i, away scores j)."""
    i, j = model["idx"][home], model["idx"][away]
    lh = np.exp(model["mu"] + model["h"] + model["att"][i] + model["dfn"][j])
    la = np.exp(model["mu"] + model["att"][j] + model["dfn"][i])
    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    rho = model["rho"]
    M[0, 0] *= max(1 - lh * la * rho, 0.0)
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()


def outcome_probs(M):
    """np.array([P(home), P(draw), P(away)]) from a scoreline matrix."""
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])


def p_over(M, line=2.5):
    i, j = np.indices(M.shape)
    return float(M[(i + j) > line].sum())


def predict_dc(model, home, away, max_goals=10):
    """Returns np.array([P(home), P(draw), P(away)])."""
    return outcome_probs(score_matrix(model, home, away, max_goals))


def fit_all(data, xi):
    ref = data["Date"].max()
    models = {}
    for lg, dfl in data.groupby("League"):
        if len(dfl) >= 150:
            models[lg] = fit_dc(dfl, xi, ref)
    return models


def blend(mkt, model, w, min_neff=None):
    if model is None:
        return mkt
    if min_neff is not None and min_neff < 10:
        w = max(w, 0.8)  # too little data on a team -> trust the market more
    p = w * np.asarray(mkt) + (1 - w) * np.asarray(model)
    return p / p.sum()


def devig(o1, ox, o2):
    inv = np.array([1 / o1, 1 / ox, 1 / o2], dtype=float)
    return inv / inv.sum(), float(inv.sum() - 1)


def logloss(P, Y):
    P, Y = np.asarray(P), np.asarray(Y)
    return float(-np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-9, 1))))


def calibrate(data, xi_grid, holdout_days=75, progress=None):
    """Walk-forward check: fit on matches before a cutoff, score the model, the market and blends
    on matches after it. Returns a DataFrame; pick the row with the lowest log loss."""
    data = add_market_probs(data)
    cutoff = data["Date"].max() - pd.Timedelta(days=holdout_days)
    rows = []
    for k, xi in enumerate(xi_grid):
        PM, PK, Y = [], [], []
        for lg, dfl in data.groupby("League"):
            train = dfl[dfl["Date"] <= cutoff]
            test = dfl[dfl["Date"] > cutoff].dropna(subset=["mH", "mD", "mA"])
            if len(train) < 300 or len(test) < 10:
                continue
            m = fit_dc(train, xi, cutoff)
            for r in test.itertuples(index=False):
                if r.HomeTeam in m["idx"] and r.AwayTeam in m["idx"]:
                    PM.append(predict_dc(m, r.HomeTeam, r.AwayTeam))
                    PK.append([r.mH, r.mD, r.mA])
                    Y.append(RES[r.FTR])
        if progress:
            progress((k + 1) / len(xi_grid))
        if len(Y) < 100:
            continue
        PM, PK, Y = np.array(PM), np.array(PK), np.array(Y)
        ll_m, ll_k = logloss(PM, Y), logloss(PK, Y)
        rps_m, rps_k = rps(PM, Y), rps(PK, Y)
        for w in np.linspace(0, 1, 11):
            blend_p = w * PK + (1 - w) * PM
            rows.append({"xi": xi, "w": round(float(w), 2), "logloss": logloss(blend_p, Y),
                         "rps": rps(blend_p, Y), "model_only": ll_m, "market_only": ll_k,
                         "rps_model": rps_m, "rps_market": rps_k, "matches": len(Y)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- team name matching
_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "sv", "fk", "the", "club", "cd", "ud", "cp"}


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return " ".join(t for t in s.split() if t not in _STOP)


_RAW_ALIASES = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Nottm Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves",
    "West Ham United": "West Ham", "Brighton & Hove Albion": "Brighton",
    "Leicester City": "Leicester", "Leeds United": "Leeds", "Norwich City": "Norwich",
    "West Bromwich Albion": "West Brom", "Queens Park Rangers": "QPR",
    "Sheffield Wednesday": "Sheffield Weds", "Sheffield United": "Sheffield United",
    "Atletico Madrid": "Ath Madrid", "Athletic Bilbao": "Ath Bilbao", "Athletic Club": "Ath Bilbao",
    "Real Sociedad": "Sociedad", "Celta Vigo": "Celta", "Rayo Vallecano": "Vallecano",
    "RCD Espanyol": "Espanol", "Espanyol": "Espanol", "Deportivo Alaves": "Alaves",
    "Paris Saint-Germain": "Paris SG", "Paris Saint Germain": "Paris SG", "PSG": "Paris SG",
    "AC Milan": "Milan", "Inter Milan": "Inter", "Internazionale": "Inter",
    "Bayern Munich": "Bayern Munich", "Borussia Dortmund": "Dortmund",
    "Borussia Monchengladbach": "M'gladbach", "Borussia M'gladbach": "M'gladbach",
    "Bayer Leverkusen": "Leverkusen", "Bayer 04 Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt", "1. FC Koln": "FC Koln", "FC Cologne": "FC Koln",
    "Cologne": "FC Koln", "Koln": "FC Koln", "Union Berlin": "Union Berlin", "1. FC Union Berlin": "Union Berlin",
    "VfB Stuttgart": "Stuttgart", "VfL Wolfsburg": "Wolfsburg", "FSV Mainz 05": "Mainz", "Mainz 05": "Mainz",
    "TSG Hoffenheim": "Hoffenheim", "1. FC Heidenheim": "Heidenheim", "FC Augsburg": "Augsburg",
    "Sporting CP": "Sp Lisbon", "Sporting Lisbon": "Sp Lisbon", "FC Porto": "Porto",
    "PSV Eindhoven": "PSV Eindhoven", "PSV": "PSV Eindhoven", "Ajax Amsterdam": "Ajax",
    "Olympique Marseille": "Marseille", "Olympique Lyonnais": "Lyon", "Olympique Lyon": "Lyon",
    "AS Monaco": "Monaco", "Stade Rennais": "Rennes", "OGC Nice": "Nice", "Lille OSC": "Lille",
    "Fenerbahce": "Fenerbahce", "Besiktas": "Besiktas", "Galatasaray": "Galatasaray",
    "Hellas Verona": "Verona", "AS Roma": "Roma", "SS Lazio": "Lazio", "SSC Napoli": "Napoli",
    "Club Brugge": "Club Brugge", "Royal Antwerp": "Antwerp", "Union Saint-Gilloise": "St. Gilloise",
    "Celtic": "Celtic", "Rangers": "Rangers", "Heart of Midlothian": "Hearts",
}
ALIASES = {norm(k): v for k, v in _RAW_ALIASES.items()}


def resolve_team(name, known, known_norm):
    """Map a SportyBet team name to a football-data.co.uk team name (or None)."""
    name = str(name).strip()
    if not name:
        return None
    if name in known:
        return name
    n = norm(name)
    alias = ALIASES.get(n)
    if alias and alias in known:
        return alias
    if n in known_norm:
        return known_norm[n]
    target = norm(alias) if alias else n
    if target in known_norm:
        return known_norm[target]
    cand = difflib.get_close_matches(target, list(known_norm), n=1, cutoff=0.78)
    return known_norm[cand[0]] if cand else None


# ----------------------------------------------------------------------------- fixture parsing
_LINE_RE = re.compile(
    r"^(?P<h>.+?)\s+(?:vs\.?|v|-|\u2013|\u2014)\s+(?P<a>.+?)[\s,;|\t]+"
    r"(?P<o1>\d+[.,]\d+)[\s,;|\t]+(?P<ox>\d+[.,]\d+)[\s,;|\t]+(?P<o2>\d+[.,]\d+)\s*$", re.I)


def _f(s):
    return float(str(s).replace(",", "."))


def parse_fixtures(text):
    """Parses lines like 'Arsenal vs Chelsea 2.10 3.40 3.20' (or tab/pipe separated).
    Returns (list of dicts, list of lines that could not be read)."""
    rows, bad = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"\t|\|", line) if p.strip()]
        try:
            if len(parts) >= 5:
                rows.append({"Home": parts[0], "Away": parts[1],
                             "Odds 1": _f(parts[-3]), "Odds X": _f(parts[-2]), "Odds 2": _f(parts[-1])})
                continue
            m = _LINE_RE.match(line)
            if m:
                rows.append({"Home": m["h"].strip(), "Away": m["a"].strip(),
                             "Odds 1": _f(m["o1"]), "Odds X": _f(m["ox"]), "Odds 2": _f(m["o2"])})
                continue
        except ValueError:
            pass
        bad.append(line)
    return rows, bad


# ----------------------------------------------------------------------------- jackpot slip
def poisson_binomial(ps):
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1 - p, p])
    return dist


def build_slip(P, max_cols):
    """Greedy: start with the single most likely outcome per match, then add extra outcomes
    (doubles/triples) where they buy the most extra hit-probability per unit of extra cost,
    without exceeding max_cols columns (full-cross-product ticket)."""
    P = np.asarray(P, float)
    n = len(P)
    order = np.argsort(-P, axis=1)
    sel = [[int(order[i, 0])] for i in range(n)]
    cols = 1
    while True:
        best, best_gain = None, 0.0
        for i in range(n):
            k = len(sel[i])
            if k >= 3:
                continue
            new_cols = cols // k * (k + 1)
            if new_cols > max_cols:
                continue
            s = P[i, sel[i]].sum()
            add = P[i, order[i, k]]
            gain = np.log((s + add) / s) / np.log((k + 1) / k)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        k = len(sel[best])
        cols = cols // k * (k + 1)
        sel[best].append(int(order[best, k]))
    return sel, cols


def slip_stats(P, sel):
    """With a full cross-product ticket, the best column has (n - uncovered) correct picks."""
    P = np.asarray(P, float)
    n = len(P)
    cover = np.array([P[i, sel[i]].sum() for i in range(n)])
    dist = poisson_binomial(1 - cover)
    return {"cover": cover, "p_all": float(np.prod(cover)),
            "p_minus1": float(dist[:2].sum()), "p_minus2": float(dist[:3].sum())}


def picks_str(sel_i):
    return "".join(LABELS[j] for j in sorted(sel_i))


# ----------------------------------------------------------------------------- ledger
def settle_pending(ledger, data):
    updated = 0
    for i, row in ledger[ledger["result"] == "Pending"].iterrows():
        if not (isinstance(row["home"], str) and isinstance(row["away"], str) and row["home"]):
            continue
        created = pd.to_datetime(row["created"])
        m = data[(data["HomeTeam"] == row["home"]) & (data["AwayTeam"] == row["away"]) &
                 (data["Date"] >= created - pd.Timedelta(days=1)) &
                 (data["Date"] <= created + pd.Timedelta(days=21))]
        if not m.empty:
            last = m.iloc[-1]
            ledger.at[i, "result"] = RESULT_TO_LABEL[last["FTR"]]
            if "mH" in m.columns and pd.notna(last["mH"]):
                ledger.at[i, "pC1"], ledger.at[i, "pCX"], ledger.at[i, "pC2"] = last["mH"], last["mD"], last["mA"]
            updated += 1
    return ledger, updated


def ledger_metrics(ledger):
    s = ledger[ledger["result"].isin(LABELS) & ledger["pF1"].notna()]
    if s.empty:
        return None
    y = s["result"].map({"1": 0, "X": 1, "2": 2}).to_numpy()
    PF = s[["pF1", "pFX", "pF2"]].to_numpy(float)
    PM = s[["pM1", "pMX", "pM2"]].to_numpy(float)
    eye = np.eye(3)[y]
    return {
        "n": len(s),
        "accuracy": float(np.mean(PF.argmax(1) == y)),
        "logloss": logloss(PF, y), "logloss_market": logloss(PM, y),
        "rps": rps(PF, y), "rps_market": rps(PM, y),
        "brier": float(np.mean(np.sum((PF - eye) ** 2, axis=1))),
    }


def rps(P, Y):
    """Ranked probability score for ordered outcomes (1, X, 2). Lower is better."""
    P, Y = np.asarray(P, float), np.asarray(Y)
    cp = np.cumsum(P, axis=1)[:, :2]
    co = np.cumsum(np.eye(3)[Y], axis=1)[:, :2]
    return float(np.mean(0.5 * np.sum((cp - co) ** 2, axis=1)))


# ----------------------------------------------------------------------------- pool-aware economics
# SportyBet Sporty 13 rules (from the rules PDF): KES 50 per combination, extra picks multiply the cost,
# 13/13 prize (base ~KES 15M) is split equally among winning tickets, 12/11 consolation amounts are dynamic.
STAKE = 50.0


def apply_shift(p, pts):
    """Manual adjustment: +pts moves that many percentage points from Away to Home (negative = reverse)."""
    p = np.asarray(p, float)
    if not pts:
        return p
    q = np.clip(p + np.array([pts, 0.0, -pts]) / 100.0, 0.01, None)
    return q / q.sum()


def crowd_shares(mkt, gamma=1.25, draw_factor=0.85):
    """PLACEHOLDER model of how other players pick: favourites over-picked (gamma > 1), draws under-picked
    (draw_factor < 1). Replace with real pick percentages whenever you have them."""
    c = np.asarray(mkt, float) ** gamma * np.array([1.0, draw_factor, 1.0])
    return c / c.sum()


def share_factor(lam):
    """Expected fraction of a split prize you keep when the number of OTHER winners ~ Poisson(lam):
    E[1/(1+K)] = (1 - exp(-lam)) / lam."""
    lam = np.asarray(lam, float)
    safe = np.maximum(lam, 1e-12)
    return np.where(lam < 1e-9, 1.0, -np.expm1(-safe) / safe)


def ticket_ev(P, C, sel, prize, n_tickets, cons12=0.0, cons11=0.0, stake=STAKE):
    """Expected value of a full-cross-product ticket.
    P = our match probabilities (n x 3), C = crowd pick shares (n x 3), sel = chosen outcomes per match.
    Other winners on a given column ~ Poisson(n_tickets * product of crowd shares on that column)."""
    P, C = np.asarray(P, float), np.asarray(C, float)
    n = len(P)
    cols = np.array(list(itertools.product(*sel)))
    rows = np.arange(n)
    ps, cs = P[rows, cols], C[rows, cols]
    p_all = ps.prod(axis=1)
    lam = n_tickets * cs.prod(axis=1)
    ev_top = float((p_all * prize * share_factor(lam)).sum())
    ev_cons = 0.0
    if cons12 or cons11:
        for pr in ps:
            d = poisson_binomial(pr)
            ev_cons += cons12 * d[n - 1] + cons11 * d[n - 2]
    cost = len(cols) * stake
    ev = ev_top + ev_cons
    return {"ncols": len(cols), "cost": cost, "p_hit": float(p_all.sum()), "ev_top": ev_top,
            "ev_cons": float(ev_cons), "ev": ev, "net": ev - cost, "roi": (ev - cost) / cost}


def optimize_slip_ev(P, C, max_cols, prize, n_tickets, cons12=0.0, cons11=0.0):
    """Pool-aware slip: (1) best single column by expected value, (2) add extra outcomes where they buy
    the most expected value per extra KES of stake, up to max_cols combinations."""
    P = np.asarray(P, float)
    n = len(P)
    kw = dict(prize=prize, n_tickets=n_tickets, cons12=cons12, cons11=cons11)
    sel = [[int(np.argmax(P[i]))] for i in range(n)]
    best = ticket_ev(P, C, sel, **kw)["ev"]
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for o in range(3):
                if o == sel[i][0]:
                    continue
                trial = [s[:] for s in sel]
                trial[i] = [o]
                v = ticket_ev(P, C, trial, **kw)["ev"]
                if v > best * (1 + 1e-12):
                    sel, best, improved = trial, v, True
    cols = 1
    while True:
        cur = ticket_ev(P, C, sel, **kw)["ev"]
        pick, best_ratio = None, 0.0
        for i in range(n):
            k = len(sel[i])
            if k >= 3:
                continue
            new_cols = cols // k * (k + 1)
            if new_cols > max_cols:
                continue
            for o in range(3):
                if o in sel[i]:
                    continue
                trial = [s[:] for s in sel]
                trial[i] = sel[i] + [o]
                ratio = (ticket_ev(P, C, trial, **kw)["ev"] - cur) / ((new_cols - cols) * STAKE)
                if ratio > best_ratio:
                    pick, best_ratio = (i, o, new_cols), ratio
        if pick is None:
            break
        sel[pick[0]].append(pick[1])
        cols = pick[2]
    return sel, cols


def next_deadline(now=None):
    """Next Saturday 17:00 Kenya time (the Sporty 13 entry deadline)."""
    now = now or datetime.now(ZoneInfo("Africa/Nairobi"))
    d = now.replace(hour=17, minute=0, second=0, microsecond=0)
    d += timedelta(days=(5 - d.weekday()) % 7)
    if d <= now:
        d += timedelta(days=7)
    return d
