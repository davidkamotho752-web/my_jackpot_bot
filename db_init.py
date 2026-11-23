"""Create the SQLite schema. Safe to run repeatedly."""
import core.db as db

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    match_date TEXT,
    league TEXT,
    home TEXT NOT NULL, home_in TEXT NOT NULL,
    away TEXT NOT NULL, away_in TEXT NOT NULL,
    o1 REAL, ox REAL, o2 REAL,
    pF1 REAL, pFX REAL, pF2 REAL,
    pM1 REAL, pMX REAL, pM2 REAL,
    pD1 REAL, pDX REAL, pD2 REAL,
    ref_source TEXT, selection TEXT, decision TEXT,
    stake REAL, odds REAL, min_odds REAL, edge_low REAL,
    bet_type TEXT DEFAULT 'REAL', action TEXT,
    result TEXT DEFAULT 'Pending',
    fthg INTEGER, ftag INTEGER, clv REAL, settled TEXT
);
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pred_id INTEGER, created TEXT, mode TEXT,
    selection TEXT, odds REAL, stake REAL,
    status TEXT DEFAULT 'Pending', pnl REAL, unit_pnl REAL, clv REAL,
    bet_type TEXT DEFAULT 'REAL', decision TEXT,
    home TEXT, away TEXT, settled TEXT
);
CREATE TABLE IF NOT EXISTS camouflage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, match TEXT, market TEXT,
    camo_type TEXT, stake REAL, odds REAL,
    outcome TEXT DEFAULT 'Pending', pnl REAL,
    reason TEXT, account_health_snapshot TEXT
);
CREATE TABLE IF NOT EXISTS tipster_picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, tipster TEXT, home_in TEXT, away_in TEXT,
    selection TEXT, odds REAL, match_date TEXT,
    status TEXT DEFAULT 'Pending', unit_pnl REAL, clv REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS team_aliases (
    typed TEXT PRIMARY KEY, official TEXT NOT NULL, created TEXT
);
CREATE TABLE IF NOT EXISTS custom_teams (
    name TEXT PRIMARY KEY, created TEXT
);
CREATE TABLE IF NOT EXISTS bankroll_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, mode TEXT, kind TEXT, amount REAL, balance REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS account_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, kind TEXT, detail TEXT, severity TEXT
);
CREATE TABLE IF NOT EXISTS param_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, name TEXT, value TEXT, evidence TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, kind TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY, value TEXT
);
"""

def main():
    with db.connect() as c:
        c.executescript(SCHEMA)
    db.set_key("schema_version", "1")
    db.log_event("init", "schema created")
    print(f"Schema ready at {db.DB_PATH}")

if __name__ == "__main__":
    main()
