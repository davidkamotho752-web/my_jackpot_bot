"""store.py - SQLite storage for the whole tool (free, one file, easy to back up).

Audit rules enforced by database triggers:
  * predictions can never be deleted, and their probabilities/odds can never be edited
    (only the result and goals can be filled in later);
  * bets can never be deleted, and their stake/odds/selection can never be edited
    (only status, pnl and settlement fields).
"""
import json
import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

TZ = ZoneInfo("Africa/Nairobi")


def now_local():
    return datetime.now(TZ)


def iso(dt):
    return dt.isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS param_versions(id INTEGER PRIMARY KEY, ts TEXT, name TEXT, value TEXT, evidence TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY, created TEXT, source TEXT, group_id TEXT, slot INTEGER,
  home_in TEXT, away_in TEXT, home TEXT, away TEXT, league TEXT,
  o1 REAL, ox REAL, o2 REAL,
  pM1 REAL, pMX REAL, pM2 REAL,
  pD1 REAL, pDX REAL, pD2 REAL,
  pF1 REAL, pFX REAL, pF2 REAL,
  pover REAL, model_ver TEXT, param_ver INTEGER,
  result TEXT DEFAULT 'Pending', fthg INTEGER, ftag INTEGER);
CREATE TABLE IF NOT EXISTS bets(
  id INTEGER PRIMARY KEY, created TEXT, pred_id INTEGER, mode TEXT, decision TEXT, reason TEXT,
  home TEXT, away TEXT, selection TEXT, odds REAL, p REAL, p_low REAL, edge_low REAL,
  stake REAL, min_odds REAL, status TEXT DEFAULT 'Pending', pnl REAL, unit_pnl REAL, settled TEXT);
CREATE TABLE IF NOT EXISTS bankroll(
  id INTEGER PRIMARY KEY, ts TEXT, mode TEXT, kind TEXT, amount REAL, note TEXT, bet_id INTEGER);
CREATE TABLE IF NOT EXISTS tickets(
  id INTEGER PRIMARY KEY, created TEXT, group_id TEXT, mode TEXT, ncols INTEGER, cost REAL,
  p_hit REAL, ev REAL, picks TEXT, assumptions TEXT);
CREATE TABLE IF NOT EXISTS rounds(
  id INTEGER PRIMARY KEY, round_date TEXT, super_pool REAL, prize2 REAL, prize3 REAL,
  winners13 INTEGER, winners12 INTEGER, winners11 INTEGER,
  payout13 REAL, payout12 REAL, payout11 REAL, tickets_est REAL, notes TEXT);
CREATE TABLE IF NOT EXISTS api_usage(
  id INTEGER PRIMARY KEY, ts TEXT, provider TEXT, cost INTEGER, remaining INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS odds_cache(
  id INTEGER PRIMARY KEY, sport_key TEXT, fetched TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS tipster_picks(
  id INTEGER PRIMARY KEY, created TEXT, tipster TEXT, home_in TEXT, away_in TEXT, home TEXT, away TEXT,
  selection TEXT, odds REAL, note TEXT, status TEXT DEFAULT 'Pending', unit_pnl REAL, clv REAL, match_date TEXT);
"""

# 'auto' rows are the tool's own calls on every match it judged (no effort from the user). They count as evidence about the model,
# but never twice: when the user also saved the same match, only the user's row counts.
AUTO_DEDUP = ("(source<>'auto' OR NOT EXISTS (SELECT 1 FROM predictions d WHERE d.source='daily' AND d.home=predictions.home "
              "AND d.away=predictions.away AND substr(d.created,1,10)=substr(predictions.created,1,10)))")

PRED_LOCKED = ["created", "source", "group_id", "slot", "home_in", "away_in", "home", "away", "league",
               "o1", "ox", "o2", "pM1", "pMX", "pM2", "pD1", "pDX", "pD2", "pF1", "pFX", "pF2",
               "pover", "model_ver", "param_ver"]
BET_LOCKED = ["created", "pred_id", "mode", "decision", "reason", "home", "away", "selection", "odds",
              "p", "p_low", "edge_low", "stake", "min_odds"]


def _trigger_sql():
    pc = " OR ".join(f"OLD.{c} IS NOT NEW.{c}" for c in PRED_LOCKED)
    bc = " OR ".join(f"OLD.{c} IS NOT NEW.{c}" for c in BET_LOCKED)
    return f"""
CREATE TRIGGER IF NOT EXISTS pred_no_delete BEFORE DELETE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS pred_no_edit BEFORE UPDATE ON predictions WHEN {pc}
BEGIN SELECT RAISE(ABORT, 'predictions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS bet_no_delete BEFORE DELETE ON bets
BEGIN SELECT RAISE(ABORT, 'bets are append-only'); END;
CREATE TRIGGER IF NOT EXISTS bet_no_edit BEFORE UPDATE ON bets WHEN {bc}
BEGIN SELECT RAISE(ABORT, 'bets are append-only'); END;
"""


class DB:
    def __init__(self, path=":memory:"):
        self.path = path
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.executescript(SCHEMA + _trigger_sql())
        cols = [r[1] for r in self.con.execute("PRAGMA table_info(tipster_picks)")]
        if "kind" not in cols:                                   # older databases: tipster picks get a daily/jackpot label
            self.con.execute("ALTER TABLE tipster_picks ADD COLUMN kind TEXT DEFAULT 'daily'")
        self.con.commit()

    # -- backup / restore (Streamlit Cloud wipes local files on reboot, so the user keeps a copy)
    def export_bytes(self):
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            dst = sqlite3.connect(tmp)
            self.con.backup(dst)
            dst.close()
            with open(tmp, "rb") as f:
                return f.read()
        finally:
            os.remove(tmp)

    def restore_bytes(self, raw):
        """Replaces everything with an uploaded backup. Returns (ok, message). Refuses files that are not a bet-engine backup."""
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                f.write(raw)
            src = sqlite3.connect(tmp)
            try:
                tabs = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                need = {"predictions", "bets", "bankroll", "settings"}
                if not need <= tabs:
                    return False, "That file is not a Bet Engine backup (expected tables are missing). Nothing was changed."
                src.backup(self.con)
            finally:
                src.close()
            self.con.executescript(SCHEMA + _trigger_sql())
            cols = [r[1] for r in self.con.execute("PRAGMA table_info(tipster_picks)")]
            if "kind" not in cols:
                self.con.execute("ALTER TABLE tipster_picks ADD COLUMN kind TEXT DEFAULT 'daily'")
            self.con.commit()
            return True, "Backup restored."
        except sqlite3.DatabaseError:
            return False, "That file could not be read as a backup. Nothing was changed."
        finally:
            os.remove(tmp)

    # -- generic helpers
    def q(self, sql, params=()):
        return pd.read_sql_query(sql, self.con, params=params)

    def x(self, sql, params=()):
        cur = self.con.execute(sql, params)
        self.con.commit()
        return cur

    def insert(self, table, row):
        cols = list(row)
        cur = self.con.execute(
            f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
            [None if (isinstance(v, float) and v != v) else v for v in row.values()])
        self.con.commit()
        return cur.lastrowid

    def get(self, key, default=None):
        r = self.con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set(self, key, value):
        self.x("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               (key, json.dumps(value)))

    def event(self, kind, detail, now=None):
        self.insert("events", {"ts": iso(now or now_local()), "kind": kind, "detail": detail})

    def log_param(self, name, value, evidence, now=None):
        return self.insert("param_versions", {"ts": iso(now or now_local()), "name": name,
                                              "value": json.dumps(value), "evidence": evidence})

    def param_version(self):
        r = self.con.execute("SELECT COALESCE(MAX(id),0) FROM param_versions").fetchone()
        return int(r[0])

    def backup(self, dest):
        out = sqlite3.connect(dest)
        self.con.backup(out)
        out.close()
