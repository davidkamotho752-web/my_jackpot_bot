"""SQLite wrapper. All modules use this; no direct sqlite3 elsewhere."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

DB_PATH = os.environ.get("ENGINE_DB", "data/engine.db")
NAIROBI = timezone(timedelta(hours=3))

def now_local():
    return datetime.now(NAIROBI)

def iso(dt):
    return dt.astimezone(NAIROBI).replace(microsecond=0).isoformat()

@contextmanager
def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def q(sql, params=()):
    with connect() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]

def x(sql, params=()):
    with connect() as c:
        cur = c.execute(sql, params)
        return cur.lastrowid

def insert(table, row):
    cols = ", ".join(row.keys())
    marks = ", ".join("?" for _ in row)
    return x(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))

def get(key, default=None):
    rows = q("SELECT value FROM kv WHERE key=?", (key,))
    return rows[0]["value"] if rows else default

def set_key(key, value):
    x("INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

def log_event(kind, detail):
    insert("events", {"ts": iso(now_local()), "kind": kind, "detail": detail})

def backup_bytes():
    with connect() as c:
        return c.serialize()

def restore_bytes(blob):
    with open(DB_PATH, "wb") as f:
        f.write(blob)
