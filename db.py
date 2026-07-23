"""SQLite persistence: everything the brain remembers lives here.

The learning loop depends on one discipline enforced in this module:
EVERY decision — buys and skips alike — is stored with its full feature
vector, and outcomes are tracked for both. Skips are the counterfactuals;
without them the trainer would only ever see what the current weights
already liked (pure survivorship bias).
"""

import json
import sqlite3
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS tokens (
    mint TEXT PRIMARY KEY,
    symbol TEXT, name TEXT,
    first_seen TEXT,
    age_hours_at_seen REAL,
    bracket TEXT,               -- fresh | young
    narrative TEXT,
    raw JSON
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    ts TEXT, mint TEXT,
    action TEXT,                -- buy | skip
    score REAL,
    price_usd REAL,
    features JSON,              -- full vector, replayable by the trainer
    weights_version INTEGER,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS outcomes (
    decision_id INTEGER PRIMARY KEY REFERENCES decisions(id),
    mint TEXT, base_price REAL,
    m5 REAL, m30 REAL, m120 REAL, m1440 REAL,   -- price multiples vs base
    rugged INTEGER DEFAULT 0,
    label TEXT,                 -- win | loss | flat | NULL while pending
    done INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    mint TEXT, symbol TEXT,
    opened_ts TEXT, entry_price REAL, size_sol REAL,
    remaining_frac REAL DEFAULT 1.0,
    tp1_done INTEGER DEFAULT 0,
    status TEXT DEFAULT 'open', -- open | closed
    exit_ts TEXT, exit_reason TEXT,
    pnl_sol REAL
);
CREATE TABLE IF NOT EXISTS weights (
    version INTEGER PRIMARY KEY,
    ts TEXT, weights JSON, note TEXT
);
CREATE TABLE IF NOT EXISTS meta_state (
    key TEXT PRIMARY KEY, value JSON, ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_mint ON decisions(mint);
CREATE INDEX IF NOT EXISTS idx_outcomes_done ON outcomes(done);
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    con = sqlite3.connect(config.DB_FILE)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def token_known(con, mint):
    return con.execute("SELECT 1 FROM tokens WHERE mint=?", (mint,)).fetchone() is not None


def save_token(con, mint, symbol, name, age_hours, bracket, narrative, raw):
    con.execute(
        "INSERT OR IGNORE INTO tokens VALUES (?,?,?,?,?,?,?,?)",
        (mint, symbol, name, now(), age_hours, bracket, narrative, json.dumps(raw)),
    )
    con.commit()


def record_decision(con, mint, action, score, price, features, weights_version, reason):
    cur = con.execute(
        "INSERT INTO decisions (ts,mint,action,score,price_usd,features,weights_version,reason)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (now(), mint, action, score, price, json.dumps(features), weights_version, reason),
    )
    con.execute(
        "INSERT INTO outcomes (decision_id, mint, base_price) VALUES (?,?,?)",
        (cur.lastrowid, mint, price),
    )
    con.commit()
    return cur.lastrowid


def pending_outcomes(con):
    return con.execute(
        "SELECT o.*, d.ts AS decision_ts FROM outcomes o"
        " JOIN decisions d ON d.id = o.decision_id WHERE o.done = 0"
    ).fetchall()


def update_outcome(con, decision_id, **fields):
    sets = ", ".join(f"{k}=?" for k in fields)
    con.execute(f"UPDATE outcomes SET {sets} WHERE decision_id=?",
                (*fields.values(), decision_id))
    con.commit()


def labeled_decisions(con, limit=2000):
    """Feature vectors with settled labels — the trainer's dataset."""
    return con.execute(
        "SELECT d.features, d.action, o.label FROM decisions d"
        " JOIN outcomes o ON o.decision_id = d.id"
        " WHERE o.done = 1 AND o.label IS NOT NULL"
        " ORDER BY d.id DESC LIMIT ?", (limit,)
    ).fetchall()


def open_positions(con):
    return con.execute("SELECT * FROM positions WHERE status='open'").fetchall()


def positions_opened_today(con):
    today = now()[:10]
    return con.execute(
        "SELECT COUNT(*) c FROM positions WHERE opened_ts LIKE ?", (today + "%",)
    ).fetchone()["c"]


def open_position(con, mint, symbol, entry_price, size_sol):
    con.execute(
        "INSERT INTO positions (mint,symbol,opened_ts,entry_price,size_sol) VALUES (?,?,?,?,?)",
        (mint, symbol, now(), entry_price, size_sol),
    )
    con.commit()


def update_position(con, pos_id, **fields):
    sets = ", ".join(f"{k}=?" for k in fields)
    con.execute(f"UPDATE positions SET {sets} WHERE id=?", (*fields.values(), pos_id))
    con.commit()


def latest_weights(con):
    row = con.execute("SELECT * FROM weights ORDER BY version DESC LIMIT 1").fetchone()
    if row:
        return row["version"], json.loads(row["weights"])
    return 0, None


def save_weights(con, weights, note):
    version, _ = latest_weights(con)
    con.execute("INSERT INTO weights VALUES (?,?,?,?)",
                (version + 1, now(), json.dumps(weights), note))
    con.commit()
    return version + 1


def get_meta(con, key, default=None):
    row = con.execute("SELECT value FROM meta_state WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_meta(con, key, value):
    con.execute("INSERT OR REPLACE INTO meta_state VALUES (?,?,?)",
                (key, json.dumps(value), now()))
    con.commit()
