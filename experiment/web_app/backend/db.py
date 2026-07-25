"""
db.py
SQLite persistence for Setu.

v2 change: users now have a real account (email + password_hash) instead of
a UUID the frontend was free to generate and claim for itself. Sessions are
tied to `users.id` exactly as before, but that id can no longer be supplied
by the caller — see main.py's get_current_user() dependency, which resolves
it from a bearer token instead. That's the actual fix that matters once this
sits behind a public URL: previously, anyone could pass any user_id in the
request body and read or create sessions under it.

Design notes (unchanged from v1):
- One writer at a time: SQLite handles concurrent readers fine, but we
  serialize writes with a threading.Lock and WAL mode so readers never
  block on a write. Fine at hackathon-demo scale; swap in Postgres if this
  ever needs real concurrent load.
"""

import json
import os
import sqlite3
import threading

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setu.db")
_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    elapsed_seconds REAL NOT NULL,
    text TEXT NOT NULL,
    assistant_reply TEXT,
    acoustic_features TEXT,
    arousal_label TEXT,
    text_emotion TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS reports (
    session_id TEXT PRIMARY KEY,
    clinician_summary TEXT NOT NULL,
    patient_message TEXT NOT NULL,
    graph_path TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS report_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    distress_score INTEGER NOT NULL,
    note TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_report_segments_session ON report_segments(session_id);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON auth_tokens(user_id);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# Accounts
# ------------------------------------------------------------------

def create_account(user_id, email, password_hash, created_at):
    with _write_lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, email.lower().strip(), password_hash, created_at),
        )
        conn.commit()
        conn.close()


def get_user_by_email(email):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_token(token, user_id, created_at):
    with _write_lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, created_at),
        )
        conn.commit()
        conn.close()


def get_user_by_token(token):
    conn = get_conn()
    row = conn.execute(
        """SELECT users.* FROM users
           JOIN auth_tokens ON auth_tokens.user_id = users.id
           WHERE auth_tokens.token = ?""",
        (token,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_token(token):
    with _write_lock:
        conn = get_conn()
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
        conn.commit()
        conn.close()


# ------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------

def create_session(session_id, user_id, started_at):
    with _write_lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO sessions (id, user_id, started_at, status) VALUES (?, ?, ?, 'active')",
            (session_id, user_id, started_at),
        )
        conn.commit()
        conn.close()


def get_session(session_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def end_session(session_id, ended_at):
    with _write_lock:
        conn = get_conn()
        conn.execute(
            "UPDATE sessions SET status = 'ended', ended_at = ? WHERE id = ?",
            (ended_at, session_id),
        )
        conn.commit()
        conn.close()


def list_sessions_for_user(user_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Turns
# ------------------------------------------------------------------

def insert_turn(
    session_id, turn_index, elapsed_seconds, text, assistant_reply,
    acoustic_features, arousal_label, text_emotion, created_at,
):
    with _write_lock:
        conn = get_conn()
        conn.execute(
            """INSERT INTO turns
               (session_id, turn_index, elapsed_seconds, text, assistant_reply,
                acoustic_features, arousal_label, text_emotion, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, turn_index, elapsed_seconds, text, assistant_reply,
                json.dumps(acoustic_features), arousal_label, json.dumps(text_emotion),
                created_at,
            ),
        )
        conn.commit()
        conn.close()


def get_turns(session_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_index ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------
# Reports
# ------------------------------------------------------------------

def save_report(session_id, clinician_summary, patient_message, graph_path, created_at):
    with _write_lock:
        conn = get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO reports
               (session_id, clinician_summary, patient_message, graph_path, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, clinician_summary, patient_message, graph_path, created_at),
        )
        conn.commit()
        conn.close()


def save_report_segments(session_id, turns):
    with _write_lock:
        conn = get_conn()
        conn.execute("DELETE FROM report_segments WHERE session_id = ?", (session_id,))
        conn.executemany(
            """INSERT INTO report_segments (session_id, turn_index, distress_score, note)
               VALUES (?, ?, ?, ?)""",
            [(session_id, t["turn_index"], t["distress_score"], t.get("note", "")) for t in turns],
        )
        conn.commit()
        conn.close()


def get_report(session_id):
    conn = get_conn()
    report_row = conn.execute(
        "SELECT * FROM reports WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not report_row:
        conn.close()
        return None
    segments = conn.execute(
        """SELECT turn_index, distress_score, note FROM report_segments
           WHERE session_id = ? ORDER BY turn_index ASC""",
        (session_id,),
    ).fetchall()
    conn.close()
    report = dict(report_row)
    report["turns"] = [dict(s) for s in segments]
    return report
