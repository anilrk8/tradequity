"""
Database initialisation and connection management.
Uses SQLite — zero setup, single file, runs fully locally.
"""

import os
import sqlite3
from pathlib import Path

# On Railway: set DB_PATH env var to point at the persistent volume mount,
# e.g.  DB_PATH=/data/stocks.db
# Locally: falls back to  <project_root>/data/stocks.db
_default = Path(__file__).parent.parent / "data" / "stocks.db"
DB_PATH = Path(os.environ.get("DB_PATH", str(_default)))


def get_connection() -> sqlite3.Connection:
    """Open (or create) the database and return a connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read performance
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create all tables and indexes if they do not already exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            sector      TEXT NOT NULL,
            universe    TEXT NOT NULL DEFAULT 'NIFTY50'
        );

        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol  TEXT    NOT NULL,
            date    TEXT    NOT NULL,   -- YYYY-MM-DD
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  INTEGER,
            PRIMARY KEY (symbol, date)
        );

        CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date
            ON ohlcv (symbol, date);

        CREATE TABLE IF NOT EXISTS indices_ohlcv (
            symbol  TEXT    NOT NULL,
            date    TEXT    NOT NULL,   -- YYYY-MM-DD
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL,
            volume  INTEGER,
            PRIMARY KEY (symbol, date)
        );

        CREATE INDEX IF NOT EXISTS idx_indices_symbol_date
            ON indices_ohlcv (symbol, date);
    """)
    conn.commit()
    conn.close()
