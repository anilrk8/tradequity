"""
Data fetcher — downloads OHLCV from Yahoo Finance (yfinance) into SQLite.

Key design decisions:
- Incremental by default: checks last stored date per symbol, only fetches new days.
- Bulk download on first run (~5-10 min for Nifty 50, 15 years).
- Daily updates are just another incremental run — takes seconds.
- Batch delays to respect Yahoo Finance rate limits.
"""

import logging
import time
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd
import yfinance as yf

from .db import get_connection, init_db
from .universe import get_stocks

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"   # Pull ~15 years of history
BATCH_SIZE = 10             # Number of stocks per batch before pausing
BATCH_DELAY_SECS = 2        # Pause between batches (avoid rate limiting)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _last_stored_date(symbol: str, conn) -> str | None:
    """Return the most recent date stored for a symbol, or None if no data."""
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM ohlcv WHERE symbol = ?", (symbol,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def _upsert_rows(df: pd.DataFrame, symbol: str, conn) -> None:
    """Write OHLCV rows into the DB, replacing any existing rows for the same date."""
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT OR REPLACE INTO ohlcv (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                row["date"],
                round(float(row["open"]),   4) if pd.notna(row["open"])   else None,
                round(float(row["high"]),   4) if pd.notna(row["high"])   else None,
                round(float(row["low"]),    4) if pd.notna(row["low"])    else None,
                round(float(row["close"]),  4) if pd.notna(row["close"])  else None,
                int(row["volume"])               if pd.notna(row["volume"]) else None,
            ),
        )
    conn.commit()


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_symbol(symbol: str, start: str, end: str, conn) -> int:
    """
    Fetch OHLCV for one symbol from `start` to `end` and write to DB.
    Returns the number of rows inserted (0 on error or no new data).
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True, timeout=30)

        if df.empty:
            logger.warning("No data returned for %s", symbol)
            return 0

        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        _upsert_rows(df, symbol, conn)
        logger.info("%s: %d rows stored", symbol, len(df))
        return len(df)

    except Exception as exc:
        logger.error("Failed to fetch %s: %s", symbol, exc)
        return 0


def bulk_download(
    universe: str = "NIFTY50",
    progress_callback: Callable[[int, int, str, int, str], None] | None = None,
) -> None:
    """
    Download (or incrementally update) all stocks in the universe.

    On first run: fetches START_DATE → today (~15 years).
    On subsequent runs: fetches only the days missing since the last stored date.

    progress_callback(done, total, symbol, rows_fetched, status) is called
    after each stock, where status is "fetched" or "already_up_to_date".
    """
    init_db()
    stocks = get_stocks(universe)
    total = len(stocks)
    today = date.today().strftime("%Y-%m-%d")

    conn = get_connection()

    # Upsert stock metadata so the stocks table stays in sync
    cur = conn.cursor()
    for stock in stocks:
        cur.execute(
            """
            INSERT OR REPLACE INTO stocks (symbol, name, sector, universe)
            VALUES (?, ?, ?, ?)
            """,
            (stock["symbol"], stock["name"], stock["sector"], universe),
        )
    conn.commit()

    for i, stock in enumerate(stocks):
        symbol = stock["symbol"]
        last_date = _last_stored_date(symbol, conn)

        if last_date:
            next_day = (
                datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if next_day >= today:
                if progress_callback:
                    progress_callback(i + 1, total, symbol, 0, "already_up_to_date")
                continue
            start = next_day
        else:
            start = START_DATE

        rows = fetch_symbol(symbol, start, today, conn)

        if progress_callback:
            progress_callback(i + 1, total, symbol, rows, "fetched")

        # Pause between batches to respect Yahoo Finance rate limits
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY_SECS)

    conn.close()


# daily_update is an alias — the logic is identical to bulk_download
daily_update = bulk_download


def fetch_custom_ticker(symbol: str, name: str, sector: str = "Custom") -> tuple[int, str | None]:
    """
    Register and download full OHLCV history for an arbitrary Yahoo Finance ticker.

    Steps:
      1. Register the symbol in the `stocks` table (universe='CUSTOM').
      2. Incrementally download OHLCV from START_DATE (or last stored date) to today.
      3. If the download returns 0 rows (ticker doesn't exist / typo), unregister it
         and return an error.

    Note: No separate probe call is made — previously a 5-day probe was used for
    validation but it caused yfinance to cache the short response, resulting in
    the full history fetch also returning only a handful of rows.

    Returns:
        (rows_fetched, error_message)  — error_message is None on success.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return 0, "Ticker symbol cannot be empty."

    conn = get_connection()
    try:
        # Step 1: register (tentative — removed on failure)
        friendly_name = name.strip() if name.strip() else symbol
        sector_clean  = sector.strip() if sector.strip() else "Custom"
        conn.execute(
            "INSERT OR REPLACE INTO stocks (symbol, name, sector, universe) "
            "VALUES (?, ?, ?, ?)",
            (symbol, friendly_name, sector_clean, "CUSTOM"),
        )
        conn.commit()

        # Step 2: incremental download
        last = _last_stored_date(symbol, conn)
        if last:
            start = (
                datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            start = START_DATE
        end = date.today().strftime("%Y-%m-%d")

        if start > end:
            return 0, None  # already up to date

        rows = fetch_symbol(symbol, start, end, conn)

        # Step 3: validate — if no data came back the ticker is likely wrong
        if rows == 0:
            conn.execute(
                "DELETE FROM stocks WHERE symbol = ? AND universe = 'CUSTOM'",
                (symbol,),
            )
            conn.commit()
            return 0, (
                f"No data found for '{symbol}' on Yahoo Finance. "
                "Check spelling — NSE stocks should end with .NS (e.g. TATAPOWER.NS)."
            )

        return rows, None

    except Exception as exc:
        logger.error("fetch_custom_ticker(%s): %s", symbol, exc)
        return 0, str(exc)
    finally:
        conn.close()


def get_custom_tickers() -> list[dict]:
    """Return all custom tickers registered in the stocks table (universe='CUSTOM')."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, name, sector FROM stocks WHERE universe = 'CUSTOM' ORDER BY name"
        )
        return [{"symbol": r[0], "name": r[1], "sector": r[2] or "Custom"} for r in cur.fetchall()]
    finally:
        conn.close()


def update_custom_tickers(
    progress_callback: Callable[[int, int, str, int, str], None] | None = None,
) -> None:
    """
    Incrementally update all custom tickers to today.
    Same logic as bulk_download but only for universe='CUSTOM' stocks.
    progress_callback(done, total, symbol, rows_fetched, status)
    """
    custom = get_custom_tickers()
    if not custom:
        return

    conn = get_connection()
    total = len(custom)
    end = date.today().strftime("%Y-%m-%d")

    for i, entry in enumerate(custom):
        symbol = entry["symbol"]
        last = _last_stored_date(symbol, conn)
        if last:
            start = (
                datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            start = START_DATE

        if start > end:
            rows, status = 0, "already_up_to_date"
        else:
            rows = fetch_symbol(symbol, start, end, conn)
            status = "fetched"

        if progress_callback:
            progress_callback(i + 1, total, symbol, rows, status)

    conn.close()


def get_data_status(universe: str = "NIFTY50") -> pd.DataFrame:
    """Return a summary DataFrame showing data availability for every stock."""
    conn = get_connection()
    stocks = get_stocks(universe)
    rows = []

    for stock in stocks:
        symbol = stock["symbol"]
        cur = conn.cursor()
        cur.execute(
            """
            SELECT MIN(date) AS first_date,
                   MAX(date) AS last_date,
                   COUNT(*)  AS trading_days
            FROM ohlcv
            WHERE symbol = ?
            """,
            (symbol,),
        )
        r = cur.fetchone()
        rows.append(
            {
                "Symbol":       symbol,
                "Name":         stock["name"],
                "Sector":       stock["sector"],
                "First Date":   r["first_date"]    or "—",
                "Last Date":    r["last_date"]     or "—",
                "Trading Days": r["trading_days"],
            }
        )

    conn.close()
    return pd.DataFrame(rows)


def has_any_data() -> bool:
    """Quick check — returns True if the ohlcv table has at least one row."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM ohlcv LIMIT 1")
        result = cur.fetchone() is not None
        conn.close()
        return result
    except Exception:
        return False


# ─── Market indices and macro data ────────────────────────────────────────────

INDEX_TICKERS: dict[str, str] = {
    "^NSEI":     "NIFTY 50 Index",
    "^NSEBANK":  "NIFTY Bank Index",
    "^INDIAVIX": "India VIX",
    "USDINR=X":  "USD/INR",
}


def _last_stored_index_date(symbol: str, conn) -> str | None:
    """Return the most recent date stored for an index symbol, or None."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM indices_ohlcv WHERE symbol = ?", (symbol,))
        row = cur.fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def fetch_indices(
    progress_callback: Callable[[int, int, str, int, str], None] | None = None,
) -> None:
    """
    Download (or incrementally update) NIFTY 50 index, NIFTY Bank, India VIX, and USD/INR.
    Stores into the 'indices_ohlcv' table.  Same incremental logic as bulk_download.
    """
    init_db()
    conn = get_connection()
    symbols = list(INDEX_TICKERS.keys())
    total = len(symbols)
    today = date.today().strftime("%Y-%m-%d")

    for i, symbol in enumerate(symbols):
        last_date = _last_stored_index_date(symbol, conn)

        if last_date:
            next_day = (
                datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if next_day >= today:
                if progress_callback:
                    progress_callback(i + 1, total, symbol, 0, "already_up_to_date")
                continue
            start = next_day
        else:
            start = START_DATE

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=today, auto_adjust=True, timeout=30)

            if df.empty:
                logger.warning("No data returned for index %s", symbol)
                if progress_callback:
                    progress_callback(i + 1, total, symbol, 0, "no_data")
                continue

            df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            cur = conn.cursor()
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT OR REPLACE INTO indices_ohlcv
                        (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        row["date"],
                        round(float(row["open"]),  4) if pd.notna(row["open"])  else None,
                        round(float(row["high"]),  4) if pd.notna(row["high"])  else None,
                        round(float(row["low"]),   4) if pd.notna(row["low"])   else None,
                        round(float(row["close"]), 4) if pd.notna(row["close"]) else None,
                        int(row["volume"])              if pd.notna(row["volume"]) else None,
                    ),
                )
            conn.commit()
            logger.info("%s (index): %d rows stored", symbol, len(df))

            if progress_callback:
                progress_callback(i + 1, total, symbol, len(df), "fetched")

        except Exception as exc:
            logger.error("Failed to fetch index %s: %s", symbol, exc)
            if progress_callback:
                progress_callback(i + 1, total, symbol, 0, "error")

    conn.close()


def has_index_data() -> bool:
    """Return True if any index data exists in the indices_ohlcv table."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM indices_ohlcv LIMIT 1")
        result = cur.fetchone() is not None
        conn.close()
        return result
    except Exception:
        return False


def get_index_status() -> pd.DataFrame:
    """Return a summary DataFrame showing data availability for each index ticker."""
    try:
        conn = get_connection()
        rows = []
        for symbol, name in INDEX_TICKERS.items():
            cur = conn.cursor()
            cur.execute(
                """
                SELECT MIN(date) AS first_date,
                       MAX(date) AS last_date,
                       COUNT(*)  AS trading_days
                FROM indices_ohlcv
                WHERE symbol = ?
                """,
                (symbol,),
            )
            r = cur.fetchone()
            rows.append({
                "Symbol":       symbol,
                "Name":         name,
                "First Date":   r["first_date"]   or "—",
                "Last Date":    r["last_date"]     or "—",
                "Trading Days": r["trading_days"],
            })
        conn.close()
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()
