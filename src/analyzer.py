"""
Seasonal analysis engine.

Core question answered: "For a given stock (or sector), how has it historically
performed during a specific calendar window defined by a start date (month+day)
and a holding period in calendar days?"

Four main functions:
  - seasonal_analysis()      : year-by-year breakdown for one stock
  - sector_seasonal_analysis(): ranks all stocks in a sector by that window
  - best_windows_for_stock() : scans all 12 start months, finds the strongest windows
  - universe_screener()      : given a fixed window, ranks ALL stocks in the universe
"""

from __future__ import annotations

from calendar import monthrange
from datetime import timedelta

import numpy as np
import pandas as pd

from .db import get_connection
from .universe import get_sectors, get_stocks, get_symbol_to_name


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_closes(symbol: str, conn) -> pd.DataFrame:
    """Load the full OHLCV series for a symbol, indexed by date."""
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM ohlcv "
        "WHERE symbol = ? ORDER BY date",
        conn,
        params=(symbol,),
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def _load_index_closes(symbol: str, conn) -> pd.DataFrame:
    """Load OHLCV for a market index / macro ticker from indices_ohlcv, indexed by date."""
    try:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM indices_ohlcv "
            "WHERE symbol = ? ORDER BY date",
            conn,
            params=(symbol,),
        )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df
    except Exception:
        return pd.DataFrame()


# ─── Date helpers ─────────────────────────────────────────────────────────────

def _safe_timestamp(year: int, month: int, day: int) -> pd.Timestamp | None:
    """Return a Timestamp, clamping `day` to the last valid day of the month."""
    try:
        last_day = monthrange(year, month)[1]
        return pd.Timestamp(year, month, min(day, last_day))
    except ValueError:
        return None


def _month_abbr(month: int) -> str:
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month - 1]


def _safe_mae(series: "pd.Series", start_price: float, favorable: bool = False) -> float | None:
    """
    Compute MAE (max adverse excursion) or MFE (max favorable excursion) as a %.
    MAE uses the intraday low series; MFE uses the intraday high series.
    Returns None if the series is all-NaN or start_price is zero.
    """
    try:
        extreme = float(series.max()) if favorable else float(series.min())
        if start_price == 0 or np.isnan(extreme):
            return None
        return round((extreme - start_price) / start_price * 100, 2)
    except Exception:
        return None


# ─── Single-year window return ─────────────────────────────────────────────────

def _window_return_by_days(
    df: pd.DataFrame,
    year: int,
    start_month: int,
    start_day: int,
    holding_days: int,
) -> dict | None:
    """
    Compute the return for one calendar year.

    The window starts on (year, start_month, start_day) — or the nearest
    trading day on or after that date — and ends after `holding_days`
    calendar days.  Returns None if there are fewer than 5 trading days.
    """
    target_start = _safe_timestamp(year, start_month, start_day)
    if target_start is None:
        return None

    target_end = target_start + timedelta(days=holding_days)

    # Use the first available trading day on or after target_start
    candidates = df[df.index >= target_start]
    if candidates.empty:
        return None
    window = df[(df.index >= candidates.index[0]) & (df.index <= target_end)]

    if len(window) < 5:
        return None

    start_price = float(window["close"].iloc[0])
    end_price   = float(window["close"].iloc[-1])
    return_pct  = (end_price - start_price) / start_price * 100

    # Normalised series: start = 100 (used for the chart)
    norm_series = (window["close"] / start_price * 100).round(2)

    return {
        "year":        year,
        "start_date":  window.index[0].strftime("%Y-%m-%d"),
        "end_date":    window.index[-1].strftime("%Y-%m-%d"),
        "trading_days": len(window),
        "start_price": round(start_price, 2),
        "end_price":   round(end_price, 2),
        "return_pct":  round(return_pct, 2),
        "max_close":   round(float(window["close"].max()), 2),
        "min_close":   round(float(window["close"].min()), 2),
        "mae_pct":     _safe_mae(window["low"],  start_price),
        "mfe_pct":     _safe_mae(window["high"], start_price, favorable=True),
        "_norm":       norm_series,
    }


# ─── Main analysis functions ───────────────────────────────────────────────────

def seasonal_analysis(
    symbol: str,
    start_month: int,
    start_day: int,
    holding_days: int,
    min_return_pct: float = 0.0,
) -> tuple[pd.DataFrame | None, dict | None]:
    """
    Seasonal analysis for a single stock.

    Parameters
    ----------
    symbol          : NSE ticker (e.g. "RELIANCE.NS")
    start_month     : 1-12
    start_day       : 1-31
    holding_days    : number of calendar days to hold from entry
    min_return_pct  : target minimum return %; used to count "target met" years

    Returns
    -------
    results_df : DataFrame
        One row per year: year, start_date, end_date, trading_days,
        start_price, end_price, return_pct, max_close, min_close,
        direction, target_met.
    summary : dict
        Aggregate stats including target-aware counts and norm_series_map.
    """
    conn = get_connection()
    df = _load_closes(symbol, conn)
    conn.close()

    if df.empty:
        return None, None

    years = sorted(df.index.year.unique())
    raw_results = []

    for year in years:
        r = _window_return_by_days(df, year, start_month, start_day, holding_days)
        if r:
            raw_results.append(r)

    if len(raw_results) < 3:
        return None, None

    norm_series_map = {r["year"]: r.pop("_norm") for r in raw_results}

    results_df = pd.DataFrame(raw_results)
    results_df["direction"]  = results_df["return_pct"].apply(lambda x: "UP" if x > 0 else "DOWN")
    results_df["target_met"] = results_df["return_pct"] >= min_return_pct

    total       = len(results_df)
    target_met  = int(results_df["target_met"].sum())
    avg_ret     = round(float(results_df["return_pct"].mean()), 2)
    target_avg  = (
        round(float(results_df.loc[results_df["target_met"], "return_pct"].mean()), 2)
        if target_met > 0 else None
    )

    summary = {
        "symbol":                symbol,
        "total_instances":       total,
        "target_met_count":      target_met,
        "target_met_label":      f"{target_met} out of {total} years",
        "avg_return_pct":        avg_ret,
        "avg_return_when_met":   target_avg,
        "target_never_met":      target_met == 0 and min_return_pct > 0,
        "best_return_pct":       round(float(results_df["return_pct"].max()), 2),
        "worst_return_pct":      round(float(results_df["return_pct"].min()), 2),
        "best_year":             int(results_df.loc[results_df["return_pct"].idxmax(), "year"]),
        "worst_year":            int(results_df.loc[results_df["return_pct"].idxmin(), "year"]),
        "min_return_pct":        min_return_pct,
        "norm_series_map":       norm_series_map,
    }

    _mae_vals = results_df["mae_pct"].dropna()
    _mfe_vals = results_df["mfe_pct"].dropna()
    summary["avg_mae_pct"]   = round(float(_mae_vals.mean()), 2) if len(_mae_vals) > 0 else None
    summary["worst_mae_pct"] = round(float(_mae_vals.min()),  2) if len(_mae_vals) > 0 else None
    summary["avg_mfe_pct"]   = round(float(_mfe_vals.mean()), 2) if len(_mfe_vals) > 0 else None

    return results_df, summary


def sector_seasonal_analysis(
    sector: str,
    start_month: int,
    start_day: int,
    holding_days: int,
    min_return_pct: float = 0.0,
    universe: str = "NIFTY50",
) -> pd.DataFrame | None:
    """
    Run seasonal_analysis for every stock in the sector and return a
    summary DataFrame sorted by target_met_count descending.
    """
    sectors = get_sectors(universe)
    symbols = sectors.get(sector, [])
    if not symbols:
        return None

    rows = []
    for symbol in symbols:
        _, summary = seasonal_analysis(symbol, start_month, start_day, holding_days, min_return_pct)
        if summary:
            rows.append(
                {
                    "Symbol":              symbol,
                    "Target Met (yrs)":    summary["target_met_count"],
                    "Out of (yrs)":        summary["total_instances"],
                    "Avg Return %":        summary["avg_return_pct"],
                    "Avg When Target Met": summary["avg_return_when_met"],
                    "Best Return %":       summary["best_return_pct"],
                    "Worst Return %":      summary["worst_return_pct"],
                }
            )

    if not rows:
        return None

    return (
        pd.DataFrame(rows)
        .sort_values("Target Met (yrs)", ascending=False)
        .reset_index(drop=True)
    )


def best_windows_for_stock(
    symbol: str,
    holding_days: int = 90,
    min_return_pct: float = 0.0,
) -> pd.DataFrame | None:
    """
    Scan all 12 possible start months (day 1) for a stock and rank windows
    by target_met_count descending.
    """
    conn = get_connection()
    df = _load_closes(symbol, conn)
    conn.close()

    if df.empty:
        return None

    years = sorted(df.index.year.unique())
    rows = []

    for start_month in range(1, 13):
        window_returns = []
        for year in years:
            r = _window_return_by_days(df, year, start_month, 1, holding_days)
            if r:
                window_returns.append(r["return_pct"])

        if len(window_returns) < 5:
            continue

        target_met = sum(1 for x in window_returns if x >= min_return_pct)
        rows.append(
            {
                "Window":           f"{_month_abbr(start_month)} 1  (+{holding_days}d)",
                "Start Month":      start_month,
                "Target Met (yrs)": target_met,
                "Out of (yrs)":     len(window_returns),
                "Avg Return %":     round(float(np.mean(window_returns)), 2),
                "Best Return %":    round(float(np.max(window_returns)), 2),
                "Worst Return %":   round(float(np.min(window_returns)), 2),
            }
        )

    if not rows:
        return None

    return (
        pd.DataFrame(rows)
        .sort_values("Target Met (yrs)", ascending=False)
        .reset_index(drop=True)
    )


def universe_screener(
    start_month: int,
    start_day: int,
    holding_days: int,
    min_return_pct: float = 0.0,
    universe: str = "NIFTY50",
) -> pd.DataFrame | None:
    """
    Given a fixed window (start month+day, holding period, return target),
    run seasonal_analysis for every stock in the universe and return a
    ranked DataFrame — best performers first, sorted by target_met_count
    then avg_return descending.

    Columns: Symbol, Name, Sector, Target Met (yrs), Out of (yrs),
             Avg Return %, Avg When Target Met, Best Return %, Worst Return %.
    """
    stocks     = get_stocks(universe)
    name_map   = get_symbol_to_name(universe)
    sector_map = {s["symbol"]: s["sector"] for s in stocks}

    rows = []
    for stock in stocks:
        symbol = stock["symbol"]
        _, summary = seasonal_analysis(
            symbol, start_month, start_day, holding_days, min_return_pct
        )
        if summary:
            rows.append(
                {
                    "Symbol":              symbol,
                    "Name":                name_map.get(symbol, symbol),
                    "Sector":              sector_map.get(symbol, ""),
                    "Target Met (yrs)":    summary["target_met_count"],
                    "Out of (yrs)":        summary["total_instances"],
                    "Avg Return %":        summary["avg_return_pct"],
                    "Avg When Target Met": summary["avg_return_when_met"],
                    "Best Return %":       summary["best_return_pct"],
                    "Worst Return %":      summary["worst_return_pct"],
                }
            )

    if not rows:
        return None

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Target Met (yrs)", "Avg Return %"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


# ─── New insight functions ─────────────────────────────────────────────────────

def monthly_return_heatmap(symbol: str) -> pd.DataFrame | None:
    """
    Return a year × month pivot of monthly returns for a stock.

    Each cell = (last_close - first_close) / first_close * 100 for that month.
    Columns: Jan..Dec.  Index: calendar year integers.
    Returns None if no data or fewer than 3 rows.
    """
    conn = get_connection()
    df = _load_closes(symbol, conn)
    conn.close()

    if df.empty:
        return None

    records = []
    for year in sorted(df.index.year.unique()):
        for month in range(1, 13):
            month_data = df[(df.index.year == year) & (df.index.month == month)]
            if len(month_data) < 3:
                continue
            start_p = float(month_data["close"].iloc[0])
            end_p   = float(month_data["close"].iloc[-1])
            if start_p > 0:
                records.append({
                    "year":       year,
                    "month":      month,
                    "return_pct": round((end_p - start_p) / start_p * 100, 2),
                })

    if not records:
        return None

    df_long = pd.DataFrame(records)
    pivot = df_long.pivot(index="year", columns="month", values="return_pct")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Only rename columns that exist in the pivot (some months may be missing for new data)
    pivot.columns = [month_names[m - 1] for m in pivot.columns]
    return pivot


def excess_return_vs_nifty(
    symbol: str,
    start_month: int,
    start_day: int,
    holding_days: int,
    min_return_pct: float = 0.0,
) -> tuple[pd.DataFrame | None, dict | None]:
    """
    Compare a stock's seasonal window return against NIFTY 50's return over the
    identical date window each year.  Also attaches India VIX level at entry.

    Returns
    -------
    results_df : columns — year, stock_return, nifty_return, excess_return,
                           beat_index, vix_at_entry, target_met
    summary    : aggregate stats; includes nifty_available / vix_available flags.
    """
    conn = get_connection()
    stock_df = _load_closes(symbol, conn)
    nifty_df = _load_index_closes("^NSEI", conn)
    vix_df   = _load_index_closes("^INDIAVIX", conn)
    conn.close()

    if stock_df.empty:
        return None, None

    nifty_available = not nifty_df.empty
    vix_available   = not vix_df.empty

    years = sorted(stock_df.index.year.unique())
    rows = []

    for year in years:
        stock_r = _window_return_by_days(stock_df, year, start_month, start_day, holding_days)
        if stock_r is None:
            continue

        nifty_ret = None
        if nifty_available:
            nifty_r = _window_return_by_days(nifty_df, year, start_month, start_day, holding_days)
            if nifty_r:
                nifty_ret = nifty_r["return_pct"]

        vix_val = None
        if vix_available:
            target_start = _safe_timestamp(year, start_month, start_day)
            if target_start is not None:
                candidates = vix_df[vix_df.index >= target_start]
                if not candidates.empty:
                    vix_val = round(float(candidates["close"].iloc[0]), 2)

        excess = round(stock_r["return_pct"] - nifty_ret, 2) if nifty_ret is not None else None

        rows.append({
            "year":          year,
            "stock_return":  stock_r["return_pct"],
            "nifty_return":  nifty_ret,
            "excess_return": excess,
            "beat_index":    (excess > 0) if excess is not None else None,
            "vix_at_entry":  vix_val,
            "target_met":    stock_r["return_pct"] >= min_return_pct,
        })

    if len(rows) < 3:
        return None, None

    results_df = pd.DataFrame(rows)

    valid_excess  = results_df["excess_return"].dropna()
    valid_nifty   = results_df["nifty_return"].dropna()
    beat_count    = int(results_df["beat_index"].dropna().sum()) if nifty_available else None
    beat_total    = int(results_df["beat_index"].notna().sum())  if nifty_available else None

    summary = {
        "nifty_available":   nifty_available,
        "vix_available":     vix_available,
        "total_years":       len(results_df),
        "avg_stock_return":  round(float(results_df["stock_return"].mean()), 2),
        "avg_nifty_return":  round(float(valid_nifty.mean()), 2) if len(valid_nifty) > 0 else None,
        "avg_excess_return": round(float(valid_excess.mean()), 2) if len(valid_excess) > 0 else None,
        "beat_index_count":  beat_count,
        "beat_index_label":  (
            f"{beat_count} of {beat_total} years"
            if beat_count is not None else "N/A"
        ),
    }

    return results_df, summary


def sector_rotation_analysis(universe: str = "NIFTY50") -> pd.DataFrame | None:
    """
    For each sector, compute the average stock return in every calendar month
    (across all constituent stocks and all years in the DB).

    Returns a pivot DataFrame — rows = sectors, columns = Jan..Dec.
    Each cell = average monthly return %.  Returns None if no data.
    """
    sectors = get_sectors(universe)
    conn = get_connection()

    sector_data: dict[str, dict[int, list[float]]] = {}

    for sector, symbols in sectors.items():
        monthly: dict[int, list[float]] = {m: [] for m in range(1, 13)}
        for symbol in symbols:
            df = _load_closes(symbol, conn)
            if df.empty:
                continue
            for year in df.index.year.unique():
                for month in range(1, 13):
                    m_data = df[(df.index.year == year) & (df.index.month == month)]
                    if len(m_data) < 3:
                        continue
                    sp = float(m_data["close"].iloc[0])
                    ep = float(m_data["close"].iloc[-1])
                    if sp > 0:
                        monthly[month].append((ep - sp) / sp * 100)
        sector_data[sector] = monthly

    conn.close()

    if not sector_data:
        return None

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for sector, monthly in sector_data.items():
        row = {"Sector": sector}
        for i, name in enumerate(month_names, 1):
            vals = monthly.get(i, [])
            row[name] = round(float(np.mean(vals)), 2) if vals else None
        rows.append(row)

    result = pd.DataFrame(rows).set_index("Sector")
    return result


def mae_analysis(
    symbol: str,
    start_month: int,
    start_day: int,
    holding_days: int,
) -> pd.DataFrame | None:
    """
    Per-year Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).

    MAE = deepest intraday low vs entry price (negative % = how far it dropped).
    MFE = highest intraday high vs entry price (positive % = best peak reached).

    Useful for stop-loss calibration: if your stop is tighter than the typical
    MAE, you'll get stopped out even on winning trades.

    Returns a DataFrame with one row per year.
    """
    conn = get_connection()
    df = _load_closes(symbol, conn)
    conn.close()

    if df.empty:
        return None

    years = sorted(df.index.year.unique())
    rows = []

    for year in years:
        r = _window_return_by_days(df, year, start_month, start_day, holding_days)
        if r is None:
            continue
        mae = r.get("mae_pct")
        mfe = r.get("mfe_pct")
        ret = r["return_pct"]
        rr  = round(ret / abs(mae), 2) if (mae is not None and mae < 0) else None
        rows.append({
            "Year":                        r["year"],
            "Final Return %":              ret,
            "MAE % (worst intraday dip)":  mae,
            "MFE % (best intraday peak)":  mfe,
            "Risk/Reward (Final÷|MAE|)":   rr,
            "Profitable":                  ret > 0,
        })

    if not rows:
        return None

    return pd.DataFrame(rows)
