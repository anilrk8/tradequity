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

    if len(raw_results) < 2:
        return None, {"error": "insufficient_years", "years_found": len(raw_results)}

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
        "low_sample_warning":    total < 5,
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
        if summary and "error" not in summary:
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
        if summary and "error" not in summary:
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
        rows.append({
            "Year":                        r["year"],
            "Final Return %":              ret,
            "MAE % (worst intraday dip)":  mae,
            "MFE % (best intraday peak)":  mfe,
            "Profitable":                  ret > 0,
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def stop_loss_survival(mae_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each candidate stop-loss level (0% to -30% in 0.5% steps), simulate
    how many trades would have been stopped out and whether they were winners
    or losers.

    A trade is "stopped out" if its MAE went deeper than the stop level.

    Returns a DataFrame with columns:
        Stop Level %  — the stop being tested (e.g. -5.0)
        Trades Stopped — total trades stopped out at this level
        Winners Stopped — profitable trades that would have been cut early
        Losers Stopped  — losing trades correctly cut
        Winners Survived — profitable trades that weathered the drawdown
        Total Trades    — total trades analysed
        Survival Rate % — % of all trades that were NOT stopped out
        Winner Preservation % — % of profitable trades that survived
    """
    total = len(mae_df)
    profitable_mask = mae_df["Profitable"].fillna(False)
    mae_vals = mae_df["MAE % (worst intraday dip)"].fillna(0)

    rows = []
    for stop in [round(-x * 0.5, 1) for x in range(1, 61)]:  # -0.5 to -30.0
        stopped      = mae_vals < stop          # MAE went below this stop
        survived     = ~stopped

        winners_stopped  = int((stopped & profitable_mask).sum())
        losers_stopped   = int((stopped & ~profitable_mask).sum())
        winners_survived = int((survived & profitable_mask).sum())
        total_stopped    = int(stopped.sum())
        total_winners    = int(profitable_mask.sum())

        rows.append({
            "Stop Level %":          stop,
            "Trades Stopped":        total_stopped,
            "Winners Stopped":       winners_stopped,
            "Losers Stopped":        losers_stopped,
            "Winners Survived":      winners_survived,
            "Total Trades":          total,
            "Survival Rate %":       round((total - total_stopped) / total * 100, 1),
            "Winner Preservation %": round(winners_survived / total_winners * 100, 1) if total_winners > 0 else None,
        })

    return pd.DataFrame(rows)


# ─── Similar Year Finder ───────────────────────────────────────────────────────

def _compute_entry_features(
    stock_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    entry_date: pd.Timestamp,
) -> dict | None:
    """
    Compute five market-condition features at a given entry date.

    Features
    --------
    vix_level          : India VIX closing value on entry day
    nifty_mom_20d      : NIFTY 50 return over the 20 trading days before entry (%)
    nifty_200dma_dist  : NIFTY 50 close vs its 200-day SMA at entry (%)
    stock_rsi14        : Stock RSI-14 at entry
    stock_200dma_dist  : Stock close vs its 200-day SMA at entry (%)
    """
    def _nearest(df: pd.DataFrame, dt: pd.Timestamp) -> pd.Timestamp | None:
        prior = df[df.index <= dt]
        return prior.index[-1] if not prior.empty else None

    # ── VIX ──────────────────────────────────────────────────────────────────
    vix_date = _nearest(vix_df, entry_date)
    if vix_date is None:
        return None
    vix_level = float(vix_df.loc[vix_date, "close"])

    # ── NIFTY 20-day momentum & 200-DMA distance ──────────────────────────────
    nifty_date = _nearest(nifty_df, entry_date)
    if nifty_date is None:
        return None
    nifty_pos = nifty_df.index.get_loc(nifty_date)
    if nifty_pos < 20:
        return None
    nifty_close = float(nifty_df["close"].iloc[nifty_pos])
    nifty_close_20ago = float(nifty_df["close"].iloc[nifty_pos - 20])
    nifty_mom_20d = (nifty_close - nifty_close_20ago) / nifty_close_20ago * 100
    nifty_200dma_dist = None
    if nifty_pos >= 200:
        nifty_200dma = float(nifty_df["close"].iloc[nifty_pos - 199 : nifty_pos + 1].mean())
        nifty_200dma_dist = (nifty_close - nifty_200dma) / nifty_200dma * 100

    # ── Stock RSI-14 & 200-DMA distance ──────────────────────────────────────
    stock_date = _nearest(stock_df, entry_date)
    if stock_date is None:
        return None
    stock_pos = stock_df.index.get_loc(stock_date)
    if stock_pos < 14:
        return None
    stock_close = float(stock_df["close"].iloc[stock_pos])

    deltas = stock_df["close"].iloc[stock_pos - 13 : stock_pos + 1].diff().dropna()
    gain = deltas.clip(lower=0).mean()
    loss = (-deltas.clip(upper=0)).mean()
    stock_rsi14 = 100.0 if loss == 0 else round(100 - 100 / (1 + gain / loss), 2)

    stock_200dma_dist = None
    if stock_pos >= 200:
        stock_200dma = float(stock_df["close"].iloc[stock_pos - 199 : stock_pos + 1].mean())
        stock_200dma_dist = (stock_close - stock_200dma) / stock_200dma * 100

    return {
        "vix_level":         round(vix_level, 2),
        "nifty_mom_20d":     round(nifty_mom_20d, 2),
        "nifty_200dma_dist": round(nifty_200dma_dist, 2) if nifty_200dma_dist is not None else None,
        "stock_rsi14":       stock_rsi14,
        "stock_200dma_dist": round(stock_200dma_dist, 2) if stock_200dma_dist is not None else None,
    }


_FEATURE_KEYS = [
    ("vix_level",         "VIX at Entry"),
    ("nifty_mom_20d",     "NIFTY 20d Mom %"),
    ("nifty_200dma_dist", "NIFTY vs 200DMA %"),
    ("stock_rsi14",       "Stock RSI-14"),
    ("stock_200dma_dist", "Stock vs 200DMA %"),
]
_FEATURE_COLS = [col for _, col in _FEATURE_KEYS]


def similar_years_analysis(
    symbol: str,
    start_month: int,
    start_day: int,
    holding_days: int,
    n_similar: int = 5,
) -> dict | None:
    """
    Find the N historically most similar years to today's market conditions
    for the given stock + seasonal window.

    Similarity is measured as normalised Euclidean distance across five features:
    VIX level, NIFTY 20-day momentum, NIFTY vs 200-DMA, Stock RSI-14, Stock vs 200-DMA.

    Returns a dict with keys:
        today_features  : dict of the 5 features for today (or None if unavailable)
        today_entry     : date string of the reference entry date used for today
        all_years       : DataFrame — every year with features + trade outcome
        similar_years   : DataFrame — top N most similar years, sorted by distance
        missing_indices : bool — True if VIX / NIFTY data is absent from DB
    """
    conn = get_connection()
    stock_df = _load_closes(symbol, conn)
    nifty_df = _load_index_closes("^NSEI", conn)
    vix_df   = _load_index_closes("^INDIAVIX", conn)
    conn.close()

    if stock_df.empty:
        return None

    missing_indices = nifty_df.empty or vix_df.empty

    years = sorted(stock_df.index.year.unique())
    rows = []

    for year in years:
        r = _window_return_by_days(stock_df, year, start_month, start_day, holding_days)
        if r is None:
            continue
        entry_date = pd.Timestamp(r["start_date"])
        feats = (
            _compute_entry_features(stock_df, nifty_df, vix_df, entry_date)
            if not missing_indices else None
        )
        row = {
            "Year":           year,
            "Entry Date":     r["start_date"],
            "Final Return %": r["return_pct"],
            "Profitable":     r["return_pct"] > 0,
        }
        if feats:
            for key, col in _FEATURE_KEYS:
                row[col] = feats.get(key)
        rows.append(row)

    if not rows:
        return None

    all_years_df = pd.DataFrame(rows)

    if missing_indices:
        return {
            "today_features":  None,
            "today_entry":     None,
            "all_years":       all_years_df,
            "similar_years":   None,
            "missing_indices": True,
        }

    # ── Compute "today" reference entry date ──────────────────────────────────
    current_year = pd.Timestamp.today().year
    target_entry = _safe_timestamp(current_year, start_month, start_day)
    if target_entry is not None:
        candidates = stock_df[stock_df.index >= target_entry]
        today_entry = candidates.index[0] if not candidates.empty else stock_df.index[-1]
    else:
        today_entry = stock_df.index[-1]

    today_feats = _compute_entry_features(stock_df, nifty_df, vix_df, today_entry)
    if today_feats is None:
        return {
            "today_features":  None,
            "today_entry":     today_entry.strftime("%Y-%m-%d"),
            "all_years":       all_years_df,
            "similar_years":   None,
            "missing_indices": False,
        }

    # ── Similarity: normalised Euclidean distance ─────────────────────────────
    feat_df = all_years_df.dropna(subset=_FEATURE_COLS).copy()
    if feat_df.empty:
        return {
            "today_features":  today_feats,
            "today_entry":     today_entry.strftime("%Y-%m-%d"),
            "all_years":       all_years_df,
            "similar_years":   None,
            "missing_indices": False,
        }

    means = feat_df[_FEATURE_COLS].mean()
    stds  = feat_df[_FEATURE_COLS].std().replace(0, 1)

    today_vec  = np.array([today_feats.get(k, 0) or 0 for k, _ in _FEATURE_KEYS], dtype=float)
    today_norm = (today_vec - means.values) / stds.values
    hist_norm  = (feat_df[_FEATURE_COLS].values - means.values) / stds.values
    distances  = np.sqrt(((hist_norm - today_norm) ** 2).sum(axis=1))

    feat_df["Similarity Distance"] = distances
    d_min, d_max = distances.min(), distances.max()
    feat_df["Similarity Score"]    = (
        100 - ((distances - d_min) / (d_max - d_min + 1e-9) * 100)
    ).round(1)

    similar = (
        feat_df.sort_values("Similarity Distance")
        .head(n_similar)
        .reset_index(drop=True)
    )

    return {
        "today_features":  today_feats,
        "today_entry":     today_entry.strftime("%Y-%m-%d"),
        "all_years":       all_years_df,
        "similar_years":   similar,
        "missing_indices": False,
    }


# ─── Volume Analysis ───────────────────────────────────────────────────────────

def volume_analysis(
    symbol: str,
    start_month: int,
    start_day: int,
    holding_days: int,
) -> dict | None:
    """
    Volume-based analytics for a stock across its full history.

    Returns a dict with four DataFrames / Series:

    monthly_avg_vol : dict {month_name -> avg_normalised_volume}
        Normalised = each day's volume / trailing 90-day avg volume.
        Gives the seasonal calendar rhythm of trading activity.

    window_rows : DataFrame  (one row per historical year in the entry window)
        Year, return_pct, entry_vol_ratio (entry day vol / 20d avg),
        window_avg_vol_ratio (mean vol in window / 90d baseline),
        direction, volume_confirmed (True if entry vol > 1.0)

    obv_by_year : dict {year -> list of cumulative OBV values normalised to 0 at entry}
        On-Balance Volume within each year's window.

    monthly_vol_df : DataFrame  (month x avg_normalised_volume, for bar chart)
    """
    conn = get_connection()
    df = _load_closes(symbol, conn)
    conn.close()

    if df.empty or "volume" not in df.columns:
        return None

    # Drop rows with null volume
    df = df.dropna(subset=["volume"])
    if len(df) < 100:
        return None

    # ── 1. Seasonal volume rhythm: normalised vol by calendar month ────────────
    # For each trading day compute vol / trailing 90-day avg vol (excluding same day)
    df = df.copy()
    df["vol_90d_avg"] = (
        df["volume"].rolling(window=90, min_periods=20).mean().shift(1)
    )
    df["vol_norm"] = df["volume"] / df["vol_90d_avg"]

    monthly_vol = {}
    for m in range(1, 13):
        vals = df.loc[df.index.month == m, "vol_norm"].dropna()
        monthly_vol[_month_abbr(m)] = round(float(vals.mean()), 3) if len(vals) > 0 else None

    monthly_vol_df = pd.DataFrame(
        [{"Month": k, "Avg Normalised Volume": v} for k, v in monthly_vol.items()]
    )

    # ── 2. Window-level volume stats per year ─────────────────────────────────
    # 20-day rolling avg for entry-day confirmation
    df["vol_20d_avg"] = (
        df["volume"].rolling(window=20, min_periods=5).mean().shift(1)
    )

    years = sorted(df.index.year.unique())
    window_rows = []
    obv_by_year = {}

    for year in years:
        r = _window_return_by_days(df, year, start_month, start_day, holding_days)
        if r is None:
            continue

        entry_date = pd.Timestamp(r["start_date"])
        end_date   = pd.Timestamp(r["end_date"])
        window_df  = df[(df.index >= entry_date) & (df.index <= end_date)].copy()

        if len(window_df) < 5:
            continue

        # Entry-day volume confirmation
        entry_row = df[df.index == entry_date]
        if entry_row.empty:
            # snap to first available day (matching _window_return_by_days logic)
            entry_row = df[df.index >= entry_date].head(1)

        entry_vol       = float(entry_row["volume"].iloc[0]) if not entry_row.empty else np.nan
        entry_20d_avg   = float(entry_row["vol_20d_avg"].iloc[0]) if not entry_row.empty else np.nan
        entry_vol_ratio = round(entry_vol / entry_20d_avg, 3) if entry_20d_avg > 0 else None

        # Window average vol vs 90-day baseline (baseline = avg of vol_90d_avg within window)
        baseline        = window_df["vol_90d_avg"].dropna()
        window_avg_vol  = float(window_df["volume"].mean())
        baseline_avg    = float(baseline.mean()) if not baseline.empty else np.nan
        window_vol_ratio = round(window_avg_vol / baseline_avg, 3) if baseline_avg > 0 else None

        ret = r["return_pct"]

        window_rows.append({
            "Year":                   year,
            "Return %":               ret,
            "Direction":              "UP" if ret > 0 else "DOWN",
            "Entry Vol Ratio":        entry_vol_ratio,   # >1 = high vol entry
            "Window Avg Vol Ratio":   window_vol_ratio,  # >1 = accumulation
            "Vol Confirmed Entry":    (entry_vol_ratio or 0) >= 1.0,
            "Accumulation Window":    (window_vol_ratio or 0) >= 1.0,
            "Vol-Price Divergence":   ret > 0 and (window_vol_ratio or 1.0) < 0.85,
        })

        # ── OBV within window ─────────────────────────────────────────────────
        obv = [0.0]
        prev_close = float(window_df["close"].iloc[0])
        for _, row in window_df.iloc[1:].iterrows():
            curr_close = float(row["close"])
            vol        = float(row["volume"])
            if curr_close > prev_close:
                obv.append(obv[-1] + vol)
            elif curr_close < prev_close:
                obv.append(obv[-1] - vol)
            else:
                obv.append(obv[-1])
            prev_close = curr_close

        # Normalise OBV to % of avg daily volume so different stocks are comparable
        avg_vol = float(window_df["volume"].mean())
        if avg_vol > 0:
            obv_norm = [round(v / avg_vol, 3) for v in obv]
        else:
            obv_norm = obv
        obv_by_year[year] = obv_norm

    if not window_rows:
        return None

    window_df_out = pd.DataFrame(window_rows)

    # ── 3. Summary signals ────────────────────────────────────────────────────
    n = len(window_df_out)
    confirmed        = int(window_df_out["Vol Confirmed Entry"].sum())
    confirmed_up     = int((window_df_out["Vol Confirmed Entry"] & (window_df_out["Direction"] == "UP")).sum())
    unconfirmed_up   = int((~window_df_out["Vol Confirmed Entry"] & (window_df_out["Direction"] == "UP")).sum())
    divergence_count = int(window_df_out["Vol-Price Divergence"].sum())
    accumulation     = int(window_df_out["Accumulation Window"].sum())

    avg_ret_confirmed   = window_df_out.loc[window_df_out["Vol Confirmed Entry"],  "Return %"].mean()
    avg_ret_unconfirmed = window_df_out.loc[~window_df_out["Vol Confirmed Entry"], "Return %"].mean()

    summary = {
        "total_years":              n,
        "vol_confirmed_entries":    confirmed,
        "vol_confirmed_pct":        round(confirmed / n * 100, 1) if n > 0 else 0,
        "confirmed_up":             confirmed_up,
        "unconfirmed_up":           unconfirmed_up,
        "accumulation_windows":     accumulation,
        "accumulation_pct":         round(accumulation / n * 100, 1) if n > 0 else 0,
        "divergence_count":         divergence_count,
        "avg_return_vol_confirmed": round(float(avg_ret_confirmed),   2) if not np.isnan(avg_ret_confirmed)   else None,
        "avg_return_unconfirmed":   round(float(avg_ret_unconfirmed), 2) if not np.isnan(avg_ret_unconfirmed) else None,
    }

    return {
        "symbol":         symbol,
        "monthly_vol_df": monthly_vol_df,
        "window_df":      window_df_out,
        "obv_by_year":    obv_by_year,
        "summary":        summary,
    }
