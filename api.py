"""
FastAPI backend for the NSE Swing Trading Seasonal Analyser.

Run with:
    uvicorn api:app --reload --port 8000

The Streamlit app (app.py) continues to run separately on port 8501.
This service exposes all analysis functions as JSON REST endpoints
for consumption by the React frontend.

CORS is open to all origins for local development — tighten before
deploying to production.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from src.analyzer import (
    seasonal_analysis,
    sector_seasonal_analysis,
    best_windows_for_stock,
    universe_screener,
    monthly_return_heatmap,
    excess_return_vs_nifty,
    sector_rotation_analysis,
    mae_analysis,
    stop_loss_survival,
    similar_years_analysis,
    volume_analysis,
)
from src.fetcher import (
    bulk_download,
    fetch_indices,
    fetch_custom_ticker,
    get_custom_tickers,
    update_custom_tickers,
    get_data_status,
    get_index_status,
    has_any_data,
    has_index_data,
    INDEX_TICKERS,
)
from src.universe import get_stocks, get_sectors, get_symbol_to_name
from src.db import init_db

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NSE Swing Trading API",
    description="Seasonal analysis engine for NSE stocks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UNIVERSE = "NIFTY500"


@app.on_event("startup")
def startup():
    init_db()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _df_to_records(df):
    """Convert a DataFrame to a JSON-safe list of dicts."""
    if df is None:
        return None
    import pandas as pd
    import numpy as np
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    # Convert numpy scalars to Python natives
    def _clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    return [{k: _clean(v) for k, v in row.items()} for row in records]


def _clean_summary(summary: dict) -> dict:
    """Strip non-serialisable norm_series_map from summary and serialise."""
    import pandas as pd
    import numpy as np

    if summary is None:
        return None

    s = {k: v for k, v in summary.items() if k != "norm_series_map"}

    # Build norm_series for fan chart: {year: [{index, value}, ...]}
    norm_map = summary.get("norm_series_map", {})
    fan_data = {}
    for year, series in norm_map.items():
        if hasattr(series, "reset_index"):
            fan_data[str(year)] = [
                {"day": i, "value": round(float(v), 4)}
                for i, v in enumerate(series.values)
                if pd.notna(v)
            ]
    s["fan_series"] = fan_data

    # Convert numpy types
    def _clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    return {k: _clean(v) if not isinstance(v, dict) else v for k, v in s.items()}


# ─── Universe & Meta ──────────────────────────────────────────────────────────

@app.get("/api/universe/stocks")
def get_universe_stocks(universe: str = Query(default=UNIVERSE)):
    """Return the list of all stocks in the universe plus any custom tickers."""
    stocks = get_stocks(universe)
    return {"stocks": stocks}


@app.get("/api/universe/sectors")
def get_universe_sectors(universe: str = Query(default=UNIVERSE)):
    sectors = get_sectors(universe)
    return {"sectors": {k: v for k, v in sectors.items()}}


@app.get("/api/universe/custom-tickers")
def list_custom_tickers():
    return {"tickers": get_custom_tickers()}


# ─── Stock Seasonal Analysis ──────────────────────────────────────────────────

@app.get("/api/analysis/seasonal")
def get_seasonal_analysis(
    symbol: str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    min_return:  float = Query(default=0.0),
):
    """
    Seasonal analysis for one stock.

    Response includes:
      - rows       : list of per-year trade rows
      - summary    : aggregate stats
      - fan_series : {year: [{day, value}]} for fan chart (indexed to 100 at entry)
    """
    results_df, summary = seasonal_analysis(symbol, start_month, start_day, holding_days, min_return)
    if results_df is None:
        if summary and summary.get("error") == "insufficient_years":
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient history: only {summary.get('years_found', 0)} completed window(s) found (need at least 3).",
            )
        raise HTTPException(status_code=404, detail="No data found for this symbol/window.")

    clean = _clean_summary(summary)
    fan = clean.pop("fan_series", {})

    # Add losing_years list for AI prompt
    losing = results_df.loc[~results_df["target_met"], "year"].tolist()
    clean["losing_years"] = [int(y) for y in losing]
    clean["win_rate_pct"] = round(clean["target_met_count"] / clean["total_instances"] * 100, 1)
    clean["median_return_pct"] = round(float(results_df["return_pct"].median()), 2)
    # Pass through low_sample_warning for the React frontend
    clean.setdefault("low_sample_warning", False)

    return {
        "rows":       _df_to_records(results_df.drop(columns=["mae_pct", "mfe_pct"], errors="ignore")),
        "summary":    clean,
        "fan_series": fan,
    }


# ─── Sector Analysis ──────────────────────────────────────────────────────────

@app.get("/api/analysis/sector")
def get_sector_analysis(
    sector: str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    min_return:  float = Query(default=0.0),
    universe: str = Query(default=UNIVERSE),
):
    result_df = sector_seasonal_analysis(sector, start_month, start_day, holding_days, min_return, universe)
    if result_df is None:
        raise HTTPException(status_code=404, detail="No data for this sector/window.")
    return {"rows": _df_to_records(result_df)}


# ─── Best Windows ─────────────────────────────────────────────────────────────

@app.get("/api/analysis/best-windows/stock")
def get_best_windows_stock(
    symbol: str,
    holding_days: int = Query(default=90, ge=5, le=365),
    min_return:   float = Query(default=12.0),
):
    result_df = best_windows_for_stock(symbol, holding_days, min_return)
    if result_df is None:
        raise HTTPException(status_code=404, detail="No data found.")
    return {"rows": _df_to_records(result_df)}


@app.get("/api/analysis/best-windows/universe")
def get_universe_screener(
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    min_return:   float = Query(default=12.0),
    universe: str = Query(default=UNIVERSE),
):
    result_df = universe_screener(start_month, start_day, holding_days, min_return, universe)
    if result_df is None:
        raise HTTPException(status_code=404, detail="No data found.")
    return {"rows": _df_to_records(result_df)}


# ─── Deep Insights ────────────────────────────────────────────────────────────

@app.get("/api/analysis/heatmap")
def get_monthly_heatmap(symbol: str):
    """Monthly return heatmap — returns {year: {month: return_pct}}."""
    pivot = monthly_return_heatmap(symbol)
    if pivot is None:
        raise HTTPException(status_code=404, detail="No data found.")
    # Convert pivot to {year: {month_name: value}}
    import pandas as pd
    result = {}
    for year in pivot.index:
        result[int(year)] = {
            col: (None if pd.isna(pivot.loc[year, col]) else round(float(pivot.loc[year, col]), 2))
            for col in pivot.columns
        }
    months = list(pivot.columns)
    return {"data": result, "months": months}


@app.get("/api/analysis/excess-return")
def get_excess_return(
    symbol: str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    min_return:  float = Query(default=0.0),
):
    import pandas as pd
    results_df, summary = excess_return_vs_nifty(symbol, start_month, start_day, holding_days, min_return)
    if results_df is None:
        raise HTTPException(status_code=404, detail="No data. Download indices first.")

    # VIX regime breakdown with target hit rate
    regime_breakdown = []
    vix_data = results_df.dropna(subset=["vix_at_entry"]).copy()
    if len(vix_data) >= 3 and summary and summary.get("vix_available"):
        vix_data["vix_regime"] = pd.cut(
            vix_data["vix_at_entry"],
            bins=[0, 14, 20, 100],
            labels=["Calm (<14)", "Normal (14\u201320)", "Elevated (>20)"],
        )
        for regime, grp in vix_data.groupby("vix_regime", observed=True):
            regime_breakdown.append({
                "regime":         str(regime),
                "years":          len(grp),
                "target_met":     int(grp["target_met"].sum()),
                "hit_rate_pct":   round(float(grp["target_met"].mean() * 100), 1),
                "avg_return_pct": round(float(grp["stock_return"].mean()), 2),
                "min_pct":        round(float(grp["stock_return"].min()), 2),
                "max_pct":        round(float(grp["stock_return"].max()), 2),
            })

    import numpy as np
    def _clean_val(v):
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
        if isinstance(v, (np.bool_,)):    return bool(v)
        return v
    clean_summary = {k: _clean_val(v) for k, v in (summary or {}).items()}

    return {
        "rows":                _df_to_records(results_df),
        "summary":             clean_summary,
        "vix_regime_breakdown": regime_breakdown,
        "min_return":          min_return,
    }


@app.get("/api/analysis/days-to-target")
def get_days_to_target(
    symbol: str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    min_return:  float = Query(default=12.0),
):
    """
    For each year where the target was met, return the first calendar day
    (from entry) that the indexed price crossed 100 + min_return.
    """
    results_df, summary = seasonal_analysis(symbol, start_month, start_day, holding_days, min_return)
    if results_df is None:
        raise HTTPException(status_code=404, detail="No data found.")

    norm_map  = summary.get("norm_series_map", {})
    threshold = 100.0 + min_return
    records   = []
    for _, row in results_df.iterrows():
        year = row["year"]
        if not row["target_met"] or year not in norm_map:
            continue
        series  = norm_map[year].values
        crossed = [i for i, v in enumerate(series) if v >= threshold]
        if crossed:
            records.append({"year": int(year), "days_to_target": crossed[0],
                            "return_pct": round(float(row["return_pct"]), 2)})

    if not records:
        return {"rows": [], "avg_days": None, "min_days": None, "max_days": None,
                "years_met": 0, "total_years": int(len(results_df))}

    days = [r["days_to_target"] for r in records]
    return {
        "rows":        records,
        "avg_days":    round(sum(days) / len(days), 1),
        "min_days":    int(min(days)),
        "max_days":    int(max(days)),
        "years_met":   len(records),
        "total_years": int(len(results_df)),
    }


@app.get("/api/analysis/sector-rotation")
def get_sector_rotation(universe: str = Query(default=UNIVERSE)):
    pivot = sector_rotation_analysis(universe)
    if pivot is None:
        raise HTTPException(status_code=404, detail="No data found.")
    import pandas as pd
    result = {}
    for sector in pivot.index:
        result[str(sector)] = {
            col: (None if pd.isna(pivot.loc[sector, col]) else round(float(pivot.loc[sector, col]), 2))
            for col in pivot.columns
        }
    months = list(pivot.columns)
    return {"data": result, "sectors": list(pivot.index), "months": months}


@app.get("/api/analysis/mae")
def get_mae_analysis(
    symbol: str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
):
    mae_df = mae_analysis(symbol, start_month, start_day, holding_days)
    if mae_df is None:
        raise HTTPException(status_code=404, detail="No data found.")
    survival_df = stop_loss_survival(mae_df)
    return {
        "mae_rows":      _df_to_records(mae_df),
        "survival_rows": _df_to_records(survival_df),
    }


@app.get("/api/analysis/similar-years")
def get_similar_years(
    symbol: str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    n_similar:   int = Query(default=5, ge=3, le=10),
):
    result = similar_years_analysis(symbol, start_month, start_day, holding_days, n_similar)
    if result is None:
        raise HTTPException(status_code=404, detail="No data found.")
    return {
        "today_features":  result.get("today_features"),
        "today_entry":     result.get("today_entry"),
        "all_years":       _df_to_records(result.get("all_years")),
        "similar_years":   _df_to_records(result.get("similar_years")),
        "missing_indices": result.get("missing_indices", False),
    }


@app.get("/api/analysis/volume")
def get_volume_analysis(
    symbol: str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
):
    """
    Volume-based analytics for one stock / window.

    Response includes:
      - monthly_vol  : [{Month, Avg Normalised Volume}] — seasonal volume rhythm
      - window_rows  : per-year volume + return stats
      - obv_by_year  : {year: [cumulative_obv_values]} — OBV within each window
      - summary      : aggregate confirmation stats
    """
    result = volume_analysis(symbol, start_month, start_day, holding_days)
    if result is None:
        raise HTTPException(status_code=404, detail="No volume data found for this symbol/window.")

    # Serialise boolean columns in window_df
    import pandas as pd
    wdf = result["window_df"].copy()
    for col in ["Vol Confirmed Entry", "Accumulation Window", "Vol-Price Divergence"]:
        if col in wdf.columns:
            wdf[col] = wdf[col].astype(bool)

    return {
        "monthly_vol":  _df_to_records(result["monthly_vol_df"]),
        "window_rows":  _df_to_records(wdf),
        "obv_by_year":  {str(k): v for k, v in result["obv_by_year"].items()},
        "summary":      result["summary"],
    }


@app.get("/api/analysis/sensitivity")
def get_entry_date_sensitivity(
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    min_return:   float = Query(default=12.0),
    top_n:        int = Query(default=20, ge=5, le=50),
    universe: str = Query(default=UNIVERSE),
):
    """
    Entry date sensitivity check for the top-N stocks from universe_screener.

    For each of the 7 dates (±3 days around the chosen entry), runs
    seasonal_analysis for every top-N stock and returns the target_met_count.

    Response:
      - base_label : the column label for the chosen entry (offset 0)
      - offsets    : list of offset dicts {offset, label, month, day}
      - rows       : [{symbol, name, <offset_label>: count, ...}]
    """
    import datetime

    # First get the ranked list for the base window
    ranked_df = universe_screener(start_month, start_day, holding_days, min_return, universe)
    if ranked_df is None:
        raise HTTPException(status_code=404, detail="No screener results for this window.")

    top_stocks = ranked_df.head(top_n)[["Symbol", "Name"]].to_dict(orient="records")
    month_abbrs = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    base_date = datetime.date(2000, start_month, start_day)
    offsets_meta = []
    for offset in range(-3, 4):
        shifted = base_date + datetime.timedelta(days=offset)
        label = f"{'+' if offset >= 0 else ''}{offset}d ({month_abbrs[shifted.month-1]} {shifted.day})"
        offsets_meta.append({"offset": offset, "label": label, "month": shifted.month, "day": shifted.day})

    rows = []
    for stock in top_stocks:
        sym = stock["Symbol"]
        row = {"symbol": sym, "name": stock["Name"]}
        for om in offsets_meta:
            _, summary = seasonal_analysis(sym, om["month"], om["day"], holding_days, min_return)
            row[om["label"]] = summary.get("target_met_count", 0) if (summary and "error" not in summary) else 0
        # Compute wobble
        counts = [row[om["label"]] for om in offsets_meta]
        row["wobble"] = max(counts) - min(counts)
        row["stability"] = "Robust" if row["wobble"] == 0 else ("Minor" if row["wobble"] <= 1 else "Fragile")
        rows.append(row)

    base_label = offsets_meta[3]["label"]   # offset=0 is index 3
    return {
        "base_label": base_label,
        "offsets":    offsets_meta,
        "rows":       rows,
    }


@app.get("/api/analysis/dashboard-summary")
def get_dashboard_summary(
    symbol: str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    min_return:  float = Query(default=12.0),
):
    """
    Run all analyses for one stock / window and return a structured numbered
    plain-English summary matching the Key Takeaways in the dashboard.

    Response keys:
      points        : list of {number, text} — the numbered takeaway sentences
      raw           : underlying numbers for each point (for building UI)
    """
    import datetime as _dt
    import numpy as np
    import pandas as pd

    month_abbrs = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    name_map    = get_symbol_to_name(UNIVERSE)
    sym_name    = name_map.get(symbol, symbol)
    win_label   = f"{month_abbrs[start_month-1]} {start_day} +{holding_days}d"

    # ── Run all analyses ──────────────────────────────────────────────────────
    res_df, s_sum = seasonal_analysis(symbol, start_month, start_day, holding_days, min_return)
    bw_df         = best_windows_for_stock(symbol, holding_days, min_return)
    ev_df, ev_sum = excess_return_vs_nifty(symbol, start_month, start_day, holding_days, min_return)
    sim_result    = similar_years_analysis(symbol, start_month, start_day, holding_days, 5)

    # Today's VIX from DB
    today_vix = None
    try:
        from src.analyzer import _load_index_closes as _lic
        from src.db import get_connection as _gc
        _conn = _gc(); _vdf = _lic("^INDIAVIX", _conn); _conn.close()
        if not _vdf.empty:
            today_vix = round(float(_vdf["close"].iloc[-1]), 2)
    except Exception:
        pass

    points = []
    raw    = {}

    # 1. Hit rate
    if s_sum and "error" not in s_sum:
        tot    = s_sum["total_instances"]
        hit_ct = s_sum["target_met_count"]
        avg_all = s_sum["avg_return_pct"]
        avg_met = s_sum.get("avg_return_when_met")
        raw["seasonal"] = {"total": tot, "hit_count": hit_ct,
                           "avg_all": avg_all, "avg_when_met": avg_met}
        if avg_met is not None:
            points.append({
                "number": 1,
                "text":   f"Target met **{hit_ct} out of {tot} years** with an average return of "
                          f"**{avg_met:+.2f}%** when the target was met (avg across all years: **{avg_all:+.2f}%**).",
            })
        else:
            points.append({
                "number": 1,
                "text":   f"Target of ≥{min_return:.0f}% was **never met** in this window "
                          f"across all {tot} years (avg return: **{avg_all:+.2f}%**).",
            })

    # 2. Days to target
    if res_df is not None and s_sum and "norm_series_map" in s_sum:
        norm_map  = s_sum["norm_series_map"]
        threshold = 100.0 + min_return
        dtd = []
        for _, row in res_df.iterrows():
            yr = row["year"]
            if not row["target_met"] or yr not in norm_map:
                continue
            series  = norm_map[yr].values
            crossed = [i for i, v in enumerate(series) if v >= threshold]
            if crossed:
                dtd.append(crossed[0])
        if dtd:
            avg_d = round(sum(dtd) / len(dtd))
            raw["days_to_target"] = {"avg": avg_d, "min": min(dtd), "max": max(dtd)}
            points.append({
                "number": 2,
                "text":   f"On average, it took **{avg_d} days** to hit the ≥{min_return:.0f}% target "
                          f"historically whenever the target was met "
                          f"(fastest: **{min(dtd)}d**, slowest: **{max(dtd)}d**).",
            })

    # 3. Best entry windows top-3
    if bw_df is not None and not bw_df.empty:
        medals = ["🥇", "🥈", "🥉"]
        cur_abbr = month_abbrs[start_month - 1]
        top3 = []
        for i, (_, r) in enumerate(bw_df.head(3).iterrows()):
            suffix = " ← your entry month" if r["Window"].startswith(cur_abbr) else ""
            top3.append(f"{medals[i]} {r['Window']} — {r['Target Met (yrs)']} of "
                        f"{r['Out of (yrs)']} yrs, avg {r['Avg Return %']:+.2f}%{suffix}")
        cur_rows  = bw_df[bw_df["Window"].str.startswith(cur_abbr)]
        extra = ""
        if not cur_rows.empty:
            cr       = cur_rows.iloc[0]
            cur_rank = list(bw_df.index).index(cr.name) + 1
            if cur_rank > 3:
                extra = (f" (Your month {cur_abbr} ranks #{cur_rank} — "
                         f"{cr['Target Met (yrs)']} of {cr['Out of (yrs)']} yrs, avg {cr['Avg Return %']:+.2f}%)")
        raw["best_windows"] = [r for r in _df_to_records(bw_df.head(3))]
        points.append({
            "number": 3,
            "text":   "Best entry windows for this stock:  \n" + "  \n".join(top3) + extra,
        })

    # 4. Beat NIFTY
    if ev_sum and ev_sum.get("nifty_available"):
        avg_exc  = ev_sum.get("avg_excess_return", 0)
        beat_lbl = ev_sum.get("beat_index_label", "")
        direction = "outperforms" if (avg_exc or 0) > 0 else "underperforms"
        raw["nifty"] = {"avg_excess": avg_exc, "beat_label": beat_lbl}
        points.append({
            "number": 4,
            "text":   f"Stock **{direction}** NIFTY in **{beat_lbl}** in this window "
                      f"(avg excess return: **{avg_exc:+.2f}%**).",
        })

    # 5. India VIX vs Returns
    if ev_df is not None and today_vix is not None:
        vix_data = ev_df.dropna(subset=["vix_at_entry"]).copy()
        if len(vix_data) >= 2:
            if today_vix < 14:
                regime_label  = "Calm (<14)"
                regime_filter = vix_data["vix_at_entry"] < 14
            elif today_vix <= 20:
                regime_label  = "Normal (14–20)"
                regime_filter = (vix_data["vix_at_entry"] >= 14) & (vix_data["vix_at_entry"] <= 20)
            else:
                regime_label  = "Elevated (>20)"
                regime_filter = vix_data["vix_at_entry"] > 20
            rd = vix_data[regime_filter]
            if len(rd) >= 1:
                rv_avg = round(float(rd["stock_return"].mean()), 2)
                rv_max = round(float(rd["stock_return"].max()), 2)
                rv_met = int(rd["target_met"].sum())
                rv_tot = len(rd)
                raw["vix_regime"] = {
                    "today_vix": today_vix, "regime": regime_label,
                    "avg_return": rv_avg, "max_return": rv_max,
                    "target_met": rv_met, "total": rv_tot,
                }
                points.append({
                    "number": 5,
                    "text":   f"India VIX today is **{today_vix}** ({regime_label} regime).  \n"
                              f"In {regime_label}-VIX years: avg return **{rv_avg:+.2f}%**, "
                              f"max **{rv_max:+.2f}%**, target met **{rv_met} of {rv_tot} times**.",
                })

    # 6. Most similar year
    if sim_result and not sim_result.get("missing_indices"):
        sim_yrs = sim_result.get("similar_years")
        if sim_yrs is not None and not sim_yrs.empty:
            top_sim = sim_yrs.iloc[0]
            sim_yr  = int(top_sim["Year"])
            sim_ret = float(top_sim["Final Return %"])
            raw["similar_year"] = {"year": sim_yr, "return_pct": round(sim_ret, 2)}
            points.append({
                "number": 6,
                "text":   f"Most similar historical year to **{_dt.date.today().year}** is "
                          f"**{sim_yr}**, when the stock returned **{sim_ret:+.2f}%** in this window.",
            })

    return {
        "symbol":    symbol,
        "sym_name":  sym_name,
        "win_label": win_label,
        "points":    points,
        "raw":       raw,
    }


@app.get("/api/analysis/compare")
def get_stock_comparison(
    symbol_a:    str,
    symbol_b:    str,
    start_month: int = Query(ge=1, le=12),
    start_day:   int = Query(ge=1, le=31),
    holding_days: int = Query(ge=5, le=365),
    min_return:  float = Query(default=12.0),
):
    """
    Side-by-side seasonal comparison of two stocks for the same window.

    Response:
      - symbol_a / symbol_b : tickers
      - name_a / name_b     : friendly names
      - win_label           : window description
      - summary_a / summary_b : aggregate stats for each stock
      - rows_a / rows_b     : per-year trade rows for each stock
      - overlap_years       : years present in BOTH series
      - correlation         : Pearson r of returns across overlap years
      - divergence_years    : years where A met target but B didn't (and vice versa)
      - days_to_target_a / days_to_target_b : avg/min/max days to first touch target
      - nifty_a / nifty_b   : beat-NIFTY summary for each stock
    """
    import numpy as np

    name_map = get_symbol_to_name(UNIVERSE)

    def _run(sym):
        df, s = seasonal_analysis(sym, start_month, start_day, holding_days, min_return)
        ev_df, ev_s = excess_return_vs_nifty(sym, start_month, start_day, holding_days, min_return)
        norm_map = s.get("norm_series_map", {}) if s else {}
        threshold = 100.0 + min_return
        dtd = []
        if df is not None:
            for _, row in df.iterrows():
                yr = row["year"]
                if not row["target_met"] or yr not in norm_map:
                    continue
                series  = norm_map[yr].values
                crossed = [i for i, v in enumerate(series) if v >= threshold]
                if crossed:
                    dtd.append(crossed[0])
        days_stat = None
        if dtd:
            days_stat = {"avg": round(sum(dtd)/len(dtd), 1), "min": min(dtd), "max": max(dtd)}
        return df, s, ev_s, days_stat

    df_a, s_a, ev_a, dtd_a = _run(symbol_a)
    df_b, s_b, ev_b, dtd_b = _run(symbol_b)

    if df_a is None and df_b is None:
        raise HTTPException(status_code=404, detail="No data found for either symbol.")

    def _clean_s(s):
        if s is None:
            return None
        import numpy as np
        def _cv(v):
            if isinstance(v, (np.integer,)):  return int(v)
            if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
            if isinstance(v, (np.bool_,)):    return bool(v)
            return v
        return {k: _cv(v) for k, v in s.items() if k != "norm_series_map"}

    # Overlap years & correlation
    years_a = set(df_a["year"].tolist()) if df_a is not None else set()
    years_b = set(df_b["year"].tolist()) if df_b is not None else set()
    overlap  = sorted(years_a & years_b)
    correlation = None
    if len(overlap) >= 3 and df_a is not None and df_b is not None:
        ra = df_a[df_a["year"].isin(overlap)].set_index("year")["return_pct"]
        rb = df_b[df_b["year"].isin(overlap)].set_index("year")["return_pct"]
        ra, rb = ra.align(rb)
        if len(ra.dropna()) >= 3:
            correlation = round(float(np.corrcoef(ra.values, rb.values)[0, 1]), 3)

    # Divergence years
    div_a_wins = []  # A met target, B didn't
    div_b_wins = []  # B met target, A didn't
    if df_a is not None and df_b is not None:
        ma = df_a[df_a["year"].isin(overlap)].set_index("year")["target_met"]
        mb = df_b[df_b["year"].isin(overlap)].set_index("year")["target_met"]
        ma, mb = ma.align(mb)
        for yr in overlap:
            if ma.get(yr) and not mb.get(yr):
                div_a_wins.append(yr)
            elif mb.get(yr) and not ma.get(yr):
                div_b_wins.append(yr)

    month_abbrs = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    win_label   = f"{month_abbrs[start_month-1]} {start_day} +{holding_days}d"

    def _ev_clean(ev):
        if ev is None:
            return None
        import numpy as np
        def _cv(v):
            if isinstance(v, (np.integer,)):  return int(v)
            if isinstance(v, (np.floating,)): return None if np.isnan(v) else float(v)
            if isinstance(v, (np.bool_,)):    return bool(v)
            return v
        return {k: _cv(v) for k, v in ev.items()}

    rows_a = _df_to_records(df_a.drop(columns=["mae_pct","mfe_pct"], errors="ignore")) if df_a is not None else []
    rows_b = _df_to_records(df_b.drop(columns=["mae_pct","mfe_pct"], errors="ignore")) if df_b is not None else []

    return {
        "symbol_a":   symbol_a,
        "symbol_b":   symbol_b,
        "name_a":     name_map.get(symbol_a, symbol_a),
        "name_b":     name_map.get(symbol_b, symbol_b),
        "win_label":  win_label,
        "summary_a":  _clean_s(s_a),
        "summary_b":  _clean_s(s_b),
        "rows_a":     rows_a,
        "rows_b":     rows_b,
        "overlap_years":    overlap,
        "correlation":      correlation,
        "div_a_wins":       div_a_wins,
        "div_b_wins":       div_b_wins,
        "days_to_target_a": dtd_a,
        "days_to_target_b": dtd_b,
        "nifty_a":    _ev_clean(ev_a),
        "nifty_b":    _ev_clean(ev_b),
        "min_return": min_return,
    }


# ─── AI Commentary ────────────────────────────────────────────────────────────

class CommentaryRequest(BaseModel):
    sym_name:       str
    win_label:      str
    summary:        dict
    today_features: dict | None = None
    similar_years:  list | None = None


@app.post("/api/ai/commentary")
def get_ai_commentary(req: CommentaryRequest):
    """
    Generate AI commentary from local Ollama Mistral.
    Returns the full text as a single JSON response (non-streaming).
    Use /api/ai/commentary/stream for SSE streaming.
    """
    from src.llm import build_seasonal_prompt, stream_commentary
    try:
        prompt = build_seasonal_prompt(
            sym_name=req.sym_name,
            win_label=req.win_label,
            summary=req.summary,
            today_features=req.today_features,
            similar_years=req.similar_years,
        )
        text = "".join(stream_commentary(prompt))
        return {"commentary": text}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/ai/commentary/stream")
def stream_ai_commentary(req: CommentaryRequest):
    """SSE streaming endpoint for AI commentary — yields text/event-stream."""
    from src.llm import build_seasonal_prompt, stream_commentary

    def _generate():
        try:
            prompt = build_seasonal_prompt(
                sym_name=req.sym_name,
                win_label=req.win_label,
                summary=req.summary,
                today_features=req.today_features,
                similar_years=req.similar_years,
            )
            for chunk in stream_commentary(prompt):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ─── Data Management ──────────────────────────────────────────────────────────

@app.get("/api/data/status")
def get_stock_data_status(universe: str = Query(default=UNIVERSE)):
    df = get_data_status(universe)
    return {"rows": _df_to_records(df), "has_data": has_any_data()}


@app.get("/api/data/index-status")
def get_index_data_status():
    df = get_index_status()
    return {
        "rows":      _df_to_records(df) if not df.empty else [],
        "has_data":  has_index_data(),
        "tickers":   INDEX_TICKERS,
    }


@app.post("/api/data/update")
def trigger_update(universe: str = Query(default=UNIVERSE)):
    """
    Trigger incremental OHLCV update for the full universe.
    Returns a summary of what was fetched.
    Note: runs synchronously — for large universes use the SSE version.
    """
    log = []
    def on_progress(done, total, symbol, rows, status):
        log.append({"done": done, "total": total, "symbol": symbol, "rows": rows, "status": status})
    bulk_download(universe=universe, progress_callback=on_progress)
    return {"log": log, "total": len(log)}


@app.post("/api/data/update/stream")
def stream_update(universe: str = Query(default=UNIVERSE)):
    """SSE streaming update — yields progress events as download runs."""
    def _generate():
        def on_progress(done, total, symbol, rows, status):
            payload = {"done": done, "total": total, "symbol": symbol, "rows": rows, "status": status}
            # This is called synchronously so we can't truly yield from a callback.
            # Events are collected and flushed — client polls for completion.
            pass
        # Run with accumulated log, yield events
        log = []
        def cb(done, total, symbol, rows, status):
            log.append({"done": done, "total": total, "symbol": symbol, "rows": rows, "status": status})
        bulk_download(universe=universe, progress_callback=cb)
        for entry in log:
            yield f"data: {json.dumps(entry)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/api/data/update-indices")
def trigger_index_update():
    log = []
    def on_progress(done, total, symbol, rows, status):
        log.append({"done": done, "total": total, "symbol": symbol, "rows": rows, "status": status})
    fetch_indices(progress_callback=on_progress)
    return {"log": log}


@app.post("/api/data/update-custom")
def trigger_custom_update():
    log = []
    def on_progress(done, total, symbol, rows, status):
        log.append({"done": done, "total": total, "symbol": symbol, "rows": rows, "status": status})
    update_custom_tickers(progress_callback=on_progress)
    return {"log": log}


class AddTickerRequest(BaseModel):
    symbol: str
    name:   str = ""
    sector: str = "Custom"


@app.post("/api/data/add-ticker")
def add_custom_ticker(req: AddTickerRequest):
    rows, err = fetch_custom_ticker(req.symbol, req.name, req.sector)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return {"rows_fetched": rows, "symbol": req.symbol.strip().upper()}
