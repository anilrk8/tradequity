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
    custom = get_custom_tickers()
    if custom:
        stocks = stocks + [{"name": c["name"], "symbol": c["symbol"], "sector": "Custom"} for c in custom]
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


@app.post("/api/data/add-ticker")
def add_custom_ticker(req: AddTickerRequest):
    rows, err = fetch_custom_ticker(req.symbol, req.name)
    if err:
        raise HTTPException(status_code=400, detail=err)
    return {"rows_fetched": rows, "symbol": req.symbol.strip().upper()}
