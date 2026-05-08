/**
 * API client for the NSE Swing Trading FastAPI backend.
 * Base URL defaults to http://localhost:8000 — override via
 * the REACT_APP_API_URL environment variable.
 */

const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function _get(path, params = {}) {
  const url = new URL(`${BASE}${path}`);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  });
  const res = await fetch(url.toString());
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function _post(path, body = {}, params = {}) {
  const url = new URL(`${BASE}${path}`);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  });
  const res = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ── Universe ──────────────────────────────────────────────────────────────────

export const getStocks        = (universe = "NIFTY500") => _get("/api/universe/stocks", { universe });
export const getSectors       = (universe = "NIFTY500") => _get("/api/universe/sectors", { universe });
export const getCustomTickers = ()                       => _get("/api/universe/custom-tickers");

// ── Analysis ──────────────────────────────────────────────────────────────────

export const getSeasonalAnalysis = (symbol, start_month, start_day, holding_days, min_return = 0) =>
  _get("/api/analysis/seasonal", { symbol, start_month, start_day, holding_days, min_return });

export const getSectorAnalysis = (sector, start_month, start_day, holding_days, min_return = 0, universe = "NIFTY500") =>
  _get("/api/analysis/sector", { sector, start_month, start_day, holding_days, min_return, universe });

export const getBestWindowsStock = (symbol, holding_days = 90, min_return = 12) =>
  _get("/api/analysis/best-windows/stock", { symbol, holding_days, min_return });

export const getUniverseScreener = (start_month, start_day, holding_days, min_return = 12, universe = "NIFTY500") =>
  _get("/api/analysis/best-windows/universe", { start_month, start_day, holding_days, min_return, universe });

export const getMonthlyHeatmap   = (symbol) => _get("/api/analysis/heatmap",        { symbol });
export const getExcessReturn     = (symbol, start_month, start_day, holding_days)   =>
  _get("/api/analysis/excess-return", { symbol, start_month, start_day, holding_days });

export const getSectorRotation   = (universe = "NIFTY500") => _get("/api/analysis/sector-rotation", { universe });

export const getMaeAnalysis      = (symbol, start_month, start_day, holding_days)   =>
  _get("/api/analysis/mae", { symbol, start_month, start_day, holding_days });

export const getSimilarYears     = (symbol, start_month, start_day, holding_days, n_similar = 5) =>
  _get("/api/analysis/similar-years", { symbol, start_month, start_day, holding_days, n_similar });

export const getVolumeAnalysis   = (symbol, start_month, start_day, holding_days) =>
  _get("/api/analysis/volume", { symbol, start_month, start_day, holding_days });

export const getEntrySensitivity = (start_month, start_day, holding_days, min_return = 12, top_n = 20, universe = "NIFTY500") =>
  _get("/api/analysis/sensitivity", { start_month, start_day, holding_days, min_return, top_n, universe });

// ── AI Commentary ─────────────────────────────────────────────────────────────

export const getAiCommentary = (sym_name, win_label, summary, today_features = null, similar_years = null) =>
  _post("/api/ai/commentary", { sym_name, win_label, summary, today_features, similar_years });

/**
 * Stream AI commentary via SSE.
 * onChunk(text) is called for each token received.
 * onDone() is called when streaming finishes.
 * onError(msg) is called on error.
 */
export function streamAiCommentary(sym_name, win_label, summary, today_features, similar_years, onChunk, onDone, onError) {
  fetch(`${BASE}/api/ai/commentary/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sym_name, win_label, summary, today_features, similar_years }),
  }).then((res) => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    function pump() {
      reader.read().then(({ done, value }) => {
        if (done) { onDone(); return; }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        lines.forEach((line) => {
          if (!line.startsWith("data: ")) return;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") { onDone(); return; }
          try {
            const msg = JSON.parse(raw);
            if (msg.error) { onError(msg.error); return; }
            if (msg.chunk) onChunk(msg.chunk);
          } catch (_) {}
        });
        pump();
      });
    }
    pump();
  }).catch((e) => onError(e.message));
}

// ── Data Management ───────────────────────────────────────────────────────────

export const getDataStatus    = (universe = "NIFTY500") => _get("/api/data/status",       { universe });
export const getIndexStatus   = ()                       => _get("/api/data/index-status");
export const triggerUpdate    = (universe = "NIFTY500") => _post("/api/data/update",      {}, { universe });
export const triggerIndexUpdate   = ()                   => _post("/api/data/update-indices");
export const triggerCustomUpdate  = ()                   => _post("/api/data/update-custom");
export const addCustomTicker  = (symbol, name = "")      => _post("/api/data/add-ticker", { symbol, name });
