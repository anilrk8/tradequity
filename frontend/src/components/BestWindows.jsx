import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { getStocks, getBestWindowsStock, getUniverseScreener, getEntrySensitivity } from "../api";
import "../styles/seasonal.css";

export default function BestWindows() {
  const [stocks, setStocks]       = useState([]);
  const [symbol, setSymbol]       = useState("");
  const [holdingDays, setHolding] = useState(90);
  const [minReturn, setMinReturn] = useState(12);
  const [mode, setMode]           = useState("stock");   // "stock" | "universe"
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [sensResult, setSensResult]   = useState(null);
  const [sensLoading, setSensLoading] = useState(false);

  // Universe screener inputs
  const today = new Date();
  const [date, setDate]   = useState(`${today.getFullYear()}-04-01`);
  const [month, setMonth] = useState(4);
  const [day, setDay]     = useState(1);

  useEffect(() => {
    getStocks().then((d) => {
      setStocks(d.stocks || []);
      if (d.stocks?.length) setSymbol(d.stocks[0].symbol);
    });
  }, []);

  const handleScan = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      if (mode === "stock") {
        const data = await getBestWindowsStock(symbol, holdingDays, minReturn);
        setResult({ mode: "stock", rows: data.rows || [] });
      } else {
        const data = await getUniverseScreener(month, day, holdingDays, minReturn);
        setResult({ mode: "universe", rows: data.rows || [] });
      }
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="tab-content">
      <div className="mode-toggle">
        <button className={`tab-btn${mode==="stock" ? " active" : ""}`} onClick={() => setMode("stock")}>
          By Stock
        </button>
        <button className={`tab-btn${mode==="universe" ? " active" : ""}`} onClick={() => setMode("universe")}>
          Universe Screener
        </button>
      </div>

      <div className="controls-row">
        {mode === "stock" ? (
          <label>Stock
            <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
              {stocks.map((s) => <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>)}
            </select>
          </label>
        ) : (
          <label>Entry Date
            <input type="date" value={date} onChange={(e) => {
              const d = new Date(e.target.value);
              setDate(e.target.value); setMonth(d.getMonth()+1); setDay(d.getDate());
            }} />
          </label>
        )}
        <label>Holding Days
          <input type="number" min={5} max={365} value={holdingDays}
            onChange={(e) => setHolding(parseInt(e.target.value) || 90)} />
        </label>
        <label>Min Return %
          <input type="number" step={0.5} value={minReturn}
            onChange={(e) => setMinReturn(parseFloat(e.target.value) || 0)} />
        </label>
        <button className="btn-primary" onClick={handleScan} disabled={loading}>
          {loading ? "Scanning…" : "Scan →"}
        </button>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {result?.rows?.length > 0 && (
        <>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart
              data={result.rows.slice(0, 20)}
              margin={{ top: 20, right: 20, bottom: 60, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis
                dataKey={result.mode === "stock" ? "Entry Month" : "Symbol"}
                angle={-40} textAnchor="end" interval={0} tick={{ fontSize: 11 }}
              />
              <YAxis />
              <Tooltip />
              <Bar dataKey={result.mode === "stock" ? "Target Met (yrs)" : "Target Met (yrs)"} radius={[3,3,0,0]}>
                {result.rows.slice(0, 20).map((entry, i) => (
                  <Cell key={i} fill={i === 0 ? "#2980b9" : i < 3 ? "#27ae60" : "#7f8c8d"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  {Object.keys(result.rows[0]).map((k) => <th key={k}>{k}</th>)}
                </tr>
              </thead>
              <tbody>
                {result.rows.map((row, i) => (
                  <tr key={i}>
                    {Object.entries(row).map(([k, v]) => (
                      <td key={k} style={{
                        color: k.includes("Return") && v != null
                          ? v >= 0 ? "#27ae60" : "#e74c3c" : undefined
                      }}>
                        {typeof v === "number" ? (k.includes("%") ? `${v > 0 ? "+" : ""}${v.toFixed(2)}%` : v) : (v ?? "—")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Entry Date Sensitivity (universe mode only) ── */}
          {result.mode === "universe" && (
            <div style={{ marginTop: 28 }}>
              <h4 className="section-header">📅 Entry Date Sensitivity (±3 Days)</h4>
              <p style={{ color: "#666", fontSize: "0.85rem", marginBottom: 10 }}>
                Shows how each top stock&apos;s <strong>Target Met count</strong> changes when you shift the entry date
                by ±3 days. <strong>Robust</strong> = no change across all 7 dates.
                <strong> Fragile</strong> = count swings by ≥2 years.
              </p>
              <button className="btn-primary" disabled={sensLoading}
                onClick={async () => {
                  setSensLoading(true); setSensResult(null);
                  try {
                    const d = await getEntrySensitivity(month, day, holdingDays, minReturn, 20);
                    setSensResult(d);
                  } catch (e) { /* silent */ }
                  finally { setSensLoading(false); }
                }}>
                {sensLoading ? "Checking…" : "Run Sensitivity Check →"}
              </button>

              {sensResult && (
                <SensitivityTable data={sensResult} />
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function SensitivityTable({ data }) {
  const { rows, offsets, base_label } = data;
  const maxYears = Math.max(...rows.flatMap((r) => offsets.map((o) => r[o.label] || 0)));

  const cellColor = (v) => {
    const pct = v / Math.max(maxYears, 1);
    if (pct >= 0.7) return { background: "#c7f2c7", color: "#1a5c1a" };
    if (pct >= 0.5) return { background: "#f7f7c7", color: "#5c5c00" };
    if (pct >= 0.3) return { background: "#f2ddc7", color: "#7a3e00" };
    return { background: "#f2c7c7", color: "#7a0000" };
  };

  const wobbleColor  = (w) => w === 0 ? "#c7f2c7" : w <= 1 ? "#f7f7c7" : "#f2c7c7";
  const stabIcon     = (s) => s === "Robust" ? "🟢" : s === "Minor" ? "🟡" : "🔴";

  return (
    <div className="table-container" style={{ marginTop: 12 }}>
      <table className="data-table">
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>Stock</th>
            {offsets.map((o) => (
              <th key={o.label} style={o.label === base_label ? { borderLeft: "2px solid #2980b9", borderRight: "2px solid #2980b9", fontWeight: 700 } : {}}>
                {o.label}
              </th>
            ))}
            <th>Wobble</th>
            <th>Stability</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td style={{ textAlign: "left", whiteSpace: "nowrap" }}>{row.name} ({row.symbol})</td>
              {offsets.map((o) => (
                <td key={o.label} style={{
                  ...cellColor(row[o.label] || 0),
                  ...(o.label === base_label ? { borderLeft: "2px solid #2980b9", borderRight: "2px solid #2980b9", fontWeight: 700 } : {}),
                }}>
                  {row[o.label] ?? 0}
                </td>
              ))}
              <td style={{ background: wobbleColor(row.wobble), fontWeight: 600 }}>{row.wobble}</td>
              <td>{stabIcon(row.stability)} {row.stability}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
