import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
  ComposedChart, Line, Legend,
} from "recharts";
import { getStocks, getVolumeAnalysis } from "../api";
import "../styles/seasonal.css";

const MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

export default function VolumeAnalysis() {
  const [stocks, setStocks]       = useState([]);
  const [symbol, setSymbol]       = useState("");
  const [holdingDays, setHolding] = useState(90);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [activeTab, setActiveTab] = useState("rhythm");

  const today = new Date();
  const mm = String(today.getMonth() + 1).padStart(2, "0");
  const dd = String(today.getDate()).padStart(2, "0");
  const [date, setDate]   = useState(`${today.getFullYear()}-${mm}-${dd}`);
  const [month, setMonth] = useState(today.getMonth() + 1);
  const [day, setDay]     = useState(today.getDate());

  useEffect(() => {
    getStocks().then((d) => {
      setStocks(d.stocks || []);
      if (d.stocks?.length) setSymbol(d.stocks[0].symbol);
    });
  }, []);

  const handleRun = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await getVolumeAnalysis(symbol, month, day, holdingDays);
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const symName = stocks.find((s) => s.symbol === symbol)?.name || symbol;

  return (
    <div className="tab-content">
      <div className="controls-row">
        <label>Stock
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {stocks.map((s) => <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>)}
          </select>
        </label>
        <label>Entry Date
          <input type="date" value={date} onChange={(e) => {
            const d = new Date(e.target.value);
            setDate(e.target.value); setMonth(d.getMonth()+1); setDay(d.getDate());
          }} />
        </label>
        <label>Holding Days
          <input type="number" min={5} max={365} value={holdingDays}
            onChange={(e) => setHolding(parseInt(e.target.value) || 90)} />
        </label>
        <button className="btn-primary" onClick={handleRun} disabled={loading}>
          {loading ? "Analysing…" : "Run Volume Analysis →"}
        </button>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <>
          <SummaryMetrics summary={result.summary} />

          <div className="tab-bar" style={{ marginTop: 16 }}>
            {[
              { id: "rhythm",  label: "📅 Seasonal Rhythm" },
              { id: "confirm", label: "📊 Vol vs Return" },
              { id: "obv",     label: "📈 OBV" },
              { id: "table",   label: "🗃 Raw Data" },
            ].map((t) => (
              <button key={t.id}
                className={`tab-btn${activeTab === t.id ? " active" : ""}`}
                onClick={() => setActiveTab(t.id)}>{t.label}</button>
            ))}
          </div>

          {activeTab === "rhythm"  && <SeasonalRhythm data={result.monthly_vol} symName={symName} />}
          {activeTab === "confirm" && <VolVsReturn rows={result.window_rows} summary={result.summary} symName={symName} />}
          {activeTab === "obv"     && <OBVChart obvByYear={result.obv_by_year} windowRows={result.window_rows} symName={symName} />}
          {activeTab === "table"   && <RawTable rows={result.window_rows} />}
        </>
      )}
    </div>
  );
}

/* ── Summary metrics ── */
function SummaryMetrics({ summary }) {
  const s = summary;
  const diff = s.avg_return_vol_confirmed != null && s.avg_return_unconfirmed != null
    ? s.avg_return_vol_confirmed - s.avg_return_unconfirmed : null;

  return (
    <>
      <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-value">{s.vol_confirmed_entries} / {s.total_years}</div>
          <div className="metric-label">Vol-Confirmed Entries</div>
        </div>
        <div className="metric-card">
          <div className={`metric-value ${s.avg_return_vol_confirmed >= 0 ? "positive" : "negative"}`}>
            {s.avg_return_vol_confirmed != null ? `${s.avg_return_vol_confirmed > 0 ? "+" : ""}${s.avg_return_vol_confirmed.toFixed(2)}%` : "—"}
          </div>
          <div className="metric-label">Avg Return (High Vol Entry)</div>
        </div>
        <div className="metric-card">
          <div className={`metric-value ${s.avg_return_unconfirmed >= 0 ? "positive" : "negative"}`}>
            {s.avg_return_unconfirmed != null ? `${s.avg_return_unconfirmed > 0 ? "+" : ""}${s.avg_return_unconfirmed.toFixed(2)}%` : "—"}
          </div>
          <div className="metric-label">Avg Return (Low Vol Entry)</div>
        </div>
        <div className="metric-card">
          <div className={`metric-value ${s.divergence_count > 2 ? "negative" : ""}`}>
            {s.divergence_count}
          </div>
          <div className="metric-label">Price-Vol Divergence Yrs</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{s.accumulation_windows} / {s.total_years}</div>
          <div className="metric-label">Accumulation Windows</div>
        </div>
      </div>
      {diff !== null && (
        <div className={`alert ${diff > 0 ? "alert-success" : "alert-info"}`}>
          {diff > 0
            ? `Vol-confirmed entries outperformed by ${diff > 0 ? "+" : ""}${diff.toFixed(2)}% on average. High-volume entry days add edge in this window.`
            : `Low-volume entries had similar or better returns in this window (${diff.toFixed(2)}% difference). Volume confirmation may not add edge here.`}
        </div>
      )}
    </>
  );
}

/* ── Seasonal Volume Rhythm ── */
function SeasonalRhythm({ data, symName }) {
  const ordered = MONTH_ORDER.map((m) => data?.find((r) => r.Month === m) || { Month: m, "Avg Normalised Volume": null });
  return (
    <div>
      <p style={{ color: "#666", fontSize: "0.85rem", margin: "12px 0 4px" }}>
        Average normalised volume per calendar month. <strong>1.0 = baseline</strong>. Above 1.0 means that month is unusually active for this stock.
      </p>
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={ordered} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="Month" />
          <YAxis domain={[0, "auto"]} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip formatter={(v) => v != null ? `${v.toFixed(2)}×` : "—"} />
          <ReferenceLine y={1} stroke="#888" strokeDasharray="5 5" label={{ value: "Baseline", position: "right", fontSize: 11 }} />
          <Bar dataKey="Avg Normalised Volume" radius={[4,4,0,0]}>
            {ordered.map((entry, i) => (
              <Cell key={i}
                fill={(entry["Avg Normalised Volume"] || 0) >= 1 ? "#27ae60" : "#e74c3c"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Volume vs Return per year ── */
function VolVsReturn({ rows, summary, symName }) {
  const divergenceYears = rows?.filter((r) => r["Vol-Price Divergence"]).map((r) => r.Year) || [];

  return (
    <div>
      <p style={{ color: "#666", fontSize: "0.85rem", margin: "12px 0 4px" }}>
        Bars = price return each year. Lines = volume ratios (right axis). &gt;1.0 = above-average volume.
      </p>
      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={rows} margin={{ top: 20, right: 50, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
          <XAxis dataKey="Year" />
          <YAxis yAxisId="left" tickFormatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`} />
          <YAxis yAxisId="right" orientation="right" domain={[0, "auto"]} tickFormatter={(v) => `${v.toFixed(1)}×`} />
          <Tooltip />
          <Legend />
          <ReferenceLine yAxisId="right" y={1} stroke="#ccc" strokeDasharray="4 4" />
          <Bar yAxisId="left" dataKey="Return %" name="Return %" radius={[3,3,0,0]}>
            {rows?.map((r, i) => <Cell key={i} fill={r.Direction === "UP" ? "#27ae60" : "#e74c3c"} />)}
          </Bar>
          <Line yAxisId="right" type="monotone" dataKey="Entry Vol Ratio"
            name="Entry Vol Ratio" stroke="#2980b9" strokeWidth={2} dot={{ r: 4 }} />
          <Line yAxisId="right" type="monotone" dataKey="Window Avg Vol Ratio"
            name="Window Avg Vol Ratio" stroke="#8e44ad" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 4 }} />
        </ComposedChart>
      </ResponsiveContainer>
      {divergenceYears.length > 0 && (
        <div className="alert alert-error" style={{ marginTop: 12 }}>
          ⚠ <strong>Price-Volume Divergence</strong> detected in years: {divergenceYears.join(", ")} — price rose but volume was well below average. These rallies may have been less reliable.
        </div>
      )}
    </div>
  );
}

/* ── OBV Chart ── */
function OBVChart({ obvByYear, windowRows, symName }) {
  const dirMap = {};
  windowRows?.forEach((r) => { dirMap[r.Year] = r.Direction; });

  return (
    <div>
      <p style={{ color: "#666", fontSize: "0.85rem", margin: "12px 0 4px" }}>
        On-Balance Volume accumulated within the trade window. Rising = buying pressure; falling = selling pressure.
        Normalised to avg daily volume. <span style={{ color: "#27ae60", fontWeight: 600 }}>Green</span> = profitable year,{" "}
        <span style={{ color: "#e74c3c", fontWeight: 600 }}>Red</span> = loss year.
      </p>
      <div style={{ overflowX: "auto" }}>
        <svg width={0} height={0}>
          <defs>
            <marker id="none" />
          </defs>
        </svg>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis type="number" dataKey="day" label={{ value: "Trading Day", position: "insideBottom", offset: -10 }} />
            <YAxis tickFormatter={(v) => `${v.toFixed(1)}`} label={{ value: "OBV (× avg vol)", angle: -90, position: "insideLeft", offset: 10 }} />
            <Tooltip formatter={(v) => `${v.toFixed(2)}×`} />
            <ReferenceLine y={0} stroke="#888" strokeDasharray="4 4" />
            {Object.entries(obvByYear || {}).map(([year, vals]) => {
              const isUp = dirMap[parseInt(year)] === "UP";
              const chartData = vals.map((v, i) => ({ day: i, value: v }));
              return (
                <Line
                  key={year}
                  data={chartData}
                  type="monotone"
                  dataKey="value"
                  dot={false}
                  name={year}
                  stroke={isUp ? "rgba(39,174,96,0.6)" : "rgba(231,76,60,0.5)"}
                  strokeWidth={1.5}
                />
              );
            })}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ── Raw Table ── */
function RawTable({ rows }) {
  if (!rows?.length) return null;
  const cols = Object.keys(rows[0]);
  return (
    <div className="table-container">
      <table className="data-table">
        <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {cols.map((c) => {
                const v = row[c];
                let style = {};
                if (c === "Return %") style.color = v >= 0 ? "#27ae60" : "#e74c3c";
                if (c === "Vol-Price Divergence" && v) style = { background: "#fff3cd" };
                return (
                  <td key={c} style={style}>
                    {typeof v === "boolean" ? (v ? "Yes" : "No") :
                     c.includes("Ratio") && v != null ? `${v.toFixed(2)}×` :
                     c === "Return %" && v != null ? `${v > 0 ? "+" : ""}${v.toFixed(2)}%` :
                     v ?? "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
