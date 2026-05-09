import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { getSectors, getSectorAnalysis } from "../api";
import "../styles/seasonal.css";

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

export default function SectorAnalysis() {
  const today = new Date();
  const mm = String(today.getMonth() + 1).padStart(2, "0");
  const dd = String(today.getDate()).padStart(2, "0");
  const defaultDate = `${today.getFullYear()}-${mm}-${dd}`;

  const [sectors, setSectors]   = useState([]);
  const [sector, setSector]     = useState("");
  const [date, setDate]         = useState(defaultDate);
  const [month, setMonth]       = useState(today.getMonth() + 1);
  const [day, setDay]           = useState(today.getDate());
  const [holdingDays, setHolding] = useState(90);
  const [minReturn, setMinReturn] = useState(0);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  useEffect(() => {
    getSectors().then((d) => {
      const names = Object.keys(d.sectors || {}).sort();
      setSectors(names);
      if (names.length) setSector(names[0]);
    });
  }, []);

  const handleAnalyse = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await getSectorAnalysis(sector, month, day, holdingDays, minReturn);
      setResult(data.rows || []);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const winLabel = `${String(day).padStart(2,"0")}-${MONTHS[month-1]} → ${holdingDays} days`;

  return (
    <div className="tab-content">
      <div className="controls-row">
        <label>Sector
          <select value={sector} onChange={(e) => setSector(e.target.value)}>
            {sectors.map((s) => <option key={s} value={s}>{s}</option>)}
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
        <label>Min Return %
          <input type="number" step={0.5} value={minReturn}
            onChange={(e) => setMinReturn(parseFloat(e.target.value) || 0)} />
        </label>
        <button className="btn-primary" onClick={handleAnalyse} disabled={loading || !sector}>
          {loading ? "Analysing…" : "Analyse Sector →"}
        </button>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <>
          <div className="section-header">
            <h3>{sector} — {winLabel}  ·  Target ≥{minReturn}%</h3>
          </div>

          <div className="metrics-row">
            <div className="metric-card">
              <div className="metric-value">{result.length}</div>
              <div className="metric-label">Stocks Analysed</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">
                {result.length ? `${(result.reduce((a,r)=>a+(r["Avg Return %"]||0),0)/result.length).toFixed(2)}%` : "—"}
              </div>
              <div className="metric-label">Avg Sector Return</div>
            </div>
            {result[0] && (
              <div className="metric-card">
                <div className="metric-value">{result[0]["Symbol"]}</div>
                <div className="metric-label">Top Stock</div>
              </div>
            )}
          </div>

          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={result} margin={{ top: 20, right: 20, bottom: 60, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="Symbol" angle={-40} textAnchor="end" interval={0} tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="Target Met (yrs)" radius={[3,3,0,0]}>
                {result.map((entry, i) => (
                  <Cell key={i}
                    fill={entry["Target Met (yrs)"] === 0 ? "#e74c3c"
                      : entry["Target Met (yrs)"] >= entry["Out of (yrs)"] * 0.7 ? "#27ae60"
                      : "#e67e22"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th><th>Target Met</th><th>Out of</th>
                  <th>Avg Return %</th><th>Avg When Met</th>
                  <th>Best %</th><th>Worst %</th>
                </tr>
              </thead>
              <tbody>
                {result.map((r, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{r["Symbol"]}</td>
                    <td>{r["Target Met (yrs)"]}</td>
                    <td>{r["Out of (yrs)"]}</td>
                    <td style={{ color: (r["Avg Return %"]||0) >= 0 ? "#27ae60" : "#e74c3c" }}>
                      {(r["Avg Return %"]||0) > 0 ? "+" : ""}{r["Avg Return %"]?.toFixed(2)}%
                    </td>
                    <td>{r["Avg When Target Met"] != null ? `+${r["Avg When Target Met"]?.toFixed(2)}%` : "—"}</td>
                    <td style={{ color: "#27ae60" }}>+{r["Best Return %"]?.toFixed(2)}%</td>
                    <td style={{ color: "#e74c3c" }}>{r["Worst Return %"]?.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
