import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { getStocks, getBestWindowsStock, getUniverseScreener } from "../api";
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
        </>
      )}
    </div>
  );
}
