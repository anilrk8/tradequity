import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, ScatterChart, Scatter, Legend, Cell, LineChart, Line,
} from "recharts";
import { getStocks, getStockComparison } from "../api";
import "../styles/seasonal.css";

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function todayStr() {
  const t = new Date();
  const mm = String(t.getMonth() + 1).padStart(2, "0");
  const dd = String(t.getDate()).padStart(2, "0");
  return `${t.getFullYear()}-${mm}-${dd}`;
}

export default function CompareStocks() {
  const today = todayStr();
  const [stocks, setStocks]       = useState([]);
  const [symA, setSymA]           = useState("");
  const [symB, setSymB]           = useState("");
  const [date, setDate]           = useState(today);
  const [month, setMonth]         = useState(new Date().getMonth() + 1);
  const [day, setDay]             = useState(new Date().getDate());
  const [holdingDays, setHolding] = useState(90);
  const [minReturn, setMinReturn] = useState(12);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  useEffect(() => {
    getStocks().then((d) => {
      const list = d.stocks || [];
      setStocks(list);
      if (list.length >= 2) { setSymA(list[0].symbol); setSymB(list[1].symbol); }
      else if (list.length === 1) { setSymA(list[0].symbol); }
    });
  }, []);

  const handleCompare = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await getStockComparison(symA, symB, month, day, holdingDays, minReturn);
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const nameA = result?.name_a || symA;
  const nameB = result?.name_b || symB;

  // Merge year-by-year rows
  const mergedRows = (() => {
    if (!result) return [];
    const mapA = {}, mapB = {};
    (result.rows_a || []).forEach((r) => { mapA[r.year] = r; });
    (result.rows_b || []).forEach((r) => { mapB[r.year] = r; });
    const years = [...new Set([...Object.keys(mapA), ...Object.keys(mapB)])].sort();
    return years.map((yr) => ({
      year:    yr,
      returnA: mapA[yr]?.return_pct ?? null,
      returnB: mapB[yr]?.return_pct ?? null,
      metA:    mapA[yr]?.target_met ?? false,
      metB:    mapB[yr]?.target_met ?? false,
    }));
  })();

  // Scatter data (overlap years only)
  const scatterRows = mergedRows.filter((r) => r.returnA != null && r.returnB != null);

  return (
    <div className="tab-content">
      {/* ── Controls ── */}
      <div className="controls-row" style={{ flexWrap: "wrap", gap: 12 }}>
        <label style={{ minWidth: 200 }}>
          Stock A
          <select value={symA} onChange={(e) => setSymA(e.target.value)}>
            {stocks.map((s) => <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>)}
          </select>
        </label>
        <label style={{ minWidth: 200 }}>
          Stock B
          <select value={symB} onChange={(e) => setSymB(e.target.value)}>
            {stocks.map((s) => <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>)}
          </select>
        </label>
        <label>
          Entry Date
          <input type="date" value={date} onChange={(e) => {
            const d = new Date(e.target.value);
            setDate(e.target.value); setMonth(d.getMonth()+1); setDay(d.getDate());
          }} />
        </label>
        <label>
          Holding Days
          <input type="number" min={5} max={365} value={holdingDays}
            onChange={(e) => setHolding(parseInt(e.target.value) || 90)} />
        </label>
        <label>
          Min Return %
          <input type="number" step={0.5} value={minReturn}
            onChange={(e) => setMinReturn(parseFloat(e.target.value) || 0)} />
        </label>
        <button className="btn-primary" onClick={handleCompare}
          disabled={loading || !symA || !symB || symA === symB}>
          {loading ? "Comparing…" : "▶ Compare"}
        </button>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <>
          <div className="section-divider" />
          <h3 style={{ textAlign: "center" }}>
            {nameA} vs {nameB} · {result.win_label} · Target ≥{minReturn}%
          </h3>

          {/* ── Scorecard ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 16 }}>
            {[
              { name: nameA, s: result.summary_a, ev: result.nifty_a, dtd: result.days_to_target_a },
              { name: nameB, s: result.summary_b, ev: result.nifty_b, dtd: result.days_to_target_b },
            ].map(({ name, s, ev, dtd }) => (
              <div key={name} style={{
                background: "#f8f9fa", borderRadius: 10, padding: 16,
                border: "1px solid #e0e0e0",
              }}>
                <h4 style={{ margin: "0 0 12px", color: "#2c3e50" }}>{name}</h4>
                {s ? (
                  <div className="metrics-row" style={{ flexDirection: "column", gap: 8 }}>
                    <ScoreRow label={`Target ≥${minReturn}% Met`}
                      value={`${s.target_met_count} / ${s.total_instances} yrs`}
                      sub={`${(s.target_met_count/s.total_instances*100).toFixed(0)}% hit rate`} />
                    <ScoreRow label="Avg Return (all yrs)" value={`${s.avg_return_pct > 0 ? "+":""}${s.avg_return_pct?.toFixed(2)}%`}
                      positive={s.avg_return_pct >= 0} />
                    <ScoreRow label="Avg When Target Met"
                      value={s.avg_return_when_met != null ? `${s.avg_return_when_met > 0 ? "+":""}${s.avg_return_when_met?.toFixed(2)}%` : "Never met"}
                      positive={s.avg_return_when_met > 0} />
                    <ScoreRow label="Best / Worst"
                      value={`${s.best_return_pct > 0?"+":""}${s.best_return_pct?.toFixed(1)}% (${s.best_year})  /  ${s.worst_return_pct?.toFixed(1)}% (${s.worst_year})`} />
                    {dtd && <ScoreRow label="Avg Days to Target" value={`${dtd.avg?.toFixed(0)}d`}
                      sub={`fastest ${dtd.min}d · slowest ${dtd.max}d`} />}
                    {ev?.nifty_available && (
                      <ScoreRow label="Beat NIFTY" value={ev.beat_index_label}
                        sub={ev.avg_excess_return != null ? `Excess avg ${ev.avg_excess_return > 0 ? "+" : ""}${ev.avg_excess_return?.toFixed(2)}%` : ""} />
                    )}
                  </div>
                ) : <p style={{ color: "#888" }}>No data available.</p>}
              </div>
            ))}
          </div>

          {/* ── Correlation callout ── */}
          {result.correlation != null && (
            <div style={{
              margin: "16px 0", padding: "12px 16px",
              background: Math.abs(result.correlation) < 0.4 ? "#eafaf1" : "#fef9e7",
              borderRadius: 8, border: `1px solid ${Math.abs(result.correlation) < 0.4 ? "#2ecc71" : "#f39c12"}`,
            }}>
              <strong>Return Correlation: {result.correlation.toFixed(2)}</strong>
              {result.correlation < 0.3
                ? "  — Low correlation: these stocks move independently, good for diversification."
                : result.correlation < 0.6
                ? "  — Moderate correlation: partial overlap in outcomes."
                : "  — High correlation: these stocks tend to move together in this window."}
              {(result.div_a_wins?.length > 0 || result.div_b_wins?.length > 0) && (
                <div style={{ marginTop: 8, fontSize: 13 }}>
                  {result.div_a_wins?.length > 0 && (
                    <div>🔵 <strong>{nameA}</strong> hit target, {nameB} didn&apos;t: <strong>{result.div_a_wins.join(", ")}</strong></div>
                  )}
                  {result.div_b_wins?.length > 0 && (
                    <div>🟠 <strong>{nameB}</strong> hit target, {nameA} didn&apos;t: <strong>{result.div_b_wins.join(", ")}</strong></div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ── Grouped bar chart ── */}
          {mergedRows.length > 0 && (
            <>
              <h4 className="section-header">📊 Year-by-Year Returns</h4>
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={mergedRows} margin={{ top: 20, right: 20, bottom: 40, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="year" angle={-45} textAnchor="end" interval={0} tick={{ fontSize: 10 }} />
                  <YAxis tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v, k) => v != null ? [`${v > 0 ? "+" : ""}${v?.toFixed(2)}%`, k] : ["—", k]} />
                  <Legend />
                  <ReferenceLine y={minReturn} stroke="#e74c3c" strokeDasharray="4 4"
                    label={{ value: `Target ${minReturn}%`, position: "right", fontSize: 11 }} />
                  <ReferenceLine y={0} stroke="#555" />
                  <Bar dataKey="returnA" name={nameA} radius={[3,3,0,0]}>
                    {mergedRows.map((r, i) => (
                      <Cell key={i} fill={r.metA ? "#27ae60" : r.returnA >= 0 ? "#2980b9" : "#e74c3c"} />
                    ))}
                  </Bar>
                  <Bar dataKey="returnB" name={nameB} radius={[3,3,0,0]}>
                    {mergedRows.map((r, i) => (
                      <Cell key={i} fill={r.metB ? "#f39c12" : r.returnB >= 0 ? "#9b59b6" : "#922b21"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </>
          )}

          {/* ── Scatter correlation ── */}
          {scatterRows.length >= 3 && (
            <>
              <h4 className="section-header">📡 Return Correlation Scatter</h4>
              <p className="chart-caption">
                Each dot = one year where both stocks have data. Top-right quadrant = both hit ≥{minReturn}% target.
              </p>
              <ResponsiveContainer width="100%" height={380}>
                <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis type="number" dataKey="returnA" name={nameA}
                    label={{ value: `${nameA} Return %`, position: "insideBottom", offset: -10 }} />
                  <YAxis type="number" dataKey="returnB" name={nameB}
                    label={{ value: `${nameB} Return %`, angle: -90, position: "insideLeft" }} />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }}
                    content={({ payload }) => {
                      if (!payload?.length) return null;
                      const d = payload[0].payload;
                      return (
                        <div style={{ background: "#fff", border: "1px solid #ddd", padding: "6px 10px", borderRadius: 6, fontSize: 12 }}>
                          <strong>{d.year}</strong><br />
                          {nameA}: {d.returnA > 0 ? "+" : ""}{d.returnA?.toFixed(2)}%<br />
                          {nameB}: {d.returnB > 0 ? "+" : ""}{d.returnB?.toFixed(2)}%
                        </div>
                      );
                    }}
                  />
                  <ReferenceLine x={minReturn} stroke="#e74c3c" strokeDasharray="4 4" />
                  <ReferenceLine y={minReturn} stroke="#e74c3c" strokeDasharray="4 4" />
                  <ReferenceLine x={0} stroke="#888" />
                  <ReferenceLine y={0} stroke="#888" />
                  <Scatter data={scatterRows} fill="#2980b9">
                    {scatterRows.map((r, i) => (
                      <Cell key={i}
                        fill={r.metA && r.metB ? "#27ae60" : r.metA ? "#2980b9" : r.metB ? "#f39c12" : "#e74c3c"} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
              <p className="chart-caption" style={{ textAlign: "center" }}>
                🟢 Both hit target · 🔵 Only {nameA} · 🟠 Only {nameB} · 🔴 Neither
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}

function ScoreRow({ label, value, sub, positive }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline",
      fontSize: 13, borderBottom: "1px solid #eee", paddingBottom: 4 }}>
      <span style={{ color: "#666" }}>{label}</span>
      <div style={{ textAlign: "right" }}>
        <span style={{
          fontWeight: 600,
          color: positive === true ? "#27ae60" : positive === false ? "#e74c3c" : "#2c3e50",
        }}>{value}</span>
        {sub && <div style={{ fontSize: 11, color: "#888" }}>{sub}</div>}
      </div>
    </div>
  );
}
