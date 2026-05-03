import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import {
  getStocks,
  getMonthlyHeatmap,
  getExcessReturn,
  getSectorRotation,
  getMaeAnalysis,
  getSimilarYears,
} from "../api";
import "../styles/seasonal.css";

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const HEAT_COLORS = (v) => {
  if (v == null) return "#e0e0e0";
  if (v > 10) return "#1a7a4c";
  if (v > 5)  return "#2ecc71";
  if (v > 0)  return "#a9dfbf";
  if (v > -5) return "#f5b7b1";
  if (v > -10) return "#e74c3c";
  return "#922b21";
};
const textForBg = (v) => (v != null && v < -5) ? "#fff" : undefined;

export default function DeepInsights() {
  const [activeTab, setActiveTab] = useState("heatmap");
  const [stocks, setStocks]       = useState([]);
  const [symbol, setSymbol]       = useState("");
  const [holdingDays, setHolding] = useState(90);

  const today = new Date();
  const [date, setDate]   = useState(`${today.getFullYear()}-04-01`);
  const [month, setMonth] = useState(4);
  const [day, setDay]     = useState(1);

  const [heatmapData,  setHeatmapData]  = useState(null);
  const [excessData,   setExcessData]   = useState(null);
  const [rotationData, setRotationData] = useState(null);
  const [maeData,      setMaeData]      = useState(null);
  const [similarData,  setSimilarData]  = useState(null);

  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    getStocks().then((d) => {
      setStocks(d.stocks || []);
      if (d.stocks?.length) setSymbol(d.stocks[0].symbol);
    });
  }, []);

  const run = async () => {
    setLoading(true); setError(null);
    try {
      if (activeTab === "heatmap") {
        const d = await getMonthlyHeatmap(symbol);
        setHeatmapData(d);
      } else if (activeTab === "excess") {
        const d = await getExcessReturn(symbol, month, day, holdingDays);
        setExcessData(d);
      } else if (activeTab === "rotation") {
        const d = await getSectorRotation();
        setRotationData(d);
      } else if (activeTab === "mae") {
        const d = await getMaeAnalysis(symbol, month, day, holdingDays);
        setMaeData(d);
      } else if (activeTab === "similar") {
        const d = await getSimilarYears(symbol, month, day, holdingDays);
        setSimilarData(d);
      }
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const subTabs = [
    { id: "heatmap",  label: "Monthly Heatmap" },
    { id: "excess",   label: "Excess vs NIFTY" },
    { id: "rotation", label: "Sector Rotation" },
    { id: "mae",      label: "MAE & Stop-Loss" },
    { id: "similar",  label: "Similar Years" },
  ];

  const needsDate = ["excess", "mae", "similar"];
  const needsStock = ["heatmap", "excess", "mae", "similar"];

  return (
    <div className="tab-content">
      <div className="tab-bar">
        {subTabs.map((t) => (
          <button
            key={t.id}
            className={`tab-btn${activeTab === t.id ? " active" : ""}`}
            onClick={() => setActiveTab(t.id)}
          >{t.label}</button>
        ))}
      </div>

      <div className="controls-row">
        {needsStock.includes(activeTab) && (
          <label>Stock
            <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
              {stocks.map((s) => <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>)}
            </select>
          </label>
        )}
        {needsDate.includes(activeTab) && (
          <label>Entry Date
            <input type="date" value={date} onChange={(e) => {
              const d = new Date(e.target.value);
              setDate(e.target.value); setMonth(d.getMonth()+1); setDay(d.getDate());
            }} />
          </label>
        )}
        {needsDate.includes(activeTab) && (
          <label>Holding Days
            <input type="number" min={5} max={365} value={holdingDays}
              onChange={(e) => setHolding(parseInt(e.target.value) || 90)} />
          </label>
        )}
        <button className="btn-primary" onClick={run} disabled={loading}>
          {loading ? "Loading…" : "Run →"}
        </button>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {/* ── Monthly Heatmap ── */}
      {activeTab === "heatmap" && heatmapData && (
        <HeatmapView data={heatmapData} />
      )}

      {/* ── Excess Return vs NIFTY ── */}
      {activeTab === "excess" && excessData?.rows?.length > 0 && (
        <div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={excessData.rows} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="Year" />
              <YAxis />
              <Tooltip formatter={(v) => `${v > 0 ? "+" : ""}${v?.toFixed(2)}%`} />
              <Legend />
              <Bar dataKey="Excess %" name="Excess Return vs NIFTY" radius={[3,3,0,0]}>
                {excessData.rows.map((entry, i) => (
                  <Cell key={i} fill={(entry["Excess %"] ?? 0) >= 0 ? "#27ae60" : "#e74c3c"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <ResultTable rows={excessData.rows} />
        </div>
      )}

      {/* ── Sector Rotation ── */}
      {activeTab === "rotation" && rotationData && (
        <RotationView data={rotationData} />
      )}

      {/* ── MAE & Stop-Loss ── */}
      {activeTab === "mae" && maeData && (
        <MaeView data={maeData} />
      )}

      {/* ── Similar Years ── */}
      {activeTab === "similar" && similarData && (
        <SimilarView data={similarData} />
      )}
    </div>
  );
}

/* ────────────── SUB-VIEWS ────────────── */

function HeatmapView({ data }) {
  if (!data?.rows?.length) return <p>No data.</p>;
  const years = [...new Set(data.rows.map((r) => r.Year))].sort();
  const pivot = {};
  data.rows.forEach((r) => { pivot[r.Year] ??= {}; pivot[r.Year][r.Month] = r["Avg Return %"]; });

  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data-table" style={{ minWidth: 600 }}>
        <thead>
          <tr>
            <th>Year</th>
            {MONTHS.map((m) => <th key={m}>{m}</th>)}
          </tr>
        </thead>
        <tbody>
          {years.map((yr) => (
            <tr key={yr}>
              <td><strong>{yr}</strong></td>
              {MONTHS.map((_, mi) => {
                const v = pivot[yr]?.[mi + 1];
                return (
                  <td key={mi} style={{
                    background: HEAT_COLORS(v),
                    color: textForBg(v),
                    textAlign: "right",
                    fontSize: 12,
                    whiteSpace: "nowrap",
                  }}>
                    {v != null ? `${v > 0 ? "+" : ""}${v.toFixed(1)}%` : "—"}
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

function RotationView({ data }) {
  if (!data?.rows?.length) return <p>No data.</p>;
  const sectors = [...new Set(data.rows.map((r) => r.Sector))];
  const months  = [...new Set(data.rows.map((r) => r.Month))].sort((a, b) => a - b);
  const pivot   = {};
  data.rows.forEach((r) => { pivot[r.Sector] ??= {}; pivot[r.Sector][r.Month] = r["Avg Return %"]; });

  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data-table" style={{ minWidth: 700 }}>
        <thead>
          <tr>
            <th>Sector</th>
            {months.map((m) => <th key={m}>{MONTHS[m - 1]}</th>)}
          </tr>
        </thead>
        <tbody>
          {sectors.map((sec) => (
            <tr key={sec}>
              <td style={{ whiteSpace: "nowrap", fontWeight: 500 }}>{sec}</td>
              {months.map((m) => {
                const v = pivot[sec]?.[m];
                return (
                  <td key={m} style={{
                    background: HEAT_COLORS(v),
                    color: textForBg(v),
                    textAlign: "right",
                    fontSize: 12,
                  }}>
                    {v != null ? `${v > 0 ? "+" : ""}${v.toFixed(1)}%` : "—"}
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

function MaeView({ data }) {
  return (
    <div>
      {/* Stop-loss survival line chart */}
      {data.survival?.length > 0 && (
        <>
          <h4 className="section-header">Stop-Loss Survival Curves</h4>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data.survival} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="Stop Loss %" tickFormatter={(v) => `${v}%`} />
              <YAxis tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
              <Tooltip formatter={(v) => `${v?.toFixed(1)}%`} />
              <Legend />
              <Line
                type="monotone" dataKey="Survival Rate %"
                stroke="#2980b9" strokeWidth={2} dot={false}
              />
              <Line
                type="monotone" dataKey="Winner Preservation %"
                stroke="#27ae60" strokeWidth={2} dot={false} strokeDasharray="5 5"
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      {/* MAE bar chart */}
      {data.mae_rows?.length > 0 && (
        <>
          <h4 className="section-header">MAE by Year</h4>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.mae_rows} margin={{ top: 10, right: 20, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="Year" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="MAE %" fill="#e74c3c" radius={[3,3,0,0]}
                name="MAE % (deepest drawdown during trade)" />
            </BarChart>
          </ResponsiveContainer>
          <ResultTable rows={data.mae_rows} />
        </>
      )}
    </div>
  );
}

function SimilarView({ data }) {
  const { current_features = {}, similar_years = [] } = data;

  const radarData = Object.entries(current_features).map(([k, v]) => ({
    metric: k, value: v,
  }));

  return (
    <div>
      {radarData.length > 0 && (
        <>
          <h4 className="section-header">Today&apos;s Market Features (z-score)</h4>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={80}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis />
              <Radar name="Features" dataKey="value" stroke="#2980b9" fill="#2980b9" fillOpacity={0.35} />
            </RadarChart>
          </ResponsiveContainer>
        </>
      )}

      {similar_years.length > 0 && (
        <>
          <h4 className="section-header">Similar Historical Years</h4>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={similar_years} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="return_pct" name="Return %" radius={[3,3,0,0]}>
                {similar_years.map((entry, i) => (
                  <Cell key={i} fill={(entry.return_pct ?? 0) >= 0 ? "#27ae60" : "#e74c3c"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <ResultTable rows={similar_years.map((r) => ({
            Year: r.year,
            Similarity: r.similarity?.toFixed(3),
            "Return %": r.return_pct?.toFixed(2),
          }))} />
        </>
      )}
    </div>
  );
}

function ResultTable({ rows }) {
  if (!rows?.length) return null;
  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>{Object.keys(rows[0]).map((k) => <th key={k}>{k}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {Object.entries(row).map(([k, v]) => (
                <td key={k} style={{
                  color: (k.includes("%") || k.includes("Return")) && v != null
                    ? parseFloat(v) >= 0 ? "#27ae60" : "#e74c3c" : undefined,
                }}>
                  {v ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
