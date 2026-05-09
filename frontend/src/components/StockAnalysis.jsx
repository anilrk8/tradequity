import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, LineChart, Line, Legend, AreaChart, Area, Cell,
} from "recharts";
import { getStocks, getSeasonalAnalysis, getDaysToTarget, streamAiCommentary } from "../api";
import "../styles/seasonal.css";

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function EntryInputs({ prefix, value, onChange }) {
  return (
    <div className="entry-inputs">
      <label>
        Entry Date
        <input
          type="date"
          value={value.date}
          onChange={(e) => {
            const d = new Date(e.target.value);
            onChange({ ...value, date: e.target.value, month: d.getMonth() + 1, day: d.getDate() });
          }}
        />
      </label>
      <label>
        Holding Days
        <input
          type="number" min={5} max={365}
          value={value.holdingDays}
          onChange={(e) => onChange({ ...value, holdingDays: parseInt(e.target.value) || 90 })}
        />
      </label>
      <label>
        Min Return %
        <input
          type="number" step={0.5}
          value={value.minReturn}
          onChange={(e) => onChange({ ...value, minReturn: parseFloat(e.target.value) || 0 })}
        />
      </label>
    </div>
  );
}

const CustomBarLabel = ({ x, y, width, value }) => {
  if (!value) return null;
  const formatted = `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
  return (
    <text x={x + width / 2} y={value >= 0 ? y - 4 : y + 14} textAnchor="middle" fontSize={10} fill="#555">
      {formatted}
    </text>
  );
};

export default function StockAnalysis() {
  const today = new Date();
  const mm = String(today.getMonth() + 1).padStart(2, "0");
  const dd = String(today.getDate()).padStart(2, "0");
  const defaultDate = `${today.getFullYear()}-${mm}-${dd}`;

  const [stocks, setStocks]         = useState([]);
  const [symbol, setSymbol]         = useState("");
  const [inputs, setInputs]         = useState({ date: defaultDate, month: today.getMonth() + 1, day: today.getDate(), holdingDays: 90, minReturn: 0 });
  const [result, setResult]         = useState(null);
  const [daysData, setDaysData]     = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [activeTab, setActiveTab]   = useState("fan");
  const [commentary, setCommentary] = useState("");
  const [aiLoading, setAiLoading]   = useState(false);

  useEffect(() => {
    getStocks().then((d) => {
      setStocks(d.stocks || []);
      if (d.stocks?.length) setSymbol(d.stocks[0].symbol);
    });
  }, []);

  const handleAnalyse = async () => {
    setLoading(true); setError(null); setResult(null); setDaysData(null); setCommentary("");
    try {
      const data = await getSeasonalAnalysis(symbol, inputs.month, inputs.day, inputs.holdingDays, inputs.minReturn);
      setResult(data);
      // Also fetch days-to-target in parallel (non-blocking)
      getDaysToTarget(symbol, inputs.month, inputs.day, inputs.holdingDays, inputs.minReturn)
        .then(setDaysData).catch(() => {});
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleAiCommentary = () => {
    if (!result) return;
    setAiLoading(true); setCommentary("");
    const sym_name = stocks.find((s) => s.symbol === symbol)?.name || symbol;
    const win_label = `${String(inputs.day).padStart(2,"0")}-${MONTHS[inputs.month-1]} → ${inputs.holdingDays} days`;
    streamAiCommentary(
      sym_name, win_label, result.summary, null, null,
      (chunk) => setCommentary((prev) => prev + chunk),
      () => setAiLoading(false),
      (err) => { setError(err); setAiLoading(false); }
    );
  };

  // Build bar chart data
  const barData = result?.rows?.map((r) => ({
    year: String(r.year),
    return: r.return_pct,
    fill: r.target_met ? "#27ae60" : r.return_pct >= 0 ? "#e67e22" : "#e74c3c",
  })) || [];

  // Build fan chart data — merge all year series by day index
  const fanData = (() => {
    if (!result?.fan_series) return [];
    const series = result.fan_series;
    const years  = Object.keys(series);
    const maxLen = Math.max(...years.map((y) => series[y].length));
    return Array.from({ length: maxLen }, (_, i) => {
      const point = { day: i };
      years.forEach((y) => { if (series[y][i]) point[y] = series[y][i].value; });
      return point;
    });
  })();

  const summary = result?.summary;

  return (
    <div className="tab-content">
      <div className="controls-row">
        <div className="control-group">
          <label>Stock
            <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
              {stocks.map((s) => (
                <option key={s.symbol} value={s.symbol}>{s.name} ({s.symbol})</option>
              ))}
            </select>
          </label>
        </div>
        <EntryInputs value={inputs} onChange={setInputs} />
        <button className="btn-primary" onClick={handleAnalyse} disabled={loading || !symbol}>
          {loading ? "Analysing…" : "Analyse →"}
        </button>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {summary?.low_sample_warning && (
        <div className="alert alert-error">
          ⚠ <strong>Very limited history</strong> — only <strong>{summary.total_instances} completed window(s)</strong> found.
          This stock may be recently listed. Results have low statistical confidence — treat with caution.
        </div>
      )}

      {summary && (
        <>
          <div className="metrics-row">
            <div className="metric-card">
              <div className="metric-value">{summary.total_instances}</div>
              <div className="metric-label">Years Analysed</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: summary.avg_return_pct >= 0 ? "#27ae60" : "#e74c3c" }}>
                {summary.avg_return_pct > 0 ? "+" : ""}{summary.avg_return_pct?.toFixed(2)}%
              </div>
              <div className="metric-label">Avg Return</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{summary.win_rate_pct?.toFixed(0)}%</div>
              <div className="metric-label">Win Rate</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: "#27ae60" }}>
                +{summary.best_return_pct?.toFixed(2)}% ({summary.best_year})
              </div>
              <div className="metric-label">Best Year</div>
            </div>
            <div className="metric-card">
              <div className="metric-value" style={{ color: "#e74c3c" }}>
                {summary.worst_return_pct?.toFixed(2)}% ({summary.worst_year})
              </div>
              <div className="metric-label">Worst Year</div>
            </div>
          </div>

          <div className="tab-bar">
            {[["fan","📈 Price Trend"],["bar","📊 Year-by-Year"],["days","⏱ Days to Target"],["table","🗃 Raw Data"]].map(([id, label]) => (
              <button key={id} className={`tab-btn${activeTab === id ? " active" : ""}`} onClick={() => setActiveTab(id)}>
                {label}
              </button>
            ))}
          </div>

          {activeTab === "fan" && (
            <div className="chart-container">
              <p className="chart-caption">Each line = one year's price trajectory indexed to 100 at entry.</p>
              <ResponsiveContainer width="100%" height={380}>
                <LineChart data={fanData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="day" label={{ value: "Trading Days", position: "insideBottom", offset: -5 }} />
                  <YAxis />
                  <Tooltip formatter={(v) => [`${v?.toFixed(2)}`, ""]} />
                  {result && Object.keys(result.fan_series).map((year) => {
                    const isWin = result.rows.find((r) => String(r.year) === year)?.target_met;
                    return (
                      <Line key={year} type="monotone" dataKey={year}
                        stroke={isWin ? "#27ae60" : "#e74c3c"} dot={false}
                        strokeWidth={1} strokeOpacity={0.6} />
                    );
                  })}
                  <ReferenceLine y={100} stroke="#888" strokeDasharray="4 4" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeTab === "bar" && (
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={barData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="year" angle={-45} textAnchor="end" interval={0} tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v) => [`${v > 0 ? "+" : ""}${v?.toFixed(2)}%`, "Return"]} />
                  <ReferenceLine y={0} stroke="#888" />
                  <Bar dataKey="return" label={<CustomBarLabel />} radius={[3,3,0,0]}>
                    {barData.map((entry, i) => (
                      <rect key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeTab === "days" && (
            <div className="chart-container">
              {daysData && daysData.years_met > 0 ? (
                <>
                  <div className="metrics-row" style={{ marginBottom: 12 }}>
                    <div className="metric-card">
                      <div className="metric-value">{daysData.avg_days?.toFixed(0)} days</div>
                      <div className="metric-label">Avg Days to Hit Target</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value" style={{ color: "#27ae60" }}>{daysData.min_days} days</div>
                      <div className="metric-label">Fastest Hit</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value" style={{ color: "#e67e22" }}>{daysData.max_days} days</div>
                      <div className="metric-label">Slowest Hit</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-value">{daysData.years_met} / {daysData.total_years}</div>
                      <div className="metric-label">Years Target Met</div>
                    </div>
                  </div>
                  <p className="chart-caption">
                    Green = at or below average speed · Orange = slower than average.
                    Only years where target was ultimately met are shown.
                  </p>
                  <ResponsiveContainer width="100%" height={340}>
                    <BarChart data={daysData.rows} margin={{ top: 20, right: 20, bottom: 30, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                      <XAxis dataKey="year" angle={-45} textAnchor="end" interval={0} tick={{ fontSize: 11 }} />
                      <YAxis label={{ value: "Days from Entry", angle: -90, position: "insideLeft", offset: 10 }} />
                      <Tooltip formatter={(v) => [`${v} days`, "Days to Hit Target"]} />
                      <ReferenceLine y={daysData.avg_days} stroke="#2980b9" strokeDasharray="5 5"
                        label={{ value: `Avg ${daysData.avg_days?.toFixed(0)}d`, position: "right", fontSize: 11 }} />
                      <Bar dataKey="days_to_target" radius={[3,3,0,0]}>
                        {daysData.rows.map((r, i) => (
                          <rect key={i} fill={r.days_to_target <= daysData.avg_days ? "#27ae60" : "#e67e22"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </>
              ) : (
                <p className="chart-caption">
                  {daysData ? `Target of ${inputs.minReturn}% was never met — no days-to-target data.` : "Run analysis to see days-to-target."}
                </p>
              )}
            </div>
          )}

          {activeTab === "table" && (
            <div className="table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Year</th><th>Entry</th><th>Exit</th>
                    <th>Trading Days</th><th>Return %</th><th>Target Met</th>
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((r) => (
                    <tr key={r.year}>
                      <td>{r.year}</td>
                      <td>{r.start_date}</td>
                      <td>{r.end_date}</td>
                      <td>{r.trading_days}</td>
                      <td style={{ color: r.return_pct >= 0 ? "#27ae60" : "#e74c3c", fontWeight: 600 }}>
                        {r.return_pct > 0 ? "+" : ""}{r.return_pct?.toFixed(2)}%
                      </td>
                      <td>{r.target_met ? "✓" : "✗"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="ai-section">
            <button className="btn-ai" onClick={handleAiCommentary} disabled={aiLoading}>
              {aiLoading ? "Mistral thinking…" : "✨ Generate AI Commentary"}
            </button>
            {commentary && (
              <blockquote className="ai-commentary">
                {commentary}{aiLoading && <span className="cursor">▌</span>}
              </blockquote>
            )}
          </div>
        </>
      )}
    </div>
  );
}
