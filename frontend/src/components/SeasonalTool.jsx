import React, { useState } from "react";
import StockAnalysis  from "./StockAnalysis";
import SectorAnalysis from "./SectorAnalysis";
import BestWindows    from "./BestWindows";
import DeepInsights   from "./DeepInsights";
import DataManagement from "./DataManagement";
import "../styles/seasonal.css";

const TABS = [
  { id: "stock",   label: "📈 Stock Analysis" },
  { id: "sector",  label: "🏭 Sector Analysis" },
  { id: "windows", label: "🔍 Best Windows" },
  { id: "deep",    label: "🧮 Deep Insights" },
  { id: "data",    label: "⚙️ Data" },
];

/**
 * SeasonalTool — self-contained seasonal analysis widget.
 *
 * Usage in your React app:
 *   import SeasonalTool from "./components/SeasonalTool";
 *   <SeasonalTool />
 *
 * The tool reads the API base URL from:
 *   process.env.REACT_APP_API_URL  (default: http://localhost:8000)
 */
export default function SeasonalTool() {
  const [activeTab, setActiveTab] = useState("stock");

  return (
    <div className="seasonal-tool-root">
      <div className="seasonal-tool-header">
        <h2 className="seasonal-tool-title">NSE Seasonal Analysis</h2>
        <p className="seasonal-tool-subtitle">
          Historical seasonality for Nifty 500 stocks powered by local SQLite + FastAPI
        </p>
      </div>

      <nav className="seasonal-tab-nav">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`seasonal-tab-btn${activeTab === t.id ? " active" : ""}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="seasonal-tab-panel">
        {activeTab === "stock"   && <StockAnalysis />}
        {activeTab === "sector"  && <SectorAnalysis />}
        {activeTab === "windows" && <BestWindows />}
        {activeTab === "deep"    && <DeepInsights />}
        {activeTab === "data"    && <DataManagement />}
      </div>
    </div>
  );
}
