import React, { useState, useEffect } from "react";
import {
  getDataStatus,
  getIndexStatus,
  triggerUpdate,
  triggerIndexUpdate,
  addCustomTicker,
} from "../api";
import "../styles/seasonal.css";

export default function DataManagement() {
  const [stockStatus, setStockStatus] = useState([]);
  const [indexStatus, setIndexStatus] = useState([]);
  const [newTicker, setNewTicker]     = useState("");
  const [newName,   setNewName]       = useState("");
  const [updating,  setUpdating]      = useState(false);
  const [message,   setMessage]       = useState(null);
  const [error,     setError]         = useState(null);

  const fetchStatus = async () => {
    try {
      const [s, i] = await Promise.all([getDataStatus(), getIndexStatus()]);
      setStockStatus(s.rows || []);
      setIndexStatus(i.rows || []);
    } catch {}
  };

  useEffect(() => { fetchStatus(); }, []);

  const handleUpdate = async () => {
    setUpdating(true); setMessage(null); setError(null);
    try {
      const data = await triggerUpdate();
      setMessage(data.message || "Update complete.");
      fetchStatus();
    } catch (e) { setError(e.message); }
    finally { setUpdating(false); }
  };

  const handleIndexUpdate = async () => {
    setUpdating(true); setMessage(null); setError(null);
    try {
      const data = await triggerIndexUpdate();
      setMessage(data.message || "Index update complete.");
      fetchStatus();
    } catch (e) { setError(e.message); }
    finally { setUpdating(false); }
  };

  const handleAddTicker = async () => {
    if (!newTicker.trim()) return;
    setUpdating(true); setMessage(null); setError(null);
    try {
      const data = await addCustomTicker(
        newTicker.trim().toUpperCase(),
        newName.trim() || newTicker.trim(),
      );
      setMessage(data.message || "Ticker added.");
      setNewTicker(""); setNewName("");
      fetchStatus();
    } catch (e) { setError(e.message); }
    finally { setUpdating(false); }
  };

  return (
    <div className="tab-content">
      <h3 className="section-header">Data Management</h3>

      {message && <div className="alert alert-success">{message}</div>}
      {error   && <div className="alert alert-error">{error}</div>}

      {/* ── Bulk Download ── */}
      <div className="dm-card">
        <div className="dm-card-title">Nifty 500 Universe</div>
        <p style={{ color: "#666", marginBottom: 12 }}>
          Downloads fresh OHLCV data for all Nifty 500 stocks from Yahoo Finance.
        </p>
        <button className="btn-primary" onClick={handleUpdate} disabled={updating}>
          {updating ? "Downloading…" : "⬇ Download / Update Stocks"}
        </button>
      </div>

      {/* ── Index Download ── */}
      <div className="dm-card">
        <div className="dm-card-title">Market Indices</div>
        <p style={{ color: "#666", marginBottom: 12 }}>
          Downloads ^NSEI (NIFTY 50) and ^NSEBANK (BANK NIFTY) for excess-return analysis.
        </p>
        <button className="btn-primary" onClick={handleIndexUpdate} disabled={updating}>
          {updating ? "Downloading…" : "⬇ Update Indices"}
        </button>
      </div>

      {/* ── Custom Ticker ── */}
      <div className="dm-card">
        <div className="dm-card-title">Add Custom Ticker</div>
        <div className="controls-row" style={{ marginTop: 8 }}>
          <label>NSE Symbol
            <input
              type="text" placeholder="e.g. TCS" value={newTicker}
              onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
              style={{ textTransform: "uppercase" }}
            />
          </label>
          <label>Display Name (optional)
            <input type="text" placeholder="e.g. Tata Consultancy" value={newName}
              onChange={(e) => setNewName(e.target.value)} />
          </label>
          <button className="btn-primary" onClick={handleAddTicker} disabled={updating || !newTicker}>
            {updating ? "Adding…" : "+ Add"}
          </button>
        </div>
      </div>

      {/* ── Status Table ── */}
      {stockStatus.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h4 className="section-header">Stock Data Status ({stockStatus.length} stocks)</h4>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  {Object.keys(stockStatus[0]).map((k) => <th key={k}>{k}</th>)}
                </tr>
              </thead>
              <tbody>
                {stockStatus.slice(0, 20).map((row, i) => (
                  <tr key={i}>
                    {Object.entries(row).map(([k, v]) => <td key={k}>{v ?? "—"}</td>)}
                  </tr>
                ))}
                {stockStatus.length > 20 && (
                  <tr><td colSpan={Object.keys(stockStatus[0]).length} style={{ textAlign: "center", color: "#999" }}>
                    … and {stockStatus.length - 20} more
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {indexStatus.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h4 className="section-header">Index Data Status</h4>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>{Object.keys(indexStatus[0]).map((k) => <th key={k}>{k}</th>)}</tr>
              </thead>
              <tbody>
                {indexStatus.map((row, i) => (
                  <tr key={i}>
                    {Object.entries(row).map(([k, v]) => <td key={k}>{v ?? "—"}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
