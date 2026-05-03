# TradeQuity — NSE Swing Trading Seasonal Analyser

A **fully local** Streamlit web app for analysing seasonal patterns in NSE (India) stocks.  
It downloads ~15 years of daily OHLCV data from Yahoo Finance, stores it in a local SQLite database, and lets you ask questions like:

> *"If I buy RELIANCE.NS every year on 1st April and hold for 90 days, how has that historically played out?"*

No cloud subscription, no API keys, no cost — runs entirely on your laptop.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+ (Anaconda recommended)
- Git

### 2. Clone & set up environment

```bash
git clone https://github.com/anilrk8/tradequity.git
cd tradequity

# Create a conda environment (or use pip in a venv)
conda create -n SmartAnalytics python=3.10 -y
conda activate SmartAnalytics

pip install -r requirements.txt
```

### 3. Launch the app

**Windows — double-click `restart.bat`** (recommended)  
This kills any stale Python/Streamlit processes, clears bytecode cache, and starts the app fresh on port 8501.

Or manually:
```bash
conda activate SmartAnalytics
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### 4. First-time data download

On first launch the database is empty. Go to the **🗄 Data Management** tab and click **Download / Update Now**.  
This fetches ~15 years of daily OHLCV for all 504 Nifty 500 stocks (~10–15 minutes).  
Every run after that is incremental (seconds).

---

## Project Structure

```
tradequity/
├── app.py                  # Streamlit UI — all tabs and rendering logic
├── src/
│   ├── db.py               # SQLite connection, table creation
│   ├── fetcher.py          # Downloads OHLCV from Yahoo Finance (yfinance)
│   ├── universe.py         # Nifty 50 & Nifty 500 stock lists with sectors
│   └── analyzer.py         # All analysis functions (seasonal, MAE, heatmap, etc.)
├── scripts/
│   └── build_nifty500.py   # One-time script to refresh the Nifty 500 stock list from NSE
├── data/
│   └── stocks.db           # SQLite database (git-ignored, lives only on your machine)
├── requirements.txt
├── restart.bat             # Windows: clean restart script
└── .streamlit/
    └── config.toml         # Streamlit server settings
```

---

## Architecture & Data Flow

```
Yahoo Finance (yfinance)
        │
        ▼
  src/fetcher.py          ← downloads OHLCV, writes to SQLite
        │
        ▼
  data/stocks.db          ← SQLite, two tables:
  ┌─────────────────┐       ohlcv          (stock prices)
  │  stocks         │       indices_ohlcv  (NIFTY, VIX, USDINR)
  │  ohlcv          │
  │  indices_ohlcv  │
  └─────────────────┘
        │
        ▼
  src/analyzer.py         ← pure analysis functions (no UI code)
        │
        ▼
  app.py                  ← Streamlit UI, reads from analyzer + fetcher
```

### Database tables

| Table | Contents |
|---|---|
| `stocks` | Symbol, name, sector, universe (`NIFTY50`, `NIFTY500`, `CUSTOM`) |
| `ohlcv` | Daily Open/High/Low/Close/Volume for every stock |
| `indices_ohlcv` | Same schema for market indices (^NSEI, ^NSEBANK, ^INDIAVIX, USDINR=X) |

---

## Features — Tab by Tab

### 📊 Stock Analysis

Pick any stock from the Nifty 500 universe (or a custom ticker you added), choose an **entry date** (month + day — year is ignored, it's seasonal) and a **holding period in calendar days**.

| Sub-feature | What it shows |
|---|---|
| **Seasonal Summary** | Win rate, average return, best/worst year, median return across all historical instances of that window |
| **Fan Chart** | Each year's price trajectory indexed to 100 at entry — lets you see the spread of outcomes |
| **Year-by-Year Bar** | Annual return for each year, coloured green/red. Hover for exact numbers |
| **MAE stats** | Average & worst intraday drawdown (Maximum Adverse Excursion) from entry |

### 🏭 Sector Analysis

Same seasonal window analysis but aggregated at the **sector level**.  
Useful for identifying which sectors seasonally outperform in a given window (e.g., "IT sector in November–January").

Shows:
- Average return across all stocks in the sector for the chosen window
- Win rate at sector level
- Year-by-year breakdown

### 🔍 Best Windows Screener

Give it a stock (or run across the whole universe) and it **scans all entry-date + holding-period combinations** to surface the historically strongest seasonal windows.

- Ranks by average return, win rate, or a composite score
- Filters by minimum number of years with data
- Useful for discovery: "show me HDFC Bank's historically best 3-month windows"

### 📉 Deep Insights

Four sub-tabs with more advanced analysis:

#### 📅 Monthly Heatmap
A **year × month grid** of average returns for a chosen stock.  
Green = historically positive month, red = negative.  
Immediately shows seasonal patterns at a glance.

#### 📈 Excess Return vs NIFTY
For any seasonal window, shows the stock's return **minus NIFTY 50's return** over the same period.  
A positive excess return means the stock beat the index.  
Also overlays India VIX at entry to show whether high/low volatility correlates with outcomes.  
*(Requires index data — download from Data Management first.)*

#### 🔄 Sector Rotation
A **sector × month heatmap** showing which sectors have historically been strong in each calendar month.  
Useful for top-down allocation decisions: "which sectors should I be looking at in May?"

#### 🎯 MAE & Stop-Loss (Stop-Loss Survival Analysis)
The most practical risk tab:

**Top metrics:** Avg MAE (how far the trade typically dips before recovering), Worst MAE ever, Avg MFE (best intraday peak reached).

**Stop-Loss Survival chart:**  
For every stop-loss level from -0.5% to -30% (in 0.5% steps), simulates:
- How many trades would have been stopped out
- Of those stopped, how many were actually **winners** (trades that would have recovered and closed profitable)
- **Winner Preservation %** — the key line: what % of your profitable trades survive at each stop level

> The sweet spot is the stop level where losers start getting cut (Losers Stopped rises) but Winner Preservation % is still above ~80%. This is your evidence-based stop-loss range for that stock + window combination.

**MAE by Year bar chart:** Each year's worst intraday dip, coloured by whether the trade ended profitable.

**MAE vs Final Return scatter:** Each dot is one year. Bottom-left = bad MAE, lost money. Top-left = had a big dip but recovered. Top-right = smooth winner. Ideal cluster is top-left quadrant.

### 🗄 Data Management

Three sections:

**Download / Update**  
Incremental OHLCV update for all 504 Nifty 500 stocks. First run takes ~10–15 min; subsequent runs take seconds (only new days are fetched).

**Market Indices & Macro**  
Downloads NIFTY 50 Index (`^NSEI`), NIFTY Bank (`^NSEBANK`), India VIX (`^INDIAVIX`), and USD/INR (`USDINR=X`). Required for the Excess vs NIFTY and VIX overlay features.

**Add Custom Ticker**  
Add any stock that isn't in the Nifty 500 — small caps, foreign ETFs, indices, etc.  
- Enter the **Yahoo Finance ticker** (NSE stocks end with `.NS`, e.g. `IRFC.NS`, `TATAPOWER.NS`)
- Optional: give it a friendly display name
- Click **Fetch & Add** — it validates the ticker, registers it in the DB, and downloads full history
- The ticker then appears in every stock dropdown across all tabs

---

## Source Files — What Does What

### `src/db.py`
- Defines `DB_PATH` (defaults to `data/stocks.db`, overridable via `DB_PATH` env var for deployment)
- `get_connection()` — opens SQLite with WAL mode for concurrent reads
- `init_db()` — creates `stocks`, `ohlcv`, `indices_ohlcv` tables on startup if they don't exist

### `src/universe.py`
- Hard-coded lists: `NIFTY50_STOCKS` (50 stocks), `NIFTY500_STOCKS` (504 stocks)
- Each entry: `{"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"}`
- Helper functions: `get_stocks()`, `get_symbols()`, `get_sectors()`, `get_symbol_to_name()`, `get_symbol_to_sector()`
- Updated by running `scripts/build_nifty500.py` whenever NSE rebalances the index

### `src/fetcher.py`
- `bulk_download(universe)` — incremental OHLCV download for entire universe (called by Data Management tab)
- `fetch_symbol(symbol, start, end, conn)` — downloads one ticker for a date range and upserts to DB
- `fetch_indices()` — downloads the four market indices into `indices_ohlcv`
- `fetch_custom_ticker(symbol, name)` — adds an arbitrary Yahoo Finance ticker to the DB
- `get_custom_tickers()` — returns all tickers registered as `CUSTOM` universe
- `get_data_status()`, `get_index_status()` — summary DataFrames for the Data Management status panels

### `src/analyzer.py`
All analysis is pure Python/pandas — no UI code here, fully testable in isolation.

| Function | Returns |
|---|---|
| `seasonal_analysis(symbol, month, day, holding_days)` | Summary dict: win rate, avg return, MAE stats, year-by-year rows |
| `sector_seasonal_analysis(sector, month, day, holding_days)` | Same but averaged across all stocks in the sector |
| `best_windows_for_stock(symbol, ...)` | DataFrame of ranked seasonal windows |
| `universe_screener(month, day, holding_days, ...)` | Cross-stock seasonal ranking for a fixed window |
| `monthly_return_heatmap(symbol)` | Year × month return pivot table |
| `excess_return_vs_nifty(symbol, month, day, holding_days)` | Per-year excess return + VIX at entry |
| `sector_rotation_analysis(universe)` | Sector × month avg return pivot |
| `mae_analysis(symbol, month, day, holding_days)` | Per-year MAE, MFE, final return |
| `stop_loss_survival(mae_df)` | Survival stats at each stop level from -0.5% to -30% |

### `app.py`
- ~1,400 lines — all Streamlit UI
- Every analysis tab uses `st.session_state` with `_res_` prefixed keys so results persist when the user changes other widgets (avoids Streamlit's rerun-on-interaction behaviour wiping your results)
- Helper functions: `_stock_selector()`, `_entry_inputs()`, `_window_label()`, `_no_data_warning()`
- One rendering function per section: `tab_stock_analysis()`, `tab_sector_analysis()`, `tab_best_windows()`, `tab_deep_insights()`, `tab_data_management()`

---

## Updating the Nifty 500 Stock List

NSE rebalances the Nifty 500 index periodically. To update the stock list:

```bash
conda activate SmartAnalytics
python scripts/build_nifty500.py
```

This downloads the latest NSE constituent CSV, maps industry labels to sector names, and patches `src/universe.py` in-place.

---

## Deployment (Optional)

Deployment files are included but deployment is currently on hold. The identified free path is:

1. **Supabase** — host the SQLite-equivalent PostgreSQL DB (free tier)
2. **Streamlit Community Cloud** — deploy app.py from this GitHub repo (free tier)

Environment variable `DB_PATH` can point to a remote database path if needed.

---

## Contributing

1. Clone the repo and create a branch: `git checkout -b feature/your-feature`
2. Make changes, test locally with `restart.bat`
3. Commit: `git commit -m "describe your change"`
4. Push and open a Pull Request: `git push origin feature/your-feature`

One important note: `data/stocks.db` is in `.gitignore` — the database never gets pushed to GitHub. Each collaborator maintains their own local copy and downloads data themselves via the Data Management tab.
