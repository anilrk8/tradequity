"""
Build the NIFTY500_STOCKS list by downloading the official NSE constituent CSV.

Usage:
    python scripts/build_nifty500.py

Output:
    Prints a Python list literal that can be pasted into src/universe.py,
    AND automatically writes it into universe.py in-place.
"""

import io
import re
import sys
import textwrap
from pathlib import Path

import requests
import pandas as pd

# ─── NSE source ───────────────────────────────────────────────────────────────
# NSE publishes an updated CSV of index constituents at this URL.
CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}

# ─── Industry → Sector mapping ────────────────────────────────────────────────
# NSE uses free-form industry strings. Map them to the same compact sector labels
# already used in NIFTY50_STOCKS.
INDUSTRY_MAP: list[tuple[str, str]] = [
    # Exact NSE industry strings first (longest/most-specific matches first)
    ("FAST MOVING CONSUMER GOODS",      "FMCG"),
    ("CONSUMER SERVICES",               "Consumer Services"),
    ("CONSUMER DURABLES",               "Consumer Goods"),
    ("OIL GAS & CONSUMABLE FUELS",      "Oil & Gas"),
    ("OIL GAS",                         "Oil & Gas"),
    ("AUTOMOBILE AND AUTO COMPONENTS",  "Automobile"),
    ("CONSTRUCTION MATERIALS",          "Cement"),
    ("MEDIA ENTERTAINMENT",             "Media & Entertainment"),
    ("MEDIA & ENTERTAINMENT",           "Media & Entertainment"),
    # Substring fallbacks
    ("BANK",                    "Banking"),
    ("INSURANCE",               "Insurance"),
    ("HOUSING FINANCE",         "Financial Services"),
    ("NBFC",                    "Financial Services"),
    ("FINANCIAL SERVICES",      "Financial Services"),
    ("FINANCE",                 "Financial Services"),
    ("BROKERAGE",               "Financial Services"),
    ("ASSET MANAGEMENT",        "Financial Services"),
    ("STOCK EXCHANGE",          "Financial Services"),
    ("INFORMATION TECHNOLOGY",  "IT"),
    ("SOFTWARE",                "IT"),
    ("COMPUTER",                "IT"),
    ("IT ",                     "IT"),
    ("FMCG",                    "FMCG"),
    ("PERSONAL CARE",           "FMCG"),
    ("FOOD",                    "FMCG"),
    ("BEVERAGES",               "FMCG"),
    ("TOBACCO",                 "FMCG"),
    ("HOUSEHOLD",               "FMCG"),
    ("RETAILING",               "Retail"),
    ("RETAIL",                  "Retail"),
    ("PHARMA",                  "Pharma"),
    ("DRUG",                    "Pharma"),
    ("HOSPITAL",                "Healthcare"),
    ("HEALTHCARE",              "Healthcare"),
    ("DIAGNOSTICS",             "Healthcare"),
    ("MEDICAL",                 "Healthcare"),
    ("AUTO",                    "Automobile"),
    ("TYRE",                    "Automobile"),
    ("OIL & GAS",               "Oil & Gas"),
    ("PETROLEUM",               "Oil & Gas"),
    ("REFINERIES",              "Oil & Gas"),
    ("GAS TRANSMISSION",        "Oil & Gas"),
    ("GAS",                     "Oil & Gas"),
    ("CONSUMABLE FUELS",        "Oil & Gas"),
    ("POWER",                   "Power"),
    ("RENEWABLE",               "Power"),
    ("SOLAR",                   "Power"),
    ("WIND",                    "Power"),
    ("ENERGY",                  "Power"),
    ("TELECOM",                 "Telecom"),
    ("STEEL",                   "Metals & Mining"),
    ("METALS",                  "Metals & Mining"),
    ("MINING",                  "Metals & Mining"),
    ("ALUMINIUM",               "Metals & Mining"),
    ("COPPER",                  "Metals & Mining"),
    ("IRON",                    "Metals & Mining"),
    ("CEMENT",                  "Cement"),
    ("REAL ESTATE",             "Real Estate"),
    ("REALTY",                  "Real Estate"),
    ("CONSTRUCTION",            "Infrastructure"),
    ("ENGINEERING",             "Infrastructure"),
    ("INFRASTRUCTURE",          "Infrastructure"),
    ("ROADS",                   "Infrastructure"),
    ("PORTS",                   "Infrastructure"),
    ("AIRPORT",                 "Infrastructure"),
    ("CAPITAL GOODS",           "Capital Goods"),
    ("INDUSTRIAL",              "Capital Goods"),
    ("MACHINERY",               "Capital Goods"),
    ("ELECTRIC EQUIPMENT",      "Capital Goods"),
    ("CHEMICAL",                "Chemicals"),
    ("SPECIALTY CHEMICAL",      "Chemicals"),
    ("FERTILISER",              "Fertilizers"),
    ("FERTILIZER",              "Fertilizers"),
    ("AGRO",                    "Agro"),
    ("AGRICULTURE",             "Agro"),
    ("TEXTILE",                 "Textiles"),
    ("APPAREL",                 "Textiles"),
    ("PAPER",                   "Paper & Packaging"),
    ("PACKAGING",               "Paper & Packaging"),
    ("GLASS",                   "Paper & Packaging"),
    ("LOGISTICS",               "Logistics"),
    ("SHIPPING",                "Logistics"),
    ("AVIATION",                "Aviation"),
    ("AIRLINE",                 "Aviation"),
    ("HOTEL",                   "Hospitality"),
    ("HOSPITALITY",             "Hospitality"),
    ("TOURISM",                 "Hospitality"),
    ("SUGAR",                   "Sugar"),
    ("SERVICES",                "Services"),
    ("CONSUMER",                "Consumer Goods"),
    ("TRADING",                 "Trading"),
    ("DIVERSIFIED",             "Diversified"),
]


def map_sector(industry: str) -> str:
    ind_upper = industry.upper().strip()
    for pattern, sector in INDUSTRY_MAP:
        if pattern in ind_upper:
            return sector
    return "Others"


def download_csv() -> pd.DataFrame:
    print(f"Downloading Nifty 500 constituent list from NSE…")
    session = requests.Session()

    # Warm-up: hit the NSE homepage to get cookies
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=15)
    except Exception:
        pass  # proceed even if warm-up fails

    resp = session.get(CSV_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    print(f"  Retrieved {len(resp.content):,} bytes  (status {resp.status_code})")

    df = pd.read_csv(io.StringIO(resp.text))
    print(f"  Parsed {len(df)} rows, columns: {list(df.columns)}")
    return df


def build_stock_list(df: pd.DataFrame) -> list[dict]:
    # NSE columns are typically: Company Name, Industry, Symbol, Series, ISIN Code
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Find the right column names
    col_name   = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None)
    col_symbol = next((c for c in df.columns if "symbol" in c.lower()), None)
    col_ind    = next((c for c in df.columns if "industry" in c.lower() or "sector" in c.lower()), None)

    if not all([col_name, col_symbol]):
        print(f"ERROR: could not identify required columns. Available: {list(df.columns)}")
        sys.exit(1)

    stocks = []
    for _, row in df.iterrows():
        symbol   = str(row[col_symbol]).strip() + ".NS"
        name     = str(row[col_name]).strip()
        industry = str(row[col_ind]).strip() if col_ind else ""
        sector   = map_sector(industry)
        stocks.append({"symbol": symbol, "name": name, "sector": sector})

    # Sort by symbol for stable ordering
    stocks.sort(key=lambda x: x["symbol"])
    return stocks


def format_list(stocks: list[dict]) -> str:
    lines = []
    max_sym  = max(len(s["symbol"]) for s in stocks)
    max_name = max(len(s["name"])   for s in stocks)
    for s in stocks:
        sym  = f'"{s["symbol"]}",'
        name = f'"{s["name"]}",'
        sect = f'"{s["sector"]}"'
        lines.append(
            f'    {{"symbol": {sym:<{max_sym+3}} "name": {name:<{max_name+3}} "sector": {sect}}},'
        )
    return "[\n" + "\n".join(lines) + "\n]"


def patch_universe_py(stocks: list[dict]) -> None:
    universe_path = Path(__file__).parent.parent / "src" / "universe.py"
    src = universe_path.read_text(encoding="utf-8")

    # Replace the NIFTY500_STOCKS placeholder / existing value
    new_list_str = format_list(stocks)
    pattern = r"NIFTY500_STOCKS\s*(?::\s*list\[dict\])?\s*=\s*\[.*?\]"
    replacement = f"NIFTY500_STOCKS: list[dict] = {new_list_str}"

    new_src, n = re.subn(pattern, replacement, src, flags=re.DOTALL)
    if n == 0:
        print("ERROR: Could not find NIFTY500_STOCKS in universe.py — patch failed.")
        sys.exit(1)

    universe_path.write_text(new_src, encoding="utf-8")
    print(f"\nSuccessfully patched src/universe.py with {len(stocks)} stocks.")


if __name__ == "__main__":
    df     = download_csv()
    stocks = build_stock_list(df)

    print(f"\nMapped {len(stocks)} stocks across sectors:")
    sector_counts: dict[str, int] = {}
    for s in stocks:
        sector_counts[s["sector"]] = sector_counts.get(s["sector"], 0) + 1
    for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>3}  {sec}")

    patch_universe_py(stocks)
    print("\nDone. Set UNIVERSE = \"NIFTY500\" in app.py to activate.")
