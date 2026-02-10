"""Shared constants for the SPY factor-investing dashboard."""

from pathlib import Path

# ── Directories ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # …/SPY/
DASHBOARD_DIR = BASE_DIR / "dashboard"

# ── CSV / JSON paths ───────────────────────────────────────────────────────
ALL_FUNDAMENTALS_CSV  = BASE_DIR / "all_fundamentals_ttm.csv"
FULL_FACTORS_CSV      = BASE_DIR / "full_factors_table.csv"
SCORED_FACTORS_CSV    = BASE_DIR / "scored_factors.csv"
TICKER_SECTOR_CSV     = BASE_DIR / "ticker_sector_map.csv"
BACKTEST_METRICS_JSON = BASE_DIR / "backtest_metrics.json"

# ── Ticker universe (deduplicated, sorted) ─────────────────────────────────
SP500_TICKERS = sorted(set([
    "MSFT","AAPL","NVDA","AMZN","AVGO","TSLA","GOOGL","APP","MO",
    "JPM","V","LLY","NFLX","XOM","MA","COST","WMT","PG","HD","BMY",
    "JNJ","ABBV","BAC","UNH","CRM","KO","ORCL","PM","WFC","INTC",
    "CSCO","IBM","CVX","GE","ABT","MCD","NOW","ACN","DIS","COF","PH",
    "MRK","UBER","T","GS","INTU","AMD","VZ","PEP","BKNG","CEG","CVS",
    "RTX","ADBE","TXN","CAT","AXP","QCOM","PGR","TMO","SPGI","MS",
    "BA","BSX","NEE","TJX","SCHW","AMGN","HON","C","AMAT","NEM",
    "UNP","SYK","CMCSA","ETN","LOW","PFE","GILD","DE","DHR","LMT",
    "ADP","COP","GEV","TMUS","ADI","MMC","LRCX","MDT","HCA",
    "MU","CB","KLAC","APH","ANET","ICE","SBUX","MCK",
]))

# ── IBKR connection ────────────────────────────────────────────────────────
IBKR_HOST     = "127.0.0.1"
IBKR_PORT     = 7496
IBKR_CLIENT_DATA     = 2    # clientId for Dataset Construction fetches
IBKR_CLIENT_BACKTEST = 3    # clientId for Backtest fetches
IBKR_CLIENT_VIX      = 5    # clientId for VIX fetches

# ── SEC EDGAR ──────────────────────────────────────────────────────────────
SEC_USER_AGENT = "kampongpiggg@gmail.com"
SEC_API_KEY    = "fc6b859f477a845e0b2a214bfc4cddf78c86f92d5d764845367b6484e7b60d41"

# ── Strategy parameters ────────────────────────────────────────────────────
REPORT_LAG_DAYS    = 90
MOM_WINDOW_DAYS    = 126
TOP_DECILE         = 10
STALE_THRESHOLD_DAYS = 35
START_DATE         = "2010-01-01"
