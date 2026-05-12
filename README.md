# SPY Factor Investing Strategy

A quantitative, factor-based stock selection strategy applied to 100 large-cap S&P 500 stocks. The system scores stocks monthly on Value, Quality, and Momentum factors, selects the top decile (10 stocks), and holds them in an equal-weight portfolio rebalanced monthly. A Streamlit dashboard displays the current picks and monitors strategy health.

## Strategy Overview

**Universe:** 100 large-cap S&P 500 stocks (deduplicated).

**Factors scored each month:**

| Category | Metrics | Standardization |
|---|---|---|
| Value | book_to_price, EV/EBIT (inverted), debt_to_equity (inverted) | Sector-neutral z-scores |
| Quality | gross_margin, operating_margin, ROE | Sector-neutral z-scores |
| Momentum | 6-month price momentum, On-Balance Volume | Global cross-sectional z-scores |

**Scoring pipeline:**
1. Winsorize raw metrics at 1st/99th percentiles within (date, sector) groups
2. Compute z-scores (sector-neutral for Value/Quality, global for Momentum)
3. Category score = mean of z-scored components within each category
4. Alpha score = equal-weight average of 3 category scores
5. Rank into deciles; select top decile (decile 10) — the 10 strongest stocks

**Portfolio construction:** Equal-weight (10% each), rebalanced on the last trading day of each month.

**Backtest results (Jan 2011 – Jan 2026, 180 months):**
- Annualized return: 27.4%
- Sharpe ratio: 1.20
- Max drawdown: -34.6%
- Annualized volatility: 22.2%
- Monthly win rate: 70.6%
- Statistical significance: t-stat = 5.31, p-value = 1.62e-07

## Data Sources

- **Fundamentals:** SEC EDGAR XBRL API (quarterly filings, TTM calculations)
- **Prices:** Interactive Brokers TWS API (15 years daily OHLCV)
- **Sectors:** Static `ticker_sector_map.csv` (100 tickers mapped to GICS sectors)

## File Structure

```
SPY/
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   ├── update_data.py          # CLI script to refresh data + picks monthly
│   ├── config.py               # Shared constants (paths, tickers, API keys, params)
│   ├── equity_curve.py         # Live portfolio tracking from trade snapshots
│   ├── cluster_buys.py         # OpenInsider scraper for insider cluster buys
│   ├── market_sentiment.py     # VIX and Fear & Greed Index fetchers
│   └── requirements.txt        # Python dependencies
├── backtest_metrics.json       # Dashboard data: metrics, current holdings, monthly picks
├── scored_factors.csv          # Full scored factor table (~181MB, not read by dashboard)
├── full_factors_table.csv      # Raw factor table (~52MB)
├── all_fundamentals_ttm.csv    # EDGAR fundamentals (~1.1MB)
├── ticker_sector_map.csv       # Ticker → sector mapping (100 rows)
├── Dataset Construction.ipynb  # Original notebook: data fetching + factor computation
└── Factor-Based Portfolio Backtest.ipynb  # Original notebook: scoring + backtest + Monte Carlo
```

## Quick Start

### 1. Install dependencies
```bash
cd SPY/dashboard
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Works immediately using the bootstrapped `backtest_metrics.json` — no IBKR connection needed.

### 3. Monthly data update (requires IBKR TWS running on localhost:7496)
```bash
python update_data.py          # Monthly: EDGAR + IBKR + rescore + update picks
python update_data.py --full   # Annual: same + full backtest to regenerate metrics
```

## Dashboard Sections

1. **Portfolio Performance** — Tracks live portfolio returns with an equity curve vs SPY benchmark, plus expected performance metrics from the backtest (annualized return, Sharpe, max drawdown, volatility, monthly win rate)
2. **Current Picks** — Top-decile portfolio for the month with factor scores (Alpha, Value, Quality, Momentum)
3. **Market Pulse** — Supplementary market context: SGD/USD rate, economic calendar, news feed, VIX level, CNN Fear & Greed Index, and insider cluster buys from OpenInsider
4. **Sector Rotation Heatmap** — Visualizes how sector allocation has shifted over the last 6 months
5. **Data Freshness** — Green/yellow/red indicator based on how recently data was updated (stale threshold: 35 days)

## Monthly Update Workflow

**1. Run the update script** (requires IBKR TWS running on localhost:7496):
```bash
cd "Factor Investing Analysis/SPY/dashboard"
python update_data.py
```
This fetches fresh EDGAR fundamentals + IBKR prices, rescores all factors, and writes the new picks into `backtest_metrics.json`.

**2. Push the updated JSON to GitHub:**
```bash
cd ..
git add backtest_metrics.json
git commit -m "Feb 2026 monthly picks"
git push
```

Streamlit Cloud auto-redeploys when the repo updates, so the dashboard will reflect the new picks within a couple minutes.

Once a year (January), run `python update_data.py --full` instead to regenerate the backtest metrics for the new universe.

## Dependencies

Python 3.10+. Key packages: `streamlit`, `pandas`, `numpy`, `scipy`, `requests`, `plotly`, `ib_insync`, `nest_asyncio`, `beautifulsoup4`.
