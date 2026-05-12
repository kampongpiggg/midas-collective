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

## Strategy Health Monitoring

The dashboard includes a **Strategy Health Badge** that runs statistical tests to determine if the live portfolio is performing as expected compared to the backtest.

### Metrics Displayed

| Metric | Description |
|--------|-------------|
| **Returns (z)** | Z-score of cumulative returns vs backtest expectations |
| **Vol** | Realized annualized volatility |
| **Win** | Monthly win rate (positive return months / total months) |
| **Sharpe** | Rolling Sharpe ratio |
| **DD** | Current drawdown from peak |

### Statistical Tests

**1. Returns Z-Score**

Measures how many standard deviations the cumulative return is from the expected value.

```
Expected monthly return: 2.28% (27.4% annual / 12)
Expected monthly std: 6.41% (22.2% annual / √12)

After n months:
  Expected cumulative return = 2.28% × n
  Expected std = 6.41% × √n
  
z = (actual_cumulative - expected_cumulative) / expected_std
```

| z-score | Probability of being this bad by chance | Interpretation |
|---------|----------------------------------------|----------------|
| z > 1.5 | Top 6.7% | Significantly outperforming |
| z ∈ [-1, 1.5] | Middle 84% | Normal variance |
| z ∈ [-2, -1) | Bottom 2-16% | Below expectations |
| z < -2 | Bottom 2.3% | Statistically unlikely if thesis holds |

**2. Volatility Chi-Squared Test**

Tests whether realized volatility differs significantly from the backtest volatility (22.2% annual).

```
H₀: σ² = expected variance
χ² = (n-1) × sample_var / expected_var
p-value from chi-squared distribution
```

- **p < 0.05**: Volatility is statistically different from backtest — risk profile has changed.

**3. Win Rate Binomial Test**

Tests whether the observed win rate is consistent with the backtest win rate (70.6%).

```
H₀: p = 0.706 (backtest win rate)
p-value = P(observing ≤ k wins | n trials, p = 0.706)
```

- **p < 0.05**: Win rate is statistically below expected.

### Strategy States

The badge displays one of the following states based on the evaluation logic:

| State | Conditions | Color | Action |
|-------|------------|-------|--------|
| **Review Thesis** | z < -2 | Black | Returns are 2+ std below expected (<2.3% chance if strategy works). Stop and investigate. |
| **Max Drawdown Breached** | DD < -34.6% | Red | Exceeded historical max drawdown. Unprecedented territory. |
| **Underperforming Backtest** | z ∈ [-2, -1) AND (win rate p<0.05 OR DD < -25%) | Red | Below expectations with supporting red flags. Discuss next steps. |
| **Lucky** | z > 1 AND vol significantly high AND Sharpe < 0.7 | Orange | Good returns but driven by excessive risk, not skill. Don't get overconfident. |
| **High Volatility** | Vol significantly high (p<0.05) AND Sharpe < 0.7 | Orange | Risk is elevated without commensurate reward. |
| **Outperforming Backtest** | z > 1.5 AND Sharpe > 1.20 | Green | Beating expectations with strong risk-adjusted returns. |
| **On Track** | Everything else | Green | Normal variance. Strategy performing as expected. |

### Interpretation Guidelines

- **z-score** tells you if cumulative returns are on track. A single bad month can drag it down temporarily — factor strategies have rough patches.
- **Volatility p-value** tells you if risk has structurally changed. High vol with good Sharpe is fine; high vol with poor Sharpe is concerning.
- **Win rate** matters less for monthly rebalancing. A few losing months don't invalidate the thesis.
- **Drawdown** is compared against historical max (-34.6%). Approaching or exceeding this level warrants attention.

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
