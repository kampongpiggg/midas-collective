"""
SPY Factor Dashboard — Streamlit app.

Run:  streamlit run app.py
"""

import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import plotly.graph_objects as go

from config import BACKTEST_METRICS_JSON, SCORED_FACTORS_CSV, SP500_TICKERS, STALE_THRESHOLD_DAYS

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPY Factor Dashboard",
    page_icon="📈",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_metrics() -> dict:
    if not BACKTEST_METRICS_JSON.exists():
        st.error(f"Missing {BACKTEST_METRICS_JSON}. Run `python update_data.py` first.")
        st.stop()
    with open(BACKTEST_METRICS_JSON) as f:
        return json.load(f)


def get_scored_factors_mtime() -> datetime | None:
    if SCORED_FACTORS_CSV.exists():
        return datetime.fromtimestamp(os.path.getmtime(SCORED_FACTORS_CSV))
    return None


metrics = load_metrics()

# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 — Monthly Stock Picks (primary use case)
# ═══════════════════════════════════════════════════════════════════════════

st.title("SPY Factor Dashboard")

rebal_date = metrics.get("last_rebalance_date", "N/A")
rebal_dt = pd.to_datetime(rebal_date)
month_label = rebal_dt.strftime("%B %Y") if rebal_date != "N/A" else "N/A"

st.header(f"Current Picks — {month_label}")
st.caption(f"Rebalance date: {rebal_date}")

holdings = metrics.get("current_holdings", [])
if holdings:
    picks_df = pd.DataFrame(holdings)
    picks_df.index = range(1, len(picks_df) + 1)
    picks_df.index.name = "#"

    display_cols = ["ticker", "sector", "weight", "alpha_score",
                    "value_score", "quality_score", "momentum_score"]
    col_labels = {
        "ticker": "Ticker",
        "sector": "Sector",
        "weight": "Weight",
        "alpha_score": "Alpha Score",
        "value_score": "Value Score",
        "quality_score": "Quality Score",
        "momentum_score": "Momentum Score",
    }
    st.dataframe(
        picks_df[display_cols].rename(columns=col_labels),
        use_container_width=True,
        height=400,
    )

    # ── TradingView mini charts ────────────────────────────────────────
    st.subheader("Price Charts (TradingView)")

    def tradingview_widget(symbol: str, width: int = 350, height: int = 220) -> str:
        return f"""
        <div class="tradingview-widget-container" style="height:{height}px;width:100%">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript"
                  src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
                  async>
          {{
            "symbol": "{symbol}",
            "width": "100%",
            "height": "{height}",
            "locale": "en",
            "dateRange": "1M",
            "colorTheme": "light",
            "isTransparent": true,
            "autosize": true,
            "largeChartUrl": ""
          }}
          </script>
        </div>
        """

    tickers = [h["ticker"] for h in holdings]
    for row_start in range(0, len(tickers), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx < len(tickers):
                with col:
                    st.components.v1.html(tradingview_widget(tickers[idx]), height=240)

    # ── Monthly picks history ──────────────────────────────────────────
    st.subheader("Monthly Picks History")

    monthly_picks = metrics.get("monthly_picks", {})
    if monthly_picks:
        history_rows = []
        for month_key in sorted(monthly_picks.keys(), reverse=True):
            entry = monthly_picks[month_key]
            history_rows.append({
                "Month": month_key,
                "Rebalance Date": entry.get("date", ""),
                "Picks": ", ".join(entry.get("tickers", [])),
                "Count": len(entry.get("tickers", [])),
            })
        history_df = pd.DataFrame(history_rows)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No monthly picks history yet. Run `python update_data.py` to populate.")

else:
    st.warning("No holdings data found. Run `python update_data.py`.")


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — Selection Frequency Heatmap
# ═══════════════════════════════════════════════════════════════════════════

st.header("Selection Frequency Heatmap")

monthly_picks = metrics.get("monthly_picks", {})
if len(monthly_picks) >= 1:
    # Last 6 months, chronological
    all_months = sorted(monthly_picks.keys(), reverse=True)[:6]
    all_months = sorted(all_months)
    n_months = len(all_months)

    # Build presence matrix: all 100 tickers × months
    # 1 = selected, 0 = not selected
    matrix = pd.DataFrame(0, index=SP500_TICKERS, columns=all_months)
    for m in all_months:
        for t in monthly_picks[m].get("tickers", []):
            if t in matrix.index:
                matrix.loc[t, m] = 1

    # Sort by total selections (most frequent at top), then alphabetically
    matrix["_total"] = matrix[all_months].sum(axis=1)
    matrix = matrix.sort_values("_total", ascending=False)
    matrix = matrix.drop(columns=["_total"])

    # Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=all_months,
        y=matrix.index.tolist(),
        colorscale=[
            [0.0, "#f1f5f9"],   # not selected — light gray
            [1.0, "#dc2626"],   # selected — red (high intensity)
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
        customdata=[["Selected" if v == 1 else "Not selected" for v in row] for row in matrix.values],
        xgap=2,
        ygap=1,
    ))

    fig.update_layout(
        yaxis=dict(
            autorange="reversed",  # most-selected tickers at top
            dtick=1,
            tickfont=dict(size=10),
        ),
        xaxis=dict(
            side="top",
            tickfont=dict(size=12),
        ),
        height=max(n_months * 50, len(matrix) * 18 + 80),
        margin=dict(l=60, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"All {len(SP500_TICKERS)} universe stocks over the last {n_months} "
        f"month{'s' if n_months != 1 else ''}. "
        "Red = selected that month. Gray = not selected. "
        "Stocks sorted by total selection frequency (most frequent at top)."
    )
else:
    st.info("Need at least 1 month of picks data for the heatmap. Run `python update_data.py`.")


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 — Data Freshness Bar
# ═══════════════════════════════════════════════════════════════════════════

st.header("Data Freshness")

# Prefer scored_factors.csv mtime if available (local), fall back to JSON rebalance date (cloud)
mtime = get_scored_factors_mtime()
if mtime is not None:
    last_updated = mtime
    last_updated_str = mtime.strftime("%Y-%m-%d %H:%M")
else:
    rebal = metrics.get("last_rebalance_date")
    last_updated = pd.to_datetime(rebal) if rebal else None
    last_updated_str = rebal if rebal else "Unknown"

if last_updated is not None:
    days_since = (datetime.now() - last_updated).days

    if days_since < 25:
        color = "green"
        status = "Fresh"
    elif days_since < STALE_THRESHOLD_DAYS:
        color = "orange"
        status = "Aging"
    else:
        color = "red"
        status = "STALE"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Last Updated", last_updated_str)
    with col2:
        st.metric("Days Since Update", days_since)
    with col3:
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background-color:{color};"
            f"color:white;text-align:center;font-size:1.4rem;font-weight:bold;margin-top:8px'>"
            f"{status}</div>",
            unsafe_allow_html=True,
        )

    if days_since >= STALE_THRESHOLD_DAYS:
        st.warning(
            f"Data is **{days_since} days old** (threshold: {STALE_THRESHOLD_DAYS} days). "
            "Run `python update_data.py` to refresh."
        )
else:
    st.info("No update history available. Run `python update_data.py` to populate.")


# ═══════════════════════════════════════════════════════════════════════════
#  Section 4 — Backtest Summary Metrics
# ═══════════════════════════════════════════════════════════════════════════

st.header("Backtest Summary")

m = metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Annualized Return", f"{m.get('ann_return', 0):.1%}")
c2.metric("Sharpe Ratio", f"{m.get('sharpe', 0):.2f}")
c3.metric("Max Drawdown", f"{m.get('max_drawdown', 0):.1%}")
c4.metric("Annualized Vol", f"{m.get('ann_vol', 0):.1%}")
c5.metric("Monthly Win Rate", f"{m.get('win_rate_monthly', 0):.1%}")

bt_start = m.get("backtest_start", "?")
bt_end   = m.get("backtest_end", "?")
n_months = m.get("num_months", "?")
st.caption(f"Backtest period: {bt_start} — {bt_end} ({n_months} months). Updated annually with new universe.")
