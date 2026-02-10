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
    page_title="The Midas Collective",
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

st.title("The Midas Collective")
st.caption("Factor Dashboard")

# ── Ticker tape of current holdings ─────────────────────────────────────
_tape_holdings = metrics.get("current_holdings", [])
if _tape_holdings:
    _tape_config = json.dumps({
        "symbols": [
            {"proName": h["ticker"], "title": h["ticker"]}
            for h in _tape_holdings
        ],
        "showSymbolLogo": True,
        "isTransparent": True,
        "displayMode": "adaptive",
        "colorTheme": "dark",
        "locale": "en",
    })
    st.components.v1.html(
        f"""<div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript"
                  src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js"
                  async>{_tape_config}</script>
        </div>""",
        height=78,
    )

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

    # ── Market widgets (Economics, Calendar, News) ────────────────────
    _widget_height = 500

    _sgdusd_config = json.dumps({
        "symbol": "FX_IDC:SGDUSD",
        "width": "100%",
        "height": str(_widget_height - 40),
        "locale": "en",
        "dateRange": "12M",
        "colorTheme": "dark",
        "isTransparent": True,
        "autosize": True,
        "largeChartUrl": "",
    })

    _cal_config = json.dumps({
        "colorTheme": "dark",
        "isTransparent": True,
        "width": "100%",
        "height": str(_widget_height - 40),
        "locale": "en",
        "importanceFilter": "-1,0,1",
        "countryFilter": "us",
    })

    _news_config = json.dumps({
        "feedMode": "market",
        "market": "stock",
        "isTransparent": True,
        "displayMode": "regular",
        "width": "100%",
        "height": str(_widget_height - 40),
        "colorTheme": "dark",
        "locale": "en",
    })

    w_col1, w_col2, w_col3 = st.columns(3)

    with w_col1:
        st.subheader("SGD / USD")
        st.components.v1.html(
            f"""<div class="tradingview-widget-container">
              <div class="tradingview-widget-container__widget"></div>
              <script type="text/javascript"
                      src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js"
                      async>{_sgdusd_config}</script>
            </div>""",
            height=_widget_height,
        )

    with w_col2:
        st.subheader("Calendar")
        st.components.v1.html(
            f"""<div class="tradingview-widget-container">
              <div class="tradingview-widget-container__widget"></div>
              <script type="text/javascript"
                      src="https://s3.tradingview.com/external-embedding/embed-widget-events.js"
                      async>{_cal_config}</script>
            </div>""",
            height=_widget_height,
        )

    with w_col3:
        st.subheader("News")
        st.components.v1.html(
            f"""<div class="tradingview-widget-container">
              <div class="tradingview-widget-container__widget"></div>
              <script type="text/javascript"
                      src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js"
                      async>{_news_config}</script>
            </div>""",
            height=_widget_height,
        )

else:
    st.warning("No holdings data found. Run `python update_data.py`.")


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — Selection Frequency Heatmap
# ═══════════════════════════════════════════════════════════════════════════

st.header("Selection Frequency Heatmap")

monthly_picks = metrics.get("monthly_picks", {})
if len(monthly_picks) >= 1:
    # Last 3 months
    recent_months = sorted(monthly_picks.keys(), reverse=True)[:3]
    n_months = len(recent_months)

    # Count how many times each ticker was selected
    freq = {t: 0 for t in SP500_TICKERS}
    for m in recent_months:
        for t in monthly_picks[m].get("tickers", []):
            if t in freq:
                freq[t] += 1

    # Sort: most frequently selected first, then alphabetically
    sorted_tickers = sorted(SP500_TICKERS, key=lambda t: (-freq[t], t))

    # Arrange into 10 × 10 grid
    grid_size = 10
    grid_z = []
    grid_text = []
    for r in range(grid_size):
        row_z = []
        row_text = []
        for c in range(grid_size):
            idx = r * grid_size + c
            if idx < len(sorted_tickers):
                ticker = sorted_tickers[idx]
                row_z.append(freq[ticker])
                row_text.append(ticker)
            else:
                row_z.append(0)
                row_text.append("")
        grid_z.append(row_z)
        grid_text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=grid_z,
        text=grid_text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale=[[0, "#000000"], [0.5, "#14532d"], [1, "#22c55e"]],
        zmin=0,
        zmax=n_months,
        showscale=True,
        colorbar=dict(
            title="Times<br>Selected",
            tickvals=list(range(n_months + 1)),
        ),
        hovertemplate="<b>%{text}</b><br>Selected %{z}/%{customdata} months<extra></extra>",
        customdata=[[n_months] * grid_size for _ in range(grid_size)],
        xgap=3,
        ygap=3,
    ))

    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"),
        height=600,
        margin=dict(l=20, r=80, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
    months_str = ", ".join(sorted(recent_months))
    st.caption(
        f"All {len(SP500_TICKERS)} universe stocks in a 10×10 grid. "
        f"Color intensity = times selected in the last {n_months} month{'s' if n_months != 1 else ''} "
        f"({months_str}). Sorted by frequency (most selected top-left)."
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

# ── Monthly picks history ──────────────────────────────────────────
st.subheader("Monthly Picks History")

monthly_picks_hist = metrics.get("monthly_picks", {})
if monthly_picks_hist:
    history_rows = []
    for month_key in sorted(monthly_picks_hist.keys(), reverse=True):
        entry = monthly_picks_hist[month_key]
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
