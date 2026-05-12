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
from cluster_buys import fetch_cluster_buys
from market_sentiment import fetch_vix, fetch_fear_greed, get_vix_color, get_fear_greed_color
from equity_curve import generate_equity_curve

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


@st.cache_data(ttl=900)
def load_cluster_buys() -> list[dict]:
    """Fetch cluster buys from OpenInsider (cached for 15 minutes)."""
    try:
        data = fetch_cluster_buys()
        if data and len(data) > 0:
            return data
    except Exception:
        pass
    # Fallback to cached data
    try:
        if BACKTEST_METRICS_JSON.exists():
            with open(BACKTEST_METRICS_JSON) as f:
                return json.load(f).get("cluster_buys", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=3600)
def load_equity_curve() -> dict:
    """Load equity curve data (cached for 1 hour)."""
    try:
        return generate_equity_curve()
    except Exception as e:
        print(f"Failed to generate equity curve: {e}")
        return {"dates": [], "portfolio_values": [], "spy_values": [], "returns": 0, "spy_returns": 0}


@st.cache_data(ttl=300)
def load_vix() -> dict:
    try:
        return fetch_vix()
    except Exception:
        return {"value": None, "change": None, "change_pct": None, "status": "N/A"}


@st.cache_data(ttl=300)
def load_fear_greed() -> dict:
    try:
        return fetch_fear_greed()
    except Exception:
        return {"value": None, "rating": "N/A", "previous_value": None, "previous_rating": None}


def get_scored_factors_mtime() -> datetime | None:
    if SCORED_FACTORS_CSV.exists():
        return datetime.fromtimestamp(os.path.getmtime(SCORED_FACTORS_CSV))
    return None


metrics = load_metrics()
cluster_buys = load_cluster_buys()
vix_data = load_vix()
fear_greed_data = load_fear_greed()
equity_curve = load_equity_curve()

# ── Title ──────────────────────────────────────────────────────────────────
st.title("The Midas Collective")

# ── Ticker tape ────────────────────────────────────────────────────────────
_tape_holdings = metrics.get("current_holdings", [])
if _tape_holdings:
    _tape_config = json.dumps({
        "symbols": [{"proName": h["ticker"], "title": h["ticker"]} for h in _tape_holdings],
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


# ═══════════════════════════════════════════════════════════════════════════
#  Section 1 — Portfolio Performance
# ═══════════════════════════════════════════════════════════════════════════

st.header("Portfolio Performance")

# ── Current Performance ───────────────────────────────────────────────────
st.subheader("Current Performance")

if equity_curve.get("dates") and len(equity_curve["dates"]) > 0:
    ec = equity_curve

    # Main metrics row
    ec1, ec2 = st.columns(2)

    with ec1:
        st.metric("Current Value", f"${ec.get('final_value', 0):,.0f}")
        # Monitoring stats below
        drawdown = ec.get("current_drawdown", 0)
        dd_color = "#22c55e" if drawdown > -10 else "#f97316" if drawdown > -25 else "#dc2626"
        rolling_sharpe = ec.get("rolling_sharpe")
        sharpe_str = f"{rolling_sharpe:.2f}" if rolling_sharpe is not None else "N/A"
        sharpe_color = "#22c55e" if rolling_sharpe and rolling_sharpe > 0.8 else "#f97316" if rolling_sharpe and rolling_sharpe > 0.3 else "#dc2626"
        st.markdown(
            f"<span style='font-size:0.85rem;color:gray'>Drawdown: </span>"
            f"<span style='font-size:0.85rem;color:{dd_color}'>{drawdown:+.1f}%</span>"
            f"<span style='font-size:0.85rem;color:gray'> · Sharpe: </span>"
            f"<span style='font-size:0.85rem;color:{sharpe_color}'>{sharpe_str}</span>",
            unsafe_allow_html=True,
        )

    with ec2:
        port_ret = ec.get("returns", 0)
        spy_ret = ec.get("spy_returns", 0)
        st.metric("Portfolio Return", f"{port_ret:+.1f}%", delta=f"{port_ret - spy_ret:+.1f}% vs SPY")
        # Win rate and thesis health
        win_count = ec.get("win_count", 0)
        total_months = ec.get("total_months", 0)
        win_rate = ec.get("win_rate")
        z_score = ec.get("z_score", 0)

        win_str = f"{win_count}/{total_months}" if total_months else "N/A"
        win_color = "#22c55e" if win_rate and win_rate >= 60 else "#f97316" if win_rate and win_rate >= 40 else "#dc2626"

        # Z-score interpretation: < -2 is concerning
        if z_score >= -1:
            z_status, z_color = "On Track", "#22c55e"
        elif z_score >= -2:
            z_status, z_color = "Monitor", "#f97316"
        else:
            z_status, z_color = "Review Thesis", "#dc2626"

        st.markdown(
            f"<span style='font-size:0.85rem;color:gray'>Win Rate: </span>"
            f"<span style='font-size:0.85rem;color:{win_color}'>{win_str}</span>"
            f"<span style='font-size:0.85rem;color:gray'> · Thesis: </span>"
            f"<span style='font-size:0.85rem;color:{z_color}'>{z_status} (z={z_score:+.1f})</span>",
            unsafe_allow_html=True,
        )

    # Equity curve chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ec["dates"],
        y=ec["portfolio_values"],
        name="Portfolio",
        line=dict(color="#22c55e", width=2),
        hovertemplate="<b>Portfolio</b><br>%{x}<br>$%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=ec["dates"],
        y=ec["spy_values"],
        name="SPY (Benchmark)",
        line=dict(color="#6b7280", width=2, dash="dot"),
        hovertemplate="<b>SPY</b><br>%{x}<br>$%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(tickformat="$,.0f"),
    )

    st.plotly_chart(fig, use_container_width=True)

    start_date = ec["dates"][0] if ec["dates"] else "?"
    end_date = ec["dates"][-1] if ec["dates"] else "?"
    st.caption(f"Live tracking: {start_date} — {end_date}")
else:
    st.info("No equity curve data available yet.")

# ── Expected Performance ──────────────────────────────────────────────────
st.subheader("Expected Performance")

m = metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Annualized Return", f"{m.get('ann_return', 0):.1%}")
c2.metric("Sharpe Ratio", f"{m.get('sharpe', 0):.2f}")
c3.metric("Max Drawdown", f"{m.get('max_drawdown', 0):.1%}")
c4.metric("Annualized Vol", f"{m.get('ann_vol', 0):.1%}")
c5.metric("Monthly Win Rate", f"{m.get('win_rate_monthly', 0):.1%}")

bt_start = m.get("backtest_start", "?")
bt_end = m.get("backtest_end", "?")
n_months = m.get("num_months", "?")
st.caption(f"Backtest period: {bt_start} — {bt_end} ({n_months} months)")


# ═══════════════════════════════════════════════════════════════════════════
#  Section 2 — Current Picks
# ═══════════════════════════════════════════════════════════════════════════

rebal_date = metrics.get("last_rebalance_date", "N/A")
rebal_dt = pd.to_datetime(rebal_date) if rebal_date != "N/A" else None
month_label = rebal_dt.strftime("%B %Y") if rebal_dt else "N/A"

st.header(f"Current Picks — {month_label}")

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
        "alpha_score": "Alpha",
        "value_score": "Value",
        "quality_score": "Quality",
        "momentum_score": "Momentum",
    }
    st.dataframe(
        picks_df[display_cols].rename(columns=col_labels),
        use_container_width=True,
        height=400,
    )
else:
    st.warning("No holdings data found. Run `python update_data.py`.")


# ═══════════════════════════════════════════════════════════════════════════
#  Section 3 — Market Pulse
# ═══════════════════════════════════════════════════════════════════════════

st.header("Market Pulse")

_widget_height = 400

# Row 1: SGD/USD, Calendar, News
w_col1, w_col2, w_col3 = st.columns(3)

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

# Row 2: VIX, Fear & Greed Gauge, Cluster Buys
w_col4, w_col5, w_col6 = st.columns(3)

with w_col4:
    st.subheader("VIX")
    vix_val = vix_data.get("value")
    vix_change = vix_data.get("change")
    vix_status = vix_data.get("status", "N/A")
    vix_color = get_vix_color(vix_val)

    if vix_val is not None:
        st.markdown(
            f"<div style='text-align:center;padding:40px 0;'>"
            f"<span style='font-size:5rem;font-weight:bold;color:{vix_color}'>{vix_val:.1f}</span>"
            f"<br><span style='font-size:1.5rem;color:{vix_color}'>{vix_status}</span>"
            f"<br><span style='font-size:1rem;color:gray'>{vix_change:+.2f} today</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("VIX data unavailable")

with w_col5:
    st.subheader("Fear & Greed Index")
    fg_val = fear_greed_data.get("value")
    fg_rating = fear_greed_data.get("rating", "N/A")

    if fg_val is not None:
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fg_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": fg_rating, "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": get_fear_greed_color(fg_val)},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#450a0a"},
                    {"range": [25, 45], "color": "#7c2d12"},
                    {"range": [45, 55], "color": "#713f12"},
                    {"range": [55, 75], "color": "#365314"},
                    {"range": [75, 100], "color": "#14532d"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": fg_val,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=_widget_height - 50,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Fear & Greed data unavailable")

with w_col6:
    st.subheader("Latest Insider Cluster Buys")
    if cluster_buys and len(cluster_buys) > 0:
        cluster_df = pd.DataFrame(cluster_buys)

        # Handle field names
        value_col = "value" if "value" in cluster_df.columns else "total_value"
        date_col = "trade_date" if "trade_date" in cluster_df.columns else "latest_filing"

        if value_col in cluster_df.columns:
            cluster_df["value_fmt"] = cluster_df[value_col].apply(lambda x: f"${x:,.0f}")
        else:
            cluster_df["value_fmt"] = "—"

        display_cols = ["ticker", "insider_count", "value_fmt"]
        display_cols = [c for c in display_cols if c in cluster_df.columns]

        col_labels = {"ticker": "Ticker", "insider_count": "Ins", "value_fmt": "Value"}

        st.dataframe(
            cluster_df[display_cols].rename(columns=col_labels).head(7),
            use_container_width=True,
            hide_index=True,
            height=_widget_height - 100,
        )
        st.markdown(
            "[View full list on OpenInsider](http://openinsider.com/screener?s=&o=&pl=3&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&fdlyl=&fdlyh=6&daysago=&xp=1&vl=200&vh=&ocl=1&och=&sic1=-1&sicl=100&sich=9999&isofficer=1&iscob=1&isceo=1&ispres=1&iscoo=1&iscfo=1&isgc=1&isvp=1&isdirector=1&grp=2&nfl=&nfh=&nil=3&nih=&nol=0&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=100&page=1)"
        )
    else:
        st.info("No cluster buys found")


# ═══════════════════════════════════════════════════════════════════════════
#  Section 4 — Sector Rotation Heatmap
# ═══════════════════════════════════════════════════════════════════════════

st.header("Sector Rotation Heatmap")

monthly_picks = metrics.get("monthly_picks", {})
if holdings and len(monthly_picks) >= 1:
    # Get sector distribution for last 6 months
    recent_months = sorted(monthly_picks.keys(), reverse=True)[:6]

    # Build sector counts per month
    sector_map = {h["ticker"]: h.get("sector", "Unknown") for h in holdings}
    sectors = sorted(set(sector_map.values()))

    heatmap_data = []
    for month in reversed(recent_months):
        tickers = monthly_picks[month].get("tickers", [])
        sector_counts = {s: 0 for s in sectors}
        for t in tickers:
            sec = sector_map.get(t, "Unknown")
            if sec in sector_counts:
                sector_counts[sec] += 1
        heatmap_data.append([sector_counts.get(s, 0) for s in sectors])

    if heatmap_data and sectors:
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=sectors,
            y=list(reversed(recent_months)),
            colorscale=[[0, "#000000"], [0.5, "#14532d"], [1, "#22c55e"]],
            showscale=True,
            colorbar=dict(title="Picks"),
            hovertemplate="<b>%{x}</b><br>%{y}: %{z} picks<extra></extra>",
            xgap=2,
            ygap=2,
        ))

        fig.update_layout(
            xaxis=dict(side="top", tickangle=-45),
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(l=20, r=20, t=80, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Number of picks per sector over the last 6 months")
    else:
        st.info("Not enough data to show sector rotation")
else:
    st.info("Need picks history for sector rotation heatmap")


# ═══════════════════════════════════════════════════════════════════════════
#  Section 5 — Data Freshness
# ═══════════════════════════════════════════════════════════════════════════

st.header("Data Freshness")

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
        color, status = "green", "Fresh"
    elif days_since < STALE_THRESHOLD_DAYS:
        color, status = "orange", "Aging"
    else:
        color, status = "red", "STALE"

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
        st.warning(f"Data is **{days_since} days old**. Run `python update_data.py` to refresh.")
else:
    st.info("No update history available.")

# Monthly picks history
with st.expander("Monthly Picks History"):
    if monthly_picks:
        history_rows = []
        for month_key in sorted(monthly_picks.keys(), reverse=True):
            entry = monthly_picks[month_key]
            history_rows.append({
                "Month": month_key,
                "Date": entry.get("date", ""),
                "Picks": ", ".join(entry.get("tickers", [])),
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No history yet.")
