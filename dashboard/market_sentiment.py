"""
Market sentiment indicators for the SPY factor dashboard.

Two groups of indicators live here:
  * General sentiment — 30-day VIX and CNN Fear & Greed.
  * 0DTE strangle-selling signals — intraday VIX1D, VIX term structure,
    SPX dealer net gamma exposure (GEX), and a composite traffic light.

Standalone module with no IBKR dependencies for use in Streamlit Cloud.
All data is sourced over plain HTTP (Yahoo Finance, CNN, CBOE).
"""

import re
from datetime import date, datetime

import requests

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _fetch_yahoo_index(symbol: str) -> dict:
    """
    Fetch the current level of a Yahoo Finance index (e.g. ^VIX, ^VIX1D).

    Returns dict with keys: value, change, change_pct (None on failure).
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        if price is None:
            return {"value": None, "change": None, "change_pct": None}

        prev_close = meta.get("previousClose", price) or price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        return {
            "value": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
        }
    except Exception as e:
        print(f"Failed to fetch {symbol}: {e}")
        return {"value": None, "change": None, "change_pct": None}


def fetch_vix() -> dict:
    """
    Fetch current 30-day VIX level from Yahoo Finance.

    Returns dict with keys: value, change, change_pct, status
    """
    out = _fetch_yahoo_index("%5EVIX")
    price = out.get("value")
    if price is None:
        return {"value": None, "change": None, "change_pct": None, "status": "N/A"}

    if price < 15:
        status = "Low"
    elif price < 20:
        status = "Normal"
    elif price < 30:
        status = "Elevated"
    else:
        status = "High"
    out["status"] = status
    return out


def fetch_fear_greed() -> dict:
    """
    Fetch CNN Fear & Greed Index.

    Returns dict with keys: value, rating, previous_value, previous_rating
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Current score
        fear_greed = data.get("fear_and_greed", {})
        score = fear_greed.get("score", 0)
        rating = fear_greed.get("rating", "N/A")

        # Previous close
        prev = fear_greed.get("previous_close", 0)
        prev_rating = fear_greed.get("previous_1_week_rating", rating)

        return {
            "value": round(score, 1),
            "rating": rating.replace("_", " ").title(),
            "previous_value": round(prev, 1) if prev else None,
            "previous_rating": prev_rating.replace("_", " ").title() if prev_rating else None,
        }
    except Exception as e:
        print(f"Failed to fetch Fear & Greed: {e}")
        return {"value": None, "rating": "N/A", "previous_value": None, "previous_rating": None}


def get_fear_greed_color(value: float) -> str:
    """Return color based on Fear & Greed score (0-100)."""
    if value is None:
        return "gray"
    if value <= 25:
        return "#dc2626"  # Extreme Fear - red
    elif value <= 45:
        return "#f97316"  # Fear - orange
    elif value <= 55:
        return "#eab308"  # Neutral - yellow
    elif value <= 75:
        return "#84cc16"  # Greed - light green
    else:
        return "#22c55e"  # Extreme Greed - green


def get_vix_color(value: float) -> str:
    """Return color based on VIX level."""
    if value is None:
        return "gray"
    if value < 15:
        return "#22c55e"  # Low - green
    elif value < 20:
        return "#84cc16"  # Normal - light green
    elif value < 30:
        return "#f97316"  # Elevated - orange
    else:
        return "#dc2626"  # High - red


# ═══════════════════════════════════════════════════════════════════════════
#  0DTE strangle-selling signals: VIX1D, VIX term structure, SPX net GEX
# ═══════════════════════════════════════════════════════════════════════════

# Colors reused across the 0DTE tiles
_GREEN = "#22c55e"
_YELLOW = "#eab308"
_RED = "#dc2626"
_GRAY = "gray"


def fetch_vix1d() -> dict:
    """
    Fetch the Cboe 1-Day Volatility Index (VIX1D) — expected S&P 500 vol over
    the *current* session, the correct tenor for 0DTE rather than 30-day VIX.

    Bands are tuned for premium selling: too low = thin premium / not worth the
    tail risk, mid = sellable, high/spike = realized vol likely to breach
    breakevens. Note VIX1D has a documented intraday "overnight bias" (drifts up
    into the close, gaps down at the open), so read spikes more than the level.

    Returns dict with keys: value, change, change_pct, status, color
    """
    out = _fetch_yahoo_index("%5EVIX1D")
    price = out.get("value")
    if price is None:
        return {**out, "status": "N/A", "color": _GRAY}

    if price < 10:
        status, color = "Very Low", _YELLOW   # premium too thin
    elif price < 18:
        status, color = "Sellable", _GREEN
    elif price < 25:
        status, color = "Elevated", _YELLOW
    else:
        status, color = "Spike", _RED         # stand down
    out["status"] = status
    out["color"] = color
    return out


def fetch_vix_term_structure() -> dict:
    """
    Build the front of the VIX term structure (VIX1D / VIX9D / 30-day VIX) and
    classify its shape. Contango (front < back) = calm, premium-selling regime;
    backwardation (front spikes above the curve) = stress, stand down.

    Returns dict with keys: vix1d, vix9d, vix, ratio, shape, color
        ratio = VIX1D / VIX9D
    """
    vix1d = _fetch_yahoo_index("%5EVIX1D").get("value")
    vix9d = _fetch_yahoo_index("%5EVIX9D").get("value")
    vix = _fetch_yahoo_index("%5EVIX").get("value")

    if vix1d is None or vix9d is None or not vix9d:
        return {"vix1d": vix1d, "vix9d": vix9d, "vix": vix,
                "ratio": None, "shape": "N/A", "color": _GRAY}

    ratio = vix1d / vix9d
    if ratio >= 1.05:
        shape, color = "Backwardation", _RED      # front-end stress
    elif ratio <= 0.95:
        shape, color = "Contango", _GREEN          # calm
    else:
        shape, color = "Flat", _YELLOW
    return {"vix1d": vix1d, "vix9d": vix9d, "vix": vix,
            "ratio": round(ratio, 3), "shape": shape, "color": color}


# OSI option symbol, e.g. "SPXW260613C04500000" -> root, expiry, C/P, strike
_OSI_RE = re.compile(r"^([A-Z\^]+?)(\d{6})([CP])(\d{8})$")


def _parse_osi(sym: str):
    m = _OSI_RE.match(sym or "")
    if not m:
        return None
    root, ymd, cp, strike = m.groups()
    try:
        exp = datetime.strptime(ymd, "%y%m%d").date()
    except ValueError:
        return None
    return root, exp, cp, int(strike) / 1000.0


def _empty_gex() -> dict:
    return {"net_gex": None, "regime": "N/A", "color": _GRAY, "spot": None,
            "call_wall": None, "put_wall": None, "within_walls": None,
            "front_expiry": None}


def fetch_net_gex(symbol: str = "_SPX", window_days: int = 7) -> dict:
    """
    Compute dealer net gamma exposure (GEX) for SPX from CBOE's free,
    ~15-min-delayed options chain. CBOE publishes per-contract gamma directly,
    so no Black-Scholes is required.

        net GEX = Σ  gamma · OI · 100 · spot² · 0.01   (calls +, puts −)

    Aggregated over expirations within `window_days` (the near-dated gamma that
    drives 0DTE hedging), in $ per 1% move. Walls are the nearest-expiry strikes
    with the largest call / put gamma — natural pinning magnets and short-strike
    anchors. Spot inside [put_wall, call_wall] suggests a pinned, sell-friendly
    range.

    Positive net GEX  -> dealers long gamma -> moves suppressed (sell-friendly).
    Negative net GEX  -> dealers short gamma -> moves amplified (stand down).

    Returns dict: net_gex ($), regime, color, spot, call_wall, put_wall,
                  within_walls, front_expiry.
    """
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        spot = data.get("current_price")
        options = data.get("options", [])
        if not spot or not options:
            return _empty_gex()

        today = date.today()
        cutoff = today.toordinal() + window_days
        spot2 = spot * spot

        net_gex = 0.0
        call_g: dict[float, float] = {}
        put_g: dict[float, float] = {}
        front_expiry = None

        # first pass: nearest expiry on/after today (the 0DTE front)
        for o in options:
            p = _parse_osi(o.get("option", ""))
            if not p:
                continue
            _, exp, _, _ = p
            if exp >= today and (front_expiry is None or exp < front_expiry):
                front_expiry = exp

        for o in options:
            p = _parse_osi(o.get("option", ""))
            if not p:
                continue
            _, exp, cp, strike = p
            if exp < today or exp.toordinal() > cutoff:
                continue
            gamma = o.get("gamma") or 0.0
            oi = o.get("open_interest") or 0.0
            g_dollars = gamma * oi * 100 * spot2 * 0.01
            net_gex += g_dollars if cp == "C" else -g_dollars
            if exp == front_expiry:
                bucket = call_g if cp == "C" else put_g
                bucket[strike] = bucket.get(strike, 0.0) + g_dollars

        call_wall = max(call_g, key=call_g.get) if call_g else None
        put_wall = max(put_g, key=put_g.get) if put_g else None
        within_walls = None
        if call_wall is not None and put_wall is not None:
            lo, hi = sorted((put_wall, call_wall))
            within_walls = lo <= spot <= hi

        if net_gex > 0:
            regime, color = "Positive (suppressed)", _GREEN
        else:
            regime, color = "Negative (amplified)", _RED

        return {
            "net_gex": net_gex,
            "regime": regime,
            "color": color,
            "spot": round(spot, 2),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "within_walls": within_walls,
            "front_expiry": front_expiry.isoformat() if front_expiry else None,
        }
    except Exception as e:
        print(f"Failed to fetch net GEX: {e}")
        return _empty_gex()


def compute_strangle_signal(gex: dict, vix1d: dict, term: dict) -> dict:
    """
    Combine the 0DTE signals into a single GREEN / YELLOW / RED traffic light
    for "can I sell a 0DTE SPY strangle right now?".

      RED   — any hard stop: short-gamma regime, VIX1D spike, term
              backwardation, or spot outside the gamma walls.
      GREEN — all clear: positive net GEX, VIX1D in the sellable band,
              contango/flat term structure, spot inside the walls.
      YELLOW — mixed; trade smaller / wider strikes or wait.

    Does NOT account for scheduled macro events (FOMC/CPI/NFP) — check the
    economic Calendar tile before selling into a number.

    Returns dict: light ("GREEN"/"YELLOW"/"RED"/"N/A"), color, reasons (list).
    """
    reasons: list[str] = []
    red = False

    net_gex = gex.get("net_gex")
    if net_gex is None:
        return {"light": "N/A", "color": _GRAY,
                "reasons": ["GEX data unavailable"]}

    # Hard stops -> RED
    if net_gex <= 0:
        red = True
        reasons.append("Net GEX negative — dealers short gamma, moves amplified")
    if gex.get("within_walls") is False:
        red = True
        reasons.append("Spot outside the gamma walls — no pinning support")
    if vix1d.get("status") == "Spike":
        red = True
        reasons.append("VIX1D spiking — realized vol likely to breach breakevens")
    if term.get("shape") == "Backwardation":
        red = True
        reasons.append("VIX term structure in backwardation — front-end stress")

    if red:
        return {"light": "RED", "color": _RED, "reasons": reasons}

    # All-clear conditions -> GREEN
    green = (
        net_gex > 0
        and vix1d.get("status") == "Sellable"
        and term.get("shape") in ("Contango", "Flat")
        and gex.get("within_walls") is True
    )
    if green:
        reasons.append("Positive net GEX, VIX1D sellable, term calm, spot pinned")
        reasons.append("Still check the economic Calendar for events today")
        return {"light": "GREEN", "color": _GREEN, "reasons": reasons}

    # Otherwise caution
    if vix1d.get("status") == "Very Low":
        reasons.append("VIX1D very low — premium may be too thin to be worth it")
    if vix1d.get("status") == "Elevated":
        reasons.append("VIX1D elevated — sell smaller / wider strikes")
    if term.get("shape") == "Flat":
        reasons.append("Term structure flat — no clear calm signal")
    if not reasons:
        reasons.append("Mixed signals — trade smaller or wait")
    return {"light": "YELLOW", "color": _YELLOW, "reasons": reasons}
