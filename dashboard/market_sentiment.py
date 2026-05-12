"""
Market sentiment indicators: VIX and Fear & Greed Index.

Standalone module with no IBKR dependencies for use in Streamlit Cloud.
"""

import requests


def fetch_vix() -> dict:
    """
    Fetch current VIX level from Yahoo Finance.

    Returns dict with keys: value, change, change_pct, status
    """
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("chart", {}).get("result", [{}])[0]
        meta = result.get("meta", {})

        price = meta.get("regularMarketPrice", 0)
        prev_close = meta.get("previousClose", price)
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        # VIX interpretation
        if price < 15:
            status = "Low"
        elif price < 20:
            status = "Normal"
        elif price < 30:
            status = "Elevated"
        else:
            status = "High"

        return {
            "value": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "status": status,
        }
    except Exception as e:
        print(f"Failed to fetch VIX: {e}")
        return {"value": None, "change": None, "change_pct": None, "status": "N/A"}


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
