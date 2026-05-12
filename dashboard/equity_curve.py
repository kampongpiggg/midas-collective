"""
Equity curve generator from portfolio snapshots.

Fetches daily prices and calculates portfolio value over time.
"""

import requests
from datetime import datetime, timedelta
import json


# Portfolio snapshots from user's trade history
SNAPSHOTS = [
    {
        "date": "2026-01-13",
        "cash": 20.41,
        "holdings": {
            "COP": 34.34519536,
            "UNP": 14.34826541,
            "INTC": 92.55121043,
            "V": 9.978175469,
            "MU": 14.0873259,
            "NVDA": 19.49468686,
            "NOW": 24.67174926,
            "MRK": 35.15194063,
            "JNJ": 17.17866566,
            "AMAT": 13.9809437,
        },
    },
    {
        "date": "2026-02-12",
        "cash": 331.46,
        "holdings": {
            "JNJ": 16.97135319,
            "NVDA": 20.59586954,
            "COP": 37.739719,
            "MRK": 36.81676863,
            "GILD": 23.95555844,
            "UNP": 15.5734254,
            "BA": 15.59659924,
            "AMAT": 14.2087082,
            "INTC": 100.7714818,
            "MU": 16.03561655,
        },
    },
    {
        "date": "2026-03-09",
        "cash": 0,
        "holdings": {
            "INTC": 91.74384758,
            "BA": 14.54956653,
            "GILD": 22.63660581,
            "UNP": 14.73387233,
            "NVDA": 19.81748535,
            "COP": 36.15013845,
            "BMY": 59.04936015,
            "MRK": 35.92418012,
            "MA": 7.013938701,
            "JNJ": 17.0815653,
        },
    },
    {
        "date": "2026-04-06",
        "cash": 19.74,
        "holdings": {
            "COP": 36.11937237,
            "NVDA": 19.96394772,
            "NOW": 35.1879109,
            "BMY": 60.17949144,
            "MRK": 36.35543075,
            "GILD": 23.75531567,
            "JNJ": 17.12399811,
            "MU": 9.741582672,
            "UNP": 15.3199361,
            "INTC": 97.24794919,
        },
    },
    {
        "date": "2026-05-11",
        "cash": 0,
        "capital_injection": 9750,  # Box spread leverage
        "holdings": {
            "ABBV": 26.00954208,
            "CRM": 30.16896398,
            "V": 17.03086845,
            "MO": 80.07033998,
            "NVDA": 25,
            "MCD": 20.01096091,
            "CVX": 30,
            "JNJ": 25,
            "MU": 7,
            "INTC": 44,
        },
    },
]

# Total capital injections (for accurate return calculation)
CAPITAL_INJECTIONS = [
    {"date": "2026-05-11", "amount": 9750},  # Box spread leverage
]


def fetch_daily_prices(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Fetch daily closing prices from Yahoo Finance.

    Returns dict mapping date string (YYYY-MM-DD) to closing price.
    """
    # Convert dates to timestamps
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    period1 = int(start_dt.timestamp())
    period2 = int(end_dt.timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("chart", {}).get("result", [{}])[0]
        timestamps = result.get("timestamp", [])
        closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])

        prices = {}
        for ts, close in zip(timestamps, closes):
            if close is not None:
                date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                prices[date_str] = close

        return prices
    except Exception as e:
        print(f"  Failed to fetch {ticker}: {e}")
        return {}


def fetch_spy_prices(start_date: str, end_date: str) -> dict:
    """Fetch SPY prices for benchmark comparison."""
    return fetch_daily_prices("SPY", start_date, end_date)


def generate_equity_curve() -> dict:
    """
    Generate equity curve data from portfolio snapshots.

    Returns dict with:
        - dates: list of date strings
        - portfolio_values: list of daily portfolio values
        - spy_values: list of SPY values (normalized to same starting value)
        - returns: cumulative return percentage (adjusted for capital injections)
        - spy_returns: SPY cumulative return percentage
        - capital_injections: total capital added after start
    """
    start_date = SNAPSHOTS[0]["date"]
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Get all unique tickers
    all_tickers = set()
    for snapshot in SNAPSHOTS:
        all_tickers.update(snapshot["holdings"].keys())

    print(f"Fetching prices for {len(all_tickers)} tickers...")

    # Fetch prices for all tickers
    ticker_prices = {}
    for ticker in all_tickers:
        print(f"  Fetching {ticker}...")
        ticker_prices[ticker] = fetch_daily_prices(ticker, start_date, end_date)

    # Fetch SPY for benchmark
    print("  Fetching SPY...")
    spy_prices = fetch_spy_prices(start_date, end_date)

    # Get all trading dates (from SPY)
    all_dates = sorted(spy_prices.keys())

    if not all_dates:
        print("No price data available")
        return {"dates": [], "portfolio_values": [], "spy_values": [], "returns": 0, "spy_returns": 0}

    # Calculate daily portfolio values
    portfolio_values = []
    current_snapshot_idx = 0

    for date in all_dates:
        # Check if we need to move to next snapshot
        while (current_snapshot_idx < len(SNAPSHOTS) - 1 and
               date >= SNAPSHOTS[current_snapshot_idx + 1]["date"]):
            current_snapshot_idx += 1

        snapshot = SNAPSHOTS[current_snapshot_idx]

        # Calculate portfolio value for this day
        daily_value = snapshot.get("cash", 0)
        for ticker, shares in snapshot["holdings"].items():
            if ticker in ticker_prices and date in ticker_prices[ticker]:
                daily_value += shares * ticker_prices[ticker][date]
            elif ticker in ticker_prices:
                # Use most recent available price
                available_dates = [d for d in ticker_prices[ticker].keys() if d <= date]
                if available_dates:
                    latest_date = max(available_dates)
                    daily_value += shares * ticker_prices[ticker][latest_date]

        portfolio_values.append(round(daily_value, 2))

    # Calculate total capital injections
    total_injections = sum(inj["amount"] for inj in CAPITAL_INJECTIONS)

    # Normalize SPY to same starting value
    initial_portfolio_value = portfolio_values[0] if portfolio_values else 0
    initial_spy_price = spy_prices.get(all_dates[0], 1)
    spy_multiplier = initial_portfolio_value / initial_spy_price if initial_spy_price else 1

    spy_values = [round(spy_prices.get(date, 0) * spy_multiplier, 2) for date in all_dates]

    # Calculate returns (adjusted for capital injections)
    final_portfolio = portfolio_values[-1] if portfolio_values else 0
    final_spy = spy_values[-1] if spy_values else 0

    # Investment return = (Final Value - Capital Injections - Initial Value) / Initial Value
    adjusted_gain = final_portfolio - total_injections - initial_portfolio_value
    portfolio_return = (adjusted_gain / initial_portfolio_value) * 100 if initial_portfolio_value else 0
    spy_return = ((final_spy / initial_portfolio_value) - 1) * 100 if initial_portfolio_value else 0

    return {
        "dates": all_dates,
        "portfolio_values": portfolio_values,
        "spy_values": spy_values,
        "returns": round(portfolio_return, 2),
        "spy_returns": round(spy_return, 2),
        "initial_value": round(initial_portfolio_value, 2),
        "final_value": round(final_portfolio, 2),
        "capital_injections": total_injections,
    }


def save_equity_curve(output_path: str = "data/equity_curve.json"):
    """Generate and save equity curve data to JSON."""
    print("Generating equity curve...")
    curve_data = generate_equity_curve()

    with open(output_path, "w") as f:
        json.dump(curve_data, f, indent=2)

    print(f"Saved equity curve to {output_path}")
    print(f"  Period: {curve_data['dates'][0]} to {curve_data['dates'][-1]}")
    print(f"  Initial value: ${curve_data['initial_value']:,.2f}")
    print(f"  Final value: ${curve_data['final_value']:,.2f}")
    print(f"  Portfolio return: {curve_data['returns']:+.2f}%")
    print(f"  SPY return: {curve_data['spy_returns']:+.2f}%")

    return curve_data


if __name__ == "__main__":
    save_equity_curve()
