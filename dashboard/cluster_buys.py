"""
OpenInsider cluster buys scraper.

Standalone module with no IBKR dependencies for use in Streamlit Cloud.
"""

import re

import requests
from bs4 import BeautifulSoup


def fetch_cluster_buys(
    min_value: int = 200_000,
    min_insiders: int = 3,
    days_back: int = 30,
) -> list[dict]:
    """
    Scrape cluster buys from openinsider.com.

    Cluster buys = multiple insiders buying around the same time.
    Filters for officers/directors with total value >= min_value in last N days.

    Returns list of dicts with keys:
        ticker, company, industry, insider_count, total_value,
        latest_filing, avg_price, titles
    """
    # OpenInsider screener URL for cluster buys (grouped by ticker)
    url = (
        "http://openinsider.com/screener?"
        f"s=&o=&pl={min_value // 1000}&ph=&ll=&lh=&fd={days_back}&fdr=&td=&tdr="
        "&fdlyl=&fdlyh=&dtefyl=&dtefyh="
        "&xa=1&xp=1&xd=1&xo=1"
        "&vl=&vh=&ocl=&och=&session=&type=&filter=cluster"
        "&sort=fd&sortdesc=1"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: Failed to fetch OpenInsider: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the main data table
    table = soup.find("table", {"class": "tinytable"})
    if not table:
        print("  WARNING: Could not find data table on OpenInsider")
        return []

    rows = table.find_all("tr")
    if len(rows) < 2:
        print("  No cluster buys found matching criteria")
        return []

    # Parse header row to find column indices
    header_row = rows[0]
    headers_text = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]

    # Column mapping (openinsider uses various column names)
    def find_col(names: list[str]) -> int:
        for name in names:
            for i, h in enumerate(headers_text):
                if name in h:
                    return i
        return -1

    col_ticker = find_col(["ticker", "symbol"])
    col_company = find_col(["company", "name"])
    col_industry = find_col(["industry", "sector"])
    col_filing = find_col(["filing", "filed"])
    col_title = find_col(["title"])
    col_price = find_col(["price"])
    col_value = find_col(["value"])

    # Aggregate by ticker for cluster detection
    ticker_data: dict[str, dict] = {}

    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        def get_cell(idx: int) -> str:
            if idx < 0 or idx >= len(cells):
                return ""
            return cells[idx].get_text(strip=True)

        ticker = get_cell(col_ticker).upper()
        if not ticker:
            continue

        company = get_cell(col_company)
        industry = get_cell(col_industry) if col_industry >= 0 else ""
        filing_date = get_cell(col_filing)
        title = get_cell(col_title) if col_title >= 0 else ""
        price_str = get_cell(col_price) if col_price >= 0 else ""
        value_str = get_cell(col_value) if col_value >= 0 else ""

        # Parse value (e.g., "$1,234,567" or "+$500,000")
        value_clean = re.sub(r"[^\d.]", "", value_str)
        try:
            value = float(value_clean) if value_clean else 0
        except ValueError:
            value = 0

        # Parse price
        price_clean = re.sub(r"[^\d.]", "", price_str)
        try:
            price = float(price_clean) if price_clean else 0
        except ValueError:
            price = 0

        # Aggregate by ticker
        if ticker not in ticker_data:
            ticker_data[ticker] = {
                "ticker": ticker,
                "company": company,
                "industry": industry,
                "insider_count": 0,
                "total_value": 0,
                "latest_filing": filing_date,
                "prices": [],
                "titles": set(),
            }

        ticker_data[ticker]["insider_count"] += 1
        ticker_data[ticker]["total_value"] += value
        if price > 0:
            ticker_data[ticker]["prices"].append(price)
        if title:
            ticker_data[ticker]["titles"].add(title)

        # Update latest filing if newer
        if filing_date > ticker_data[ticker]["latest_filing"]:
            ticker_data[ticker]["latest_filing"] = filing_date

    # Filter for actual clusters (min_insiders+) and min value
    cluster_buys = []
    for data in ticker_data.values():
        if data["insider_count"] >= min_insiders and data["total_value"] >= min_value:
            avg_price = sum(data["prices"]) / len(data["prices"]) if data["prices"] else 0
            cluster_buys.append({
                "ticker": data["ticker"],
                "company": data["company"],
                "industry": data["industry"],
                "insider_count": data["insider_count"],
                "total_value": int(data["total_value"]),
                "latest_filing": data["latest_filing"],
                "avg_price": round(avg_price, 2),
                "titles": ", ".join(sorted(data["titles"])),
            })

    # Sort by total value descending
    cluster_buys.sort(key=lambda x: x["total_value"], reverse=True)

    print(f"  Found {len(cluster_buys)} cluster buys ({min_insiders}+ insiders, >=${min_value:,})")
    return cluster_buys
