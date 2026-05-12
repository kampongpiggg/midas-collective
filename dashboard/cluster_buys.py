"""
OpenInsider cluster buys scraper.

Standalone module with no IBKR dependencies for use in Streamlit Cloud.
"""

import re

import requests
from bs4 import BeautifulSoup


def fetch_cluster_buys(
    min_value_k: int = 200,
    min_insiders: int = 3,
    days_back: int = 30,
) -> list[dict]:
    """
    Scrape cluster buys from openinsider.com.

    Uses OpenInsider's grouped screener with filters:
    - Officers, directors, C-suite only (no 10% owners)
    - Purchases only (no grants/awards)
    - Grouped by ticker (grp=2)
    - Min 3 insiders (nil=3)
    - Min $200k total value (vl=200)

    Returns list of dicts with keys:
        ticker, company, industry, insider_count, trade_date,
        filing_date, price, qty, value
    """
    # OpenInsider screener with proper filters for cluster buys
    # grp=2 = group by ticker, nil=3 = min 3 insiders, vl=200 = min $200k value
    # isofficer, iscob, isceo, etc. = officer/director filters
    # xp=1 = exclude private, ocl=1 = min 1% ownership change
    url = (
        f"http://openinsider.com/screener?s=&o=&pl=3&ph=&ll=&lh="
        f"&fd={days_back}&fdr=&td=0&tdr=&fdlyl=&fdlyh=6&daysago="
        f"&xp=1&vl={min_value_k}&vh=&ocl=1&och="
        f"&sic1=-1&sicl=100&sich=9999"
        f"&isofficer=1&iscob=1&isceo=1&ispres=1&iscoo=1&iscfo=1&isgc=1&isvp=1&isdirector=1"
        f"&grp=2&nfl=&nfh=&nil={min_insiders}&nih=&nol=0&noh="
        f"&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=100&page=1"
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
    headers_text = [th.get_text(strip=True).lower().replace("\xa0", " ") for th in header_row.find_all(["th", "td"])]

    def find_col(names: list[str]) -> int:
        for name in names:
            for i, h in enumerate(headers_text):
                if name in h:
                    return i
        return -1

    col_filing = find_col(["filing"])
    col_trade = find_col(["trade date", "trade"])
    col_ticker = find_col(["ticker"])
    col_company = find_col(["company"])
    col_industry = find_col(["industry"])
    col_insiders = find_col(["ins"])
    col_price = find_col(["price"])
    col_qty = find_col(["qty"])
    col_value = find_col(["value"])

    cluster_buys = []

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
        trade_date = get_cell(col_trade) if col_trade >= 0 else ""

        # Parse insider count
        ins_str = get_cell(col_insiders) if col_insiders >= 0 else "0"
        try:
            insider_count = int(ins_str)
        except ValueError:
            insider_count = 0

        # Parse price
        price_str = get_cell(col_price) if col_price >= 0 else ""
        price_clean = re.sub(r"[^\d.]", "", price_str)
        try:
            price = float(price_clean) if price_clean else 0
        except ValueError:
            price = 0

        # Parse quantity
        qty_str = get_cell(col_qty) if col_qty >= 0 else ""
        qty_clean = re.sub(r"[^\d]", "", qty_str)
        try:
            qty = int(qty_clean) if qty_clean else 0
        except ValueError:
            qty = 0

        # Parse value
        value_str = get_cell(col_value) if col_value >= 0 else ""
        value_clean = re.sub(r"[^\d]", "", value_str)
        try:
            value = int(value_clean) if value_clean else 0
        except ValueError:
            value = 0

        cluster_buys.append({
            "ticker": ticker,
            "company": company,
            "industry": industry,
            "insider_count": insider_count,
            "trade_date": trade_date,
            "filing_date": filing_date,
            "price": round(price, 2),
            "qty": qty,
            "value": value,
        })

    # Sort by value descending
    cluster_buys.sort(key=lambda x: x["value"], reverse=True)

    print(f"  Found {len(cluster_buys)} cluster buys ({min_insiders}+ insiders, >=${min_value_k}k)")
    return cluster_buys
