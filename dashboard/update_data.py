#!/usr/bin/env python
"""
Monthly data-update CLI for the SPY factor-investing strategy.

Usage
-----
    python update_data.py            # monthly: EDGAR + IBKR + rescore + picks
    python update_data.py --full     # annual : same + full backtest → metrics
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import nest_asyncio
import numpy as np
import pandas as pd
import requests
from ib_insync import IB, Stock, util
from scipy.stats import ttest_1samp

from config import (
    ALL_FUNDAMENTALS_CSV,
    BACKTEST_METRICS_JSON,
    BASE_DIR,
    FULL_FACTORS_CSV,
    IBKR_CLIENT_BACKTEST,
    IBKR_CLIENT_DATA,
    IBKR_HOST,
    IBKR_PORT,
    MOM_WINDOW_DAYS,
    REPORT_LAG_DAYS,
    SCORED_FACTORS_CSV,
    SEC_API_KEY,
    SEC_USER_AGENT,
    SP500_TICKERS,
    START_DATE,
    TICKER_SECTOR_CSV,
    TOP_DECILE,
)

# ═══════════════════════════════════════════════════════════════════════════
#  EDGAR helpers  (from Dataset Construction.ipynb cells bd19c1c5, 51976640)
# ═══════════════════════════════════════════════════════════════════════════

def _get_cik_from_ticker(
    ticker: str,
    user_agent: str,
    api_key: Optional[str] = None,
) -> str:
    """Resolve ticker → CIK using SEC's company_tickers.json."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    if api_key:
        headers["X-API-KEY"] = api_key

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    ticker = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Could not find CIK for ticker {ticker}")


def fetch_edgar_fundamentals(
    ticker: str,
    user_agent: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch fundamentals from SEC EDGAR companyfacts API and return a
    quarterly-only dataframe with TTM metrics.
    """
    cik_str = _get_cik_from_ticker(ticker, user_agent=user_agent, api_key=api_key)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    if api_key:
        headers["X-API-KEY"] = api_key

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    facts = data.get("facts", {}).get("us-gaap", {})

    # ── Tag mapping ────────────────────────────────────────────────────
    tag_map = {
        "total_equity": [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "Equity", "PartnersCapital", "MembersEquity",
        ],
        "long_term_debt_noncurrent": [
            "LongTermDebtNoncurrent", "DebtNoncurrent", "LongTermBorrowings",
            "LongTermDebtAndCapitalLeaseObligations",
            "LongTermDebtAndFinanceLeaseObligations", "LongTermDebt",
            "LongTermLoans", "NotesPayableNoncurrent", "LoansPayableNoncurrent",
            "ConvertibleDebtNoncurrent", "UnsecuredDebtNoncurrent",
            "SecuredDebtNoncurrent", "SeniorLongTermDebt",
            "SubordinatedLongTermDebt", "MortgageNotesPayableNoncurrent",
            "DebenturesNoncurrent", "CommercialPaperNoncurrent",
            "FinanceLeaseLiabilityNoncurrent", "CapitalLeaseObligationsNoncurrent",
            "DebtInstrumentCarryingAmount", "LongTermPortionOfDebt",
            "LongTermPortionOfBorrowings",
        ],
        "long_term_debt_current": [
            "LongTermDebtCurrent", "DebtCurrent", "CurrentPortionOfLongTermDebt",
            "CurrentPortionOfLongTermBorrowings", "CurrentPortionOfNotesPayable",
            "CurrentPortionOfDebt", "CurrentPortionOfBorrowings",
            "CurrentPortionOfBankLoans", "CurrentPortionOfConvertibleDebt",
            "CurrentPortionOfFinanceLeaseLiability",
            "CurrentPortionOfCapitalLeaseObligation", "ShortTermBorrowings",
            "ShortTermNotesPayable", "BankOverdrafts", "CommercialPaper",
            "DebtCurrentExcludingFinanceLeases",
        ],
        "cash_and_equivalents": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsAndShortTermInvestments",
            "CashAndCashEquivalents",
        ],
        "revenue": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues", "Revenue", "SalesRevenueNet",
            "SalesRevenueGoodsNet", "ProductRevenue", "ServiceRevenue",
        ],
        "gross_profit": [
            "GrossProfit", "GrossProfitLoss",
            "OperatingRevenueLessCostOfRevenue", "RevenuesLessCostOfSales",
        ],
        "operating_income": [
            "OperatingIncomeLoss", "IncomeLossFromOperations",
            "OperatingProfitLoss", "OperatingProfit", "IncomeFromOperations",
            "OperatingIncomeLossContinuingOperations",
            "OperatingIncomeLossContinuingOperationsAndIncomeLossFromEquityMethodInvestments",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ],
        "net_income": [
            "NetIncomeLoss", "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
        "dep_amort": [
            "DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
            "DepreciationAndAmortizationOfPropertyPlantAndEquipment",
        ],
        "interest_expense": [
            "InterestExpense", "InterestAndDebtExpense",
            "InterestExpenseBorrowings",
        ],
        "income_tax_expense": [
            "IncomeTaxExpenseBenefit", "IncomeTaxExpense", "IncomeTaxProvision",
        ],
        "shares_outstanding": [
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
        ],
    }

    # ── Raw period collector ───────────────────────────────────────────
    periods: dict = {}

    def _add_fact_series(field_name, tag_candidates, unit_preference=("USD", "shares")):
        for tag in tag_candidates:
            if tag not in facts:
                continue
            units_dict = facts[tag].get("units", {})
            for unit in unit_preference:
                if unit not in units_dict:
                    continue
                for item in units_dict[unit]:
                    end = item.get("end")
                    val = item.get("val")
                    form = item.get("form", "")
                    fp   = item.get("fp", "")
                    fy   = item.get("fy", None)
                    if end is None or val is None:
                        continue
                    try:
                        dt = datetime.fromisoformat(end)
                    except Exception:
                        continue
                    key = dt.date()
                    if key not in periods:
                        periods[key] = {
                            "fiscal_period_end": dt.date(),
                            "form": form, "fp": fp, "fy": fy,
                        }
                    if field_name not in periods[key] or periods[key][field_name] is None:
                        periods[key][field_name] = val
                break

    for field_name, tag_candidates in tag_map.items():
        if field_name == "shares_outstanding":
            _add_fact_series(field_name, tag_candidates, unit_preference=("shares", "SHARES"))
        else:
            _add_fact_series(field_name, tag_candidates, unit_preference=("USD",))

    if not periods:
        raise ValueError("No EDGAR facts found for this ticker.")

    df = pd.DataFrame(list(periods.values()))
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])

    # ── Total Debt ─────────────────────────────────────────────────────
    lt_non = df.get("long_term_debt_noncurrent", np.nan)
    lt_cur = df.get("long_term_debt_current", np.nan)
    df["total_debt"] = lt_non.fillna(0) + lt_cur.fillna(0)
    df.loc[df["total_debt"] == 0, "total_debt"] = np.nan

    # ── Period type ────────────────────────────────────────────────────
    def _ptype(r):
        fp = str(r.get("fp", "")).upper()
        form = str(r.get("form", "")).upper()
        if fp == "FY" or form in ("10-K", "20-F", "40-F"):
            return "annual"
        return "quarterly"

    df["period_type"] = df.apply(_ptype, axis=1)
    df["fy"] = df["fy"].fillna(df["fiscal_period_end"].dt.year)

    # ── EBIT = operating_income ────────────────────────────────────────
    if "operating_income" not in df.columns:
        df["operating_income"] = np.nan
    df["ebit"] = df["operating_income"].astype(float)

    # ── Q1–Q4 construction ─────────────────────────────────────────────
    flow_cols  = ["revenue", "gross_profit", "operating_income", "ebit", "net_income"]
    stock_cols = ["total_equity", "total_debt", "cash_and_equivalents", "shares_outstanding"]

    df_q  = df[df["period_type"] == "quarterly"].copy()
    df_fy = df[df["period_type"] == "annual"].copy()

    q4_rows = []
    for fy, grp in df_fy.groupby("fy"):
        fy_row = grp.sort_values("fiscal_period_end").iloc[-1]
        qrows = df_q[df_q["fy"] == fy].sort_values("fiscal_period_end")

        qmap: dict = {}
        for _, r in qrows.iterrows():
            fp = str(r.get("fp", "")).upper()
            if fp in ("Q1", "Q2", "Q3") and fp not in qmap:
                qmap[fp] = r

        if len(qmap) < 3 and len(qrows) >= 3:
            rt = list(qrows.itertuples())
            qmap = {"Q1": rt[0], "Q2": rt[1], "Q3": rt[2]}

        if len(qmap) < 3:
            continue

        q1, q2, q3 = qmap["Q1"], qmap["Q2"], qmap["Q3"]
        q4 = {
            "fiscal_period_end": fy_row["fiscal_period_end"],
            "period_type": "quarterly", "fp": "Q4", "fy": fy,
        }
        for col in stock_cols:
            q4[col] = fy_row.get(col, np.nan)
        for col in flow_cols:
            fyv = fy_row.get(col, np.nan)
            if pd.isna(fyv):
                q4[col] = np.nan
            else:
                parts = [getattr(q1, col, np.nan), getattr(q2, col, np.nan), getattr(q3, col, np.nan)]
                if any(pd.isna(p) for p in parts):
                    q4[col] = np.nan
                else:
                    q4[col] = float(fyv) - sum(float(p) for p in parts)
        q4_rows.append(q4)

    if q4_rows:
        df_q = pd.concat([df_q, pd.DataFrame(q4_rows)], ignore_index=True)

    # ── TTM ────────────────────────────────────────────────────────────
    df_q = df_q.sort_values("fiscal_period_end").reset_index(drop=True)
    for col in flow_cols:
        df_q[col] = df_q[col].astype(float)
        df_q[col + "_ttm"] = df_q[col].rolling(window=4, min_periods=4).sum()

    # ── Final output ───────────────────────────────────────────────────
    base_cols = [
        "fiscal_period_end",
        "total_equity", "total_debt", "cash_and_equivalents",
        "revenue", "gross_profit", "operating_income", "ebit",
        "net_income", "shares_outstanding",
    ]
    ttm_cols = [c + "_ttm" for c in flow_cols]
    for col in base_cols:
        if col not in df_q.columns:
            df_q[col] = np.nan

    fund_df = df_q[base_cols + ttm_cols].sort_values("fiscal_period_end").reset_index(drop=True)
    return fund_df


# ── cell 51976640: loop wrapper ────────────────────────────────────────────
def fetch_all_fundamentals(
    tickers: list[str],
    user_agent: str = SEC_USER_AGENT,
    api_key: str | None = SEC_API_KEY,
    save_path=ALL_FUNDAMENTALS_CSV,
) -> pd.DataFrame:
    """Fetch EDGAR fundamentals for all tickers and save to CSV."""
    all_dfs = []
    for ticker in tickers:
        try:
            print(f"  Fetching fundamentals for {ticker} ...")
            fund_df = fetch_edgar_fundamentals(ticker, user_agent=user_agent, api_key=api_key)
            fund_df = fund_df.copy()
            fund_df["ticker"] = ticker
            all_dfs.append(fund_df)
        except Exception as e:
            print(f"  WARNING: Failed to fetch {ticker}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
    else:
        combined = pd.DataFrame()

    combined.to_csv(save_path, index=False)
    print(f"  Saved combined fundamentals ({combined.shape}) to {save_path}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════
#  Monthly fundamentals panel  (cell bff73dcb)
# ═══════════════════════════════════════════════════════════════════════════

def prepare_monthly_fundamentals(
    csv_path=ALL_FUNDAMENTALS_CSV,
    freq: str = "BMS",
) -> pd.DataFrame:
    """Monthly 'as-of' panel: first business day of each month, backward-merged."""
    df = pd.read_csv(csv_path)
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
    df = df.sort_values(["ticker", "fiscal_period_end"])

    out_frames = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("fiscal_period_end")
        start = grp["fiscal_period_end"].min().normalize()
        end   = grp["fiscal_period_end"].max().normalize()
        monthly_idx = pd.date_range(start=start, end=end, freq=freq)
        monthly_dates = pd.DataFrame({"asof_date": monthly_idx})
        merged = pd.merge_asof(
            monthly_dates.sort_values("asof_date"),
            grp.sort_values("fiscal_period_end"),
            left_on="asof_date",
            right_on="fiscal_period_end",
            direction="backward",
        )
        merged["ticker"] = ticker
        out_frames.append(merged)

    monthly = pd.concat(out_frames, ignore_index=True)
    monthly = monthly.set_index(["ticker", "asof_date"]).sort_index()
    return monthly


# ═══════════════════════════════════════════════════════════════════════════
#  IBKR price fetch  (cell 78e2c027)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_ohlcv_ibkr(tickers, client_id=IBKR_CLIENT_DATA):
    """Fetch 15 years of daily OHLCV from IBKR TWS."""
    nest_asyncio.apply()
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=client_id)
    dfs = []
    for symbol in tickers:
        contract = Stock(symbol, "SMART", "USD")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="15 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
        )
        df = util.df(bars)[["date", "open", "high", "low", "close", "volume"]]
        df["ticker"] = symbol
        dfs.append(df)
    ib.disconnect()
    return pd.concat(dfs, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Factor computation  (cells 0f3cbd38, 042d0b07, 13ea9ba6, 8ae039a7)
# ═══════════════════════════════════════════════════════════════════════════

def _to_float(x, default=np.nan):
    if x is None:
        return default
    if pd.isna(x):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def compute_value_factors(price_df, fund_df, date_range, report_lag_days=REPORT_LAG_DAYS):
    date_index = pd.to_datetime(pd.Index(date_range)).sort_values()
    rows = []
    for asof in date_index:
        asof = pd.to_datetime(asof)
        px_idx = price_df.index[price_df.index <= asof]
        if len(px_idx) == 0:
            rows.append({"date": asof, "book_to_price": np.nan, "ev_ebit": np.nan, "debt_to_equity": np.nan})
            continue
        price_val = float(price_df.loc[px_idx[-1], "close"])

        cutoff = asof - timedelta(days=report_lag_days)
        eligible = fund_df[fund_df["fiscal_period_end"] <= cutoff]
        if eligible.empty:
            rows.append({"date": asof, "book_to_price": np.nan, "ev_ebit": np.nan, "debt_to_equity": np.nan})
            continue

        f = eligible.iloc[-1]
        equity_val = _to_float(f.get("total_equity"))
        debt_val   = _to_float(f.get("total_debt"))
        cash_val   = _to_float(f.get("cash_and_equivalents", 0.0))
        ebit_ttm   = _to_float(f.get("ebit_ttm"))
        shares_val = _to_float(f.get("shares_outstanding"))

        # book_to_price
        if pd.notna(equity_val) and pd.notna(shares_val) and shares_val != 0 and price_val != 0:
            book_to_price = (equity_val / shares_val) / price_val
        else:
            book_to_price = np.nan

        # EV / EBIT
        mkt_cap = price_val * shares_val if pd.notna(shares_val) else np.nan
        if pd.notna(mkt_cap) and pd.notna(ebit_ttm) and ebit_ttm != 0:
            ev = mkt_cap + (debt_val if pd.notna(debt_val) else 0.0) - (cash_val if pd.notna(cash_val) else 0.0)
            ev_ebit = ev / ebit_ttm
        else:
            ev_ebit = np.nan

        # debt_to_equity
        if pd.notna(debt_val) and pd.notna(equity_val) and equity_val != 0:
            debt_to_equity = debt_val / equity_val
        else:
            debt_to_equity = np.nan

        rows.append({"date": asof, "book_to_price": book_to_price, "ev_ebit": ev_ebit, "debt_to_equity": debt_to_equity})

    return pd.DataFrame(rows).set_index("date").sort_index()


def compute_quality_factors(fund_df, date_range, report_lag_days=REPORT_LAG_DAYS):
    date_index = pd.to_datetime(pd.Index(date_range)).sort_values()
    rows = []
    for asof in date_index:
        asof = pd.to_datetime(asof)
        cutoff = asof - timedelta(days=report_lag_days)
        eligible = fund_df[fund_df["fiscal_period_end"] <= cutoff]
        if eligible.empty:
            rows.append({"date": asof, "gross_margin": np.nan, "operating_margin": np.nan, "roe": np.nan})
            continue

        f = eligible.iloc[-1]
        revenue      = _to_float(f.get("revenue"))
        gross_profit = _to_float(f.get("gross_profit"))
        op_inc       = _to_float(f.get("operating_income"))
        net_inc      = _to_float(f.get("net_income"))
        equity       = _to_float(f.get("total_equity"))

        gross_margin     = gross_profit / revenue if pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0 else np.nan
        operating_margin = op_inc / revenue if pd.notna(op_inc) and pd.notna(revenue) and revenue != 0 else np.nan
        roe              = net_inc / equity if pd.notna(net_inc) and pd.notna(equity) and equity != 0 else np.nan

        rows.append({"date": asof, "gross_margin": gross_margin, "operating_margin": operating_margin, "roe": roe})

    return pd.DataFrame(rows).set_index("date").sort_index()


def compute_momentum_factors(price_df, date_range, mom_window_days=MOM_WINDOW_DAYS):
    price_df = price_df.sort_index()
    date_index = pd.to_datetime(pd.Index(date_range)).sort_values()

    # Precompute OBV
    obv_series = []
    obv = 0
    prev_close = None
    for dt, row in price_df.iterrows():
        c, v = row["close"], row["volume"]
        if prev_close is None:
            prev_close = c
            obv_series.append((dt, obv))
            continue
        if c > prev_close:
            obv += v
        elif c < prev_close:
            obv -= v
        prev_close = c
        obv_series.append((dt, obv))

    obv_df = pd.DataFrame(obv_series, columns=["dt", "obv"]).set_index("dt")

    rows = []
    for asof in date_index:
        asof = pd.to_datetime(asof)
        px_idx = price_df.index[price_df.index <= asof]
        if len(px_idx) == 0:
            rows.append({"date": asof, "mom_6m": np.nan, "obv": np.nan})
            continue
        asof_trading = px_idx[-1]

        # 6m momentum
        mom_start = asof_trading - timedelta(days=mom_window_days)
        hist_mom = price_df.loc[mom_start:asof_trading]
        if len(hist_mom) < 2:
            mom_6m = np.nan
        else:
            mom_6m = hist_mom["close"].iloc[-1] / hist_mom["close"].iloc[0] - 1

        # OBV
        obv_idx = obv_df.index[obv_df.index <= asof_trading]
        obv_val = obv_df.loc[obv_idx[-1], "obv"] if len(obv_idx) > 0 else np.nan

        rows.append({"date": asof, "mom_6m": mom_6m, "obv": obv_val})

    return pd.DataFrame(rows).set_index("date").sort_index()


def build_factor_table(raw_df, monthly_fund_df, tickers=None, report_lag_days=REPORT_LAG_DAYS, mom_window_days=MOM_WINDOW_DAYS):
    raw_df = raw_df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    if tickers is None:
        tickers_price = set(raw_df["ticker"].unique())
        if isinstance(monthly_fund_df.index, pd.MultiIndex) and "ticker" in monthly_fund_df.index.names:
            tickers_fund = set(monthly_fund_df.index.get_level_values("ticker").unique())
        else:
            tickers_fund = set(monthly_fund_df["ticker"].unique())
        tickers = sorted(tickers_price & tickers_fund)

    all_factors = []
    for ticker in tickers:
        px = raw_df[raw_df["ticker"] == ticker].copy()
        if px.empty:
            continue
        price_df = px.sort_values("date").set_index("date")[["close", "volume"]]
        date_range = pd.to_datetime(price_df.index.unique()).sort_values()

        if isinstance(monthly_fund_df.index, pd.MultiIndex) and "ticker" in monthly_fund_df.index.names:
            fd = monthly_fund_df.xs(ticker, level="ticker").copy()
        else:
            fd = monthly_fund_df[monthly_fund_df["ticker"] == ticker].copy()
        if fd.empty:
            continue
        fund_df = fd.reset_index().sort_values("fiscal_period_end")

        value_df   = compute_value_factors(price_df, fund_df, date_range, report_lag_days)
        quality_df = compute_quality_factors(fund_df, date_range, report_lag_days)
        mom_df     = compute_momentum_factors(price_df, date_range, mom_window_days)

        df = pd.concat([value_df, quality_df, mom_df], axis=1)
        df["ticker"] = ticker
        df = df.reset_index().set_index(["ticker", "date"]).sort_index()
        all_factors.append(df)

    if not all_factors:
        return pd.DataFrame()
    return pd.concat(all_factors, axis=0).sort_index()


# ═══════════════════════════════════════════════════════════════════════════
#  Scoring  (cell ffd5965b)
# ═══════════════════════════════════════════════════════════════════════════

def winsorize_series(s, lower=0.01, upper=0.99):
    s = s.astype(float)
    if s.notna().sum() < 10:
        return s
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)


def zscore(s):
    s = s.astype(float)
    if s.notna().sum() < 2:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def score_factors(factors_csv=FULL_FACTORS_CSV, sector_csv=TICKER_SECTOR_CSV, save_path=SCORED_FACTORS_CSV):
    """Load raw factors, compute z-scores and alpha, save scored CSV."""
    factors = pd.read_csv(factors_csv, low_memory=False)
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    factors["ticker"] = factors["ticker"].astype(str).str.upper()
    factors = factors.sort_values(["date", "ticker"]).reset_index(drop=True)

    sector_df = pd.read_csv(sector_csv)
    sector_df["ticker"] = sector_df["ticker"].astype(str).str.upper()
    factors = factors.merge(sector_df, on="ticker", how="left")

    raw_cols = ["book_to_price", "ev_ebit", "debt_to_equity",
                "gross_margin", "operating_margin", "roe", "mom_6m", "obv"]
    for col in raw_cols:
        factors[col] = pd.to_numeric(factors[col], errors="coerce")

    factors[raw_cols] = factors.groupby(["date", "sector"])[raw_cols].transform(winsorize_series)

    # Directional components
    factors["val_book_to_price"]   = factors["book_to_price"]
    factors["val_ev_ebit"]         = -factors["ev_ebit"]
    factors["val_debt_to_equity"]  = -factors["debt_to_equity"]
    factors["qlt_gross_margin"]    = factors["gross_margin"]
    factors["qlt_operating_margin"]= factors["operating_margin"]
    factors["qlt_roe"]             = factors["roe"]
    factors["mom_mom6"]            = factors["mom_6m"]
    factors["mom_obv"]             = factors["obv"]

    value_cols   = ["val_book_to_price", "val_ev_ebit", "val_debt_to_equity"]
    quality_cols = ["qlt_gross_margin", "qlt_operating_margin", "qlt_roe"]
    momentum_cols= ["mom_mom6", "mom_obv"]

    for col in value_cols + quality_cols:
        factors[col + "_z"] = factors.groupby(["date", "sector"])[col].transform(zscore)
    for col in momentum_cols:
        factors[col + "_z"] = factors.groupby("date")[col].transform(zscore)

    value_z_cols   = [c + "_z" for c in value_cols]
    quality_z_cols = [c + "_z" for c in quality_cols]
    mom_z_cols     = [c + "_z" for c in momentum_cols]

    factors["value_score"]    = factors[value_z_cols].mean(axis=1, skipna=True)
    factors["quality_score"]  = factors[quality_z_cols].mean(axis=1, skipna=True)
    factors["momentum_score"] = factors[mom_z_cols].mean(axis=1, skipna=True)
    factors["alpha_score"]    = (factors["value_score"] + factors["quality_score"] + factors["momentum_score"]) / 3.0

    def decile_rank(s, n=10):
        r = s.rank(method="first")
        return pd.qcut(r, q=n, labels=False, duplicates="drop") + 1

    factors["alpha_decile"] = factors.groupby("date")["alpha_score"].transform(lambda s: decile_rank(s))

    factors.to_csv(save_path, index=True)
    print(f"  Saved scored factors ({factors.shape}) to {save_path}")
    return factors


# ═══════════════════════════════════════════════════════════════════════════
#  Backtest  (cells af30eb28, 55f40628, a9f12883)
# ═══════════════════════════════════════════════════════════════════════════

def run_long_top_decile_backtest(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    rebal_dates: list,
    top_decile: int = TOP_DECILE,
    decile_col: str = "alpha_decile",
) -> pd.DataFrame:
    """Long-only, equal-weight, top-decile strategy. Benchmark = SPY."""
    print("  Fetching SPY OHLCV from IBKR for benchmark...")
    spy_raw = fetch_ohlcv_ibkr(["SPY"], client_id=IBKR_CLIENT_BACKTEST)
    spy_raw["date"] = pd.to_datetime(spy_raw["date"])
    spy_raw = spy_raw.sort_values("date").reset_index(drop=True)
    spy_raw["bench_ret"] = spy_raw["close"].pct_change()
    bench_ret = spy_raw[["date", "bench_ret"]].dropna(subset=["bench_ret"]).reset_index(drop=True)

    port_ret_records = []
    for i in range(len(rebal_dates) - 1):
        reb_date = rebal_dates[i]
        next_reb_date = rebal_dates[i + 1]
        f_slice = factors[factors["date"] == reb_date].copy()
        long_names = f_slice.loc[f_slice[decile_col] == top_decile, "ticker"].unique()
        if len(long_names) == 0:
            continue
        w = 1.0 / len(long_names)
        mask = (prices["date"] > reb_date) & (prices["date"] <= next_reb_date)
        p_slice = prices.loc[mask & prices["ticker"].isin(long_names)].copy()
        if p_slice.empty:
            continue
        p_slice["weight"] = w
        daily_port = (
            p_slice.groupby("date")
            .apply(lambda df: (df["weight"] * df["ret_1d"]).sum())
            .rename("port_ret")
            .reset_index()
        )
        port_ret_records.append(daily_port)

    if not port_ret_records:
        return pd.DataFrame(columns=["date", "port_ret", "bench_ret"])

    port_ret_df = pd.concat(port_ret_records, ignore_index=True).sort_values("date").reset_index(drop=True)
    results = pd.merge(port_ret_df, bench_ret, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return results


def compute_portfolio_metrics(results: pd.DataFrame, freq: int = 252) -> dict:
    """Return dict of portfolio metrics (matching notebook output)."""
    df = results.dropna(subset=["port_ret", "bench_ret"]).copy()
    if df.empty:
        return {}
    df = df.sort_values("date").reset_index(drop=True)
    r = df["port_ret"]

    total_pnl = (1 + r).prod() - 1
    n_days = len(r)
    ann_return = (1 + total_pnl) ** (freq / n_days) - 1 if n_days > 0 else np.nan
    ann_vol = r.std(ddof=0) * np.sqrt(freq) if r.std(ddof=0) > 0 else np.nan
    sharpe  = (r.mean() / r.std(ddof=0)) * np.sqrt(freq) if r.std(ddof=0) > 0 else np.nan

    equity = (1 + r).cumprod()
    max_dd = (equity / equity.cummax() - 1.0).min()

    df["month"] = df["date"].dt.to_period("M")
    month_ret = df.groupby("month")["port_ret"].apply(lambda x: (1 + x).prod() - 1)
    win_rate = float((month_ret > 0).mean()) if len(month_ret) > 0 else np.nan

    return {
        "ann_return": round(float(ann_return), 4),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(float(max_dd), 4),
        "ann_vol": round(float(ann_vol), 4),
        "win_rate_monthly": round(float(win_rate), 4),
        "num_months": int(len(month_ret)),
    }


def ttest_monthly_returns(results):
    df = results.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly_ret = df.groupby("month")["port_ret"].apply(lambda x: (1 + x).prod() - 1).to_timestamp()
    t_stat, p_two = ttest_1samp(monthly_ret, popmean=0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    return {"monthly_t_stat": float(t_stat), "monthly_p_value_one_sided": float(p_one), "num_months": len(monthly_ret)}


# ═══════════════════════════════════════════════════════════════════════════
#  JSON helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_metrics_json() -> dict:
    if BACKTEST_METRICS_JSON.exists():
        with open(BACKTEST_METRICS_JSON) as f:
            return json.load(f)
    return {}


def _save_metrics_json(data: dict):
    with open(BACKTEST_METRICS_JSON, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved metrics JSON to {BACKTEST_METRICS_JSON}")


def _extract_current_picks(scored_df: pd.DataFrame) -> tuple[str, list[dict], list[str]]:
    """
    From scored factors, get the latest rebalance date's top-decile picks.
    Returns (rebal_date_str, holdings_list, tickers_list).
    """
    scored_df["date"] = pd.to_datetime(scored_df["date"])
    last_date = scored_df["date"].max()
    f_slice = scored_df[scored_df["date"] == last_date].copy()
    top = f_slice[f_slice["alpha_decile"] == TOP_DECILE].sort_values("alpha_score", ascending=False)

    sector_df = pd.read_csv(TICKER_SECTOR_CSV)
    sector_map = dict(zip(sector_df["ticker"].str.upper(), sector_df["sector"]))

    n = len(top)
    w = round(1.0 / n, 2) if n > 0 else 0

    holdings = []
    tickers = []
    for _, row in top.iterrows():
        t = row["ticker"]
        tickers.append(t)
        holdings.append({
            "ticker": t,
            "sector": sector_map.get(t, "Unknown"),
            "weight": w,
            "alpha_score": round(float(row["alpha_score"]), 3),
            "value_score": round(float(row["value_score"]), 3),
            "quality_score": round(float(row["quality_score"]), 3),
            "momentum_score": round(float(row["momentum_score"]), 3),
        })

    return str(last_date.date()), holdings, tickers


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry points
# ═══════════════════════════════════════════════════════════════════════════

def run_monthly():
    """Default mode: EDGAR fundamentals + IBKR prices + rescore + update picks."""
    print("\n=== STEP 1/5: Fetching EDGAR fundamentals ===")
    fetch_all_fundamentals(SP500_TICKERS)

    print("\n=== STEP 2/5: Fetching IBKR prices ===")
    t0 = time.time()
    raw_df = fetch_ohlcv_ibkr(SP500_TICKERS, client_id=IBKR_CLIENT_DATA)
    print(f"  Price data: {raw_df.shape[0]} rows in {(time.time()-t0)/60:.1f} min")

    print("\n=== STEP 3/5: Building factor table ===")
    monthly_fund = prepare_monthly_fundamentals()
    factors_table = build_factor_table(raw_df, monthly_fund)
    factors_table.to_csv(FULL_FACTORS_CSV, index=True)
    print(f"  Saved full_factors_table.csv ({factors_table.shape})")

    print("\n=== STEP 4/5: Scoring factors ===")
    scored = score_factors()

    print("\n=== STEP 5/5: Updating picks in backtest_metrics.json ===")
    rebal_date_str, holdings, tickers = _extract_current_picks(scored)
    month_key = rebal_date_str[:7]  # "YYYY-MM"

    metrics = _load_metrics_json()
    metrics["last_rebalance_date"] = rebal_date_str
    metrics["current_holdings"] = holdings
    if "monthly_picks" not in metrics:
        metrics["monthly_picks"] = {}
    metrics["monthly_picks"][month_key] = {"date": rebal_date_str, "tickers": tickers}
    _save_metrics_json(metrics)

    print("\nDone (monthly update).")


def run_full():
    """Full mode: monthly update + full backtest → regenerate metrics."""
    # First do the monthly update steps
    print("\n=== STEP 1/6: Fetching EDGAR fundamentals ===")
    fetch_all_fundamentals(SP500_TICKERS)

    print("\n=== STEP 2/6: Fetching IBKR prices ===")
    t0 = time.time()
    raw_df = fetch_ohlcv_ibkr(SP500_TICKERS, client_id=IBKR_CLIENT_DATA)
    print(f"  Price data: {raw_df.shape[0]} rows in {(time.time()-t0)/60:.1f} min")

    print("\n=== STEP 3/6: Building factor table ===")
    monthly_fund = prepare_monthly_fundamentals()
    factors_table = build_factor_table(raw_df, monthly_fund)
    factors_table.to_csv(FULL_FACTORS_CSV, index=True)
    print(f"  Saved full_factors_table.csv ({factors_table.shape})")

    print("\n=== STEP 4/6: Scoring factors ===")
    scored = score_factors()

    print("\n=== STEP 5/6: Running full backtest ===")
    # Prepare prices_df
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    prices_df = raw_df.sort_values(["ticker", "date"]).copy()
    prices_df["ret_1d"] = prices_df.groupby("ticker")["close"].pct_change()
    prices_df = prices_df[["date", "ticker", "close", "ret_1d"]].dropna(subset=["ret_1d"])

    scored["date"] = pd.to_datetime(scored["date"])
    common_dates = sorted(set(scored["date"].unique()) & set(prices_df["date"].unique()))
    factors_aligned = scored[scored["date"].isin(common_dates)].copy()

    f_dates = pd.Series(sorted(factors_aligned["date"].unique()))
    rebal_dates = f_dates.groupby([f_dates.dt.year, f_dates.dt.month]).max().tolist()

    backtest_results = run_long_top_decile_backtest(factors_aligned, prices_df, rebal_dates)
    bt_metrics = compute_portfolio_metrics(backtest_results)

    print("\n=== STEP 6/6: Saving backtest_metrics.json ===")
    rebal_date_str, holdings, tickers = _extract_current_picks(scored)
    month_key = rebal_date_str[:7]

    old = _load_metrics_json()
    old_picks = old.get("monthly_picks", {})

    metrics = {
        "computed_at": datetime.now().isoformat(),
        "backtest_start": str(backtest_results["date"].min().date()) if not backtest_results.empty else None,
        "backtest_end": str(backtest_results["date"].max().date()) if not backtest_results.empty else None,
        **bt_metrics,
        "last_rebalance_date": rebal_date_str,
        "current_holdings": holdings,
        "monthly_picks": old_picks,
    }
    metrics["monthly_picks"][month_key] = {"date": rebal_date_str, "tickers": tickers}
    _save_metrics_json(metrics)

    print("\nDone (full update).")


# ═══════════════════════════════════════════════════════════════════════════
#  __main__
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPY Factor Strategy — Data Updater")
    parser.add_argument("--full", action="store_true", help="Run full backtest (annual / universe change)")
    args = parser.parse_args()

    if args.full:
        run_full()
    else:
        run_monthly()
