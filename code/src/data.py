# Run once if yfinance is not installed.
# %pip install yfinance --upgrade --no-cache-dir
import pydtmc 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import yfinance as yf
from scipy.stats import norm
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

pd.set_option("display.max_columns", 140)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

print("yfinance version:", yf.__version__)

sector_etfs = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer": "XLY",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Real Estate": "XLRE",
}

# Default example portfolio. Replace this with your own portfolio_input dataframe or CSV path.
company_universe = {
    "Technology": ["AAPL", "MSFT", "NVDA", "CRM", "ORCL", "ADBE", "CSCO", "INTC"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO"],
    "Industrials": ["CAT", "BA", "GE", "HON", "UNP", "UPS", "LMT", "DE"],
    "Consumer": ["AMZN", "WMT", "HD", "MCD", "SBUX", "NKE", "TGT", "COST"],
    "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV", "TMO", "ABT", "LLY"],
    "Financials": ["JPM", "BAC", "GS", "MS", "C", "WFC", "USB", "PNC"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "O", "PSA", "CCI", "DLR"],
}

# Option A: leave as None to use company_universe above.
# Option B: set to a dataframe with at least a ticker column.
# Option C: set to a CSV path with at least a ticker column.
# Supported optional columns include sector_group/sector, sector_etf, ead/exposure, and lgd.
portfolio_input = None

# Example custom input with the minimum one-company portfolio:
# portfolio_input = pd.DataFrame({
#     "ticker": ["AAPL"],
#     "sector_group": ["Technology"],
#     "ead": [100_000_000],
#     "lgd": [0.45],
# })


def standardize_portfolio_input(portfolio_input=None, company_universe=None, sector_etfs=None):
    sector_etfs = sector_etfs or {}

    if portfolio_input is None:
        if not company_universe:
            raise ValueError("Provide either portfolio_input or company_universe.")
        rows = []
        for sector_name, sector_tickers in company_universe.items():
            for ticker in sector_tickers:
                rows.append({
                    "ticker": ticker,
                    "sector_group": sector_name,
                    "sector_etf": sector_etfs.get(sector_name, "SPY"),
                })
        portfolio_df = pd.DataFrame(rows)
    elif isinstance(portfolio_input, (str, bytes)):
        portfolio_df = pd.read_csv(portfolio_input)
    else:
        portfolio_df = pd.DataFrame(portfolio_input).copy()

    if portfolio_df.empty:
        raise ValueError("Portfolio must contain at least one company.")

    normalized_columns = {col.lower().strip(): col for col in portfolio_df.columns}
    column_aliases = {
        "ticker": ["ticker", "symbol", "borrower_ticker", "company_ticker", "name"],
        "sector_group": ["sector_group", "sector", "industry_group", "gics_sector"],
        "sector_etf": ["sector_etf", "factor_etf", "sector_factor", "etf"],
    }

    rename_map = {}
    for standard_name, aliases in column_aliases.items():
        for alias in aliases:
            if alias in normalized_columns:
                rename_map[normalized_columns[alias]] = standard_name
                break
    portfolio_df = portfolio_df.rename(columns=rename_map)

    if "ticker" not in portfolio_df.columns:
        raise ValueError("Portfolio input must include a ticker/symbol column.")

    portfolio_df["ticker"] = portfolio_df["ticker"].astype(str).str.upper().str.strip()
    portfolio_df = portfolio_df.loc[portfolio_df["ticker"].ne("")].drop_duplicates("ticker")

    if portfolio_df.empty:
        raise ValueError("Portfolio must contain at least one valid ticker.")

    if "sector_group" not in portfolio_df.columns:
        portfolio_df["sector_group"] = "User Portfolio"
    portfolio_df["sector_group"] = portfolio_df["sector_group"].fillna("User Portfolio").astype(str).str.strip()
    portfolio_df.loc[portfolio_df["sector_group"].eq(""), "sector_group"] = "User Portfolio"

    if "sector_etf" not in portfolio_df.columns:
        portfolio_df["sector_etf"] = portfolio_df["sector_group"].map(sector_etfs)
    else:
        portfolio_df["sector_etf"] = portfolio_df["sector_etf"].fillna(
            portfolio_df["sector_group"].map(sector_etfs)
        )
    portfolio_df["sector_etf"] = portfolio_df["sector_etf"].fillna("SPY").astype(str).str.upper().str.strip()
    portfolio_df.loc[portfolio_df["sector_etf"].eq(""), "sector_etf"] = "SPY"

    ordered_cols = ["ticker", "sector_group", "sector_etf"]
    optional_cols = [col for col in ["ead", "lgd"] if col in portfolio_df.columns]
    remaining_cols = [col for col in portfolio_df.columns if col not in ordered_cols + optional_cols]
    return portfolio_df[ordered_cols + optional_cols + remaining_cols].reset_index(drop=True)


company_df = standardize_portfolio_input(
    portfolio_input=portfolio_input,
    company_universe=company_universe,
    sector_etfs=sector_etfs,
)

print("Number of companies:", len(company_df))
print("Number of sector groups:", company_df["sector_group"].nunique())
company_df.head()


