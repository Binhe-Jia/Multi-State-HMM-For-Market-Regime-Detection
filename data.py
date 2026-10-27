# Run once if yfinance is not installed.
# %pip install yfinance --upgrade --no-cache-dir
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

pd.set_option("display.max_columns", 140)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

print("yfinance version:", yf.__version__)



def load_portfolio_input(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a portfolio from a CSV file.

    The CSV must contain one asset per row.

    Required columns:
        ticker, weight

    Optional columns:
        name, sector

    Returns:
        A portfolio DataFrame.
    """
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Portfolio file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)


    if df.empty:
        raise ValueError("Portfolio dataset is empty.")

    required_columns = {"ticker", "weight"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df
    

    

def download_market_data(
    df: pd.DataFrame,
    market_ticker: str,
    interval: str,
    period: str,
) -> pd.DataFrame:
    """
    Download TSX index and portfolio market data.
    Note that market_ticker represents the S&P/TSX Composite and should be added separately because it is not necessarily part of the custom portfolio.
    """
    portfolio_tickers = df["ticker"].tolist()
    download_tickers = [market_ticker] + portfolio_tickers

    historical_data = yf.download(
        download_tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
    )

    return historical_data



def extract_close_prices(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract closing prices and return one column per ticker.
    """
    return historical_data["Close"].copy()


def validate_downloaded_data(
    close_prices: pd.DataFrame,
    expected_tickers: list[str],
) -> None:
    """
    Validate the closing-price data returned by Yahoo Finance.
    """
    if close_prices.empty:
        raise ValueError("Closing-price data is empty.")

    missing_tickers = set(expected_tickers) - set(close_prices.columns)

    if missing_tickers:
        raise ValueError(
            f"No closing-price column returned for: {missing_tickers}"
        )

    for ticker in expected_tickers:
        ticker_prices = close_prices[ticker]

        if ticker_prices.dropna().empty:
            raise ValueError(f"No valid prices returned for {ticker}.")

        if (ticker_prices.dropna() <= 0).any():
            raise ValueError(f"Nonpositive prices found for {ticker}.")

    if close_prices.index.has_duplicates:
        raise ValueError("Closing-price data contains duplicate dates.")

    if not close_prices.index.is_monotonic_increasing:
        raise ValueError("Closing-price dates are not chronological.")



def calculate_log_return(close_prices:pd.DataFrame):
    """
    Produce the numerical returns used to define observation categories.
    """
    return np.log(close_prices / close_prices.shift(1))



def extract_market_returns(
    log_returns: pd.DataFrame,
    market_ticker: str,
) -> pd.Series:
    """
    Extract the market-index log-return series used by the HMM.
    """
    market_returns = log_returns[market_ticker].dropna().copy()
    market_returns.name = "market_return"

    return market_returns


def split_time_series(
    market_returns: pd.Series,
    split_date: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Split market returns into nonoverlapping training and test periods.

    The split date is the first date of the test period.
    """
    split_date = pd.Timestamp(split_date)

    train = market_returns.loc[
        market_returns.index < split_date
    ].copy()

    test = market_returns.loc[
        market_returns.index >= split_date
    ].copy()

    if train.empty:
        raise ValueError("Training period is empty.")

    if test.empty:
        raise ValueError("Test period is empty.")

    return train, test



def fit_return_bins(
    training_returns: pd.Series,
    number_of_symbols: int = 5,
) -> np.ndarray:
    """
    Calculate quantile bin edges from training returns.

    Returns one more edge than the requested number of symbols.
    """
    if number_of_symbols < 2:
        raise ValueError(
            "At least two observation symbols are required."
        )

    clean_returns = training_returns.dropna()

    if clean_returns.empty:
        raise ValueError("Training returns are empty.")

    quantile_levels = []

    for i in range(1, number_of_symbols):
        quantile_levels.append(i / number_of_symbols)

    quantile_values = clean_returns.quantile(
        quantile_levels
    ).to_numpy()

    bin_edges = np.concatenate(
        ([-np.inf], quantile_values, [np.inf])
    )

    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError(
            "Quantile boundaries are not unique. "
            "Use fewer observation symbols."
        )

    return bin_edges



def discretize_market_returns(
    market_returns: pd.Series,
    bin_edges: np.ndarray,
    symbol_labels: list[str],
) -> pd.Series:
    """
    Convert continuous returns into discrete observation symbols.
    """
    if len(bin_edges) != len(symbol_labels) + 1:
        raise ValueError("Wrong number of edges.")

    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError("Bin edges must be strictly increasing.")

    observations = pd.cut(
        market_returns,
        bins=bin_edges,
        labels=symbol_labels,
        include_lowest=True,
    )

    unassigned = market_returns.notna() & observations.isna()

    if unassigned.any():
        raise ValueError(
            "Some market returns were not assigned an observation symbol."
        )

    observations.name = "observation"

    return observations

    



def save_processed_data(
    output_dir: str | Path,
    close_prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    training_returns: pd.Series,
    test_returns: pd.Series,
    bin_edges: np.ndarray,
    training_symbols: pd.Series,
    test_symbols: pd.Series,
) -> dict[str, Path]:
    """
    Save processed market data and return the generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "close_prices": output_dir / "close_prices.csv",
        "log_returns": output_dir / "log_returns.csv",
        "training_returns": output_dir / "training_returns.csv",
        "test_returns": output_dir / "test_returns.csv",
        "bin_edges": output_dir / "return_bin_edges.csv",
        "training_symbols": output_dir / "training_symbols.csv",
        "test_symbols": output_dir / "test_symbols.csv",
    }

    close_prices.to_csv(
        output_paths["close_prices"],
        index_label="date",
    )
    log_returns.to_csv(
        output_paths["log_returns"],
        index_label="date",
    )
    training_returns.to_frame(
        name="market_return"
    ).to_csv(
        output_paths["training_returns"],
        index_label="date",
    )
    test_returns.to_frame(
        name="market_return"
    ).to_csv(
        output_paths["test_returns"],
        index_label="date",
    )

    bin_edges_df = pd.DataFrame(
        {
            "edge_number": range(len(bin_edges)),
            "edge": bin_edges,
        }
    )
    bin_edges_df.to_csv(
        output_paths["bin_edges"],
        index=False,
    )

    training_symbols.astype("string").to_frame(
        name="observation"
    ).to_csv(
        output_paths["training_symbols"],
        index_label="date",
    )
    test_symbols.astype("string").to_frame(
        name="observation"
    ).to_csv(
        output_paths["test_symbols"],
        index_label="date",
    )

    return output_paths


def main() -> None:
    """
    Run an end-to-end smoke test of the market-data pipeline.
    Note this is just a test for the data.py functionality, and we still allow customized input
    """
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "portfolio.csv"

    market_ticker = "^GSPTSE"
    interval = "1d"
    period = "10y"
    split_date = "2024-01-01"

    symbol_labels = [
        "very_negative",
        "negative",
        "neutral",
        "positive",
        "very_positive",
    ]

    print("Loading portfolio...")
    portfolio_df = load_portfolio_input(csv_path)
    print(portfolio_df)

    print("\nDownloading market data...")
    historical_data = download_market_data(
        portfolio_df,
        market_ticker,
        interval,
        period,
    )

    close_prices = extract_close_prices(historical_data)

    expected_tickers = [
        market_ticker,
        *portfolio_df["ticker"].tolist(),
    ]

    validate_downloaded_data(
        close_prices,
        expected_tickers,
    )

    print(
        "\nDownloaded date range:",
        close_prices.index.min(),
        "to",
        close_prices.index.max(),
    )
    print("Closing-price shape:", close_prices.shape)

    log_returns = calculate_log_return(close_prices)

    market_returns = extract_market_returns(
        log_returns,
        market_ticker,
    )

    training_returns, test_returns = split_time_series(
        market_returns,
        split_date,
    )

    bin_edges = fit_return_bins(
        training_returns,
        number_of_symbols=len(symbol_labels),
    )

    training_symbols = discretize_market_returns(
        training_returns,
        bin_edges,
        symbol_labels,
    )

    test_symbols = discretize_market_returns(
        test_returns,
        bin_edges,
        symbol_labels,
    )

    print("\nTraining observations:", len(training_symbols))
    print("Test observations:", len(test_symbols))
    print("Bin edges:", bin_edges)

    print("\nTraining symbol proportions:")
    print(
        training_symbols.value_counts(
            normalize=True,
        ).sort_index()
    )

    print("\nTest symbol proportions:")
    print(
        test_symbols.value_counts(
            normalize=True,
        ).sort_index()
    )

    processed_dir = project_root / "data" / "processed"

    saved_files = save_processed_data(
        output_dir=processed_dir,
        close_prices=close_prices,
        log_returns=log_returns,
        training_returns=training_returns,
        test_returns=test_returns,
        bin_edges=bin_edges,
        training_symbols=training_symbols,
        test_symbols=test_symbols,
    )

    print("\nSaved processed files:")

    for name, path in saved_files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
