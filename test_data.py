"""Tests for the market-data preparation pipeline."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import (
    calculate_log_return,
    discretize_market_returns,
    download_market_data,
    extract_close_prices,
    extract_market_returns,
    fit_return_bins,
    load_portfolio_input,
    save_processed_data,
    split_time_series,
    validate_downloaded_data,
)


def test_load_portfolio_input_reads_valid_csv(tmp_path):
    portfolio = pd.DataFrame(
        {
            "ticker": ["RY.TO", "ENB.TO"],
            "weight": [0.6, 0.4],
        }
    )
    csv_path = tmp_path / "portfolio.csv"
    portfolio.to_csv(csv_path, index=False)

    result = load_portfolio_input(csv_path)

    pd.testing.assert_frame_equal(result, portfolio)


def test_load_portfolio_input_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_portfolio_input(tmp_path / "missing.csv")


def test_load_portfolio_input_rejects_missing_required_column(tmp_path):
    csv_path = tmp_path / "portfolio.csv"
    pd.DataFrame({"ticker": ["RY.TO"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_portfolio_input(csv_path)


def test_download_market_data_passes_all_tickers_to_yfinance():
    portfolio = pd.DataFrame(
        {
            "ticker": ["RY.TO", "ENB.TO"],
            "weight": [0.6, 0.4],
        }
    )
    downloaded = pd.DataFrame({"placeholder": [1.0]})

    with patch("src.data.yf.download", return_value=downloaded) as mock_download:
        result = download_market_data(
            portfolio,
            market_ticker="^GSPTSE",
            interval="1d",
            period="1y",
        )

    assert result is downloaded
    mock_download.assert_called_once_with(
        ["^GSPTSE", "RY.TO", "ENB.TO"],
        period="1y",
        interval="1d",
        auto_adjust=True,
    )


def test_extract_close_prices_returns_one_column_per_ticker():
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "^GSPTSE"),
            ("Close", "RY.TO"),
            ("Open", "^GSPTSE"),
            ("Open", "RY.TO"),
        ],
        names=["Price", "Ticker"],
    )
    historical_data = pd.DataFrame(
        [
            [20_000.0, 100.0, 19_900.0, 99.0],
            [20_100.0, 101.0, 20_000.0, 100.0],
        ],
        index=dates,
        columns=columns,
    )

    result = extract_close_prices(historical_data)

    assert list(result.columns) == ["^GSPTSE", "RY.TO"]
    np.testing.assert_allclose(result["RY.TO"], [100.0, 101.0])


def test_validate_downloaded_data_accepts_valid_prices():
    prices = pd.DataFrame(
        {
            "^GSPTSE": [20_000.0, 20_100.0],
            "RY.TO": [100.0, 101.0],
        },
        index=pd.date_range("2024-01-02", periods=2, freq="D"),
    )

    result = validate_downloaded_data(prices, ["^GSPTSE", "RY.TO"])

    assert result is None


def test_validate_downloaded_data_rejects_missing_ticker():
    prices = pd.DataFrame(
        {"^GSPTSE": [20_000.0, 20_100.0]},
        index=pd.date_range("2024-01-02", periods=2, freq="D"),
    )

    with pytest.raises(ValueError, match="RY.TO"):
        validate_downloaded_data(prices, ["^GSPTSE", "RY.TO"])


def test_calculate_and_extract_market_log_returns():
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    prices = pd.DataFrame(
        {
            "^GSPTSE": [100.0, 110.0, 121.0],
            "RY.TO": [50.0, 50.0, 55.0],
        },
        index=dates,
    )

    log_returns = calculate_log_return(prices)
    market_returns = extract_market_returns(log_returns, "^GSPTSE")

    assert market_returns.name == "market_return"
    assert list(market_returns.index) == list(dates[1:])
    np.testing.assert_allclose(market_returns, [np.log(1.1), np.log(1.1)])


def test_split_time_series_has_no_overlapping_boundary():
    dates = pd.date_range("2023-12-29", periods=5, freq="D")
    returns = pd.Series(np.arange(5, dtype=float), index=dates)

    train, test = split_time_series(returns, "2024-01-01")

    assert (train.index < pd.Timestamp("2024-01-01")).all()
    assert (test.index >= pd.Timestamp("2024-01-01")).all()
    assert train.index.intersection(test.index).empty


def test_fit_bins_and_discretize_returns():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    training_returns = pd.Series(np.arange(10, dtype=float), index=dates)
    labels = [
        "very_negative",
        "negative",
        "neutral",
        "positive",
        "very_positive",
    ]

    bin_edges = fit_return_bins(training_returns, number_of_symbols=5)
    observations = discretize_market_returns(
        training_returns,
        bin_edges,
        labels,
    )

    assert len(bin_edges) == 6
    assert np.isneginf(bin_edges[0])
    assert np.isposinf(bin_edges[-1])
    assert np.all(np.diff(bin_edges) > 0)
    assert observations.name == "observation"
    assert not observations.isna().any()
    assert observations.value_counts().sort_index().tolist() == [2, 2, 2, 2, 2]


def test_discretization_uses_infinite_edges_for_future_extremes():
    training_returns = pd.Series(np.arange(10, dtype=float))
    labels = [
        "very_negative",
        "negative",
        "neutral",
        "positive",
        "very_positive",
    ]
    bin_edges = fit_return_bins(training_returns, number_of_symbols=5)
    test_returns = pd.Series([-100.0, 100.0])

    observations = discretize_market_returns(
        test_returns,
        bin_edges,
        labels,
    )

    assert observations.iloc[0] == "very_negative"
    assert observations.iloc[1] == "very_positive"


def test_save_processed_data_writes_expected_files(tmp_path):
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    close_prices = pd.DataFrame(
        {
            "^GSPTSE": [100.0, 101.0, 102.0],
            "RY.TO": [50.0, 51.0, 52.0],
        },
        index=dates,
    )
    log_returns = calculate_log_return(close_prices)
    training_returns = pd.Series(
        [0.01, -0.01],
        index=dates[:2],
        name="market_return",
    )
    test_returns = pd.Series(
        [0.02],
        index=dates[2:],
        name="market_return",
    )
    bin_edges = np.array([-np.inf, 0.0, np.inf])
    training_symbols = pd.Series(
        ["positive", "negative"],
        index=dates[:2],
        name="observation",
    )
    test_symbols = pd.Series(
        ["positive"],
        index=dates[2:],
        name="observation",
    )

    output_paths = save_processed_data(
        output_dir=tmp_path,
        close_prices=close_prices,
        log_returns=log_returns,
        training_returns=training_returns,
        test_returns=test_returns,
        bin_edges=bin_edges,
        training_symbols=training_symbols,
        test_symbols=test_symbols,
    )

    assert all(path.is_file() for path in output_paths.values())

    saved_prices = pd.read_csv(
        output_paths["close_prices"],
        index_col="date",
    )
    saved_bins = pd.read_csv(output_paths["bin_edges"])

    assert list(saved_prices.columns) == ["^GSPTSE", "RY.TO"]
    np.testing.assert_allclose(saved_prices["RY.TO"], [50.0, 51.0, 52.0])
    np.testing.assert_allclose(saved_bins["edge"], bin_edges)
