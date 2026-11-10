"""Run the complete project workflow from data preparation to saved results.

This module will connect the focused modules without placing analysis logic in the
notebook. The notebook can then load saved tables and figures for interpretation.
"""
from src.data import (
    load_portfolio_input,
    download_market_data,
    extract_close_prices,
    validate_downloaded_data,
    calculate_log_return,
    extract_market_returns,
    split_time_series,
    fit_return_bins,
    discretize_market_returns,
    save_processed_data,
)

from src.hmm import (
    create_initial_transition_guess,
    create_initial_emission_guess,
    fit_discrete_hmm,
    decode_observations,
    save_hmm_outputs,
)
from pathlib import Path
import pandas as pd


def run_pipeline(
    portfolio_path: str | Path,
    market_ticker: str,
    interval: str,
    period: str,
    split_date: str,
    possible_states: list[str],
    possible_symbols: list[str],
    output_root: str | Path,
) -> dict:
    """
    Run data preparation, HMM fitting, chronological decoding,
    and output saving.

    Implementation steps:

    1. Load the portfolio CSV with load_portfolio_input(). The result is
       the portfolio DataFrame containing ticker and weight columns.

    2. Download the market index and portfolio prices with
       download_market_data(). Pass the portfolio DataFrame together with
       market_ticker, interval, and period.

    3. Extract adjusted closing prices with extract_close_prices(). Build
       expected_tickers from market_ticker and the portfolio ticker column,
       then call validate_downloaded_data().

    4. Calculate log returns with calculate_log_return(). Extract the market
       index return Series with extract_market_returns(). The complete
       log-return DataFrame must be retained for the later portfolio
       backtest.

    5. Split the market returns with split_time_series(). The training
       period must contain dates before split_date, and the test period must
       begin on split_date.

    6. Fit the return-category boundaries with fit_return_bins() using only
       training_returns. Set number_of_symbols to len(possible_symbols).

    7. Call discretize_market_returns() twice. Use the same training bin
       edges for both training_returns and test_returns. The results are the
       training and test observation-symbol Series.

    8. Create the initial transition matrix with
       create_initial_transition_guess(len(possible_states)). Create the
       initial emission matrix with
       create_initial_emission_guess(possible_symbols).

    9. Fit the model with fit_discrete_hmm(). Pass only training_symbols to
       the fitting function together with the state names, symbol names, and
       initial probability matrices.

    10. Concatenate training_symbols and test_symbols in chronological order
        with pandas.concat(). Decode the combined sequence once with
        decode_observations(). The HMM parameters remain fitted only on the
        training period, while decoding the combined sequence preserves the
        filtered-state distribution across the train/test boundary.

    11. Select the training and test rows from the decoded probability
        DataFrames using the original training and test date indices. The
        test filtered probabilities will later be used by the strategy.
        Posterior probabilities are for retrospective analysis only.

    12. Create data/processed and outputs/tables/hmm paths under
        output_root. Save processed data with save_processed_data() and save
        fitted matrices and decoded results with save_hmm_outputs().

    13. Return a dictionary containing the prepared data, fitted model,
        complete decoded results, period-specific filtered probabilities,
        and generated output paths. This allows later modules to use the
        results without downloading or fitting the model again.

    The returned dictionary should have the general structure:

        {
            "data": {
                "portfolio": portfolio_df,
                "close_prices": close_prices,
                "log_returns": log_returns,
                "training_returns": training_returns,
                "test_returns": test_returns,
                "bin_edges": bin_edges,
                "training_symbols": training_symbols,
                "test_symbols": test_symbols,
            },
            "model": model,
            "decoding": {
                "all_results": decoded_results,
                "training_filtered_probabilities": ...,
                "test_filtered_probabilities": ...,
            },
            "saved_paths": {
                "data": data_paths,
                "hmm": hmm_paths,
            },
        }
    """

    output_root = Path(output_root)

    portfolio_df = load_portfolio_input(portfolio_path)

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
        number_of_symbols=len(possible_symbols),
    )

    training_symbols = discretize_market_returns(
        training_returns,
        bin_edges,
        possible_symbols,
    )

    test_symbols = discretize_market_returns(
        test_returns,
        bin_edges,
        possible_symbols,
    )

    transition_guess = create_initial_transition_guess(
        len(possible_states)
    )

    emission_guess = create_initial_emission_guess(
        possible_symbols
    )

    model = fit_discrete_hmm(
        training_symbols,
        possible_states,
        possible_symbols,
        transition_guess,
        emission_guess,
    )

    all_symbols = pd.concat(
        [training_symbols, test_symbols]
    )

    decoded_results = decode_observations(
        model,
        all_symbols,
    )

    filtered_probabilities = decoded_results[
        "filtered_probabilities"
    ]
    posterior_probabilities = decoded_results[
        "posterior_probabilities"
    ]

    training_filtered_probabilities = filtered_probabilities.loc[
        training_symbols.index
    ].copy()
    test_filtered_probabilities = filtered_probabilities.loc[
        test_symbols.index
    ].copy()
    training_posterior_probabilities = posterior_probabilities.loc[
        training_symbols.index
    ].copy()
    test_posterior_probabilities = posterior_probabilities.loc[
        test_symbols.index
    ].copy()

    processed_dir = output_root / "data" / "processed"
    hmm_output_dir = output_root / "outputs" / "tables" / "hmm"

    data_paths = save_processed_data(
        output_dir=processed_dir,
        close_prices=close_prices,
        log_returns=log_returns,
        training_returns=training_returns,
        test_returns=test_returns,
        bin_edges=bin_edges,
        training_symbols=training_symbols,
        test_symbols=test_symbols,
    )

    hmm_paths = save_hmm_outputs(
        output_dir=hmm_output_dir,
        model=model,
        decoded_results=decoded_results,
    )

    return {
        "data": {
            "portfolio": portfolio_df,
            "close_prices": close_prices,
            "log_returns": log_returns,
            "training_returns": training_returns,
            "test_returns": test_returns,
            "bin_edges": bin_edges,
            "training_symbols": training_symbols,
            "test_symbols": test_symbols,
        },
        "model": model,
        "decoding": {
            "all_results": decoded_results,
            "training_filtered_probabilities": (
                training_filtered_probabilities
            ),
            "test_filtered_probabilities": (
                test_filtered_probabilities
            ),
            "training_posterior_probabilities": (
                training_posterior_probabilities
            ),
            "test_posterior_probabilities": (
                test_posterior_probabilities
            ),
        },
        "saved_paths": {
            "data": data_paths,
            "hmm": hmm_paths,
        },
    }


def main() -> None:
    """
    Run the pipeline using the project's standard settings.

    Implementation steps:

    1. Find the project root from the location of this file.
    2. Define the portfolio path, market ticker, interval, period, and
       chronological split date.
    3. Define the ordered hidden-state and observation-symbol labels.
    4. Call run_pipeline() with those settings.
    5. Print a short summary of saved paths or fitted matrices. Detailed
       analysis and plotting should remain outside main().

    To run this, run python -m pipline.py
    """
    project_root = Path(__file__).resolve().parents[1]

    possible_states = [
        "low_risk",
        "high_risk",
    ]

    possible_symbols = [
        "very_negative",
        "negative",
        "neutral",
        "positive",
        "very_positive",
    ]

    results = run_pipeline(
        portfolio_path=project_root / "data" / "portfolio.csv",
        market_ticker="^GSPTSE",
        interval="1d",
        period="10y",
        split_date="2024-01-01",
        possible_states=possible_states,
        possible_symbols=possible_symbols,
        output_root=project_root,
    )

    print("Pipeline completed.")
    print("\nTransition matrix:")
    print(results["model"].p)
    print("\nEmission matrix:")
    print(results["model"].e)
    print("\nSaved paths:")
    print(results["saved_paths"])


if __name__ == "__main__":
    main()
