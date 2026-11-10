"""Compare a two-state and a three-state discrete HMM.

Candidate models
----------------
    2 states: ["low", "high"]
    3 states: ["low", "medium", "high"]
States are listed from lower to higher risk, as run_pipeline() requires.

Design (fixed before looking at any test-period result)
-------------------------------------------------------
* One chronological split at SPLIT_DATE = 2024-01-01. There are no folds.
* Each candidate is first run through run_pipeline(), so it uses exactly the
  same data preparation, bin fitting, initial guess, fitting and decoding as
  the rest of the project. Each run saves to its own folder
  (outputs/tables/hmm_2_states/, outputs/tables/hmm_3_states/).
* Responsibility "fit each state count from multiple random
  initializations": after the pipeline fit, the same model is refitted from
  N_RANDOM_RESTARTS random starting points with fit_discrete_hmm(), using the
  pipeline run's own training symbols (same data, no extra download).
* Responsibility "retain converged fits and record log likelihoods":
  every start is recorded; failed starts are kept with their error message
  and excluded from the comparison.
* Responsibility "compare with justified criteria": the model is chosen by
  BIC on the TRAINING period only (AIC and log likelihood are also reported).
  The test-period (2024+) log likelihood is reported for information only and
  is NOT used to choose, following rule 6 of the project outline.
* Responsibility "check stability across initializations": convergence
  rate, spread of training log likelihoods, and the largest difference in
  any matrix entry between the best fit and the other near-best fits.
  (Stability across sample periods is not assessed, because this design
  deliberately uses a single split date.)

Initial-state distribution
--------------------------
PyDTMC does not estimate an initial distribution; it uses a uniform one for
the state before the first observation. It is therefore not counted as a
free parameter.

Label switching
---------------
Baum-Welch can swap state labels. After every fit the states are checked
with the tail-risk score of Section 6 of the outline,
    v_i = sum_l c_l^2 * B[i, l],   c = (-2, -1, 0, 1, 2),
and a column records whether "low" < ("medium" <) "high" in v. For the
stability comparison, matrices are put in risk order first so that
relabelled copies of the same solution count as the same solution.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydtmc
from pydtmc import HiddenMarkovModel

if __package__:
    from .hmm import (
        decode_observations,
        fit_discrete_hmm,
        validate_probability_matrix,
    )
    from .pipeline import run_pipeline
else:
    from hmm import (
        decode_observations,
        fit_discrete_hmm,
        validate_probability_matrix,
    )
    from pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Settings (move to config.py once it exists)
# ---------------------------------------------------------------------------

MARKET_TICKER = "^GSPTSE"
INTERVAL = "1d"
PERIOD = "10y"
SPLIT_DATE = "2024-01-01"

POSSIBLE_SYMBOLS = [
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "very_positive",
]

CANDIDATE_MODELS: dict[int, list[str]] = {
    2: ["low", "high"],
    3: ["low", "medium", "high"],
}

N_RANDOM_RESTARTS = 5          # random starts per model, on top of the pipeline fit
RANDOM_SEED = 348
STAY_PROBABILITY_RANGE = (0.80, 0.98)
EMISSION_CONCENTRATION = 2.0
LOG_LIKELIHOOD_TOLERANCE = 1.0  # fits within this of the best count as "near-best"


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def count_free_parameters(number_of_states: int, number_of_symbols: int) -> int:
    """K(K-1) transition parameters + K(L-1) emission parameters."""
    k, l = number_of_states, number_of_symbols
    return k * (k - 1) + k * (l - 1)


def information_criteria(
    log_likelihood: float,
    number_of_parameters: int,
    number_of_observations: int,
) -> dict[str, float]:
    """AIC = 2p - 2 ell,  BIC = p log(n) - 2 ell. Smaller is better."""
    if number_of_observations <= 0:
        raise ValueError("Number of observations must be positive.")
    return {
        "aic": float(2 * number_of_parameters - 2 * log_likelihood),
        "bic": float(
            number_of_parameters * np.log(number_of_observations)
            - 2 * log_likelihood
        ),
    }


def emission_risk_score(emission_matrix: np.ndarray) -> np.ndarray:
    """Tail-risk score v_i = sum_l c_l^2 B[i, l], c evenly spaced on [-2, 2]."""
    emission_matrix = np.asarray(emission_matrix, dtype=float)
    c = np.linspace(-2.0, 2.0, emission_matrix.shape[1])
    return emission_matrix @ (c**2)


def labels_in_risk_order(emission_matrix: np.ndarray) -> bool:
    """True if the state order (low -> high) has increasing tail-risk score."""
    return bool(np.all(np.diff(emission_risk_score(emission_matrix)) > 0))


def risk_ordered_matrices(
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute states into increasing tail-risk order (for comparisons only)."""
    order = np.argsort(emission_risk_score(emission_matrix), kind="stable")
    return (
        np.asarray(transition_matrix)[np.ix_(order, order)],
        np.asarray(emission_matrix)[order, :],
    )


def training_log_likelihood(
    model: HiddenMarkovModel,
    training_symbols: pd.Series,
) -> float:
    """Log likelihood of the training symbols (uniform initial distribution)."""
    return float(decode_observations(model, training_symbols)["log_probability"])


def test_log_likelihood(
    model: HiddenMarkovModel,
    training_symbols: pd.Series,
    test_symbols: pd.Series,
) -> float:
    """
    Log likelihood of the test symbols given the training history,

        ell(test | train) = ell(train + test) - ell(train).

    This is the same as decoding the test period starting from the filtered
    state distribution on the last training date, which is how run_pipeline()
    carries the filter across the split. Parameters are not refitted.
    Reported for information only.
    """
    all_symbols = pd.concat([training_symbols, test_symbols])
    joint = float(decode_observations(model, all_symbols)["log_probability"])
    return joint - training_log_likelihood(model, training_symbols)


# ---------------------------------------------------------------------------
# Random initializations
# ---------------------------------------------------------------------------


def create_random_initial_guess(
    number_of_states: int,
    number_of_symbols: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One random starting point for Baum-Welch.

    Transition: each diagonal entry ~ Uniform(STAY_PROBABILITY_RANGE) so the
    start still describes persistent regimes; the rest of the row is split
    among the other states with a flat Dirichlet draw.

    Emission: each row ~ Dirichlet(EMISSION_CONCENTRATION * ones). Rows are
    then sorted by tail-risk score so that, like the pipeline's own guess,
    row 0 matches "low" and the last row matches "high".
    """
    k = number_of_states
    low, high = STAY_PROBABILITY_RANGE

    transition_guess = np.zeros((k, k))
    for i in range(k):
        stay = rng.uniform(low, high)
        transition_guess[i, np.arange(k) != i] = (
            rng.dirichlet(np.ones(k - 1)) * (1.0 - stay)
        )
        transition_guess[i, i] = stay

    emission_guess = rng.dirichlet(
        np.full(number_of_symbols, EMISSION_CONCENTRATION), size=k
    )
    emission_guess = emission_guess[
        np.argsort(emission_risk_score(emission_guess), kind="stable")
    ]

    validate_probability_matrix(transition_guess, (k, k), "Transition matrix")
    validate_probability_matrix(
        emission_guess, (k, number_of_symbols), "Emission matrix"
    )
    return transition_guess, emission_guess


# ---------------------------------------------------------------------------
# Fitting one candidate model
# ---------------------------------------------------------------------------


def _fit_record(
    source: str,
    model: HiddenMarkovModel | None,
    training_symbols: pd.Series,
    error: str | None = None,
) -> dict:
    """Summarise one fit (converged or failed) as a dictionary."""
    if model is None:
        return {
            "source": source,
            "converged": False,
            "log_likelihood": np.nan,
            "labels_in_risk_order": np.nan,
            "error": error,
            "model": None,
        }
    return {
        "source": source,
        "converged": True,
        "log_likelihood": training_log_likelihood(model, training_symbols),
        "labels_in_risk_order": labels_in_risk_order(model.e),
        "error": None,
        "model": model,
    }


def run_random_restarts(
    training_symbols: pd.Series,
    possible_states: list[str],
    possible_symbols: list[str],
    n_restarts: int,
    seed: int,
) -> list[dict]:
    """Refit the model from n_restarts random starting points."""
    rng = np.random.default_rng([seed, len(possible_states)])
    records = []
    for restart in range(1, n_restarts + 1):
        transition_guess, emission_guess = create_random_initial_guess(
            len(possible_states), len(possible_symbols), rng
        )
        source = f"random_{restart}"
        try:
            model = fit_discrete_hmm(
                training_symbols,
                possible_states,
                possible_symbols,
                transition_guess,
                emission_guess,
            )
            records.append(_fit_record(source, model, training_symbols))
        except Exception as error:  # PyDTMC raises ValueError if it does not converge
            records.append(
                _fit_record(source, None, training_symbols, f"{type(error).__name__}: {error}")
            )
    return records


def initialization_stability(records: list[dict]) -> dict:
    """How much the fitted model depends on its starting point."""
    converged = [r for r in records if r["converged"]]
    summary = {
        "n_starts": len(records),
        "n_converged": len(converged),
        "log_likelihood_range": np.nan,
        "n_near_best": 0,
        "max_transition_difference": np.nan,
        "max_emission_difference": np.nan,
    }
    if not converged:
        return summary

    best = max(converged, key=lambda r: r["log_likelihood"])
    lls = np.array([r["log_likelihood"] for r in converged])
    near_best = [
        r for r in converged
        if best["log_likelihood"] - r["log_likelihood"] <= LOG_LIKELIHOOD_TOLERANCE
    ]
    best_p, best_e = risk_ordered_matrices(best["model"].p, best["model"].e)
    p_diffs, e_diffs = [], []
    for r in near_best:
        p, e = risk_ordered_matrices(r["model"].p, r["model"].e)
        p_diffs.append(np.abs(p - best_p).max())
        e_diffs.append(np.abs(e - best_e).max())

    summary.update(
        log_likelihood_range=float(lls.max() - lls.min()),
        n_near_best=len(near_best),
        max_transition_difference=float(max(p_diffs)),
        max_emission_difference=float(max(e_diffs)),
    )
    return summary


def fit_candidate(
    number_of_states: int,
    project_root: Path,
    portfolio_path: Path,
    n_restarts: int = N_RANDOM_RESTARTS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Run the pipeline for one state count, then add random restarts.

    If the pipeline's own Baum-Welch fit fails to converge, run_pipeline()
    raises and no data is returned; this is recorded and the caller supplies
    the training data from the other candidate's run for the restarts.
    """
    possible_states = CANDIDATE_MODELS[number_of_states]
    output_name = f"hmm_{number_of_states}_states"
    try:
        pipeline_results = run_pipeline(
            portfolio_path=portfolio_path,
            market_ticker=MARKET_TICKER,
            interval=INTERVAL,
            period=PERIOD,
            split_date=SPLIT_DATE,
            possible_states=possible_states,
            possible_symbols=POSSIBLE_SYMBOLS,
            output_root=project_root,
        )
        pipeline_error = None
    except ValueError as error:
        pipeline_results = None
        pipeline_error = f"{type(error).__name__}: {error}"

    return {
        "number_of_states": number_of_states,
        "possible_states": possible_states,
        "output_name": output_name,
        "pipeline_results": pipeline_results,
        "pipeline_error": pipeline_error,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def check_same_data(runs: list[dict]) -> dict:
    """
    Both pipeline runs download data separately. Stop if the training
    returns, test returns or bin edges differ, because the likelihoods would
    then not be comparable.
    """
    successful = [r["pipeline_results"] for r in runs if r["pipeline_results"] is not None]
    if not successful:
        errors = "; ".join(f"K={r['number_of_states']}: {r['pipeline_error']}" for r in runs)
        raise RuntimeError(f"No pipeline run succeeded. {errors}")

    reference = successful[0]["data"]
    for other in successful[1:]:
        data = other["data"]
        if not reference["training_returns"].equals(data["training_returns"]):
            raise RuntimeError("Training returns differ between pipeline runs.")
        if not reference["test_returns"].equals(data["test_returns"]):
            raise RuntimeError("Test returns differ between pipeline runs.")
        if not np.array_equal(reference["bin_edges"], data["bin_edges"]):
            raise RuntimeError("Bin edges differ between pipeline runs.")
    return reference


def select_model(comparison_table: pd.DataFrame) -> dict:
    """
    Decision rule: the candidate with the lowest training-period BIC.

    BIC is used because it penalises the extra parameters of the
    three-state model (18 vs 10 with five symbols) more strongly than AIC,
    which favours the more interpretable model unless the extra state is
    clearly supported by the training data. AIC's choice is reported too.
    """
    table = comparison_table.dropna(subset=["bic"])
    if table.empty:
        raise RuntimeError("No candidate model converged.")
    bic_choice = int(table.loc[table["bic"].idxmin(), "number_of_states"])
    aic_choice = int(table.loc[table["aic"].idxmin(), "number_of_states"])

    by_k = table.set_index("number_of_states")
    lr_statistic = None
    if {2, 3}.issubset(by_k.index):
        # Reported only: the chi-squared reference distribution is not
        # valid when comparing different numbers of hidden states.
        lr_statistic = float(
            2 * (by_k.loc[3, "log_likelihood"] - by_k.loc[2, "log_likelihood"])
        )

    return {
        "selected_number_of_states": bic_choice,
        "selected_states": CANDIDATE_MODELS[bic_choice],
        "rule": "lowest training-period BIC",
        "aic_choice": aic_choice,
        "criteria_agree": bic_choice == aic_choice,
        "likelihood_ratio_statistic": lr_statistic,
        "test_period_used_for_selection": False,
    }


def run_model_selection(
    project_root: str | Path,
    portfolio_path: str | Path | None = None,
    n_restarts: int = N_RANDOM_RESTARTS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Fit and compare the two-state and three-state models.

    Returns
    -------
    dict with
        comparison_table : one row per model (training criteria, stability,
                           test log likelihood for information)
        fit_table        : one row per start (pipeline fit + random restarts)
        selection        : decision and reasons
        best_models      : best-fitting HiddenMarkovModel per state count
        pipeline_runs    : the raw run_pipeline() output per state count
        data             : the shared prepared data
    """
    project_root = Path(project_root)
    if portfolio_path is None:
        portfolio_path = project_root / "data" / "portfolio.csv"
    portfolio_path = Path(portfolio_path)

    runs = [
        fit_candidate(k, project_root, portfolio_path, n_restarts, seed)
        for k in CANDIDATE_MODELS
    ]
    data = check_same_data(runs)
    training_symbols = data["training_symbols"]
    test_symbols = data["test_symbols"]
    n_train = len(training_symbols)

    comparison_rows, fit_rows, best_models = [], [], {}
    for run in runs:
        k = run["number_of_states"]

        if run["pipeline_results"] is not None:
            records = [_fit_record("pipeline", run["pipeline_results"]["model"], training_symbols)]
        else:
            records = [_fit_record("pipeline", None, training_symbols, run["pipeline_error"])]

        records += run_random_restarts(
            training_symbols, run["possible_states"], POSSIBLE_SYMBOLS, n_restarts, seed
        )

        for r in records:
            fit_rows.append(
                {
                    "number_of_states": k,
                    "source": r["source"],
                    "converged": r["converged"],
                    "log_likelihood": r["log_likelihood"],
                    "labels_in_risk_order": r["labels_in_risk_order"],
                    "error": r["error"],
                }
            )

        stability = initialization_stability(records)
        n_params = count_free_parameters(k, len(POSSIBLE_SYMBOLS))
        row = {
            "number_of_states": k,
            "states": ", ".join(run["possible_states"]),
            "n_parameters": n_params,
            "n_training_observations": n_train,
            "best_fit_source": None,
            "log_likelihood": np.nan,
            "log_likelihood_per_obs": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "labels_in_risk_order": np.nan,
            "test_log_likelihood_info_only": np.nan,
            "test_log_likelihood_per_obs_info_only": np.nan,
            "n_test_observations": len(test_symbols),
            **stability,
        }

        converged = [r for r in records if r["converged"]]
        if converged:
            best = max(converged, key=lambda r: r["log_likelihood"])
            best_models[k] = best["model"]
            test_ll = test_log_likelihood(best["model"], training_symbols, test_symbols)
            row.update(
                best_fit_source=best["source"],
                log_likelihood=best["log_likelihood"],
                log_likelihood_per_obs=best["log_likelihood"] / n_train,
                labels_in_risk_order=best["labels_in_risk_order"],
                test_log_likelihood_info_only=test_ll,
                test_log_likelihood_per_obs_info_only=test_ll / len(test_symbols),
                **information_criteria(best["log_likelihood"], n_params, n_train),
            )
        else:
            best_models[k] = None
        comparison_rows.append(row)

    comparison_table = pd.DataFrame(comparison_rows)
    return {
        "comparison_table": comparison_table,
        "fit_table": pd.DataFrame(fit_rows),
        "selection": select_model(comparison_table),
        "best_models": best_models,
        "pipeline_runs": {r["number_of_states"]: r["pipeline_results"] for r in runs},
        "data": data,
        "settings": {
            "market_ticker": MARKET_TICKER,
            "interval": INTERVAL,
            "period": PERIOD,
            "split_date": SPLIT_DATE,
            "training_start": training_symbols.index[0].date().isoformat(),
            "training_end": training_symbols.index[-1].date().isoformat(),
            "test_start": test_symbols.index[0].date().isoformat(),
            "test_end": test_symbols.index[-1].date().isoformat(),
            "possible_symbols": POSSIBLE_SYMBOLS,
            "candidate_models": {str(k): v for k, v in CANDIDATE_MODELS.items()},
            "n_random_restarts": n_restarts,
            "random_seed": seed,
            "stay_probability_range": list(STAY_PROBABILITY_RANGE),
            "emission_concentration": EMISSION_CONCENTRATION,
            "log_likelihood_tolerance": LOG_LIKELIHOOD_TOLERANCE,
            "initial_distribution": "uniform (PyDTMC default, not estimated)",
            "bin_edges": [float(x) for x in data["bin_edges"]],
        },
    }


def save_model_selection_outputs(
    results: dict,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save comparison tables, best-fit matrices and run metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "comparison_table": output_dir / "model_comparison.csv",
        "fit_table": output_dir / "fits_by_initialization.csv",
        "summary": output_dir / "model_selection_summary.json",
    }
    results["comparison_table"].to_csv(paths["comparison_table"], index=False)
    results["fit_table"].to_csv(paths["fit_table"], index=False)

    for k, model in results["best_models"].items():
        if model is None:
            continue
        states, symbols = list(model.states), list(model.symbols)
        p_path = output_dir / f"best_transition_matrix_{k}_states.csv"
        e_path = output_dir / f"best_emission_matrix_{k}_states.csv"
        pd.DataFrame(model.p, index=states, columns=states).to_csv(p_path, index_label="state")
        pd.DataFrame(model.e, index=states, columns=symbols).to_csv(e_path, index_label="state")
        paths[f"best_transition_matrix_{k}_states"] = p_path
        paths[f"best_emission_matrix_{k}_states"] = e_path

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "pydtmc_version": pydtmc.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "settings": results["settings"],
        "selection": results["selection"],
    }
    with open(paths["summary"], "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, default=str)
    return paths


def main() -> None:
    """
    Run model selection with the project's standard settings.

    Saves pipeline outputs to outputs/tables/hmm_2_states/ and
    outputs/tables/hmm_3_states/, and the comparison to
    outputs/tables/model_selection/.
    """
    project_root = Path(__file__).resolve().parents[1]
    results = run_model_selection(project_root)
    paths = save_model_selection_outputs(
        results, project_root / "outputs" / "tables" / "model_selection"
    )

    columns = [
        "number_of_states", "best_fit_source", "n_parameters", "log_likelihood",
        "aic", "bic", "n_converged", "n_starts", "log_likelihood_range",
        "labels_in_risk_order", "test_log_likelihood_per_obs_info_only",
    ]
    print(results["comparison_table"][columns].to_string(index=False))
    selection = results["selection"]
    print(
        f"\nSelected: {selection['selected_number_of_states']} states "
        f"{selection['selected_states']} ({selection['rule']}); "
        f"AIC chooses {selection['aic_choice']} states."
    )
    print("\nSaved:", *paths.values(), sep="\n  ")


if __name__ == "__main__":
    main()

