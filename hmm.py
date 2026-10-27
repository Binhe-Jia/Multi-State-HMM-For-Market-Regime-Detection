from pydtmc import MarkovChain, HiddenMarkovModel
from pathlib import Path
import json
import networkx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf
import scipy
import sklearn as skl

def validate_symbol_sequence(
    observations: pd.Series,
    possible_symbols: list[str],
) -> None:
    """
    Check that the observation sequence is nonempty and contains
    only recognized symbols.
    """
    if observations.empty:
        raise ValueError("Empty observations input.")

    if not possible_symbols:
        raise ValueError("Possible-symbol list is empty.")

    if len(possible_symbols) != len(set(possible_symbols)):
        raise ValueError("Possible-symbol list contains duplicates.")

    if observations.isna().any():
        raise ValueError("Observation sequence contains missing values.")

    observed_symbols = set(observations)
    allowed_symbols = set(possible_symbols)

    unknown_symbols = observed_symbols - allowed_symbols

    if unknown_symbols:
        raise ValueError(
            f"Observation sequence contains unknown symbols: "
            f"{unknown_symbols}"
        )

def validate_probability_matrix(
    matrix: np.ndarray,
    expected_shape: tuple[int, int],
    matrix_name: str,
) -> None:
    """
    Validate an initial transition or emission matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("The matrix is not an NumPy array")
    if matrix.shape != expected_shape:
        raise ValueError(f"The matrix dimension does not match the expected shape.\nActual:{matrix.shape}\nExpected:{expected_shape}" )
    if (matrix < 0).any():
        raise ValueError(
            f"{matrix_name} contains negative probabilities."
        )
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(
            f"Rows of {matrix_name} must sum to one. "
            f"Current row sums: {row_sums}"
        )

def create_initial_transition_guess(
    number_of_states: int,
    stay_probability: float = 0.90,
) -> np.ndarray:
    """
    Create an initial persistent hidden-state transition matrix.
    """
    if number_of_states < 2:
        raise ValueError("At least two hidden states are required.")
    if not 0.0 < stay_probability < 1.0:
        raise ValueError("Stay probability must be between zero and one.")
    switch_probability = (1.0 - stay_probability) / (number_of_states - 1)
    transition_guess = np.full(
        (number_of_states, number_of_states),
        switch_probability,
        dtype=float,
    )
    np.fill_diagonal(transition_guess,stay_probability)
    validate_probability_matrix(transition_guess,expected_shape=(number_of_states, number_of_states),matrix_name="Transition matrix")

    return transition_guess


def create_initial_emission_guess(
    possible_symbols: list[str],
) -> np.ndarray:
    """
    Create initial emission probabilities for a two-state model.
    """
    expected_symbols = [
        "very_negative",
        "negative",
        "neutral",
        "positive",
        "very_positive",
    ]
    if possible_symbols != expected_symbols:
        raise ValueError("Possible symbols must use the expected order.")
    raw_weights = np.array(
        [
            [1, 2, 4, 2, 1],
            [4, 2, 1, 2, 4],
        ],
        dtype=float,
    )
    emission_guess = (raw_weights/ raw_weights.sum(axis=1, keepdims=True))
    validate_probability_matrix(emission_guess,expected_shape=(2, len(possible_symbols)),matrix_name="Emission matrix")
    return emission_guess


def fit_discrete_hmm(
    training_symbols: pd.Series,
    possible_states: list[str],
    possible_symbols: list[str],
    transition_guess: np.ndarray,
    emission_guess: np.ndarray,
) -> HiddenMarkovModel:
    """
    Fit a discrete HMM using the Baum-Welch algorithm.
    """
    validate_symbol_sequence(training_symbols, possible_symbols)
    if len(possible_states) < 2:
        raise ValueError("At least two hidden states are required.")
    if len(possible_states) != len(set(possible_states)):
        raise ValueError("Possible-state list contains duplicates.")
    number_of_states = len(possible_states)
    number_of_symbols = len(possible_symbols)
    validate_probability_matrix(
        transition_guess,
        expected_shape=(number_of_states,number_of_states,),
        matrix_name="Transition matrix",
    )

    validate_probability_matrix(
        emission_guess,
        expected_shape=(
            number_of_states,
            number_of_symbols,
        ),
        matrix_name="Emission matrix",
    )
    observation_sequence = (training_symbols.astype(str).tolist())
    model = HiddenMarkovModel.fit(
        fitting_type="baum-welch",
        possible_states=possible_states,
        possible_symbols=possible_symbols,
        p_guess=transition_guess,
        e_guess=emission_guess,
        symbols=observation_sequence,
    )
    return model
    


def decode_observations(
    model: HiddenMarkovModel,
    observations: pd.Series,
) -> dict:
    """
    Decode an observation sequence with a fitted HMM.
    """
    validate_symbol_sequence(observations,list(model.symbols))

    symbol_sequence = observations.astype(str).tolist()
    result = model.decode(symbol_sequence, use_scaling=True)

    if result is None:
        raise ValueError("The HMM could not decode the observations.")
    (log_probability,
        posterior,
        backward,
        forward,
        scaling_factors,
    ) = result

    states = list(model.states)
    posterior_probabilities = pd.DataFrame(
        posterior.T,
        index=observations.index,
        columns=states,
    )
    filtered_probabilities = pd.DataFrame(
        forward[:, 1:].T,
        index=observations.index,
        columns=states,
    )
    backward_probabilities = pd.DataFrame(
        backward[:, 1:].T,
        index=observations.index,
        columns=states,
    )
    scaling_factors = pd.Series(
        scaling_factors[1:],
        index=observations.index,
        name="scaling_factor",
    )
    return {
        "log_probability": log_probability,
        "posterior_probabilities": posterior_probabilities,
        "filtered_probabilities": filtered_probabilities,
        "backward_probabilities": backward_probabilities,
        "scaling_factors": scaling_factors,
    }



def get_filtered_probabilities(
    model: HiddenMarkovModel,
    observations: pd.Series,
) -> pd.DataFrame:
    """
    Return state probabilities based only on observations available
    through each date.
    """
    decoded_results = decode_observations(model, observations)
    filtered_probabilities = decoded_results["filtered_probabilities"].copy()
    row_sums = filtered_probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("Filtered state probabilities do not sum to one.")
    return filtered_probabilities


def get_model_parameters(
    model: HiddenMarkovModel,
) -> dict:
    """
    Return the fitted states, symbols, transition matrix,
    and emission matrix in a convenient structure.
    """
    states = list(model.states)
    symbols = list(model.symbols)

    transition_matrix = pd.DataFrame(
        model.p.copy(),
        index=states,
        columns=states,
    )

    emission_matrix = pd.DataFrame(
        model.e.copy(),
        index=states,
        columns=symbols,
    )

    return {
        "states": states,
        "symbols": symbols,
        "transition_matrix": transition_matrix,
        "emission_matrix": emission_matrix,
    }


def save_hmm_outputs(
    output_dir: str | Path,
    model: HiddenMarkovModel,
    decoded_results: dict | None = None,
) -> dict[str, Path]:
    """
    Save fitted HMM parameters and optional decoding results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "transition_matrix": output_dir / "transition_matrix.csv",
        "emission_matrix": output_dir / "emission_matrix.csv",
        "model_summary": output_dir / "model_summary.json",
    }

    parameters = get_model_parameters(model)

    parameters["transition_matrix"].to_csv(
        output_paths["transition_matrix"],
        index_label="state",
    )
    parameters["emission_matrix"].to_csv(
        output_paths["emission_matrix"],
        index_label="state",
    )

    summary = {
        "states": parameters["states"],
        "symbols": parameters["symbols"],
    }

    if decoded_results is not None:
        output_paths["filtered_probabilities"] = (
            output_dir / "filtered_probabilities.csv"
        )
        output_paths["posterior_probabilities"] = (
            output_dir / "posterior_probabilities.csv"
        )

        decoded_results["filtered_probabilities"].to_csv(
            output_paths["filtered_probabilities"],
            index_label="date",
        )
        decoded_results["posterior_probabilities"].to_csv(
            output_paths["posterior_probabilities"],
            index_label="date",
        )

        summary["log_probability"] = float(
            decoded_results["log_probability"]
        )

    with open(
        output_paths["model_summary"],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    return output_paths
