"""Tests for HMM fitting, probabilities, and saved outputs."""

import json

import numpy as np
import pandas as pd
from pydtmc import HiddenMarkovModel

from src.hmm import save_hmm_outputs


def test_save_hmm_outputs_writes_parameters_and_decoding_results(tmp_path):
    states = ["low_risk", "high_risk"]
    symbols = ["negative", "neutral", "positive"]
    transition_matrix = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
        ]
    )
    emission_matrix = np.array(
        [
            [0.2, 0.6, 0.2],
            [0.4, 0.2, 0.4],
        ]
    )
    model = HiddenMarkovModel(
        transition_matrix,
        emission_matrix,
        states,
        symbols,
    )
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    filtered_probabilities = pd.DataFrame(
        [[0.7, 0.3], [0.4, 0.6]],
        index=dates,
        columns=states,
    )
    posterior_probabilities = pd.DataFrame(
        [[0.6, 0.4], [0.4, 0.6]],
        index=dates,
        columns=states,
    )
    decoded_results = {
        "log_probability": -4.5,
        "filtered_probabilities": filtered_probabilities,
        "posterior_probabilities": posterior_probabilities,
    }

    output_paths = save_hmm_outputs(
        output_dir=tmp_path,
        model=model,
        decoded_results=decoded_results,
    )

    assert all(path.is_file() for path in output_paths.values())

    saved_transition = pd.read_csv(
        output_paths["transition_matrix"],
        index_col="state",
    )
    saved_emission = pd.read_csv(
        output_paths["emission_matrix"],
        index_col="state",
    )

    np.testing.assert_allclose(saved_transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(saved_emission.sum(axis=1), 1.0)

    with open(
        output_paths["model_summary"],
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    assert summary["states"] == states
    assert summary["symbols"] == symbols
    assert summary["log_probability"] == -4.5
