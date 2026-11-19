"""Shared project configuration.

Planned contents:
- S&P/TSX Composite ticker and sample dates
- train/test split dates
- random seeds and HMM initialization counts
- two-state and three-state model settings
- trading-rule thresholds and transaction costs
- canonical data and output paths
"""

# Settings for model selection and evaluation

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

N_RANDOM_RESTARTS = 0          # random starts per model, on top of the pipeline fit
RANDOM_SEED = 348
STAY_PROBABILITY_RANGE = (0.80, 0.98)
EMISSION_CONCENTRATION = 2.0
LOG_LIKELIHOOD_TOLERANCE = 1.0  # fits within this of the best count as "near-best"
