from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# This file contains predictions for ALL tested K values.
INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "elo_k_experiment_predictions.csv"
)

# Authoritative feature dataset used to define the chronological split.
FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports"

BOOTSTRAP_REPLICATES = 2_000
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95

SELECTED_K = 20
EPSILON = 1e-6

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


# ============================================================
# Create authoritative test match IDs
# ============================================================

def load_authoritative_test_ids() -> set:
    """
    Recreate the chronological split from the feature dataset.

    This is used to verify whether the Elo prediction file
    contains every expected test match.
    """

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURES_PATH}\n"
            "Run src/build_features.py first."
        )

    features = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["date"],
    )

    required_columns = {
        "match_id",
        "date",
        "target",
    }

    missing_columns = required_columns - set(features.columns)

    if missing_columns:
        raise ValueError(
            "Feature dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if features["match_id"].duplicated().any():
        raise ValueError(
            "Duplicate match_id values found in feature dataset."
        )

    features = features.loc[
        features["target"].isin([0, 1])
    ].copy()

    features["date"] = pd.to_datetime(
        features["date"],
        errors="coerce",
    )

    if features["date"].isna().any():
        raise ValueError(
            "Feature dataset contains invalid dates."
        )

    unique_dates = sorted(
        features["date"].drop_duplicates().tolist()
    )

    train_date_count = int(
        len(unique_dates) * TRAIN_FRACTION
    )

    validation_date_count = int(
        len(unique_dates) * VALIDATION_FRACTION
    )

    test_start = (
        train_date_count + validation_date_count
    )

    test_dates = set(
        unique_dates[test_start:]
    )

    test_rows = features.loc[
        features["date"].isin(test_dates)
    ]

    return set(test_rows["match_id"])


# ============================================================
# Load and validate test predictions
# ============================================================

def load_test_predictions() -> pd.DataFrame:
    """Load held-out test predictions for Elo K=20."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{INPUT_PATH}\n\n"
            "Run src/experiment_elo.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "match_id",
        "split",
        "target",
        "k_value",
        "team_1_win_probability",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Prediction file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print("\nPrediction file")
    print("-" * 70)
    print(f"File: {INPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"K values: {sorted(df['k_value'].dropna().unique())}")

    print("\nRows by K and split:")
    print(
        df.groupby(["k_value", "split"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    # --------------------------------------------------------
    # Filter selected K
    # --------------------------------------------------------

    selected_df = df.loc[
        df["k_value"] == SELECTED_K
    ].copy()

    if selected_df.empty:
        raise ValueError(
            f"No predictions found for Elo K={SELECTED_K}."
        )

    test_df = selected_df.loc[
        selected_df["split"] == "test"
    ].copy()

    if test_df.empty:
        raise ValueError(
            f"No test predictions found for Elo K={SELECTED_K}."
        )

    # --------------------------------------------------------
    # Validate IDs for uniqueness
    # --------------------------------------------------------

    if test_df["match_id"].duplicated().any():
        duplicate_ids = (
            test_df.loc[
                test_df["match_id"].duplicated(keep=False),
                "match_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate test match IDs found for "
            f"K={SELECTED_K}. Examples: {duplicate_ids[:10]}"
        )

    # --------------------------------------------------------
    # Validate all K values use the same test matches
    # --------------------------------------------------------

    all_test_rows = df.loc[
        df["split"] == "test"
    ].copy()

    k_values = sorted(
        all_test_rows["k_value"].dropna().unique()
    )

    expected_ids = set(
        test_df["match_id"]
    )

    for k_value in k_values:

        k_test_df = all_test_rows.loc[
            all_test_rows["k_value"] == k_value
        ]

        current_ids = set(
            k_test_df["match_id"]
        )

        if k_test_df["match_id"].duplicated().any():
            raise ValueError(
                f"Duplicate test match IDs found for K={k_value}."
            )

        if current_ids != expected_ids:
            missing_ids = expected_ids - current_ids
            extra_ids = current_ids - expected_ids

            raise ValueError(
                "Test match IDs differ across K values. "
                "Paired bootstrap would be invalid.\n"
                f"K={k_value}\n"
                f"Missing IDs: {len(missing_ids)}\n"
                f"Extra IDs: {len(extra_ids)}"
            )

    # --------------------------------------------------------
    # Compare against authoritative feature test IDs
    # --------------------------------------------------------

    authoritative_test_ids = load_authoritative_test_ids()
    prediction_test_ids = set(test_df["match_id"])

    missing_from_predictions = (
        authoritative_test_ids - prediction_test_ids
    )

    unexpected_in_predictions = (
        prediction_test_ids - authoritative_test_ids
    )

    print("\nAuthoritative test-set consistency check")
    print("-" * 70)
    print(
        "Expected test matches from feature dataset: "
        f"{len(authoritative_test_ids):,}"
    )
    print(
        "K=20 test predictions: "
        f"{len(prediction_test_ids):,}"
    )
    print(
        "Expected matches missing from predictions: "
        f"{len(missing_from_predictions):,}"
    )
    print(
        "Unexpected prediction matches: "
        f"{len(unexpected_in_predictions):,}"
    )

    if missing_from_predictions:
        print("\nMissing match IDs:")
        print(
            sorted(missing_from_predictions)[:20]
        )

    if unexpected_in_predictions:
        print("\nUnexpected match IDs:")
        print(
            sorted(unexpected_in_predictions)[:20]
        )

    if (
        missing_from_predictions
        or unexpected_in_predictions
    ):
        raise ValueError(
            "Prediction test matches do not exactly match "
            "the authoritative feature test set. "
            "Stopping before bootstrap."
        )

    # --------------------------------------------------------
    # Validate targets and probabilities
    # --------------------------------------------------------

    test_df = test_df.dropna(
        subset=[
            "target",
            "team_1_win_probability",
        ]
    ).copy()

    test_df["target"] = test_df["target"].astype(int)

    if not test_df["target"].isin([0, 1]).all():
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    if test_df["target"].nunique() != 2:
        raise ValueError(
            "The test set must contain both outcome classes."
        )

    test_df["team_1_win_probability"] = np.clip(
        test_df["team_1_win_probability"].astype(float),
        EPSILON,
        1 - EPSILON,
    )

    test_df = test_df.sort_values(
        "match_id"
    ).reset_index(drop=True)

    return test_df


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate probability-quality metrics."""

    probabilities = np.clip(
        probabilities.astype(float),
        EPSILON,
        1 - EPSILON,
    )

    return {
        "auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


# ============================================================
# Bootstrap confidence intervals
# ============================================================

def run_bootstrap(
    y_true: np.ndarray,
    elo_probabilities: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    """
    Bootstrap test matches with replacement.

    The same sampled match indices are used for Elo and the
    50% probability baseline, enabling paired comparisons.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_size = len(y_true)

    baseline_probabilities = np.full(
        sample_size,
        0.5,
        dtype=float,
    )

    bootstrap_rows = []
    skipped_auc_replicates = 0

    for replicate in range(
        BOOTSTRAP_REPLICATES
    ):

        sampled_indices = rng.integers(
            low=0,
            high=sample_size,
            size=sample_size,
        )

        sampled_y = y_true[
            sampled_indices
        ]

        sampled_elo = elo_probabilities[
            sampled_indices
        ]

        sampled_baseline = baseline_probabilities[
            sampled_indices
        ]

        # AUC is undefined if a sample contains one class only.
        if np.unique(sampled_y).size < 2:
            skipped_auc_replicates += 1
            continue

        elo_metrics = calculate_metrics(
            sampled_y,
            sampled_elo,
        )

        baseline_metrics = calculate_metrics(
            sampled_y,
            sampled_baseline,
        )

        bootstrap_rows.append(
            {
                "replicate": replicate,

                "elo_auc": elo_metrics["auc"],
                "elo_log_loss": elo_metrics["log_loss"],
                "elo_brier_score": elo_metrics["brier_score"],

                "baseline_auc": baseline_metrics["auc"],
                "baseline_log_loss": baseline_metrics["log_loss"],
                "baseline_brier_score": baseline_metrics[
                    "brier_score"
                ],

                # Positive means Elo has higher/worse loss.
                "auc_difference_vs_0_5": (
                    elo_metrics["auc"] - 0.5
                ),

                "log_loss_difference_vs_baseline": (
                    elo_metrics["log_loss"]
                    - baseline_metrics["log_loss"]
                ),

                "brier_difference_vs_baseline": (
                    elo_metrics["brier_score"]
                    - baseline_metrics["brier_score"]
                ),
            }
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    if bootstrap_df.empty:
        raise RuntimeError(
            "No valid bootstrap replicates were produced."
        )

    return (
        bootstrap_df,
        skipped_auc_replicates,
    )


# ============================================================
# Confidence interval summary
# ============================================================

def percentile_interval(
    values: pd.Series,
) -> tuple[float, float]:
    """Return a two-sided percentile confidence interval."""

    alpha = 1 - CONFIDENCE_LEVEL

    lower = values.quantile(
        alpha / 2
    )

    upper = values.quantile(
        1 - alpha / 2
    )

    return (
        float(lower),
        float(upper),
    )


def build_summary(
    bootstrap_df: pd.DataFrame,
    point_estimates: dict[str, float],
) -> pd.DataFrame:

    metric_definitions = [
        {
            "metric": "ROC AUC",
            "point_estimate": point_estimates["auc"],
            "bootstrap_column": "elo_auc",
            "comparison": "AUC - 0.5",
            "difference_column": "auc_difference_vs_0_5",
            "direction": "Higher is better",
        },
        {
            "metric": "Log loss",
            "point_estimate": point_estimates["log_loss"],
            "bootstrap_column": "elo_log_loss",
            "comparison": "Elo log loss - baseline log loss",
            "difference_column": (
                "log_loss_difference_vs_baseline"
            ),
            "direction": "Lower is better",
        },
        {
            "metric": "Brier score",
            "point_estimate": point_estimates["brier_score"],
            "bootstrap_column": "elo_brier_score",
            "comparison": "Elo Brier - baseline Brier",
            "difference_column": (
                "brier_difference_vs_baseline"
            ),
            "direction": "Lower is better",
        },
    ]

    summary_rows = []

    for definition in metric_definitions:

        metric_ci = percentile_interval(
            bootstrap_df[
                definition["bootstrap_column"]
            ]
        )

        difference_ci = percentile_interval(
            bootstrap_df[
                definition["difference_column"]
            ]
        )

        summary_rows.append(
            {
                "metric": definition["metric"],
                "point_estimate": definition["point_estimate"],
                "ci_95_lower": metric_ci[0],
                "ci_95_upper": metric_ci[1],
                "comparison": definition["comparison"],
                "difference_95_lower": difference_ci[0],
                "difference_95_upper": difference_ci[1],
                "direction": definition["direction"],
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        f"BOOTSTRAP EVALUATION — ODI ELO K={SELECTED_K}"
    )
    print("=" * 70)

    test_df = load_test_predictions()

    y_true = test_df[
        "target"
    ].to_numpy(dtype=int)

    elo_probabilities = test_df[
        "team_1_win_probability"
    ].to_numpy(dtype=float)

    baseline_probabilities = np.full(
        len(y_true),
        0.5,
        dtype=float,
    )

    point_estimates = calculate_metrics(
        y_true,
        elo_probabilities,
    )

    baseline_metrics = calculate_metrics(
        y_true,
        baseline_probabilities,
    )

    print("\nFinal test-set details")
    print("-" * 70)
    print(f"Test matches: {len(test_df):,}")
    print(
        "Team 1 actual win rate: "
        f"{y_true.mean():.4f}"
    )

    print("\nPoint estimates")
    print("-" * 70)

    print(
        f"Elo K={SELECTED_K} AUC: "
        f"{point_estimates['auc']:.6f}"
    )

    print(
        f"Elo K={SELECTED_K} log loss: "
        f"{point_estimates['log_loss']:.6f}"
    )

    print(
        f"Elo K={SELECTED_K} Brier score: "
        f"{point_estimates['brier_score']:.6f}"
    )

    print("\n50% probability baseline")
    print("-" * 70)

    print(
        f"Baseline AUC: "
        f"{baseline_metrics['auc']:.6f}"
    )

    print(
        f"Baseline log loss: "
        f"{baseline_metrics['log_loss']:.6f}"
    )

    print(
        f"Baseline Brier score: "
        f"{baseline_metrics['brier_score']:.6f}"
    )

    print(
        f"\nRunning {BOOTSTRAP_REPLICATES:,} "
        "bootstrap replicates..."
    )

    bootstrap_df, skipped_replicates = run_bootstrap(
        y_true=y_true,
        elo_probabilities=elo_probabilities,
    )

    summary_df = build_summary(
        bootstrap_df=bootstrap_df,
        point_estimates=point_estimates,
    )

    summary_path = (
        OUTPUT_DIR
        / "bootstrap_confidence_intervals.csv"
    )

    bootstrap_path = (
        OUTPUT_DIR
        / "bootstrap_replicate_results.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    print("\n95% Bootstrap Confidence Intervals")
    print("=" * 70)

    display_columns = [
        "metric",
        "point_estimate",
        "ci_95_lower",
        "ci_95_upper",
        "comparison",
        "difference_95_lower",
        "difference_95_upper",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nBootstrap details")
    print("-" * 70)
    print(
        f"Requested replicates: "
        f"{BOOTSTRAP_REPLICATES:,}"
    )
    print(
        f"Valid replicates: "
        f"{len(bootstrap_df):,}"
    )
    print(
        f"Skipped single-class replicates: "
        f"{skipped_replicates:,}"
    )
    print(
        f"Random seed: {RANDOM_STATE}"
    )
    print(
        f"Confidence level: "
        f"{CONFIDENCE_LEVEL * 100:.0f}%"
    )

    print("\nSaved files")
    print("-" * 70)
    print(summary_path)
    print(bootstrap_path)

    print("\nInterpretation reminder:")
    print(
        "These intervals quantify sampling uncertainty within "
        "this historical test period. They do not account for "
        "model-selection, data-source, or future-tournament "
        "uncertainty."
    )


if __name__ == "__main__":
    main()