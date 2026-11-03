"""
Diagnose probability calibration for the current ODI predictor.

Uses:
    reports/model_comparison_predictions.csv

Filters:
    split == "test"
    model == "Elo K=20"

Outputs:
    reports/elo_k20_test_calibration.csv
    reports/elo_k20_test_calibration.png
    reports/elo_k20_test_probability_distribution.png
    reports/elo_k20_calibration_summary.txt
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "Elo K=20"
SPLIT_NAME = "test"

TARGET_COLUMN = "target"
PROBABILITY_COLUMN = "probability"

# Ten fixed-width probability bins:
# 0.0–0.1, 0.1–0.2, ..., 0.9–1.0
NUMBER_OF_BINS = 10


# ============================================================
# 2. LOAD AND VALIDATE PREDICTIONS
# ============================================================

def load_test_predictions() -> pd.DataFrame:
    """Load saved predictions and select Elo K=20 test matches."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_PATH}\n\n"
            "Run src/compare_models.py first."
        )

    dataframe = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "match_id",
        "date",
        "target",
        "split",
        "model",
        "probability",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Prediction file is missing columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = dataframe[
        (dataframe["split"] == SPLIT_NAME)
        & (dataframe["model"] == MODEL_NAME)
    ].copy()

    if dataframe.empty:
        raise ValueError(
            f"No predictions found for model='{MODEL_NAME}' "
            f"and split='{SPLIT_NAME}'."
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    dataframe["target"] = pd.to_numeric(
        dataframe["target"],
        errors="coerce",
    )

    dataframe["probability"] = pd.to_numeric(
        dataframe["probability"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["date", "target", "probability"]
    ).copy()

    if not dataframe["target"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if dataframe["probability"].lt(0).any() or dataframe[
        "probability"
    ].gt(1).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if dataframe["match_id"].duplicated().any():
        raise ValueError(
            "Duplicate match IDs found for Elo K=20 test predictions."
        )

    dataframe["target"] = dataframe["target"].astype(int)

    dataframe = dataframe.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    return dataframe


# ============================================================
# 3. CALCULATE CALIBRATION TABLE
# ============================================================

def calculate_calibration_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group predictions into fixed-width probability bins.

    Mean predicted probability:
        Average probability assigned by the model in that bin.

    Observed win rate:
        Actual proportion of Team 1 wins in that bin.
    """

    bin_edges = np.linspace(
        0,
        1,
        NUMBER_OF_BINS + 1,
    )

    bin_labels = [
        f"{bin_edges[index]:.1f}–{bin_edges[index + 1]:.1f}"
        for index in range(NUMBER_OF_BINS)
    ]

    dataframe = dataframe.copy()

    dataframe["probability_bin"] = pd.cut(
        dataframe["probability"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    )

    calibration_table = (
        dataframe.groupby(
            "probability_bin",
            observed=False,
        )
        .agg(
            matches=("target", "count"),
            mean_predicted_probability=("probability", "mean"),
            observed_win_rate=("target", "mean"),
        )
        .reset_index()
    )

    calibration_table["absolute_calibration_error"] = (
        calibration_table["mean_predicted_probability"]
        - calibration_table["observed_win_rate"]
    ).abs()

    calibration_table["weighted_calibration_error"] = (
        calibration_table["matches"]
        / len(dataframe)
        * calibration_table["absolute_calibration_error"]
    )

    return calibration_table


# ============================================================
# 4. CALCULATE SUMMARY METRICS
# ============================================================

def calculate_summary(
    dataframe: pd.DataFrame,
    calibration_table: pd.DataFrame,
) -> dict:
    """Calculate overall probability metrics and approximate ECE."""

    y_true = dataframe["target"].to_numpy()

    probabilities = np.clip(
        dataframe["probability"].to_numpy(),
        1e-6,
        1 - 1e-6,
    )

    non_empty_bins = calibration_table[
        calibration_table["matches"] > 0
    ]

    expected_calibration_error = (
        non_empty_bins["weighted_calibration_error"].sum()
    )

    return {
        "model": MODEL_NAME,
        "split": SPLIT_NAME,
        "matches": len(dataframe),
        "mean_predicted_probability": probabilities.mean(),
        "actual_team_1_win_rate": y_true.mean(),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "expected_calibration_error_10_bins": (
            expected_calibration_error
        ),
    }


# ============================================================
# 5. CREATE RELIABILITY DIAGRAM
# ============================================================

def create_reliability_plot(
    calibration_table: pd.DataFrame,
    summary: dict,
) -> Path:
    """Plot predicted probability against observed win rate."""

    non_empty_bins = calibration_table[
        calibration_table["matches"] > 0
    ]

    figure, axis = plt.subplots(figsize=(8, 8))

    # Perfect calibration reference.
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    # Model calibration.
    axis.plot(
        non_empty_bins["mean_predicted_probability"],
        non_empty_bins["observed_win_rate"],
        marker="o",
        linewidth=2,
        label="Elo K=20",
    )

    # Show bin counts beside points.
    for _, row in non_empty_bins.iterrows():
        axis.annotate(
            f"n={int(row['matches'])}",
            (
                row["mean_predicted_probability"],
                row["observed_win_rate"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    axis.set_title(
        "Elo K=20 — Test Set Calibration"
    )

    axis.set_xlabel(
        "Mean predicted probability of Team 1 winning"
    )

    axis.set_ylabel(
        "Observed Team 1 win rate"
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()

    output_path = (
        REPORTS_DIR
        / "elo_k20_test_calibration.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# ============================================================
# 6. CREATE PROBABILITY DISTRIBUTION PLOT
# ============================================================

def create_probability_distribution_plot(
    dataframe: pd.DataFrame,
) -> Path:
    """Show how often the model produces different probabilities."""

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.hist(
        dataframe["probability"],
        bins=np.linspace(0, 1, 21),
        edgecolor="black",
    )

    axis.set_title(
        "Elo K=20 — Test Prediction Distribution"
    )

    axis.set_xlabel(
        "Predicted probability of Team 1 winning"
    )

    axis.set_ylabel(
        "Number of matches"
    )

    axis.set_xlim(0, 1)
    axis.grid(True, axis="y", alpha=0.3)

    figure.tight_layout()

    output_path = (
        REPORTS_DIR
        / "elo_k20_test_probability_distribution.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    """Run calibration diagnostics and save results."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_test_predictions()

    calibration_table = calculate_calibration_table(
        dataframe
    )

    summary = calculate_summary(
        dataframe,
        calibration_table,
    )

    # --------------------------------------------------------
    # Save calibration table
    # --------------------------------------------------------

    calibration_table_path = (
        REPORTS_DIR
        / "elo_k20_test_calibration.csv"
    )

    calibration_table.to_csv(
        calibration_table_path,
        index=False,
    )

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    reliability_plot_path = create_reliability_plot(
        calibration_table,
        summary,
    )

    probability_distribution_path = (
        create_probability_distribution_plot(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        REPORTS_DIR
        / "elo_k20_calibration_summary.txt"
    )

    summary_lines = [
        "ELO K=20 CALIBRATION DIAGNOSTICS",
        "=" * 50,
        "",
    ]

    for metric, value in summary.items():
        if isinstance(value, float):
            summary_lines.append(
                f"{metric}: {value:.6f}"
            )
        else:
            summary_lines.append(
                f"{metric}: {value}"
            )

    summary_lines.extend(
        [
            "",
            "Interpretation:",
            "- A perfectly calibrated model follows the diagonal",
            "  in the reliability diagram.",
            "- If observed win rate is below predicted probability,",
            "  the model is overconfident in that bin.",
            "- If observed win rate is above predicted probability,",
            "  the model is underconfident in that bin.",
            "- ECE is a bin-based summary and depends on the chosen",
            "  number of bins; interpret it alongside the plot.",
        ]
    )

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("ELO K=20 TEST CALIBRATION")
    print("=" * 72)

    print(
        calibration_table[
            [
                "probability_bin",
                "matches",
                "mean_predicted_probability",
                "observed_win_rate",
                "absolute_calibration_error",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("OVERALL SUMMARY")
    print("=" * 72)

    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")

    print("\n" + "=" * 72)
    print("SAVED FILES")
    print("=" * 72)

    print(calibration_table_path)
    print(reliability_plot_path)
    print(probability_distribution_path)
    print(summary_path)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nERROR")
        print("-" * 72)
        print(error)
        sys.exit(1)