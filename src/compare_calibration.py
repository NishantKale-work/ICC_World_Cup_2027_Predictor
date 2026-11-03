"""
Compare Elo K=20 calibration on validation and test sets.

Input:
    reports/model_comparison_predictions.csv

Outputs:
    reports/elo_k20_calibration_comparison.csv
    reports/elo_k20_validation_vs_test_calibration.png
    reports/elo_k20_validation_vs_test_calibration_summary.txt
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import brier_score_loss, log_loss


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
SPLITS = ["validation", "test"]

NUMBER_OF_BINS = 10

REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "target",
    "split",
    "model",
    "probability",
]


# ============================================================
# 2. LOAD AND VALIDATE PREDICTIONS
# ============================================================

def load_predictions() -> pd.DataFrame:
    """Load Elo K=20 validation and test predictions."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_PATH}\n\n"
            "Run src/compare_models.py first."
        )

    dataframe = pd.read_csv(PREDICTIONS_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Prediction file is missing columns: {missing_columns}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe = dataframe[
        (dataframe["model"] == MODEL_NAME)
        & (dataframe["split"].isin(SPLITS))
    ].copy()

    if dataframe.empty:
        raise ValueError(
            f"No predictions found for {MODEL_NAME} "
            f"on splits {SPLITS}."
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
        subset=[
            "match_id",
            "date",
            "target",
            "probability",
        ]
    ).copy()

    if not dataframe["target"].isin([0, 1]).all():
        raise ValueError("Target values must be 0 or 1.")

    if (
        dataframe["probability"].lt(0).any()
        or dataframe["probability"].gt(1).any()
    ):
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    dataframe["target"] = dataframe["target"].astype(int)

    for split_name in SPLITS:
        split_dataframe = dataframe[
            dataframe["split"] == split_name
        ]

        if split_dataframe["match_id"].duplicated().any():
            raise ValueError(
                f"Duplicate match IDs found in {split_name} predictions."
            )

    return dataframe.sort_values(
        ["split", "date", "match_id"]
    ).reset_index(drop=True)


# ============================================================
# 3. CALCULATE CALIBRATION TABLE
# ============================================================

def calculate_calibration_table(
    dataframe: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Calculate observed outcomes within fixed probability bins."""

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

    calibration_table["split"] = split_name

    return calibration_table


# ============================================================
# 4. CALCULATE OVERALL METRICS
# ============================================================

def calculate_summary(
    dataframe: pd.DataFrame,
    calibration_table: pd.DataFrame,
    split_name: str,
) -> dict:
    """Calculate overall probability metrics and 10-bin ECE."""

    y_true = dataframe["target"].to_numpy()

    probabilities = np.clip(
        dataframe["probability"].to_numpy(),
        1e-6,
        1 - 1e-6,
    )

    non_empty_bins = calibration_table[
        calibration_table["matches"] > 0
    ]

    ece = non_empty_bins[
        "weighted_calibration_error"
    ].sum()

    return {
        "model": MODEL_NAME,
        "split": split_name,
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
        "expected_calibration_error_10_bins": ece,
    }


# ============================================================
# 5. CREATE SIDE-BY-SIDE RELIABILITY PLOT
# ============================================================

def create_comparison_plot(
    calibration_tables: dict[str, pd.DataFrame],
) -> Path:
    """Plot validation and test calibration on the same axes."""

    figure, axis = plt.subplots(figsize=(8, 8))

    # Perfect calibration reference.
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for split_name in SPLITS:
        calibration_table = calibration_tables[split_name]

        non_empty_bins = calibration_table[
            calibration_table["matches"] > 0
        ]

        axis.plot(
            non_empty_bins["mean_predicted_probability"],
            non_empty_bins["observed_win_rate"],
            marker="o",
            linewidth=2,
            label=split_name.capitalize(),
        )

        # Annotate the number of matches in each bin.
        for _, row in non_empty_bins.iterrows():
            axis.annotate(
                f"{int(row['matches'])}",
                (
                    row["mean_predicted_probability"],
                    row["observed_win_rate"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_title(
        "Elo K=20 — Validation vs Test Calibration"
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
        / "elo_k20_validation_vs_test_calibration.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# ============================================================
# 6. MAIN
# ============================================================

def main() -> None:
    """Run validation-versus-test calibration comparison."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_predictions()

    calibration_tables = {}
    summary_rows = []

    for split_name in SPLITS:
        split_dataframe = dataframe[
            dataframe["split"] == split_name
        ].copy()

        if split_dataframe.empty:
            raise ValueError(
                f"No Elo K=20 predictions found for {split_name}."
            )

        calibration_table = calculate_calibration_table(
            split_dataframe,
            split_name,
        )

        calibration_tables[split_name] = calibration_table

        summary = calculate_summary(
            split_dataframe,
            calibration_table,
            split_name,
        )

        summary_rows.append(summary)

    # --------------------------------------------------------
    # Combine and save calibration tables
    # --------------------------------------------------------

    combined_calibration_table = pd.concat(
        calibration_tables.values(),
        ignore_index=True,
    )

    calibration_table_path = (
        REPORTS_DIR
        / "elo_k20_calibration_comparison.csv"
    )

    combined_calibration_table.to_csv(
        calibration_table_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save summary metrics
    # --------------------------------------------------------

    summary_dataframe = pd.DataFrame(summary_rows)

    summary_path = (
        REPORTS_DIR
        / "elo_k20_validation_vs_test_calibration_summary.txt"
    )

    summary_path.write_text(
        summary_dataframe.to_string(index=False),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Create plot
    # --------------------------------------------------------

    plot_path = create_comparison_plot(
        calibration_tables
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("ELO K=20 — VALIDATION VS TEST CALIBRATION")
    print("=" * 72)

    for split_name in SPLITS:
        print(f"\n{split_name.upper()}")

        print(
            calibration_tables[split_name][
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
    print("OVERALL METRICS")
    print("=" * 72)

    print(
        summary_dataframe.to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("SAVED FILES")
    print("=" * 72)

    print(calibration_table_path)
    print(plot_path)
    print(summary_path)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nERROR")
        print("-" * 72)
        print(error)
        sys.exit(1)