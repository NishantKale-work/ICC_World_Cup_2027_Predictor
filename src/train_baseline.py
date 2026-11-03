
"""
Train and evaluate the ODI Logistic Regression baseline.

Split:
    Chronological 70% / 15% / 15%, based on unique dates.
    Matches from the same date stay in the same split.

Input:
    data/processed/odi_features.csv

Outputs:
    models/logistic_regression_baseline.joblib
    reports/baseline_model_evaluation.txt
    reports/baseline_test_predictions.csv
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "odi_features.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "logistic_regression_baseline.joblib"
REPORT_PATH = REPORTS_DIR / "baseline_model_evaluation.txt"
PREDICTIONS_PATH = REPORTS_DIR / "baseline_test_predictions.csv"


# ---------------------------------------------------------------------
# Configuration — current feature-builder schema
# ---------------------------------------------------------------------

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "team_1_elo",
    "team_2_elo",
    "elo_difference",
    "team_1_recent_form",
    "team_2_recent_form",
    "recent_form_difference",
    "team_1_matches_played",
    "team_2_matches_played",
]

TARGET_COLUMN = "target"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load features and retain decided matches with valid targets."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}\n"
            "Run src/build_features.py first."
        )

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "date",
        "match_id",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Feature dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["date", TARGET_COLUMN]
    ).copy()

    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)

    if not dataframe[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target column must contain only 0 and 1.")

    dataframe = dataframe.sort_values(
        ["date", "match_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("No decided matches with valid targets were found.")

    return dataframe


# ---------------------------------------------------------------------
# Chronological split by unique dates
# ---------------------------------------------------------------------

def chronological_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically by unique dates.

    This prevents matches played on the same date from being divided
    between training, validation, and test sets.
    """

    unique_dates = np.array(
        sorted(dataframe["date"].drop_duplicates())
    )

    number_of_dates = len(unique_dates)

    if number_of_dates < 3:
        raise ValueError(
            "At least 3 unique dates are required for splitting."
        )

    train_end = int(number_of_dates * TRAIN_RATIO)
    validation_end = int(
        number_of_dates * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_end = max(1, train_end)
    validation_end = max(train_end + 1, validation_end)
    validation_end = min(validation_end, number_of_dates - 1)

    train_dates = set(unique_dates[:train_end])
    validation_dates = set(unique_dates[train_end:validation_end])
    test_dates = set(unique_dates[validation_end:])

    train_data = dataframe[
        dataframe["date"].isin(train_dates)
    ].copy()

    validation_data = dataframe[
        dataframe["date"].isin(validation_dates)
    ].copy()

    test_data = dataframe[
        dataframe["date"].isin(test_dates)
    ].copy()

    # Confirm no date overlap between splits.
    train_date_set = set(train_data["date"].unique())
    validation_date_set = set(validation_data["date"].unique())
    test_date_set = set(test_data["date"].unique())

    if train_date_set & validation_date_set:
        raise RuntimeError("Date overlap detected: train vs validation.")

    if train_date_set & test_date_set:
        raise RuntimeError("Date overlap detected: train vs test.")

    if validation_date_set & test_date_set:
        raise RuntimeError("Date overlap detected: validation vs test.")

    if train_data.empty or validation_data.empty or test_data.empty:
        raise ValueError("One of the chronological splits is empty.")

    return train_data, validation_data, test_data


# ---------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------

def create_model_pipeline() -> Pipeline:
    """Create preprocessing and Logistic Regression pipeline."""

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_preprocessor,
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate classification and probability-quality metrics."""

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
    }


def format_metrics(
    title: str,
    metrics: dict[str, float],
) -> list[str]:
    """Format metrics for console and report."""

    lines = [title]

    for metric_name, value in metrics.items():
        lines.append(f"  {metric_name}: {value:.4f}")

    return lines


def describe_split(
    name: str,
    dataframe: pd.DataFrame,
) -> str:
    """Describe split size and date range."""

    return (
        f"{name:<12} {len(dataframe):>6,} matches | "
        f"{dataframe['date'].min().date()} to "
        f"{dataframe['date'].max().date()} | "
        f"{dataframe['date'].nunique():,} unique dates"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("ODI WIN PREDICTION — LOGISTIC REGRESSION BASELINE")
    print("=" * 80)

    dataframe = load_data()

    print(f"\nDecided matches available: {len(dataframe):,}")
    print(f"Unique match dates: {dataframe['date'].nunique():,}")

    train_data, validation_data, test_data = chronological_split(
        dataframe
    )

    print("\nChronological date-based split:")
    print(describe_split("Train:", train_data))
    print(describe_split("Validation:", validation_data))
    print(describe_split("Test:", test_data))

    print("\nSplit validation:")
    print("No match date appears in more than one split.")

    # Prepare datasets.
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data[TARGET_COLUMN]

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    # Train on training data only.
    pipeline = create_model_pipeline()

    print("\nTraining Logistic Regression...")
    pipeline.fit(X_train, y_train)

    # Validation evaluation.
    validation_probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
    )

    # Final test evaluation.
    test_probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
    )

    # Simple 50% probability reference baseline.
    baseline_probabilities = np.full(
        len(y_test),
        0.5,
        dtype=float,
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_probabilities,
    )

    test_confusion_matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    test_classification_report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=[
            "Team 1 did not win",
            "Team 1 won",
        ],
        zero_division=0,
    )

    # Save test predictions.
    prediction_columns = [
        column
        for column in [
            "match_id",
            "date",
            "team_1",
            "team_2",
            "winner",
            TARGET_COLUMN,
        ]
        if column in test_data.columns
    ]

    predictions_output = test_data[prediction_columns].copy()

    predictions_output[
        "predicted_team_1_win_probability"
    ] = test_probabilities

    predictions_output[
        "predicted_team_1_win"
    ] = test_predictions

    predictions_output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # Save model artifact.
    model_artifact = {
        "model": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "model_name": "Logistic Regression",
        "split_method": "chronological unique-date split",
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(model_artifact, MODEL_PATH)

    # Build report.
    report_lines = [
        "ODI WIN PREDICTION — BASELINE MODEL EVALUATION",
        "=" * 80,
        "",
        "DATA SPLIT",
        "-" * 80,
        f"Total decided matches: {len(dataframe):,}",
        f"Total unique dates: {dataframe['date'].nunique():,}",
        describe_split("Train:", train_data),
        describe_split("Validation:", validation_data),
        describe_split("Test:", test_data),
        "",
        "Split validation:",
        "Chronological split based on unique dates.",
        "No date appears in more than one split.",
        "",
    ]

    report_lines.extend(
        format_metrics("VALIDATION METRICS", validation_metrics)
    )
    report_lines.append("")

    report_lines.extend(
        format_metrics(
            "TEST METRICS — LOGISTIC REGRESSION",
            test_metrics,
        )
    )
    report_lines.append("")

    report_lines.extend(
        format_metrics(
            "TEST METRICS — 50% PROBABILITY BASELINE",
            baseline_metrics,
        )
    )

    report_lines.extend(
        [
            "",
            "TEST CONFUSION MATRIX",
            "-" * 80,
            "Rows = actual class [0, 1]; columns = predicted class [0, 1]",
            str(test_confusion_matrix),
            "",
            "CLASSIFICATION REPORT",
            "-" * 80,
            test_classification_report,
            "",
            "FEATURES",
            "-" * 80,
        ]
    )

    report_lines.extend(FEATURE_COLUMNS)

    report_lines.extend(
        [
            "",
            f"Saved model: {MODEL_PATH}",
            f"Saved test predictions: {PREDICTIONS_PATH}",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # Console output.
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)

    for line in format_metrics(
        "Logistic Regression",
        validation_metrics,
    ):
        print(line)

    print("\n" + "=" * 80)
    print("TEST METRICS — LOGISTIC REGRESSION")
    print("=" * 80)

    for line in format_metrics(
        "Logistic Regression",
        test_metrics,
    ):
        print(line)

    print("\n" + "=" * 80)
    print("TEST METRICS — 50% PROBABILITY BASELINE")
    print("=" * 80)

    for line in format_metrics(
        "Baseline",
        baseline_metrics,
    ):
        print(line)

    print("\nTest confusion matrix:")
    print(test_confusion_matrix)

    print("\nClassification report:")
    print(test_classification_report)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved test predictions: {PREDICTIONS_PATH}")
    print(f"Evaluation report saved: {REPORT_PATH}")

    print("\nBaseline training and evaluation complete.")


if __name__ == "__main__":
    main()
