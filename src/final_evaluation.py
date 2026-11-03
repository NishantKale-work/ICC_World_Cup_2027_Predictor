from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT_DIR / "data" / "processed" / "odi_features.csv"
OUTPUT_DIR = ROOT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_RESULTS_PATH = OUTPUT_DIR / "final_validation_results.csv"
TEST_RESULTS_PATH = OUTPUT_DIR / "final_test_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "final_evaluation_summary.txt"

TARGET_COLUMN = "target"
ELO_DIFFERENCE_COLUMN = "elo_difference"

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

# Main chronological split
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# Split the training period chronologically:
# earlier 80% trains the base model,
# later 20% fits the calibration model.
BASE_TRAIN_FRACTION = 0.80

RANDOM_STATE = 42


# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------

def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}\n"
            "Run src/build_features.py first."
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    required_columns = [
        "date",
        "match_id",
        TARGET_COLUMN,
        ELO_DIFFERENCE_COLUMN,
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    df = df[df[TARGET_COLUMN].isin([0, 1])].copy()
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No decided matches with valid targets found.")

    if df["date"].isna().any():
        raise ValueError("Some matches have missing dates.")

    if df["match_id"].duplicated().any():
        raise ValueError("Duplicate match_id values found.")

    return df


# ---------------------------------------------------------
# Chronological split by unique date
# ---------------------------------------------------------

def chronological_split(df):
    unique_dates = sorted(df["date"].drop_duplicates().tolist())

    train_date_count = int(len(unique_dates) * TRAIN_FRACTION)
    validation_date_count = int(
        len(unique_dates) * VALIDATION_FRACTION
    )

    train_dates = set(unique_dates[:train_date_count])

    validation_start = train_date_count
    validation_end = train_date_count + validation_date_count

    validation_dates = set(
        unique_dates[validation_start:validation_end]
    )

    test_dates = set(unique_dates[validation_end:])

    train_df = df[df["date"].isin(train_dates)].copy()
    validation_df = df[df["date"].isin(validation_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)

    return train_df, validation_df, test_df


# ---------------------------------------------------------
# Chronological split of training data for calibration
# ---------------------------------------------------------

def split_base_and_calibration_data(train_df):
    unique_dates = sorted(train_df["date"].drop_duplicates().tolist())

    split_index = int(len(unique_dates) * BASE_TRAIN_FRACTION)

    base_train_dates = set(unique_dates[:split_index])
    calibration_dates = set(unique_dates[split_index:])

    base_train_df = train_df[
        train_df["date"].isin(base_train_dates)
    ].copy()

    calibration_df = train_df[
        train_df["date"].isin(calibration_dates)
    ].copy()

    if base_train_df.empty or calibration_df.empty:
        raise ValueError(
            "Base training or calibration split is empty. "
            "Adjust BASE_TRAIN_FRACTION."
        )

    assert base_train_dates.isdisjoint(calibration_dates)

    return base_train_df, calibration_df


# ---------------------------------------------------------
# Base Logistic Regression model
# ---------------------------------------------------------

def create_base_model():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ---------------------------------------------------------
# Probability benchmarks
# ---------------------------------------------------------

def elo_expected_probabilities(elo_difference):
    """
    Convert Elo difference into expected win probability.

    Assumes:
        elo_difference = Team 1 Elo - Team 2 Elo

    Standard Elo expected-score formula:
        P(Team 1 wins) = 1 / (1 + 10 ** (-difference / 400))
    """
    elo_difference = np.asarray(elo_difference, dtype=float)

    probabilities = 1 / (
        1 + np.power(10, -elo_difference / 400)
    )

    # Avoid exact 0 or 1 probabilities for numerical stability.
    return np.clip(probabilities, 1e-6, 1 - 1e-6)


def fifty_percent_probabilities(number_of_matches):
    return np.full(number_of_matches, 0.5, dtype=float)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(y_true, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "auc": roc_auc_score(y_true, probabilities),
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


def evaluate_probability_methods(
    y_true,
    elo_probabilities,
    uncalibrated_probabilities,
    calibrated_probabilities,
):
    baseline_probabilities = fifty_percent_probabilities(len(y_true))

    results = pd.DataFrame([
        {
            "method": "50% probability baseline",
            **calculate_metrics(y_true, baseline_probabilities),
        },
        {
            "method": "Raw Elo expected probability",
            **calculate_metrics(y_true, elo_probabilities),
        },
        {
            "method": "Uncalibrated Logistic Regression",
            **calculate_metrics(y_true, uncalibrated_probabilities),
        },
        {
            "method": "Chronology-safe sigmoid calibration",
            **calculate_metrics(y_true, calibrated_probabilities),
        },
    ])

    return results


# ---------------------------------------------------------
# Fit chronology-safe calibration model
# ---------------------------------------------------------

def fit_calibration_model(base_model, X_calibration, y_calibration):
    calibration_scores = base_model.decision_function(
        X_calibration
    ).reshape(-1, 1)

    sigmoid_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    sigmoid_model.fit(
        calibration_scores,
        y_calibration,
    )

    return sigmoid_model


def get_calibrated_probabilities(
    base_model,
    calibration_model,
    X,
):
    scores = base_model.decision_function(X).reshape(-1, 1)

    probabilities = calibration_model.predict_proba(scores)[:, 1]

    return probabilities


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def run_final_evaluation():
    df = load_data()

    train_df, validation_df, test_df = chronological_split(df)

    base_train_df, calibration_df = split_base_and_calibration_data(
        train_df
    )

    X_base_train = base_train_df[FEATURE_COLUMNS]
    y_base_train = base_train_df[TARGET_COLUMN].astype(int)

    X_calibration = calibration_df[FEATURE_COLUMNS]
    y_calibration = calibration_df[TARGET_COLUMN].astype(int)

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)

    print("\nDATASET")
    print("-" * 75)
    print(f"Total decided matches: {len(df):,}")
    print(f"Total unique dates: {df['date'].nunique():,}")

    print(
        f"\nBase model training: {len(base_train_df):,} matches | "
        f"{base_train_df['date'].min().date()} to "
        f"{base_train_df['date'].max().date()}"
    )

    print(
        f"Calibration window: {len(calibration_df):,} matches | "
        f"{calibration_df['date'].min().date()} to "
        f"{calibration_df['date'].max().date()}"
    )

    print(
        f"Validation: {len(validation_df):,} matches | "
        f"{validation_df['date'].min().date()} to "
        f"{validation_df['date'].max().date()}"
    )

    print(
        f"Test: {len(test_df):,} matches | "
        f"{test_df['date'].min().date()} to "
        f"{test_df['date'].max().date()}"
    )

    # -----------------------------------------------------
    # Fit models
    # -----------------------------------------------------

    print("\nFitting base Logistic Regression...")
    base_model = create_base_model()
    base_model.fit(X_base_train, y_base_train)

    print("Fitting sigmoid calibration on later training window...")
    calibration_model = fit_calibration_model(
        base_model,
        X_calibration,
        y_calibration,
    )

    # -----------------------------------------------------
    # Validation comparison
    # -----------------------------------------------------

    print("\nVALIDATION COMPARISON")
    print("-" * 75)

    validation_uncalibrated = base_model.predict_proba(
        X_validation
    )[:, 1]

    validation_calibrated = get_calibrated_probabilities(
        base_model,
        calibration_model,
        X_validation,
    )

    validation_elo_probabilities = elo_expected_probabilities(
        validation_df[ELO_DIFFERENCE_COLUMN]
    )

    validation_results = evaluate_probability_methods(
        y_true=y_validation,
        elo_probabilities=validation_elo_probabilities,
        uncalibrated_probabilities=validation_uncalibrated,
        calibrated_probabilities=validation_calibrated,
    )

    print(validation_results.to_string(index=False))

    # Select using validation log loss, then Brier score.
    # The 50% baseline is a reference, not a candidate model.
    candidate_results = validation_results[
        validation_results["method"] != "50% probability baseline"
    ].copy()

    candidate_results = candidate_results.sort_values(
        by=["log_loss", "brier_score"],
        ascending=[True, True],
    ).reset_index(drop=True)

    selected_method = candidate_results.iloc[0]["method"]

    print(f"\nSelected using validation: {selected_method}")

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------
    # Test results are reported for all methods.
    # The test set is not used to select the method.

    print("\nFINAL TEST COMPARISON")
    print("-" * 75)

    test_uncalibrated = base_model.predict_proba(
        X_test
    )[:, 1]

    test_calibrated = get_calibrated_probabilities(
        base_model,
        calibration_model,
        X_test,
    )

    test_elo_probabilities = elo_expected_probabilities(
        test_df[ELO_DIFFERENCE_COLUMN]
    )

    test_results = evaluate_probability_methods(
        y_true=y_test,
        elo_probabilities=test_elo_probabilities,
        uncalibrated_probabilities=test_uncalibrated,
        calibrated_probabilities=test_calibrated,
    )

    print(test_results.to_string(index=False))

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    validation_results.to_csv(
        VALIDATION_RESULTS_PATH,
        index=False,
    )

    test_results.to_csv(
        TEST_RESULTS_PATH,
        index=False,
    )

    summary_lines = [
        "ODI FINAL MODEL EVALUATION",
        "=" * 75,
        "",
        "Chronological split by unique date was used.",
        "Calibration was fitted using a later chronological training window.",
        "Validation was used to select the candidate model.",
        "The test set was evaluated after method selection.",
        "",
        "VALIDATION RESULTS",
        validation_results.to_string(index=False),
        "",
        f"Selected method based on validation log loss: {selected_method}",
        "",
        "FINAL TEST RESULTS",
        test_results.to_string(index=False),
        "",
        "Interpretation:",
        "- AUC measures ranking ability; higher is better.",
        "- Log loss and Brier score measure probability quality; lower is better.",
        "- Accuracy is measured using a 0.5 probability threshold.",
        "- Raw Elo is a meaningful strength-based benchmark.",
        "- The 50% baseline is a simple reference benchmark.",
        "- Results apply to this historical test period and may not",
        "  generalize perfectly to the 2027 World Cup.",
        "",
        "Important limitation:",
        "The base model was trained on the earlier 80% of the training period.",
        "The later 20% was reserved for calibration, so the base model does",
        "not use all pre-validation training matches.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nSaved files:")
    print(f"Validation results: {VALIDATION_RESULTS_PATH}")
    print(f"Test results:       {TEST_RESULTS_PATH}")
    print(f"Summary:            {SUMMARY_PATH}")


if __name__ == "__main__":
    run_final_evaluation()