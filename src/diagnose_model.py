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

ERA_RESULTS_PATH = OUTPUT_DIR / "diagnostic_results_by_year.csv"
BUCKET_RESULTS_PATH = OUTPUT_DIR / "diagnostic_probability_buckets.csv"
SUMMARY_PATH = OUTPUT_DIR / "diagnostic_summary.txt"

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

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
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
        raise ValueError("No decided matches found.")

    if df["date"].isna().any():
        raise ValueError("Some matches have missing dates.")

    if df["match_id"].duplicated().any():
        raise ValueError("Duplicate match IDs found.")

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
# Chronological calibration split
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
        raise ValueError("Base training or calibration split is empty.")

    assert base_train_dates.isdisjoint(calibration_dates)

    return base_train_df, calibration_df


# ---------------------------------------------------------
# Models
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


def elo_expected_probabilities(elo_difference):
    elo_difference = np.asarray(elo_difference, dtype=float)

    probabilities = 1 / (
        1 + np.power(10, -elo_difference / 400)
    )

    return np.clip(probabilities, 1e-6, 1 - 1e-6)


def fit_calibration_model(base_model, X_calibration, y_calibration):
    scores = base_model.decision_function(
        X_calibration
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(scores, y_calibration)

    return calibrator


def get_calibrated_probabilities(base_model, calibrator, X):
    scores = base_model.decision_function(X).reshape(-1, 1)

    return calibrator.predict_proba(scores)[:, 1]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "matches": len(y_true),
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
        "actual_team_1_win_rate": y_true.mean(),
        "mean_predicted_probability": probabilities.mean(),
    }


# ---------------------------------------------------------
# Year-by-year diagnostics
# ---------------------------------------------------------

def evaluate_by_year(test_df):
    results = []

    methods = {
        "Raw Elo": "elo_probability",
        "Uncalibrated Logistic Regression": "logistic_probability",
        "Calibrated Logistic Regression": "calibrated_probability",
    }

    for year, year_df in test_df.groupby(test_df["date"].dt.year):
        y_true = year_df[TARGET_COLUMN].astype(int)

        for method_name, probability_column in methods.items():
            probabilities = year_df[probability_column].to_numpy()

            # AUC requires both outcome classes in the subset.
            if y_true.nunique() < 2:
                metrics = {
                    "matches": len(year_df),
                    "accuracy": accuracy_score(
                        y_true,
                        (probabilities >= 0.5).astype(int),
                    ),
                    "auc": np.nan,
                    "log_loss": log_loss(
                        y_true,
                        np.clip(probabilities, 1e-6, 1 - 1e-6),
                        labels=[0, 1],
                    ),
                    "brier_score": brier_score_loss(
                        y_true,
                        probabilities,
                    ),
                    "actual_team_1_win_rate": y_true.mean(),
                    "mean_predicted_probability": probabilities.mean(),
                }
            else:
                metrics = calculate_metrics(y_true, probabilities)

            results.append({
                "year": year,
                "method": method_name,
                **metrics,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Probability bucket diagnostics
# ---------------------------------------------------------

def evaluate_probability_buckets(test_df):
    methods = {
        "Raw Elo": "elo_probability",
        "Uncalibrated Logistic Regression": "logistic_probability",
        "Calibrated Logistic Regression": "calibrated_probability",
    }

    bucket_edges = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    results = []

    for method_name, probability_column in methods.items():
        method_df = test_df.copy()

        method_df["probability"] = method_df[probability_column]

        method_df["probability_bucket"] = pd.cut(
            method_df["probability"],
            bins=bucket_edges,
            include_lowest=True,
            right=True,
        )

        for bucket, bucket_df in method_df.groupby(
            "probability_bucket",
            observed=False,
        ):
            if bucket_df.empty:
                continue

            results.append({
                "method": method_name,
                "probability_bucket": str(bucket),
                "matches": len(bucket_df),
                "mean_predicted_probability": (
                    bucket_df["probability"].mean()
                ),
                "actual_team_1_win_rate": (
                    bucket_df[TARGET_COLUMN].mean()
                ),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Main diagnostic run
# ---------------------------------------------------------

def run_diagnostics():
    df = load_data()

    train_df, validation_df, test_df = chronological_split(df)

    base_train_df, calibration_df = split_base_and_calibration_data(
        train_df
    )

    X_base_train = base_train_df[FEATURE_COLUMNS]
    y_base_train = base_train_df[TARGET_COLUMN].astype(int)

    X_calibration = calibration_df[FEATURE_COLUMNS]
    y_calibration = calibration_df[TARGET_COLUMN].astype(int)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)

    print("\nDIAGNOSTIC DATASET")
    print("-" * 75)
    print(f"Total decided matches: {len(df):,}")
    print(f"Test matches: {len(test_df):,}")
    print(
        f"Test period: {test_df['date'].min().date()} to "
        f"{test_df['date'].max().date()}"
    )

    print("\nFitting Logistic Regression...")
    base_model = create_base_model()
    base_model.fit(X_base_train, y_base_train)

    print("Fitting chronology-safe sigmoid calibration...")
    calibrator = fit_calibration_model(
        base_model,
        X_calibration,
        y_calibration,
    )

    # -----------------------------------------------------
    # Generate test probabilities
    # -----------------------------------------------------

    test_df = test_df.copy()

    test_df["elo_probability"] = elo_expected_probabilities(
        test_df[ELO_DIFFERENCE_COLUMN]
    )

    test_df["logistic_probability"] = base_model.predict_proba(
        X_test
    )[:, 1]

    test_df["calibrated_probability"] = (
        get_calibrated_probabilities(
            base_model,
            calibrator,
            X_test,
        )
    )

    # -----------------------------------------------------
    # Overall test comparison
    # -----------------------------------------------------

    print("\nOVERALL TEST METRICS")
    print("-" * 75)

    overall_results = []

    for method_name, probability_column in {
        "Raw Elo": "elo_probability",
        "Uncalibrated Logistic Regression": "logistic_probability",
        "Calibrated Logistic Regression": "calibrated_probability",
    }.items():
        metrics = calculate_metrics(
            test_df[TARGET_COLUMN],
            test_df[probability_column],
        )

        overall_results.append({
            "method": method_name,
            **metrics,
        })

    overall_results = pd.DataFrame(overall_results)
    print(overall_results.to_string(index=False))

    # -----------------------------------------------------
    # Year-by-year performance
    # -----------------------------------------------------

    print("\nTEST PERFORMANCE BY YEAR")
    print("-" * 75)

    year_results = evaluate_by_year(test_df)
    print(year_results.to_string(index=False))

    # -----------------------------------------------------
    # Probability bucket analysis
    # -----------------------------------------------------

    print("\nPROBABILITY BUCKET DIAGNOSTICS")
    print("-" * 75)

    bucket_results = evaluate_probability_buckets(test_df)
    print(bucket_results.to_string(index=False))

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    year_results.to_csv(
        ERA_RESULTS_PATH,
        index=False,
    )

    bucket_results.to_csv(
        BUCKET_RESULTS_PATH,
        index=False,
    )

    summary_lines = [
        "ODI MODEL DIAGNOSTICS",
        "=" * 75,
        "",
        f"Total decided matches: {len(df):,}",
        f"Test matches: {len(test_df):,}",
        f"Test period: {test_df['date'].min().date()} to "
        f"{test_df['date'].max().date()}",
        "",
        "OVERALL TEST METRICS",
        overall_results.to_string(index=False),
        "",
        "TEST PERFORMANCE BY YEAR",
        year_results.to_string(index=False),
        "",
        "PROBABILITY BUCKET DIAGNOSTICS",
        bucket_results.to_string(index=False),
        "",
        "Interpretation guidance:",
        "- AUC measures ranking ability; higher is better.",
        "- Log loss and Brier score measure probability quality; lower is better.",
        "- In probability buckets, mean predicted probability should be",
        "  reasonably close to the actual Team 1 win rate.",
        "- Small buckets are noisy and should not be overinterpreted.",
        "- Year-to-year variation may reflect changing teams, formats,",
        "  match conditions, and limited sample sizes.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nSaved diagnostic files:")
    print(f"By-year results: {ERA_RESULTS_PATH}")
    print(f"Probability buckets: {BUCKET_RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_diagnostics()