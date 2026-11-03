
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
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

RESULTS_PATH = OUTPUT_DIR / "calibration_experiment_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "calibration_experiment_summary.txt"

TARGET_COLUMN = "target"

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
# Base model
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
# Metrics
# ---------------------------------------------------------

def evaluate_model(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y, predictions),
        "auc": roc_auc_score(y, probabilities),
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y, probabilities),
    }


# ---------------------------------------------------------
# Calibration experiment
# ---------------------------------------------------------

def run_experiment():
    df = load_data()
    train_df, validation_df, test_df = chronological_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN].astype(int)

    print("\nDATASET")
    print("-" * 60)
    print(f"Decided matches: {len(df):,}")
    print(f"Train: {len(train_df):,} matches")
    print(f"Validation: {len(validation_df):,} matches")
    print(f"Test: {len(test_df):,} matches")
    print("Test set will not be used in this experiment.")

    results = []

    # -----------------------------------------------------
    # 1. Uncalibrated model
    # -----------------------------------------------------

    print("\nFitting uncalibrated Logistic Regression...")

    base_model = create_base_model()
    base_model.fit(X_train, y_train)

    base_metrics = evaluate_model(
        base_model,
        X_validation,
        y_validation,
    )

    results.append({
        "method": "Uncalibrated Logistic Regression",
        **base_metrics,
    })

    # -----------------------------------------------------
    # 2. Sigmoid calibration
    # -----------------------------------------------------

    print("Fitting sigmoid-calibrated Logistic Regression...")

    sigmoid_model = CalibratedClassifierCV(
        estimator=clone(create_base_model()),
        method="sigmoid",
        cv=5,
    )

    sigmoid_model.fit(X_train, y_train)

    sigmoid_metrics = evaluate_model(
        sigmoid_model,
        X_validation,
        y_validation,
    )

    results.append({
        "method": "Sigmoid calibration",
        **sigmoid_metrics,
    })

    # -----------------------------------------------------
    # 3. Isotonic calibration
    # -----------------------------------------------------

    print("Fitting isotonic-calibrated Logistic Regression...")

    isotonic_model = CalibratedClassifierCV(
        estimator=clone(create_base_model()),
        method="isotonic",
        cv=5,
    )

    isotonic_model.fit(X_train, y_train)

    isotonic_metrics = evaluate_model(
        isotonic_model,
        X_validation,
        y_validation,
    )

    results.append({
        "method": "Isotonic calibration",
        **isotonic_metrics,
    })

    # -----------------------------------------------------
    # Save and display results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["log_loss", "brier_score"],
        ascending=[True, True],
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_PATH, index=False)

    best_result = results_df.iloc[0]

    summary_lines = [
        "ODI CALIBRATION EXPERIMENT SUMMARY",
        "=" * 60,
        "",
        "Calibration methods were compared on chronological validation data.",
        "The test set was not used.",
        "",
        results_df.to_string(index=False),
        "",
        f"Best validation method: {best_result['method']}",
        f"Log loss: {best_result['log_loss']:.4f}",
        f"Brier score: {best_result['brier_score']:.4f}",
        "",
        "Lower log loss and Brier score indicate better probability quality.",
        "AUC measures ranking ability; calibration does not necessarily improve it.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print(f"\nBest validation method: {best_result['method']}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_experiment()
