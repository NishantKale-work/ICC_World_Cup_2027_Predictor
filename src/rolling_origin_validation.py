from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_features_with_venue_merged.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

WINDOW_RESULTS_FILE = (
    REPORTS_DIR / "rolling_origin_window_results.csv"
)

SUMMARY_RESULTS_FILE = (
    REPORTS_DIR / "rolling_origin_summary.csv"
)

PREDICTIONS_FILE = (
    REPORTS_DIR / "rolling_origin_predictions.csv"
)

REPORT_FILE = (
    REPORTS_DIR / "rolling_origin_validation.txt"
)

TARGET_COLUMN = "target"
DATE_COLUMN = "date"
MATCH_ID_COLUMN = "match_id"

RANDOM_STATE = 42

# Each window:
# - Train: all matches up to test start minus 6 years
# - Calibration: the following 4 years
# - Test: the next 2 years
#
# This gives an expanding training set and keeps test data
# strictly later than both training and calibration data.

TEST_WINDOWS = [
    ("2017-01-01", "2018-12-31"),
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
    ("2025-01-01", "2026-12-31"),
]

BASELINE_FEATURES = [
    "team_1_elo",
    "team_2_elo",
    "elo_difference",
    "team_1_recent_form",
    "team_2_recent_form",
    "recent_form_difference",
    "team_1_matches_played",
    "team_2_matches_played",
]

VENUE_FEATURES = [
    "team_1_venue_matches",
    "team_2_venue_matches",
    "venue_familiarity_difference",
    "team_1_home",
    "team_2_home",
    "home_advantage_difference",
]

VENUE_MODEL_FEATURES = BASELINE_FEATURES + VENUE_FEATURES


# ============================================================
# Utility functions
# ============================================================

def make_logistic_regression() -> Pipeline:
    """Create the logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def elo_probability(
    elo_difference: np.ndarray | pd.Series,
) -> np.ndarray:
    """Convert Elo difference into team-1 win probability."""
    elo_difference = np.asarray(
        elo_difference,
        dtype=float,
    )

    probabilities = 1.0 / (
        1.0 + np.power(10.0, -elo_difference / 400.0)
    )

    return np.clip(
        probabilities,
        1e-6,
        1.0 - 1e-6,
    )


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate standard binary classification metrics."""
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predictions,
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

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def fit_calibrated_model(
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    features: list[str],
    target_column: str,
) -> CalibratedClassifierCV:
    """
    Fit the model on the training period and fit sigmoid
    calibration on the later calibration period.
    """
    y_train = train_df[target_column].astype(int)

    model = make_logistic_regression()

    model.fit(
        train_df[features],
        y_train,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="sigmoid",
    )

    calibrated_model.fit(
        calibration_df[features],
        calibration_df[target_column].astype(int),
    )

    return calibrated_model


def get_optional_team_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================
# Main rolling-origin evaluation
# ============================================================

def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn",
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print("=" * 75)
    print("ROLLING-ORIGIN VALIDATION")
    print("=" * 75)
    print(f"Input: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = (
        [
            MATCH_ID_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
        ]
        + VENUE_MODEL_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The input file is missing required columns:\n"
            f"{missing_columns}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            MATCH_ID_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
        ]
    ).copy()

    df = df[
        df[TARGET_COLUMN].isin([0, 1])
    ].copy()

    # Only matches with complete venue features are included,
    # ensuring all three models use the same matches per window.
    df = df.dropna(
        subset=VENUE_FEATURES
    ).copy()

    df = df.sort_values(
        [DATE_COLUMN, MATCH_ID_COLUMN]
    ).reset_index(drop=True)

    if df[MATCH_ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate match IDs found after filtering."
        )

    print(f"\nTotal eligible decided matches: {len(df)}")
    print(
        f"Date range: {df[DATE_COLUMN].min().date()} "
        f"to {df[DATE_COLUMN].max().date()}"
    )

    all_window_results = []
    all_predictions = []

    # --------------------------------------------------------
    # Run each chronological window
    # --------------------------------------------------------

    for window_number, (
        test_start_text,
        test_end_text,
    ) in enumerate(TEST_WINDOWS, start=1):

        test_start = pd.Timestamp(test_start_text)
        test_end = pd.Timestamp(test_end_text)

        # Calibration period is the 4 years immediately
        # preceding the test period.
        calibration_start = (
            test_start - pd.DateOffset(years=4)
        )

        calibration_end = (
            test_start - pd.Timedelta(days=1)
        )

        # Training period ends before calibration begins.
        train_end = (
            calibration_start - pd.Timedelta(days=1)
        )

        train_df = df[
            df[DATE_COLUMN] <= train_end
        ].copy()

        calibration_df = df[
            (
                df[DATE_COLUMN] >= calibration_start
            )
            & (
                df[DATE_COLUMN] <= calibration_end
            )
        ].copy()

        test_df = df[
            (
                df[DATE_COLUMN] >= test_start
            )
            & (
                df[DATE_COLUMN] <= test_end
            )
        ].copy()

        print("\n" + "=" * 75)
        print(f"WINDOW {window_number}")
        print("=" * 75)

        print(
            f"Train:      {len(train_df):4} matches | "
            f"through {train_end.date()}"
        )

        print(
            f"Calibration:{len(calibration_df):4} matches | "
            f"{calibration_start.date()} to "
            f"{calibration_end.date()}"
        )

        print(
            f"Test:       {len(test_df):4} matches | "
            f"{test_start.date()} to "
            f"{test_end.date()}"
        )

        if train_df.empty:
            print("Skipping window: training data is empty.")
            continue

        if calibration_df.empty:
            print("Skipping window: calibration data is empty.")
            continue

        if test_df.empty:
            print("Skipping window: test data is empty.")
            continue

        if train_df[TARGET_COLUMN].nunique() < 2:
            print(
                "Skipping window: training data contains "
                "only one target class."
            )
            continue

        if calibration_df[TARGET_COLUMN].nunique() < 2:
            print(
                "Skipping window: calibration data contains "
                "only one target class."
            )
            continue

        y_test = test_df[
            TARGET_COLUMN
        ].astype(int).to_numpy()

        # ----------------------------------------------------
        # Model 1: Baseline logistic regression
        # ----------------------------------------------------

        print("Fitting baseline LR...")

        baseline_model = fit_calibrated_model(
            train_df=train_df,
            calibration_df=calibration_df,
            features=BASELINE_FEATURES,
            target_column=TARGET_COLUMN,
        )

        baseline_probabilities = (
            baseline_model.predict_proba(
                test_df[BASELINE_FEATURES]
            )[:, 1]
        )

        # ----------------------------------------------------
        # Model 2: Logistic regression + venue features
        # ----------------------------------------------------

        print("Fitting venue-feature LR...")

        venue_model = fit_calibrated_model(
            train_df=train_df,
            calibration_df=calibration_df,
            features=VENUE_MODEL_FEATURES,
            target_column=TARGET_COLUMN,
        )

        venue_probabilities = (
            venue_model.predict_proba(
                test_df[VENUE_MODEL_FEATURES]
            )[:, 1]
        )

        # ----------------------------------------------------
        # Model 3: Raw Elo K20
        # ----------------------------------------------------

        raw_elo_probabilities = elo_probability(
            test_df["elo_difference"].to_numpy()
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        model_probabilities = {
            "Baseline LR (calibrated)": baseline_probabilities,
            "Venue LR (calibrated)": venue_probabilities,
            "Raw Elo K20": raw_elo_probabilities,
        }

        for model_name, probabilities in (
            model_probabilities.items()
        ):
            metrics = calculate_metrics(
                y_test,
                probabilities,
            )

            all_window_results.append(
                {
                    "window": window_number,
                    "test_start": test_start.date(),
                    "test_end": test_end.date(),
                    "train_matches": len(train_df),
                    "calibration_matches": len(calibration_df),
                    "test_matches": len(test_df),
                    "model": model_name,
                    **metrics,
                }
            )

            print(f"\n{model_name}")
            print(
                f"  Accuracy:   "
                f"{metrics['accuracy']:.6f}"
            )
            print(
                f"  ROC AUC:    "
                f"{metrics['roc_auc']:.6f}"
            )
            print(
                f"  Log loss:   "
                f"{metrics['log_loss']:.6f}"
            )
            print(
                f"  Brier score:"
                f" {metrics['brier_score']:.6f}"
            )

        # ----------------------------------------------------
        # Save match-level predictions
        # ----------------------------------------------------

        prediction_df = pd.DataFrame(
            {
                MATCH_ID_COLUMN: test_df[
                    MATCH_ID_COLUMN
                ].to_numpy(),
                DATE_COLUMN: test_df[
                    DATE_COLUMN
                ].dt.strftime("%Y-%m-%d").to_numpy(),
                "window": window_number,
                TARGET_COLUMN: y_test,
                "baseline_probability": baseline_probabilities,
                "venue_probability": venue_probabilities,
                "raw_elo_k20_probability": raw_elo_probabilities,
            }
        )

        team_1_column = get_optional_team_column(
            test_df,
            [
                "team_1",
                "team1",
                "team_1_name",
                "team1_name",
            ],
        )

        team_2_column = get_optional_team_column(
            test_df,
            [
                "team_2",
                "team2",
                "team_2_name",
                "team2_name",
            ],
        )

        if team_1_column:
            prediction_df.insert(
                2,
                "team_1",
                test_df[team_1_column].to_numpy(),
            )

        if team_2_column:
            insert_position = (
                3 if team_1_column else 2
            )

            prediction_df.insert(
                insert_position,
                "team_2",
                test_df[team_2_column].to_numpy(),
            )

        all_predictions.append(prediction_df)

    # ========================================================
    # Save detailed results
    # ========================================================

    if not all_window_results:
        raise RuntimeError(
            "No rolling-origin windows completed. "
            "Check the data and date ranges."
        )

    results_df = pd.DataFrame(
        all_window_results
    )

    results_df.to_csv(
        WINDOW_RESULTS_FILE,
        index=False,
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # ========================================================
    # Summarize model performance across windows
    # ========================================================

    metric_columns = [
        "accuracy",
        "roc_auc",
        "log_loss",
        "brier_score",
    ]

    summary_df = (
        results_df
        .groupby("model")[metric_columns]
        .agg(["mean", "std"])
    )

    summary_df.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary_df.columns
    ]

    summary_df = summary_df.reset_index()

    summary_df.to_csv(
        SUMMARY_RESULTS_FILE,
        index=False,
    )

    # ========================================================
    # Text report
    # ========================================================

    report_lines = [
        "ROLLING-ORIGIN VALIDATION REPORT",
        "=" * 75,
        f"Input: {INPUT_FILE}",
        "",
        "Method:",
        (
            "For each test window, training uses only earlier matches. "
            "The immediately preceding four years are used for sigmoid "
            "calibration, and the following two years are used for test."
        ),
        (
            "All three models are evaluated on the same test matches "
            "within each window."
        ),
        "",
        "Window-level results:",
        "-" * 75,
    ]

    for _, row in results_df.iterrows():
        report_lines.extend(
            [
                (
                    f"Window {row['window']} | "
                    f"{row['test_start']} to {row['test_end']} | "
                    f"{row['model']}"
                ),
                (
                    f"  N={row['test_matches']} | "
                    f"Accuracy={row['accuracy']:.6f} | "
                    f"AUC={row['roc_auc']:.6f} | "
                    f"LogLoss={row['log_loss']:.6f} | "
                    f"Brier={row['brier_score']:.6f}"
                ),
                "",
            ]
        )

    report_lines.extend(
        [
            "Mean and standard deviation across windows:",
            "-" * 75,
            summary_df.to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            ),
            "",
            "Interpretation caution:",
            (
                "Rolling-origin results provide a stronger view of "
                "temporal consistency, but they do not guarantee future "
                "performance. Windows may differ in size and cricket "
                "conditions. Compare models on the same windows and "
                "inspect each window rather than relying only on averages."
            ),
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # ========================================================
    # Console summary
    # ========================================================

    print("\n" + "=" * 75)
    print("ROLLING-ORIGIN SUMMARY")
    print("=" * 75)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved outputs:")
    print(f"  {WINDOW_RESULTS_FILE}")
    print(f"  {SUMMARY_RESULTS_FILE}")
    print(f"  {PREDICTIONS_FILE}")
    print(f"  {REPORT_FILE}")

    print("\nRolling-origin validation complete.")


if __name__ == "__main__":
    main()