"""
Compare ODI match prediction models using a strict chronological split.

Models compared:
1. Elo K=10
2. Elo K=20
3. Elo K=30
4. Elo K=40
5. Logistic Regression - raw probabilities
6. Logistic Regression - sigmoid calibrated probabilities
7. Random Forest - raw probabilities

Model selection:
- Validation log loss is the primary criterion.
- Validation Brier score is the secondary criterion.
- Test results are reported only after validation-based selection.

Expected input files:
- data/processed/odi_features.csv
- reports/elo_k_experiment_predictions.csv

Outputs:
- reports/model_comparison_validation.csv
- reports/model_comparison_test.csv
- reports/model_comparison_predictions.csv
- reports/model_comparison_summary.txt
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "odi_features.csv"
ELO_PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "elo_k_experiment_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "target"
DATE_COLUMN = "date"
MATCH_ID_COLUMN = "match_id"

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

ELO_K_VALUES = [10, 20, 30, 40]

PROBABILITY_COLUMN_CANDIDATES = [
    "team_1_win_probability",
    "team_1_probability",
    "elo_probability",
    "predicted_probability",
    "probability",
]

SPLIT_COLUMN_CANDIDATES = [
    "split",
    "dataset_split",
    "data_split",
]

# Common values used in experiment output files.
SPLIT_ALIASES = {
    "validation": "validation",
    "val": "validation",
    "valid": "validation",
    "test": "test",
    "train": "train",
}


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def print_section(title: str) -> None:
    """Print a readable section heading."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Keep probabilities away from exactly zero or one."""
    return np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1 - 1e-6,
    )


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Calculate classification and probability-quality metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = clip_probabilities(probabilities)

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
    }


def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first matching column from a list of candidates."""
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


# ============================================================
# 3. LOAD AND VALIDATE FEATURE DATA
# ============================================================

def load_feature_data() -> pd.DataFrame:
    """Load the engineered ODI features and validate their structure."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURES_PATH}\n\n"
            "Run build_match_dataset.py and build_features.py first."
        )

    dataframe = pd.read_csv(FEATURES_PATH)

    require_columns(
        dataframe,
        [
            MATCH_ID_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
            *FEATURE_COLUMNS,
        ],
        "Feature dataset",
    )

    dataframe[DATE_COLUMN] = pd.to_datetime(
        dataframe[DATE_COLUMN],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            MATCH_ID_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
        ]
    ).copy()

    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe[TARGET_COLUMN].isin([0, 1])
    ].copy()

    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)

    dataframe = dataframe.sort_values(
        [DATE_COLUMN, MATCH_ID_COLUMN]
    ).reset_index(drop=True)

    if dataframe[MATCH_ID_COLUMN].duplicated().any():
        duplicate_count = dataframe[MATCH_ID_COLUMN].duplicated().sum()

        raise ValueError(
            f"Feature dataset contains {duplicate_count} duplicate match IDs."
        )

    if dataframe.empty:
        raise ValueError("No valid decided matches found in feature dataset.")

    return dataframe


# ============================================================
# 4. CREATE EXACT CHRONOLOGICAL SPLIT
# ============================================================

def create_chronological_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique dates so matches played on the same date
    cannot appear in different datasets.
    """
    unique_dates = np.array(
        sorted(dataframe[DATE_COLUMN].unique())
    )

    total_dates = len(unique_dates)

    train_date_count = int(total_dates * TRAIN_FRACTION)
    validation_date_count = int(
        total_dates * VALIDATION_FRACTION
    )

    train_dates = unique_dates[:train_date_count]

    validation_dates = unique_dates[
        train_date_count:
        train_date_count + validation_date_count
    ]

    test_dates = unique_dates[
        train_date_count + validation_date_count:
    ]

    train_dataframe = dataframe[
        dataframe[DATE_COLUMN].isin(train_dates)
    ].copy()

    validation_dataframe = dataframe[
        dataframe[DATE_COLUMN].isin(validation_dates)
    ].copy()

    test_dataframe = dataframe[
        dataframe[DATE_COLUMN].isin(test_dates)
    ].copy()

    print_section("CHRONOLOGICAL SPLIT")

    print(f"Unique dates: {total_dates:,}")

    print(
        f"Train:      {len(train_dataframe):,} matches | "
        f"{train_dataframe[DATE_COLUMN].min().date()} to "
        f"{train_dataframe[DATE_COLUMN].max().date()}"
    )

    print(
        f"Validation: {len(validation_dataframe):,} matches | "
        f"{validation_dataframe[DATE_COLUMN].min().date()} to "
        f"{validation_dataframe[DATE_COLUMN].max().date()}"
    )

    print(
        f"Test:       {len(test_dataframe):,} matches | "
        f"{test_dataframe[DATE_COLUMN].min().date()} to "
        f"{test_dataframe[DATE_COLUMN].max().date()}"
    )

    if (
        train_dataframe[DATE_COLUMN].max()
        >= validation_dataframe[DATE_COLUMN].min()
    ):
        raise AssertionError("Train and validation dates overlap.")

    if (
        validation_dataframe[DATE_COLUMN].max()
        >= test_dataframe[DATE_COLUMN].min()
    ):
        raise AssertionError("Validation and test dates overlap.")

    return (
        train_dataframe.reset_index(drop=True),
        validation_dataframe.reset_index(drop=True),
        test_dataframe.reset_index(drop=True),
    )


# ============================================================
# 5. TRAIN LOGISTIC REGRESSION AND RANDOM FOREST
# ============================================================

def create_logistic_regression_pipeline() -> Pipeline:
    """Create the same Logistic Regression pipeline as final_evaluation.py."""
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


def create_random_forest_pipeline() -> Pipeline:
    """
    Create a single pre-specified Random Forest challenger.

    This is intentionally not heavily tuned. The validation set should
    compare a reasonable challenger, not become a tuning playground.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_statistical_models(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Train Logistic Regression and Random Forest.

    The first 80% of training dates fit the model.
    The later 20% of training dates are reserved for calibration.
    This mirrors the existing Logistic Regression evaluation design.
    """
    unique_train_dates = np.array(
        sorted(train_dataframe[DATE_COLUMN].unique())
    )

    base_train_date_count = int(
        len(unique_train_dates) * BASE_TRAIN_FRACTION
    )

    base_train_dates = unique_train_dates[
        :base_train_date_count
    ]

    calibration_dates = unique_train_dates[
        base_train_date_count:
    ]

    base_train = train_dataframe[
        train_dataframe[DATE_COLUMN].isin(base_train_dates)
    ].copy()

    calibration_dataframe = train_dataframe[
        train_dataframe[DATE_COLUMN].isin(calibration_dates)
    ].copy()

    X_base = base_train[FEATURE_COLUMNS]
    y_base = base_train[TARGET_COLUMN]

    X_calibration = calibration_dataframe[FEATURE_COLUMNS]
    y_calibration = calibration_dataframe[TARGET_COLUMN]

    X_validation = validation_dataframe[FEATURE_COLUMNS]
    y_validation = validation_dataframe[TARGET_COLUMN]

    X_test = test_dataframe[FEATURE_COLUMNS]
    y_test = test_dataframe[TARGET_COLUMN]

    print_section("STATISTICAL MODEL TRAINING")

    print(
        f"Base model training: {len(base_train):,} matches | "
        f"{base_train[DATE_COLUMN].min().date()} to "
        f"{base_train[DATE_COLUMN].max().date()}"
    )

    print(
        f"Calibration window:  {len(calibration_dataframe):,} matches | "
        f"{calibration_dataframe[DATE_COLUMN].min().date()} to "
        f"{calibration_dataframe[DATE_COLUMN].max().date()}"
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic_model = create_logistic_regression_pipeline()

    logistic_model.fit(X_base, y_base)

    lr_calibration_scores = logistic_model.decision_function(
        X_calibration
    ).reshape(-1, 1)

    sigmoid_calibrator = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    sigmoid_calibrator.fit(
        lr_calibration_scores,
        y_calibration,
    )

    lr_validation_raw = logistic_model.predict_proba(
        X_validation
    )[:, 1]

    lr_test_raw = logistic_model.predict_proba(
        X_test
    )[:, 1]

    lr_validation_calibration_scores = (
        logistic_model.decision_function(X_validation)
        .reshape(-1, 1)
    )

    lr_test_calibration_scores = (
        logistic_model.decision_function(X_test)
        .reshape(-1, 1)
    )

    lr_validation_calibrated = sigmoid_calibrator.predict_proba(
        lr_validation_calibration_scores
    )[:, 1]

    lr_test_calibrated = sigmoid_calibrator.predict_proba(
        lr_test_calibration_scores
    )[:, 1]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest_model = create_random_forest_pipeline()

    random_forest_model.fit(X_base, y_base)

    rf_validation = random_forest_model.predict_proba(
        X_validation
    )[:, 1]

    rf_test = random_forest_model.predict_proba(
        X_test
    )[:, 1]

    return {
        "Logistic Regression (raw)": {
            "validation": clip_probabilities(lr_validation_raw),
            "test": clip_probabilities(lr_test_raw),
        },
        "Logistic Regression (calibrated)": {
            "validation": clip_probabilities(
                lr_validation_calibrated
            ),
            "test": clip_probabilities(
                lr_test_calibrated
            ),
        },
        "Random Forest (raw)": {
            "validation": clip_probabilities(rf_validation),
            "test": clip_probabilities(rf_test),
        },
    }


# ============================================================
# 6. LOAD ELO EXPERIMENT PREDICTIONS
# ============================================================

def load_elo_predictions(
    expected_dataframe: pd.DataFrame,
    split_name: str,
) -> dict[str, np.ndarray]:
    """
    Load saved Elo predictions for a specific split.

    The script supports common probability column names, but requires
    match IDs, K values, targets, and a split column.
    """
    if not ELO_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Elo prediction file not found:\n{ELO_PREDICTIONS_PATH}\n\n"
            "Run src/experiment_elo.py first and confirm it creates "
            "reports/elo_k_experiment_predictions.csv."
        )

    elo_dataframe = pd.read_csv(ELO_PREDICTIONS_PATH)

    require_columns(
        elo_dataframe,
        [
            MATCH_ID_COLUMN,
            "k_value",
            TARGET_COLUMN,
        ],
        "Elo prediction dataset",
    )

    probability_column = find_first_existing_column(
        elo_dataframe,
        PROBABILITY_COLUMN_CANDIDATES,
    )

    if probability_column is None:
        raise ValueError(
            "Could not identify the Elo probability column.\n"
            f"Expected one of: {PROBABILITY_COLUMN_CANDIDATES}\n"
            f"Available columns: {list(elo_dataframe.columns)}"
        )

    split_column = find_first_existing_column(
        elo_dataframe,
        SPLIT_COLUMN_CANDIDATES,
    )

    if split_column is None:
        raise ValueError(
            "Could not identify the split column in the Elo prediction file.\n"
            f"Expected one of: {SPLIT_COLUMN_CANDIDATES}\n"
            f"Available columns: {list(elo_dataframe.columns)}"
        )

    elo_dataframe["normalized_split"] = (
        elo_dataframe[split_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(SPLIT_ALIASES)
    )

    if elo_dataframe["normalized_split"].isna().any():
        unexpected_values = (
            elo_dataframe.loc[
                elo_dataframe["normalized_split"].isna(),
                split_column,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Unrecognized split labels in Elo predictions: "
            f"{unexpected_values}"
        )

    elo_dataframe[TARGET_COLUMN] = pd.to_numeric(
        elo_dataframe[TARGET_COLUMN],
        errors="coerce",
    )

    elo_dataframe[probability_column] = pd.to_numeric(
        elo_dataframe[probability_column],
        errors="coerce",
    )

    elo_dataframe = elo_dataframe[
        elo_dataframe["normalized_split"] == split_name
    ].copy()

    expected_ids = set(
        expected_dataframe[MATCH_ID_COLUMN].tolist()
    )

    predictions_by_k = {}

    for k_value in ELO_K_VALUES:
        k_dataframe = elo_dataframe[
            elo_dataframe["k_value"] == k_value
        ].copy()

        if k_dataframe[MATCH_ID_COLUMN].duplicated().any():
            raise ValueError(
                f"Elo K={k_value} contains duplicate match IDs "
                f"for split '{split_name}'."
            )

        actual_ids = set(
            k_dataframe[MATCH_ID_COLUMN].tolist()
        )

        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids

        if missing_ids or extra_ids:
            raise ValueError(
                f"Elo K={k_value} match IDs do not align with "
                f"the feature dataset for split '{split_name}'.\n"
                f"Missing IDs: {len(missing_ids)}\n"
                f"Extra IDs: {len(extra_ids)}"
            )

        k_dataframe = k_dataframe.set_index(
            MATCH_ID_COLUMN
        ).loc[
            expected_dataframe[MATCH_ID_COLUMN]
        ].reset_index()

        if not np.array_equal(
            k_dataframe[TARGET_COLUMN].astype(int).to_numpy(),
            expected_dataframe[TARGET_COLUMN].astype(int).to_numpy(),
        ):
            raise ValueError(
                f"Targets do not align for Elo K={k_value}, "
                f"split '{split_name}'."
            )

        predictions_by_k[f"Elo K={k_value}"] = clip_probabilities(
            k_dataframe[probability_column].to_numpy()
        )

    return predictions_by_k


# ============================================================
# 7. EVALUATE AND SAVE RESULTS
# ============================================================

def evaluate_models(
    split_name: str,
    dataframe: pd.DataFrame,
    predictions_by_model: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate metrics and assemble match-level predictions."""
    y_true = dataframe[TARGET_COLUMN].to_numpy()

    metric_rows = []
    prediction_rows = []

    for model_name, probabilities in predictions_by_model.items():
        probabilities = clip_probabilities(probabilities)

        metrics = calculate_metrics(
            y_true,
            probabilities,
        )

        metric_rows.append(
            {
                "split": split_name,
                "model": model_name,
                **metrics,
            }
        )

        model_predictions = pd.DataFrame(
            {
                MATCH_ID_COLUMN: dataframe[MATCH_ID_COLUMN].to_numpy(),
                DATE_COLUMN: dataframe[DATE_COLUMN].to_numpy(),
                TARGET_COLUMN: y_true,
                "split": split_name,
                "model": model_name,
                "probability": probabilities,
            }
        )

        prediction_rows.append(model_predictions)

    metrics_dataframe = pd.DataFrame(metric_rows)

    predictions_dataframe = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    return metrics_dataframe, predictions_dataframe


def main() -> None:
    """Run the complete comparison experiment."""
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load features and create exact split
    # --------------------------------------------------------

    dataframe = load_feature_data()

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_chronological_split(dataframe)

    # --------------------------------------------------------
    # Train statistical models
    # --------------------------------------------------------

    statistical_predictions = train_statistical_models(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )

    # --------------------------------------------------------
    # Load Elo predictions for validation and test
    # --------------------------------------------------------

    validation_elo_predictions = load_elo_predictions(
        validation_dataframe,
        "validation",
    )

    test_elo_predictions = load_elo_predictions(
        test_dataframe,
        "test",
    )

    # --------------------------------------------------------
    # Combine predictions
    # --------------------------------------------------------

    validation_predictions = {
        **validation_elo_predictions,
        **{
            model_name: model_predictions["validation"]
            for model_name, model_predictions
            in statistical_predictions.items()
        },
    }

    test_predictions = {
        **test_elo_predictions,
        **{
            model_name: model_predictions["test"]
            for model_name, model_predictions
            in statistical_predictions.items()
        },
    }

    # --------------------------------------------------------
    # Evaluate validation and test
    # --------------------------------------------------------

    validation_metrics, validation_prediction_rows = evaluate_models(
        "validation",
        validation_dataframe,
        validation_predictions,
    )

    test_metrics, test_prediction_rows = evaluate_models(
        "test",
        test_dataframe,
        test_predictions,
    )

    # --------------------------------------------------------
    # Select model using validation only
    # --------------------------------------------------------

    validation_ranked = validation_metrics.sort_values(
        ["log_loss", "brier_score"],
        ascending=[True, True],
    ).reset_index(drop=True)

    selected_model = validation_ranked.iloc[0]["model"]

    validation_ranked["validation_rank"] = (
        np.arange(len(validation_ranked)) + 1
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    validation_metrics_path = (
        REPORTS_DIR / "model_comparison_validation.csv"
    )

    test_metrics_path = (
        REPORTS_DIR / "model_comparison_test.csv"
    )

    predictions_path = (
        REPORTS_DIR / "model_comparison_predictions.csv"
    )

    summary_path = (
        REPORTS_DIR / "model_comparison_summary.txt"
    )

    validation_ranked.to_csv(
        validation_metrics_path,
        index=False,
    )

    test_metrics.to_csv(
        test_metrics_path,
        index=False,
    )

    all_predictions = pd.concat(
        [
            validation_prediction_rows,
            test_prediction_rows,
        ],
        ignore_index=True,
    )

    all_predictions.to_csv(
        predictions_path,
        index=False,
    )

    selected_test_row = test_metrics[
        test_metrics["model"] == selected_model
    ].iloc[0]

    summary_lines = [
        "ODI MODEL COMPARISON SUMMARY",
        "=" * 50,
        "",
        "Selection criterion:",
        "1. Lowest validation log loss",
        "2. Lowest validation Brier score as tie-breaker",
        "",
        f"Selected model: {selected_model}",
        "",
        "Selected model validation metrics:",
        validation_ranked[
            validation_ranked["model"] == selected_model
        ].to_string(index=False),
        "",
        "Selected model test metrics:",
        selected_test_row.to_string(),
        "",
        "Important:",
        "The model was selected using validation metrics only.",
        "Test metrics are reported for final assessment and were not",
        "used to choose the model.",
    ]

    summary_text = "\n".join(summary_lines)

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_section("VALIDATION MODEL COMPARISON")

    print(
        validation_ranked[
            [
                "validation_rank",
                "model",
                "accuracy",
                "auc",
                "log_loss",
                "brier_score",
            ]
        ].to_string(index=False)
    )

    print_section("TEST MODEL COMPARISON")

    print(
        test_metrics[
            [
                "model",
                "accuracy",
                "auc",
                "log_loss",
                "brier_score",
            ]
        ]
        .sort_values("log_loss")
        .to_string(index=False)
    )

    print_section("VALIDATION-SELECTED MODEL")

    print(f"Selected model: {selected_model}")

    print("\nSelected model's test metrics:")
    print(selected_test_row.to_string())

    print_section("SAVED FILES")

    print(validation_metrics_path)
    print(test_metrics_path)
    print(predictions_path)
    print(summary_path)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nERROR")
        print("-" * 72)
        print(error)
        sys.exit(1)