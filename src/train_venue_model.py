
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "venue_logistic_regression_2027.joblib"

TARGET_COLUMN = "target"
DATE_COLUMN = "date"
MATCH_ID_COLUMN = "match_id"

# Final model training cutoff.
# All matches on or before this date can be used to fit the model.
TRAIN_END_DATE = pd.Timestamp("2023-06-25")

# Calibration period:
# Matches after TRAIN_END_DATE and on or before this date are used
# to fit the probability calibration layer.
CALIBRATION_END_DATE = pd.Timestamp("2026-10-25")

RANDOM_STATE = 42

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

ALL_FEATURES = BASELINE_FEATURES + VENUE_FEATURES


# ============================================================
# Model creation
# ============================================================

def make_logistic_regression() -> Pipeline:
    """
    Create the venue-feature logistic regression pipeline.

    The imputer and scaler are fitted only on the training data.
    """
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


# ============================================================
# Data loading and validation
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load and validate the historical venue-feature dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset was not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        MATCH_ID_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    ] + ALL_FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The input dataset is missing required columns:\n"
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

    # Keep only matches with a valid date, ID, and decided result.
    df = df.dropna(
        subset=[
            MATCH_ID_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
        ]
    ).copy()

    df = df[df[TARGET_COLUMN].isin([0, 1])].copy()

    # Keep only matches involving the working 2027 teams,
    # for which all venue features were generated.
    df = df.dropna(subset=VENUE_FEATURES).copy()

    df = df.sort_values(
        [DATE_COLUMN, MATCH_ID_COLUMN]
    ).reset_index(drop=True)

    if df[MATCH_ID_COLUMN].duplicated().any():
        duplicate_count = int(
            df[MATCH_ID_COLUMN].duplicated().sum()
        )

        raise ValueError(
            f"Found {duplicate_count} duplicate match IDs "
            "after filtering."
        )

    if df.empty:
        raise ValueError(
            "No eligible decided matches remain after filtering."
        )

    return df


# ============================================================
# Main training workflow
# ============================================================

def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAIN FINAL VENUE MODEL FOR 2027")
    print("=" * 70)

    print(f"Input dataset:\n{INPUT_FILE}")

    df = load_dataset()

    # --------------------------------------------------------
    # Chronological training and calibration split
    # --------------------------------------------------------

    train_df = df[
        df[DATE_COLUMN] <= TRAIN_END_DATE
    ].copy()

    calibration_df = df[
        (df[DATE_COLUMN] > TRAIN_END_DATE)
        & (df[DATE_COLUMN] <= CALIBRATION_END_DATE)
    ].copy()

    if train_df.empty:
        raise ValueError(
            "The training period contains no eligible matches."
        )

    if calibration_df.empty:
        raise ValueError(
            "The calibration period contains no eligible matches."
        )

    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()

    y_calibration = (
        calibration_df[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    if len(np.unique(y_train)) != 2:
        raise ValueError(
            "The training data must contain both target classes."
        )

    if len(np.unique(y_calibration)) != 2:
        raise ValueError(
            "The calibration data must contain both target classes."
        )

    print("\nDataset summary")
    print("-" * 70)

    print(f"Eligible decided matches: {len(df)}")
    print(
        f"Overall date range: "
        f"{df[DATE_COLUMN].min().date()} to "
        f"{df[DATE_COLUMN].max().date()}"
    )

    print("\nTraining period")
    print(
        f"Matches: {len(train_df)} | "
        f"{train_df[DATE_COLUMN].min().date()} to "
        f"{train_df[DATE_COLUMN].max().date()}"
    )

    print("\nCalibration period")
    print(
        f"Matches: {len(calibration_df)} | "
        f"{calibration_df[DATE_COLUMN].min().date()} to "
        f"{calibration_df[DATE_COLUMN].max().date()}"
    )

    print(f"\nFeature count: {len(ALL_FEATURES)}")

    # --------------------------------------------------------
    # Fit venue-feature logistic regression
    # --------------------------------------------------------

    print("\nFitting venue-feature logistic regression...")

    base_model = make_logistic_regression()

    base_model.fit(
        train_df[ALL_FEATURES],
        y_train,
    )

    # --------------------------------------------------------
    # Fit sigmoid calibration on later historical matches
    # --------------------------------------------------------

    print("Fitting sigmoid probability calibration...")

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid",
    )

    calibrated_model.fit(
        calibration_df[ALL_FEATURES],
        y_calibration,
    )

    # --------------------------------------------------------
    # Save model and metadata
    # --------------------------------------------------------

    model_artifact = {
        "model": calibrated_model,
        "feature_columns": ALL_FEATURES,
        "baseline_feature_columns": BASELINE_FEATURES,
        "venue_feature_columns": VENUE_FEATURES,
        "target_column": TARGET_COLUMN,
        "model_name": "Venue-Feature Logistic Regression",
        "calibration_method": "sigmoid",
        "training_start_date": (
            train_df[DATE_COLUMN].min().strftime("%Y-%m-%d")
        ),
        "training_end_date": TRAIN_END_DATE.strftime("%Y-%m-%d"),
        "calibration_start_date": (
            calibration_df[DATE_COLUMN].min().strftime("%Y-%m-%d")
        ),
        "calibration_end_date": (
            calibration_df[DATE_COLUMN].max().strftime("%Y-%m-%d")
        ),
        "training_matches": int(len(train_df)),
        "calibration_matches": int(len(calibration_df)),
        "training_data_source": str(INPUT_FILE),
        "random_state": RANDOM_STATE,
        "notes": (
            "Final tournament prediction artifact. "
            "The model was fitted on historical matches through "
            f"{TRAIN_END_DATE.date()} and sigmoid-calibrated on "
            "later matches through {CALIBRATION_END_DATE.date()}. "
            "The historical evaluation test period was not used "
            "as a separate final holdout in this production fit."
        ),
    }

    joblib.dump(
        model_artifact,
        MODEL_FILE,
    )

    print("\nModel saved successfully")
    print("-" * 70)
    print(f"Artifact:\n{MODEL_FILE}")

    # --------------------------------------------------------
    # Reload and verify saved artifact
    # --------------------------------------------------------

    print("\nReloading saved model for verification...")

    saved_artifact = joblib.load(MODEL_FILE)

    saved_model = saved_artifact["model"]
    saved_features = saved_artifact["feature_columns"]

    sample = calibration_df[
        saved_features
    ].head(5)

    sample_probabilities = saved_model.predict_proba(
        sample
    )[:, 1]

    if not np.all(np.isfinite(sample_probabilities)):
        raise ValueError(
            "Saved model produced non-finite probabilities."
        )

    if not np.all(
        (sample_probabilities >= 0)
        & (sample_probabilities <= 1)
    ):
        raise ValueError(
            "Saved model produced probabilities outside [0, 1]."
        )

    print("Saved artifact loaded successfully.")
    print(f"Saved model type: {type(saved_model).__name__}")
    print(f"Saved feature count: {len(saved_features)}")
    print(
        "Sample predicted probabilities:",
        np.round(sample_probabilities, 4).tolist(),
    )

    print("\nTraining completed.")
    print(
        "This artifact is ready for the next step: "
        "building 2027 match features from each team's "
        "pre-tournament history and the scheduled venue."
    )


if __name__ == "__main__":
    main()

