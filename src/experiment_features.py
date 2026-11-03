
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT_DIR / "data" / "processed" / "odi_features.csv"
OUTPUT_DIR = ROOT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "feature_experiment_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "feature_experiment_summary.txt"

TARGET_COLUMN = "target"

# Keep the same chronological split proportions as the baseline.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

RANDOM_STATE = 42


# ---------------------------------------------------------
# Feature sets
# ---------------------------------------------------------

FEATURE_SETS = {
    "Current baseline: Elo + recent form + experience": [
        "team_1_elo",
        "team_2_elo",
        "elo_difference",
        "team_1_recent_form",
        "team_2_recent_form",
        "recent_form_difference",
        "team_1_matches_played",
        "team_2_matches_played",
    ],
    "Elo only": [
        "team_1_elo",
        "team_2_elo",
        "elo_difference",
    ],
    "Recent form only": [
        "team_1_recent_form",
        "team_2_recent_form",
        "recent_form_difference",
    ],
    "Elo + recent form": [
        "team_1_elo",
        "team_2_elo",
        "elo_difference",
        "team_1_recent_form",
        "team_2_recent_form",
        "recent_form_difference",
    ],
}


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

    required_columns = ["date", TARGET_COLUMN]
    for feature_list in FEATURE_SETS.values():
        required_columns.extend(feature_list)

    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(
            "The feature dataset is missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # Keep only decided matches with valid binary targets.
    df = df[df[TARGET_COLUMN].isin([0, 1])].copy()

    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No decided matches with valid targets were found.")

    if df["date"].isna().any():
        raise ValueError("Some matches have missing dates.")

    if df["match_id"].duplicated().any():
        raise ValueError("Duplicate match_id values found in feature dataset.")

    return df


# ---------------------------------------------------------
# Chronological split by unique date
# ---------------------------------------------------------

def chronological_split(df):
    unique_dates = sorted(df["date"].drop_duplicates().tolist())

    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates for a reliable split.")

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

    # Confirm that dates do not cross split boundaries.
    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)

    return train_df, validation_df, test_df


# ---------------------------------------------------------
# Model and evaluation
# ---------------------------------------------------------

def create_model():
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
# Run experiments
# ---------------------------------------------------------

def run_experiments():
    df = load_data()
    train_df, validation_df, test_df = chronological_split(df)

    print("\nDATASET")
    print("-" * 60)
    print(f"Decided matches: {len(df):,}")
    print(f"Unique dates: {df['date'].nunique():,}")
    print(
        f"Train: {len(train_df):,} matches | "
        f"{train_df['date'].min().date()} to "
        f"{train_df['date'].max().date()}"
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

    X_train_target = train_df[TARGET_COLUMN].astype(int)
    y_validation = validation_df[TARGET_COLUMN].astype(int)

    results = []

    for experiment_name, feature_columns in FEATURE_SETS.items():
        print(f"\nRunning: {experiment_name}")

        X_train = train_df[feature_columns]
        X_validation = validation_df[feature_columns]

        model = create_model()
        model.fit(X_train, X_train_target)

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        result = {
            "experiment": experiment_name,
            "feature_count": len(feature_columns),
            "features": ", ".join(feature_columns),
            **metrics,
        }

        results.append(result)

        print(f"Features: {len(feature_columns)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Log loss: {metrics['log_loss']:.4f}")
        print(f"Brier score: {metrics['brier_score']:.4f}")

    results_df = pd.DataFrame(results)

    # Sort primarily by log loss, then Brier score.
    # Lower is better for both probability-quality metrics.
    results_df = results_df.sort_values(
        by=["log_loss", "brier_score"],
        ascending=[True, True],
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_PATH, index=False)

    best_result = results_df.iloc[0]

    summary_lines = [
        "ODI FEATURE EXPERIMENT SUMMARY",
        "=" * 60,
        "",
        "Selection was based on chronological validation data only.",
        "The test set was not used in these experiments.",
        "",
        "Results ranked by log loss, then Brier score:",
        "",
        results_df[
            [
                "experiment",
                "feature_count",
                "accuracy",
                "auc",
                "log_loss",
                "brier_score",
            ]
        ].to_string(index=False),
        "",
        f"Best validation experiment: {best_result['experiment']}",
        f"Best validation log loss: {best_result['log_loss']:.4f}",
        f"Best validation Brier score: {best_result['brier_score']:.4f}",
        "",
        "Important: This is a feature-selection result, not final test performance.",
    ]

    summary_text = "\n".join(summary_lines)
    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(
        results_df[
            [
                "experiment",
                "accuracy",
                "auc",
                "log_loss",
                "brier_score",
            ]
        ].to_string(index=False)
    )

    print(f"\nBest validation experiment: {best_result['experiment']}")
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    run_experiments()
