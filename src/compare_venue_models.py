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

PREDICTIONS_FILE = REPORTS_DIR / "venue_test_predictions.csv"
COMPARISON_CSV_FILE = REPORTS_DIR / "venue_model_comparison.csv"
COMPARISON_TXT_FILE = REPORTS_DIR / "venue_model_comparison.txt"
BOOTSTRAP_CSV_FILE = REPORTS_DIR / "venue_bootstrap_results.csv"
BOOTSTRAP_TXT_FILE = REPORTS_DIR / "venue_bootstrap_results.txt"

TARGET_COLUMN = "target"
DATE_COLUMN = "date"
MATCH_ID_COLUMN = "match_id"

TRAIN_END_DATE = pd.Timestamp("2019-06-06")
VALIDATION_END_DATE = pd.Timestamp("2023-06-25")

RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 2_000
CONFIDENCE_LEVEL = 0.95

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
# Utility functions
# ============================================================

def find_column(df: pd.DataFrame, candidates: list[str], description: str) -> str:
    """Return the first candidate column found in the dataframe."""
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find a column for {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def make_logistic_regression() -> Pipeline:
    """Create the same imputation/scaling/logistic-regression pipeline."""
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


def elo_probability(elo_difference: pd.Series | np.ndarray) -> np.ndarray:
    """Convert an Elo difference into a team-1 win probability."""
    elo_difference = np.asarray(elo_difference, dtype=float)

    probabilities = 1.0 / (
        1.0 + np.power(10.0, -elo_difference / 400.0)
    )

    return np.clip(probabilities, 1e-6, 1.0 - 1e-6)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate classification and probability-quality metrics."""
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, probabilities),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probabilities)
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.6f}"


# ============================================================
# Date-clustered paired bootstrap
# ============================================================

def clustered_paired_bootstrap(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    baseline_probabilities: np.ndarray,
    venue_probabilities: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = RANDOM_STATE,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """
    Paired bootstrap that resamples test dates, not individual matches.

    Every match played on a sampled date is included together. If a date is
    sampled more than once, its matches are included more than once.

    Differences are calculated as:

        venue model metric - baseline model metric

    Positive differences are better for accuracy/AUC.
    Negative differences are better for log loss/Brier score.
    """
    rng = np.random.default_rng(random_state)

    dates = pd.to_datetime(test_df[DATE_COLUMN]).dt.normalize()
    unique_dates = np.array(sorted(dates.unique()))

    date_to_indices = {
        date: np.flatnonzero(dates.to_numpy() == date)
        for date in unique_dates
    }

    metric_names = [
        "accuracy",
        "roc_auc",
        "log_loss",
        "brier_score",
    ]

    observed_baseline = calculate_metrics(
        y_true,
        baseline_probabilities,
    )
    observed_venue = calculate_metrics(
        y_true,
        venue_probabilities,
    )

    observed_differences = {
        metric: observed_venue[metric] - observed_baseline[metric]
        for metric in metric_names
    }

    bootstrap_differences = {
        metric: []
        for metric in metric_names
    }

    valid_replicates = {
        metric: 0
        for metric in metric_names
    }

    for _ in range(n_resamples):
        sampled_dates = rng.choice(
            unique_dates,
            size=len(unique_dates),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [date_to_indices[date] for date in sampled_dates]
        )

        sampled_y = y_true[sampled_indices]
        sampled_baseline = baseline_probabilities[sampled_indices]
        sampled_venue = venue_probabilities[sampled_indices]

        baseline_metrics = calculate_metrics(
            sampled_y,
            sampled_baseline,
        )
        venue_metrics = calculate_metrics(
            sampled_y,
            sampled_venue,
        )

        for metric in metric_names:
            baseline_value = baseline_metrics[metric]
            venue_value = venue_metrics[metric]

            if pd.isna(baseline_value) or pd.isna(venue_value):
                continue

            difference = venue_value - baseline_value
            bootstrap_differences[metric].append(difference)
            valid_replicates[metric] += 1

    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    results = []

    for metric in metric_names:
        values = np.asarray(
            bootstrap_differences[metric],
            dtype=float,
        )

        if len(values) == 0:
            ci_lower = np.nan
            ci_upper = np.nan
        else:
            ci_lower, ci_upper = np.percentile(
                values,
                [lower_percentile, upper_percentile],
            )

        crosses_zero = (
            pd.isna(ci_lower)
            or pd.isna(ci_upper)
            or (ci_lower <= 0 <= ci_upper)
        )

        results.append(
            {
                "metric": metric,
                "baseline_test": observed_baseline[metric],
                "venue_test": observed_venue[metric],
                "venue_minus_baseline": observed_differences[metric],
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence_level": confidence_level,
                "valid_bootstrap_replicates": valid_replicates[metric],
                "total_bootstrap_replicates": n_resamples,
                "ci_crosses_zero": crosses_zero,
            }
        )

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================

def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn",
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    print("=" * 70)
    print("VENUE MODEL COMPARISON")
    print("=" * 70)
    print(f"Input: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Resolve key columns. These candidates make the script tolerant of
    # minor naming differences in the source dataset.
    date_column = find_column(
        df,
        [DATE_COLUMN, "start_date", "match_date"],
        "match date",
    )
    match_id_column = find_column(
        df,
        [MATCH_ID_COLUMN, "id", "matchid"],
        "match ID",
    )

    # The remainder of the script uses the configured standard names.
    if date_column != DATE_COLUMN:
        df = df.rename(columns={date_column: DATE_COLUMN})

    if match_id_column != MATCH_ID_COLUMN:
        df = df.rename(columns={match_id_column: MATCH_ID_COLUMN})

    required_columns = (
        [MATCH_ID_COLUMN, DATE_COLUMN, TARGET_COLUMN]
        + ALL_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from the input file:\n"
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

    df = df[df[TARGET_COLUMN].isin([0, 1])].copy()

    # Only working 2027-team matches have venue features.
    # This is the same evaluated subset used in the previous comparison.
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

    # Fixed chronological split, unchanged from the previous script.
    train_mask = df[DATE_COLUMN] <= TRAIN_END_DATE

    validation_mask = (
        (df[DATE_COLUMN] > TRAIN_END_DATE)
        & (df[DATE_COLUMN] <= VALIDATION_END_DATE)
    )

    test_mask = df[DATE_COLUMN] > VALIDATION_END_DATE

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            "One or more chronological splits are empty. "
            "Check the dates in the input dataset."
        )

    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()
    y_validation = validation_df[TARGET_COLUMN].astype(int).to_numpy()
    y_test = test_df[TARGET_COLUMN].astype(int).to_numpy()

    print("\nData split")
    print("-" * 70)

    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        print(
            f"{split_name:12} {len(split_df):4} matches | "
            f"{split_df[DATE_COLUMN].min().date()} to "
            f"{split_df[DATE_COLUMN].max().date()}"
        )

    print(f"\nTotal evaluated matches: {len(df)}")

    # --------------------------------------------------------
    # Baseline logistic regression
    # --------------------------------------------------------

    print("\nTraining baseline logistic regression...")

    baseline_model = make_logistic_regression()

    baseline_model.fit(
        train_df[BASELINE_FEATURES],
        y_train,
    )

    baseline_calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(baseline_model),
        method="sigmoid",
    )

    baseline_calibrated_model.fit(
        validation_df[BASELINE_FEATURES],
        y_validation,
    )

    baseline_test_probabilities = (
        baseline_calibrated_model.predict_proba(
            test_df[BASELINE_FEATURES]
        )[:, 1]
    )

    # --------------------------------------------------------
    # Logistic regression with venue features
    # --------------------------------------------------------

    print("Training venue-feature logistic regression...")

    venue_model = make_logistic_regression()

    venue_model.fit(
        train_df[ALL_FEATURES],
        y_train,
    )

    venue_calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(venue_model),
        method="sigmoid",
    )

    venue_calibrated_model.fit(
        validation_df[ALL_FEATURES],
        y_validation,
    )

    venue_test_probabilities = (
        venue_calibrated_model.predict_proba(
            test_df[ALL_FEATURES]
        )[:, 1]
    )

    # --------------------------------------------------------
    # Raw Elo K20 benchmark
    # --------------------------------------------------------

    if "elo_difference" not in test_df.columns:
        raise ValueError(
            "The required elo_difference column is missing."
        )

    raw_elo_test_probabilities = elo_probability(
        test_df["elo_difference"].to_numpy()
    )

    # --------------------------------------------------------
    # Match-level test predictions
    # --------------------------------------------------------

    predictions = pd.DataFrame(
        {
            MATCH_ID_COLUMN: test_df[MATCH_ID_COLUMN].to_numpy(),
            DATE_COLUMN: test_df[DATE_COLUMN].dt.strftime(
                "%Y-%m-%d"
            ).to_numpy(),
            TARGET_COLUMN: y_test,
            "baseline_probability": baseline_test_probabilities,
            "venue_probability": venue_test_probabilities,
            "raw_elo_k20_probability": raw_elo_test_probabilities,
        }
    )

    # Add team names when available.
    team_1_column = next(
        (
            column
            for column in [
                "team_1",
                "team1",
                "team_1_name",
                "team1_name",
            ]
            if column in test_df.columns
        ),
        None,
    )

    team_2_column = next(
        (
            column
            for column in [
                "team_2",
                "team2",
                "team_2_name",
                "team2_name",
            ]
            if column in test_df.columns
        ),
        None,
    )

    if team_1_column:
        predictions.insert(
            2,
            "team_1",
            test_df[team_1_column].to_numpy(),
        )

    if team_2_column:
        insert_position = 3 if team_1_column else 2
        predictions.insert(
            insert_position,
            "team_2",
            test_df[team_2_column].to_numpy(),
        )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(f"\nSaved match-level predictions:\n{PREDICTIONS_FILE}")

    # --------------------------------------------------------
    # Aggregate test metrics
    # --------------------------------------------------------

    model_probabilities = {
        "Baseline LR (calibrated)": baseline_test_probabilities,
        "Baseline + Venue LR (calibrated)": venue_test_probabilities,
        "Raw Elo K20": raw_elo_test_probabilities,
    }

    comparison_rows = []

    for model_name, probabilities in model_probabilities.items():
        metrics = calculate_metrics(y_test, probabilities)

        comparison_rows.append(
            {
                "model": model_name,
                "test_matches": len(y_test),
                **metrics,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        COMPARISON_CSV_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Date-clustered paired bootstrap
    # --------------------------------------------------------

    print(
        f"\nRunning {BOOTSTRAP_RESAMPLES:,} paired "
        "date-cluster bootstrap resamples..."
    )

    bootstrap_df = clustered_paired_bootstrap(
        test_df=test_df,
        y_true=y_test,
        baseline_probabilities=baseline_test_probabilities,
        venue_probabilities=venue_test_probabilities,
        n_resamples=BOOTSTRAP_RESAMPLES,
        random_state=RANDOM_STATE,
        confidence_level=CONFIDENCE_LEVEL,
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_CSV_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Text reports
    # --------------------------------------------------------

    comparison_lines = [
        "VENUE MODEL COMPARISON",
        "=" * 70,
        f"Input file: {INPUT_FILE}",
        f"Evaluated working-team matches: {len(df)}",
        "",
        "Chronological split:",
        (
            f"Train:      {len(train_df)} | "
            f"{train_df[DATE_COLUMN].min().date()} to "
            f"{train_df[DATE_COLUMN].max().date()}"
        ),
        (
            f"Validation: {len(validation_df)} | "
            f"{validation_df[DATE_COLUMN].min().date()} to "
            f"{validation_df[DATE_COLUMN].max().date()}"
        ),
        (
            f"Test:       {len(test_df)} | "
            f"{test_df[DATE_COLUMN].min().date()} to "
            f"{test_df[DATE_COLUMN].max().date()}"
        ),
        "",
        "Test metrics:",
        "-" * 70,
    ]

    for _, row in comparison_df.iterrows():
        comparison_lines.append(str(row["model"]))
        comparison_lines.append(
            f"  Accuracy:   {row['accuracy']:.6f}"
        )
        comparison_lines.append(
            f"  ROC AUC:    {row['roc_auc']:.6f}"
        )
        comparison_lines.append(
            f"  Log loss:   {row['log_loss']:.6f}"
        )
        comparison_lines.append(
            f"  Brier:      {row['brier_score']:.6f}"
        )
        comparison_lines.append("")

    comparison_lines.extend(
        [
            "Methodology note:",
            (
                "The venue model is evaluated on the same test matches "
                "as the baseline. The test period has already been "
                "examined in earlier model comparisons, so this result "
                "is exploratory rather than a pristine final holdout."
            ),
        ]
    )

    COMPARISON_TXT_FILE.write_text(
        "\n".join(comparison_lines),
        encoding="utf-8",
    )

    bootstrap_lines = [
        "DATE-CLUSTERED PAIRED BOOTSTRAP",
        "=" * 70,
        f"Test matches: {len(test_df)}",
        f"Unique test dates: {test_df[DATE_COLUMN].nunique()}",
        f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}",
        f"Random seed: {RANDOM_STATE}",
        f"Confidence interval: {CONFIDENCE_LEVEL:.0%}",
        "",
        (
            "Difference is venue model minus baseline model. "
            "Positive is better for accuracy/AUC; negative is "
            "better for log loss/Brier."
        ),
        "",
    ]

    for _, row in bootstrap_df.iterrows():
        crosses_zero = bool(row["ci_crosses_zero"])

        if crosses_zero:
            interpretation = "CI crosses zero; difference is uncertain"
        else:
            interpretation = "CI excludes zero"

        bootstrap_lines.extend(
            [
                str(row["metric"]),
                (
                    f"  Baseline: {row['baseline_test']:.6f} | "
                    f"Venue: {row['venue_test']:.6f}"
                ),
                (
                    f"  Difference: {row['venue_minus_baseline']:.6f}"
                ),
                (
                    f"  95% CI: [{row['ci_lower']:.6f}, "
                    f"{row['ci_upper']:.6f}]"
                ),
                (
                    f"  Valid replicates: "
                    f"{row['valid_bootstrap_replicates']}/"
                    f"{row['total_bootstrap_replicates']}"
                ),
                f"  Interpretation: {interpretation}",
                "",
            ]
        )

    bootstrap_lines.extend(
        [
            "Important limitation:",
            (
                "This bootstrap estimates uncertainty conditional on "
                "the current test period and fitted models. It does "
                "not undo prior inspection of the test set or establish "
                "that the improvement will generalize to future matches."
            ),
        ]
    )

    BOOTSTRAP_TXT_FILE.write_text(
        "\n".join(bootstrap_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\nTest model comparison")
    print("-" * 70)
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nPaired date-cluster bootstrap")
    print("-" * 70)
    print(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved reports:")
    print(f"  {PREDICTIONS_FILE}")
    print(f"  {COMPARISON_CSV_FILE}")
    print(f"  {COMPARISON_TXT_FILE}")
    print(f"  {BOOTSTRAP_CSV_FILE}")
    print(f"  {BOOTSTRAP_TXT_FILE}")

    print("\nDone.")


if __name__ == "__main__":
    main()