from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)


# ============================================================
# Paths and configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "odi_matches_master.csv"
)

FEATURES_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "odi_features.csv"
)

REPORTS_DIR = ROOT_DIR / "reports"

RESULTS_PATH = REPORTS_DIR / "elo_k_experiment_results.csv"
PREDICTIONS_PATH = REPORTS_DIR / "elo_k_experiment_predictions.csv"
SUMMARY_PATH = REPORTS_DIR / "elo_k_experiment_summary.txt"

INITIAL_ELO = 1500
K_VALUES = [10, 20, 30, 40]
EPSILON = 1e-6

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


# ============================================================
# Data loading and validation
# ============================================================

def load_and_align_data():
    """
    Load the master match data and feature data.

    The feature dataset is the source of truth for:
    - Which decided matches are evaluated
    - Match dates
    - Chronological split boundaries

    The master dataset supplies:
    - Team names
    - Winner
    """

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found:\n{MASTER_PATH}"
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{FEATURES_PATH}\n"
            "Run src/build_features.py first."
        )

    master = pd.read_csv(MASTER_PATH)
    features = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["date"],
    )

    master_required = {
        "match_id",
        "date",
        "team_1",
        "team_2",
        "winner",
    }

    features_required = {
        "match_id",
        "date",
        "target",
    }

    missing_master = master_required - set(master.columns)
    missing_features = features_required - set(features.columns)

    if missing_master:
        raise ValueError(
            "Master dataset is missing columns: "
            f"{sorted(missing_master)}"
        )

    if missing_features:
        raise ValueError(
            "Feature dataset is missing columns: "
            f"{sorted(missing_features)}"
        )

    if master["match_id"].duplicated().any():
        raise ValueError(
            "Duplicate match_id values found in master dataset."
        )

    if features["match_id"].duplicated().any():
        raise ValueError(
            "Duplicate match_id values found in feature dataset."
        )

    # Keep only binary decided matches from the feature dataset.
    features = features.loc[
        features["target"].isin([0, 1])
    ].copy()

    features["date"] = pd.to_datetime(
        features["date"],
        errors="coerce",
    )

    if features["date"].isna().any():
        raise ValueError(
            "Feature dataset contains missing or invalid dates."
        )

    # Match rows by ID, rather than relying on row order.
    aligned = features[
        ["match_id", "date", "target"]
    ].merge(
        master[
            [
                "match_id",
                "date",
                "team_1",
                "team_2",
                "winner",
            ]
        ],
        on="match_id",
        how="left",
        suffixes=("_features", "_master"),
        validate="one_to_one",
        indicator=True,
    )

    unmatched = aligned.loc[
        aligned["_merge"] != "both"
    ]

    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} feature matches could not be "
            "matched to the master dataset by match_id."
        )

    # Confirm the match dates agree between both datasets.
    date_mismatch = (
        aligned["date_features"]
        != aligned["date_master"]
    )

    if date_mismatch.any():
        mismatch_ids = aligned.loc[
            date_mismatch,
            "match_id",
        ].head(10).tolist()

        raise ValueError(
            "Match dates differ between master and feature data. "
            f"Example match IDs: {mismatch_ids}"
        )

    aligned = aligned.drop(
        columns=["date_master", "_merge"]
    )

    aligned = aligned.rename(
        columns={"date_features": "date"}
    )

    # Confirm the target agrees with the master winner.
    expected_target = (
        aligned["winner"] == aligned["team_1"]
    ).astype(int)

    invalid_winners = ~(
        (aligned["winner"] == aligned["team_1"])
        | (aligned["winner"] == aligned["team_2"])
    )

    if invalid_winners.any():
        raise ValueError(
            "Some feature matches have winners that do not "
            "match either participating team."
        )

    target_mismatch = (
        aligned["target"].astype(int)
        != expected_target
    )

    if target_mismatch.any():
        mismatch_ids = aligned.loc[
            target_mismatch,
            "match_id",
        ].head(10).tolist()

        raise ValueError(
            "Targets differ between feature and master data. "
            f"Example match IDs: {mismatch_ids}"
        )

    aligned = aligned.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    return aligned


# ============================================================
# Exact chronological split from final_evaluation.py
# ============================================================

def create_chronological_splits(df):
    """
    Reproduce final_evaluation.py's exact split logic.

    Important: validation date count is calculated separately
    from train date count, matching the original script.
    """

    unique_dates = sorted(
        df["date"].drop_duplicates().tolist()
    )

    train_date_count = int(
        len(unique_dates) * TRAIN_FRACTION
    )

    validation_date_count = int(
        len(unique_dates) * VALIDATION_FRACTION
    )

    train_dates = set(
        unique_dates[:train_date_count]
    )

    validation_start = train_date_count

    validation_end = (
        train_date_count + validation_date_count
    )

    validation_dates = set(
        unique_dates[
            validation_start:validation_end
        ]
    )

    test_dates = set(
        unique_dates[validation_end:]
    )

    # Verify no date crosses split boundaries.
    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)

    split_map = {}

    for date in train_dates:
        split_map[date] = "train"

    for date in validation_dates:
        split_map[date] = "validation"

    for date in test_dates:
        split_map[date] = "test"

    df = df.copy()
    df["split"] = df["date"].map(split_map)

    if df["split"].isna().any():
        raise ValueError(
            "Some matches were not assigned to a split."
        )

    return df


# ============================================================
# Elo functions
# ============================================================

def expected_score(rating_a, rating_b):
    """Expected win probability for Team A."""

    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


def generate_elo_predictions(matches, k_value):
    """
    Rebuild Elo ratings for a given K value.

    Every match on the same date uses ratings from the start
    of that date. Rating changes are applied after all matches
    from that date have been processed.

    This prevents same-day results from leaking into predictions
    for other matches on the same date.
    """

    ratings = {}
    prediction_rows = []

    for match_date, daily_matches in matches.groupby(
        "date",
        sort=True,
    ):

        start_of_day_ratings = ratings.copy()
        daily_rating_changes = {}

        for _, match in daily_matches.iterrows():

            team_1 = match["team_1"]
            team_2 = match["team_2"]
            winner = match["winner"]

            rating_1 = start_of_day_ratings.get(
                team_1,
                INITIAL_ELO,
            )

            rating_2 = start_of_day_ratings.get(
                team_2,
                INITIAL_ELO,
            )

            probability_team_1 = expected_score(
                rating_1,
                rating_2,
            )

            actual_team_1 = int(
                winner == team_1
            )

            prediction_rows.append(
                {
                    "match_id": match["match_id"],
                    "date": match_date,
                    "team_1": team_1,
                    "team_2": team_2,
                    "winner": winner,
                    "target": actual_team_1,
                    "k_value": k_value,
                    "team_1_elo_pre_match": rating_1,
                    "team_2_elo_pre_match": rating_2,
                    "elo_difference": rating_1 - rating_2,
                    "team_1_win_probability": probability_team_1,
                    "split": match["split"],
                }
            )

            rating_change = k_value * (
                actual_team_1 - probability_team_1
            )

            daily_rating_changes[team_1] = (
                daily_rating_changes.get(team_1, 0)
                + rating_change
            )

            daily_rating_changes[team_2] = (
                daily_rating_changes.get(team_2, 0)
                - rating_change
            )

        # Apply Elo updates after all matches for the date.
        for team, change in daily_rating_changes.items():

            previous_rating = start_of_day_ratings.get(
                team,
                INITIAL_ELO,
            )

            ratings[team] = previous_rating + change

    return pd.DataFrame(prediction_rows)


# ============================================================
# Evaluation metrics
# ============================================================

def calculate_metrics(actual, probabilities):

    actual = np.asarray(
        actual,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1 - EPSILON,
    )

    predicted = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "matches": len(actual),
        "accuracy": accuracy_score(
            actual,
            predicted,
        ),
        "auc": roc_auc_score(
            actual,
            probabilities,
        ),
        "log_loss": log_loss(
            actual,
            probabilities,
            labels=[0, 1],
        ),
        "brier_score": brier_score_loss(
            actual,
            probabilities,
        ),
        "actual_team_1_win_rate": actual.mean(),
        "mean_predicted_team_1_win_probability": (
            probabilities.mean()
        ),
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("ODI ELO K-VALUE EXPERIMENT")
    print("Using exact split logic from final_evaluation.py")
    print("=" * 75)

    matches = load_and_align_data()
    matches = create_chronological_splits(matches)

    print("\nDATASET CHECK")
    print("-" * 75)
    print(f"Decided matches: {len(matches):,}")
    print(f"Unique dates:    {matches['date'].nunique():,}")

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        split_df = matches.loc[
            matches["split"] == split_name
        ]

        print(
            f"{split_name.capitalize():<12}: "
            f"{len(split_df):,} matches | "
            f"{split_df['date'].min().date()} to "
            f"{split_df['date'].max().date()}"
        )

    all_results = []
    all_predictions = []

    for k_value in K_VALUES:

        print(
            f"\nRebuilding Elo ratings with K = {k_value}..."
        )

        predictions = generate_elo_predictions(
            matches=matches,
            k_value=k_value,
        )

        all_predictions.append(predictions)

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            split_df = predictions.loc[
                predictions["split"] == split_name
            ]

            metrics = calculate_metrics(
                actual=split_df["target"],
                probabilities=split_df[
                    "team_1_win_probability"
                ],
            )

            all_results.append(
                {
                    "k_value": k_value,
                    "split": split_name,
                    **metrics,
                }
            )

    results_df = pd.DataFrame(all_results)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Select K using validation only
    # --------------------------------------------------------

    validation_results = results_df.loc[
        results_df["split"] == "validation"
    ].copy()

    validation_results = validation_results.sort_values(
        ["log_loss", "brier_score"],
        ascending=True,
    )

    selected_k = int(
        validation_results.iloc[0]["k_value"]
    )

    selected_validation = validation_results.iloc[0]

    selected_test = results_df.loc[
        (results_df["split"] == "test")
        & (results_df["k_value"] == selected_k)
    ].iloc[0]

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    summary = [
        "ODI ELO K-VALUE EXPERIMENT",
        "=" * 75,
        "",
        "Split logic:",
        "Reused the exact unique-date split logic from final_evaluation.py.",
        "",
        "Selected K:",
        str(selected_k),
        "",
        "Selection criterion:",
        "Lowest validation log loss; Brier score used as secondary criterion.",
        "",
        "Validation results:",
        validation_results.to_string(index=False),
        "",
        "Test results:",
        results_df.loc[
            results_df["split"] == "test"
        ].sort_values("k_value").to_string(index=False),
        "",
        "Selected K validation metrics:",
        f"Accuracy:  {selected_validation['accuracy']:.4f}",
        f"AUC:       {selected_validation['auc']:.4f}",
        f"Log loss:  {selected_validation['log_loss']:.4f}",
        f"Brier:     {selected_validation['brier_score']:.4f}",
        "",
        "Selected K test metrics:",
        f"Accuracy:  {selected_test['accuracy']:.4f}",
        f"AUC:       {selected_test['auc']:.4f}",
        f"Log loss:  {selected_test['log_loss']:.4f}",
        f"Brier:     {selected_test['brier_score']:.4f}",
        "",
        "Important:",
        "K was selected using validation results only.",
        "Test results were not used to select K.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("VALIDATION RESULTS — K SELECTION")
    print("=" * 75)

    print(
        validation_results[
            [
                "k_value",
                "matches",
                "accuracy",
                "auc",
                "log_loss",
                "brier_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 75)
    print("TEST RESULTS — HELD-OUT EVALUATION")
    print("=" * 75)

    print(
        results_df.loc[
            results_df["split"] == "test"
        ].sort_values("k_value")[
            [
                "k_value",
                "matches",
                "accuracy",
                "auc",
                "log_loss",
                "brier_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 75)
    print("SELECTED K")
    print("=" * 75)

    print(f"Selected K: {selected_k}")
    print(
        "Selection criterion: lowest validation log loss, "
        "with Brier score as secondary comparison."
    )

    print("\nSelected K — validation:")
    print(f"Accuracy:  {selected_validation['accuracy']:.4f}")
    print(f"AUC:       {selected_validation['auc']:.4f}")
    print(f"Log loss:  {selected_validation['log_loss']:.4f}")
    print(f"Brier:     {selected_validation['brier_score']:.4f}")

    print("\nSelected K — test:")
    print(f"Accuracy:  {selected_test['accuracy']:.4f}")
    print(f"AUC:       {selected_test['auc']:.4f}")
    print(f"Log loss:  {selected_test['log_loss']:.4f}")
    print(f"Brier:     {selected_test['brier_score']:.4f}")

    print("\nSaved files:")
    print(f"- {RESULTS_PATH}")
    print(f"- {PREDICTIONS_PATH}")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()