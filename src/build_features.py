
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import sys

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "odi_matches_master.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "odi_features.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_validation.txt"

# Elo configuration
INITIAL_ELO = 1500.0
ELO_K = 20.0

# Recent form configuration
FORM_WINDOW = 5


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected Elo score for team A against team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def safe_team_name(value) -> str | None:
    """Normalize a team name while preserving missing values."""
    if pd.isna(value):
        return None

    name = str(value).strip()
    return name if name else None


def get_team_state(
    team: str,
    ratings: dict,
    form: dict,
    matches_played: dict,
) -> tuple[float, float, int]:
    """Return a team's pre-match rating, recent form, and match count."""
    rating = ratings.get(team, INITIAL_ELO)
    recent_results = list(form.get(team, []))
    recent_form = (
        sum(recent_results) / len(recent_results)
        if recent_results
        else 0.5
    )
    played = matches_played.get(team, 0)

    return rating, recent_form, played


# ---------------------------------------------------------
# Main feature-building pipeline
# ---------------------------------------------------------

def main() -> None:
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found:\n{INPUT_PATH}")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "match_id",
        "date",
        "team_1",
        "team_2",
        "winner",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"ERROR: Missing required columns: {sorted(missing_columns)}")
        sys.exit(1)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team_1"] = df["team_1"].apply(safe_team_name)
    df["team_2"] = df["team_2"].apply(safe_team_name)
    df["winner"] = df["winner"].apply(safe_team_name)

    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        print(f"ERROR: Found {invalid_dates} rows with invalid dates.")
        sys.exit(1)

    invalid_teams = int(
        (df["team_1"].isna() | df["team_2"].isna()).sum()
    )
    if invalid_teams:
        print(f"ERROR: Found {invalid_teams} rows with missing team names.")
        sys.exit(1)

    if df["match_id"].duplicated().any():
        duplicate_count = int(df["match_id"].duplicated().sum())
        print(f"ERROR: Found {duplicate_count} duplicate match IDs.")
        sys.exit(1)

    # Stable ordering makes output reproducible.
    df = df.sort_values(
        ["date", "match_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Team state is updated only after an entire date has been processed.
    ratings: dict[str, float] = {}
    form: dict[str, deque] = defaultdict(
        lambda: deque(maxlen=FORM_WINDOW)
    )
    matches_played: dict[str, int] = defaultdict(int)

    feature_rows = []
    dates_with_multiple_matches = 0
    same_day_feature_checks = 0

    # Group matches by date to prevent same-day information leakage.
    for match_date, day_matches in df.groupby("date", sort=True):
        day_matches = day_matches.sort_values(
            "match_id",
            kind="mergesort",
        )

        if len(day_matches) > 1:
            dates_with_multiple_matches += 1

        # Snapshot all team states at the start of the date.
        start_of_day_ratings = ratings.copy()
        start_of_day_form = {
            team: list(results)
            for team, results in form.items()
        }
        start_of_day_matches_played = matches_played.copy()

        # Build every feature row using only start-of-day information.
        for _, row in day_matches.iterrows():
            team_1 = row["team_1"]
            team_2 = row["team_2"]
            winner = row["winner"]

            rating_1, form_1, played_1 = get_team_state(
                team_1,
                start_of_day_ratings,
                start_of_day_form,
                start_of_day_matches_played,
            )

            rating_2, form_2, played_2 = get_team_state(
                team_2,
                start_of_day_ratings,
                start_of_day_form,
                start_of_day_matches_played,
            )

            target = None
            if winner == team_1:
                target = 1
            elif winner == team_2:
                target = 0

            feature_rows.append(
                {
                    "match_id": row["match_id"],
                    "date": match_date,
                    "team_1": team_1,
                    "team_2": team_2,
                    "winner": winner,
                    "target": target,
                    "team_1_elo": rating_1,
                    "team_2_elo": rating_2,
                    "elo_difference": rating_1 - rating_2,
                    "team_1_recent_form": form_1,
                    "team_2_recent_form": form_2,
                    "recent_form_difference": form_1 - form_2,
                    "team_1_matches_played": played_1,
                    "team_2_matches_played": played_2,
                }
            )

        # -------------------------------------------------
        # Apply all results after feature creation for date
        # -------------------------------------------------

        # Calculate Elo changes from the same start-of-day
        # ratings, then aggregate changes by team.
        daily_elo_deltas = defaultdict(float)
        daily_form_results = defaultdict(list)
        daily_match_counts = defaultdict(int)

        for _, row in day_matches.iterrows():
            team_1 = row["team_1"]
            team_2 = row["team_2"]
            winner = row["winner"]

            # No-result or invalid winner: no rating/form update.
            if winner not in {team_1, team_2}:
                continue

            rating_1 = start_of_day_ratings.get(team_1, INITIAL_ELO)
            rating_2 = start_of_day_ratings.get(team_2, INITIAL_ELO)

            expected_1 = expected_score(rating_1, rating_2)

            actual_1 = 1.0 if winner == team_1 else 0.0
            actual_2 = 1.0 - actual_1

            delta_1 = ELO_K * (actual_1 - expected_1)
            delta_2 = ELO_K * (actual_2 - (1.0 - expected_1))

            daily_elo_deltas[team_1] += delta_1
            daily_elo_deltas[team_2] += delta_2

            daily_form_results[team_1].append(actual_1)
            daily_form_results[team_2].append(actual_2)

            daily_match_counts[team_1] += 1
            daily_match_counts[team_2] += 1

        # Apply aggregated Elo changes.
        for team, delta in daily_elo_deltas.items():
            old_rating = start_of_day_ratings.get(team, INITIAL_ELO)
            ratings[team] = old_rating + delta

        # Add form observations in deterministic match order.
        for team, results in daily_form_results.items():
            for result in results:
                form[team].append(result)

        # Update match counts.
        for team, count in daily_match_counts.items():
            matches_played[team] = (
                start_of_day_matches_played.get(team, 0) + count
            )

    features = pd.DataFrame(feature_rows)

    # Preserve a consistent column order.
    output_columns = [
        "match_id",
        "date",
        "team_1",
        "team_2",
        "winner",
        "target",
        "team_1_elo",
        "team_2_elo",
        "elo_difference",
        "team_1_recent_form",
        "team_2_recent_form",
        "recent_form_difference",
        "team_1_matches_played",
        "team_2_matches_played",
    ]

    features = features[output_columns]
    features.to_csv(OUTPUT_PATH, index=False)

    # -----------------------------------------------------
    # Validation report
    # -----------------------------------------------------

    target_counts = features["target"].value_counts(dropna=False)
    decided_matches = int(features["target"].notna().sum())
    excluded_matches = int(features["target"].isna().sum())

    report_lines = [
        "ODI FEATURE BUILD VALIDATION",
        "=" * 60,
        f"Input rows: {len(df):,}",
        f"Output rows: {len(features):,}",
        f"Date range: {features['date'].min().date()} to "
        f"{features['date'].max().date()}",
        f"Unique match IDs: {features['match_id'].nunique():,}",
        f"Duplicate match IDs: {int(features['match_id'].duplicated().sum())}",
        f"Dates with multiple matches: {dates_with_multiple_matches:,}",
        "",
        "Target validation:",
        f"Decided matches: {decided_matches:,}",
        f"Excluded no-result/tie/invalid-winner matches: {excluded_matches:,}",
        f"Target 1 (team_1 wins): {int(target_counts.get(1.0, 0)):,}",
        f"Target 0 (team_2 wins): {int(target_counts.get(0.0, 0)):,}",
        "",
        "Feature validation:",
        f"Missing Elo difference: {int(features['elo_difference'].isna().sum())}",
        f"Missing recent form difference: "
        f"{int(features['recent_form_difference'].isna().sum())}",
        f"Missing match count features: "
        f"{int(features[['team_1_matches_played', 'team_2_matches_played']].isna().sum().sum())}",
        "",
        "Leakage protection:",
        "Features for all matches on a date were created from the same "
        "start-of-day team-state snapshot.",
        "Match results were applied only after feature creation for that date.",
        "No-result, tie, or invalid-winner matches do not update Elo, form, "
        "or match counts.",
        "",
        f"Saved features: {OUTPUT_PATH}",
    ]

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
