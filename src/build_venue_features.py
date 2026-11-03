from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_matches_with_resolved_cities.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_features_with_venue.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "venue_feature_validation.txt"
)

WORLD_CUP_CONFIG_PATH = PROJECT_ROOT / "config"


# ============================================================
# HOME VENUE RULES
# ============================================================

# These are explicit venue-country matches.
# Do not use broad regions such as "United Kingdom" for Ireland.
TEAM_HOME_COUNTRIES = {
    "Afghanistan": {"afghanistan"},
    "Australia": {"australia"},
    "Bangladesh": {"bangladesh"},
    "England": {"england"},
    "India": {"india"},
    "Ireland": {"ireland", "northern ireland"},
    "Netherlands": {"netherlands"},
    "New Zealand": {"new zealand"},
    "Pakistan": {"pakistan"},
    "Scotland": {"scotland"},
    "South Africa": {"south africa"},
    "Sri Lanka": {"sri lanka"},
    "Zimbabwe": {"zimbabwe"},
}


# West Indies is a composite team, so it does not have one
# single home-country value. These are explicit host territories.
WEST_INDIES_HOME_COUNTRIES = {
    "antigua and barbuda",
    "barbados",
    "dominica",
    "grenada",
    "guyana",
    "jamaica",
    "saint kitts and nevis",
    "saint lucia",
    "saint vincent and the grenadines",
    "trinidad and tobago",
}


# Normalization for known country-name variants.
COUNTRY_ALIASES = {
    "st kitts and nevis": "saint kitts and nevis",
    "st. kitts and nevis": "saint kitts and nevis",
    "st lucia": "saint lucia",
    "st. lucia": "saint lucia",
    "st vincent and the grenadines": "saint vincent and the grenadines",
    "st. vincent and the grenadines": "saint vincent and the grenadines",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value) -> str:
    """Normalize a text value for comparisons."""
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().lower().split())


def normalize_country(value) -> str:
    """Normalize country names and known aliases."""
    country = normalize_text(value)

    if country in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[country]

    return country


def load_world_cup_teams() -> set[str]:
    """Load the working 2027 World Cup team list from config."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from config.world_cup_2027_teams import WORLD_CUP_2027_TEAMS

    return set(WORLD_CUP_2027_TEAMS)


def is_home_match(team: str, venue_country: str) -> int:
    """
    Return 1 if the venue country is a recognized home venue
    for the team; otherwise return 0.
    """
    team_name = str(team).strip()
    country = normalize_country(venue_country)

    if not team_name or not country:
        return 0

    if team_name == "West Indies":
        return int(country in WEST_INDIES_HOME_COUNTRIES)

    allowed_countries = TEAM_HOME_COUNTRIES.get(team_name, set())

    return int(country in allowed_countries)


def get_match_target(row) -> float | None:
    """
    Target:
      1 = team_1 won
      0 = team_2 won
      None = no result, tie, or invalid winner
    """
    winner = str(row["winner"]).strip()
    team_1 = str(row["team_1"]).strip()
    team_2 = str(row["team_2"]).strip()

    if winner == team_1:
        return 1

    if winner == team_2:
        return 0

    return None


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data() -> pd.DataFrame:
    print("Loading resolved ODI dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "match_id",
        "date",
        "team_1",
        "team_2",
        "winner",
        "city_resolved",
        "country_resolved",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        bad_rows = int(df["date"].isna().sum())
        raise ValueError(
            f"Found {bad_rows} rows with invalid dates."
        )

    if df["match_id"].isna().any():
        raise ValueError("Found missing match IDs.")

    if df["match_id"].duplicated().any():
        duplicates = int(df["match_id"].duplicated().sum())
        raise ValueError(
            f"Found {duplicates} duplicate match IDs in input."
        )

    for column in [
        "team_1",
        "team_2",
        "city_resolved",
        "country_resolved",
    ]:
        if df[column].isna().any():
            missing_count = int(df[column].isna().sum())
            raise ValueError(
                f"Found {missing_count} missing values in {column}."
            )

    df["team_1"] = df["team_1"].astype(str).str.strip()
    df["team_2"] = df["team_2"].astype(str).str.strip()
    df["city_resolved"] = df["city_resolved"].astype(str).str.strip()
    df["country_resolved"] = (
        df["country_resolved"].astype(str).str.strip()
    )

    # Stable ordering for reproducibility.
    df = df.sort_values(
        ["date", "match_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


# ============================================================
# BUILD VENUE FEATURES
# ============================================================

def build_venue_features(
    df: pd.DataFrame,
    world_cup_teams: set[str],
) -> pd.DataFrame:
    """
    Create venue familiarity and home-venue features.

    Venue familiarity:
      Number of prior ODI appearances by each team at the
      same normalized venue.

    Leakage protection:
      - Only matches before the current date are counted.
      - All matches on the same date use the same snapshot.
      - Venue counts are updated after all features for that
        date have been created.
    """

    # Key: (team, normalized city, normalized country)
    venue_appearances = defaultdict(int)

    feature_rows = []

    grouped_dates = df.groupby("date", sort=True)

    for match_date, date_matches in grouped_dates:
        # Snapshot at the start of the date.
        start_of_day_snapshot = dict(venue_appearances)

        date_feature_rows = []

        for _, row in date_matches.iterrows():
            team_1 = row["team_1"]
            team_2 = row["team_2"]

            city = normalize_text(row["city_resolved"])
            country = normalize_country(row["country_resolved"])

            venue_key_1 = (team_1, city, country)
            venue_key_2 = (team_2, city, country)

            team_1_venue_matches = start_of_day_snapshot.get(
                venue_key_1,
                0,
            )

            team_2_venue_matches = start_of_day_snapshot.get(
                venue_key_2,
                0,
            )

            team_1_home = is_home_match(
                team_1,
                row["country_resolved"],
            )

            team_2_home = is_home_match(
                team_2,
                row["country_resolved"],
            )

            target = get_match_target(row)

            feature_row = {
                "match_id": row["match_id"],
                "date": row["date"],
                "team_1": team_1,
                "team_2": team_2,
                "city_resolved": row["city_resolved"],
                "country_resolved": row["country_resolved"],
                "team_1_venue_matches": team_1_venue_matches,
                "team_2_venue_matches": team_2_venue_matches,
                "venue_familiarity_difference": (
                    team_1_venue_matches - team_2_venue_matches
                ),
                "team_1_home": team_1_home,
                "team_2_home": team_2_home,
                "home_advantage_difference": (
                    team_1_home - team_2_home
                ),
                "target": target,
            }

            date_feature_rows.append(feature_row)

        # Save all features for the date before updating counts.
        feature_rows.extend(date_feature_rows)

        # Update venue appearances only after feature creation
        # for every match on this date.
        for _, row in date_matches.iterrows():
            team_1 = row["team_1"]
            team_2 = row["team_2"]

            city = normalize_text(row["city_resolved"])
            country = normalize_country(row["country_resolved"])

            venue_key_1 = (team_1, city, country)
            venue_key_2 = (team_2, city, country)

            venue_appearances[venue_key_1] += 1
            venue_appearances[venue_key_2] += 1

    features = pd.DataFrame(feature_rows)

    # Keep only matches where both teams are in the working
    # 2027 World Cup team set.
    tournament_features = features[
        features["team_1"].isin(world_cup_teams)
        & features["team_2"].isin(world_cup_teams)
    ].copy()

    return tournament_features.reset_index(drop=True)


# ============================================================
# VALIDATION REPORT
# ============================================================

def build_validation_report(
    input_df: pd.DataFrame,
    features: pd.DataFrame,
    world_cup_teams: set[str],
) -> str:
    report_lines = []

    report_lines.append(
        "WORLD CUP 2027 VENUE FEATURE VALIDATION"
    )
    report_lines.append("=" * 65)

    report_lines.append(f"Input rows: {len(input_df):,}")
    report_lines.append(
        f"Working World Cup teams: {len(world_cup_teams)}"
    )
    report_lines.append(
        "Matches with both teams in working set: "
        f"{len(features):,}"
    )

    report_lines.append(
        "Date range: "
        f"{input_df['date'].min().date()} to "
        f"{input_df['date'].max().date()}"
    )

    report_lines.append(
        f"Unique match IDs: {features['match_id'].nunique():,}"
    )

    duplicate_ids = int(features["match_id"].duplicated().sum())

    report_lines.append(
        f"Duplicate match IDs: {duplicate_ids:,}"
    )

    dates_with_multiple_matches = int(
        (input_df.groupby("date").size() > 1).sum()
    )

    report_lines.append(
        "Dates with multiple matches: "
        f"{dates_with_multiple_matches:,}"
    )

    report_lines.append("")
    report_lines.append("Target validation:")

    decided_matches = int(features["target"].notna().sum())
    excluded_matches = int(features["target"].isna().sum())

    target_1_count = int((features["target"] == 1).sum())
    target_0_count = int((features["target"] == 0).sum())

    report_lines.append(
        f"Decided matches: {decided_matches:,}"
    )
    report_lines.append(
        "Excluded no-result/tie/invalid-winner matches: "
        f"{excluded_matches:,}"
    )
    report_lines.append(
        f"Target 1 (team_1 wins): {target_1_count:,}"
    )
    report_lines.append(
        f"Target 0 (team_2 wins): {target_0_count:,}"
    )

    report_lines.append("")
    report_lines.append("Venue feature validation:")

    venue_feature_columns = [
        "team_1_venue_matches",
        "team_2_venue_matches",
        "venue_familiarity_difference",
    ]

    home_feature_columns = [
        "team_1_home",
        "team_2_home",
        "home_advantage_difference",
    ]

    missing_venue_features = int(
        features[venue_feature_columns].isna().sum().sum()
    )

    missing_home_features = int(
        features[home_feature_columns].isna().sum().sum()
    )

    report_lines.append(
        "Missing venue familiarity features: "
        f"{missing_venue_features:,}"
    )
    report_lines.append(
        f"Missing home features: {missing_home_features:,}"
    )

    report_lines.append(
        "Team 1 home flags: "
        f"{int(features['team_1_home'].sum()):,}"
    )
    report_lines.append(
        "Team 2 home flags: "
        f"{int(features['team_2_home'].sum()):,}"
    )

    report_lines.append("")
    report_lines.append("Home-venue rule validation:")

    for team in sorted(world_cup_teams):
        team_rows = features[
            (features["team_1"] == team)
            | (features["team_2"] == team)
        ]

        home_rows = team_rows[
            (team_rows["team_1"] == team)
            & (team_rows["team_1_home"] == 1)
            |
            (team_rows["team_2"] == team)
            & (team_rows["team_2_home"] == 1)
        ]

        report_lines.append(
            f"{team}: {len(home_rows):,} home-venue matches"
        )

    report_lines.append("")
    report_lines.append("Leakage protection:")
    report_lines.append(
        "Venue familiarity features use only matches "
        "before the current date."
    )
    report_lines.append(
        "All matches on a date use the same start-of-day "
        "venue-count snapshot."
    )
    report_lines.append(
        "Venue appearance counts are updated after feature "
        "creation for the date."
    )
    report_lines.append(
        "Historical matches against teams outside the 14-team "
        "set may contribute to venue familiarity, but are not "
        "included in the output rows."
    )

    report_lines.append("")
    report_lines.append("Ireland home-country rule:")
    report_lines.append(
        "Ireland home venues are restricted to Ireland and "
        "Northern Ireland."
    )
    report_lines.append(
        "England, Scotland, Wales, and other countries do not "
        "count as Ireland home venues."
    )

    return "\n".join(report_lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_data()
    world_cup_teams = load_world_cup_teams()

    features = build_venue_features(
        df=df,
        world_cup_teams=world_cup_teams,
    )

    # Final integrity checks.
    if features["match_id"].duplicated().any():
        raise ValueError(
            "Output contains duplicate match IDs."
        )

    if features[
        [
            "team_1_venue_matches",
            "team_2_venue_matches",
            "venue_familiarity_difference",
            "team_1_home",
            "team_2_home",
            "home_advantage_difference",
        ]
    ].isna().any().any():
        raise ValueError(
            "Output contains missing venue/home features."
        )

    # Save output.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # Save validation report.
    report = build_validation_report(
        input_df=df,
        features=features,
        world_cup_teams=world_cup_teams,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(report)

    print("")
    print(f"Saved features: {OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()