from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_features_with_venue_merged.csv"
)

RESOLVED_MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_matches_with_resolved_cities.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "venue_logistic_regression_2027.joblib"
)

DEFAULT_CUTOFF_DATE = "2026-10-25"


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
# Home-region configuration
# ============================================================

HOME_REGIONS = {
    "Afghanistan": {"Afghanistan"},
    "Australia": {"Australia"},
    "Bangladesh": {"Bangladesh"},
    "England": {"England"},
    "India": {"India"},
    "Ireland": {"Ireland", "Northern Ireland"},
    "Netherlands": {"Netherlands"},
    "New Zealand": {"New Zealand"},
    "Pakistan": {"Pakistan"},
    "Scotland": {"Scotland"},
    "South Africa": {"South Africa"},
    "Sri Lanka": {"Sri Lanka"},
    "Zimbabwe": {"Zimbabwe"},
    "West Indies": {
        "Antigua and Barbuda",
        "Barbados",
        "Dominica",
        "Grenada",
        "Guyana",
        "Jamaica",
        "Saint Kitts and Nevis",
        "Saint Lucia",
        "Saint Vincent and the Grenadines",
        "Trinidad and Tobago",
    },
}


# ============================================================
# Column helpers
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "")
        .replace(",", "")
    )


def normalize_country(value) -> str:
    text = normalize_text(value)

    aliases = {
        "england": "england",
        "united kingdom": "england",
        "uk": "england",
        "republic of ireland": "ireland",
        "northern ireland": "northern ireland",
        "new zealand": "new zealand",
        "south africa": "south africa",
        "west indies": "west indies",
    }

    return aliases.get(text, text)


# ============================================================
# Load historical data
# ============================================================

def load_historical_data() -> pd.DataFrame:
    """Join model features with resolved venue geography by unique match_id."""
    for file_path in (HISTORICAL_FILE, RESOLVED_MATCHES_FILE):
        if not file_path.exists():
            raise FileNotFoundError(f"Required historical file not found:\\n{file_path}")

    features = pd.read_csv(HISTORICAL_FILE)
    resolved = pd.read_csv(RESOLVED_MATCHES_FILE)

    features = features.rename(columns={
        find_column(features, ["date", "start_date", "match_date"], "feature match date"): "date",
        find_column(features, ["match_id", "id", "matchid"], "feature match ID"): "match_id",
        find_column(features, ["team_1", "team1", "team_1_name", "team1_name"], "feature Team 1"): "team_1",
        find_column(features, ["team_2", "team2", "team_2_name", "team2_name"], "feature Team 2"): "team_2",
    })

    resolved = resolved.rename(columns={
        find_column(resolved, ["match_id", "id", "matchid"], "resolved match ID"): "match_id",
        find_column(resolved, ["date", "start_date", "match_date"], "resolved match date"): "resolved_date",
        find_column(resolved, ["team_1", "team1", "team_1_name", "team1_name"], "resolved Team 1"): "resolved_team_1",
        find_column(resolved, ["team_2", "team2", "team_2_name", "team2_name"], "resolved Team 2"): "resolved_team_2",
        find_column(resolved, ["city_resolved", "city", "venue_city"], "resolved city"): "city_resolved",
        find_column(resolved, ["country_resolved", "country"], "resolved country"): "country_resolved",
    })

    features["match_id"] = features["match_id"].astype("string").str.strip()
    resolved["match_id"] = resolved["match_id"].astype("string").str.strip()

    geography = resolved[[
        "match_id", "resolved_date", "resolved_team_1", "resolved_team_2",
        "city_resolved", "country_resolved",
    ]].copy()

    if geography["match_id"].duplicated().any():
        raise ValueError("Resolved geography contains duplicate match IDs.")

    df = features.merge(geography, on="match_id", how="left", validate="one_to_one")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["resolved_date"] = pd.to_datetime(df["resolved_date"], errors="coerce")

    date_mismatch = df["date"].notna() & df["resolved_date"].notna() & df["date"].ne(df["resolved_date"])
    if date_mismatch.any():
        raise ValueError(f"Date mismatch found for {int(date_mismatch.sum())} joined matches.")

    for left, right in [("team_1", "resolved_team_1"), ("team_2", "resolved_team_2")]:
        mismatch = (
            df[left].notna() & df[right].notna()
            & df[left].astype(str).str.casefold().ne(df[right].astype(str).str.casefold())
        )
        if mismatch.any():
            raise ValueError(f"{left} mismatch found for {int(mismatch.sum())} joined matches.")

    required = ["date", "match_id", "team_1", "team_2", "city_resolved", "country_resolved"] + BASELINE_FEATURES
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Joined historical dataset is missing required columns: {missing}")

    df = df.dropna(subset=["date", "match_id", "team_1", "team_2"]).copy()
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    missing_geo = df["city_resolved"].isna() | df["country_resolved"].isna()
    if missing_geo.any():
        raise ValueError(f"{int(missing_geo.sum())} matches have missing resolved city/country after joining.")

    return df


# ============================================================
# Reconstruct each team's latest pre-tournament state
# ============================================================

def get_latest_team_state(
    df: pd.DataFrame,
    team_name: str,
    cutoff_date: pd.Timestamp,
) -> dict:
    """
    Get the team's latest historical Elo, form, and matches-played
    snapshot before the requested cutoff date.

    The team may appear as either Team 1 or Team 2 in historical rows.
    """

    history = df[df["date"] <= cutoff_date].copy()

    team_as_1 = history[
        history["team_1"].str.casefold() == team_name.casefold()
    ].copy()

    team_as_2 = history[
        history["team_2"].str.casefold() == team_name.casefold()
    ].copy()

    if not team_as_1.empty:
        team_as_1["team_elo"] = team_as_1["team_1_elo"]
        team_as_1["team_form"] = team_as_1["team_1_recent_form"]
        team_as_1["team_matches_played"] = (
            team_as_1["team_1_matches_played"]
        )

    if not team_as_2.empty:
        team_as_2["team_elo"] = team_as_2["team_2_elo"]
        team_as_2["team_form"] = team_as_2["team_2_recent_form"]
        team_as_2["team_matches_played"] = (
            team_as_2["team_2_matches_played"]
        )

    team_history = pd.concat(
        [team_as_1, team_as_2],
        ignore_index=True,
    )

    if team_history.empty:
        # Explicit cold-start fallback for teams absent from the local dataset.
        # These are neutral priors, NOT measured team strength.
        # Elo 1500 is the conventional starting rating; form 0.5 is neutral.
        return {
            "elo": 1500.0,
            "recent_form": 0.5,
            "matches_played": 0.0,
            "latest_match_date": pd.NaT,
            "state_source": "COLD_START_ASSUMPTION",
        }

    team_history = team_history.sort_values(
        ["date", "match_id"]
    )

    latest = team_history.iloc[-1]

    return {
        "elo": float(latest["team_elo"]),
        "recent_form": float(latest["team_form"]),
        "matches_played": float(latest["team_matches_played"]),
        "latest_match_date": latest["date"],
        "state_source": "HISTORICAL_DATA",
    }


# ============================================================
# Venue familiarity
# ============================================================

def get_venue_matches(
    df: pd.DataFrame,
    team_name: str,
    venue_city: str,
    venue_country: str,
    cutoff_date: pd.Timestamp,
) -> int:
    """Count prior matches for a team at the same city and country."""

    history = df[df["date"] <= cutoff_date].copy()

    team_mask = (
        history["team_1"].str.casefold().eq(team_name.casefold())
        | history["team_2"].str.casefold().eq(team_name.casefold())
    )

    city_mask = (
        history["city_resolved"].map(normalize_text)
        == normalize_text(venue_city)
    )

    country_mask = (
        history["country_resolved"].map(normalize_country)
        == normalize_country(venue_country)
    )

    return int(
        (team_mask & city_mask & country_mask).sum()
    )


# ============================================================
# Home advantage
# ============================================================

def is_home_team(
    team_name: str,
    venue_country: str,
) -> int:
    regions = HOME_REGIONS.get(team_name, set())

    normalized_venue_country = normalize_text(venue_country)

    normalized_regions = {
        normalize_text(region)
        for region in regions
    }

    return int(
        normalized_venue_country in normalized_regions
    )


# ============================================================
# Build one fixture's features
# ============================================================

def build_match_features(
    df: pd.DataFrame,
    team_1: str,
    team_2: str,
    venue_city: str,
    venue_country: str,
    cutoff_date: str,
) -> pd.DataFrame:

    cutoff = pd.Timestamp(cutoff_date)

    team_1_state = get_latest_team_state(
        df,
        team_1,
        cutoff,
    )

    team_2_state = get_latest_team_state(
        df,
        team_2,
        cutoff,
    )

    cold_start_teams = [
        team for team, state in [
            (team_1, team_1_state),
            (team_2, team_2_state),
        ]
        if state.get("state_source") == "COLD_START_ASSUMPTION"
    ]
    if cold_start_teams:
        print(
            "WARNING: Cold-start assumption used for "
            + ", ".join(cold_start_teams)
            + ". Elo=1500, recent_form=0.5, matches_played=0; "
              "these are not observed historical team ratings."
        )

    team_1_venue_matches = get_venue_matches(
        df,
        team_1,
        venue_city,
        venue_country,
        cutoff,
    )

    team_2_venue_matches = get_venue_matches(
        df,
        team_2,
        venue_city,
        venue_country,
        cutoff,
    )

    team_1_home = is_home_team(
        team_1,
        venue_country,
    )

    team_2_home = is_home_team(
        team_2,
        venue_country,
    )

    features = {
        "team_1_elo": team_1_state["elo"],
        "team_2_elo": team_2_state["elo"],
        "elo_difference": (
            team_1_state["elo"] - team_2_state["elo"]
        ),

        "team_1_recent_form": team_1_state["recent_form"],
        "team_2_recent_form": team_2_state["recent_form"],
        "recent_form_difference": (
            team_1_state["recent_form"]
            - team_2_state["recent_form"]
        ),

        "team_1_matches_played": (
            team_1_state["matches_played"]
        ),
        "team_2_matches_played": (
            team_2_state["matches_played"]
        ),

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
    }

    return pd.DataFrame([features])


# ============================================================
# Predict fixture
# ============================================================

def predict_fixture(
    team_1: str,
    team_2: str,
    venue_city: str,
    venue_country: str,
    cutoff_date: str,
) -> None:

    historical_df = load_historical_data()

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Saved venue model not found:\n{MODEL_FILE}"
        )

    artifact = joblib.load(MODEL_FILE)

    model = artifact["model"]
    expected_features = artifact["feature_columns"]

    match_features = build_match_features(
        df=historical_df,
        team_1=team_1,
        team_2=team_2,
        venue_city=venue_city,
        venue_country=venue_country,
        cutoff_date=cutoff_date,
    )

    # Ensure the model receives features in its expected order.
    match_features = match_features[expected_features]

    probability_team_1 = float(
        model.predict_proba(match_features)[0, 1]
    )

    probability_team_2 = 1.0 - probability_team_1

    print("\n" + "=" * 70)
    print("2027 FIXTURE PREDICTION")
    print("=" * 70)

    print(f"{team_1} vs {team_2}")
    print(f"Venue: {venue_city}, {venue_country}")
    print(f"Historical cutoff: {cutoff_date}")

    print("\nModel features")
    print("-" * 70)
    print(match_features.to_string(index=False))

    print("\nPredicted probabilities")
    print("-" * 70)
    print(f"{team_1}: {probability_team_1:.1%}")
    print(f"{team_2}: {probability_team_2:.1%}")

    print("\nReminder: these are model estimates, not guarantees.")


# ============================================================
# Command-line entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a 2027 ODI World Cup fixture."
    )

    parser.add_argument(
        "--team1",
        required=True,
        help="First team",
    )

    parser.add_argument(
        "--team2",
        required=True,
        help="Second team",
    )

    parser.add_argument(
        "--city",
        required=True,
        help="Venue city",
    )

    parser.add_argument(
        "--country",
        required=True,
        help="Venue country",
    )

    parser.add_argument(
        "--cutoff",
        default=DEFAULT_CUTOFF_DATE,
        help="Historical data cutoff date, YYYY-MM-DD",
    )

    args = parser.parse_args()

    predict_fixture(
        team_1=args.team1,
        team_2=args.team2,
        venue_city=args.city,
        venue_country=args.country,
        cutoff_date=args.cutoff,
    )


if __name__ == "__main__":
    main()