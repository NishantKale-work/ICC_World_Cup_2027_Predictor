
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_features.csv"
)

VENUE_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_features_with_venue.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_features_with_venue_merged.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "venue_feature_merge_validation.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

VENUE_COLUMNS = [
    "team_1_venue_matches",
    "team_2_venue_matches",
    "venue_familiarity_difference",
    "team_1_home",
    "team_2_home",
    "home_advantage_difference",
]


# ============================================================
# LOAD WORKING 2027 TEAM LIST
# ============================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.world_cup_2027_teams import WORLD_CUP_2027_TEAMS
except ImportError as exc:
    raise ImportError(
        "Could not import WORLD_CUP_2027_TEAMS from "
        "config/world_cup_2027_teams.py. "
        "Check that the file exists and defines this variable."
    ) from exc

WORLD_CUP_TEAMS = set(WORLD_CUP_2027_TEAMS)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "match_id" not in df.columns:
        raise ValueError(
            f"{dataset_name} does not contain match_id."
        )

    if df["match_id"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing match IDs."
        )

    if df["match_id"].duplicated().any():
        duplicate_count = int(
            df["match_id"].duplicated().sum()
        )

        raise ValueError(
            f"{dataset_name} contains "
            f"{duplicate_count} duplicate match IDs."
        )

    return df


# ============================================================
# IDENTIFY MATCHES INVOLVING WORKING 2027 TEAMS
# ============================================================

def identify_tournament_matches(
    df: pd.DataFrame,
) -> pd.Series:
    required_columns = ["team_1", "team_2"]

    missing_columns = (
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing team columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        df["team_1"].isin(WORLD_CUP_TEAMS)
        & df["team_2"].isin(WORLD_CUP_TEAMS)
    )


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    base_df: pd.DataFrame,
    venue_df: pd.DataFrame,
) -> None:

    missing_venue_columns = (
        set(VENUE_COLUMNS) - set(venue_df.columns)
    )

    if missing_venue_columns:
        raise ValueError(
            "Venue dataset is missing columns: "
            f"{sorted(missing_venue_columns)}"
        )

    base_ids = set(base_df["match_id"])
    venue_ids = set(venue_df["match_id"])

    venue_ids_not_in_base = venue_ids - base_ids

    if venue_ids_not_in_base:
        raise ValueError(
            "Venue dataset contains match IDs absent "
            "from the base feature dataset. "
            f"Count: {len(venue_ids_not_in_base):,}"
        )

    tournament_mask = identify_tournament_matches(
        base_df
    )

    tournament_ids = set(
        base_df.loc[tournament_mask, "match_id"]
    )

    missing_tournament_ids = (
        tournament_ids - venue_ids
    )

    if missing_tournament_ids:
        raise ValueError(
            "Some matches involving the working 2027 teams "
            "are missing from the venue dataset. "
            f"Count: {len(missing_tournament_ids):,}"
        )

    non_tournament_ids = (
        base_ids - tournament_ids
    )

    unexpected_venue_ids = (
        venue_ids & non_tournament_ids
    )

    if unexpected_venue_ids:
        raise ValueError(
            "Venue dataset unexpectedly contains matches "
            "outside the working 2027 team set. "
            f"Count: {len(unexpected_venue_ids):,}"
        )

    print("Input validation:")
    print(f"Base feature rows: {len(base_df):,}")
    print(f"Venue feature rows: {len(venue_df):,}")
    print(
        "Matches involving working 2027 teams: "
        f"{len(tournament_ids):,}"
    )
    print(
        "Matches outside working 2027 team set: "
        f"{len(non_tournament_ids):,}"
    )
    print(
        "Expected matches without venue features: "
        f"{len(non_tournament_ids):,}"
    )


# ============================================================
# MERGE
# ============================================================

def merge_features(
    base_df: pd.DataFrame,
    venue_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge venue features onto the baseline feature dataset
    using match_id.

    The baseline dataset remains the left-hand dataset.
    Matches outside the working 2027 team set are expected
    to have missing venue features.
    """

    existing_venue_columns = [
        column
        for column in VENUE_COLUMNS
        if column in base_df.columns
    ]

    if existing_venue_columns:
        raise ValueError(
            "Base feature dataset already contains venue "
            "feature columns. Inspect the input before "
            "proceeding. Existing columns: "
            f"{existing_venue_columns}"
        )

    venue_subset = venue_df[
        ["match_id"] + VENUE_COLUMNS
    ].copy()

    merged_df = base_df.merge(
        venue_subset,
        on="match_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_rows = merged_df[
        merged_df["_merge"] != "both"
    ]

    tournament_mask = identify_tournament_matches(
        merged_df
    )

    unmatched_tournament_rows = unmatched_rows[
        identify_tournament_matches(unmatched_rows)
    ]

    if not unmatched_tournament_rows.empty:
        raise ValueError(
            "Some matches involving the working 2027 teams "
            "did not receive venue features. "
            f"Count: {len(unmatched_tournament_rows):,}"
        )

    print(
        "Expected unmatched non-tournament rows: "
        f"{len(unmatched_rows):,}"
    )

    merged_df = merged_df.drop(
        columns=["_merge"]
    )

    return merged_df


# ============================================================
# VALIDATION REPORT
# ============================================================

def create_report(
    base_df: pd.DataFrame,
    venue_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> str:

    lines = []

    tournament_mask = identify_tournament_matches(
        merged_df
    )

    tournament_rows = merged_df.loc[
        tournament_mask
    ]

    non_tournament_rows = merged_df.loc[
        ~tournament_mask
    ]

    lines.append("VENUE FEATURE MERGE VALIDATION")
    lines.append("=" * 65)

    lines.append(
        f"Base feature rows: {len(base_df):,}"
    )
    lines.append(
        f"Venue feature rows: {len(venue_df):,}"
    )
    lines.append(
        f"Merged feature rows: {len(merged_df):,}"
    )

    lines.append(
        "Matches involving working 2027 teams: "
        f"{len(tournament_rows):,}"
    )
    lines.append(
        "Matches outside working 2027 team set: "
        f"{len(non_tournament_rows):,}"
    )

    lines.append(
        "Base unique match IDs: "
        f"{base_df['match_id'].nunique():,}"
    )
    lines.append(
        "Merged unique match IDs: "
        f"{merged_df['match_id'].nunique():,}"
    )

    lines.append(
        "Duplicate merged match IDs: "
        f"{int(merged_df['match_id'].duplicated().sum()):,}"
    )

    lines.append("")
    lines.append(
        "Venue feature missing-value check:"
    )

    lines.append(
        "Expected missing venue values on "
        "non-tournament-team matches:"
    )

    for column in VENUE_COLUMNS:
        missing_all = int(
            merged_df[column].isna().sum()
        )

        missing_tournament = int(
            tournament_rows[column].isna().sum()
        )

        lines.append(
            f"{column}: {missing_all:,} missing overall; "
            f"{missing_tournament:,} missing among "
            "working 2027 team matches"
        )

    lines.append("")
    lines.append("Merge integrity:")

    lines.append(
        "Base row count preserved: "
        f"{len(base_df) == len(merged_df)}"
    )

    lines.append(
        "Base match IDs preserved: "
        f"{set(base_df['match_id']) == set(merged_df['match_id'])}"
    )

    lines.append(
        "All working 2027 team matches have venue features: "
        f"{not tournament_rows[VENUE_COLUMNS].isna().any().any()}"
    )

    lines.append(
        "Non-tournament matches allowed to have missing "
        "venue features: True"
    )

    lines.append(
        "One-to-one merge validated: True"
    )

    lines.append("")
    lines.append(
        "Source datasets were not modified."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("Loading baseline feature dataset...")
    base_df = load_dataset(
        BASE_FEATURES_PATH,
        "Base feature dataset",
    )

    print("Loading venue feature dataset...")
    venue_df = load_dataset(
        VENUE_FEATURES_PATH,
        "Venue feature dataset",
    )

    validate_inputs(
        base_df=base_df,
        venue_df=venue_df,
    )

    print("\nMerging features...")

    merged_df = merge_features(
        base_df=base_df,
        venue_df=venue_df,
    )

    # --------------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------------

    if merged_df["match_id"].duplicated().any():
        raise ValueError(
            "Merged dataset contains duplicate match IDs."
        )

    if len(merged_df) != len(base_df):
        raise ValueError(
            "Merged dataset row count differs from base "
            "dataset. "
            f"Base: {len(base_df):,}; "
            f"Merged: {len(merged_df):,}"
        )

    if set(merged_df["match_id"]) != set(
        base_df["match_id"]
    ):
        raise ValueError(
            "Merged dataset does not preserve all base "
            "match IDs."
        )

    tournament_mask = identify_tournament_matches(
        merged_df
    )

    tournament_rows = merged_df.loc[
        tournament_mask
    ]

    missing_tournament_features = int(
        tournament_rows[VENUE_COLUMNS].isna().sum().sum()
    )

    if missing_tournament_features > 0:
        raise ValueError(
            "Matches involving the working 2027 teams "
            "contain missing venue features. "
            f"Missing values: {missing_tournament_features:,}"
        )

    non_tournament_rows = merged_df.loc[
        ~tournament_mask
    ]

    print("\nFinal checks passed:")
    print(
        f"All base rows preserved: {len(merged_df):,}"
    )
    print(
        "Matches with venue features: "
        f"{len(tournament_rows):,}"
    )
    print(
        "Matches without venue features "
        "(outside working team set): "
        f"{len(non_tournament_rows):,}"
    )

    # --------------------------------------------------------
    # Save merged dataset
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save validation report
    # --------------------------------------------------------

    report = create_report(
        base_df=base_df,
        venue_df=venue_df,
        merged_df=merged_df,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("\n")
    print(report)

    print("\nSaved merged dataset:")
    print(OUTPUT_PATH)

    print("\nSaved validation report:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
