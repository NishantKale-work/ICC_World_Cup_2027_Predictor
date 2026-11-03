"""
Build and validate the canonical men's ODI match dataset.

Primary source:
    data/raw/odis_male_json.zip

Secondary comparison source:
    data/raw/archive (1).zip
        ODI_Match_info.csv

Outputs:
    data/processed/odi_matches_master.csv
    reports/match_dataset_validation.txt

The script does not modify the original ZIP archives.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_JSON_ZIP = RAW_DIR / "odis_male_json.zip"
SECONDARY_CSV_ZIP = RAW_DIR / "archive (1).zip"

OUTPUT_CSV = PROCESSED_DIR / "odi_matches_master.csv"
VALIDATION_REPORT = REPORTS_DIR / "match_dataset_validation.txt"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """Normalize text for comparisons without changing the stored value."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_team_pair(team_1: Any, team_2: Any) -> tuple[str, str]:
    """Create an order-independent team pair for matching."""
    teams = sorted([normalize_text(team_1), normalize_text(team_2)])
    return teams[0], teams[1]


def get_match_date(info: dict[str, Any]) -> str | None:
    """Return the first match date in ISO format."""
    dates = info.get("dates", [])

    if not dates:
        return None

    parsed = pd.to_datetime(dates[0], errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y-%m-%d")


def classify_result(outcome: dict[str, Any]) -> str:
    """
    Classify match result.

    Possible values:
        win
        tie
        no_result
        other
    """
    if outcome.get("winner"):
        return "win"

    result_text = normalize_text(outcome.get("result", ""))

    if "tie" in result_text:
        return "tie"

    if "no result" in result_text or "abandoned" in result_text:
        return "no_result"

    if outcome.get("method"):
        return "other"

    return "other"


def extract_margin(outcome: dict[str, Any]) -> tuple[str | None, float | None]:
    """Extract the winning margin, if available."""
    by = outcome.get("by", {})

    if not isinstance(by, dict):
        return None, None

    if "runs" in by:
        return "runs", float(by["runs"])

    if "wickets" in by:
        return "wickets", float(by["wickets"])

    if "innings" in by:
        return "innings", float(by["innings"])

    return None, None


def make_comparison_key(
    date: Any,
    team_1: Any,
    team_2: Any,
) -> tuple[str, str, str]:
    """Build a comparison key from date and order-independent teams."""
    parsed_date = pd.to_datetime(date, errors="coerce")

    date_string = (
        parsed_date.strftime("%Y-%m-%d")
        if not pd.isna(parsed_date)
        else ""
    )

    first_team, second_team = normalize_team_pair(team_1, team_2)

    return date_string, first_team, second_team


# ---------------------------------------------------------------------
# Primary source: structured ODI JSON
# ---------------------------------------------------------------------

def load_primary_odi_json() -> pd.DataFrame:
    """Load all men's ODI matches from the Cricsheet-style JSON archive."""

    if not PRIMARY_JSON_ZIP.exists():
        raise FileNotFoundError(
            f"Primary archive not found: {PRIMARY_JSON_ZIP}"
        )

    records = []

    with zipfile.ZipFile(PRIMARY_JSON_ZIP, "r") as archive:
        json_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".json")
        ]

        print(f"JSON files found in primary archive: {len(json_members):,}")

        for index, member_name in enumerate(json_members, start=1):
            try:
                with archive.open(member_name) as file:
                    match_data = json.load(file)

                info = match_data.get("info", {})

                # Strictly retain men's ODI matches.
                if normalize_text(info.get("gender")) != "male":
                    continue

                if normalize_text(info.get("match_type")) != "odi":
                    continue

                teams = info.get("teams", [])

                if not isinstance(teams, list) or len(teams) != 2:
                    continue

                team_1, team_2 = teams

                outcome = info.get("outcome", {})
                if not isinstance(outcome, dict):
                    outcome = {}

                toss = info.get("toss", {})
                if not isinstance(toss, dict):
                    toss = {}

                margin_type, margin_value = extract_margin(outcome)

                result_type = classify_result(outcome)

                records.append(
                    {
                        "match_id": Path(member_name).stem,
                        "date": get_match_date(info),
                        "team_1": team_1,
                        "team_2": team_2,
                        "winner": outcome.get("winner"),
                        "result_type": result_type,
                        "result_detail": outcome.get("result"),
                        "method": outcome.get("method"),
                        "margin_type": margin_type,
                        "margin_value": margin_value,
                        "toss_winner": toss.get("winner"),
                        "toss_decision": toss.get("decision"),
                        "venue": info.get("venue"),
                        "city": info.get("city"),
                        "event_name": (
                            info.get("event", {}).get("name")
                            if isinstance(info.get("event"), dict)
                            else None
                        ),
                        "match_type": info.get("match_type"),
                        "gender": info.get("gender"),
                        "source": "odis_male_json.zip",
                    }
                )

            except Exception as error:
                print(
                    f"Warning: Could not process {member_name}: "
                    f"{type(error).__name__}: {error}"
                )

            if index % 500 == 0:
                print(f"Processed {index:,}/{len(json_members):,} files")

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise ValueError(
            "No men's ODI matches were extracted. "
            "Check the archive and JSON schema."
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    dataframe = dataframe.sort_values(
        by=["date", "match_id"],
        na_position="last",
    ).reset_index(drop=True)

    return dataframe


# ---------------------------------------------------------------------
# Secondary source: match-info CSV
# ---------------------------------------------------------------------

def load_secondary_csv() -> pd.DataFrame | None:
    """Load ODI match-info CSV from the secondary archive."""

    if not SECONDARY_CSV_ZIP.exists():
        print(
            "\nSecondary CSV archive not found. "
            "Skipping cross-source comparison."
        )
        return None

    expected_member = "ODI_Match_info.csv"

    with zipfile.ZipFile(SECONDARY_CSV_ZIP, "r") as archive:
        members = archive.namelist()

        if expected_member not in members:
            print(
                f"\n{expected_member} not found in "
                f"{SECONDARY_CSV_ZIP.name}. Skipping comparison."
            )
            return None

        with archive.open(expected_member) as file:
            dataframe = pd.read_csv(file)

    required_columns = {"date", "team1", "team2"}

    if not required_columns.issubset(dataframe.columns):
        print(
            "\nSecondary CSV is missing required columns. "
            "Skipping cross-source comparison."
        )
        return None

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    return dataframe


# ---------------------------------------------------------------------
# Validation and comparison
# ---------------------------------------------------------------------

def validate_primary_dataset(dataframe: pd.DataFrame) -> list[str]:
    """Run basic integrity checks on the primary dataset."""

    findings = []

    findings.append(f"Primary match rows: {len(dataframe):,}")
    findings.append(
        f"Date range: {dataframe['date'].min()} to "
        f"{dataframe['date'].max()}"
    )

    findings.append(
        f"Missing dates: {int(dataframe['date'].isna().sum()):,}"
    )

    findings.append(
        f"Missing winners: {int(dataframe['winner'].isna().sum()):,}"
    )

    findings.append("\nResult-type counts:")
    result_counts = dataframe["result_type"].value_counts(dropna=False)

    for result_type, count in result_counts.items():
        findings.append(f"  {result_type}: {count:,}")

    duplicate_match_ids = int(dataframe["match_id"].duplicated().sum())

    findings.append(
        f"\nDuplicate match IDs: {duplicate_match_ids:,}"
    )

    duplicate_combinations = dataframe.duplicated(
        subset=["date", "team_1", "team_2"],
        keep=False,
    )

    findings.append(
        "Rows involved in repeated date/team combinations: "
        f"{int(duplicate_combinations.sum()):,}"
    )

    findings.append("\nMost common teams:")
    team_counts = pd.concat(
        [dataframe["team_1"], dataframe["team_2"]]
    ).value_counts()

    for team, count in team_counts.head(15).items():
        findings.append(f"  {team}: {count:,} appearances")

    return findings


def compare_with_secondary(
    primary: pd.DataFrame,
    secondary: pd.DataFrame | None,
) -> list[str]:
    """Compare primary JSON records with the secondary CSV source."""

    findings = []

    if secondary is None:
        findings.append(
            "Secondary source comparison was skipped."
        )
        return findings

    primary_keys = set(
        primary.apply(
            lambda row: make_comparison_key(
                row["date"],
                row["team_1"],
                row["team_2"],
            ),
            axis=1,
        )
    )

    secondary_keys = set(
        secondary.apply(
            lambda row: make_comparison_key(
                row["date"],
                row["team1"],
                row["team2"],
            ),
            axis=1,
        )
    )

    overlap = primary_keys.intersection(secondary_keys)
    only_primary = primary_keys - secondary_keys
    only_secondary = secondary_keys - primary_keys

    findings.append(f"Secondary CSV rows: {len(secondary):,}")
    findings.append(f"Primary unique comparison keys: {len(primary_keys):,}")
    findings.append(
        f"Secondary unique comparison keys: {len(secondary_keys):,}"
    )
    findings.append(f"Matching date/team combinations: {len(overlap):,}")
    findings.append(
        f"Keys only in primary JSON source: {len(only_primary):,}"
    )
    findings.append(
        f"Keys only in secondary CSV source: {len(only_secondary):,}"
    )

    findings.append("\nSecondary CSV result values:")
    if "result" in secondary.columns:
        result_counts = secondary["result"].value_counts(dropna=False)

        for result, count in result_counts.items():
            findings.append(f"  {result}: {count:,}")

    findings.append("\nSecondary CSV winner status:")
    if "winner" in secondary.columns:
        missing_winners = int(secondary["winner"].isna().sum())
        present_winners = int(secondary["winner"].notna().sum())

        findings.append(f"  Winner present: {present_winners:,}")
        findings.append(f"  Winner missing: {missing_winners:,}")

    return findings


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

def save_validation_report(
    primary: pd.DataFrame,
    primary_findings: list[str],
    comparison_findings: list[str],
) -> None:
    """Save validation findings to a readable text file."""

    with VALIDATION_REPORT.open("w", encoding="utf-8") as file:
        file.write("ODI MASTER DATASET VALIDATION\n")
        file.write("=" * 80 + "\n\n")

        file.write("PRIMARY DATASET FINDINGS\n")
        file.write("-" * 80 + "\n")

        for finding in primary_findings:
            file.write(f"{finding}\n")

        file.write("\nSECONDARY SOURCE COMPARISON\n")
        file.write("-" * 80 + "\n")

        for finding in comparison_findings:
            file.write(f"{finding}\n")

        file.write("\nMASTER DATASET COLUMNS\n")
        file.write("-" * 80 + "\n")

        for column in primary.columns:
            file.write(f"{column}\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("BUILDING MEN'S ODI MASTER MATCH DATASET")
    print("=" * 80)

    primary = load_primary_odi_json()

    print(f"\nExtracted men's ODI matches: {len(primary):,}")

    # Save the canonical match-level dataset.
    primary.to_csv(OUTPUT_CSV, index=False)

    print(f"\nMaster dataset saved to:\n{OUTPUT_CSV}")

    primary_findings = validate_primary_dataset(primary)

    secondary = load_secondary_csv()

    comparison_findings = compare_with_secondary(
        primary=primary,
        secondary=secondary,
    )

    save_validation_report(
        primary=primary,
        primary_findings=primary_findings,
        comparison_findings=comparison_findings,
    )

    print("\n" + "=" * 80)
    print("PRIMARY DATASET FINDINGS")
    print("=" * 80)

    for finding in primary_findings:
        print(finding)

    print("\n" + "=" * 80)
    print("SECONDARY SOURCE COMPARISON")
    print("=" * 80)

    for finding in comparison_findings:
        print(finding)

    print("\nValidation report saved to:")
    print(VALIDATION_REPORT)

    print("\nDone. No model has been trained yet.")


if __name__ == "__main__":
    main()