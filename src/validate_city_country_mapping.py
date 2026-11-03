
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "odi_matches_master.csv"
MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "city_country_mapping.csv"


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------
def load_data():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_PATH}")

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"City-country mapping not found: {MAPPING_PATH}")

    master = pd.read_csv(MASTER_PATH)
    mapping = pd.read_csv(MAPPING_PATH)

    required_master_columns = {"date", "venue", "city", "team_1", "team_2"}
    required_mapping_columns = {"city", "country", "mapping_status"}

    missing_master_columns = required_master_columns - set(master.columns)
    missing_mapping_columns = required_mapping_columns - set(mapping.columns)

    if missing_master_columns:
        raise ValueError(
            f"Master dataset is missing columns: {sorted(missing_master_columns)}"
        )

    if missing_mapping_columns:
        raise ValueError(
            f"Mapping file is missing columns: {sorted(missing_mapping_columns)}"
        )

    return master, mapping


# ---------------------------------------------------------
# Normalize city labels for comparison only
# ---------------------------------------------------------
def normalize_city(value):
    if pd.isna(value):
        return ""

    return str(value).strip().casefold()


# ---------------------------------------------------------
# Validation report
# ---------------------------------------------------------
def validate_mapping(master, mapping):
    master = master.copy()
    mapping = mapping.copy()

    master["city_normalized"] = master["city"].apply(normalize_city)
    mapping["city_normalized"] = mapping["city"].apply(normalize_city)

    # Ignore blank city labels in the mapping file
    mapping_nonblank = mapping[mapping["city_normalized"] != ""].copy()

    master_cities = set(
        master.loc[master["city_normalized"] != "", "city_normalized"].unique()
    )

    mapped_cities = set(mapping_nonblank["city_normalized"].unique())

    # -----------------------------------------------------
    # 1. Basic counts
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("CITY-COUNTRY MAPPING VALIDATION")
    print("=" * 70)

    print(f"Master dataset rows: {len(master):,}")
    print(f"Unique nonblank master cities: {len(master_cities):,}")
    print(f"Mapping rows, excluding blank city: {len(mapping_nonblank):,}")

    # -----------------------------------------------------
    # 2. Master cities missing from mapping
    # -----------------------------------------------------
    missing_cities = sorted(master_cities - mapped_cities)

    print("\n" + "-" * 70)
    print("1. CITIES IN MASTER DATASET BUT MISSING FROM MAPPING")
    print("-" * 70)

    if missing_cities:
        for city in missing_cities:
            print(f"- {city}")
    else:
        print("None. Every nonblank master city has a mapping row.")

    # -----------------------------------------------------
    # 3. Cities requiring review or missing country
    # -----------------------------------------------------
    review_rows = mapping_nonblank[
        mapping_nonblank["mapping_status"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("REVIEW_REQUIRED")
    ]

    blank_country_rows = mapping_nonblank[
        mapping_nonblank["country"].isna()
        | mapping_nonblank["country"].astype(str).str.strip().eq("")
    ]

    print("\n" + "-" * 70)
    print("2. MAPPING ROWS MARKED REVIEW_REQUIRED")
    print("-" * 70)

    if review_rows.empty:
        print("None.")
    else:
        print(
            review_rows[
                ["city", "country", "mapping_status", "notes"]
            ].to_string(index=False)
        )

    print("\n" + "-" * 70)
    print("3. MAPPING ROWS WITH BLANK COUNTRY")
    print("-" * 70)

    if blank_country_rows.empty:
        print("None.")
    else:
        print(
            blank_country_rows[
                ["city", "country", "mapping_status", "notes"]
            ].to_string(index=False)
        )

    # -----------------------------------------------------
    # 4. Duplicate city entries in mapping
    # -----------------------------------------------------
    duplicate_rows = mapping_nonblank[
        mapping_nonblank.duplicated(
            subset=["city_normalized"],
            keep=False
        )
    ].sort_values("city_normalized")

    print("\n" + "-" * 70)
    print("4. DUPLICATE CITY ENTRIES")
    print("-" * 70)

    if duplicate_rows.empty:
        print("None.")
    else:
        print(
            duplicate_rows[
                ["city", "country", "mapping_status", "notes"]
            ].to_string(index=False)
        )

    # -----------------------------------------------------
    # 5. Matches with blank city
    # -----------------------------------------------------
    blank_city_matches = master[
        master["city"].isna()
        | master["city"].astype(str).str.strip().eq("")
    ].copy()

    print("\n" + "-" * 70)
    print("5. MATCHES WITH BLANK CITY")
    print("-" * 70)

    print(f"Number of matches with blank city: {len(blank_city_matches):,}")

    if blank_city_matches.empty:
        print("None.")
    else:
        venue_summary = (
            blank_city_matches
            .groupby("venue", dropna=False)
            .agg(
                match_count=("match_id", "count"),
                first_date=("date", "min"),
                last_date=("date", "max"),
            )
            .reset_index()
            .sort_values("match_count", ascending=False)
        )

        print("\nBlank-city matches grouped by venue:")
        print(venue_summary.to_string(index=False))

    # -----------------------------------------------------
    # 6. Final status
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if (
        not missing_cities
        and review_rows.empty
        and blank_country_rows.empty
        and duplicate_rows.empty
    ):
        print("City mapping checks passed.")
    else:
        print("Mapping still needs attention. Review the sections above.")

    if not blank_city_matches.empty:
        print(
            "\nBlank city values are a separate issue. "
            "We should resolve them using venue information before "
            "creating venue-based features."
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    master, mapping = load_data()
    validate_mapping(master, mapping)


if __name__ == "__main__":
    main()
