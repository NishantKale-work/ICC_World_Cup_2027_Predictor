
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "odi_matches_master.csv"
CITY_MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "city_country_mapping.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "odi_matches_with_resolved_cities.csv"
)


# ---------------------------------------------------------
# Explicit venue-to-city mappings
#
# These are used only when the original city field is blank.
# Do not add a venue here unless its location is unambiguous.
# ---------------------------------------------------------
VENUE_LOCATION_MAP = {
    "Harare Sports Club": ("Harare", "Zimbabwe"),
    "Rangiri Dambulla International Stadium": ("Dambulla", "Sri Lanka"),
    "Sydney Cricket Ground": ("Sydney", "Australia"),
    "Melbourne Cricket Ground": ("Melbourne", "Australia"),
    "Adelaide Oval": ("Adelaide", "Australia"),
    "Dubai International Cricket Stadium": ("Dubai", "United Arab Emirates"),
    "Pallekele International Cricket Stadium": ("Kandy", "Sri Lanka"),
    "Sharjah Cricket Stadium": ("Sharjah", "United Arab Emirates"),
    "Rawalpindi Cricket Stadium": ("Rawalpindi", "Pakistan"),
    "Multan Cricket Stadium": ("Multan", "Pakistan"),
    "Queenstown Events Centre": ("Queenstown", "New Zealand"),
    "Chittagong Divisional Stadium": ("Chittagong", "Bangladesh"),
    "Mombasa Sports Club Ground": ("Mombasa", "Kenya"),
    "Dubai Sports City Cricket Stadium": ("Dubai", "United Arab Emirates"),
    "Perth Stadium": ("Perth", "Australia"),
    "Galle International Stadium": ("Galle", "Sri Lanka"),
    "Bulawayo Athletic Club": ("Bulawayo", "Zimbabwe"),
    "Sheikhupura Stadium": ("Sheikhupura", "Pakistan"),
    "Niaz Stadium": ("Hyderabad", "Pakistan"),
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_text(value):
    """Normalize text for matching without changing source values."""
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().casefold().split())


def is_blank(value):
    return pd.isna(value) or str(value).strip() == ""


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
def load_data():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_PATH}")

    if not CITY_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"City-country mapping not found: {CITY_MAPPING_PATH}"
        )

    master = pd.read_csv(MASTER_PATH)
    city_mapping = pd.read_csv(CITY_MAPPING_PATH)

    required_master_columns = {"city", "venue"}
    required_mapping_columns = {"city", "country", "mapping_status"}

    missing_master_columns = required_master_columns - set(master.columns)
    missing_mapping_columns = required_mapping_columns - set(city_mapping.columns)

    if missing_master_columns:
        raise ValueError(
            f"Master dataset is missing columns: {sorted(missing_master_columns)}"
        )

    if missing_mapping_columns:
        raise ValueError(
            f"City mapping is missing columns: {sorted(missing_mapping_columns)}"
        )

    return master, city_mapping


# ---------------------------------------------------------
# Build city-country lookup
# ---------------------------------------------------------
def build_city_country_lookup(city_mapping):
    lookup = {}

    for _, row in city_mapping.iterrows():
        city = row["city"]
        country = row["country"]
        status = row["mapping_status"]

        if is_blank(city):
            continue

        if is_blank(country):
            continue

        if str(status).strip().upper() == "REVIEW_REQUIRED":
            continue

        lookup[normalize_text(city)] = str(country).strip()

    return lookup


# ---------------------------------------------------------
# Resolve cities
# ---------------------------------------------------------
def resolve_cities(master, city_mapping):
    df = master.copy()

    city_country_lookup = build_city_country_lookup(city_mapping)

    # Preserve the original city column.
    df["city_resolved"] = df["city"]
    df["country_resolved"] = ""
    df["city_resolution_status"] = ""

    # Normalize venue names for lookup.
    normalized_venue_map = {
        normalize_text(venue): (city, country)
        for venue, (city, country) in VENUE_LOCATION_MAP.items()
    }

    for index, row in df.iterrows():
        original_city = row["city"]
        venue = row["venue"]

        # -------------------------------------------------
        # Case 1: Original city is present
        # -------------------------------------------------
        if not is_blank(original_city):
            city_text = str(original_city).strip()
            normalized_city = normalize_text(city_text)

            df.at[index, "city_resolved"] = city_text

            country = city_country_lookup.get(normalized_city, "")

            if country:
                df.at[index, "country_resolved"] = country
                df.at[index, "city_resolution_status"] = "CITY_FROM_SOURCE"
            else:
                df.at[index, "country_resolved"] = ""
                df.at[index, "city_resolution_status"] = (
                    "CITY_PRESENT_COUNTRY_REVIEW"
                )

            continue

        # -------------------------------------------------
        # Case 2: City is blank; try explicit venue mapping
        # -------------------------------------------------
        normalized_venue = normalize_text(venue)

        if normalized_venue in normalized_venue_map:
            resolved_city, resolved_country = normalized_venue_map[
                normalized_venue
            ]

            df.at[index, "city_resolved"] = resolved_city
            df.at[index, "country_resolved"] = resolved_country
            df.at[index, "city_resolution_status"] = "RESOLVED_FROM_VENUE"

        # -------------------------------------------------
        # Case 3: City is blank and venue is not mapped
        # -------------------------------------------------
        else:
            df.at[index, "city_resolved"] = ""
            df.at[index, "country_resolved"] = ""
            df.at[index, "city_resolution_status"] = (
                "UNRESOLVED_BLANK_CITY"
            )

    return df


# ---------------------------------------------------------
# Print report
# ---------------------------------------------------------
def print_report(original_df, resolved_df):
    print("\n" + "=" * 70)
    print("BLANK-CITY RESOLUTION REPORT")
    print("=" * 70)

    print(f"Master dataset rows: {len(original_df):,}")
    print(
        "Original blank city rows:",
        original_df["city"].apply(is_blank).sum()
    )

    print("\nResolution status counts:")
    print(
        resolved_df["city_resolution_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    unresolved = resolved_df[
        resolved_df["city_resolution_status"] == "UNRESOLVED_BLANK_CITY"
    ]

    print("\n" + "-" * 70)
    print("UNRESOLVED BLANK-CITY VENUES")
    print("-" * 70)

    if unresolved.empty:
        print("None. All blank city values were resolved.")
    else:
        unresolved_venues = (
            unresolved.groupby("venue", dropna=False)
            .size()
            .reset_index(name="match_count")
            .sort_values("match_count", ascending=False)
        )

        print(unresolved_venues.to_string(index=False))

    print("\n" + "-" * 70)
    print("SAMPLE RESOLVED RECORDS")
    print("-" * 70)

    resolved_samples = resolved_df[
        resolved_df["city_resolution_status"] == "RESOLVED_FROM_VENUE"
    ]

    if resolved_samples.empty:
        print("No venue-based city resolutions were made.")
    else:
        columns_to_show = [
            column
            for column in [
                "date",
                "venue",
                "city",
                "city_resolved",
                "country_resolved",
                "city_resolution_status",
            ]
            if column in resolved_samples.columns
        ]

        print(
            resolved_samples[columns_to_show]
            .head(15)
            .to_string(index=False)
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    master, city_mapping = load_data()

    resolved_df = resolve_cities(master, city_mapping)

    # Save to a new file. Original master and mapping are untouched.
    resolved_df.to_csv(OUTPUT_PATH, index=False)

    print_report(master, resolved_df)

    print("\n" + "=" * 70)
    print("OUTPUT SAVED")
    print("=" * 70)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
