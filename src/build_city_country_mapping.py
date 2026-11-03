from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "odi_matches_master.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "city_country_mapping.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input dataset: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"city", "venue"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    # Extract distinct city-venue combinations.
    city_venue = (
        df[["city", "venue"]]
        .drop_duplicates()
        .sort_values(["city", "venue"], na_position="last")
        .reset_index(drop=True)
    )

    # Extract distinct city names.
    cities = (
        df[["city"]]
        .drop_duplicates()
        .sort_values("city", na_position="last")
        .reset_index(drop=True)
    )

    # Add blank review fields. Countries will be filled only after verification.
    cities["country"] = ""
    cities["mapping_status"] = "REVIEW_REQUIRED"
    cities["notes"] = ""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cities.to_csv(OUTPUT_FILE, index=False)

    print(f"Master dataset rows: {len(df):,}")
    print(f"Unique city-venue combinations: {len(city_venue):,}")
    print(f"Unique city values, including missing: {len(cities):,}")
    print(f"\nMapping file created:\n{OUTPUT_FILE}")

    print("\nPreview:")
    print(cities.head(30).to_string(index=False))

    print("\nMissing city rows:")
    print(df[df["city"].isna()][["date", "venue", "team_1", "team_2"]]
          .head(20)
          .to_string(index=False))


if __name__ == "__main__":
    main()