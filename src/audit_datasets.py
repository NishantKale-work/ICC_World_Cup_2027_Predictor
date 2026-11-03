"""
Dataset audit for the ICC Cricket World Cup 2027 Predictor project.

This script:
- Scans ZIP archives in data/raw/
- Lists archive contents
- Audits CSV files for dimensions, columns, missing values, and date ranges
- Samples JSON cricket match files and reports their structure
- Saves audit outputs in reports/

It does not extract, modify, or delete the original ZIP files.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Limit the number of JSON files parsed from each archive.
# This keeps large archives, such as all_male_json.zip, manageable.
JSON_SAMPLE_SIZE = 3

# Candidate column names that may contain dates.
DATE_COLUMN_CANDIDATES = {
    "date",
    "match_date",
    "start_date",
    "start",
    "datetime",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def format_bytes(size: int) -> str:
    """Convert bytes into a readable file size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TB"


def get_archive_type(filename: str) -> str:
    """Return a simple file type label based on the archive member."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".csv":
        return "CSV"
    if suffix == ".json":
        return "JSON"
    if suffix in {".xlsx", ".xls"}:
        return "Excel"
    if suffix == ".txt":
        return "Text"

    return suffix.upper().replace(".", "") or "Unknown"


def identify_date_columns(columns: list[str]) -> list[str]:
    """Identify likely date columns from a list of column names."""
    date_columns = []

    for column in columns:
        normalized = str(column).strip().lower()

        if normalized in DATE_COLUMN_CANDIDATES:
            date_columns.append(column)
        elif "date" in normalized:
            date_columns.append(column)

    return date_columns


def inspect_json_structure(data: Any) -> dict[str, Any]:
    """Summarize the structure of a JSON object."""
    summary: dict[str, Any] = {
        "top_level_type": type(data).__name__,
    }

    if isinstance(data, dict):
        summary["top_level_keys"] = list(data.keys())

        info = data.get("info")
        if isinstance(info, dict):
            summary["info_keys"] = list(info.keys())

            teams = info.get("teams")
            if isinstance(teams, list):
                summary["teams"] = teams

            dates = info.get("dates")
            if dates is not None:
                summary["dates"] = dates

            outcome = info.get("outcome")
            if isinstance(outcome, dict):
                summary["outcome"] = outcome

            toss = info.get("toss")
            if isinstance(toss, dict):
                summary["toss"] = toss

            summary["match_type"] = info.get("match_type")
            summary["venue"] = info.get("venue")
            summary["event"] = info.get("event")

        innings = data.get("innings")
        if isinstance(innings, list):
            summary["innings_count"] = len(innings)

    elif isinstance(data, list):
        summary["list_length"] = len(data)

        if data:
            summary["first_item_type"] = type(data[0]).__name__

    return summary


# ---------------------------------------------------------------------
# CSV audit
# ---------------------------------------------------------------------

def audit_csv(
    archive: zipfile.ZipFile,
    member_name: str,
    archive_name: str,
) -> dict[str, Any]:
    """Read and audit a CSV file directly from a ZIP archive."""

    result: dict[str, Any] = {
        "archive": archive_name,
        "member": member_name,
        "type": "CSV",
    }

    try:
        with archive.open(member_name) as file:
            # Read the CSV into a DataFrame.
            dataframe = pd.read_csv(file, low_memory=False)

        result["rows"] = int(dataframe.shape[0])
        result["columns_count"] = int(dataframe.shape[1])
        result["columns"] = [str(column) for column in dataframe.columns]

        result["duplicate_rows"] = int(dataframe.duplicated().sum())

        missing_values = dataframe.isna().sum()
        result["columns_with_missing_values"] = {
            str(column): int(count)
            for column, count in missing_values.items()
            if count > 0
        }

        # Identify likely date columns and attempt date parsing.
        date_columns = identify_date_columns(list(dataframe.columns))
        date_results = {}

        for column in date_columns:
            parsed_dates = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

            valid_dates = parsed_dates.dropna()

            date_results[str(column)] = {
                "valid_date_count": int(valid_dates.shape[0]),
                "invalid_or_missing_date_count": int(
                    parsed_dates.isna().sum()
                ),
                "minimum_date": (
                    str(valid_dates.min().date())
                    if not valid_dates.empty
                    else None
                ),
                "maximum_date": (
                    str(valid_dates.max().date())
                    if not valid_dates.empty
                    else None
                ),
            }

        result["date_columns"] = date_results

        # Show a small preview without dumping the entire dataset.
        result["sample_rows"] = (
            dataframe.head(3)
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        )

    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"

    return result


# ---------------------------------------------------------------------
# JSON audit
# ---------------------------------------------------------------------

def audit_json_samples(
    archive: zipfile.ZipFile,
    json_members: list[str],
    archive_name: str,
) -> list[dict[str, Any]]:
    """Inspect a small sample of JSON files from an archive."""

    results = []

    sample_members = json_members[:JSON_SAMPLE_SIZE]

    for member_name in sample_members:
        result: dict[str, Any] = {
            "archive": archive_name,
            "member": member_name,
            "type": "JSON",
        }

        try:
            with archive.open(member_name) as file:
                data = json.load(file)

            result["structure"] = inspect_json_structure(data)

        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"

        results.append(result)

    return results


# ---------------------------------------------------------------------
# Main archive audit
# ---------------------------------------------------------------------

def audit_archive(zip_path: Path) -> dict[str, Any]:
    """Audit one ZIP archive."""

    archive_result: dict[str, Any] = {
        "archive": zip_path.name,
        "archive_size": format_bytes(zip_path.stat().st_size),
        "members": [],
        "file_type_counts": {},
        "csv_audits": [],
        "json_samples": [],
        "errors": [],
    }

    print(f"\n{'=' * 80}")
    print(f"Auditing: {zip_path.name}")
    print(f"Archive size: {archive_result['archive_size']}")
    print("=" * 80)

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            all_members = [
                member
                for member in archive.namelist()
                if not member.endswith("/")
            ]

            archive_result["member_count"] = len(all_members)
            archive_result["members"] = all_members

            file_type_counts: dict[str, int] = {}

            for member_name in all_members:
                file_type = get_archive_type(member_name)
                file_type_counts[file_type] = (
                    file_type_counts.get(file_type, 0) + 1
                )

            archive_result["file_type_counts"] = file_type_counts

            print(f"Files inside archive: {len(all_members)}")
            print(f"File types: {file_type_counts}")

            # Audit CSV files.
            csv_members = [
                member
                for member in all_members
                if member.lower().endswith(".csv")
            ]

            for member_name in csv_members:
                print(f"\n  Auditing CSV: {member_name}")

                csv_result = audit_csv(
                    archive=archive,
                    member_name=member_name,
                    archive_name=zip_path.name,
                )

                archive_result["csv_audits"].append(csv_result)

                if "error" in csv_result:
                    print(f"    ERROR: {csv_result['error']}")
                else:
                    print(
                        f"    Rows: {csv_result['rows']:,} | "
                        f"Columns: {csv_result['columns_count']}"
                    )
                    print(f"    Columns: {csv_result['columns']}")

                    if csv_result["date_columns"]:
                        print(
                            f"    Date information: "
                            f"{csv_result['date_columns']}"
                        )

            # Sample JSON files.
            json_members = [
                member
                for member in all_members
                if member.lower().endswith(".json")
            ]

            if json_members:
                print(
                    f"\n  Sampling {min(JSON_SAMPLE_SIZE, len(json_members))} "
                    f"of {len(json_members):,} JSON files..."
                )

                json_results = audit_json_samples(
                    archive=archive,
                    json_members=json_members,
                    archive_name=zip_path.name,
                )

                archive_result["json_samples"] = json_results

                for json_result in json_results:
                    print(f"\n    JSON file: {json_result['member']}")

                    if "error" in json_result:
                        print(f"    ERROR: {json_result['error']}")
                    else:
                        print(
                            "    Structure: "
                            f"{json_result['structure']}"
                        )

    except zipfile.BadZipFile:
        error_message = "This file is not a valid ZIP archive."
        archive_result["errors"].append(error_message)
        print(f"ERROR: {error_message}")

    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        archive_result["errors"].append(error_message)
        print(f"ERROR: {error_message}")

    return archive_result


# ---------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------

def save_reports(results: list[dict[str, Any]]) -> None:
    """Save the complete audit results as JSON and a readable text report."""

    json_report_path = REPORTS_DIR / "dataset_audit.json"
    text_report_path = REPORTS_DIR / "dataset_audit.txt"

    # Save machine-readable JSON report.
    with json_report_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False, default=str)

    # Save readable text report.
    with text_report_path.open("w", encoding="utf-8") as file:
        file.write("ODI WORLD CUP 2027 PREDICTOR — DATASET AUDIT\n")
        file.write("=" * 80 + "\n\n")

        for result in results:
            file.write(f"Archive: {result['archive']}\n")
            file.write(f"Size: {result['archive_size']}\n")
            file.write(
                f"Files inside archive: "
                f"{result.get('member_count', 'Unknown')}\n"
            )
            file.write(
                f"File types: {result.get('file_type_counts', {})}\n\n"
            )

            for csv_result in result.get("csv_audits", []):
                file.write(f"CSV: {csv_result['member']}\n")

                if "error" in csv_result:
                    file.write(f"  ERROR: {csv_result['error']}\n")
                else:
                    file.write(f"  Rows: {csv_result['rows']:,}\n")
                    file.write(
                        f"  Columns: {csv_result['columns_count']}\n"
                    )
                    file.write(
                        f"  Column names: {csv_result['columns']}\n"
                    )
                    file.write(
                        f"  Duplicate rows: "
                        f"{csv_result['duplicate_rows']:,}\n"
                    )
                    file.write(
                        f"  Missing values: "
                        f"{csv_result['columns_with_missing_values']}\n"
                    )
                    file.write(
                        f"  Date information: "
                        f"{csv_result['date_columns']}\n"
                    )

                file.write("\n")

            for json_result in result.get("json_samples", []):
                file.write(f"JSON sample: {json_result['member']}\n")

                if "error" in json_result:
                    file.write(f"  ERROR: {json_result['error']}\n")
                else:
                    file.write(
                        f"  Structure: {json_result['structure']}\n"
                    )

                file.write("\n")

            file.write("-" * 80 + "\n\n")

    print("\nAudit reports saved:")
    print(f"  JSON report: {json_report_path}")
    print(f"  Text report: {text_report_path}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    """Run the dataset audit."""

    print("ODI WORLD CUP 2027 PREDICTOR — DATASET AUDIT")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")

    if not RAW_DATA_DIR.exists():
        print("\nERROR: data/raw/ directory does not exist.")
        return

    zip_files = sorted(RAW_DATA_DIR.glob("*.zip"))

    if not zip_files:
        print("\nERROR: No ZIP files found in data/raw/.")
        print("Copy the dataset ZIP files into data/raw/ and run again.")
        return

    print(f"\nFound {len(zip_files)} ZIP archive(s).")

    results = []

    for zip_path in zip_files:
        archive_result = audit_archive(zip_path)
        results.append(archive_result)

    save_reports(results)

    print("\nAudit complete.")
    print(
        "Important: this is an initial audit only. "
        "We have not merged datasets or trained a model."
    )


if __name__ == "__main__":
    main()