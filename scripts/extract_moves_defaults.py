#!/usr/bin/env python3
"""
Extract default data tables from the EPA MOVES SQL dump.

This script parses the MariaDB dump file from the EPA MOVES GitHub repository
and exports selected tables to CSV files for use in NeuralMOVES aggregation layers.

Usage:
    python scripts/extract_moves_defaults.py [--sql-dump path/to/movesdb.sql]

If no SQL dump is provided, the script will look for movesdb.zip in the current
directory or attempt to download it from the EPA GitHub repo.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# ── Configuration ──────────────────────────────────────────────────────────────

EPA_MOVES_DB_URL = (
    "https://github.com/USEPA/EPA_MOVES_Model/raw/master/"
    "database/Setup/movesdb20241112.zip"
)

# Tables to extract: table_name -> (column_names, output_filename)
TABLES = {
    "sourcetypeagedistribution": {
        "columns": ["sourceTypeID", "yearID", "ageID", "ageFraction"],
        "output": "age_distribution.csv",
    },
    "avft": {
        "columns": [
            "sourceTypeID", "modelYearID", "fuelTypeID",
            "engTechID", "fuelEngFraction",
        ],
        "output": "avft.csv",
    },
    # sampleVehiclePopulation has AVFT-equivalent fuel/eng fractions
    # (the avft table is typically empty in defaults; this table has the data)
    "samplevehiclepopulation": {
        "columns": [
            "sourceTypeModelYearID", "sourceTypeID", "modelYearID",
            "fuelTypeID", "engTechID", "regClassID",
            "stmyFuelEngFraction", "stmyFraction",
        ],
        "output": "sample_vehicle_population.csv",
    },
    "sourcetypeyear": {
        "columns": [
            "yearID", "sourceTypeID", "salesGrowthFactor",
            "sourceTypePopulation", "migrationrate",
        ],
        "output": "source_type_population.csv",
    },
    "sourcetypeyearvmt": {
        "columns": ["yearID", "sourceTypeID", "VMT"],
        "output": "source_type_year_vmt.csv",
    },
    "monthvmtfraction": {
        "columns": ["sourceTypeID", "monthID", "monthVMTFraction"],
        "output": "month_vmt_fraction.csv",
    },
    "dayvmtfraction": {
        "columns": [
            "sourceTypeID", "monthID", "roadTypeID",
            "dayID", "dayVMTFraction",
        ],
        "output": "day_vmt_fraction.csv",
    },
    "hourvmtfraction": {
        "columns": [
            "sourceTypeID", "roadTypeID", "dayID",
            "hourID", "hourVMTFraction",
        ],
        "output": "hour_vmt_fraction.csv",
    },
    "hpmsvtypeyear": {
        "columns": [
            "HPMSVtypeID", "yearID", "VMTGrowthFactor", "HPMSBaseYearVMT",
        ],
        "output": "hpms_vtype_year.csv",
    },
    "zonemonthhour": {
        "columns": [
            "monthID", "zoneID", "hourID", "temperature",
            "relHumidity", "heatIndex", "specificHumidity", "molWaterFraction",
        ],
        "output": "zone_month_hour.csv",
    },
    "avgspeeddistribution": {
        "columns": [
            "sourceTypeID", "roadTypeID", "hourDayID",
            "avgSpeedBinID", "avgSpeedFraction",
        ],
        "output": "avg_speed_distribution.csv",
    },
    "emissionratebyage": {
        "columns": [
            "sourceBinID", "polProcessID", "opModeID", "ageGroupID",
            "meanBaseRate", "meanBaseRateCV", "meanBaseRateIM",
            "meanBaseRateIMCV", "dataSourceId",
        ],
        "output": "emission_rate_by_age.csv",
    },
}


# ── SQL Parsing ────────────────────────────────────────────────────────────────

# Pattern to match: INSERT INTO `tablename` VALUES
INSERT_RE = re.compile(
    r"INSERT INTO `(\w+)` VALUES\s*", re.IGNORECASE
)

# Pattern to parse individual value tuples: (v1,v2,...),
# Handles NULL, numbers, and quoted strings
VALUE_TUPLE_RE = re.compile(
    r"\(([^)]*)\)"
)


def parse_value(v: str) -> str:
    """Parse a single SQL value into a string for CSV output."""
    v = v.strip()
    if v.upper() == "NULL":
        return ""
    # Remove surrounding quotes
    if (v.startswith("'") and v.endswith("'")) or \
       (v.startswith('"') and v.endswith('"')):
        return v[1:-1]
    return v


def extract_tables(sql_stream, target_tables: dict[str, dict]) -> dict[str, list]:
    """
    Parse a SQL dump stream and extract rows for target tables.

    Parameters
    ----------
    sql_stream : text stream
        The SQL dump file opened for reading
    target_tables : dict
        Mapping of lowercase table name -> config dict

    Returns
    -------
    dict[str, list[list[str]]]
        Mapping of table name -> list of row value lists
    """
    results = {name: [] for name in target_tables}
    target_names = set(target_tables.keys())

    current_table = None
    buffer = ""

    for line in sql_stream:
        # Check for INSERT INTO statement
        m = INSERT_RE.match(line)
        if m:
            table_name = m.group(1).lower()
            if table_name in target_names:
                current_table = table_name
                # The values start after "VALUES" on this line
                rest = line[m.end():]
                buffer = rest
            else:
                current_table = None
                buffer = ""
            continue

        if current_table is None:
            continue

        # Accumulate lines for current INSERT
        buffer += line

        # Check if this line ends the INSERT (contains a semicolon)
        if ";" in line:
            # Parse all value tuples from the buffer
            for match in VALUE_TUPLE_RE.finditer(buffer):
                raw = match.group(1)
                # Split by comma, but handle quoted strings
                values = _split_sql_values(raw)
                parsed = [parse_value(v) for v in values]
                results[current_table].append(parsed)
            current_table = None
            buffer = ""

    return results


def _split_sql_values(raw: str) -> list[str]:
    """Split SQL values string by comma, respecting quotes."""
    values = []
    current = []
    in_quote = False
    quote_char = None

    for ch in raw:
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
        elif ch in ("'", '"'):
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch == ",":
            values.append("".join(current))
            current = []
        else:
            current.append(ch)

    if current:
        values.append("".join(current))

    return values


# ── Aggregation for county meteorology ─────────────────────────────────────────

def aggregate_county_meteorology(zone_rows: list[list[str]]) -> list[list[str]]:
    """
    Aggregate zoneMonthHour data to county-level monthly averages.

    zoneMonthHour has: monthID, zoneID, hourID, temperature, relHumidity, ...
    We want: countyID, monthID, avg_temperature_F, avg_relHumidity

    In MOVES, zoneID is typically countyID * 10 + zoneIndex.
    We average across hours for each county-month combination.
    """
    from collections import defaultdict

    # Group by (zoneID, monthID) and average temp/humidity across hours
    groups = defaultdict(lambda: {"temp_sum": 0.0, "humid_sum": 0.0, "count": 0})

    for row in zone_rows:
        if len(row) < 5:
            continue
        try:
            month_id = int(row[0])
            zone_id = int(row[1])
            temp = float(row[3]) if row[3] else None
            humid = float(row[4]) if row[4] else None
        except (ValueError, IndexError):
            continue

        if temp is not None and humid is not None:
            # Derive county from zone: countyID = zoneID // 10
            county_id = zone_id // 10
            key = (county_id, month_id)
            groups[key]["temp_sum"] += temp
            groups[key]["humid_sum"] += humid
            groups[key]["count"] += 1

    # Produce averaged rows
    result = []
    for (county_id, month_id), agg in sorted(groups.items()):
        if agg["count"] > 0:
            avg_temp = agg["temp_sum"] / agg["count"]
            avg_humid = agg["humid_sum"] / agg["count"]
            result.append([str(county_id), str(month_id),
                          f"{avg_temp:.4f}", f"{avg_humid:.4f}"])

    return result


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_fraction_sums(rows, columns, group_cols, fraction_col,
                           expected_sum=1.0, tolerance=0.01, table_name=""):
    """Validate that fraction columns sum correctly within groups."""
    from collections import defaultdict

    col_indices = {c: i for i, c in enumerate(columns)}
    group_indices = [col_indices[c] for c in group_cols]
    frac_index = col_indices[fraction_col]

    groups = defaultdict(float)
    for row in rows:
        try:
            key = tuple(row[i] for i in group_indices)
            groups[key] += float(row[frac_index])
        except (ValueError, IndexError):
            continue

    violations = 0
    for key, total in groups.items():
        if abs(total - expected_sum) > tolerance:
            violations += 1
            if violations <= 5:
                print(f"  WARNING [{table_name}] group {key}: "
                      f"sum={total:.6f}, expected={expected_sum}")

    if violations > 5:
        print(f"  ... and {violations - 5} more violations")
    elif violations == 0:
        print(f"  OK: all {len(groups)} groups sum to ~{expected_sum}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract MOVES default data from SQL dump"
    )
    parser.add_argument(
        "--sql-dump",
        type=str,
        default=None,
        help="Path to the unzipped .sql dump file",
    )
    parser.add_argument(
        "--zip-file",
        type=str,
        default=None,
        help="Path to movesdb*.zip file (will be unzipped automatically)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (default: src/neuralmoves/defaults/)",
    )
    args = parser.parse_args()

    # Determine output directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / "src" / "neuralmoves" / "defaults"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find or download the SQL dump
    sql_path = None
    if args.sql_dump:
        sql_path = Path(args.sql_dump)
    elif args.zip_file:
        zip_path = Path(args.zip_file)
    else:
        # Look for zip in common locations
        candidates = [
            Path("movesdb.zip"),
            repo_root / "movesdb.zip",
            Path("movesdb20241112.zip"),
        ]
        zip_path = None
        for c in candidates:
            if c.exists():
                zip_path = c
                break

        if zip_path is None:
            print("MOVES database dump not found locally.")
            print(f"Downloading from: {EPA_MOVES_DB_URL}")
            zip_path = Path("movesdb.zip")
            urlretrieve(EPA_MOVES_DB_URL, str(zip_path))
            print(f"Downloaded to: {zip_path}")

    # Open the SQL dump
    if sql_path and sql_path.exists():
        print(f"Reading SQL dump: {sql_path}")
        f = open(sql_path, "r", encoding="utf-8", errors="replace")
    else:
        print(f"Extracting from zip: {zip_path}")
        zf = zipfile.ZipFile(str(zip_path), "r")
        names = zf.namelist()
        sql_name = [n for n in names if n.endswith(".sql")][0]
        print(f"  SQL file inside zip: {sql_name}")
        raw = zf.open(sql_name)
        f = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")

    # Parse the SQL dump
    print(f"\nParsing {len(TABLES)} tables from SQL dump...")
    print("This may take a few minutes for the 390 MB file.\n")

    results = extract_tables(f, TABLES)
    f.close()

    # Write CSV files
    for table_name, config in TABLES.items():
        rows = results[table_name]
        columns = config["columns"]
        output_file = output_dir / config["output"]

        # Special handling for zoneMonthHour → county_meteorology
        if table_name == "zonemonthhour":
            print(f"[{table_name}] {len(rows)} raw rows → aggregating to county-monthly...")
            county_rows = aggregate_county_meteorology(rows)
            county_columns = ["countyID", "monthID", "temperature_F", "relHumidity_pct"]
            county_output = output_dir / "county_meteorology.csv"
            with open(county_output, "w", newline="") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(county_columns)
                writer.writerows(county_rows)
            print(f"  → {county_output} ({len(county_rows)} rows)")
            # Also write raw zone data (skip if too large)
            if len(rows) > 0:
                with open(output_file, "w", newline="") as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(columns)
                    writer.writerows(rows)
                print(f"  → {output_file} ({len(rows)} rows, raw)")
            continue

        if len(rows) == 0:
            print(f"[{table_name}] WARNING: No data found!")
            continue

        print(f"[{table_name}] {len(rows)} rows extracted")

        # Validate column count
        expected_cols = len(columns)
        mismatches = sum(1 for r in rows if len(r) != expected_cols)
        if mismatches > 0:
            print(f"  WARNING: {mismatches} rows have wrong column count "
                  f"(expected {expected_cols})")

        # Write CSV
        with open(output_file, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"  → {output_file}")

    # Run validations
    print("\n── Validating fraction sums ──")

    if results["sourcetypeagedistribution"]:
        validate_fraction_sums(
            results["sourcetypeagedistribution"],
            TABLES["sourcetypeagedistribution"]["columns"],
            group_cols=["sourceTypeID", "yearID"],
            fraction_col="ageFraction",
            table_name="sourceTypeAgeDistribution",
        )

    if results["monthvmtfraction"]:
        validate_fraction_sums(
            results["monthvmtfraction"],
            TABLES["monthvmtfraction"]["columns"],
            group_cols=["sourceTypeID"],
            fraction_col="monthVMTFraction",
            table_name="monthVMTFraction",
        )

    if results["dayvmtfraction"]:
        validate_fraction_sums(
            results["dayvmtfraction"],
            TABLES["dayvmtfraction"]["columns"],
            group_cols=["sourceTypeID", "monthID", "roadTypeID"],
            fraction_col="dayVMTFraction",
            table_name="dayVMTFraction",
        )

    if results["hourvmtfraction"]:
        validate_fraction_sums(
            results["hourvmtfraction"],
            TABLES["hourvmtfraction"]["columns"],
            group_cols=["sourceTypeID", "roadTypeID", "dayID"],
            fraction_col="hourVMTFraction",
            table_name="hourVMTFraction",
        )

    print("\n── Summary ──")
    for table_name, config in TABLES.items():
        n = len(results[table_name])
        out = config["output"]
        status = "OK" if n > 0 else "EMPTY"
        print(f"  {out:40s} {n:>10,} rows  [{status}]")

    print(f"\nOutput directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
