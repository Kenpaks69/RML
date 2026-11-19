"""
Raw Dataset Ingestion and File Scanner for RegQuest ML.
Discovers and loads historical credential logbook CSV files.
"""

import os
import csv
from typing import List, Dict, Any, Tuple
from ..common.paths import RAW_DATA_DIR, ML_ROOT


def discover_raw_files(raw_dir: str = None) -> List[str]:
    """
    Finds all historical CSV files in data/raw/ or project directory.
    Sorted chronologically by month name where possible.
    """
    target_dir = raw_dir or str(RAW_DATA_DIR)
    if not os.path.exists(target_dir):
        # Fallback to ML/data/raw
        target_dir = os.path.join(str(ML_ROOT), "data", "raw")
        if not os.path.exists(target_dir):
            return []

    month_order = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    csv_files = []
    for f in os.listdir(target_dir):
        if f.lower().endswith(".csv") and not f.startswith("cleaned_") and not f.startswith("daily_"):
            csv_files.append(os.path.join(target_dir, f))

    def sort_key(filepath: str) -> Tuple[int, str]:
        fname = os.path.basename(filepath).lower()
        for idx, m in enumerate(month_order):
            if m in fname:
                return (idx, fname)
        return (99, fname)

    return sorted(csv_files, key=sort_key)


def load_raw_csv_records(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a single monthly CSV registry into a list of row dictionaries.
    Handles encodings and strips whitespace from headers.
    """
    records = []
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as fp:
                reader = csv.reader(fp)
                header = next(reader, None)
                if not header:
                    return []
                clean_header = [col.strip() for col in header]

                for row in reader:
                    if not any(cell.strip() for cell in row):
                        continue
                    row_dict = {}
                    for i, col_name in enumerate(clean_header):
                        val = row[i].strip() if i < len(row) else ""
                        row_dict[col_name] = val
                    row_dict["_source_file"] = os.path.basename(file_path)
                    records.append(row_dict)
            break
        except Exception:
            continue

    return records


def load_all_raw_datasets(raw_dir: str = None) -> List[Dict[str, Any]]:
    """
    Loads all discovered monthly raw datasets into a single record list.
    """
    files = discover_raw_files(raw_dir)
    all_records = []
    for fp in files:
        rows = load_raw_csv_records(fp)
        all_records.extend(rows)
    return all_records


def load_raw_csv_files(raw_dir: str = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Loads all raw csv records and returns (records, file_paths).
    """
    files = discover_raw_files(raw_dir)
    records = []
    for fp in files:
        records.extend(load_raw_csv_records(fp))
    return records, files


def load_processed_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads cleaned records or daily time series from processed CSV file.
    """
    records = []
    if not os.path.exists(file_path):
        return records
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            records.append(dict(row))
    return records

