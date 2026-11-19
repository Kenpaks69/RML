"""
Data Validation, Date Anomaly Detection, and Quality Auditing for RegQuest ML.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional


MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}


def extract_month_from_filename(filename: str) -> Optional[int]:
    """Extracts month integer (1-12) from source filename."""
    lower = filename.lower()
    for m_str, m_num in MONTH_MAP.items():
        if m_str in lower:
            return m_num
    return None


def calculate_business_days(start_date: datetime, end_date: datetime) -> int:
    """Calculates number of operating business days (Mon-Fri) between two dates."""
    if start_date > end_date:
        return 0
    cur = start_date
    biz_days = 0
    while cur <= end_date:
        if cur.weekday() < 5:  # Monday to Friday
            biz_days += 1
        cur += timedelta(days=1)
    return biz_days


def is_date_anomaly(date_str: str) -> Tuple[bool, str]:
    """
    Checks if a raw date string is anomalous or requires deterministic repair.
    Returns (is_anomaly, reason).
    """
    if not date_str or not str(date_str).strip():
        return True, "empty_date"

    s = str(date_str).strip()

    # Excel epoch serial numbers (e.g. '45678')
    if re.fullmatch(r"^\d{5}$", s):
        return True, "excel_serial_number"

    # Day 00 anomaly (e.g. '2025-07-00')
    if re.search(r"-\d{2}-00$|/\d{2}/00$", s):
        return True, "day_zero"

    # Day > 31 anomaly (e.g. '2025-07-32')
    match_day = re.search(r"[/-](\d{1,2})$", s)
    if match_day and int(match_day.group(1)) > 31:
        return True, "day_out_of_bounds"

    # Repetitive format (e.g. '2025-07-07-2025')
    if len(s) > 10 and re.search(r"\d{4}.*\d{4}", s):
        return True, "repetitive_year_or_format"

    return False, "valid"
