"""
Academic and Operational Calendar Utilities for RegQuest ML.
"""

from datetime import datetime, timedelta
from typing import List, Set

# Academic operational calendar based on registrar seasonal demand cycles
PEAK_ACADEMIC_MONTHS: Set[int] = {1, 6, 8, 9}  # Jan (2nd sem), Jun (Graduation), Aug-Sep (1st sem / boards)
SLOW_ACADEMIC_MONTHS: Set[int] = {3, 12}       # Mar (break / finalization), Dec (holiday recess)


def is_peak_season(month: int) -> bool:
    """Returns True if given month falls in registrar peak season."""
    return month in PEAK_ACADEMIC_MONTHS


def is_slow_season(month: int) -> bool:
    """Returns True if given month falls in registrar slow recess season."""
    return month in SLOW_ACADEMIC_MONTHS


def generate_future_operating_dates(
    start_date_str: str,
    number_of_days: int = 5
) -> List[str]:
    """
    Generates a list of upcoming operating dates (skipping Saturdays & Sundays).
    """
    cur = datetime.strptime(start_date_str, "%Y-%m-%d")
    operating_dates = []
    while len(operating_dates) < number_of_days:
        if cur.weekday() < 5:  # Monday to Friday
            operating_dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return operating_dates
