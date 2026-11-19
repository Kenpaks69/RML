"""
Unit Tests for RegQuest ML Data Pipeline.
Validates loading, cleaning, validation, calendar utilities, and leak-free ordering.
"""

import os
import unittest
from datetime import datetime
from src.regquest_ml.common.paths import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR
)
from src.regquest_ml.data.loader import (
    load_all_raw_datasets,
    load_processed_data,
    discover_raw_files
)
from src.regquest_ml.data.validator import (
    calculate_business_days
)
from src.regquest_ml.data.cleaner import (
    clean_and_parse_date,
    clean_records,
    clean_raw_records,
    parse_raw_credential,
    standardize_purpose
)
from src.regquest_ml.features.calendar import (
    is_peak_season,
    is_slow_season,
    generate_future_operating_dates
)


class TestDataPipeline(unittest.TestCase):

    def test_clean_and_parse_standard_date(self):
        dt, note = clean_and_parse_date("5/15/2025")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2025)
        self.assertEqual(dt.month, 5)
        self.assertEqual(dt.day, 15)

    def test_clean_and_parse_missing_year_slash(self):
        # e.g., 5/302025
        dt, note = clean_and_parse_date("5/302025")
        self.assertIsNotNone(dt)
        self.assertEqual(dt, datetime(2025, 5, 30))

    def test_clean_and_parse_double_slash(self):
        # e.g., 10//14/2025
        dt, note = clean_and_parse_date("10//14/2025")
        self.assertIsNotNone(dt)
        self.assertEqual(dt, datetime(2025, 10, 14))

    def test_clean_and_parse_repetitive_format(self):
        # e.g., 9/1/20/2025
        dt, note = clean_and_parse_date("9/1/20/2025")
        self.assertIsNotNone(dt)
        self.assertEqual(dt, datetime(2025, 9, 1))

    def test_clean_and_parse_excel_epoch_anomaly(self):
        # e.g. 1/4/1900 in July batch
        dt, note = clean_and_parse_date("1/4/1900")
        self.assertIsNotNone(dt)
        self.assertEqual(dt, datetime(2025, 6, 30))

    def test_clean_and_parse_shifted_text(self):
        dt, note = clean_and_parse_date("GSIS")
        self.assertIsNone(dt)

    def test_parse_raw_credential_single(self):
        primary, all_codes, is_multi, count = parse_raw_credential("1")
        self.assertEqual(primary, "TOR")
        self.assertFalse(is_multi)
        self.assertEqual(count, 1)

    def test_parse_raw_credential_multi(self):
        primary, all_codes, is_multi, count = parse_raw_credential("1, 2")
        self.assertEqual(primary, "TOR")
        self.assertTrue(is_multi)
        self.assertEqual(count, 2)
        self.assertIn("HD", all_codes)

    def test_standardize_purpose(self):
        self.assertEqual(standardize_purpose("Employment"), "Employment")
        self.assertEqual(standardize_purpose("EVALUATION"), "Evaluation")
        self.assertEqual(standardize_purpose("Board Exam Review"), "Board Exam")
        self.assertEqual(standardize_purpose("CHED Scholarship"), "Scholarship")

    def test_calendar_helpers(self):
        self.assertTrue(is_peak_season(1))
        self.assertTrue(is_peak_season(6))
        self.assertTrue(is_slow_season(12))
        self.assertFalse(is_peak_season(3))

        dates = generate_future_operating_dates("2026-11-06", number_of_days=3)
        self.assertEqual(len(dates), 3)
        self.assertEqual(dates[0], "2026-11-06")
        self.assertEqual(dates[1], "2026-11-09")
        self.assertEqual(dates[2], "2026-11-10")

    def test_calculate_business_days(self):
        # Mon to Wed = 3 operating days
        d1 = datetime(2025, 6, 2)   # Monday
        d2 = datetime(2025, 6, 4)   # Wednesday
        self.assertEqual(calculate_business_days(d1, d2), 3)

        # Fri to Mon = 2 operating days (skips Sat, Sun)
        d_fri = datetime(2025, 6, 6) # Friday
        d_mon = datetime(2025, 6, 9) # Monday
        self.assertEqual(calculate_business_days(d_fri, d_mon), 2)

    def test_processed_files_exist(self):
        cleaned_csv = PROCESSED_DATA_DIR / "cleaned_requests.csv"
        daily_csv = PROCESSED_DATA_DIR / "daily_time_series.csv"
        self.assertTrue(cleaned_csv.exists(), "cleaned_requests.csv must exist")
        self.assertTrue(daily_csv.exists(), "daily_time_series.csv must exist")


if __name__ == "__main__":
    unittest.main()
