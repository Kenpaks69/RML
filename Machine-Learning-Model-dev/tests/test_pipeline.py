"""
Unit Tests for RegQuest ML Data Processing Pipeline.
Validates loading, cleaning, date standardization, credential mapping, and leak-free ordering.
"""

import os
import unittest
from datetime import datetime
from ML.src.pipeline import (
    clean_and_parse_date,
    parse_raw_credential,
    standardize_purpose,
    calculate_business_days,
    PROCESSED_DATA_DIR
)


class TestPipeline(unittest.TestCase):

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
        cleaned_csv = os.path.join(PROCESSED_DATA_DIR, "cleaned_requests.csv")
        daily_csv = os.path.join(PROCESSED_DATA_DIR, "daily_time_series.csv")
        self.assertTrue(os.path.exists(cleaned_csv), "cleaned_requests.csv must exist")
        self.assertTrue(os.path.exists(daily_csv), "daily_time_series.csv must exist")


if __name__ == "__main__":
    unittest.main()

