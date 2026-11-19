"""
Unit Tests for RegQuest ML Monthly Workload Forecasting and Classification.
Tests workload threshold categorization, monthly aggregation, and API service.
"""

import unittest
from pydantic import ValidationError
from src.regquest_ml.workload.classifier import (
    classify_workload_tier,
    calculate_monthly_workload,
    WORKLOAD_THRESHOLDS,
    WORKLOAD_DESCRIPTIONS
)
from app.schemas.forecasting import MonthlyWorkloadRequest, MonthlyWorkloadResponse
from app.services.forecasting_service import ForecastingService
from app.dependencies import ModelManager


class TestWorkload(unittest.TestCase):

    def setUp(self):
        self.manager = ModelManager.get_instance()
        self.service = ForecastingService(self.manager)

    def test_classify_workload_tier(self):
        self.assertEqual(classify_workload_tier(250), "Light")
        self.assertEqual(classify_workload_tier(500), "Moderate")
        self.assertEqual(classify_workload_tier(800), "Heavy")
        self.assertEqual(classify_workload_tier(1200), "Very Heavy")

    def test_calculate_monthly_workload(self):
        dummy_daily = [
            {"date": f"2026-11-{i:02d}", "predicted_requests": 25}
            for i in range(1, 21)
        ]
        res = calculate_monthly_workload(dummy_daily, "2026-11")
        self.assertEqual(res["total_predicted_requests"], 500)
        self.assertEqual(res["operating_days_count"], 20)
        self.assertEqual(res["average_daily_requests"], 25.0)
        self.assertEqual(res["workload_classification"], "Moderate")
        self.assertIn("workload_description", res)

    def test_monthly_workload_service_and_schema(self):
        req = MonthlyWorkloadRequest(target_month="2026-12")
        res = self.service.forecast_monthly_workload(req.target_month)
        resp = MonthlyWorkloadResponse(
            status="success",
            month=res["month"],
            predicted_total_requests=res["predicted_total_requests"],
            operating_days=res["operating_days"],
            average_daily_requests=res["average_daily_requests"],
            workload_level=res["workload_level"],
            classification_basis=res["classification_basis"],
            model_version=res["model_version"],
            explanation=res["explanation"]
        )
        self.assertEqual(resp.status, "success")
        self.assertEqual(resp.month, "2026-12")
        self.assertGreater(resp.predicted_total_requests, 0)
        self.assertIn(resp.workload_level, ["Light", "Moderate", "Heavy", "Very Heavy"])

    def test_monthly_workload_validation(self):
        with self.assertRaises(ValidationError):
            MonthlyWorkloadRequest(target_month="2026/12")  # Invalid format

        with self.assertRaises(ValidationError):
            MonthlyWorkloadRequest(target_month="12-2026")  # Invalid format


if __name__ == "__main__":
    unittest.main()
