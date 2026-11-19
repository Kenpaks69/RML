"""
API Integration and Schema Validation Tests for RegQuest ML Microservice.
Tests all request/response models, endpoint functions, validations, and edge cases.
"""

import unittest
from pydantic import ValidationError
from ML.app.dependencies import ModelManager
from ML.app.schemas import (
    ProcessingTimeRequest,
    ProcessingTimeResponse,
    DailyForecastRequest,
    DailyForecastResponse,
    MonthlyWorkloadRequest,
    MonthlyWorkloadResponse,
    HealthCheckResponse
)
from ML.app.routers.health import health_check
from ML.app.routers.processing_time import predict_processing_time
from ML.app.routers.daily_forecast import forecast_daily_demand
from ML.app.routers.monthly_workload import forecast_monthly_workload


class TestMLAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manager = ModelManager.get_instance()

    def test_health_check(self):
        resp = health_check(self.manager)
        self.assertIsInstance(resp, HealthCheckResponse)
        self.assertEqual(resp.status, "ok")
        self.assertTrue(resp.model_availability["daily_forecaster"])
        self.assertTrue(resp.model_availability["processing_baseline"])
        self.assertIn("daily_forecaster", resp.model_versions)

    def test_predict_processing_time_tor(self):
        req = ProcessingTimeRequest(
            credential_type="TOR",
            purpose="Employment",
            date_requested="2026-11-05",
            num_documents=1
        )
        resp = predict_processing_time(req, self.manager)
        self.assertIsInstance(resp, ProcessingTimeResponse)
        self.assertEqual(resp.status, "success")
        self.assertGreater(resp.predicted_processing_time, 0)
        self.assertEqual(resp.unit, "operating_days")
        self.assertFalse(resp.is_empirical)
        self.assertIsNotNone(resp.disclaimer)

    def test_predict_processing_time_multi_doc(self):
        req_single = ProcessingTimeRequest(credential_type="HD", num_documents=1)
        resp_single = predict_processing_time(req_single, self.manager)

        req_multi = ProcessingTimeRequest(credential_type="HD", num_documents=2)
        resp_multi = predict_processing_time(req_multi, self.manager)

        self.assertGreater(resp_multi.predicted_processing_time, resp_single.predicted_processing_time)

    def test_forecast_daily_demand(self):
        req = DailyForecastRequest(
            start_date="2026-11-09",
            number_of_operating_days=10
        )
        resp = forecast_daily_demand(req, self.manager)
        self.assertIsInstance(resp, DailyForecastResponse)
        self.assertEqual(resp.status, "success")
        self.assertEqual(len(resp.forecast), 10)
        self.assertGreater(resp.total_predicted_requests, 0)

        for item in resp.forecast:
            self.assertGreaterEqual(item.predicted_requests, 0)
            self.assertLessEqual(item.lower_bound_90, item.predicted_requests)
            self.assertGreaterEqual(item.upper_bound_90, item.predicted_requests)

    def test_forecast_monthly_workload(self):
        req = MonthlyWorkloadRequest(target_month="2026-12")
        resp = forecast_monthly_workload(req, self.manager)
        self.assertIsInstance(resp, MonthlyWorkloadResponse)
        self.assertEqual(resp.status, "success")
        self.assertEqual(resp.month, "2026-12")
        self.assertGreater(resp.predicted_total_requests, 0)
        self.assertGreater(resp.operating_days, 15)
        self.assertIn(resp.workload_level, ["Light", "Moderate", "Heavy", "Very Heavy"])
        self.assertEqual(resp.classification_basis, "historical_demand")
        self.assertIn("explanation", resp.model_dump())

    def test_invalid_date_format_handling(self):
        with self.assertRaises(ValidationError):
            ProcessingTimeRequest(
                credential_type="TOR",
                date_requested="05-11-2026"  # Bad format
            )

    def test_invalid_operating_days_range(self):
        with self.assertRaises(ValidationError):
            DailyForecastRequest(
                start_date="2026-11-09",
                number_of_operating_days=150  # Exceeds max 60
            )

    def test_invalid_monthly_format(self):
        with self.assertRaises(ValidationError):
            MonthlyWorkloadRequest(
                target_month="2026/12"  # Bad format
            )


if __name__ == "__main__":
    unittest.main()
