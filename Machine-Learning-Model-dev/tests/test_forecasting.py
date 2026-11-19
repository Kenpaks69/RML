"""
Unit Tests for RegQuest ML Daily Demand Forecasting.
Tests feature engineering, candidate model training, autoregressive rollout, intervals, and API service.
"""

import unittest
import numpy as np
from pydantic import ValidationError
from src.regquest_ml.features.daily_demand import extract_daily_features
from src.regquest_ml.forecasting.predict import DailyDemandForecaster, FEATURE_KEYS
from app.schemas.forecasting import DailyForecastRequest, DailyForecastResponse
from app.services.forecasting_service import ForecastingService
from app.dependencies import ModelManager


class TestForecasting(unittest.TestCase):

    def setUp(self):
        self.manager = ModelManager.get_instance()
        self.service = ForecastingService(self.manager)

    def test_feature_engineering_lags_and_rolling(self):
        records = [
            {"date": f"2026-11-{i:02d}", "total_requests": float(10 + i * 2), "is_operating_day": 1}
            for i in range(1, 15)
        ]
        feats = extract_daily_features(records)
        self.assertEqual(len(feats), 14)

        # Check that lag_1 matches previous day's requests
        self.assertEqual(feats[1]["lag_1"], feats[0]["target_requests"])
        self.assertEqual(feats[2]["lag_1"], feats[1]["target_requests"])
        self.assertIn("rolling_mean_5", feats[5])
        self.assertIn("rolling_std_5", feats[5])

    def test_candidate_models_fit_and_predict(self):
        X = np.random.uniform(10, 50, size=(40, len(FEATURE_KEYS)))
        y = np.random.uniform(15, 60, size=(40,))

        for m_type in ["random_forest", "decision_tree", "linear_regression"]:
            forecaster = DailyDemandForecaster(model_type=m_type)
            forecaster.fit(X, y)
            preds = forecaster.predict(X[:5])
            self.assertEqual(len(preds), 5)
            self.assertTrue(all(p >= 0 for p in preds))

    def test_autoregressive_rollout_and_intervals(self):
        X = np.random.uniform(10, 50, size=(40, len(FEATURE_KEYS)))
        y = np.random.uniform(15, 60, size=(40,))

        forecaster = DailyDemandForecaster(model_type="random_forest")
        forecaster.fit(X, y)

        dates = ["2026-11-09", "2026-11-10", "2026-11-11"]
        recent = [30.0, 35.0, 40.0, 45.0, 50.0]
        rollout = forecaster.autoregressive_rollout(dates, recent)

        self.assertEqual(len(rollout), 3)
        for item in rollout:
            self.assertIn("predicted_requests", item)
            self.assertGreaterEqual(item["predicted_requests"], 0)
            self.assertLessEqual(item["lower_bound_90"], item["predicted_requests"])
            self.assertGreaterEqual(item["upper_bound_90"], item["predicted_requests"])

    def test_daily_forecast_service_and_schema(self):
        req = DailyForecastRequest(start_date="2026-11-09", number_of_operating_days=5)
        res = self.service.forecast_daily_demand(req.start_date, req.number_of_operating_days)
        resp = DailyForecastResponse(
            status="success",
            forecast=res["forecast"],
            model_version=res["model_version"],
            start_date=res["start_date"],
            total_operating_days=res["total_operating_days"],
            total_predicted_requests=res["total_predicted_requests"]
        )
        self.assertEqual(resp.status, "success")
        self.assertEqual(len(resp.forecast), 5)
        self.assertGreater(resp.total_predicted_requests, 0)

    def test_daily_forecast_request_validation(self):
        with self.assertRaises(ValidationError):
            DailyForecastRequest(start_date="invalid-date", number_of_operating_days=10)

        with self.assertRaises(ValidationError):
            DailyForecastRequest(start_date="2026-11-09", number_of_operating_days=100)  # Max 60


if __name__ == "__main__":
    unittest.main()
