"""
Unit Tests for RegQuest ML Processing Time Estimation.
Tests predictor logic, document multiplier rules, baseline training, and API integration.
"""

import unittest
from datetime import datetime
from src.regquest_ml.processing_time.predict import (
    ProcessingTimePredictor,
    ADMINISTRATIVE_STANDARD_DAYS,
    add_business_days
)
from src.regquest_ml.processing_time.train import train_processing_time_baseline
from app.schemas.processing_time import ProcessingTimeRequest, ProcessingTimeResponse
from app.services.processing_service import ProcessingService
from app.dependencies import ModelManager


class TestProcessingTime(unittest.TestCase):

    def setUp(self):
        self.predictor = ProcessingTimePredictor()
        self.manager = ModelManager.get_instance()
        self.service = ProcessingService(self.manager)

    def test_predictor_default_sla(self):
        res = self.predictor.predict_one("TOR", "Employment")
        self.assertIn("predicted_processing_time", res)
        self.assertGreater(res["predicted_processing_time"], 0)
        self.assertEqual(res["unit"], "operating_days")
        self.assertFalse(res["is_empirical"])
        self.assertIn("disclaimer", res)

    def test_predictor_multi_document_scaling(self):
        single = self.predictor.predict_one("HD", num_docs=1)
        multi = self.predictor.predict_one("HD", num_docs=3)
        self.assertGreater(multi["predicted_processing_time"], single["predicted_processing_time"])

    def test_add_business_days(self):
        # Starting Friday 2026-11-06, adding 1 business day should be Monday 2026-11-09
        start = datetime(2026, 11, 6)
        res = add_business_days(start, 1)
        self.assertEqual(res, datetime(2026, 11, 9))

    def test_train_baseline_with_dummy_records(self):
        dummy_records = [
            {"primary_credential": "TOR", "scheduled_lead_business_days": 5.0} for _ in range(12)
        ] + [
            {"primary_credential": "COE", "scheduled_lead_business_days": 2.0} for _ in range(12)
        ]
        predictor = train_processing_time_baseline(dummy_records)
        self.assertIn("TOR", predictor.standards)
        self.assertEqual(predictor.standards["TOR"], 5.0)

    def test_processing_service_and_schema(self):
        req = ProcessingTimeRequest(credential_type="TOR", purpose="Employment", num_documents=2)
        result = self.service.predict_processing_time(
            credential_type=req.credential_type,
            purpose=req.purpose,
            num_documents=req.num_documents
        )
        resp = ProcessingTimeResponse(
            status="success",
            predicted_processing_time=result["predicted_processing_time"],
            unit=result["unit"],
            model_version=result["model_version"],
            basis=result["basis"],
            is_empirical=result["is_empirical"],
            disclaimer=result.get("disclaimer")
        )
        self.assertEqual(resp.status, "success")
        self.assertGreater(resp.predicted_processing_time, 0)


if __name__ == "__main__":
    unittest.main()
