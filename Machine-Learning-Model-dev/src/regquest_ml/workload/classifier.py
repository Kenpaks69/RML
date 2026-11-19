"""
Monthly Workload Volume Forecasting and 4-Tier Operational Classifier.
Derived from daily Random Forest rollout aggregations.
"""

from typing import Dict, Any, List

WORKLOAD_THRESHOLDS = {
    "light_max": 300,
    "moderate_max": 650,
    "heavy_max": 900
}

WORKLOAD_DESCRIPTIONS = {
    "Light": "Low transaction volume; standard registrar staffing is sufficient.",
    "Moderate": "Normal academic operational load; standard queue distribution.",
    "Heavy": "High transaction volume; recommend expedited processing shifts.",
    "Very Heavy": "Peak volume (enrollment / graduation); recommend temporary counter expansion."
}


def classify_workload_tier(total_volume: float) -> str:
    """Classifies total monthly request volume into operational workload tier."""
    if total_volume < WORKLOAD_THRESHOLDS["light_max"]:
        return "Light"
    elif total_volume <= WORKLOAD_THRESHOLDS["moderate_max"]:
        return "Moderate"
    elif total_volume <= WORKLOAD_THRESHOLDS["heavy_max"]:
        return "Heavy"
    else:
        return "Very Heavy"


def calculate_monthly_workload(
    daily_predictions: List[Dict[str, Any]],
    target_month: str = "2026-11"
) -> Dict[str, Any]:
    """
    Sums daily request predictions across operating days to derive monthly workload.
    """
    total_predicted = sum(p["predicted_requests"] for p in daily_predictions)
    op_days = len(daily_predictions)
    daily_avg = round(total_predicted / max(1, op_days), 1)

    tier = classify_workload_tier(total_predicted)
    desc = WORKLOAD_DESCRIPTIONS.get(tier, "Normal operations.")

    # Capacity utilization based on nominal threshold of 1000 requests/month
    capacity_pct = round((total_predicted / 1000.0) * 100.0, 1)

    return {
        "forecast_month": target_month,
        "total_predicted_requests": int(round(total_predicted)),
        "operating_days_count": op_days,
        "average_daily_requests": daily_avg,
        "workload_classification": tier,
        "workload_description": desc,
        "capacity_utilization_pct": capacity_pct,
        "source": "ml_service"
    }
