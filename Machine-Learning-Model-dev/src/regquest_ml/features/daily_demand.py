"""
Autoregressive Lag and Rolling Statistical Feature Pipeline for Daily Demand.
Guarantees strictly backward-looking features to prevent data leakage.
"""

import math
from datetime import datetime
from typing import List, Dict, Any
from .calendar import is_peak_season, is_slow_season
from ..data.cleaner import aggregate_daily_time_series


def extract_daily_features(
    records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Computes calendar, seasonal, and strictly backward-looking autoregressive
    lag/rolling features from chronological daily records.
    """
    sorted_records = sorted(records, key=lambda x: x["date"])
    featured_records = []
    history = []

    for i, rec in enumerate(sorted_records):
        dt = datetime.strptime(rec["date"], "%Y-%m-%d")
        total_req = float(rec["total_requests"])

        day_of_week = dt.weekday()
        month = dt.month
        day_of_month = dt.day

        # Calendar features
        is_peak = int(is_peak_season(month))
        is_slow = int(is_slow_season(month))
        is_month_start = int(day_of_month <= 5)
        is_month_end = int(day_of_month >= 25)

        # Autoregressive Lags (prior history only)
        lag_1 = history[-1] if len(history) >= 1 else total_req
        lag_2 = history[-2] if len(history) >= 2 else lag_1
        lag_3 = history[-3] if len(history) >= 3 else lag_2
        lag_5 = history[-5] if len(history) >= 5 else lag_1
        lag_10 = history[-10] if len(history) >= 10 else lag_5

        # Rolling Statistics (strictly backward-looking)
        w5 = history[-5:] if len(history) >= 5 else (history if history else [total_req])
        w10 = history[-10:] if len(history) >= 10 else (history if history else [total_req])

        mean_5 = sum(w5) / len(w5)
        mean_10 = sum(w10) / len(w10)
        var_5 = sum((x - mean_5) ** 2 for x in w5) / len(w5) if len(w5) > 1 else 0.0
        std_5 = math.sqrt(var_5)
        min_5 = min(w5)
        max_5 = max(w5)

        feat_dict = {
            "date": rec["date"],
            "target_requests": total_req,
            "day_of_week": day_of_week,
            "month": month,
            "day_of_month": day_of_month,
            "is_operating_day": int(rec.get("is_operating_day", 1)),
            "is_weekend": int(rec.get("is_weekend", 0)),
            "is_peak_academic_season": is_peak,
            "is_slow_academic_season": is_slow,
            "is_month_start": is_month_start,
            "is_month_end": is_month_end,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_5": lag_5,
            "lag_10": lag_10,
            "rolling_mean_5": mean_5,
            "rolling_mean_10": mean_10,
            "rolling_std_5": std_5,
            "rolling_min_5": min_5,
            "rolling_max_5": max_5
        }

        featured_records.append(feat_dict)
        history.append(total_req)

    return featured_records


build_daily_features_table = extract_daily_features
