"""
Feature Engineering Pipelines for RegQuest ML Capabilities.
Supports:
1. Daily time-series feature engineering with strictly backward-looking lag/rolling features.
2. Individual request feature engineering for processing duration turnaround.
3. Operating calendar sequence generation for multi-day forecasting horizons.
"""

import os
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Academic calendar indicators based on university operational cycles
PEAK_ACADEMIC_MONTHS = {1, 6, 8, 9}  # Jan (2nd sem start), Jun (Graduation), Aug-Sep (1st sem / board exams)
SLOW_ACADEMIC_MONTHS = {3, 12}       # Mar (break / finalization), Dec (holiday recess)


def extract_daily_features(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes calendar, seasonal, and strictly backward-looking autoregressive lag/rolling features
    from a list of daily time-series records sorted chronologically.
    Prevents future-to-past data leakage.
    """
    sorted_records = sorted(records, key=lambda x: x["date"])
    
    featured_records = []
    
    # Track historical sequence of total_requests
    history = []
    
    for i, rec in enumerate(sorted_records):
        dt = datetime.strptime(rec["date"], "%Y-%m-%d")
        total_req = float(rec["total_requests"])
        
        # Calendar & Seasonality
        day_of_week = dt.weekday()
        month = dt.month
        day_of_month = dt.day
        
        is_peak = int(month in PEAK_ACADEMIC_MONTHS)
        is_slow = int(month in SLOW_ACADEMIC_MONTHS)
        is_month_start = int(day_of_month <= 5)
        is_month_end = int(day_of_month >= 25)
        
        # Backward-looking lags
        lag_1 = history[-1] if len(history) >= 1 else total_req
        lag_2 = history[-2] if len(history) >= 2 else lag_1
        lag_3 = history[-3] if len(history) >= 3 else lag_2
        lag_5 = history[-5] if len(history) >= 5 else lag_1
        lag_10 = history[-10] if len(history) >= 10 else lag_5
        
        # Rolling windows (last 5 and 10 operating days)
        w5 = history[-5:] if len(history) >= 5 else (history if history else [total_req])
        w10 = history[-10:] if len(history) >= 10 else (history if history else [total_req])
        
        mean_5 = sum(w5) / len(w5)
        mean_10 = sum(w10) / len(w10)
        
        var_5 = sum((x - mean_5) ** 2 for x in w5) / len(w5)
        std_5 = math.sqrt(var_5)
        
        min_5 = min(w5)
        max_5 = max(w5)
        
        feat = {
            "date": rec["date"],
            "target_requests": total_req,
            "day_of_week": day_of_week,
            "month": month,
            "day_of_month": day_of_month,
            "is_operating_day": int(rec.get("is_operating_day", 1 if day_of_week < 5 else 0)),
            "is_weekend": int(rec.get("is_weekend", 1 if day_of_week >= 5 else 0)),
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
        
        featured_records.append(feat)
        history.append(total_req)
        
    return featured_records


def generate_future_operating_dates(start_date_str: str, num_operating_days: int) -> List[str]:
    """
    Generates a list of upcoming operating dates (Monday-Friday) starting on or after start_date.
    """
    dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    operating_dates = []
    cur = dt
    
    while len(operating_dates) < num_operating_days:
        if cur.weekday() < 5:  # Monday to Friday
            operating_dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
        
    return operating_dates


def get_request_features(credential_type: str, purpose: str, date_requested_str: str, num_documents: int = 1) -> Dict[str, Any]:
    """
    Extracts features for an individual document request turnaround prediction.
    Only uses attributes known at request submission time (prevents data leakage).
    """
    dt = datetime.strptime(date_requested_str, "%Y-%m-%d") if date_requested_str else datetime.now()
    
    cred = (credential_type or "TOR").strip().upper()
    purp = (purpose or "Employment").strip()
    
    return {
        "credential_type": cred,
        "purpose": purp,
        "is_multi_document": int(num_documents > 1),
        "num_documents": max(1, int(num_documents)),
        "day_of_week": dt.weekday(),
        "month": dt.month,
        "is_peak_academic_season": int(dt.month in PEAK_ACADEMIC_MONTHS)
    }

