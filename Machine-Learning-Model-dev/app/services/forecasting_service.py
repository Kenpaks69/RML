"""
Forecasting Domain Service.
Encapsulates daily demand autoregressive rollout and monthly workload aggregations.
"""

from datetime import datetime
import calendar
from typing import Dict, Any, List, Optional
from ..dependencies import ModelManager
from ..schemas.forecasting import DailyForecastItem


class ForecastingService:
    def __init__(self, model_manager: ModelManager):
        self.manager = model_manager

    def generate_operating_dates(self, start_date_str: str, num_days: int) -> List[str]:
        from datetime import timedelta
        cur = datetime.strptime(start_date_str, "%Y-%m-%d")
        dates = []
        while len(dates) < num_days:
            if cur.weekday() < 5:  # Monday to Friday
                dates.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
        return dates

    def forecast_daily_demand(
        self,
        start_date: str,
        number_of_operating_days: int = 20
    ) -> Dict[str, Any]:
        """
        Executes multi-step autoregressive rollout for future operating days.
        """
        if not self.manager.daily_forecaster:
            raise RuntimeError("Daily demand forecasting model artifact is not loaded.")

        operating_dates = self.generate_operating_dates(start_date, number_of_operating_days)
        recent_history = self.manager.metadata.get("recent_history_buffer", [35.0] * 15)

        forecast_items = self.manager.daily_forecaster.autoregressive_rollout(
            operating_dates=operating_dates,
            recent_history=recent_history
        )

        total_predicted = sum(item["predicted_requests"] for item in forecast_items)

        formatted_items = [
            DailyForecastItem(
                date=item["date"],
                weekday=item["weekday"],
                predicted_requests=item["predicted_requests"],
                lower_bound_90=item["lower_bound_90"],
                upper_bound_90=item["upper_bound_90"]
            )
            for item in forecast_items
        ]

        return {
            "forecast": formatted_items,
            "model_version": self.manager.daily_forecaster.model_version,
            "start_date": start_date,
            "total_operating_days": len(formatted_items),
            "total_predicted_requests": total_predicted
        }

    def forecast_monthly_workload(
        self,
        target_month: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Aggregates daily predictions over a target month to produce monthly workload tier.
        """
        if not self.manager.daily_forecaster:
            raise RuntimeError("Daily demand forecasting model artifact is not loaded.")

        if target_month:
            year, month = map(int, target_month.split("-"))
        else:
            now = datetime.now()
            if now.month == 12:
                year, month = now.year + 1, 1
            else:
                year, month = now.year, now.month + 1

        target_month_str = f"{year:04d}-{month:02d}"
        num_days_in_month = calendar.monthrange(year, month)[1]

        operating_dates = []
        for day in range(1, num_days_in_month + 1):
            dt = datetime(year, month, day)
            if dt.weekday() < 5:
                operating_dates.append(dt.strftime("%Y-%m-%d"))

        if not operating_dates:
            raise ValueError("No operating days found for target month.")

        recent_history = self.manager.metadata.get("recent_history_buffer", [35.0] * 15)

        forecast_items = self.manager.daily_forecaster.autoregressive_rollout(
            operating_dates=operating_dates,
            recent_history=recent_history
        )

        predicted_total = sum(item["predicted_requests"] for item in forecast_items)
        num_op_days = len(operating_dates)
        daily_avg = round(predicted_total / num_op_days, 1) if num_op_days > 0 else 0.0

        thresholds = self.manager.metadata.get("monthly_workload_thresholds", {
            "light_max": 500,
            "moderate_max": 850,
            "heavy_max": 1150
        })

        if predicted_total < thresholds.get("light_max", 500):
            level = "Light"
            desc = "Demand is anticipated to be below typical operational capacity."
        elif predicted_total <= thresholds.get("moderate_max", 850):
            level = "Moderate"
            desc = "Demand is in line with standard registrar operational capacity."
        elif predicted_total <= thresholds.get("heavy_max", 1150):
            level = "Heavy"
            desc = "High volume anticipated. Consider preparing additional verification personnel."
        else:
            level = "Very Heavy"
            desc = "Peak volume anticipated (e.g. enrollment/graduation surge). Maximum staffing recommended."

        explanation = (
            f"Forecasted {predicted_total} total credential requests across {num_op_days} operating days "
            f"(average {daily_avg} requests/day). Classified as '{level}' based on 2025 historical demand quartiles. {desc}"
        )

        return {
            "month": target_month_str,
            "predicted_total_requests": predicted_total,
            "operating_days": num_op_days,
            "average_daily_requests": daily_avg,
            "workload_level": level,
            "classification_basis": "historical_demand",
            "model_version": self.manager.daily_forecaster.model_version,
            "explanation": explanation
        }
