# RegQuest Machine Learning Microservice

A production-ready Machine Learning package and microservice for the **RegQuest** Academic Credential Request Management System.

---

## 1. Capabilities Overview

### A. Document Processing Time Prediction (`/predict/processing-time`)
- **Objective:** Estimates the processing turnaround duration (in operating business days) of an individual credential request.
- **Features:** Credential Type (`TOR`, `HD`, `CAV`, `COE`, `CAR`, etc.), Request Purpose, Submission Date, Multi-Document Package Flag, Number of Documents.
- **Model:** Defensible administrative standard and scheduled duration baseline (`processing-time-baseline-v1`).
- **Data Limitation Notice:** In the historical 2025 registrar logbooks, only 2.7% (250 / 9,265 rows) recorded turnaround duration, representing scheduled target lead times rather than physical completion times. In accordance with strict development principles, the service returns `is_empirical: false` with explanatory disclaimers until production logs accumulate empirical completion timestamps.

### B. Daily Incoming Request Forecasting (`/forecast/daily`)
- **Objective:** Forecasts incoming credential requests expected on upcoming registrar operating days (Monday–Friday).
- **Features:** Calendar day of week, day of month, month, academic seasonality indicators (peak enrollment/graduation vs. recess), backward-looking autoregressive lags ($t-1, t-2, t-3, t-5, t-10$), and 5-day / 10-day rolling averages and volatility.
- **Champion Model:** **Random Forest Regressor** (50 trees, `max_depth=4`), crowned champion after beating all candidate models on chronological cross-validation:
  - **Random Forest (Champion):** MAE **18.97**, MedAE **12.74**, RMSE **27.10** requests/day
  - **Decision Tree Regressor:** MAE 19.11, MedAE 14.67, RMSE 26.98 requests/day
  - **Linear Regression:** MAE 23.79, MedAE 20.54, RMSE 30.35 requests/day
  - **Rolling 5-Day Average Baseline:** MAE 22.93, MedAE 18.15, RMSE 29.81 requests/day
- **Inference:** Autoregressive rollout engine generating multi-step point estimates and 90% prediction intervals.

### C. Monthly Workload Forecasting (`/forecast/monthly-workload`)
- **Objective:** Forecasts total expected monthly request volume and classifies registrar workload into operational tiers.
- **Formula:** $\text{Monthly Forecast} = \sum_{t \in \text{Operating Days}} \hat{y}_t$
- **Workload Classification Tiers (Derived from 2025 Historical Demand Quartiles):**
  - **Light:** $< 500$ requests/month (e.g., December recess)
  - **Moderate:** $500 - 850$ requests/month (e.g., April, November, February)
  - **Heavy:** $850 - 1,150$ requests/month (e.g., July, September, January)
  - **Very Heavy:** $> 1,150$ requests/month (e.g., June graduation, August enrollment peaks)

---

## 2. Directory Structure

```
ML/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── configs/
│   ├── data_config.yaml
│   ├── processing_time_config.yaml
│   ├── daily_forecast_config.yaml
│   └── workload_config.yaml
│
├── data/
│   ├── raw/                        # Original 12 monthly CSV logbooks (untouched)
│   ├── interim/                    # Interim datasets
│   ├── processed/                  # Cleaned tabular records & daily time series
│   │   ├── cleaned_requests.csv    # 9,265 validated records
│   │   ├── daily_time_series.csv   # 245 operating day observations
│   │   └── rejected_records.csv    # Unresolved records log (0 records)
│   └── external/                   # External reference calendars
│
├── notebooks/
│   ├── 00_system_and_data_audit.ipynb
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_cleaning_and_preprocessing.ipynb
│   ├── 03_processing_time_prediction.ipynb
│   ├── 04_daily_request_forecasting.ipynb
│   ├── 05_monthly_workload_forecasting.ipynb
│   ├── 06_model_evaluation_and_comparison.ipynb
│   └── 99_experiments.ipynb
│
├── src/
│   └── regquest_ml/
│       ├── __init__.py
│       ├── common/paths.py         # Centralized paths
│       ├── data/                   # Data loader, validator, cleaner
│       ├── features/               # Calendar, daily demand, processing time features
│       ├── processing_time/        # Turnaround prediction & baseline training
│       ├── forecasting/            # Daily forecaster rollout & model selection
│       ├── workload/               # Monthly 4-tier workload classifier
│       └── evaluation/             # Walk-forward CV & regression metrics
│
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── api/
│   │   └── routes/                 # processing_time, daily_forecast, monthly_workload, health
│   ├── schemas/                    # Pydantic v2 schemas (processing_time, forecasting)
│   └── services/                   # Domain services (ProcessingService, ForecastingService)
│
├── scripts/
│   ├── train_processing_time.py    # Fits & persists processing time baseline
│   ├── train_daily_forecast.py     # Fits & persists champion Random Forest forecaster
│   └── evaluate_models.py          # 4-split walk-forward CV comparative scoreboard
│
├── artifacts/
│   ├── models/
│   │   ├── processing_time/        # processing_baseline.joblib
│   │   └── daily_forecast/         # daily_forecaster.joblib
│   └── metadata/                   # metadata.json
│
├── reports/
│   ├── data_quality/               # data_quality_report.json
│   ├── evaluation/                 # evaluation_report.json
│   └── figures/                    # Visualizations and diagnostic plots
│
└── tests/
    ├── test_data_pipeline.py
    ├── test_processing_time.py
    ├── test_forecasting.py
    └── test_workload.py
```

---

## 3. Quick Start & Execution

### Training Models via Scripts
```powershell
# 1. Train Processing Time Baseline
& "D:\Capstone\ML\venv\Scripts\python.exe" D:\Capstone\ML\scripts\train_processing_time.py

# 2. Train Champion Daily Forecaster
& "D:\Capstone\ML\venv\Scripts\python.exe" D:\Capstone\ML\scripts\train_daily_forecast.py

# 3. Run Comparative Evaluation Scoreboard
& "D:\Capstone\ML\venv\Scripts\python.exe" D:\Capstone\ML\scripts\evaluate_models.py
```

### Running Automated Test Suites
```powershell
& "D:\Capstone\ML\venv\Scripts\pytest.exe" D:\Capstone\ML\tests\
```
*(All 49 unit and API integration tests pass with 100% success rate)*

### Starting the FastAPI Microservice
```powershell
& "D:\Capstone\ML\venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
Interactive OpenAPI Swagger docs: `http://localhost:8001/docs`

---

## 4. API Endpoints Reference

### `GET /health`
Returns microservice status, artifact availability, versions, and uptime.

### `POST /predict/processing-time`
Request:
```json
{
  "credential_type": "TOR",
  "purpose": "Employment",
  "date_requested": "2026-11-05",
  "num_documents": 1
}
```
Response:
```json
{
  "status": "success",
  "predicted_processing_time": 6.0,
  "unit": "operating_days",
  "model_version": "processing-time-baseline-v1",
  "basis": "administrative_standard",
  "is_empirical": false,
  "disclaimer": "Notice: Estimated processing duration reflects administrative registrar guidelines and scheduled turnaround. Actual empirical processing timestamps are not yet recorded in sufficient historical volume."
}
```

### `POST /forecast/daily`
Request:
```json
{
  "start_date": "2026-11-09",
  "number_of_operating_days": 5
}
```
Response:
```json
{
  "status": "success",
  "forecast": [
    {
      "date": "2026-11-09",
      "weekday": "Monday",
      "predicted_requests": 46,
      "lower_bound_90": 16,
      "upper_bound_90": 76
    }
  ],
  "model_version": "daily-forecaster-random_forest-v1",
  "start_date": "2026-11-09",
  "total_operating_days": 5,
  "total_predicted_requests": 230
}
```

### `POST /forecast/monthly-workload`
Request:
```json
{
  "target_month": "2026-12"
}
```
Response:
```json
{
  "status": "success",
  "month": "2026-12",
  "predicted_total_requests": 380,
  "operating_days": 23,
  "average_daily_requests": 16.5,
  "workload_level": "Light",
  "classification_basis": "historical_demand",
  "model_version": "daily-forecaster-random_forest-v1",
  "explanation": "Forecasted 380 total credential requests across 23 operating days (average 16.5 requests/day). Classified as 'Light' based on 2025 historical demand quartiles."
}
```

---

## 5. Django Backend & Frontend Integration

1. **Backend Integration (`Capstone-Backend`):**
   - Configured via `credentials/ml_client.py` and `credentials/views.py`.
   - Flake8 compliant with 0 errors (`max-line-length = 88`).
   - All 7 integration tests pass (`Ran 7 tests in 44.375s. OK`).
2. **Frontend Integration (`Capstone-Frontend`):**
   - Live daily forecasts and monthly badges rendered in `PredictedInsightCard.jsx`.
   - Real-time turnaround estimate query on `TrackStatusPage.jsx` with instant arithmetic fallback.
