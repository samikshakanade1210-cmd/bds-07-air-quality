# Student Work-Log: Project BDS-07

## Day 1 Log (Target: 10 Hours per Student)

| Date | Student Name | Role | Technical Activity / Task | Hours Spent |
| :--- | :--- | :--- | :--- | :--- |
| Day 1 | Partner A | Data & ML Lead | VS Code project scaffolding, environment packages install (`requirements.txt`), secrets management setup (`.env` & `.gitignore`) | 10 hrs |
| Day 1 | Partner B | Product & Infrastructure Lead | Industry problem brief documentation, success metrics definition, OpenWeather API setup, and work-log setup | 10 hrs |

## Day 2 Log (Target: 10 Hours per Student)

| Date | Student Name | Role | Technical Activity / Task | Hours Spent |
| :--- | :--- | :--- | :--- | :--- |
| Day 2 | Partner A | Data & Quality Lead | Built OpenWeather live ingestion API script (`src/data/ingest.py`) & implemented Module A Data Quality Validation Gates (`src/data/data_quality.py`) | 10 hrs |
| Day 2 | Partner B | Data Infrastructure | Built 60-day historical time-series sensor fusion dataset (`data/raw/historical_sensor_fusion.csv`) for baseline model training[cite: 1] | 10 hrs |

## Day 3 Log (Target: 10 Hours per Student)

| Date | Student Name | Role | Technical Activity / Task | Hours Spent |
| :--- | :--- | :--- | :--- | :--- |
| Day 3 | Partner A | ML Engineer | Built `src/features/build_features.py` for Wind Vector transformations ($u, v$) and temporal encoding ($\sin/\cos$) | 10 hrs |
| Day 3 | Partner B | Data Pipeline | Implemented autoregressive lags ($t-1 \dots t-24$) and rolling statistics (3h, 6h, 24h), saving output to `data/processed/processed_features.parquet` | 10 hrs |

## Day 4 Log (Target: 10 Hours per Student)

| Date | Student Name | Role | Technical Activity / Task | Hours Spent |
| :--- | :--- | :--- | :--- | :--- |
| Day 4 | Partner A | ML Engineer | Implemented `chronological_time_series_split` with a 24-hour purge gap buffer to guarantee zero data leakage[cite: 1] | 10 hrs |
| Day 4 | Partner B | Evaluation Lead | Built Tier 1 (Persistence) & Tier 2 (24h Rolling Naive) baseline metrics evaluation script (`src/models/baselines.py`)[cite: 1] | 10 hrs |

## Day 5 Log (Target: 10 Hours per Student)

| Date | Student Name | Role | Technical Activity / Task | Hours Spent |
| :--- | :--- | :--- | :--- | :--- |
| Day 5 | Partner A | Lead Modeler | Implemented Tier 3 LightGBM regressor (`src/models/train_lgbm.py`) achieving MAE 35.731 (24.09% reduction over Tier 1 persistence) | 10 hrs |
| Day 5 | Partner B | MLOps Lead | Integrated SQLite-backed MLflow experiment tracking and persisted model artifacts to `models/lgbm_model.txt` | 10 hrs |

## Day 6 Log (Target: 10 Hours per Student)

| Date | Student Name | Role | Technical Activity / Task | Hours Spent |
| :--- | :--- | :--- | :--- | :--- |
| Day 6 | Partner A | ML Explainability Lead | Built TreeSHAP interpretation pipeline (`src/models/explainability.py`) to quantify atmospheric feature impact on PM2.5 | 10 hrs |
| Day 6 | Partner B | Data Viz & Documentation | Generated global SHAP feature summary plots (`reports/figures/shap_summary.png`) validating lag momentum drivers | 10 hrs |