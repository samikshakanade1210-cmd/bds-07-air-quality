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