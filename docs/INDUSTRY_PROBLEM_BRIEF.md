# Industry Problem Brief: Project BDS-07

## 1. Project Overview & Problem Statement
- **Project Title:** BDS-07 | Hyperlocal Air Quality Forecasting with Sensor Fusion and Explainable Source Signals
- **Core Challenge:** Air-quality levels vary significantly across location, weather, and time. Public dashboards provide current values without local predictive insights[cite: 1].
- **Objective:** Build a working, stakeholder-testable prototype that integrates data collection, missing-value treatment, weather joins, regression forecasting, and geospatial dashboards with full ML explainability (SHAP)[cite: 1].

## 2. Key Stakeholders & Target Personas
- **Citizens:** Need accurate, hyperlocal real-time AQI predictions to plan daily outdoor exposure[cite: 1].
- **Urban Planners & Environmental Researchers:** Need explainable source signals to identify pollution drivers (e.g., wind speed impact vs. traffic emission)[cite: 1].

## 3. Measurable Success Metrics
- **Model Evaluation:** MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)[cite: 1].
- **System Metrics:** Data completeness, interval coverage, and spatial error handling[cite: 1].

## 4. Risks & Misuse Cases
- **Sensor Failure:** System must handle missing sensor signals without defaulting to false "clean air" predictions[cite: 1].