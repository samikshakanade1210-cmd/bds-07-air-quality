import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.baselines import chronological_time_series_split

def train_tier3_lightgbm(df_path: str):
    df = pd.read_parquet(df_path)
    train_df, test_df = chronological_time_series_split(df, train_ratio=0.8, purge_hours=24)
    
    # Define features and target
    ignore_cols = ['timestamp', 'city', 'pm2_5']
    features = [c for c in train_df.columns if c not in ignore_cols]
    target = 'pm2_5'
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Fix Windows space path issue for MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Hyperlocal_AQI_Forecasting")
    
    with mlflow.start_run(run_name="Tier3_LightGBM"):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Calculate MAE reduction % over Tier 1 Persistence (47.072)
        baseline_mae = 47.072
        mae_reduction = ((baseline_mae - mae) / baseline_mae) * 100
        
        # Log metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE_Reduction_Pct", mae_reduction)
        
        print("\n--- TIER 3 LIGHTGBM RESULTS ---")
        print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f}")
        print(f"MAE Reduction vs Tier 1: {mae_reduction:.2f}% (Acceptance Target >= 25%)")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        model_out = "models/lgbm_model.txt"
        model.booster_.save_model(model_out)
        print(f"[SUCCESS] Model saved to {model_out}")

if __name__ == "__main__":
    train_tier3_lightgbm("data/processed/processed_features.parquet")