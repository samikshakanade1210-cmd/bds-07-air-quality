import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

def chronological_time_series_split(df: pd.DataFrame, train_ratio: float = 0.8, purge_hours: int = 24):
    """Section 3: Zero-Leakage Validation Split.
    Applies chronological split with a purge gap buffer to avoid temporal leakage.
    """
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    
    # Purge gap buffer between train and test sets
    test_start = train_end + purge_hours
    
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[test_start:].copy()
    
    print(f"[SPLIT COMPLETE] Train samples: {len(train_df)} | Test samples: {len(test_df)} (Purge gap: {purge_hours}h)")
    return train_df, test_df

def evaluate_baseline_models(df_path: str):
    """Section 4: Tier 1 Persistence and Tier 2 Rolling Baselines."""
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Processed feature file missing at {df_path}")
        
    df = pd.read_parquet(df_path)
    train_df, test_df = chronological_time_series_split(df, train_ratio=0.8, purge_hours=24)
    
    # Actual Target (PM2.5 concentration)
    y_true = test_df['pm2_5'].values
    
    # Tier 1 Baseline: Persistence (Predict y_hat_{t+h} = y_t)
    y_pred_tier1 = test_df['pm2_5_lag_1'].values
    
    # Tier 2 Baseline: 24h Rolling Mean Naive Predictor
    y_pred_tier2 = test_df['pm2_5_roll_mean_24h'].values
    
    # Calculate Evaluation Metrics
    mae_t1 = mean_absolute_error(y_true, y_pred_tier1)
    rmse_t1 = np.sqrt(mean_squared_error(y_true, y_pred_tier1))
    
    mae_t2 = mean_absolute_error(y_true, y_pred_tier2)
    rmse_t2 = np.sqrt(mean_squared_error(y_true, y_pred_tier2))
    
    print("\n--- BASELINE EVALUATION RESULTS ---")
    print(f"Tier 1 (Persistence Baseline)  -> MAE: {mae_t1:.3f} | RMSE: {rmse_t1:.3f}")
    print(f"Tier 2 (24h Rolling Mean Naive) -> MAE: {mae_t2:.3f} | RMSE: {rmse_t2:.3f}")

if __name__ == "__main__":
    processed_path = "data/processed/processed_features.parquet"
    evaluate_baseline_models(processed_path)