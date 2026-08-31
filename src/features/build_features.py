import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root directory to sys.path so 'src' can be imported anywhere
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.data_quality import clean_air_quality_data


def build_atmospheric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Module B: Sensor Fusion & Feature Engineering.
    Calculates Wind Vectors (u, v), Diurnal Cycles, Lags, and Rolling Statistics.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["city", "timestamp"]).reset_index(drop=True)

    # 1. Temporal Cycles (Diurnal capture for rush hours)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # 2. Wind Vectors (u, v resolution for circular continuity at 0°/360°)
    if "wind_speed" in df.columns and "wind_deg" in df.columns:
        rad = np.radians(df["wind_deg"])
        df["u_wind"] = -df["wind_speed"] * np.sin(rad)
        df["v_wind"] = -df["wind_speed"] * np.cos(rad)

    # Group by city for time-series computations
    processed_dfs = []
    for city, group in df.groupby("city"):
        group = group.copy()

        # 3. Autoregressive Lags (t-1, t-2, t-3, t-6, t-12, t-24)
        for lag in [1, 2, 3, 6, 12, 24]:
            group[f"pm2_5_lag_{lag}"] = group["pm2_5"].shift(lag)

        # 4. Rolling Statistics (3h, 6h, 24h)
        for window in [3, 6, 24]:
            group[f"pm2_5_roll_mean_{window}h"] = (
                group["pm2_5"].shift(1).rolling(window=window).mean()
            )
            group[f"pm2_5_roll_std_{window}h"] = (
                group["pm2_5"].shift(1).rolling(window=window).std()
            )

        processed_dfs.append(group)

    df_featured = (
        pd.concat(processed_dfs, axis=0).dropna().reset_index(drop=True)
    )
    return df_featured


if __name__ == "__main__":
    raw_path = "data/raw/historical_sensor_fusion.csv"
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        clean_df = clean_air_quality_data(raw_df)
        featured_df = build_atmospheric_features(clean_df)

        os.makedirs("data/processed", exist_ok=True)
        out_path = "data/processed/processed_features.parquet"
        featured_df.to_parquet(out_path, index=False)
        print(
            f"[SUCCESS] Feature Engineering Complete! Processed dataset shape: {featured_df.shape}"
        )
        print(f"Saved to {out_path}")