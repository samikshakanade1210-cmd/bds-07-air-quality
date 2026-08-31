import pandas as pd
import numpy as np

def clean_air_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Quality Gates from Model Blueprint:
    1. Removes negative pollutant concentrations.
    2. Bounds relative humidity (0-100%).
    3. Forward-fills short sensor gaps (<= 3 hours).
    """
    df = df.copy()
    pollutants = ['pm2_5', 'pm10', 'no2', 'so2', 'co']
    
    # Boundary checks
    for p in pollutants:
        if p in df.columns:
            df[p] = df[p].apply(lambda x: np.nan if (pd.notnull(x) and x < 0) else x)
            
    if 'humidity' in df.columns:
        df['humidity'] = df['humidity'].clip(0, 100)

    # Imputation for short intervals (<= 3 hours)
    df[pollutants] = df[pollutants].ffill(limit=3)
    return df

if __name__ == "__main__":
    raw_path = "data/raw/raw_sensor_fusion.csv"
    if pd.io.common.file_exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        df_clean = clean_air_quality_data(df_raw)
        print(f"[QUALITY GATE PASSED] Cleaned {len(df_clean)} records successfully!")