import os
import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.baselines import chronological_time_series_split

def generate_shap_explainability(df_path: str, model_path: str):
    """Section 5: Explainability Pipeline.
    Computes SHAP values to quantify feature contributions for PM2.5 predictions.
    """
    if not os.path.exists(df_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Processed dataset or trained model artifact missing!")
        
    df = pd.read_parquet(df_path)
    _, test_df = chronological_time_series_split(df, train_ratio=0.8, purge_hours=24)
    
    ignore_cols = ['timestamp', 'city', 'pm2_5']
    features = [c for c in test_df.columns if c not in ignore_cols]
    X_test = test_df[features]
    
    # Load trained LightGBM model
    model = lgb.Booster(model_file=model_path)
    
    # Compute TreeSHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Save SHAP Summary Plot
    os.makedirs("reports/figures", exist_ok=True)
    summary_plot_path = "reports/figures/shap_summary.png"
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    
    print(f"[SUCCESS] SHAP Summary Plot generated and saved to {summary_plot_path}")

if __name__ == "__main__":
    generate_shap_explainability(
        df_path="data/processed/processed_features.parquet",
        model_path="models/lgbm_model.txt"
    )