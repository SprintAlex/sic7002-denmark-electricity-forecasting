"""SHAP — interprétabilité du XGBoost day-ahead (demandé dans la proposal).

Produit le beeswarm (effet + direction de chaque feature) et le bar plot
(importance moyenne). Run: HORIZON=24 .venv/bin/python -m src.models.shap_analysis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor

from config import PROCESSED_DIR, ROOT
from src.features import build_features, temporal_split, HORIZON
from src.models.xgb_model import feature_cols, TARGET

FIG = ROOT / "reports" / "figures"


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)
    cols = feature_cols(df)
    df = df.dropna(subset=cols + [TARGET]).reset_index(drop=True)
    tr, va, te = temporal_split(df)

    model = XGBRegressor()
    model.load_model(str(PROCESSED_DIR / "xgb_spot.json"))

    # échantillon de test pour SHAP (rapide)
    Xte = te[cols].sample(min(3000, len(te)), random_state=42)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xte)

    shap.summary_plot(sv, Xte, show=False, max_display=15)
    plt.title(f"SHAP beeswarm — prix spot day-ahead (H={HORIZON})")
    plt.tight_layout(); plt.savefig(FIG / "07_shap_beeswarm.png", dpi=110, bbox_inches="tight")
    plt.close()

    shap.summary_plot(sv, Xte, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP — importance moyenne |valeur|")
    plt.tight_layout(); plt.savefig(FIG / "08_shap_bar.png", dpi=110, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    print("Top 12 features (|SHAP| moyen):")
    print(mean_abs.head(12).round(2).to_string())
    print(f"\n✅ figures : 07_shap_beeswarm.png, 08_shap_bar.png")


if __name__ == "__main__":
    main()
