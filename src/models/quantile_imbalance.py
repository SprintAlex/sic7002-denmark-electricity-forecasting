"""Prévision probabiliste du prix de déséquilibre (imbalance) par quantiles.

Répond à la partie "probabilistic imbalance forecasting / QRF" de la proposal :
au lieu d'une prévision ponctuelle, on produit des intervalles (P10, P50, P90)
via XGBoost en régression quantile. On vérifie la couverture empirique
(≈ 80% des réalisations doivent tomber dans [P10, P90]).

Run: HORIZON=24 .venv/bin/python -m src.models.quantile_imbalance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from config import PROCESSED_DIR, ROOT
from src.features import build_features, temporal_split
from src.models.xgb_model import feature_cols

TARGET = "imbalance"
FIG = ROOT / "reports" / "figures"
QUANTILES = [0.1, 0.5, 0.9]


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)
    cols = feature_cols(df)
    df = df.dropna(subset=cols + [TARGET]).reset_index(drop=True)
    tr, va, te = temporal_split(df)
    Xtr, ytr, Xte, yte = tr[cols], tr[TARGET], te[cols], te[TARGET]
    print(f"train={len(tr):,} test={len(te):,} | cible=imbalance (prix de déséquilibre)")

    preds = {}
    for q in QUANTILES:
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                         n_estimators=400, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
        m.fit(Xtr, ytr)
        preds[q] = m.predict(Xte)

    p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
    coverage = ((yte.values >= p10) & (yte.values <= p90)).mean()
    print(f"\nMAE médiane (P50) : {mean_absolute_error(yte, p50):.2f}")
    print(f"Couverture [P10,P90] : {coverage*100:.1f}%  (cible ~80%)")
    print(f"Largeur moyenne intervalle : {(p90 - p10).mean():.1f} EUR/MWh")

    # plot : réel vs intervalle sur 1 semaine de test (DK1)
    s = te[te.zone == "DK1"].head(168).reset_index()
    idx = s["index"]
    pos = te.index.get_indexer(idx)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(s)), p10[pos], p90[pos], alpha=.3, label="P10-P90")
    ax.plot(range(len(s)), p50[pos], label="P50 (médiane)", lw=1.3)
    ax.plot(range(len(s)), s[TARGET].values, "k.", ms=3, label="réel")
    ax.set_title("Prévision probabiliste imbalance — DK1, 1 semaine test")
    ax.set_ylabel("EUR/MWh"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "09_imbalance_quantiles.png", dpi=110)
    print(f"\n✅ figure : 09_imbalance_quantiles.png")


if __name__ == "__main__":
    main()
