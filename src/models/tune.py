"""Tuning XGBoost day-ahead — recherche sur la validation (pas de fuite test).

Grille compacte sur les params clés, sélection par MAE de validation,
report final sur le test. Run: HORIZON=24 .venv/bin/python -m src.models.tune
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import itertools
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import PROCESSED_DIR
from src.features import build_features, temporal_split
from src.models.xgb_model import feature_cols, TARGET, directional_accuracy

GRID = {
    "max_depth": [6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5],
}


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)
    cols = feature_cols(df)
    df = df.dropna(subset=cols + [TARGET]).reset_index(drop=True)
    tr, va, te = temporal_split(df)
    Xtr, ytr, Xva, yva, Xte, yte = (tr[cols], tr[TARGET], va[cols], va[TARGET],
                                    te[cols], te[TARGET])

    keys = list(GRID)
    best, best_mae = None, np.inf
    print(f"{len(list(itertools.product(*GRID.values())))} configs évaluées sur validation\n")
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(keys, combo))
        m = XGBRegressor(n_estimators=800, subsample=0.8, colsample_bytree=0.8,
                         n_jobs=-1, random_state=42, early_stopping_rounds=40,
                         eval_metric="mae", **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        mae = mean_absolute_error(yva, m.predict(Xva))
        print(f"  {p} -> val MAE {mae:.2f} (n_trees={m.best_iteration})")
        if mae < best_mae:
            best_mae, best, best_model = mae, p, m

    pred = best_model.predict(Xte)
    ref = te["spot_price_eur_lag24"]
    print(f"\n>>> MEILLEUR: {best} | val MAE {best_mae:.2f}")
    print(f"    TEST  MAE={mean_absolute_error(yte, pred):.2f}  "
          f"RMSE={np.sqrt(mean_squared_error(yte, pred)):.2f}  "
          f"DirAcc={directional_accuracy(yte.values, pred, ref.values)*100:.1f}%")
    print(f"    (baseline non-tuné : MAE 25.63)")
    best_model.save_model(str(PROCESSED_DIR / "xgb_spot_tuned.json"))
    print("✅ modèle tuné sauvé : data/processed/xgb_spot_tuned.json")


if __name__ == "__main__":
    main()
