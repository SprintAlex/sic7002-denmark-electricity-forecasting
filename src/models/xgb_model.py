"""Baseline XGBoost — prévision du prix spot day-ahead (EUR/MWh).

Compare à un baseline naïf (persistance T-24). Métriques : MAE, RMSE,
directional accuracy. Importance des features via SHAP (interprétabilité,
demandée dans la proposal). Split temporel strict (aucune fuite).

Run: .venv/bin/python -m src.models.xgb_model
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import PROCESSED_DIR, ROOT, FORECAST_COLS
from src.features import build_features, temporal_split

TARGET = "spot_price_eur"
FIG = ROOT / "reports" / "figures"

# features = calendaire + lags/rolling (prix & imbalance passés) + drivers exogènes
# + prévisions day-ahead éolien/solaire (connues la veille, sans fuite).
# EXCLUS (fuite) : prix courant, imbalance/mfrr courants, gap_flags.
EXOG = ["consumption_mwh", "wind_speed", "radia_glob", "temp_dry"] + FORECAST_COLS

# hyperparamètres retenus : Optuna (80 essais, CV TimeSeriesSplit) + loss pseudo-Huber.
# cf. reports/optuna_xgb.log, src.models.xgb_experiments, src.models.xgb_forecast_exog.
BEST_PARAMS = dict(learning_rate=0.01116, n_estimators=496, max_depth=10,
                   subsample=0.73377, colsample_bytree=0.69384, min_child_weight=3,
                   gamma=0.21781, reg_alpha=0.38953, reg_lambda=0.00012,
                   objective="reg:pseudohubererror")


def feature_cols(df):
    cols = [c for c in df.columns if c.startswith(("sin_", "cos_", "is_"))]
    cols += [c for c in df.columns if "_lag" in c or "_roll" in c]
    cols += EXOG + ["zone_id"]
    return cols


def directional_accuracy(y_true, y_pred, ref):
    """% de fois où le signe de la variation (vs T-24) est bien prédit."""
    true_dir = np.sign(y_true - ref)
    pred_dir = np.sign(y_pred - ref)
    mask = true_dir != 0
    return (true_dir[mask] == pred_dir[mask]).mean()


def evaluate(name, y_true, y_pred, ref):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    da = directional_accuracy(y_true.values, y_pred, ref.values)
    print(f"  {name:20s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:5.3f}  DirAcc={da*100:5.1f}%")
    return mae, rmse, r2, da


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)

    cols = feature_cols(df)
    # retire les lignes sans historique complet (168h initiales de lags)
    df = df.dropna(subset=cols + [TARGET]).reset_index(drop=True)

    tr, va, te = temporal_split(df)
    # entraînement sur train+val (n_estimators fixé par Optuna -> pas d'early stopping)
    trva = pd.concat([tr, va]).sort_values("timestamp_utc")
    print(f"train+val={len(trva):,}  test={len(te):,}  | {len(cols)} features")

    Xtrva, ytrva = trva[cols], trva[TARGET]
    Xte, yte = te[cols], te[TARGET]

    model = XGBRegressor(**BEST_PARAMS, tree_method="hist",
                         n_jobs=-1, random_state=42, verbosity=0)
    model.fit(Xtrva, ytrva)

    pred = model.predict(Xte)
    ref = te["spot_price_eur_lag24"]  # référence pour directional accuracy

    print("\n=== TEST (2024 S2) ===")
    evaluate("Naïf (persist T-24)", yte, ref.values, ref)
    evaluate("XGBoost", yte, pred, ref)

    # --- prédictions vs réel (1 semaine) ---
    s = te[te.zone == "DK1"].head(168)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s["timestamp_utc"], s[TARGET], label="réel", lw=1.5)
    ax.plot(s["timestamp_utc"], model.predict(s[cols]), label="XGBoost", lw=1.5)
    ax.set_title("XGBoost vs réel — DK1, 1 semaine de test"); ax.legend()
    ax.set_ylabel("EUR/MWh"); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(FIG / "06_xgb_predictions.png", dpi=110)
    print(f"\n  -> {FIG/'06_xgb_predictions.png'}")

    # --- importance (gain) ---
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    print("\nTop 10 features (gain XGBoost):")
    print(imp.head(10).round(3).to_string())

    model.save_model(str(PROCESSED_DIR / "xgb_spot.json"))
    print("\n✅ modèle sauvé : data/processed/xgb_spot.json")


if __name__ == "__main__":
    main()
