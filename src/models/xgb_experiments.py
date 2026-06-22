"""Expériences pour faire baisser MAE/RMSE au-delà du tuning (day-ahead honnête).

Part des MEILLEURS hyperparamètres Optuna (xgb_optuna) et teste deux leviers
gratuits en code, sans réintroduire de fuite :

  #5  objectif d'apprentissage : squared (RMSE) vs absolute (MAE) vs pseudo-Huber
  #4  cible : prix brut vs RÉSIDU sur la persistance lag24 (on prédit prix - lag24,
      puis on rajoute lag24). Recentre la cible, aide souvent les arbres.

Entraînement sur train+val (comme le run Optuna final), test intouché (2024 S2).
Run: HORIZON=24 .venv/bin/python -m src.models.xgb_experiments
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import PROCESSED_DIR
from src.features import build_features, temporal_split, HORIZON
from src.models.xgb_model import feature_cols, TARGET, directional_accuracy

SEED = 42
# meilleurs params Optuna (80 essais, CV TimeSeriesSplit) — voir reports/optuna_xgb.log
BEST = dict(learning_rate=0.01116, n_estimators=496, max_depth=10,
            subsample=0.73377, colsample_bytree=0.69384, min_child_weight=3,
            gamma=0.21781, reg_alpha=0.38953, reg_lambda=0.00012)

OBJECTIVES = {
    "squared":     "reg:squarederror",
    "absolute":    "reg:absoluteerror",
    "pseudohuber": "reg:pseudohubererror",
}


def fit_predict(objective, Xtrva, ytrva, Xte):
    m = XGBRegressor(**BEST, objective=objective, tree_method="hist",
                     random_state=SEED, n_jobs=-1, verbosity=0)
    m.fit(Xtrva, ytrva)
    return m.predict(Xte)


def score(name, yte, pred, ref):
    mae = mean_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    r2 = r2_score(yte, pred)
    da = directional_accuracy(np.asarray(yte), np.asarray(pred), np.asarray(ref)) * 100
    print(f"  {name:28s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:5.3f}  DirAcc={da:5.1f}%")
    return mae, rmse, r2, da


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)
    cols = feature_cols(df)
    df = df.dropna(subset=cols + [TARGET]).reset_index(drop=True)
    tr, va, te = temporal_split(df)
    trva = pd.concat([tr, va]).sort_values("timestamp_utc")

    Xtrva, Xte = trva[cols], te[cols]
    yte = te[TARGET].to_numpy()
    ref = te["spot_price_eur_lag24"].to_numpy()           # persistance = baseline directionnel
    lag_trva = trva["spot_price_eur_lag24"].to_numpy()

    print(f"HORIZON={HORIZON} | train+val {len(trva):,} | test {len(te):,} | {len(cols)} features\n")
    print(f"  {'référence':28s} ", end="")
    score("(naïf persist T-24)", yte, ref, ref)
    print()

    results = {}
    # #5 — cible = prix brut, on varie l'objectif
    print("[#5] cible = prix brut :")
    for tag, obj in OBJECTIVES.items():
        pred = fit_predict(obj, Xtrva, trva[TARGET], Xte)
        results[("brut", tag)] = score(f"prix brut / {tag}", yte, pred, ref)

    # #4 — cible = résidu (prix - lag24), reconstruit en ajoutant lag24
    print("\n[#4] cible = résidu vs lag24 (prix - lag24) :")
    yresid = (trva[TARGET].to_numpy() - lag_trva)
    for tag, obj in OBJECTIVES.items():
        pred_resid = fit_predict(obj, Xtrva, pd.Series(yresid, index=trva.index), Xte)
        pred = pred_resid + ref                            # reconstruction du prix
        results[("résidu", tag)] = score(f"résidu / {tag}", yte, pred, ref)

    # récap meilleur par MAE et par RMSE
    best_mae = min(results.items(), key=lambda kv: kv[1][0])
    best_rmse = min(results.items(), key=lambda kv: kv[1][1])
    print(f"\n>>> meilleur MAE  : {best_mae[0]} -> MAE {best_mae[1][0]:.2f} "
          f"(RMSE {best_mae[1][1]:.2f}, R2 {best_mae[1][2]:.3f})")
    print(f">>> meilleur RMSE : {best_rmse[0]} -> RMSE {best_rmse[1][1]:.2f} "
          f"(MAE {best_rmse[1][0]:.2f}, R2 {best_rmse[1][2]:.3f})")
    print(f"    rappels : Optuna brut/squared MAE 25.49 RMSE 38.86 | LSTM 128x2 MAE 24.78")


if __name__ == "__main__":
    main()
