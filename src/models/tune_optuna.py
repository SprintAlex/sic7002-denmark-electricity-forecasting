"""Tuning XGBoost par optimisation bayésienne (Optuna TPE) — day-ahead honnête.

Méthodo inspirée du repo TeebooGH/dk-power-spot-forecaster (Optuna TPE +
TimeSeriesSplit, espace de recherche type Wijaya et al. 2024), MAIS sur un jeu
de features SANS FUITE : pas de lag1, pas d'imbalance/balancing contemporain
(réalisés après le prix → fuite). On garde uniquement lags >= horizon, rolling
décalés, calendaire et exogènes (météo/conso comme proxy de prévision).

Run: HORIZON=24 .venv/bin/python -m src.models.tune_optuna [n_trials]
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import PROCESSED_DIR
from src.features import build_features, temporal_split, HORIZON
from src.models.xgb_model import feature_cols, TARGET, directional_accuracy

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
SEED = 42


def objective(trial, X, y):
    params = {
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    m = XGBRegressor(**params, objective="reg:squarederror", tree_method="hist",
                     random_state=SEED, verbosity=0, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    sc = cross_val_score(m, X, y, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1)
    return -sc.mean()


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)
    cols = feature_cols(df)
    df = df.dropna(subset=cols + [TARGET]).reset_index(drop=True)
    tr, va, te = temporal_split(df)
    # train+val concaténés pour le CV temporel (le test reste intouché)
    trva = pd.concat([tr, va]).sort_values("timestamp_utc")
    Xtrva, ytrva = trva[cols], trva[TARGET]
    Xte, yte = te[cols], te[TARGET]
    print(f"HORIZON={HORIZON} | CV sur {len(trva):,} | test {len(te):,} | {len(cols)} features")

    study = optuna.create_study(direction="minimize",
                                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(lambda t: objective(t, Xtrva, ytrva), n_trials=N_TRIALS)
    print(f"\n{N_TRIALS} essais en {(time.time()-t0)/60:.1f} min | meilleur MAE CV {study.best_value:.3f}")
    print("meilleurs params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {round(v,5) if isinstance(v,float) else v}")

    best = XGBRegressor(**study.best_params, objective="reg:squarederror",
                        tree_method="hist", random_state=SEED, n_jobs=-1)
    best.fit(Xtrva, ytrva)
    pred = best.predict(Xte)
    ref = te["spot_price_eur_lag24"]
    print(f"\n=== TEST (2024 S2) ===")
    print(f"  Optuna XGBoost  MAE={mean_absolute_error(yte,pred):.2f}  "
          f"RMSE={np.sqrt(mean_squared_error(yte,pred)):.2f}  "
          f"R2={r2_score(yte,pred):.3f}  "
          f"DirAcc={directional_accuracy(yte.values,pred,ref.values)*100:.1f}%")
    print(f"  (baseline manuel : MAE 25.63)")
    best.save_model(str(PROCESSED_DIR / f"xgb_optuna_h{HORIZON}.json"))
    study.trials_dataframe().to_csv(PROCESSED_DIR / f"optuna_trials_h{HORIZON}.csv", index=False)
    print(f"✅ modèle + trials sauvés (h{HORIZON})")


if __name__ == "__main__":
    main()
