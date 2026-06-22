"""#1 — ajoute les prévisions day-ahead éolien/solaire (EDS) comme exogènes.

Mesure le gain réel de la météo "production" honnête (ForecastDayAhead, connue
la veille) vs le modèle actuel. On garde la meilleure loss trouvée (pseudo-Huber).
Compare : features actuelles  vs  + prévisions  vs  + prévisions + net load.

Run: HORIZON=24 .venv/bin/python -m src.models.xgb_forecast_exog
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import PROCESSED_DIR, START_DATE, END_DATE
from src.features import build_features, temporal_split, HORIZON
from src.models.xgb_model import feature_cols, TARGET, directional_accuracy
from src.ingest.forecasts import fetch_forecasts

SEED = 42
BEST = dict(learning_rate=0.01116, n_estimators=496, max_depth=10,
            subsample=0.73377, colsample_bytree=0.69384, min_child_weight=3,
            gamma=0.21781, reg_alpha=0.38953, reg_lambda=0.00012)
FC_COLS = ["fc_offshore_da", "fc_onshore_da", "fc_solar_da", "fc_wind_da", "fc_renew_da"]
CACHE = PROCESSED_DIR / "forecasts.parquet"


def load_forecasts():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    print("  pull EDS Forecasts_Hour 2022-2024 ...")
    fc = fetch_forecasts(START_DATE, END_DATE)
    fc.to_parquet(CACHE)
    print(f"  -> {len(fc):,} lignes mises en cache")
    return fc


def run(name, cols, trva, te, ref):
    m = XGBRegressor(**BEST, objective="reg:pseudohubererror", tree_method="hist",
                     random_state=SEED, n_jobs=-1, verbosity=0)
    m.fit(trva[cols], trva[TARGET])
    pred = m.predict(te[cols])
    yte = te[TARGET].to_numpy()
    mae, rmse, r2 = (mean_absolute_error(yte, pred),
                     np.sqrt(mean_squared_error(yte, pred)), r2_score(yte, pred))
    da = directional_accuracy(yte, pred, ref) * 100
    print(f"  {name:34s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:5.3f}  DirAcc={da:5.1f}%  ({len(cols)} feat)")
    return mae, rmse, r2


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)

    fc = load_forecasts()
    fc["timestamp_utc"] = pd.to_datetime(fc["timestamp_utc"], utc=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.merge(fc, on=["timestamp_utc", "zone"], how="left")
    df["net_load_da"] = df["consumption_mwh"] - df["fc_renew_da"]   # demande résiduelle (proxy)

    base = feature_cols(df)
    cov = df[FC_COLS].notna().all(axis=1).mean()
    print(f"HORIZON={HORIZON} | couverture forecasts mergés : {cov*100:.1f}%\n")

    sets = {
        "actuel (réf)":                 base,
        "+ prévisions éol/sol DA":      base + FC_COLS,
        "+ prévisions + net load":      base + FC_COLS + ["net_load_da"],
    }
    all_cols = list(dict.fromkeys(sum(sets.values(), [])))
    d = df.dropna(subset=all_cols + [TARGET]).reset_index(drop=True)
    tr, va, te = temporal_split(d)
    trva = pd.concat([tr, va]).sort_values("timestamp_utc")
    ref = te["spot_price_eur_lag24"].to_numpy()
    print(f"train+val {len(trva):,} | test {len(te):,}\n")

    res = {k: run(k, c, trva, te, ref) for k, c in sets.items()}
    b = res["actuel (réf)"]
    print(f"\n  rappel pseudo-Huber sans forecasts : MAE {b[0]:.2f} RMSE {b[1]:.2f} R2 {b[2]:.3f}")


if __name__ == "__main__":
    main()
