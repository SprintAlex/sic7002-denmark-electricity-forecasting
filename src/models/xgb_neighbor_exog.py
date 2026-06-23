"""#1 prix zones voisines (lags) + #2 pondération récence — validation du gain.

Part du meilleur XGBoost actuel (Optuna + pseudo-Huber + forecasts day-ahead) et
ajoute, sans fuite :
  #1  lags (T-24, T-168) des prix DE / SE3 / SE4 / NO2 / SYSTEM (couplage de marché).
      Les prix voisins CONTEMPORAINS sont exclus (même enchère que DK -> fuite).
  #2  sample_weight croissant dans le temps (demi-vie ~180 j) pour recoller au régime
      récent (corrige test MAE > val MAE, le shift S1->S2 2024).

Run: HORIZON=24 .venv/bin/python -m src.models.xgb_neighbor_exog
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
from src.models.xgb_model import feature_cols, TARGET, BEST_PARAMS, directional_accuracy
from src.ingest.neighbor_prices import fetch_neighbor_prices, NB_COLS

SEED = 42
LAGS = [24, 168]
HALFLIFE_DAYS = 180.0
CACHE = PROCESSED_DIR / "neighbor_prices.parquet"
NB_LAG_COLS = [f"{c}_lag{l}" for c in NB_COLS for l in LAGS]


def load_neighbor_lags():
    if CACHE.exists():
        nb = pd.read_parquet(CACHE)
    else:
        print("  pull EDS Elspotprices (voisins) ...")
        nb = fetch_neighbor_prices(START_DATE, END_DATE)
        nb.to_parquet(CACHE)
        print(f"  -> {len(nb):,} heures en cache")
    nb["timestamp_utc"] = pd.to_datetime(nb["timestamp_utc"], utc=True)
    # squelette horaire continu -> shift par lignes = shift horaire exact (pas de trou)
    full = pd.date_range(nb["timestamp_utc"].min(), nb["timestamp_utc"].max(), freq="h")
    nb = nb.set_index("timestamp_utc").reindex(full).rename_axis("timestamp_utc").reset_index()
    for c in NB_COLS:
        for l in LAGS:
            nb[f"{c}_lag{l}"] = nb[c].shift(l)
    return nb[["timestamp_utc"] + NB_LAG_COLS]


def recency_weight(ts):
    age_days = (ts.max() - ts).dt.total_seconds() / 86400.0
    return (0.5 ** (age_days / HALFLIFE_DAYS)).to_numpy()


def fit_eval(name, cols, trva, te, ref, weighted=False):
    m = XGBRegressor(**BEST_PARAMS, tree_method="hist", random_state=SEED,
                     n_jobs=-1, verbosity=0)
    sw = recency_weight(trva["timestamp_utc"]) if weighted else None
    m.fit(trva[cols], trva[TARGET], sample_weight=sw)
    pred = m.predict(te[cols])
    yte = te[TARGET].to_numpy()
    mae, rmse, r2 = (mean_absolute_error(yte, pred),
                     np.sqrt(mean_squared_error(yte, pred)), r2_score(yte, pred))
    da = directional_accuracy(yte, pred, ref) * 100
    print(f"  {name:36s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  R2={r2:5.3f}  DirAcc={da:5.1f}%  ({len(cols)} feat)")
    return mae, rmse, r2


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    nb = load_neighbor_lags()
    df = df.merge(nb, on="timestamp_utc", how="left")

    # feature_cols ramasse tout "_lag" -> on retire les lags voisins pour isoler le delta
    base = [c for c in feature_cols(df) if c not in NB_LAG_COLS]
    cov = df[NB_LAG_COLS].notna().all(axis=1).mean()
    print(f"HORIZON={HORIZON} | couverture lags voisins : {cov*100:.1f}%\n")

    d = df.dropna(subset=base + NB_LAG_COLS + [TARGET]).reset_index(drop=True)
    tr, va, te = temporal_split(d)
    trva = pd.concat([tr, va]).sort_values("timestamp_utc")
    ref = te["spot_price_eur_lag24"].to_numpy()
    print(f"train+val {len(trva):,} | test {len(te):,}\n")

    fit_eval("actuel (forecasts)",          base,               trva, te, ref)
    fit_eval("#1 + lags prix voisins",      base + NB_LAG_COLS, trva, te, ref)
    fit_eval("#2 + récence (poids temporel)", base + NB_LAG_COLS, trva, te, ref, weighted=True)
    print(f"\n  rappel canonique actuel : MAE 23.26 RMSE 37.78 R2 0.626")


if __name__ == "__main__":
    main()
