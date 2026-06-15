"""Feature engineering pour XGBoost / LSTM.

Produit un dataset model-ready à partir du CSV nettoyé :
  - lags auto-régressifs (T-1, T-24, T-168) sur le prix et l'imbalance
  - rolling stats (moyenne/écart 24h, 168h)
  - calendaire cyclique (heure, jour, mois en sin/cos) + weekend + fériés DK
  - split temporel SANS fuite (train < val < test dans le temps)

⚠️ Tous les lags/rolling calculés PAR ZONE, en respectant l'ordre temporel
(shift positif uniquement) -> aucune fuite du futur.
Run: .venv/bin/python -m src.features
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import holidays

from config import PROCESSED_DIR, ZONES

CSV = PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv"
OUT = PROCESSED_DIR / "features.parquet"

TARGETS = ["spot_price_eur", "imbalance"]
# horizon de prévision (h) : 1 = nowcast (utilise lag1) ; 24 = day-ahead réaliste.
# En day-ahead on ne dispose que d'infos jusqu'à H-24 -> lag min = horizon.
HORIZON = int(__import__("os").environ.get("HORIZON", "24"))
LAGS = [h for h in [1, 24, 168] if h >= HORIZON]
ROLL_WINDOWS = [24, 168]
ROLL_SHIFT = HORIZON  # la fenêtre rolling exclut tout ce qui est < H-horizon
DK_HOLIDAYS = holidays.Denmark(years=range(2021, 2026))


def _cyclical(s, period):
    return np.sin(2 * np.pi * s / period), np.cos(2 * np.pi * s / period)


def build_features(df):
    df = df.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    ts = df["timestamp_utc"]

    # --- calendaire cyclique ---
    df["sin_hour"], df["cos_hour"] = _cyclical(ts.dt.hour, 24)
    df["sin_dow"], df["cos_dow"] = _cyclical(ts.dt.dayofweek, 7)
    df["sin_month"], df["cos_month"] = _cyclical(ts.dt.month, 12)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = ts.dt.date.isin(DK_HOLIDAYS).astype(int)

    # --- lags + rolling PAR ZONE (shift -> pas de fuite) ---
    parts = []
    for z in ZONES:
        g = df[df.zone == z].copy()
        for col in TARGETS:
            for lag in LAGS:
                g[f"{col}_lag{lag}"] = g[col].shift(lag)
            for w in ROLL_WINDOWS:
                # shift(horizon) avant rolling : exclut tout ce qui n'est pas
                # disponible à l'instant de prévision (pas de fuite à l'horizon)
                base = g[col].shift(ROLL_SHIFT)
                g[f"{col}_rollmean{w}"] = base.rolling(w).mean()
                g[f"{col}_rollstd{w}"] = base.rolling(w).std()
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)


def temporal_split(df, train_end="2024-01-01", val_end="2024-07-01"):
    """Découpe temporelle stricte (pas de shuffle)."""
    t = df["timestamp_utc"]
    train = df[t < train_end]
    val = df[(t >= train_end) & (t < val_end)]
    test = df[t >= val_end]
    return train, val, test


if __name__ == "__main__":
    df = pd.read_csv(CSV, parse_dates=["timestamp_utc"])
    feat = build_features(df)

    feat_cols = [c for c in feat.columns
                 if c.endswith(tuple(f"lag{l}" for l in LAGS))
                 or "roll" in c or c.startswith(("sin_", "cos_", "is_"))]
    print(f"dataset: {feat.shape} | {len(feat_cols)} features créées")
    print("features:", feat_cols)

    # sanity check fuite : lag24 == prix d'il y a 24h ?
    z = feat[feat.zone == "DK1"].sort_values("timestamp_utc").reset_index(drop=True)
    chk = (z["spot_price_eur"].iloc[168] == z["spot_price_eur_lag168"].iloc[336])
    print(f"\n[check fuite] spot lag168 cohérent : {chk}")

    tr, va, te = temporal_split(feat)
    print(f"\nsplit temporel : train={len(tr):,} (<2024) "
          f"val={len(va):,} (2024 S1) test={len(te):,} (2024 S2)")

    feat.to_parquet(OUT)
    print(f"\n✅ {OUT} ({feat.shape[0]:,} lignes × {feat.shape[1]} col)")
