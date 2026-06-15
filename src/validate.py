"""Validation du dataset final : doublons, continuité, plages d'unités."""
import pandas as pd

from config import ZONES, DATA_COLS

# bornes de sanity par variable (prix négatifs tolérés : marché réel)
RANGES = {
    "spot_price_eur": (-500, 5000),
    "imbalance": (-5000, 5000),
    "mfrr_activated": (-3000, 3000),
    "consumption_mwh": (0, 10000),
    "wind_speed": (0, 50),
    "radia_glob": (0, 1500),
    "temp_dry": (-40, 45),
}


def validate(df):
    ok = True
    # 1. doublons
    dupes = df.duplicated(subset=["timestamp_utc", "zone"]).sum()
    print(f"[doublons (timestamp_utc, zone)] {dupes}")
    ok &= dupes == 0

    # 2. continuité horaire par zone
    for zone in ZONES:
        ts = pd.to_datetime(df.loc[df["zone"] == zone, "timestamp_utc"]).sort_values()
        gaps = ts.diff().dropna()
        bad = gaps[gaps != pd.Timedelta("1h")]
        print(f"[continuité {zone}] {len(bad)} trous horaires")
        ok &= bad.empty

    # 3. plages d'unités
    for col, (lo, hi) in RANGES.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        n_out = ((v < lo) | (v > hi)).sum()
        print(f"[plage {col:16s}] min={v.min():.1f} max={v.max():.1f} "
              f"hors[{lo},{hi}]={n_out}")

    # 4. rapport manquants
    print("\n--- manquants ---")
    for col in DATA_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"  {col:16s}: {n:6d} NaN ({n/len(df)*100:.2f}%)")
    return ok
