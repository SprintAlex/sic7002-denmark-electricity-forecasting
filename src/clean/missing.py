"""Gestion des trous : forward-fill court, flag pour les trous longs.

Règle équipe : gap <= 2h -> forward-fill ; gap > 2h -> garder NaN + flag.
Le ffill se fait PAR zone (jamais à travers la frontière de zone).
"""
import numpy as np
import pandas as pd


def forward_fill_with_flag(df, cols, max_gap=2):
    """ffill les runs de NaN <= max_gap, flag les plus longs (*_gap_flag)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        is_null = df[col].isna()
        run_id = (~is_null).cumsum()
        run_len = is_null.groupby(run_id).transform("sum")
        long_gap = is_null & (run_len > max_gap)
        df[f"{col}_gap_flag"] = long_gap
        df[col] = df[col].ffill()
        df.loc[long_gap, col] = np.nan  # ré-ouvre les trous longs après ffill
    return df


def ffill_by_zone(df, cols, zones, max_gap=2):
    """Applique forward_fill_with_flag indépendamment par zone."""
    parts = []
    for zone in zones:
        z = df[df["zone"] == zone].sort_values("timestamp_utc")
        parts.append(forward_fill_with_flag(z, cols, max_gap))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["timestamp_utc", "zone"]).reset_index(drop=True)
