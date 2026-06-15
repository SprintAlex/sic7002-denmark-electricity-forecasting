"""Météo DMI par zone : wind_speed, radia_glob, temp_dry.

Moyenne spatiale non pondérée sur les stations de chaque zone (config.STATIONS).
"""
import pandas as pd

from config import STATIONS, DMI_PARAMETERS, RAW_DIR
from src.api.dmi import fetch_zone


def fetch_weather(start, end):
    zone_frames = []
    for zone, stations in STATIONS.items():
        param_frames = []
        for param in DMI_PARAMETERS:
            # cache par (zone, param) : un blip réseau ne perd pas la progression
            cache = RAW_DIR / f"weather_{zone}_{param}.parquet"
            if cache.exists():
                print(f"  DMI {zone} / {param} [cache]")
                param_frames.append(pd.read_parquet(cache))
                continue
            print(f"  DMI {zone} / {param}")
            df_p = fetch_zone(stations, param, start, end)
            df_p.to_parquet(cache)
            param_frames.append(df_p)
        # merge des 3 variables sur timestamp
        z = param_frames[0]
        for pf in param_frames[1:]:
            z = z.merge(pf, on="timestamp_utc", how="outer")
        z["zone"] = zone
        zone_frames.append(z)
    out = pd.concat(zone_frames, ignore_index=True)
    return out.drop_duplicates(subset=["timestamp_utc", "zone"])
