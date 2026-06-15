"""Météo DMI par zone : wind_speed, radia_glob, temp_dry.

Moyenne spatiale non pondérée sur les stations de chaque zone (config.STATIONS).
"""
import pandas as pd

from config import STATIONS, DMI_PARAMETERS
from src.api.dmi import fetch_zone


def fetch_weather(start, end):
    zone_frames = []
    for zone, stations in STATIONS.items():
        param_frames = []
        for param in DMI_PARAMETERS:
            print(f"  DMI {zone} / {param}")
            param_frames.append(fetch_zone(stations, param, start, end))
        # merge des 3 variables sur timestamp
        z = param_frames[0]
        for pf in param_frames[1:]:
            z = z.merge(pf, on="timestamp_utc", how="outer")
        z["zone"] = zone
        zone_frames.append(z)
    out = pd.concat(zone_frames, ignore_index=True)
    return out.drop_duplicates(subset=["timestamp_utc", "zone"])
