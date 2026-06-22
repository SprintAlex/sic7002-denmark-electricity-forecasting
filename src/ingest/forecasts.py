"""Prévisions day-ahead éolien/solaire (MWh) DK1/DK2 — dataset EDS Forecasts_Hour.

ForecastDayAhead = prévision publiée la veille → DISPONIBLE avant le fixing du
prix spot day-ahead, donc utilisable comme feature SANS fuite (contrairement à
ForecastIntraday/Current, plus proches du temps réel).

Couverture vérifiée 100 % 2022-2024 (Offshore Wind, Onshore Wind, Solar), par zone.
On expose 3 colonnes brutes + 2 agrégats (vent total, renouvelable total).
"""
import pandas as pd

from config import ZONES, TZ_UTC
from src.api.eds import eds_fetch

COL_TIME, COL_AREA, COL_TYPE, COL_DA = "HourUTC", "PriceArea", "ForecastType", "ForecastDayAhead"
TYPE_MAP = {"Offshore Wind": "fc_offshore_da", "Onshore Wind": "fc_onshore_da", "Solar": "fc_solar_da"}


def fetch_forecasts(start, end):
    recs = eds_fetch("Forecasts_Hour", start, end,
                     columns=[COL_TIME, COL_AREA, COL_TYPE, COL_DA],
                     filters={COL_AREA: ZONES}, sort=COL_TIME)
    df = pd.DataFrame(recs)
    cols = ["timestamp_utc", "zone"] + list(TYPE_MAP.values()) + ["fc_wind_da", "fc_renew_da"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["timestamp_utc"] = pd.to_datetime(df[COL_TIME]).dt.tz_localize(TZ_UTC)  # déjà UTC
    df = df.rename(columns={COL_AREA: "zone"})
    df = df[df[COL_TYPE].isin(TYPE_MAP)]
    wide = (df.pivot_table(index=["timestamp_utc", "zone"], columns=COL_TYPE,
                           values=COL_DA, aggfunc="mean")
              .rename(columns=TYPE_MAP).reset_index())
    for c in TYPE_MAP.values():
        if c not in wide:
            wide[c] = pd.NA
    wide["fc_wind_da"] = wide["fc_offshore_da"].fillna(0) + wide["fc_onshore_da"].fillna(0)
    wide["fc_renew_da"] = wide["fc_wind_da"] + wide["fc_solar_da"].fillna(0)
    return wide[cols].drop_duplicates(subset=["timestamp_utc", "zone"])
