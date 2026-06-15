"""Consommation horaire (MWh) DK1/DK2.

schema_probe : ConsumptionDK3619CodeHour (prévu par l'équipe) = 404, n'existe pas.
ProductionConsumptionSettlement MARCHE : HourUTC, PriceArea (DK1/DK2) et
GrossConsumptionMWh = conso brute horaire par zone, déjà en MWh (pas de /1000).
"""
import pandas as pd

from config import ZONES, TZ_UTC
from src.api.eds import eds_fetch

COL_TIME = "HourUTC"
COL_AREA = "PriceArea"
COL_CONS = "GrossConsumptionMWh"


def fetch_consumption(start, end):
    recs = eds_fetch("ProductionConsumptionSettlement", start, end,
                     columns=[COL_TIME, COL_AREA, COL_CONS],
                     filters={COL_AREA: ZONES}, sort=COL_TIME)
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone", "consumption_mwh"])
    df["timestamp_utc"] = pd.to_datetime(df[COL_TIME]).dt.tz_localize(TZ_UTC)
    df[COL_CONS] = pd.to_numeric(df[COL_CONS], errors="coerce")
    df = df.rename(columns={COL_AREA: "zone", COL_CONS: "consumption_mwh"})
    out = (df.groupby(["timestamp_utc", "zone"], as_index=False)["consumption_mwh"].sum())
    return out
