"""Prix spot day-ahead (EUR/MWh) DK1/DK2.

schema_probe : DayAheadPrices renvoie 0 enregistrement via l'API (cassé/vide) ;
Elspotprices MARCHE et expose HourUTC (déjà en UTC, pas de conversion DST nécessaire),
PriceArea (DK1/DK2 présents) et SpotPriceEUR. On utilise donc Elspotprices.
"""
import pandas as pd

from config import ZONES, TZ_UTC
from src.api.eds import eds_fetch

COL_TIME = "HourUTC"
COL_AREA = "PriceArea"
COL_PRICE = "SpotPriceEUR"


def fetch_prices(start, end):
    recs = eds_fetch("Elspotprices", start, end,
                     columns=[COL_TIME, COL_AREA, COL_PRICE],
                     filters={COL_AREA: ZONES}, sort=COL_TIME)
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone", "spot_price_eur"])
    # HourUTC est déjà en UTC -> localiser directement, pas de conversion locale->UTC
    df["timestamp_utc"] = pd.to_datetime(df[COL_TIME]).dt.tz_localize(TZ_UTC)
    df = df.rename(columns={COL_AREA: "zone", COL_PRICE: "spot_price_eur"})
    out = df[["timestamp_utc", "zone", "spot_price_eur"]]
    return out.drop_duplicates(subset=["timestamp_utc", "zone"])
