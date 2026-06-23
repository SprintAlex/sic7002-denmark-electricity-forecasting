"""Prix spot des zones voisines (couplage de marché) — EDS Elspotprices.

DK1/DK2 sont fortement couplés à l'Allemagne (DE), la Suède (SE3/SE4), la Norvège
(NO2) et au prix SYSTEM nordique. Ces prix sont un driver majeur du prix danois.

⚠️ Le prix day-ahead voisin CONTEMPORAIN sort de la MÊME enchère que DK (déterminé
simultanément) → l'utiliser tel quel = fuite. On n'expose que les prix bruts ici ;
les features ne prendront que des LAGS (>= horizon), construits en aval.
"""
import pandas as pd

from config import TZ_UTC
from src.api.eds import eds_fetch

AREAS = ["DE", "SE3", "SE4", "NO2", "SYSTEM"]
COL_TIME, COL_AREA, COL_PRICE = "HourUTC", "PriceArea", "SpotPriceEUR"
NB_COLS = [f"price_{a.lower()}" for a in AREAS]


def fetch_neighbor_prices(start, end):
    recs = eds_fetch("Elspotprices", start, end,
                     columns=[COL_TIME, COL_AREA, COL_PRICE],
                     filters={COL_AREA: AREAS}, sort=COL_TIME)
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc"] + NB_COLS)
    df["timestamp_utc"] = pd.to_datetime(df[COL_TIME]).dt.tz_localize(TZ_UTC)  # déjà UTC
    wide = df.pivot_table(index="timestamp_utc", columns=COL_AREA,
                          values=COL_PRICE, aggfunc="mean")
    wide.columns = [f"price_{c.lower()}" for c in wide.columns]
    return wide.reset_index().sort_values("timestamp_utc").reset_index(drop=True)
