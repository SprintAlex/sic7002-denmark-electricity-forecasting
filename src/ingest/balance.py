"""Prix de déséquilibre + activation mFRR depuis RegulatingBalancePowerdata.

schema_probe : PowerSystemRightNow (temps réel) n'a pas d'historique imbalance
avant 2024. RegulatingBalancePowerdata est HORAIRE, par PriceArea, et couvre
2022-2024 avec :
  - ImbalancePriceEUR  -> prix de déséquilibre (2e cible du projet, EUR/MWh)
  - mFRRUpActBal / mFRRDownActBal -> activation mFRR (MWh), on prend le net
C'est la source cohérente pour la research question ("stochastic imbalance prices").
"""
import pandas as pd

from config import ZONES, TZ_UTC
from src.api.eds import eds_fetch

COLS = ["HourUTC", "PriceArea", "ImbalancePriceEUR",
        "mFRRUpActBal", "mFRRDownActBal"]


def fetch_balance(start, end):
    recs = eds_fetch("RegulatingBalancePowerdata", start, end,
                     columns=COLS, filters={"PriceArea": ZONES}, sort="HourUTC")
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone", "imbalance", "mfrr_activated"])
    df["timestamp_utc"] = pd.to_datetime(df["HourUTC"]).dt.tz_localize(TZ_UTC)
    for c in ("ImbalancePriceEUR", "mFRRUpActBal", "mFRRDownActBal"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # imbalance = prix de déséquilibre ; mfrr_activated = activation nette (up - down)
    df["imbalance"] = df["ImbalancePriceEUR"]
    df["mfrr_activated"] = df["mFRRUpActBal"].fillna(0) - df["mFRRDownActBal"].fillna(0)
    df = df.rename(columns={"PriceArea": "zone"})
    out = df[["timestamp_utc", "zone", "imbalance", "mfrr_activated"]]
    return out.drop_duplicates(subset=["timestamp_utc", "zone"])
