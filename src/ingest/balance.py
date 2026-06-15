"""Imbalance + mFRR depuis PowerSystemRightNow.

schema_probe a montré :
  - le dataset 'Realtime Electricity Market Data' / 'ConsumptionDK3619CodeHour'
    planifiés par l'équipe n'existent pas (404) ;
  - PowerSystemRightNow EXISTE encore et contient ImbalanceDK1/2 + mFRR_ActivatedDK1/2
    (contrairement au message de Thibaud). MAIS résolution 1 minute (Minutes1UTC).
On agrège donc à l'heure (moyenne) et on reshape wide -> long par zone.

⚠️ À confirmer : couverture historique de ce dataset jusqu'à 2022 (à vérifier au build).
"""
import pandas as pd

from config import ZONES
from src.api.eds import eds_fetch
from src.clean.timeutils import ensure_utc

COLS = ["Minutes1UTC", "ImbalanceDK1", "ImbalanceDK2",
        "mFRR_ActivatedDK1", "mFRR_ActivatedDK2"]


def fetch_balance(start, end):
    recs = eds_fetch("PowerSystemRightNow", start, end, columns=COLS, sort="Minutes1UTC")
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone", "imbalance", "mfrr_activated"])

    # Minutes1UTC est déjà en UTC -> heure tronquée, agrégation horaire (moyenne)
    df["timestamp_utc"] = pd.to_datetime(df["Minutes1UTC"]).dt.tz_localize("UTC").dt.floor("H")
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    hourly = df.groupby("timestamp_utc")[COLS[1:]].mean().reset_index()

    # wide -> long par zone
    parts = []
    for zone in ZONES:
        parts.append(pd.DataFrame({
            "timestamp_utc": hourly["timestamp_utc"],
            "zone": zone,
            "imbalance": hourly[f"Imbalance{zone}"],
            "mfrr_activated": hourly[f"mFRR_Activated{zone}"],
        }))
    out = pd.concat(parts, ignore_index=True)
    return ensure_utc(out).drop_duplicates(subset=["timestamp_utc", "zone"])
