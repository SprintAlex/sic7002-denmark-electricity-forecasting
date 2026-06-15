"""Imbalance + mFRR depuis PowerSystemRightNow.

schema_probe a montré :
  - le dataset 'Realtime Electricity Market Data' / 'ConsumptionDK3619CodeHour'
    planifiés par l'équipe n'existent pas (404) ;
  - PowerSystemRightNow EXISTE encore et contient ImbalanceDK1/2 + mFRR_ActivatedDK1/2
    (contrairement au message de Thibaud). MAIS résolution 1 minute (Minutes1UTC).
On agrège donc à l'heure (moyenne) et on reshape wide -> long par zone.

⚠️ À confirmer : couverture historique de ce dataset jusqu'à 2022 (à vérifier au build).
"""
from datetime import datetime, timedelta

import pandas as pd

from config import ZONES
from src.api.eds import eds_fetch
from src.clean.timeutils import ensure_utc

COLS = ["Minutes1UTC", "ImbalanceDK1", "ImbalanceDK2",
        "mFRR_ActivatedDK1", "mFRR_ActivatedDK2"]


def _month_chunks(start, end, days=30):
    s, e = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=days - 1), e)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt + timedelta(days=1)


def fetch_balance(start, end):
    # PowerSystemRightNow est en résolution 1-min : sur 3 ans l'API tronque
    # (total peu fiable -> pagination stoppe à ~100k). On pull par chunks de 30j.
    recs = []
    for s, e in _month_chunks(start, end):
        recs.extend(eds_fetch("PowerSystemRightNow", s, e, columns=COLS, sort="Minutes1UTC"))
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "zone", "imbalance", "mfrr_activated"])

    # Minutes1UTC est déjà en UTC -> heure tronquée, agrégation horaire (moyenne)
    df["timestamp_utc"] = pd.to_datetime(df["Minutes1UTC"]).dt.tz_localize("UTC").dt.floor("h")
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
