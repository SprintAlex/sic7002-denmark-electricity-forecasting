"""Cherche un dataset EDS d'imbalance HISTORIQUE (2022-2023) par zone.

PowerSystemRightNow = temps réel (pas d'historique imbalance avant 2024).
On teste les datasets de balancing/regulating connus d'Energinet, qui sont
horaires et historiques. Throttle 12s (rate-limit EDS).
"""
import json
import time
import requests

EDS = "https://api.energidataservice.dk/dataset"

CANDIDATES = [
    "RegulatingBalancePowerdata",
    "ImbalancePrice",
    "BalancingPowerPriceData",
    "mFRRReservesDK1",
    "mFRReservesDK1",
    "FcrReservesDK1",
    "RealtimeMarket",
    "ElspotpricesImbalance",
    "ImbalanceSettlement",
    "BalancingMarketData",
]


def get(url, **p):
    for _ in range(6):
        r = requests.get(url, params=p, timeout=60)
        if r.status_code == 429:
            print("   429 → 65s"); time.sleep(65); continue
        return r
    return r


def probe(ds):
    print(f"\n--- {ds} ---")
    # teste sur une date HISTORIQUE (2022) pour vérifier la couverture
    r = get(f"{EDS}/{ds}", start="2022-06-01T00:00", end="2022-06-01T03:00", limit=3)
    if not r.ok:
        print(f"  HTTP {r.status_code}: {r.text[:90]}"); return
    recs = r.json().get("records", [])
    if not recs:
        print("  (0 enreg en 2022 — pas d'historique)"); return
    cols = list(recs[0].keys())
    imb = [c for c in cols if "mbalance" in c or "FRR" in c or "egulat" in c.lower()]
    print(f"  ✅ COLONNES: {cols}")
    print(f"     -> imbalance/regul/mFRR: {imb}")
    print(f"     ÉCH: {json.dumps(recs[0], ensure_ascii=False)[:300]}")


if __name__ == "__main__":
    for ds in CANDIDATES:
        probe(ds); time.sleep(12)
