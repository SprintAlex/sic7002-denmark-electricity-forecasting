"""
Schema probe — étape qui débloque le pipeline.

Inspecte les schémas RÉELS des datasets (les noms de colonnes ont changé en
2025-2026 et étaient devinés dans le notebook de Duc). On pull 1 seule journée
par source et on imprime les colonnes + un échantillon.

Run: .venv/bin/python notebooks/schema_probe.py
"""
import json
import requests

EDS_BASE = "https://api.energidataservice.dk/dataset"
DAY = ("2024-06-01T00:00", "2024-06-02T00:00")


def probe_eds(dataset, filters=None):
    params = {"start": DAY[0], "end": DAY[1], "limit": 5, "offset": 0}
    if filters:
        params["filter"] = json.dumps(filters, separators=(",", ":"))
    print(f"\n{'='*70}\nEDS dataset: {dataset}\n{'='*70}")
    try:
        r = requests.get(f"{EDS_BASE}/{dataset}", params=params, timeout=60)
        print(f"URL: {r.url}")
        if not r.ok:
            print(f"  HTTP {r.status_code}: {r.text[:300]}")
            return
        recs = r.json().get("records", [])
        if not recs:
            print("  (aucun enregistrement)")
            return
        print(f"  COLONNES: {list(recs[0].keys())}")
        print(f"  ÉCHANTILLON: {json.dumps(recs[0], indent=2, ensure_ascii=False)}")
    except Exception as ex:
        print(f"  ERREUR: {ex}")


def list_eds_datasets():
    """Liste les datasets EDS dont le nom matche un mot-clé."""
    print(f"\n{'='*70}\nRecherche datasets EDS (imbalance / realtime / consumption)\n{'='*70}")
    try:
        r = requests.get("https://api.energidataservice.dk/datasets", timeout=60)
        names = [d.get("name", d) if isinstance(d, dict) else d for d in r.json()]
        for kw in ("mbalance", "ealtime", "onsumption", "ower", "FRR", "ayAhead"):
            hits = [n for n in names if kw in str(n)]
            if hits:
                print(f"  [{kw}] -> {hits}")
    except Exception as ex:
        print(f"  ERREUR listing: {ex}")


def probe_dmi():
    print(f"\n{'='*70}\nDMI metObs — structure d'une observation\n{'='*70}")
    url = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"
    params = {
        "parameterId": "wind_speed",
        "datetime": "2024-06-01T00:00:00Z/2024-06-01T02:00:00Z",
        "limit": 3,
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        print(f"URL: {r.url}  -> HTTP {r.status_code}")
        if not r.ok:
            print(f"  {r.text[:300]}")
            return
        feats = r.json().get("features", [])
        if feats:
            print(f"  PROPERTIES: {list(feats[0]['properties'].keys())}")
            print(f"  ÉCHANTILLON: {json.dumps(feats[0]['properties'], indent=2)}")
    except Exception as ex:
        print(f"  ERREUR: {ex}")


if __name__ == "__main__":
    list_eds_datasets()
    probe_eds("DayAheadPrices", {"PriceArea": ["DK1", "DK2"]})
    probe_eds("ConsumptionDK3619CodeHour", {"PriceArea": ["DK1", "DK2"]})
    # Realtime: nom de dataset à confirmer via le listing ci-dessus
    for cand in ("RealtimeMarket", "RealTimeMarket", "ElectricityBalance", "PowerSystemRightNow"):
        probe_eds(cand)
    probe_dmi()
