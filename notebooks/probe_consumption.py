"""Trouve le bon dataset de consommation (ConsumptionDK3619CodeHour = 404).

Liste correctement les datasets EDS puis probe les candidats conso.
Économe en appels (rate-limit brutal) : 12s entre les requêtes.
"""
import json
import time
import requests

EDS = "https://api.energidataservice.dk"


def get(url, **params):
    for _ in range(6):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:
            print("   429 → 65s"); time.sleep(65); continue
        return r
    return r


def list_all():
    r = get(f"{EDS}/datasets")
    try:
        data = r.json()
    except Exception:
        print("non-JSON:", r.text[:300]); return []
    print("type:", type(data).__name__,
          "| keys/len:", list(data.keys()) if isinstance(data, dict) else len(data))
    # explorer la structure
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"  clé '{k}': {type(v).__name__} "
                  f"({len(v) if hasattr(v,'__len__') else v})")
            if isinstance(v, list) and v and isinstance(v[0], (str, dict)):
                names = [x if isinstance(x, str) else x.get("name", x) for x in v]
                cons = sorted(n for n in names if "onsum" in str(n).lower())
                print("   datasets conso:", cons)
                return names
    return data


def probe(dataset):
    print(f"\n--- {dataset} ---")
    r = get(f"{EDS}/dataset/{dataset}",
            start="2024-06-01T00:00", end="2024-06-01T03:00", limit=3)
    if not r.ok:
        print(f"  HTTP {r.status_code}: {r.text[:120]}"); return
    recs = r.json().get("records", [])
    if recs:
        print("  COLONNES:", list(recs[0].keys()))
        print("  ÉCH:", json.dumps(recs[0], ensure_ascii=False))
    else:
        print("  (0 enregistrement)")


if __name__ == "__main__":
    names = list_all(); time.sleep(12)
    # candidats conso horaire par zone connus de l'API EDS
    for cand in ["ConsumptionPerGroupSettlement", "ConsumptionDE35Hour",
                 "ConsumptionIndustry", "ProductionConsumptionSettlement",
                 "ConsumptionHour"]:
        probe(cand); time.sleep(12)
