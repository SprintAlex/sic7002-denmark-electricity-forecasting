"""
Schema probe v2 — résout les inconnues laissées par v1.

v1 a montré : ConsumptionDK3619CodeHour 404, PowerSystemRightNow vivant (1-min),
DayAheadPrices vide avec filtre. Ici on :
  - liste les datasets EDS contenant 'onsumption' / 'price' / 'spot'
  - probe DayAheadPrices SANS filtre (voir vraies valeurs PriceArea + couverture)
  - probe Elspotprices (fallback prix)
  - confirme la résolution de PowerSystemRightNow

EDS rate-limit ~ agressif → on espace les appels (8s) et on attend en tête si besoin.
Run: .venv/bin/python notebooks/schema_probe2.py
"""
import json
import time
import sys
import requests

EDS = "https://api.energidataservice.dk/dataset"


def get(url, **params):
    for attempt in range(6):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:
            wait = 60
            try:
                wait = int(r.json().get("message", "").split("in ")[1].split(" ")[0]) + 3
            except Exception:
                pass
            print(f"   429 → attente {wait}s...")
            time.sleep(min(wait, 305))
            continue
        return r
    return r


def list_datasets():
    print("=" * 70, "\nDATASETS EDS (mots-clés conso / prix)\n", "=" * 70)
    r = get("https://api.energidataservice.dk/datasets")
    try:
        data = r.json()
    except Exception:
        print("  réponse non-JSON:", r.text[:200]); return
    # structure inconnue → normaliser en liste de noms
    if isinstance(data, dict):
        data = data.get("datasets") or data.get("result") or list(data.values())
    names = []
    for d in data:
        names.append(d.get("name") if isinstance(d, dict) else str(d))
    for kw in ("onsumption", "Consumption", "Price", "Spot", "Elspot", "Load"):
        hits = sorted({n for n in names if n and kw.lower() in n.lower()})
        if hits:
            print(f"  [{kw}] {hits}")
    print(f"  (total {len(names)} datasets)")


def probe(dataset, columns=None, filters=None, sort=None):
    print("\n" + "=" * 70, f"\n{dataset}\n", "=" * 70)
    p = {"start": "2024-06-01T00:00", "end": "2024-06-01T06:00", "limit": 5, "offset": 0}
    if columns:
        p["columns"] = ",".join(columns)
    if filters:
        p["filter"] = json.dumps(filters, separators=(",", ":"))
    if sort:
        p["sort"] = sort
    r = get(f"{EDS}/{dataset}", **p)
    if not r.ok:
        print(f"  HTTP {r.status_code}: {r.text[:200]}"); return
    recs = r.json().get("records", [])
    if not recs:
        print("  (0 enregistrement)"); return
    print(f"  COLONNES: {list(recs[0].keys())}")
    print(f"  ÉCHANTILLON: {json.dumps(recs[0], ensure_ascii=False)}")
    # valeurs distinctes de PriceArea si présent
    if "PriceArea" in recs[0]:
        r2 = get(f"{EDS}/{dataset}", start="2024-06-01T00:00", end="2024-06-01T01:00",
                 columns="PriceArea", limit=100)
        areas = sorted({x.get("PriceArea") for x in r2.json().get("records", [])})
        print(f"  PriceArea distinctes: {areas}")


if __name__ == "__main__":
    if "--wait" in sys.argv:
        print("Attente cooldown 305s avant probe..."); time.sleep(305)
    list_datasets(); time.sleep(8)
    probe("DayAheadPrices"); time.sleep(8)            # sans filtre → vraies valeurs
    probe("Elspotprices"); time.sleep(8)              # fallback prix historique
    probe("PowerSystemRightNow", columns=["Minutes1UTC", "ImbalanceDK1", "ImbalanceDK2",
          "mFRR_ActivatedDK1", "mFRR_ActivatedDK2"])
