"""Client Energi Data Service.

Porté depuis le notebook de Duc (cell 8) — gère les 3 pièges de l'API EDS :
  - datetime ISO 'YYYY-MM-DDTHH:MM' (pas juste la date)
  - filter en JSON COMPACT (separators sans espaces, sinon HTTP 400)
  - sort = nom de colonne seul (pas de 'ASC'/'DESC')
+ gestion du rate-limit 429 (cooldown respecté) et pagination par offset.
"""
import json
import time
import requests

from config import EDS_BASE


def _to_dt(d: str) -> str:
    return d if "T" in d else f"{d}T00:00"


def eds_fetch(dataset, start, end, columns=None, filters=None,
              sort=None, limit=100_000, pause=0.5):
    """Récupère tous les enregistrements d'un dataset EDS sur [start, end]."""
    params = {"start": _to_dt(start), "end": _to_dt(end),
              "limit": limit, "offset": 0}
    if columns:
        params["columns"] = ",".join(columns)
    if filters:
        params["filter"] = json.dumps(filters, separators=(",", ":"))
    if sort:
        params["sort"] = sort

    records = []
    while True:
        r = _get_with_retry(f"{EDS_BASE}/{dataset}", params)
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("records", [])
        records.extend(batch)
        total = payload.get("total", 0)
        if len(records) >= total or not batch:
            break
        params["offset"] += limit
        time.sleep(pause)
    return records


def _get_with_retry(url, params, max_retries=6):
    for _ in range(max_retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:
            wait = 60
            try:
                wait = int(r.json()["message"].split("in ")[1].split(" ")[0]) + 3
            except Exception:
                pass
            print(f"  EDS 429 → attente {min(wait, 305)}s")
            time.sleep(min(wait, 305))
            continue
        return r
    return r
