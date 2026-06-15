# Danish Electricity Spot Prices — Data Pipeline

Pipeline d'ingestion + nettoyage pour la prévision des prix spot de l'électricité danoise
(DK1/DK2, 2022-2024). Projet SIC7002 — phase données pour les modèles XGBoost & LSTM.

## Sortie

`data/processed/denmark_electricity_dataset_2022_2024.csv` — une ligne par (heure UTC, zone) :

```
timestamp_utc | zone | spot_price_eur | imbalance | mfrr_activated | consumption_mwh | wind_speed | radia_glob | temp_dry
```

Forward-fill des trous < 2h, colonnes `*_gap_flag` pour les trous plus longs.

## Sources

| Source | Dataset | Variables |
|--------|---------|-----------|
| Energi Data Service | `DayAheadPrices` | `spot_price_eur` |
| Energi Data Service | `PowerSystemRightNow` (1-min → horaire) | `imbalance`, `mfrr_activated` |
| Energi Data Service | *(dataset conso — voir notebooks/schema_probe)* | `consumption_mwh` |
| DMI Open Data | `metObs` (sans clé) | `wind_speed`, `radia_glob`, `temp_dry` |

⚠️ Plusieurs datasets prévus par l'équipe ont changé en 2025-2026 — voir
`notebooks/schema_probe*.py` pour les schémas réels validés en live.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# 1. Vérifier les schémas réels des APIs (étape de débloquage)
.venv/bin/python notebooks/schema_probe.py

# 2. Construire le dataset complet
.venv/bin/python -m src.build_dataset
```

## Architecture

- `config.py` — périmètre, stations DMI, schéma de sortie
- `src/api/` — clients EDS + DMI (gestion rate-limit, pagination, DST)
- `src/ingest/` — un module par source → DataFrame long `(timestamp_utc, zone, …)`
- `src/clean/` — DST-safe UTC, skeleton horaire, merge, forward-fill+flag
- `src/validate.py` — doublons, continuité, plages d'unités
- `inspiration/` — notebook initial de l'équipe (référence)

## Pièges gérés

- **DST** : timestamps locaux danois → heure dupliquée en octobre / manquante en mars.
  Géré via `tz_localize(..., ambiguous='infer', nonexistent='shift_forward')`.
- **Prix négatifs** : réels sur ce marché, conservés.
- **Rate-limit EDS** : cooldown 429 respecté automatiquement.
