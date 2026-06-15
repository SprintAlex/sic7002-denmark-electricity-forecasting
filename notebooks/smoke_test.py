"""Smoke test : valide le code de bout en bout sur une petite fenêtre.

Teste les 4 sources sur 2 jours + merge + clean + validate, sans lancer le run
complet (3 ans / DMI 30-60 min). Vérifie aussi la couverture historique 2022.
Run: .venv/bin/python -m notebooks.smoke_test
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import ZONES, DATA_COLS, STATIONS
from src.ingest.prices import fetch_prices
from src.ingest.balance import fetch_balance
from src.ingest.consumption import fetch_consumption
from src.api.dmi import fetch_zone
from src.clean.merge import merge_all
from src.clean.missing import ffill_by_zone
from src.validate import validate

S, E = "2024-06-01", "2024-06-02"


def show(name, df):
    print(f"\n### {name}: {df.shape}")
    print(df.head(3).to_string())
    return df


print("=== 1. PRICES (Elspotprices) ===")
prices = show("prices", fetch_prices(S, E))

print("\n=== 2. BALANCE (PowerSystemRightNow, 1min->1h) ===")
balance = show("balance", fetch_balance(S, E))

print("\n=== 3. CONSUMPTION (ProductionConsumptionSettlement) ===")
cons = show("consumption", fetch_consumption(S, E))

print("\n=== 4. WEATHER (DMI, 2 stations/zone, wind_speed seul) ===")
wrows = []
for z in ZONES:
    df = fetch_zone(STATIONS[z][:2], "wind_speed", S, E)
    df["zone"] = z
    wrows.append(df)
weather = show("weather (wind only)", pd.concat(wrows, ignore_index=True))

print("\n=== 5. MERGE + CLEAN (2 jours) ===")
# weather ici n'a que wind_speed -> ajouter colonnes manquantes pour le test
for c in ("radia_glob", "temp_dry"):
    weather[c] = pd.NA
df = merge_all(prices, balance, cons, weather)
df = ffill_by_zone(df, DATA_COLS, ZONES)
print(f"merged: {df.shape}")
validate(df)

print("\n=== 6. COUVERTURE HISTORIQUE 2022 ===")
for name, fn in [("prices", fetch_prices), ("balance", fetch_balance),
                 ("consumption", fetch_consumption)]:
    try:
        d = fn("2022-01-01", "2022-01-02")
        print(f"  {name} 2022-01-01: {len(d)} lignes "
              f"({'OK' if len(d) else 'VIDE ⚠️'})")
    except Exception as ex:
        print(f"  {name} 2022: ERREUR {ex}")
