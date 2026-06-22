"""Orchestrateur : ingest -> merge -> clean -> validate -> export.

Run depuis la racine : .venv/bin/python -m src.build_dataset
Caches parquet dans data/raw/ ; pulls bruts CSV + final dans data/processed/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (START_DATE, END_DATE, ZONES, DATA_COLS, FINAL_COLS,
                    RAW_DIR, PROCESSED_DIR)
from src.ingest.prices import fetch_prices
from src.ingest.balance import fetch_balance
from src.ingest.consumption import fetch_consumption
from src.ingest.weather import fetch_weather
from src.ingest.forecasts import fetch_forecasts
from src.clean.merge import merge_all
from src.clean.missing import ffill_by_zone
from src.validate import validate


def _cached(name, fn):
    """Pull une source avec cache parquet + dump CSV brut."""
    cache = RAW_DIR / f"{name}.parquet"
    if cache.exists():
        print(f"[{name}] cache")
        return pd.read_parquet(cache)
    print(f"[{name}] téléchargement...")
    df = fn(START_DATE, END_DATE)
    df.to_parquet(cache)
    df.to_csv(PROCESSED_DIR / f"raw_{name}.csv", index=False)
    print(f"[{name}] {len(df):,} lignes")
    return df


def main():
    prices = _cached("prices", fetch_prices)
    balance = _cached("balance", fetch_balance)
    consumption = _cached("consumption", fetch_consumption)
    weather = _cached("weather", fetch_weather)
    forecasts = _cached("forecasts", fetch_forecasts)

    df = merge_all(prices, balance, consumption, weather, forecasts)
    df = ffill_by_zone(df, DATA_COLS, ZONES)

    print("\n=== VALIDATION ===")
    validate(df)

    flag_cols = [c for c in df.columns if c.endswith("_gap_flag")]
    out = df[FINAL_COLS + flag_cols].copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    path = PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv"
    out.to_csv(path, index=False)
    print(f"\n✅ Export : {path} ({len(out):,} lignes × {out.shape[1]} col)")


if __name__ == "__main__":
    main()
