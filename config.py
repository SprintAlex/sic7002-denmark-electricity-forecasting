"""Configuration centrale du pipeline."""
from pathlib import Path
import pytz

# --- Périmètre (figé par l'équipe) ---
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
ZONES = ["DK1", "DK2"]

# --- Fuseaux ---
TZ_DK = pytz.timezone("Europe/Copenhagen")
TZ_UTC = pytz.UTC

# --- Chemins ---
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Endpoints ---
EDS_BASE = "https://api.energidataservice.dk/dataset"
DMI_BASE = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"

# --- Stations DMI par zone (de inspiration/, à valider) ---
# DK1 = Jutland + Funen, DK2 = Zealand + Bornholm
STATIONS_DK1 = ["06030", "06041", "06060", "06079", "06081",
                "06096", "06102", "06118", "06156", "06180"]
STATIONS_DK2 = ["06170", "06183", "06190", "06193", "06194", "06197", "06280"]
STATIONS = {"DK1": STATIONS_DK1, "DK2": STATIONS_DK2}

DMI_PARAMETERS = ["wind_speed", "radia_glob", "temp_dry"]

# --- Prévisions day-ahead production (EDS Forecasts_Hour, ForecastDayAhead) ---
# Connues la veille -> exogènes honnêtes pour le day-ahead (pas de fuite).
FORECAST_COLS = ["fc_offshore_da", "fc_onshore_da", "fc_solar_da",
                 "fc_wind_da", "fc_renew_da"]

# --- Schéma de sortie (ordre figé) ---
FINAL_COLS = ["timestamp_utc", "zone", "spot_price_eur", "imbalance",
              "mfrr_activated", "consumption_mwh", "wind_speed",
              "radia_glob", "temp_dry"] + FORECAST_COLS

DATA_COLS = FINAL_COLS[2:]  # colonnes sujettes au forward-fill / flag
