"""Client DMI Open Data (metObs, OGC Features API).

Pas de clé requise depuis le 2/12/2025. Schéma confirmé par schema_probe :
properties = {parameterId, created, value, observed, stationId}, 'observed' en UTC ISO.

Corrige le bug du notebook : fetch_dmi_zone ne prend PAS de clé API.
"""
import time
import requests
import pandas as pd

from config import DMI_BASE, TZ_UTC


def fetch_station(station_id, parameter, start, end, limit=10_000):
    """Observations horaires d'une station pour une variable."""
    params = {
        "stationId": station_id,
        "parameterId": parameter,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit,
        "offset": 0,
    }
    rows = []
    while True:
        r = requests.get(DMI_BASE, params=params, timeout=60)
        r.raise_for_status()
        feats = r.json().get("features", [])
        for f in feats:
            p = f["properties"]
            rows.append({"timestamp_utc": pd.to_datetime(p["observed"]),
                         "station_id": station_id, parameter: p.get("value")})
        if len(feats) < limit:
            break
        params["offset"] += limit
        time.sleep(0.2)
    return pd.DataFrame(rows, columns=["timestamp_utc", "station_id", parameter])


def fetch_zone(stations, parameter, start, end, chunk_days=30):
    """Moyenne horaire non pondérée d'une variable sur les stations d'une zone."""
    from src.clean.timeutils import ensure_utc  # local import to avoid cycle
    frames = []
    for st in stations:
        for s, e in _chunks(start, end, chunk_days):
            try:
                frames.append(fetch_station(st, parameter, s, e))
            except Exception as ex:
                print(f"   DMI station {st} [{s}->{e}]: {ex}")
            time.sleep(0.1)
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", parameter])
    df = pd.concat(frames, ignore_index=True)
    df = ensure_utc(df)
    df[parameter] = pd.to_numeric(df[parameter], errors="coerce")
    df["timestamp_utc"] = df["timestamp_utc"].dt.floor("H")
    # moyenne intra-station par heure, puis moyenne entre stations
    per_station = df.groupby(["timestamp_utc", "station_id"])[parameter].mean().reset_index()
    return per_station.groupby("timestamp_utc")[parameter].mean().reset_index()


def _chunks(start, end, days):
    from datetime import datetime, timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=days - 1), e)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt + timedelta(days=1)
