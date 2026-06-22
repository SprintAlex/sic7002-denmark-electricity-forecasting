"""Merge de toutes les sources sur le skeleton horaire (timestamp_utc, zone)."""
from src.clean.timeutils import build_skeleton, ensure_utc


def merge_all(prices, balance, consumption, weather, forecasts=None):
    """left-join chaque source sur le skeleton ; expose les trous en NaN."""
    df = build_skeleton()
    for src in (prices, balance, consumption, weather, forecasts):
        if src is None or src.empty:
            continue
        df = df.merge(ensure_utc(src), on=["timestamp_utc", "zone"], how="left")
    return df.drop_duplicates(subset=["timestamp_utc", "zone"], keep="last")
