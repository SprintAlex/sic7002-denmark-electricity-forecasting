"""Utilitaires temps — conversion DST-safe et skeleton horaire UTC.

Le piège central du projet : les timestamps Energinet sont en heure locale
danoise → une heure dupliquée en octobre, une manquante en mars. On localise
avec ambiguous='infer' / nonexistent='shift_forward' avant de convertir en UTC.
"""
import pandas as pd

from config import TZ_DK, TZ_UTC, ZONES, START_DATE, END_DATE


def to_utc(series, tz_local=TZ_DK):
    """Série datetime locale danoise -> UTC, DST-safe."""
    s = pd.to_datetime(series)
    if s.dt.tz is None:
        s = s.dt.tz_localize(tz_local, ambiguous="infer", nonexistent="shift_forward")
    return s.dt.tz_convert(TZ_UTC)


def ensure_utc(df, col="timestamp_utc"):
    """Garantit que la colonne timestamp est tz-aware UTC."""
    s = pd.to_datetime(df[col])
    s = s.dt.tz_localize(TZ_UTC) if s.dt.tz is None else s.dt.tz_convert(TZ_UTC)
    return df.assign(**{col: s})


def build_skeleton(start=START_DATE, end=END_DATE, zones=ZONES):
    """Backbone : index horaire UTC continu × zones (produit cartésien)."""
    full = pd.date_range(start=f"{start} 00:00", end=f"{end} 23:00",
                         freq="H", tz=TZ_UTC)
    return pd.DataFrame([(ts, z) for ts in full for z in zones],
                        columns=["timestamp_utc", "zone"])
