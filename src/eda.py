"""EDA — analyse exploratoire du dataset prix élec DK.

Produit les figures et stats qui alimentent les slides "Price Behavior" :
  - distribution des prix (non-normalité, prix négatifs)
  - ACF (saisonnalité 24h / 168h)
  - profils horaire & hebdomadaire
  - spread DK1-DK2 (couplage de marché)
  - corrélations features <-> prix
Figures dans reports/figures/. Run: .venv/bin/python -m src.eda
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from config import PROCESSED_DIR, ROOT

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CSV = PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv"


def load():
    df = pd.read_csv(CSV, parse_dates=["timestamp_utc"])
    df = df.sort_values(["zone", "timestamp_utc"]).reset_index(drop=True)
    df["hour"] = df["timestamp_utc"].dt.hour
    df["dow"] = df["timestamp_utc"].dt.dayofweek  # 0=lundi
    df["month"] = df["timestamp_utc"].dt.month
    return df


def stats(df):
    print("=" * 60, "\nSTATS DESCRIPTIVES (prix spot EUR/MWh)\n", "=" * 60)
    for z in ["DK1", "DK2"]:
        s = df.loc[df.zone == z, "spot_price_eur"].dropna()
        print(f"\n{z}: n={len(s)} mean={s.mean():.1f} std={s.std():.1f} "
              f"min={s.min():.1f} max={s.max():.1f}")
        print(f"   skew={s.skew():.2f} kurtosis={s.kurtosis():.2f} "
              f"| % prix négatifs: {(s < 0).mean()*100:.2f}%")
        print(f"   quantiles 1/50/99: {s.quantile(.01):.1f} / "
              f"{s.median():.1f} / {s.quantile(.99):.1f}")


def fig_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, z in enumerate(["DK1", "DK2"]):
        s = df.loc[df.zone == z, "spot_price_eur"].dropna()
        ax[i].hist(s, bins=120, color="#2b6cb0", alpha=.8)
        ax[i].axvline(0, color="red", ls="--", lw=1, label="0 EUR/MWh")
        ax[i].set_title(f"Distribution prix spot — {z} (non-normale)")
        ax[i].set_xlabel("EUR/MWh"); ax[i].legend()
    fig.tight_layout(); fig.savefig(FIG / "01_distribution_prix.png", dpi=110)
    print(f"  -> {FIG/'01_distribution_prix.png'}")


def fig_acf(df):
    s = df.loc[df.zone == "DK1", "spot_price_eur"].dropna()
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_acf(s, lags=200, ax=ax, alpha=.05)
    for lag in (24, 48, 168):
        ax.axvline(lag, color="green", ls=":", lw=1)
    ax.set_title("ACF prix spot DK1 — pics à 24h (journalier) et 168h (hebdo)")
    fig.tight_layout(); fig.savefig(FIG / "02_acf_prix.png", dpi=110)
    print(f"  -> {FIG/'02_acf_prix.png'}")


def fig_profiles(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for z in ["DK1", "DK2"]:
        df[df.zone == z].groupby("hour")["spot_price_eur"].mean().plot(ax=ax[0], label=z)
        df[df.zone == z].groupby("dow")["spot_price_eur"].mean().plot(ax=ax[1], label=z)
    ax[0].set_title("Profil horaire moyen"); ax[0].set_xlabel("heure"); ax[0].legend()
    ax[1].set_title("Profil hebdomadaire moyen"); ax[1].set_xlabel("jour (0=lun)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "03_profils.png", dpi=110)
    print(f"  -> {FIG/'03_profils.png'}")


def fig_spread(df):
    p = df.pivot_table(index="timestamp_utc", columns="zone", values="spot_price_eur")
    spread = (p["DK1"] - p["DK2"]).dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(spread, bins=160, color="#805ad5")
    ax.set_title(f"Spread DK1-DK2 — couplage de marché "
                 f"(% à 0±1€: {(spread.abs() <= 1).mean()*100:.0f}%)")
    ax.set_xlabel("EUR/MWh"); ax.set_xlim(-50, 50)
    fig.tight_layout(); fig.savefig(FIG / "04_spread.png", dpi=110)
    print(f"  -> {FIG/'04_spread.png'}")


def fig_corr(df):
    cols = ["spot_price_eur", "imbalance", "consumption_mwh",
            "wind_speed", "radia_glob", "temp_dry"]
    c = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title("Corrélations features <-> prix")
    fig.tight_layout(); fig.savefig(FIG / "05_correlations.png", dpi=110)
    print(f"  -> {FIG/'05_correlations.png'}")
    print("\nCorrélations avec spot_price_eur:")
    print(c["spot_price_eur"].drop("spot_price_eur").round(3).to_string())


if __name__ == "__main__":
    df = load()
    stats(df)
    print("\nFIGURES:")
    fig_distribution(df); fig_acf(df); fig_profiles(df)
    fig_spread(df); fig_corr(df)
    print("\n✅ EDA terminée — figures dans reports/figures/")
