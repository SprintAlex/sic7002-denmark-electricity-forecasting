"""LSTM day-ahead — prévision prix spot, comparaison directe avec XGBoost.

Même jeu de features no-leakage (HORIZON=24) que XGBoost, mais présenté en
SÉQUENCES glissantes (168h = 1 semaine) pour exploiter la structure temporelle.
Standardisation fit sur le train uniquement. Split temporel strict.

Run: HORIZON=24 .venv/bin/python -m src.models.lstm_model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import PROCESSED_DIR, ROOT, ZONES
from src.features import build_features, temporal_split
from src.models.xgb_model import feature_cols, TARGET, directional_accuracy

torch.manual_seed(42)
np.random.seed(42)
SEQ_LEN = 168
FIG = ROOT / "reports" / "figures"
DEV = "cpu"


class LSTMReg(nn.Module):
    # archi retenue par le tuning (lstm_tune.py, sélection sur val MAE) : 128x2, dropout 0.3
    def __init__(self, n_feat, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # dernier pas de temps


def make_sequences(df, cols, train_end, val_end):
    """Fenêtres glissantes par zone ; chaque séquence assignée par le temps de sa cible."""
    seqs, ys, times = [], [], []
    for z in ZONES:
        g = df[df.zone == z].sort_values("timestamp_utc").reset_index(drop=True)
        X = g[cols].values.astype(np.float32)
        y = g[TARGET].values.astype(np.float32)
        t = g["timestamp_utc"].values
        for i in range(SEQ_LEN - 1, len(g)):
            seqs.append(X[i - SEQ_LEN + 1: i + 1])
            ys.append(y[i]); times.append(t[i])
    seqs = np.stack(seqs); ys = np.array(ys, dtype=np.float32)
    times = pd.to_datetime(times)
    tr = times < train_end
    va = (times >= train_end) & (times < val_end)
    te = times >= val_end
    return seqs, ys, times, tr, va, te


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)
    cols = feature_cols(df)
    df = df.dropna(subset=cols + [TARGET]).reset_index(drop=True)

    X, y, times, tr, va, te = make_sequences(df, cols, "2024-01-01", "2024-07-01")
    print(f"séquences: train={tr.sum():,} val={va.sum():,} test={te.sum():,} "
          f"| seq_len={SEQ_LEN} feat={len(cols)}")

    # standardisation fit sur le train (features + cible)
    fsc = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
    Xs = fsc.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    ym, ys_ = y[tr].mean(), y[tr].std()
    yz = ((y - ym) / ys_).astype(np.float32)

    def loader(mask, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs[mask]), torch.from_numpy(yz[mask]))
        return DataLoader(ds, batch_size=256, shuffle=shuffle)

    dl_tr, dl_va = loader(tr, True), loader(va, False)
    model = LSTMReg(len(cols)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    lossf = nn.MSELoss()

    best_mae, best_state, patience = np.inf, None, 0
    for epoch in range(40):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            loss = lossf(model(xb.to(DEV)), yb.to(DEV))
            loss.backward(); opt.step()
        # val MAE en unités réelles
        model.eval(); preds = []
        with torch.no_grad():
            for xb, _ in dl_va:
                preds.append(model(xb.to(DEV)).cpu().numpy())
        vpred = np.concatenate(preds) * ys_ + ym
        vmae = mean_absolute_error(y[va], vpred)
        print(f"  epoch {epoch+1:2d}  val MAE {vmae:.2f}")
        if vmae < best_mae - 0.05:
            best_mae, best_state, patience = vmae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                print("  early stop"); break

    model.load_state_dict(best_state)
    model.eval(); preds = []
    with torch.no_grad():
        dl_te = loader(te, False)
        for xb, _ in dl_te:
            preds.append(model(xb.to(DEV)).cpu().numpy())
    tpred = np.concatenate(preds) * ys_ + ym
    yte = y[te]

    # référence directional = lag24 (présent dans les features, dé-standardisé)
    lag24_idx = cols.index("spot_price_eur_lag24")
    ref = X[te][:, -1, lag24_idx]
    mae = mean_absolute_error(yte, tpred)
    rmse = np.sqrt(mean_squared_error(yte, tpred))
    da = directional_accuracy(yte, tpred, ref)
    print(f"\n=== TEST (2024 S2) — LSTM ===")
    print(f"  LSTM     MAE={mae:.2f}  RMSE={rmse:.2f}  DirAcc={da*100:.1f}%")
    print(f"  (XGBoost day-ahead : MAE 25.63  RMSE 38.80  DirAcc 72.6%)")

    # plot 1 semaine
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(yte[:168], label="réel", lw=1.5)
    ax.plot(tpred[:168], label="LSTM", lw=1.5)
    ax.set_title("LSTM vs réel — 1 semaine de test"); ax.legend(); ax.set_ylabel("EUR/MWh")
    fig.tight_layout(); fig.savefig(FIG / "10_lstm_predictions.png", dpi=110)
    torch.save(model.state_dict(), PROCESSED_DIR / "lstm_spot.pt")
    print(f"\n✅ figure 10_lstm_predictions.png | modèle lstm_spot.pt")


if __name__ == "__main__":
    main()
