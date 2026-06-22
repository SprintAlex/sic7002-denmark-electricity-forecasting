"""Tuning LSTM — recherche sur une grille curée d'hyperparamètres.

Mêmes features no-leakage que XGBoost. On explore taille cachée, nb de couches,
dropout, learning rate et lookback. Sélection par MAE de validation (2024 S1),
report final sur le test (2024 S2). Inspiré de Kılıç et al. (2024).

Run: HORIZON=24 .venv/bin/python -m src.models.lstm_tune
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import PROCESSED_DIR, ZONES
from src.features import build_features, temporal_split, HORIZON
from src.models.xgb_model import feature_cols, TARGET, directional_accuracy

torch.manual_seed(42); np.random.seed(42)

# grille curée (raisonnable sur CPU) : 6 configs
GRID = [
    dict(hidden=64,  layers=2, dropout=0.2, lr=1e-3, seq=168),
    dict(hidden=128, layers=2, dropout=0.2, lr=1e-3, seq=168),
    dict(hidden=64,  layers=2, dropout=0.3, lr=5e-4, seq=168),
    dict(hidden=128, layers=2, dropout=0.3, lr=5e-4, seq=168),
    dict(hidden=64,  layers=1, dropout=0.1, lr=1e-3, seq=48),
    dict(hidden=128, layers=3, dropout=0.3, lr=5e-4, seq=168),
]


class LSTMReg(nn.Module):
    def __init__(self, n_feat, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(df, cols, seq_len, train_end, val_end):
    seqs, ys, times = [], [], []
    for z in ZONES:
        g = df[df.zone == z].sort_values("timestamp_utc").reset_index(drop=True)
        X = g[cols].values.astype(np.float32)
        y = g[TARGET].values.astype(np.float32)
        t = g["timestamp_utc"].values
        for i in range(seq_len - 1, len(g)):
            seqs.append(X[i - seq_len + 1: i + 1]); ys.append(y[i]); times.append(t[i])
    seqs = np.stack(seqs); ys = np.array(ys, np.float32); times = pd.to_datetime(times)
    return (seqs, ys, times < train_end,
            (times >= train_end) & (times < val_end), times >= val_end)


def train_eval(cfg, df, cols):
    X, y, tr, va, te = make_sequences(df, cols, cfg["seq"], "2024-01-01", "2024-07-01")
    fsc = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
    Xs = fsc.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    ym, ysd = y[tr].mean(), y[tr].std()
    yz = ((y - ym) / ysd).astype(np.float32)

    def loader(m, sh): return DataLoader(TensorDataset(torch.from_numpy(Xs[m]),
                       torch.from_numpy(yz[m])), batch_size=256, shuffle=sh)
    dl_tr, dl_va = loader(tr, True), loader(va, False)
    model = LSTMReg(len(cols), cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    lossf = nn.MSELoss()

    best, best_state, pat = np.inf, None, 0
    for ep in range(30):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
        model.eval(); p = []
        with torch.no_grad():
            for xb, _ in dl_va: p.append(model(xb).numpy())
        vmae = mean_absolute_error(y[va], np.concatenate(p) * ysd + ym)
        if vmae < best - 1e-3:
            best, best_state, pat = vmae, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 4: break

    model.load_state_dict(best_state); model.eval(); p = []
    with torch.no_grad():
        for xb, _ in loader(te, False): p.append(model(xb).numpy())
    tpred = np.concatenate(p) * ysd + ym
    yte = y[te]
    lag24_idx = cols.index("spot_price_eur_lag24")
    ref = X[te][:, -1, lag24_idx]
    return dict(val_mae=best,
                mae=mean_absolute_error(yte, tpred),
                rmse=np.sqrt(mean_squared_error(yte, tpred)),
                r2=r2_score(yte, tpred),
                da=directional_accuracy(yte, tpred, ref) * 100)


def main():
    df = pd.read_csv(PROCESSED_DIR / "denmark_electricity_dataset_2022_2024.csv",
                     parse_dates=["timestamp_utc"])
    df = build_features(df)
    df["zone_id"] = (df["zone"] == "DK2").astype(int)
    cols = feature_cols(df)
    df = df.dropna(subset=cols + [TARGET]).reset_index(drop=True)

    print(f"HORIZON={HORIZON} | {len(GRID)} configs LSTM\n")
    results = []
    for i, cfg in enumerate(GRID, 1):
        r = train_eval(cfg, df, cols)
        results.append((cfg, r))
        print(f"  [{i}/{len(GRID)}] {cfg} -> val {r['val_mae']:.2f} | "
              f"test MAE {r['mae']:.2f} RMSE {r['rmse']:.2f} R2 {r['r2']:.3f} DA {r['da']:.1f}%")

    best_cfg, best_r = min(results, key=lambda x: x[1]["val_mae"])
    print(f"\n>>> MEILLEUR (val): {best_cfg}")
    print(f"    TEST MAE={best_r['mae']:.2f} RMSE={best_r['rmse']:.2f} "
          f"R2={best_r['r2']:.3f} DirAcc={best_r['da']:.1f}%")
    print(f"    (LSTM baseline non-tuné : MAE 26.40 | XGBoost Optuna : voir log)")


if __name__ == "__main__":
    main()
