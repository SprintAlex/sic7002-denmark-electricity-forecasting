# Synthèse — Prévision des prix de l'électricité au Danemark (SIC7002)

## 1. Objectif

Prévoir les prix de gros de l'électricité danoise (zones **DK1** et **DK2**, **2022-2024**, pas horaire)
avec deux familles de modèles — **XGBoost** (arbres) et **LSTM** (réseau récurrent) — et les comparer.
Deux cibles : le **prix spot day-ahead** (déterministe) et le **prix de déséquilibre** (stochastique).

## 2. Données

Dataset horaire, une ligne par (heure UTC, zone) : `denmark_electricity_dataset_2022_2024.csv`
(52 608 lignes). Trois sources publiques (APIs), agrégées et nettoyées :

| Variable | Source (dataset) |
|---|---|
| `spot_price_eur` (cible) | Energi Data Service — **`Elspotprices`** |
| `imbalance` = prix de déséquilibre (cible 2) | Energi Data Service — **`RegulatingBalancePowerdata`** (`ImbalancePriceEUR`) |
| `mfrr_activated` | idem (mFRR up − down) |
| `consumption_mwh` | Energi Data Service — **`ProductionConsumptionSettlement`** |
| `wind_speed`, `radia_glob`, `temp_dry` | DMI Open Data (`metObs`, moyenne par zone) |

**Corrections clés vs spécification initiale** (datasets périmés en 2025-2026, vérifiés en direct) :
`DayAheadPrices` vide → `Elspotprices` ; `ConsumptionDK3619CodeHour` (404) → `ProductionConsumptionSettlement` ;
`PowerSystemRightNow` sans historique imbalance avant 2024 → `RegulatingBalancePowerdata` (complet 2022-2024) ;
DMI sans clé depuis 12/2025.

## 3. Pipeline de données

Ingestion (avec gestion du rate-limit et cache) → uniformisation **UTC** (neutralise le piège du
changement d'heure été/hiver) → squelette horaire continu → fusion des 4 sources → nettoyage
(forward-fill des trous ≤ 2 h, drapeau `*_gap_flag` pour les trous plus longs) → validation
automatique (0 doublon, 0 rupture horaire, bornes physiques, prix négatifs conservés).
**Qualité finale : ~100 % de couverture, ~0.1 % NaN (heures-frontière DST uniquement).**

## 4. Analyse exploratoire (EDA)

- **Prix non-normaux** : moyenne ~125 €/MWh, écart-type ~115, **skew 2.07, kurtosis 5.56**, 2.6 % de
  prix négatifs → justifie d'abandonner les modèles linéaires.
- **Saisonnalité** (ACF) : pics nets à **24 h** (journalier) et **168 h** (hebdomadaire).
- **Couplage de marché** : spread DK1-DK2 concentré à 0 (interconnexion non congestionnée la
  plupart du temps).

## 5. Feature engineering

- Lags auto-régressifs (day-ahead : T-24, T-168), rolling stats (moyenne/écart 24 h, 168 h) décalées
  à l'horizon → **aucune fuite du futur** (vérifié).
- Calendaire cyclique (sin/cos heure, jour, mois), week-end, jours fériés danois.
- **Split temporel strict** : train < 2024, validation S1-2024, test S2-2024 (jamais d'aléatoire).
- Distinction **nowcast (1 h, utilise T-1)** vs **day-ahead (24 h, lag min = 24 h)** — on retient le
  day-ahead, conforme au nom et défendable.

## 6. Résultats — XGBoost vs LSTM (test S2-2024, prix spot day-ahead)

| Modèle | MAE | RMSE | Directional Acc. | Vitesse | Interprétable |
|---|---|---|---|---|---|
| Naïf (persistance T-24) | 36.50 | 55.75 | — | — | — |
| **XGBoost** | **25.63** | **38.80** | 72.6 % | secondes | ✅ SHAP |
| **LSTM** (168 h, 2 couches) | 26.40 | 42.42 | **75.1 %** | minutes | ❌ |

- Les deux battent nettement le naïf (~30 %). **Quasi à égalité** : XGBoost meilleur sur l'erreur
  ponctuelle + rapide + interprétable ; LSTM légèrement meilleur sur la direction.
- **SHAP** : drivers = `lag24` (38), puis **`wind_speed` (23)** et `temp_dry` (7) → le vent et la
  température pèsent réellement (thèse "windy Denmark" confirmée), pas que l'auto-régression.
- **Tuning** : gains marginaux (test MAE 25.75 ≈ 25.63) → le mur est la difficulté du problème.

## 7. Prévision probabiliste du déséquilibre

Régression quantile (P10/P50/P90) sur le prix d'imbalance : P50 MAE 42.8, **couverture [P10,P90]
= 67.8 %** (cible 80 %). L'imbalance est nettement plus dur que le spot et les intervalles
sous-couvrent → confirme la nécessité d'approches probabilistes (argument Sideratos de la proposal).

## 8. Limites & perspectives

- **Distribution shift S1→S2 2024** (val MAE < test MAE) sur les deux modèles → marché 2024 instable.
- Features météo = *actuals* (proxy de prévisions ; en production on utiliserait des forecasts).
- Quantiles sous-calibrés → QRF ou calibration conforme à explorer.
- Pistes : horizon multi-pas, détection de régime (pics), backtest P&L.

## 9. Reproductibilité

```bash
python -m src.build_dataset        # reconstruit le dataset (APIs + nettoyage)
python -m src.eda                  # figures EDA
HORIZON=24 python -m src.models.xgb_model          # XGBoost day-ahead
HORIZON=24 python -m src.models.shap_analysis      # interprétabilité
HORIZON=24 python -m src.models.quantile_imbalance # imbalance probabiliste
HORIZON=24 python -m src.models.lstm_model         # LSTM
```
Figures dans `reports/figures/` (10 PNG). Code versionné (git).
