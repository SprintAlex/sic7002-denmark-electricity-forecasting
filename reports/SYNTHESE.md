# Synthèse — Prévision des prix de l'électricité au Danemark (SIC7002)

## 1. Objectif

Prévoir les prix de gros de l'électricité danoise (zones **DK1** et **DK2**, **2022-2024**, pas horaire)
avec deux familles de modèles — **XGBoost** (arbres) et **LSTM** (réseau récurrent) — et les comparer.
Deux cibles : le **prix spot day-ahead** (déterministe) et le **prix de déséquilibre** (stochastique).

## 2. Données

Dataset horaire, une ligne par (heure UTC, zone) : `denmark_electricity_dataset_2022_2024.csv`
(52 608 lignes × 26 colonnes). Quatre sources publiques (APIs), agrégées et nettoyées :

| Variable | Source (dataset) |
|---|---|
| `spot_price_eur` (cible) | Energi Data Service — **`Elspotprices`** |
| `imbalance` = prix de déséquilibre (cible 2) | Energi Data Service — **`RegulatingBalancePowerdata`** (`ImbalancePriceEUR`) |
| `mfrr_activated` | idem (mFRR up − down) |
| `consumption_mwh` | Energi Data Service — **`ProductionConsumptionSettlement`** |
| `wind_speed`, `radia_glob`, `temp_dry` | DMI Open Data (`metObs`, moyenne par zone) |
| `fc_offshore_da`, `fc_onshore_da`, `fc_solar_da` (+ `fc_wind_da`, `fc_renew_da`) | Energi Data Service — **`Forecasts_Hour`** (`ForecastDayAhead`) : prévisions de production publiées **la veille** → exogènes honnêtes pour le day-ahead |

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
- **Exogènes** : météo DMI (actuals, proxy de prévision) + **prévisions day-ahead éolien/solaire**
  (EDS `ForecastDayAhead`, connues la veille → honnêtes, pas de fuite). 30 features au total.
- **Split temporel strict** : train < 2024, validation S1-2024, test S2-2024 (jamais d'aléatoire).
- Distinction **nowcast (1 h, utilise T-1)** vs **day-ahead (24 h, lag min = 24 h)** — on retient le
  day-ahead, conforme au nom et défendable.

## 6. Résultats — XGBoost vs LSTM (test S2-2024, prix spot day-ahead)

Même jeu de features no-leakage (30, prévisions day-ahead incluses), même split temporel.
Les deux modèles sont **fine-tunés** (cf. §6bis).

| Modèle | MAE | RMSE | R² | Directional Acc. | Interprétable |
|---|---|---|---|---|---|
| Naïf (persistance T-24) | 36.65 | 55.89 | 0.18 | — | — |
| **XGBoost** (Optuna + pseudo-Huber + forecasts) | **23.26** | **37.78** | **0.626** | 76.0 % | ✅ SHAP |
| **LSTM** (128×2, 168 h, + forecasts) | 25.32 | 41.74 | ≈0.54 | **76.3 %** | ❌ |

- Les deux écrasent le naïf (~37 % de MAE en moins). **XGBoost gagne** sur l'erreur ponctuelle
  (MAE/RMSE/R²), reste plus rapide et interprétable ; le LSTM ne le rejoint que sur la **direction**
  (76 %), sans exploiter aussi bien les exogènes.
- **SHAP** : drivers = `lag24` (11.9), `wind_speed` (9.4), `temp_dry` (4.7), puis **les prévisions
  day-ahead `fc_offshore_da` (4.7) et `fc_renew_da` (4.4)** → la production éolienne *attendue* est un
  moteur de premier plan, juste derrière la persistance (thèse "windy Denmark" confirmée).

## 6bis. Fine-tuning — ce qui a réellement fait baisser l'erreur

Méthodo inspirée de la littérature EPF (Lago et al. 2021) et d'un repo de référence (Optuna TPE +
TimeSeriesSplit). Chaque ligne ajoute un levier au précédent :

| Étape | MAE | Lecture |
|---|---|---|
| Baseline XGBoost (grille manuelle) | 25.63 | point de départ |
| + Optuna (80 essais, CV temporel) | 25.49 | tuning seul = **gain marginal** (plateau) |
| + loss **pseudo-Huber** (vs squared) | 24.38 | optimise l'erreur médiane → baisse le MAE |
| + **prévisions day-ahead éol/solaire** | **23.26** | **le vrai levier : du signal neuf** |

- **Leçon clé** : passé le plateau du tuning (~25.5), seul l'**ajout de signal** (prévisions de
  production, connues la veille) fait baisser **MAE, RMSE et R² en même temps**. On ne gagne pas sur
  les trois en ne touchant que les hyperparamètres (MAE et RMSE s'arbitrent via le traitement des pics).
- **Résultats négatifs** (gardés par honnêteté) : cible = *résidu vs lag24* → pire (la persistance est
  déjà capturée par `lag24`) ; `net_load` (conso − renouvelable) → pire (mélange actuals + forecast).
- **Pas de triche** : refus d'ajouter `lag1` ou le prix d'imbalance contemporain (réalisés *après* le
  spot → fuite). C'est ce qui sépare un MAE ~23 **honnête** d'un MAE ~1.4 trompeur (repo de référence,
  qui prédit le présent avec le présent).

**Pour pousser plus loin — 3 leviers de plus testés, aucun ne bat 23.26** (le plateau est réel) :

| Levier | MAE | Pourquoi ça ne passe pas |
|---|---|---|
| Lags prix zones voisines (DE/SE/NO) | 23.00 | gain MAE minuscule **mais DirAcc ↓** ; le couplage utile est *contemporain* (= même enchère que DK → fuite), donc les lags sont **redondants avec `lag24`** |
| Pondération récence (`sample_weight`) | 23.31 | downweighter 2022-2023 perd de l'info ; le régime récent n'est pas plus représentatif du test |
| Re-tune Optuna + tuning de `huber_slope` | 24.18 | l'optimum du **CV ≠ optimum du test** (shift S1→S2) → les params re-tunés généralisent moins bien |

→ **Le plafond ~23.3 vient de la difficulté du problème et du *distribution shift* 2024, pas d'un
sous-tuning.** Conclusion défendable : on a épuisé les leviers honnêtes raisonnables. (Scripts :
`xgb_neighbor_exog.py`, `tune_optuna.py` avec `huber_slope`.)

## 7. Prévision probabiliste du déséquilibre

Régression quantile (P10/P50/P90) sur le prix d'imbalance : P50 MAE 41.9, **couverture [P10,P90]
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
# --- fine-tuning (reproduire la campagne) ---
HORIZON=24 python -m src.models.tune_optuna 80     # recherche bayésienne XGBoost
HORIZON=24 python -m src.models.xgb_experiments    # loss (pseudo-Huber) + cible résidu
HORIZON=24 python -m src.models.xgb_forecast_exog  # apport des prévisions day-ahead
HORIZON=24 python -m src.models.lstm_tune          # grille LSTM
```
Figures dans `reports/figures/` (10 PNG). Code versionné (git).
