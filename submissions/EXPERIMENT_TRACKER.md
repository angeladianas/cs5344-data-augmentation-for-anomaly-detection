# Experiment Tracker

## UNSW-NB15 ($N = 70,000$)

| Rank | Recipe | Filter ($\tau$) | Local $S_{\text{val}}$ | AUPRC ECOD | AUPRC IForest | Kaggle Public Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 (Submitted) | 65% Deep TabDDPM + 35% Grouped GMM V2 | $\le 0.05$ | 0.63032 | 0.66507 | 0.59557 | 0.60013 |
| 2 | 64% Deep TabDDPM + 36% Grouped GMM V2 | $\le 0.05$ | 0.63253 | 0.66611 | 0.59894 | 0.59984 |
| 3 | 62% Deep TabDDPM + 38% Grouped GMM V2 | $\le 0.05$ | 0.62804 | 0.66577 | 0.59032 | 0.59723 |
| 4 | 68% Deep TabDDPM + 32% Grouped GMM V2 | $\le 0.05$ | 0.63043 | 0.65990 | 0.60096 | 0.59504 |
| 5 | 51% TabDDPM + 49% Grouped GMM V2 | $\le 0.05$ | 0.62650 | 0.63612 | 0.61688 | 0.59500 |
| 6 | 52% TabDDPM + 48% Grouped GMM V2 | $\le 0.05$ | - | - | - | 0.59312 |
| 7 | 65% Deep TabDDPM + 35% Grouped GMM V2 (5-Fold LightGBM) | $\le 0.05$ | 0.63492 | 0.65573 | 0.61412 | 0.59302 |
| 8 | 50% TabDDPM + 50% Grouped GMM V2 | $\le 0.05$ | - | - | - | 0.59232 |
| 9 | 44% TabDDPM + 56% Grouped GMM V2 (130k Pool) | $\le 0.05$ | 0.61994 | 0.62673 | 0.61315 | 0.58614 |
| 10 | 70% TabDDPM + 30% Grouped GMM V2 | $\le 0.05$ | 0.60416 | - | - | 0.56895 |
| 11 | 70% Pre-Filtered TabDDPM + 30% Grouped GMM V2 | $\le 0.05$ | - | - | - | 0.53646 |
| 12 | 50% TabDDPM + 50% Ungrouped GMM V1 | $\le 0.02$ | - | - | - | 0.53122 |
| 13 | 100% TabDDPM | $\le 0.05$ | - | - | - | 0.48217 |

## NSL-KDD ($N = 40,000$)

| Rank | Recipe | Filter ($\tau$) | Local $S_{\text{val}}$ | AUPRC ECOD | AUPRC IForest | Kaggle Public Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 70% Deep TabDDPM + 30% Grouped GMM V2 | $\le 0.04$ | 0.95498 | 0.94033 | 0.96962 | 0.93115 |
| 2 (Submitted) | 65% Deep TabDDPM + 35% Grouped GMM V2 | $\le 0.04$ | 0.95429 | 0.93863 | 0.96995 | 0.93113 |
| 3 | 75% Deep TabDDPM + 25% Grouped GMM V2 | $\le 0.04$ | 0.95557 | 0.94120 | 0.96994 | 0.93099 |
| 3 | 59% TabDDPM + 41% Grouped GMM V2 (85k Pool) | $\le 0.04$ | 0.95388 | 0.93748 | 0.97049 | 0.93099 |
| 5 | 69% Deep TabDDPM + 31% Grouped GMM V2 | $\le 0.04$ | 0.95513 | 0.93995 | 0.97030 | 0.93097 |
| 6 | 60% TabDDPM + 40% Grouped GMM V2 | $\le 0.04$ | - | - | - | 0.93092 |
| 7 | 80% Deep TabDDPM + 20% Grouped GMM V2 | $\le 0.04$ | 0.95570 | 0.94164 | 0.96976 | 0.93083 |
| 8 | 75% TabDDPM + 25% Grouped GMM V2 (50k Pool) | $\le 0.04$ | - | - | - | 0.93072 |
| 9 | 70% TabDDPM + 30% Ungrouped GMM V1 | $\le 0.05$ | - | - | - | 0.93050 |
| 10 | 70% TabDDPM + 30% Grouped GMM V2 (52k Pool) | $\le 0.05$ | 0.95222 | 0.93532 | 0.96913 | 0.93029 |

## Recipe Terminology

- TabDDPM (Standard vs. Deep Pool): Early runs sampled from a smaller candidate pool (50,000 rows for NSL-KDD, 70,000 rows for UNSW-NB15), where strict boundary filtering ran out of clean diffusion rows at high TabDDPM ratios. Deep TabDDPM expands the raw candidate pool (110,000 rows for NSL-KDD, 210,000 rows for UNSW-NB15) so high TabDDPM ratios only select low-probability normal rows.
- Pre-Filtered TabDDPM: An early TabDDPM export that was pre-sorted by lowest anomaly probability before ensembling, which dropped rare normal protocols and lowered performance.
- Ungrouped GMM V1 vs. Grouped GMM V2: GMM V1 fits a single Gaussian mixture across the entire dataset. Grouped GMM V2 fits separate Gaussian mixtures for each protocol group (`tcp`, `udp`, `other`), preventing continuous feature distributions from mixing across protocols.
- LightGBM Filter (Single vs. 5-Fold): Standard runs use a single LightGBM classifier to reject candidates with anomaly probability above $\tau$. The 5-Fold variant averages predictions across five stratified folds.
