# Competition Submissions

This folder contains the synthetic normal datasets generated for the NSL-KDD and UNSW-NB15 anomaly detection tracks. Both files use a 65% TabDDPM and 35% protocol-grouped GMM V2 ensemble with LightGBM boundary filtering.

## Submission Files

| File | Dataset | Rows | Ensemble Split | LightGBM Threshold | Local Validation | Kaggle Public Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`submission_nsl_kdd.csv`](submission_nsl_kdd.csv) | NSL-KDD | 40,000 | 26,000 TabDDPM (65%) + 14,000 GMM V2 (35%) | $\tau \le 0.04$ | 0.95429 | 0.93113 |
| [`submission_unsw_nb15.csv`](submission_unsw_nb15.csv) | UNSW-NB15 | 70,000 | 45,500 TabDDPM (65%) + 24,500 GMM V2 (35%) | $\tau \le 0.05$ | 0.63032 | 0.60013 |

## Ensemble Methodology

The generation pipeline combines two models and applies two post-processing steps:

1. TabDDPM generates 65% of the final dataset. It samples from the trained reverse diffusion checkpoint (`results/models/{dataset}/tabddpm_best.pt`) over 1,000 cosine-scheduled timesteps. Diffusion models capture non-linear feature relationships and heavy-tailed numeric distributions, which improves downstream ECOD detection.

2. Protocol-grouped GMM V2 generates the remaining 35%. Separate Gaussian mixture models are fitted per protocol group (`tcp` and `non-tcp` for NSL-KDD; `tcp`, `udp`, and `other` for UNSW-NB15), with component-conditional donor sampling for categorical columns. Grouping by protocol prevents continuous parameters from bleeding across different traffic types and maintains mode coverage for Isolation Forest.

3. A supervised LightGBM classifier (`results/models/boundary_filter_{dataset}.pkl`), trained on the labeled training data (`is_anomaly == 0` vs. `is_anomaly == 1`), scores all generated rows. Candidates with an anomaly probability above $\tau$ ($\tau = 0.04$ for NSL-KDD, $\tau = 0.05$ for UNSW-NB15) are removed so synthetic normal points do not overlap with attack regions.

4. Numeric columns are rounded to match the decimal precision of `train.csv`. Domain rules are also applied to UNSW-NB15: TCP window and sequence fields (`swin`, `stcpb`, `dtcpb`, `dwin`) are set to zero for non-TCP rows, `tcprtt` is set to `synack + ackdat`, and mean packet sizes (`smean`, `dmean`) are recomputed from byte and packet counts.

## Reproduction

Run the pipeline from the repository root to regenerate both files from the trained models:

```bash
python run_pipeline.py
```

Historical runs and ratio comparisons are recorded in [`EXPERIMENT_TRACKER.md`](EXPERIMENT_TRACKER.md).
