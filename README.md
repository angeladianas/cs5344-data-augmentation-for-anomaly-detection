# CS5344: Tabular Data Augmentation for Anomaly Detection

**Course:** CS5344 Big Data Analytics Technology (Semester 1)  
**Team:** Group 39 (Diana, Marco, Tan Yi Kai)  
**Task:** Generate high-utility synthetic normal network traffic for downstream unsupervised anomaly detectors (**ECOD** and **Isolation Forest**).

---

## 1. Executive Summary & Submission Results

This repository contains the complete, reproducible source code for Group 39's final competition submissions on **NSL-KDD** and **UNSW-NB15**. 

Our framework integrates a **Dual-Generator Ensemble** with **Supervised Boundary Negative Filtering** and **Network RFC Physics Repair**:
1. **Deep TabDDPM (65%)**: Continuous-discrete hybrid score-based diffusion model capturing high-order non-linear feature couplings and heavy-tailed traffic flows.
2. **Protocol-Grouped GMM V2 (35%)**: Stratified Gaussian Mixture Models fitted per protocol group (`tcp`, `udp`, `other`), paired with component-conditional donor sampling to preserve multi-modal categorical combinations.
3. **LightGBM Negative Boundary Filter**: Discards synthetic candidates whose predicted anomaly probability exceeds boundary rejection threshold $\tau$ ($\tau \le 0.04$ for NSL-KDD, $\tau \le 0.05$ for UNSW-NB15) to guarantee clean normal separation from attack manifolds.
4. **Physical Invariant Calibration**: Enforces deterministic network protocol constraints (RFC TCP window/sequence rules, timing identity $\text{tcprtt} = \text{synack} + \text{ackdat}$, and byte/packet count consistency).

### Official Competition Scores

| Dataset | Submission Target | Ensemble Split | Boundary Threshold ($\tau$) | Local Validation ($S_{\text{val}}$) | Kaggle Public Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NSL-KDD** | Exactly 40,000 rows | 65% TabDDPM + 35% GMM V2 | $\tau \le 0.04$ | **0.95429** | **0.93113** |
| **UNSW-NB15** | Exactly 70,000 rows | 65% TabDDPM + 35% GMM V2 | $\tau \le 0.05$ | **0.63032** | **0.60013** |

*Evaluation Metric:* $S = \frac{1}{2} \left( \text{AUPRC}_{\text{ECOD}} + \text{AUPRC}_{\text{IForest}} \right)$, evaluated against hidden ground truth labels.

---

## 2. One-Command Master Reproduction

To regenerate both verified submission CSV files directly from the trained models:

```bash
python run_pipeline.py
```

### Dataset-Specific Execution
```bash
python run_pipeline.py --dataset nsl_kdd     # Generates submissions/submission_nsl_kdd.csv (40,000 rows)
python run_pipeline.py --dataset unsw_nb15   # Generates submissions/submission_unsw_nb15.csv (70,000 rows)
```

The pipeline automatically:
1. Loads the trained diffusion checkpoints (`results/models/{dataset}/tabddpm_best.pt`) and protocol-grouped GMM models.
2. Draws raw candidate pools with adaptive headroom.
3. Decodes continuous and categorical features via `TabularDecoder`.
4. Prunes candidates crossing the LightGBM decision boundary (`results/models/boundary_filter_{dataset}.pkl`).
5. Applies decimal precision alignment and physical protocol invariants.
6. Formats and exports submission files to `submissions/`.

---

## 3. Interactive Walkthrough & Exploratory Workings

For an interactive, cell-by-cell walkthrough with visualizations, mathematical justifications, ratio sweeps, and boundary filtering curves, open:
- [`notebook/step_5_ensemble.ipynb`](notebook/step_5_ensemble.ipynb)

All notebooks across the project's development milestones are organized under [`notebook/`](notebook/) to showcase exploratory workings and sprint analyses, while the core production algorithms have been fully refactored into the modular Python codebase (`models/` and `utils/`).

A comprehensive version-by-version log of candidate sources, filtering parameters, local validation scores ($S_{\text{val}}$), and official Kaggle public scores is maintained in:
- [`submissions/EXPERIMENT_TRACKER.md`](submissions/EXPERIMENT_TRACKER.md)

---

## 4. Repository Structure

```
cs5344-data-augmentation-for-anomaly-detection/
├── run_pipeline.py                 # Master reproduction CLI entry point
├── pyproject.toml                  # Project dependencies and tool configurations
├── uv.lock                         # Exact reproducible package lockfile
│
├── data/                           # Labeled training sets
│   ├── NSL-KDD/train.csv           # 30,000 training flows (41 features + is_anomaly)
│   └── UNSW-NB15/train.csv         # 50,000 training flows (42 features + is_anomaly)
│
├── models/                         # Core generative model architectures
│   ├── tabddpm/                    # Score-based continuous-discrete diffusion
│   │   ├── diffusion.py            # Gaussian continuous & Multinomial categorical math
│   │   ├── modules.py              # Residual MLP denoiser with sinusoidal time embeddings
│   │   ├── trainer.py              # Training loop with AdamW and cosine scheduling
│   │   └── sampler.py              # Reverse diffusion candidate sampler
│   ├── ensemble.py                 # Master dual-generator TabDDPM + GMM V2 ensemble engine
│   ├── gmm_baseline.py             # BIC-optimized Gaussian Mixture Model engine
│   └── gmm_candidates.py           # Protocol grouping and conditional donor sampling
│
├── utils/                          # Common transformations & evaluation harness
│   ├── dataset_configs.py          # Strict feature semantic schemas
│   ├── domain_rules.py             # Network RFC protocol invariants & precision alignment
│   ├── preprocessing.py            # Data auditing, quantile scaling, and split generation
│   ├── decoding.py                 # TabularDecoder restoring exact dtypes and bounds
│   ├── boundary_filter.py          # Supervised LightGBM boundary filter
│   └── evaluator.py                # Kaggle-equivalent benchmark (ECOD + 10-seed IForest)
│
├── scripts/                        # Pipelines and training utilities
│   └── train_tabddpm.py            # GPU training entry point for TabDDPM models
│
├── submissions/                    # Official submission outputs and records
│   ├── submission_nsl_kdd.csv      # NSL-KDD submission (40,000 rows, 42 columns)
│   ├── submission_unsw_nb15.csv    # UNSW-NB15 submission (70,000 rows, 43 columns)
│   ├── EXPERIMENT_TRACKER.md       # Complete 13-run ablation matrix and history
│   └── README.md                   # Submission documentation and methodology
│
├── results/                        # Serialized model weights & performance artifacts
│   ├── models/                     # Trained TabDDPM weights (.pt) and LightGBM filters (.pkl)
│   └── boundary_filter_performance.png
│
├── notebook/                       # Exploratory research & demonstration notebooks
│   ├── step_5_ensemble.ipynb       # Master end-to-end ensemble walkthrough
│   ├── preprocessing/              # Exploratory data audit and quantile scaling
│   └── models/                     # Exploratory LightGBM rejection boundary experiments
│
└── tests/                          # Automated verification and audit suite
    ├── conftest.py                 # Threading and OpenMP safety fixtures
    ├── test_submission_integrity.py# Exact row count, schema, bounds & SHA-256 locks
    ├── test_adversarial.py         # Adversarial robustness and leakage tests
    ├── test_tabddpm.py             # Unit and integration tests for diffusion engine
    ├── test_gmm_candidates.py      # Protocol grouping and domain invariant tests
    ├── test_decoding.py            # TabularDecoder inversion and bounds tests
    ├── test_evaluator.py           # Evaluator metric correctness tests
    └── test_preprocessing.py       # Data split and transform tests
```

---

## 5. Environment Setup & Dependency Management

The project uses [`uv`](https://github.com/astral-sh/uv) inside Python $\ge 3.10$ for deterministic, lockfile-pinned environment management.

### Installation
```bash
# 1. Clone repository
git clone https://github.com/angeladianas/cs5344-data-augmentation-for-anomaly-detection.git
cd cs5344-data-augmentation-for-anomaly-detection

# 2. Sync virtual environment from lockfile
uv sync
```

To activate the virtual environment:
```bash
source .venv/bin/activate
```

---

## 6. Testing, Quality Assurance & Verification

All modules adhere to strict PEP 8 formatting, complete type annotations, and automated test coverage.

### Run Full Test Suite (66 Tests)
```bash
uv run pytest
```
Includes:
- **`test_submission_integrity.py`**: Verifies exact row counts (40k / 70k), column schema matching, zero nulls, protocol constraints, and bitwise SHA-256 hash invariance.
- **`test_adversarial.py`**: Asserts resilience against pathological inputs, distribution shifts, and zero-leakage constraints.
- **`test_tabddpm.py`** & **`test_gmm_candidates.py`**: Verifies generative mathematical correctness and donor sampling logic.

### Linting & Formatting Check (Zero Warnings)
```bash
uv run ruff check .
uv run ruff format --check .
```

---

## 7. Submission Verification Checklist

- [x] **NSL-KDD Output**: Exactly 40,000 normal synthetic samples.
- [x] **UNSW-NB15 Output**: Exactly 70,000 normal synthetic samples.
- [x] **Header Alignment**: `id` column as first column, exact feature names matching training data, no `is_anomaly`.
- [x] **Zero Nulls**: 100% complete rows without missing values.
- [x] **Domain Physical Constraints**: RFC-compliant TCP sequence numbers, window sizes, and timing identities enforced.
- [x] **Zero Warnings**: Strict Ruff linting and formatting compliance across all files.
- [x] **Deterministic Reproduction**: Seeded execution reproducing exact submitted CSV files.

---

## 8. Authors & Acknowledgments

- **Diana** (Group 39)
- **Marco** (Group 39)
- **Tan Yi Kai** (Group 39)

*Course Instructors:* National University of Singapore, School of Computing — CS5344 Big Data Analytics Technology.
