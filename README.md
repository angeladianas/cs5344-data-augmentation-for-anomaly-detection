# CS5344: Tabular Data Augmentation for Anomaly Detection (Group 39)

## Reproducing Competition Submissions (Grading & Evaluation)

This repository provides an automated, unified pipeline to reproduce the exact Kaggle competition submissions for both **NSL-KDD** ($N = 40,000$) and **UNSW-NB15** ($N = 70,000$).

### 1. One-Command Master Pipeline Execution
To execute the end-to-end pipeline directly from our trained models (**65% Deep TabDDPM + 35% Protocol-Grouped GMM V2 + LightGBM Negative Boundary Filtering + RFC Physics Cleaning**):
```bash
python run_pipeline.py
```
This generates candidates directly from the trained TabDDPM checkpoints (`results/models/{dataset}/tabddpm_best.pt`) and Protocol-Grouped GMM V2 models, filters them via LightGBM boundary critics (`results/models/boundary_filter_{dataset}.pkl`), enforces network physics invariants, and exports:
- `submissions/submission_nsl_kdd.csv` (40,000 rows, 42 cols, 0 nulls, Kaggle Public: **0.93113**)
- `submissions/submission_unsw_nb15.csv` (70,000 rows, 43 cols, 0 nulls, Kaggle Public: **0.60013**)

### 2. Interactive Step-by-Step Walkthrough Notebook
For an interactive, cell-by-cell walkthrough with visualizations, mathematical justifications, ratio sweeps, and boundary filtering curves, open:
- [`notebook/step_5_ensemble.ipynb`](notebook/step_5_ensemble.ipynb)

### 3. Submission History & Experiment Tracker
A complete version-by-version log of candidate sources, filtering parameters, local validation scores ($S_{\text{val}}$), and official Kaggle public scores is maintained in:
- [`submissions/EXPERIMENT_TRACKER.md`](submissions/EXPERIMENT_TRACKER.md)

---

## Development Guide
Setup guidelines to ensure that we have the same environment and code linting.

### 1. Local Setup
---
#### a. Environment and extension
In your VSCode, if you're developing locally,
1. Install `ruff` extension, this helps linting your Python code.
2. Install `uv` (can use brew or any other method you prefer), we'll use this to manage our environments instead of pip (both in VSCode and in Kaggle Notebook).
3. Some basic libraries and dependencies have been added to `pyproject.toml` and locked under `uv.lock`. You can start a new virtual environment by typing `uv venv` in the project root and then activate that virtual environment (typically by doing `source ~/<your_path_to_this_folder>/.venv/bin/activate`). Tips: to automatically select the venv everytime you open this project, open the command palate (Cmd + Shift + p) > Python: Select Interpreter > Enter interpreter path... > put `.venv/bin/activate` as the path > Enter. Open a new terminal window in VSCode, then you should be in your venv already (one easy way to verify is to run `which python3`, it should point to the python3 version in your venv path).
4. Start syncing the dependencies by using `uv sync`.
5. If you want to add new libraries, you can do so by adding `uv add <package>`. This will automatically update the lockfile (`uv.lock` and `pyproject.toml`). If you're updating the `pyproject.toml` in your pull requests, add somebody else as the reviewer of the pull request and inform people to update their environment accordingly. This is to prevent environment conflicts. 

#### b. Further things on Linting
This is more important for those doing the results consolidation and anyone creating the templates for others to use. This is to ensure that we all have a similar code style.
1. Create a `.vscode` folder in the project root. 
2. Create a file `settings.json`. Add this to the file:
```json
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
        }
    },
    "ruff.configuration": "pyproject.toml"
}
```

### 2. Git workflow
---
1. Create a new branch, based off the main branch.
2. Add, commit, and push your changes to that branch.
3. Create a pull request.
4. If you're updating the `pyproject.toml`/any other shared files in your pull requests, add somebody else as the reviewer of the pull request. Else, just merge the commit to the main branch directly by yourself. 

Idea: At the initial stages, we can just push our individual file into `tmp_notebook` folder. Then the consolidation team will help to refactor the notebooks into ready-to-use Python scripts. 
Note: always remember to keep your branch and main branch up-to-date by using git pull!
For branching strategies, up to your preference. You can always create a new branch based off main for every changes you created. Or you can create a long-lived branch. 

```bash
# 1. Go to the main branch
git checkout main

# 2. Pull changes from main
git pull origin main

# 3. Create and checkout to a new branch (replace your_branch_name with your desired branch name)
git checkout -b your_branch_name

# 4. Add changes
git add xyz.py

# 5. Commit changes, add your commit message
git commit -m "feat: your commit message"

# 6. Push the changes
git push origin your_branch_name

# 7. Go to Github and create your pull request
```

### 3. Using this repository in Kaggle notebook
---
1. Link your Kaggle notebook with your Github account (Notebook > File > Link to Github). You also need to create a PAT (Personal Access Token) in Github so that you do not need to repeatedly enter your credentials when connecting to Github from Kaggle. Steps:
- Go to your Github Settings > Developer Settings > Tokens (Classic) > Generate new token (Fine grained, repo-scope) > Set everything up (Permissions needed: Contents (Read only)). 
- Do not push your changes from Kaggle notebook, so to avoid this mistake, we set the Permissions to just read only. 
- Copy and save the token.
- Go to your Kaggle notebook again > Add-ons > Secrets > Add a secret named `GH_TOKEN` and add your github token.
2. Turn on your internet access in the notebook (Settings > Turn on internet).
3. Setup
```python
from kaggle_secrets import UserSecretsClient
import os

# 1. Get the token safely
user_secrets = UserSecretsClient()
token = user_secrets.get_secret("GH_TOKEN")
repo_name = "cs5344-data-augmentation-for-anomaly-detection"

# 2. Construct the URL with the token
repo_url = f"https://angeladianas:{token}@github.com/angeladianas/{repo_name}.git"

# 3. Clone without being asked for a username
if not os.path.exists(repo_name):
    print("Cloning private repository...")
    !git clone {repo_url}
else:
    print("Repo already exists. Pulling latest...")
    %cd {repo_name}
    !git pull {repo_url} origin main

%cd {repo_name}

!uv pip install -r pyproject.toml --system
```

2. If you want to install new libraries, you can use uv add. Then let Diana knows on what new libraries you need and she'll add it to the `pyproject.toml` and `uv.lock`.

```python
# 1. Install the new library and update the lockfile
!uv add <package>

# 2. Verify the lockfile was updated
# NOTE: this does not mean you have updated your lockfile in git. You still need to push your changes as per the git workflow.
!ls -l uv.lock
```

## Shared dataset preprocessing

Both dataset notebooks in `notebook/preprocessing/` use the same pipeline:

```python
from utils.dataset_configs import NSL_KDD, UNSW_NB15
from utils.preprocessing import run_preprocessing

nsl = run_preprocessing("data/NSL-KDD/train.csv", NSL_KDD)
unsw = run_preprocessing("data/UNSW-NB15/train.csv", UNSW_NB15)
```

Dataset configurations declare categorical, binary, discrete, integer, continuous,
and bounded-rate features. The pipeline audits the data, creates duplicate-aware
development/validation splits, fits generator rules on development normals, and
exports GMM/TabDDPM inputs and an audit report. Boundary vocabularies use all
development rows. UNSW service `-` is preserved; its packet `rate` is unbounded,
and `is_ftp_login` preserves the released discrete states rather than forcing binary.

Artifacts are saved separately under `output/nsl_kdd/` and `output/unsw_nb15/`,
including CSV copies. Re-running overwrites the selected dataset's artifacts.
Source-row IDs are provenance and must be excluded from training features.
`TabularDecoder.from_directory(result["model_input_output"])` loads the matching
decoder. Float32 inverse errors are measured with per-row precision bounds;
check the reconstruction audit, especially for large TCP sequence values.
Zero-indicator modelling remains a later experiment. This pipeline does not train
generators or establish downstream detection performance.

Generate the GMM share of the proposal's oversampled candidate pool with:

```bash
uv run python -m models.gmm_candidates_v1  # original ungrouped model
uv run python -m models.gmm_candidates_v2  # protocol-grouped models
```

V1 and V2 outputs are kept separately under each dataset's
`gmm_candidates_v1/` and `gmm_candidates_v2/` directories.

This fits separate best-converged-by-BIC GMMs on development-normal protocol
groups: TCP and non-TCP for NSL-KDD, and TCP, UDP, and other protocols for
UNSW-NB15. Each group generates its share of the 50,000 NSL-KDD or 60,000
UNSW-NB15 candidates, proportionally to its share of development-normal rows;
the group samples are recombined before post-processing. The selected
group-specific models and per-row group/component assignments are saved. It
also applies the saved LightGBM
boundary filter from `results/models/` with the default rejection threshold
`0.10`, retaining the original pool and writing accepted rows to
`filtered_samples.csv` and `filtered_samples.parquet`. Per-candidate anomaly
scores are saved to `boundary_filter_scores.csv`. NSL-KDD `src_bytes`,
`dst_bytes`, `dst_host_count`, and `dst_host_srv_count` are rank-mapped to
their development-normal empirical marginals before inverse transformation,
preserving observed tails and zero frequencies. Every numerical output column
is also rounded to the maximum meaningful decimal precision found in the
original training CSV; the per-column precision is recorded in metadata. Set
`--tau-reject` to change the boundary-filter threshold. These are still only
GMM candidates, not final submissions: the proposal combines them with
TabDDPM candidates and selects the required final row count. For NSL-KDD,
`num_compromised` and `num_root` are resampled jointly from development-normal
rows matching each candidate's protocol, service, flag, and login context;
sampling falls back to same-protocol rows and then all development normals.
For UNSW-NB15, the TCP window and sequence fields (`swin`, `stcpb`, `dtcpb`,
and `dwin`) are set to zero for non-TCP records. Generated `tcprtt` is derived
from `synack + ackdat` and bounded to 3.5 seconds; when the sum exceeds the
bound, both components are scaled proportionally to preserve the timing
identity. `response_body_len` is rank-calibrated separately for HTTP traffic
(the `http` service or a positive HTTP-method count) and non-HTTP traffic to
match development-normal tails. `ct_state_ttl` is reconstructed from
development-normal values to match its zero frequency within each
`(proto, state)` context, with protocol and global fallbacks for unseen
contexts. Generated `smean` and `dmean` are derived from bytes per packet and
rounded to the source's integer representation; byte/packet consistency is
checked against the resulting half-packet rounding tolerance. If a generated
packet count is zero, the corresponding byte count is set to zero and the
derived mean is zero; the number of such byte-count repairs is recorded in
metadata.

Run utility tests with `uv run pytest`.

## Disclaimer
---
AI is used in generating docstring, enhancing the sentences, and other formatting. 
