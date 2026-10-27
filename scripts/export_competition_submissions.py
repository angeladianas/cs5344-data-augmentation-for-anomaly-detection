r"""Export competition submissions using the Unified 65:35 Dual-Generator Ensemble with LightGBM Filtering."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ensemble import (
    export_competition_submission,
    fit_and_sample_gmm,
    load_tabddpm_sampler,
    sample_tabddpm_candidates,
)
from utils.domain_rules import enforce_domain_invariants

__all__ = [
    "enforce_domain_invariants",
    "export_competition_submission",
    "fit_and_sample_gmm",
    "load_tabddpm_sampler",
    "sample_tabddpm_candidates",
]

if __name__ == "__main__":
    print(
        "Exporting competition submissions directly from trained models (65:35 TabDDPM + Grouped GMM V2)..."
    )
    export_competition_submission("nsl_kdd", target_n=40000, alpha=0.65, tau_reject=0.04)
    export_competition_submission("unsw_nb15", target_n=70000, alpha=0.65, tau_reject=0.05)
    print("All submissions exported and verified successfully!")
