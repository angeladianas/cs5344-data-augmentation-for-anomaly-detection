"""CS5344 Master Reproducibility Pipeline Entry Point.

This script reproduces the competition submissions for both NSL-KDD (40,000 rows)
and UNSW-NB15 (70,000 rows) by generating candidates directly from the two best
generative models (65% Deep TabDDPM + 35% Protocol-Grouped GMM V2) and filtering
with the LightGBM Boundary Critic.

Usage:
    python run_pipeline.py
    python run_pipeline.py --dataset unsw_nb15
    python run_pipeline.py --dataset nsl_kdd
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ensemble import export_competition_submission


def main() -> None:
    r"""Parse command-line arguments and reproduce competition submissions."""
    parser = argparse.ArgumentParser(
        description="Reproduce CS5344 final competition submissions from trained TabDDPM + Grouped GMM V2 models."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "nsl_kdd", "unsw_nb15"],
        default="all",
        help="Dataset to generate (default: all)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="TabDDPM mixture weight (default: 0.65 for 65%% TabDDPM : 35%% Grouped GMM V2)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Optional override for LightGBM boundary rejection threshold (default: 0.04 for NSL, 0.05 for UNSW)",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("CS5344: Tabular Data Augmentation for Anomaly Detection")
    print(
        "Master Reproducibility Pipeline (Direct Model Generation: 65% TabDDPM + 35% Grouped GMM V2)"
    )
    print("=" * 75)

    if args.dataset in ["all", "nsl_kdd"]:
        export_competition_submission(
            dataset_name="nsl_kdd",
            target_n=40000,
            alpha=args.alpha,
            tau_reject=args.tau if args.tau is not None else 0.04,
        )

    if args.dataset in ["all", "unsw_nb15"]:
        export_competition_submission(
            dataset_name="unsw_nb15",
            target_n=70000,
            alpha=args.alpha,
            tau_reject=args.tau if args.tau is not None else 0.05,
        )

    print("\n[SUCCESS] Master pipeline execution completed successfully.")
    print("Output submissions:")
    print("  - submissions/submission_nsl_kdd.csv (40,000 rows)")
    print("  - submissions/submission_unsw_nb15.csv (70,000 rows)")


if __name__ == "__main__":
    main()
