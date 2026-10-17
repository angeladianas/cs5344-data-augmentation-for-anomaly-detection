"""Run the protocol-grouped GMM candidate generator (V2)."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.gmm_candidates import main

if __name__ == "__main__":
    main(generation_version="v2")
