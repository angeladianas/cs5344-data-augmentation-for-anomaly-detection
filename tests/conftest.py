"""Pytest configuration and global environment fixtures for CS5344.

Configures environment variables to avoid OpenMP and threading conflicts
between scikit-learn/joblib and PyTorch on macOS ARM64 systems.
"""

import os

# Prevent OpenMP runtime collision between scikit-learn/joblib and PyTorch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
