"""Shared runtime initialization for reproducible numeric experiments."""

from __future__ import annotations

import os
import sys


NUMERIC_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_numeric_thread_limits() -> None:
    """Force the process-level numeric thread policy before fingerprinting."""

    for name in NUMERIC_THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            torch_module.set_num_threads(1)
            torch_module.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting the interop pool only before parallel
            # work starts. Repeated calls remain safe because the first call
            # already installed the required single-thread policy.
            pass


__all__ = [
    "NUMERIC_THREAD_ENVIRONMENT_VARIABLES",
    "apply_numeric_thread_limits",
]
