from __future__ import annotations

"""Persistent cache locations for repository command-line entrypoints."""

import os
from pathlib import Path
from typing import MutableMapping


def configure_runtime_caches(
    repo_root: str | Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Route large runtime caches beside the repository unless overridden."""
    env = os.environ if environ is None else environ
    root = Path(
        env.get("FAITHFUL_INTERVENTIONS_CACHE_ROOT", Path(repo_root) / ".cache")
    ).expanduser().resolve()
    # Keep shared-node CPU usage bounded unless the caller explicitly opts out.
    # These environment variables must be set before importing torch/numpy so
    # BLAS/OpenMP libraries do not fan out across every host core.
    cpu_threads = str(env.get("FAITHFUL_INTERVENTIONS_CPU_THREADS", "2"))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = cpu_threads

    defaults = {
        "XDG_CACHE_HOME": root,
        "VLLM_CACHE_ROOT": root / "vllm",
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
        "TRITON_CACHE_DIR": root / "triton",
        "HF_HOME": root / "huggingface",
        "HF_DATASETS_CACHE": root / "huggingface" / "datasets",
        "PIP_CACHE_DIR": root / "pip",
        "CUDA_CACHE_PATH": root / "cuda",
    }
    for name, path in defaults.items():
        env.setdefault(name, str(path))
    for path in defaults.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return root
