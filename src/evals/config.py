from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Default Hugging Face repositories & paths
DEFAULT_HF_DATA_REPO: str = "UniverseTBD/AstroBridge-Data"
DEFAULT_HF_EVALS_SUBDIR: str = "evals/spectra"
DEFAULT_MODEL_REPO: str = "UniverseTBD/astrobridge-model-v7"
DEFAULT_BASE_LLM_REPO: str = "Qwen/Qwen3.5-9B"


def get_repo_root() -> Path:
    """Find the root directory of the Astrobridge repository."""
    return Path(__file__).resolve().parent.parent.parent


def get_hf_data_repo() -> str:
    """Return the Hugging Face dataset repository identifier."""
    return os.environ.get("ASTROBRIDGE_HF_DATA_REPO", DEFAULT_HF_DATA_REPO)


def get_hf_evals_subdir() -> str:
    """Return the Hugging Face subdirectory containing benchmark parquets."""
    return os.environ.get("ASTROBRIDGE_HF_EVALS_SUBDIR", DEFAULT_HF_EVALS_SUBDIR)


def get_hf_benchmark_subpath(filename: str, subdir: Optional[str] = None) -> str:
    """Return the relative repository path for a given benchmark file on HF."""
    base_subdir = subdir or get_hf_evals_subdir()
    return f"{base_subdir.rstrip('/')}/{filename}"


def get_default_cache_dir() -> Path:
    env_dir = os.environ.get("ASTROBRIDGE_DATA_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return get_repo_root() / "data" / "benchmarks"

