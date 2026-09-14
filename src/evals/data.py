from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from huggingface_hub import hf_hub_download

_REPO_ID = "UniverseTBD/AstroBridge-Data"
_HF_BENCHMARK_DIR = "captions/spectra/benchmarks"

BENCHMARK_FILES: Dict[str, str] = {
    "redshift": "redshift.parquet",
    "distance": "redshift.parquet",
    "caption_distance": "redshift.parquet",
    "source_class": "source_class.parquet",
    "source": "source_class.parquet",
    "caption_source": "source_class.parquet",
    "subclass": "subclass.parquet",
    "caption_subclass": "subclass.parquet",
    "emission_lines": "emission_lines.parquet",
    "emission": "emission_lines.parquet",
    "caption_emission_lines": "emission_lines.parquet",
}


def _find_local_benchmark_file(filename: str) -> Optional[Path]:
    """Search common local paths for pre-downloaded benchmark files."""
    candidates = [
        Path("eval_results/benchmarks") / filename,
        Path("benchmarks") / filename,
        Path(__file__).resolve().parent.parent.parent / "eval_results" / "benchmarks" / filename,
        Path(__file__).resolve().parent.parent.parent / "benchmarks" / filename,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


@functools.lru_cache(maxsize=8)
def load_benchmark_dataset(benchmark_name: str) -> pd.DataFrame:
    """Load a benchmark dataset with spectra and ground truth.

    Checks local benchmark directory first, falling back to Hugging Face
    dataset download if not found locally.

    Args:
        benchmark_name: Benchmark identifier (e.g. 'redshift', 'source_class',
                        'subclass', 'emission_lines' or their task equivalents).

    Returns:
        pd.DataFrame containing spectra, object_id, sample_id, survey, and ground_truth.
    """
    key = benchmark_name.strip().lower()
    if key not in BENCHMARK_FILES:
        valid_keys = sorted(set(BENCHMARK_FILES.keys()))
        raise ValueError(f"Unknown benchmark '{benchmark_name}'. Valid benchmarks: {valid_keys}")

    filename = BENCHMARK_FILES[key]
    local_path = _find_local_benchmark_file(filename)

    if local_path is not None:
        print(f"Loading benchmark '{key}' from local file: {local_path}")
        df = pd.read_parquet(local_path)
    else:
        hf_subpath = f"{_HF_BENCHMARK_DIR}/{filename}"
        print(f"Downloading benchmark '{key}' from HF {_REPO_ID}:{hf_subpath}...")
        downloaded_path = hf_hub_download(
            repo_id=_REPO_ID,
            filename=hf_subpath,
            repo_type="dataset",
        )
        df = pd.read_parquet(downloaded_path)

    # Standardize identifier columns
    if "object_id" in df.columns:
        df["sample_id"] = df["object_id"].astype(str)
        if "wiki_entity_id" not in df.columns:
            df["wiki_entity_id"] = df["sample_id"]
    elif "wiki_entity_id" in df.columns:
        df["sample_id"] = df["wiki_entity_id"].astype(str)
        if "object_id" not in df.columns:
            df["object_id"] = df["sample_id"]

    print(f"Loaded benchmark '{key}': {len(df)} samples.")
    return df


@functools.lru_cache(maxsize=1)
def load_all_benchmark_spectra() -> pd.DataFrame:
    """Load and deduplicate spectra across all 4 benchmark datasets.

    Provides the 717 unique spectra evaluated across the benchmark suite,
    ideal for pre-generating cached captions.

    Returns:
        pd.DataFrame containing unique spectra deduplicated by object_id.
    """
    benchmarks = ["redshift", "source_class", "subclass", "emission_lines"]
    dfs: List[pd.DataFrame] = []

    for name in benchmarks:
        df = load_benchmark_dataset(name)
        # Keep essential spectrum columns
        common_cols = [c for c in ["sample_id", "object_id", "wiki_entity_id", "survey", "spectrum"] if c in df.columns]
        dfs.append(df[common_cols].copy())

    combined = pd.concat(dfs, ignore_index=True)
    dedup = combined.drop_duplicates(subset=["object_id"]).reset_index(drop=True)
    print(f"Loaded all benchmark spectra: {len(dedup)} unique spectra across {benchmarks}.")
    return dedup


def load_test_spectra(task_or_benchmark: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Load test spectra for evaluation or caption generation.

    If task_or_benchmark is specified, loads that dedicated benchmark dataset.
    Otherwise loads all unique benchmark spectra (717 samples).
    """
    if task_or_benchmark and task_or_benchmark.lower() in BENCHMARK_FILES:
        return load_benchmark_dataset(task_or_benchmark)
    return load_all_benchmark_spectra()


def load_emission_line_ground_truth() -> pd.DataFrame:
    """Load emission line benchmark ground truth."""
    return load_benchmark_dataset("emission_lines")


def load_test_spectra_emission_lines(**kwargs) -> pd.DataFrame:
    """Load emission line benchmark dataset (200 samples with query lines)."""
    return load_benchmark_dataset("emission_lines")


def load_test_spectra_by_category(
    target_column: str,
    active_keys: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load source class or subclass benchmark dataset."""
    col = target_column.lower().strip()
    if "subclass" in col:
        return load_benchmark_dataset("subclass")
    return load_benchmark_dataset("source_class")
