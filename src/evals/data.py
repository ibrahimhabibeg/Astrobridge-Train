import functools
import os
from pathlib import Path
from typing import Optional
import pandas as pd
from huggingface_hub import hf_hub_download

_REPO_ID = "UniverseTBD/AstroBridge-Data"
_SPECTRA_FILE = "observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet"
_V7_SUBSET_SPECTRA_FILE = "observations/spectra/desi_sdss_subset_crossmatch_nolan_1.0arcsec.parquet"
_EMISSION_LINES_FILE = "observations/spectra/extracted_emission_lines.csv"
_TYPES_FILE = "observations/spectra/extracted_types.csv"


@functools.lru_cache(maxsize=1)
def load_test_spectra() -> pd.DataFrame:
    """Downloads the AstroBridge spectra parquet, filters to test split, deduplicates."""
    print("Loading dataset...")
def _load_legacy_test_spectra() -> pd.DataFrame:
    """Downloads the AstroBridge spectra parquet, filters to test split, deduplicates (400 samples)."""
    print("Loading legacy dataset (desi_sdss_crossmatch_nolan_1.0arcsec.parquet)...")
    parquet_path = hf_hub_download(
        repo_id=_REPO_ID, filename=_SPECTRA_FILE, repo_type="dataset"
    )
    df = pd.read_parquet(parquet_path)

    print("Filtering and deduplicating data...")
    df_test = df[df["split"] == "test"]
    df_test = df_test.drop_duplicates(subset=["wiki_entity_id"])
    print(f"Found {len(df_test)} unique test samples.")
    df_test = df_test.drop_duplicates(subset=["wiki_entity_id"]).reset_index(drop=True)
    print(f"Found {len(df_test)} unique legacy test samples.")
    return df_test


def _find_v7_splits_path() -> Path:
    candidates = [
        Path(__file__).parent / "splits" / "v7_splits.parquet",
        Path(__file__).resolve().parent / "splits" / "v7_splits.parquet",
        Path("src/evals/data/splits/v7_splits.parquet"),
        Path("eval_results/v7_splits.parquet"),
        Path("/root/src/evals/data/splits/v7_splits.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find v7_splits.parquet in any of: {[str(c) for c in candidates]}"
    )


def _load_v7_test_spectra() -> pd.DataFrame:
    """Loads clean v7 validation + test spectra (641 unique samples).

    Uses the v7 stratified split policy and reads raw spectra from
    desi_sdss_subset_crossmatch_nolan_1.0arcsec.parquet, matching
    the exact training pipeline canonical object loader.
    """
    splits_path = _find_v7_splits_path()
    print(f"Loading v7 manifest splits from {splits_path}...")
    v7_manifest = pd.read_parquet(splits_path)

    # Use both validation and test data
    val_test_mask = v7_manifest["split"].isin(["val", "test"]) & v7_manifest["has_spectra"]
    v7_val_test = v7_manifest[val_test_mask]
    target_object_ids = set(v7_val_test["object_id"].astype(str))

    from captioner.data.spectra_dataset import load_spectra_table
    print(f"Loading v7 spectra from {_V7_SUBSET_SPECTRA_FILE}...")
    spectra_df = load_spectra_table(
        hf_path=_REPO_ID,
        files=[_V7_SUBSET_SPECTRA_FILE],
    )

    df_matched = spectra_df[spectra_df["object_id"].astype(str).isin(target_object_ids)].copy()
    df_matched = df_matched.drop_duplicates(subset=["wiki_entity_id"]).reset_index(drop=True)
    print(f"Found {len(df_matched)} unique v7 (val + test) test samples.")
    return df_matched


@functools.lru_cache(maxsize=4)
def load_test_spectra(split_version: Optional[str] = None) -> pd.DataFrame:
    """Downloads or filters test spectra.

    Args:
        split_version: 'legacy' (default, 400 samples from upstream test split)
                       or 'v7' (641 unique val+test spectra from v7 stratified split).
                       If None, falls back to env var ASTROBRIDGE_SPLIT or 'legacy'.
    """
    version = (split_version or os.environ.get("ASTROBRIDGE_SPLIT", "legacy")).lower().strip()
    if version in ("v7", "v7_val_test", "v7-val-test"):
        return _load_v7_test_spectra()
    elif version in ("legacy", "upstream", "v5", "v6"):
        return _load_legacy_test_spectra()
    else:
        raise ValueError(
            f"Unknown split_version '{version}'. Expected 'legacy' or 'v7'."
        )


def load_emission_line_ground_truth() -> pd.DataFrame:
    """Downloads the emission lines ground truth CSV."""
    print("Loading emission lines ground truth...")
    csv_path = hf_hub_download(
        repo_id=_REPO_ID, filename=_EMISSION_LINES_FILE, repo_type="dataset"
    )
    return pd.read_csv(csv_path)


def load_test_spectra_emission_lines() -> pd.DataFrame:
def load_test_spectra_emission_lines(split_version: Optional[str] = None) -> pd.DataFrame:
    """Test spectra filtered to those with emission line annotations."""
    df_spectra = load_test_spectra()
    df_spectra = load_test_spectra(split_version=split_version)
    df_lines = load_emission_line_ground_truth()
    valid_ids = set(df_lines["wiki_entity_id"])
    df_test_lines = df_spectra[df_spectra["wiki_entity_id"].isin(valid_ids)]
    df_test_lines = df_spectra[df_spectra["wiki_entity_id"].isin(valid_ids)].reset_index(drop=True)
    print(f"Found {len(df_test_lines)} test spectra matching emission line ground truth.")
    return df_test_lines


@functools.lru_cache(maxsize=1)
def _load_types_ground_truth() -> pd.DataFrame:
    """Downloads the source/subclass types ground truth CSV."""
    print("Loading source classification ground truth...")
    csv_path = hf_hub_download(
        repo_id=_REPO_ID, filename=_TYPES_FILE, repo_type="dataset"
    )
    return pd.read_csv(csv_path)


def load_test_spectra_by_category(
    target_column: str,
    active_keys: list[str],
    split_version: Optional[str] = None,
) -> pd.DataFrame:
    """Test spectra filtered and merged by a categorical column (class or subclass).

    Args:
        target_column: Column name in the types CSV to filter on ('class' or 'subclass').
        active_keys: List of values in target_column to keep.
        split_version: 'legacy' (default) or 'v7'.
    """
    df_spectra = load_test_spectra()
    df_spectra = load_test_spectra(split_version=split_version)
    df_types = _load_types_ground_truth()

    df_types = df_types[df_types[target_column].isin(active_keys)]

    df_merged = df_spectra.merge(
        df_types[["wiki_entity_id", target_column]],
        on="wiki_entity_id",
        how="inner",
    )
    ).reset_index(drop=True)
    print(f"Found {len(df_merged)} test spectra matching active {target_column} values.")
    return df_merged
