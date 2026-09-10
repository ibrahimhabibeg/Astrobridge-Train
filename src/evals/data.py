import functools
import pandas as pd
from huggingface_hub import hf_hub_download

_REPO_ID = "UniverseTBD/AstroBridge-Data"
_SPECTRA_FILE = "observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet"
_EMISSION_LINES_FILE = "observations/spectra/extracted_emission_lines.csv"
_TYPES_FILE = "observations/spectra/extracted_types.csv"


@functools.lru_cache(maxsize=1)
def load_test_spectra() -> pd.DataFrame:
    """Downloads the AstroBridge spectra parquet, filters to test split, deduplicates."""
    print("Loading dataset...")
    parquet_path = hf_hub_download(
        repo_id=_REPO_ID, filename=_SPECTRA_FILE, repo_type="dataset"
    )
    df = pd.read_parquet(parquet_path)

    print("Filtering and deduplicating data...")
    df_test = df[df["split"] == "test"]
    df_test = df_test.drop_duplicates(subset=["wiki_entity_id"])
    print(f"Found {len(df_test)} unique test samples.")
    return df_test


def load_emission_line_ground_truth() -> pd.DataFrame:
    """Downloads the emission lines ground truth CSV."""
    print("Loading emission lines ground truth...")
    csv_path = hf_hub_download(
        repo_id=_REPO_ID, filename=_EMISSION_LINES_FILE, repo_type="dataset"
    )
    return pd.read_csv(csv_path)


def load_test_spectra_emission_lines() -> pd.DataFrame:
    """Test spectra filtered to those with emission line annotations."""
    df_spectra = load_test_spectra()
    df_lines = load_emission_line_ground_truth()
    valid_ids = set(df_lines["wiki_entity_id"])
    df_test_lines = df_spectra[df_spectra["wiki_entity_id"].isin(valid_ids)]
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
) -> pd.DataFrame:
    """Test spectra filtered and merged by a categorical column (class or subclass).

    Args:
        target_column: Column name in the types CSV to filter on ('class' or 'subclass').
        active_keys: List of values in target_column to keep.
    """
    df_spectra = load_test_spectra()
    df_types = _load_types_ground_truth()

    df_types = df_types[df_types[target_column].isin(active_keys)]

    df_merged = df_spectra.merge(
        df_types[["wiki_entity_id", target_column]],
        on="wiki_entity_id",
        how="inner",
    )
    print(f"Found {len(df_merged)} test spectra matching active {target_column} values.")
    return df_merged
