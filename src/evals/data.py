import pandas as pd
from huggingface_hub import hf_hub_download

def load_test_spectra() -> pd.DataFrame:
    """
    Downloads the AstroBridge-Data spectra parquet from HuggingFace,
    filters to the 'test' split, and deduplicates by wiki_entity_id.
    """
    print("Loading dataset...")
    parquet_path = hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/desi_sdss_crossmatch_nolan_1.0arcsec.parquet",
        repo_type="dataset"
    )
    df = pd.read_parquet(parquet_path)
    
    print("Filtering and deduplicating data...")
    df_test = df[df['split'] == 'test']
    df_test = df_test.drop_duplicates(subset=['wiki_entity_id'])
    print(f"Found {len(df_test)} unique test samples.")
    
    return df_test

def load_emission_line_ground_truth() -> pd.DataFrame:
    """
    Downloads the AstroBridge-Data extracted emission lines CSV from HuggingFace.
    """
    print("Loading emission lines ground truth...")
    csv_path = hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/extracted_emission_lines.csv",
        repo_type="dataset"
    )
    df = pd.read_csv(csv_path)
    return df

def load_test_spectra_emission_lines() -> pd.DataFrame:
    """
    Downloads spectra and emission lines datasets, filters to the 'test' split,
    and keeps only samples that have emission line ground truth annotations.
    """
    df_spectra = load_test_spectra()
    df_lines = load_emission_line_ground_truth()
    valid_ids = set(df_lines["wiki_entity_id"])
    df_test_lines = df_spectra[df_spectra["wiki_entity_id"].isin(valid_ids)]
    print(f"Found {len(df_test_lines)} test spectra matching emission line ground truth.")
    return df_test_lines
