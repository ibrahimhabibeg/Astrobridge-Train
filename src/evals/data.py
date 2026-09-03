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
