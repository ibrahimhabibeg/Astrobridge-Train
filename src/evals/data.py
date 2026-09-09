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

def load_source_classification_ground_truth() -> pd.DataFrame:
    """
    Downloads the AstroBridge-Data extracted types CSV from HuggingFace.
    """
    print("Loading source classification ground truth...")
    csv_path = hf_hub_download(
        repo_id="UniverseTBD/AstroBridge-Data",
        filename="observations/spectra/extracted_types.csv",
        repo_type="dataset"
    )
    df = pd.read_csv(csv_path)
    return df

def load_test_spectra_source_classification(active_classes_keys: list[str]) -> pd.DataFrame:
    """
    Downloads spectra and extracted types, filters to the 'test' split,
    keeps only samples matching active classes, and merges the class labels.
    """
    df_spectra = load_test_spectra()
    df_types = load_source_classification_ground_truth()
    
    # Filter to active classes
    df_types = df_types[df_types['class'].isin(active_classes_keys)]
    
    # Merge
    df_test_types = df_spectra.merge(df_types[['wiki_entity_id', 'class']], on='wiki_entity_id', how='inner')
    print(f"Found {len(df_test_types)} test spectra matching active source classes.")
    return df_test_types

def load_test_spectra_subclass_classification(active_classes_keys: list[str]) -> pd.DataFrame:
    """
    Downloads spectra and extracted types, filters to the 'test' split,
    keeps only samples matching active subclasses, and merges the subclass labels.
    """
    df_spectra = load_test_spectra()
    df_types = load_source_classification_ground_truth()
    
    # Filter to active subclasses
    df_types = df_types[df_types['subclass'].isin(active_classes_keys)]
    
    # Merge
    df_test_types = df_spectra.merge(df_types[['wiki_entity_id', 'subclass']], on='wiki_entity_id', how='inner')
    print(f"Found {len(df_test_types)} test spectra matching active source subclasses.")
    return df_test_types
