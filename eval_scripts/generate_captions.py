#!/usr/bin/env python
"""Generate and cache captions for astronomical spectra across evaluation benchmarks.

Usage:
    uv run eval_scripts/generate_captions.py \
        --responder eval_configs/caption_responders/astrobridge.yaml \
        --output eval_results/cached_captions/astrobridge_captions.jsonl \
        --limit 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional
import dotenv
import numpy as np
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.caption_responders import CaptionSample, get_caption_responder
from evals.data import (
    load_all_benchmark_spectra,
    load_benchmark_dataset,
)


def run_caption_generation(
    responder_config: dict,
    output_path: str,
    benchmark: str = "all",
    limit: Optional[int] = None,
    batch_size: int = 32,
    caption_prompt: Optional[str] = None,
    device: Optional[str] = None,
    df: Optional[Any] = None,
) -> int:
    """Generate and save captions for benchmark spectra to output_path. Returns total count generated."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initializing caption responder '{responder_config.get('responder_type')}' on {device}...")
    responder = get_caption_responder(responder_config, device)

    # Determine benchmark or use provided dataframe
    if df is not None:
        df_test = df
    else:
        bench = benchmark or "all"
        print(f"Loading benchmark data for target '{bench}'...")
        if bench == "all":
            df_test = load_all_benchmark_spectra()
        else:
            df_test = load_benchmark_dataset(bench)

    if limit is not None:
        print(f"Limiting to first {limit} samples.")
        df_test = df_test.head(limit)

    total_expected = len(df_test)
    print(f"Starting caption generation for {total_expected} spectra...")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    total_generated = 0
    with open(output_path, "w") as out_f:
        for batch_df in tqdm(list(chunker(df_test, batch_size)), desc="Generating Captions"):
            samples = []
            surveys = batch_df["survey"].tolist() if "survey" in batch_df.columns else ["sdss"] * len(batch_df)

            for i, (_, row) in enumerate(batch_df.iterrows()):
                spec_data = row["spectrum"]
                flux = np.array(spec_data["flux"])
                wavelength = np.array(spec_data["lambda"])
                mask = (
                    np.array(spec_data["mask"]).astype(bool)
                    if "mask" in spec_data
                    else np.zeros_like(flux, dtype=bool)
                )
                ivar = np.array(spec_data["ivar"]) if "ivar" in spec_data else None
                samples.append(
                    CaptionSample(
                        sample_id=str(row["sample_id"]),
                        wavelength=wavelength,
                        flux=flux,
                        mask=mask,
                        survey=surveys[i],
                        ivar=ivar,
                    )
                )

            captions = responder.generate_captions(samples, prompt_override=caption_prompt)
            for c in captions:
                rec = {
                    "sample_id": c.sample_id,
                    "object_id": c.sample_id,
                    "survey": c.survey,
                    "caption": c.caption,
                    "responder_type": c.responder_type,
                    "model_id": c.model_id,
                    "caption_prompt": c.caption_prompt,
                }
                out_f.write(json.dumps(rec) + "\n")
                total_generated += 1

            out_f.flush()

    print(f"Successfully generated and saved {total_generated}/{total_expected} captions to {output_path}")
    return total_generated


def main():
    parser = argparse.ArgumentParser(description="Pre-generate and cache captions for benchmark spectra.")
    parser.add_argument("--responder", type=str, required=True, help="Path to caption responder YAML config.")
    parser.add_argument("--output", type=str, required=True, help="Output path for captions.jsonl.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all", "redshift", "distance", "source_class", "source", "subclass", "emission_lines"],
        help="Benchmark to generate captions for (default: 'all' generates for all 717 unique spectra).",
    )
    parser.add_argument("--dataset-filter", type=str, default=None, help="Alias for --benchmark.")
    parser.add_argument("--split", type=str, default=None, help="Deprecated (datasets are pre-stratified benchmarks).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of spectra.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for caption generation.")
    parser.add_argument("--caption-prompt", type=str, default=None, help="Override captioning prompt.")
    args = parser.parse_args()

    dotenv.load_dotenv()

    with open(args.responder, "r") as f:
        responder_config = yaml.safe_load(f)

    if args.caption_prompt:
        responder_config["caption_prompt"] = args.caption_prompt

    bench = args.dataset_filter or args.benchmark
    run_caption_generation(
        responder_config=responder_config,
        output_path=args.output,
        benchmark=bench,
        limit=args.limit,
        batch_size=args.batch_size,
        caption_prompt=args.caption_prompt,
    )


if __name__ == "__main__":
    main()
