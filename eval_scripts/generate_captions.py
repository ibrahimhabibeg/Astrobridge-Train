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
import dotenv
import numpy as np
import yaml
from tqdm import tqdm

from evals.caption_responders import CaptionSample, get_caption_responder
from evals.data import (
    load_all_benchmark_spectra,
    load_benchmark_dataset,
)


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

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing caption responder '{responder_config.get('responder_type')}' on {device}...")
    responder = get_caption_responder(responder_config, device)

    # Determine benchmark
    bench = args.dataset_filter or args.benchmark
    print(f"Loading benchmark data for target '{bench}'...")
    if bench == "all":
        df_test = load_all_benchmark_spectra()
    else:
        df_test = load_benchmark_dataset(bench)

    if args.limit is not None:
        print(f"Limiting to first {args.limit} samples.")
        df_test = df_test.head(args.limit)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    total_generated = 0
    with open(args.output, "w") as out_f:
        for batch_df in tqdm(list(chunker(df_test, args.batch_size)), desc="Generating Captions"):
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

            captions = responder.generate_captions(samples)
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

    print(f"Successfully generated and saved {total_generated} captions to {args.output}")


if __name__ == "__main__":
    main()
