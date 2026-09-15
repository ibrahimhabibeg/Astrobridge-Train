from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional
import dotenv
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.data import load_all_benchmark_spectra, load_benchmark_dataset
from evals.responders import SpectrumSample, get_responder


def run_caption_generation(
    responder_config: dict[str, Any],
    output_path: str | Path,
    benchmark: str = "all",
    limit: Optional[int] = None,
    batch_size: int = 32,
    caption_prompt: Optional[str] = None,
    device: Optional[str] = None,
    df: Optional[Any] = None,
) -> int:
    import torch

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    responder = get_responder(responder_config, device)

    if df is not None:
        df_test = df
    elif benchmark == "all":
        df_test = load_all_benchmark_spectra()
    else:
        df_test = load_benchmark_dataset(benchmark)

    if limit is not None:
        df_test = df_test.head(limit)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = df_test.to_dict(orient="records")
    total_generated = 0

    with open(output_path, "w") as f:
        for i in tqdm(range(0, len(rows), batch_size), desc="Generating Captions"):
            batch_rows = rows[i : i + batch_size]
            samples = [SpectrumSample.from_row(r) for r in batch_rows]
            captions = responder.generate_captions(samples, prompt_override=caption_prompt)
            for c in captions:
                record = {
                    "sample_id": c.sample_id,
                    "survey": c.survey,
                    "caption": c.caption,
                    "responder_type": c.responder_type,
                    "model_id": c.model_id,
                    "caption_prompt": c.caption_prompt,
                }
                f.write(json.dumps(record) + "\n")
                total_generated += 1
            f.flush()

    return total_generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and cache captions for benchmark spectra.")
    parser.add_argument("--responder", type=str, required=True, help="Path to responder YAML config.")
    parser.add_argument("--output", type=str, required=True, help="Output path for captions.jsonl.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all", "distance", "source", "subclass", "emission_lines"],
        help="Benchmark dataset to generate captions for (default: all).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of spectra.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--caption-prompt", type=str, default=None, help="Optional caption prompt override.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu).")
    args = parser.parse_args()

    dotenv.load_dotenv()

    with open(args.responder, "r") as f:
        responder_config = yaml.safe_load(f)

    if args.caption_prompt:
        responder_config["caption_prompt"] = args.caption_prompt

    total = run_caption_generation(
        responder_config=responder_config,
        output_path=args.output,
        benchmark=args.benchmark,
        limit=args.limit,
        batch_size=args.batch_size,
        caption_prompt=args.caption_prompt,
        device=args.device,
    )
    print(f"Generated {total} captions to {args.output}")


if __name__ == "__main__":
    main()
