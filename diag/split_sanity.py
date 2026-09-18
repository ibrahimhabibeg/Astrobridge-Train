#!/usr/bin/env python
"""Does the written manifest's split match what configs/data.yaml actually asked for?

Reads `outputs/manifest/manifest.parquet` (produced by `scripts/00_build_manifest.py` /
`write_manifest`) and checks it against `cfg.splits` — no network, no re-download, cheap enough to
run right before every training launch, not just once. This is the check that would have caught
the tier_overrides wiring being wrong (typo'd tier name, fraction not applied, stratification
silently dropped) before a training run silently used the old global fractions for every tier.

Checks:
  - every object_id appears in exactly one split (the same invariant assign_splits asserts
    in-process, re-verified independently from what actually landed on disk)
  - each tier's actual val/test fraction is within tolerance of what cfg.splits (with
    tier_overrides applied) says it should be
  - within spectra specifically, each `survey` stratum (desi/sdss) independently matches the
    fraction too — a flat pooled 80/20 across both surveys could still land here as a red
    herring if one survey silently dominates the draw

Usage:
    uv run python diag/split_sanity.py
    uv run python diag/split_sanity.py --manifest outputs/manifest/manifest.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from captioner.utils.config import load_config  # noqa: E402

TOLERANCE = 0.03  # fraction points; small strata round to the nearest object, so exact isn't fair


def _expected_fractions(cfg, tier: str) -> tuple[float, float]:
    overrides = dict(cfg.splits.get("tier_overrides", {}))
    if tier in overrides:
        entry = overrides[tier]
        return float(entry.get("val", cfg.splits.val)), float(entry.get("test", cfg.splits.test))
    return float(cfg.splits.val), float(cfg.splits.test)


def _check_group(label: str, sub: pd.DataFrame, expect_val: float, expect_test: float) -> list[str]:
    n = len(sub)
    if n == 0:
        return [f"  SKIP {label}: 0 objects"]
    counts = sub["split"].value_counts()
    got_val = counts.get("val", 0) / n
    got_test = counts.get("test", 0) / n
    problems = []
    ok_val = abs(got_val - expect_val) <= TOLERANCE
    ok_test = abs(got_test - expect_test) <= TOLERANCE
    status = "PASS" if (ok_val and ok_test) else "FAIL"
    print(
        f"  {status} {label:<24s} n={n:5d}  val={got_val:.3f} (want {expect_val:.3f})"
        f"  test={got_test:.3f} (want {expect_test:.3f})"
    )
    if not ok_val:
        problems.append(f"{label}: val fraction {got_val:.3f} outside tolerance of {expect_val:.3f}")
    if not ok_test:
        problems.append(f"{label}: test fraction {got_test:.3f} outside tolerance of {expect_test:.3f}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None, help="defaults to cfg.manifest.parquet")
    args = parser.parse_args()

    cfg = load_config("base", "data")
    manifest_path = Path(args.manifest or cfg.manifest.parquet)
    if not manifest_path.exists():
        print(f"FAIL: {manifest_path} does not exist — run scripts/00_build_manifest.py first.")
        sys.exit(1)

    df = pd.read_parquet(manifest_path)
    for col in ("object_id", "tier", "split"):
        if col not in df.columns:
            print(f"FAIL: manifest is missing column {col!r} — wrong file, or a schema change upstream.")
            sys.exit(1)

    print(f"Manifest: {manifest_path}  ({len(df)} objects)\n")

    problems: list[str] = []

    dup = df.groupby("object_id")["split"].nunique()
    n_dup = int((dup > 1).sum())
    print(f"{'PASS' if n_dup == 0 else 'FAIL'} object_id in more than one split: {n_dup}")
    if n_dup:
        problems.append(f"{n_dup} object_id(s) appear in more than one split — a leak.")

    print("\nPer-tier val/test fractions vs configs/data.yaml (splits.val/test + tier_overrides):")
    for tier in sorted(df["tier"].dropna().unique()):
        expect_val, expect_test = _expected_fractions(cfg, tier)
        problems += _check_group(tier, df[df["tier"] == tier], expect_val, expect_test)

    if "survey" in df.columns and "spectra" in set(df["tier"]):
        print("\nspectra, by survey (stratification actually held, not just the pooled total):")
        spectra = df[df["tier"] == "spectra"]
        expect_val, expect_test = _expected_fractions(cfg, "spectra")
        for survey in sorted(spectra["survey"].dropna().unique()):
            problems += _check_group(
                f"spectra/{survey}", spectra[spectra["survey"] == survey], expect_val, expect_test
            )

    print()
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("PASS — splits match configs/data.yaml. Safe to launch training against this manifest.")


if __name__ == "__main__":
    main()
