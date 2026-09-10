import json
import os
from typing import Any, Dict

import pandas as pd

from .classification import classification_report
from .multilabel import multilabel_report
from ..data import load_test_spectra


def _load_metadata(results_dir: str) -> dict:
    """Load metadata.json from a results directory, or return {}."""
    path = os.path.join(results_dir, "metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------

def _prepare_classification_df(results_dir: str) -> pd.DataFrame:
    """Load results.jsonl and standardize prediction column."""
    results_path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(results_path):
        return pd.DataFrame()

    df = pd.read_json(results_path, lines=True)
    if df.empty:
        return df

    # Standardize prediction column
    if "model_answer" in df.columns:
        df["pred"] = df["model_answer"].fillna("UNKNOWN")
    elif "parsed" in df.columns:
        df["pred"] = df["parsed"].fillna("UNKNOWN")
    else:
        df["pred"] = "UNKNOWN"

    # Merge survey info for per-survey breakdowns
    test_spectra = load_test_spectra()
    df = df.merge(
        test_spectra[["wiki_entity_id", "survey"]],
        on="wiki_entity_id",
        how="left",
    )

    return df


def _compute_and_print_classification(
    df: pd.DataFrame,
    labels: list,
    *,
    ordinal: bool = False,
    results_dir: str,
) -> None:
    """Compute classification metrics, print summary, and save JSON."""
    if df.empty:
        return

    y_true = df["correct_answer"].tolist()
    y_pred = df["pred"].tolist()

    overall = classification_report(y_true, y_pred, labels, ordinal=ordinal)
    if overall["total_samples"] == overall["format_errors"]:
        print("All predictions were format errors — no metrics to compute.")
        return

    def print_metrics(name: str, m: dict):
        print(f"\n=== Classification Metrics ({name}) ===")
        print(f"Total Samples: {m['total_samples']} | Format Errors: {m['format_errors']}")
        acc_str = f"Accuracy: {m['global_accuracy']*100:.1f}%"
        mae_str = f" | MAE: {m['mean_absolute_error']:.3f} classes" if "mean_absolute_error" in m else ""
        print(f"{acc_str}{mae_str} | Macro F1: {m['macro_f1']:.3f}")

    print_metrics("Overall", overall)

    metrics: Dict[str, Any] = {
        "metadata": _load_metadata(results_dir),
        **overall,
    }

    # Per-survey breakdown
    if "survey" in df.columns:
        metrics["by_survey"] = {}
        for survey in sorted(df["survey"].dropna().unique()):
            survey_df = df[df["survey"] == survey]
            yt_s = survey_df["correct_answer"].tolist()
            yp_s = survey_df["pred"].tolist()
            survey_metrics = classification_report(yt_s, yp_s, labels, ordinal=ordinal)
            if survey_metrics["total_samples"] > survey_metrics["format_errors"]:
                metrics["by_survey"][survey] = survey_metrics
                print_metrics(survey.upper(), survey_metrics)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


# ---------------------------------------------------------------------------
# Emission line pipeline
# ---------------------------------------------------------------------------

def _prepare_emission_df(results_dir: str) -> pd.DataFrame:
    """Load results.jsonl and standardize ground truth / prediction columns."""
    results_path = os.path.join(results_dir, "results.jsonl")
    if not os.path.exists(results_path):
        return pd.DataFrame()

    df = pd.read_json(results_path, lines=True)
    if df.empty:
        return df

    # Ground truth: dict of {line_name: snr}
    if "ground_truth_lines" in df.columns:
        df["gt"] = df["ground_truth_lines"]
    else:
        df["gt"] = [{}] * len(df)
    df["gt"] = df["gt"].apply(lambda x: x if isinstance(x, dict) else {})

    # Predictions: list of line names, or None if parser failed
    if "predicted_lines" in df.columns:
        df["pred_raw"] = df["predicted_lines"]
    elif "parsed" in df.columns:
        df["pred_raw"] = df["parsed"]
    else:
        df["pred_raw"] = None

    return df


def _compute_and_print_emission(
    df: pd.DataFrame,
    canonical_lines: list,
    results_dir: str,
) -> None:
    """Compute emission line metrics, print summary, and save JSON."""
    if df.empty:
        return

    # Format errors: samples where the parser returned None (not a list)
    format_error_mask = df["pred_raw"].apply(lambda x: not isinstance(x, list))
    n_format_errors = int(format_error_mask.sum())

    # Filter to successfully parsed samples
    valid_df = df[~format_error_mask].copy()
    gt_dicts = valid_df["gt"].tolist()
    pred_sets = [set(p) for p in valid_df["pred_raw"].tolist()]

    report = multilabel_report(
        gt_dicts, pred_sets, canonical_lines, n_format_errors=n_format_errors
    )

    metrics = {
        "metadata": _load_metadata(results_dir),
        **report,
    }

    # Print summary
    sample = report["sample_level"]
    micro = report["dataset_micro_level"]
    macro = report["dataset_macro_level"]
    snr = report["snr_weighted"]

    print(f"\n=== Emission Line Metrics ===")
    print(
        f"Samples: {report['total_samples']} | "
        f"Exact Match: {report['exact_match_rate']*100:.1f}% | "
        f"Format Errors: {report['format_errors']}"
    )
    print(
        f"Sample Means -> Prec: {sample['mean_precision']*100:.1f}% | "
        f"Rec: {sample['mean_recall']*100:.1f}% | "
        f"F1: {sample['mean_f1']:.3f} | "
        f"SNR-F1: {snr['mean_snr_weighted_f1']:.3f}"
    )
    print(
        f"Micro Totals -> Prec: {micro['micro_precision']*100:.1f}% | "
        f"Rec: {micro['micro_recall']*100:.1f}% | "
        f"F1: {micro['micro_f1']:.3f} | "
        f"SNR-F1: {micro['micro_snr_weighted_f1']:.3f}"
    )
    print(f"Macro Means  -> Line F1: {macro['mean_line_f1']:.3f}")

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_and_save_metrics(results_dir: str, task: Any) -> None:
    """Load results, compute all metrics for the task, and save metrics.json.

    Dispatches to the appropriate metric pipeline based on ``task.name``.

    Args:
        results_dir: Path to the evaluation results directory containing
            ``results.jsonl`` and optionally ``metadata.json``.
        task: A task object with a ``.name`` attribute. Must also have
            task-specific attributes (e.g., ``.scheme`` for distance
            classification, ``.categories`` for source/subclass,
            ``.canonical_lines`` for emission lines).
    """
    if task.name == "emission_lines":
        df = _prepare_emission_df(results_dir)
        _compute_and_print_emission(df, task.canonical_lines, results_dir)

    elif task.name == "distance_classification":
        df = _prepare_classification_df(results_dir)
        _compute_and_print_classification(
            df, task.scheme.labels, ordinal=True, results_dir=results_dir
        )

    elif task.name in ("source_classification", "subclass_classification"):
        df = _prepare_classification_df(results_dir)
        _compute_and_print_classification(
            df, task.categories, ordinal=False, results_dir=results_dir
        )

    else:
        raise ValueError(f"Unknown task name for metrics: {task.name}")

