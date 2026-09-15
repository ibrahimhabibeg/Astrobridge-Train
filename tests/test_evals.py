from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evals.data import load_all_benchmark_spectra, load_benchmark_dataset
from src.evals.frontier import MockFrontierJudge, get_frontier_judge
from src.evals.metrics.reporter import compute_caption_metrics
from src.evals.responders import MockCaptionResponder, SpectrumSample, get_responder
from src.evals.tasks import (
    DistanceTask,
    EmissionLineTask,
    SourceTask,
    SubclassTask,
    get_task,
    parse_multi_choice,
    parse_single_choice,
)


def test_spectrum_sample_from_row():
    row = {
        "sample_id": "test_123",
        "survey": "sdss",
        "spectrum": {
            "lambda": [4000.0, 5000.0, 6000.0],
            "flux": [1.0, 2.5, 0.8],
            "mask": [0, 0, 1],
            "ivar": [1.0, 1.0, 0.5],
        },
    }
    sample = SpectrumSample.from_row(row)
    assert sample.sample_id == "test_123"
    assert sample.survey == "sdss"
    assert len(sample.wavelength) == 3
    assert len(sample.flux) == 3
    assert sample.mask[2] == True
    assert sample.ivar is not None


def test_distance_task():
    # Test initialization with custom dataframe and arbitrary labels
    dummy_df = pd.DataFrame([
        {"ground_truth": {"redshift_bin": "distant", "z": 2.5}},
        {"ground_truth": {"redshift_bin": "nearby", "z": 0.03}},
        {"ground_truth": {"redshift_bin": "intermediate", "z": 0.3}},
    ])
    task_custom = DistanceTask(benchmark_df=dummy_df)
    assert task_custom.ordered_labels == ["nearby", "intermediate", "distant"]
    assert task_custom.label_to_letter == {"nearby": "A", "intermediate": "B", "distant": "C"}
    assert task_custom.extract_ground_truth({"ground_truth": {"redshift_bin": "nearby"}}) == "A"
    assert task_custom.extract_ground_truth({"ground_truth": {"redshift_bin": "distant"}}) == "C"

    # Test default initialization from benchmark dataset
    task = DistanceTask()
    assert task.extract_ground_truth({"ground_truth": {"redshift_bin": "<0.1"}}) == "A"
    assert task.extract_ground_truth({"ground_truth": {"redshift_bin": "0.1-0.5"}}) == "B"
    assert task.extract_ground_truth({"ground_truth": {"redshift_bin": ">0.5"}}) == "C"

    prompt = task.build_frontier_prompt("Caption: Redshift around 0.2.")
    assert "Allowed categories:" in prompt
    assert "A: <0.1" in prompt
    assert "B: 0.1-0.5" in prompt
    assert "C: >0.5" in prompt

    assert task.default_parse("FINAL ANSWER: B") == "B"
    assert task.default_parse("Therefore, the answer is A.") == "A"
    assert task.default_parse("Random gibberish without option") is None


def test_source_task():
    task = SourceTask(active_classes={"GALAXY": "Galaxy", "QSO": "Quasar"})
    assert task.extract_ground_truth({"source_type": "GALAXY"}) == "Galaxy"
    assert task.extract_ground_truth({"class": "QSO"}) == "Quasar"

    prompt = task.build_frontier_prompt("Caption: Luminous quasar with broad lines.")
    assert "A: Galaxy" in prompt
    assert "B: Quasar" in prompt
    assert "FINAL ANSWER: [Letter]" in prompt

    assert task.default_parse("FINAL ANSWER: A") == "Galaxy"
    assert task.default_parse("FINAL ANSWER: B") == "Quasar"
    assert task.default_parse("Therefore, the answer is A.") == "Galaxy"
    assert task.default_parse("FINAL ANSWER: Quasar") == "Quasar"
    assert task.default_parse("This is clearly a QSO.") == "Quasar"
    assert task.default_parse("Random text without matches") is None


def test_subclass_task():
    task = SubclassTask(
        active_classes={"AGN": "AGN", "STARBURST": "Starburst", "STARFORMING": "Starforming"}
    )
    assert task.extract_ground_truth({"subclass": "STARBURST"}) == "Starburst"

    prompt = task.build_frontier_prompt("Caption: Intense star-forming regions.")
    assert "A: AGN" in prompt
    assert "B: Starburst" in prompt
    assert "C: Starforming" in prompt
    assert "FINAL ANSWER: [Letter]" in prompt

    assert task.default_parse("FINAL ANSWER: A") == "AGN"
    assert task.default_parse("FINAL ANSWER: B") == "Starburst"
    assert task.default_parse("The category is C") == "Starforming"
    assert task.default_parse("FINAL ANSWER: Starburst") == "Starburst"
    assert task.default_parse("Based on lines, this is an AGN.") == "AGN"
    assert task.default_parse("Unrelated text") is None


def test_emission_line_task():
    task = EmissionLineTask()
    row = {
        "detected_lines": ["HALPHA", "OIII_5007"],
        "absent_lines": ["HBETA"],
        "line_details": {
            "HALPHA": {"snr": 12.5},
            "OIII_5007": {"snr": 8.0},
            "HBETA": {"snr": 1.2},
        },
    }
    gt = task.extract_ground_truth(row)
    assert gt == {"HALPHA": 12.5, "OIII_5007": 8.0}

    prompt = task.build_frontier_prompt(
        "Strong H-alpha and [O III] detected.",
        item={"candidate_query_lines": ["HALPHA", "OIII_5007", "HBETA"]},
    )
    assert "A: HALPHA" in prompt
    assert "B: OIII_5007" in prompt
    assert "C: HBETA" in prompt

    parse_fn = task.get_parse_fn(item={"candidate_query_lines": ["HALPHA", "OIII_5007", "HBETA"]})
    parsed = parse_fn("FINAL ANSWER: A, B")
    assert parsed == ["HALPHA", "OIII_5007"]

    parsed_none = parse_fn("FINAL ANSWER: NONE")
    assert parsed_none == []


def test_mock_responder():
    config = {"responder_type": "mock", "mock_caption": "Mock test spectrum."}
    responder = get_responder(config, device="cpu")
    samples = [
        SpectrumSample(
            sample_id="s1",
            wavelength=np.array([4000, 5000]),
            flux=np.array([1.0, 2.0]),
            mask=np.array([False, False]),
        )
    ]
    captions = responder.generate_captions(samples)
    assert len(captions) == 1
    assert captions[0].sample_id == "s1"
    assert captions[0].caption == "Mock test spectrum."


def test_mock_frontier():
    config = {"frontier_type": "mock", "mock_answer": "B"}
    judge = get_frontier_judge(config)
    resp = judge.predict("Test prompt", parse_fn=lambda x: "B" if "B" in x else None)
    assert resp.parsed == "B"
    assert resp.forced_fallback == False

    fb_judge = MockFrontierJudge({"mock_answer": "A", "simulate_fallback": True})
    resp2 = fb_judge.predict(
        "Test prompt",
        parse_fn=lambda x: "A" if "FINAL ANSWER: A" in x else None,
        fallback_tag="FINAL ANSWER: ",
    )
    assert resp2.parsed == "A"
    assert resp2.forced_fallback == True


def test_compute_caption_metrics_classification(tmp_path: Path):
    preds_file = tmp_path / "predictions.jsonl"
    records = [
        {
            "sample_id": "1",
            "task": "distance",
            "ground_truth": "A",
            "caption": {"text": "Short caption."},
            "frontier_evaluation": {
                "prediction": "A",
                "raw_response": "FINAL ANSWER: A",
                "forced_fallback": False,
                "is_correct": True,
            },
        },
        {
            "sample_id": "2",
            "task": "distance",
            "ground_truth": "B",
            "caption": {"text": "Another caption text."},
            "frontier_evaluation": {
                "prediction": "B",
                "raw_response": "FINAL ANSWER: B",
                "forced_fallback": False,
                "is_correct": True,
            },
        },
        {
            "sample_id": "3",
            "task": "distance",
            "ground_truth": "A",
            "caption": {"text": "Third caption."},
            "frontier_evaluation": {
                "prediction": "B",
                "raw_response": "FINAL ANSWER: B",
                "forced_fallback": False,
                "is_correct": False,
            },
        },
    ]
    with open(preds_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    bundle = compute_caption_metrics(tmp_path)
    assert bundle["total_samples"] == 3
    assert pytest.approx(bundle["task_metrics"]["accuracy"], 0.01) == 2 / 3
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "report.md").exists()


def test_compute_caption_metrics_multilabel(tmp_path: Path):
    preds_file = tmp_path / "predictions.jsonl"
    records = [
        {
            "sample_id": "1",
            "task": "emission_lines",
            "ground_truth": {"HALPHA": 10.0, "HBETA": 5.0},
            "caption": {"text": "Caption 1."},
            "frontier_evaluation": {
                "prediction": ["HALPHA", "HBETA"],
                "raw_response": "EMISSION LINES: HALPHA, HBETA",
                "forced_fallback": False,
                "is_correct": True,
            },
        },
        {
            "sample_id": "2",
            "task": "emission_lines",
            "ground_truth": {"HALPHA": 8.0},
            "caption": {"text": "Caption 2."},
            "frontier_evaluation": {
                "prediction": ["HALPHA", "OIII_5007"],
                "raw_response": "EMISSION LINES: HALPHA, OIII_5007",
                "forced_fallback": False,
                "is_correct": False,
            },
        },
    ]
    with open(preds_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    bundle = compute_caption_metrics(tmp_path)
    assert bundle["total_samples"] == 2
    assert "sample_f1" in bundle["task_metrics"]
    assert "dataset_f1" in bundle["task_metrics"]
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "report.md").exists()


def test_task_registry_resolution():
    assert isinstance(get_task("distance"), DistanceTask)
    assert isinstance(get_task("source"), SourceTask)
    assert isinstance(get_task("subclass"), SubclassTask)
    assert isinstance(get_task("emission_lines"), EmissionLineTask)


def test_choice_parsers():
    # Single choice
    assert parse_single_choice("FINAL ANSWER: A", allowed_letters={"A", "B"}) == "A"
    assert parse_single_choice("FINAL ANSWER: [B]", allowed_letters={"A", "B"}) == "B"
    assert parse_single_choice("FINAL ANSWER: (A)", allowed_letters={"A", "B"}) == "A"
    assert parse_single_choice("FINAL ANSWER: **B**", allowed_letters={"A", "B"}) == "B"
    assert parse_single_choice("FINAL ANSWER: Option A", allowed_letters={"A", "B"}) == "A"
    assert parse_single_choice("Therefore, the answer is B.", allowed_letters={"A", "B"}) == "B"
    assert parse_single_choice("The category is A", allowed_letters={"A", "B"}) == "A"
    assert parse_single_choice("FINAL ANSWER: C", allowed_letters={"A", "B"}) is None
    assert parse_single_choice("Random text") is None

    # Multi choice
    assert parse_multi_choice("FINAL ANSWER: A, B", allowed_letters={"A", "B", "C"}) == ["A", "B"]
    assert parse_multi_choice("FINAL ANSWER: [A, C]", allowed_letters={"A", "B", "C"}) == ["A", "C"]
    assert parse_multi_choice("FINAL ANSWER: **A**, **B**", allowed_letters={"A", "B", "C"}) == ["A", "B"]
    assert parse_multi_choice("FINAL ANSWER: A and B", allowed_letters={"A", "B", "C"}) == ["A", "B"]
    assert parse_multi_choice("FINAL ANSWER: NONE", allowed_letters={"A", "B", "C"}) == []
    assert parse_multi_choice("No lines present. FINAL ANSWER: NONE") == []
    assert parse_multi_choice("Random text without choices", allowed_letters={"A", "B"}) == []
