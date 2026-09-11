import json
import os
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

from evals.caption_responders import (
    CaptionSample,
    get_caption_responder,
)
from evals.caption_tasks import (
    get_caption_task,
    CaptionDistanceTask,
    CaptionEmissionLineTask,
    CategoricalCaptionTask,
)
from evals.frontier import get_frontier_model
from evals.caption_eval import compute_caption_metrics
from eval_scripts.run_caption_eval_local import run_caption_evaluation


class TestCaptionEval(unittest.TestCase):

    def test_mock_caption_responder(self):
        config = {"responder_type": "mock", "caption_prompt": "Custom prompt testing"}
        responder = get_caption_responder(config, "cpu")

        samples = [
            CaptionSample(
                sample_id="test_1",
                wavelength=[4000, 5000, 6000],
                flux=[1.0, 2.0, 1.5],
                mask=[False, False, False],
                survey="sdss",
            )
        ]
        captions = responder.generate_captions(samples)
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0].sample_id, "test_1")
        self.assertIn("Synthetic astronomical spectrum", captions[0].caption)
        self.assertEqual(captions[0].caption_prompt, "Custom prompt testing")
        self.assertEqual(captions[0].responder_type, "mock")

    def test_mock_caption_responder_prompt_override(self):
        config = {"responder_type": "mock", "caption_prompt": "Base prompt"}
        responder = get_caption_responder(config, "cpu")

        samples = [
            CaptionSample(
                sample_id="test_1",
                wavelength=[4000, 5000],
                flux=[1.0, 2.0],
                mask=[False, False],
                survey="desi",
            )
        ]
        captions = responder.generate_captions(samples, prompt_override="Overridden prompt")
        self.assertEqual(captions[0].caption_prompt, "Overridden prompt")

    def test_frontier_model_mock(self):
        config = {"frontier_type": "mock", "mock_answer": "C"}
        model = get_frontier_model(config)

        # Distance task test
        resp = model.predict("Classify distance... FINAL ANSWER: [Letter]", parse_fn=lambda x: "C")
        self.assertEqual(resp.parsed, "C")
        self.assertIn("FINAL ANSWER: C", resp.raw_text)

        # Emission line task test
        def parse_lines(text):
            return ["Hα", "Hβ"] if "Hα" in text else []

        resp_lines = model.predict("EMISSION LINES: candidate list...", parse_fn=parse_lines)
        self.assertIn("Hα", resp_lines.parsed)

        # Fallback simulation
        fb_config = {"frontier_type": "mock", "mock_answer": "A", "simulate_fallback": True}
        fb_model = get_frontier_model(fb_config)
        fb_resp = fb_model.predict(
            "Classify distance",
            parse_fn=lambda x: "A" if "FINAL ANSWER: A" in x else None,
            fallback_tag="\n\nFINAL ANSWER: ",
        )
        self.assertTrue(fb_resp.forced_fallback)
        self.assertEqual(fb_resp.parsed, "A")

    def test_caption_distance_task(self):
        task = get_caption_task("caption_distance", scheme="3-group")
        prompt = task.build_frontier_prompt("The spectrum shows a galaxy at z=0.05 with narrow emission.")
        self.assertIn("Allowed categories:", prompt)
        self.assertIn("The spectrum shows a galaxy at z=0.05", prompt)
        self.assertIn("FINAL ANSWER: [Letter]", prompt)

        parsed = task.default_parse("After thinking, here is the result.\nFINAL ANSWER: B")
        self.assertEqual(parsed, "B")

        # Ground truth extraction
        gt = task.extract_ground_truth({"Z": 0.05})
        self.assertEqual(gt, "A")
        gt_high = task.extract_ground_truth({"Z": 0.5})
        self.assertEqual(gt_high, "C")

    def test_caption_emission_line_task(self):
        task = get_caption_task("caption_emission_lines")
        prompt = task.build_frontier_prompt("Strong Balmer lines with H-alpha and H-beta detected.")
        self.assertIn("Allowed candidate lines:", prompt)
        self.assertIn("Strong Balmer lines with H-alpha", prompt)
        self.assertIn("EMISSION LINES: line1, line2, ...", prompt)

        parsed = task.default_parse("The spectrum has features.\nEMISSION LINES: Hα, Hβ")
        self.assertIn("Hα", parsed)
        self.assertIn("Hβ", parsed)

        parsed_none = task.default_parse("EMISSION LINES: NONE")
        self.assertEqual(parsed_none, [])

    def test_caption_categorical_tasks(self):
        source_task = get_caption_task(
            "caption_source",
            active_classes={"GALAXY": "Galaxy", "QSO": "Quasar"},
        )
        prompt = source_task.build_frontier_prompt("Broad emission lines and blue continuum.")
        self.assertIn("Galaxy, Quasar", prompt)
        self.assertIn("FINAL ANSWER: [Category]", prompt)

        parsed = source_task.default_parse("FINAL ANSWER: Quasar")
        self.assertEqual(parsed, "Quasar")

        gt = source_task.extract_ground_truth({"class": "QSO"})
        self.assertEqual(gt, "Quasar")

        subclass_task = get_caption_task(
            "caption_subclass",
            active_classes={"AGN": "AGN", "STARBURST": "Starburst"},
        )
        gt_sub = subclass_task.extract_ground_truth({"subclass": "AGN"})
        self.assertEqual(gt_sub, "AGN")

    def test_compute_caption_metrics_single_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preds_file = os.path.join(tmpdir, "predictions.jsonl")
            records = [
                {
                    "sample_id": "1",
                    "ground_truth": "A",
                    "caption": {"text": "A brief description of sample 1."},
                    "frontier_evaluation": {
                        "prediction": "A",
                        "raw_response": "FINAL ANSWER: A",
                        "is_correct": True,
                        "forced_fallback": False,
                    },
                },
                {
                    "sample_id": "2",
                    "ground_truth": "B",
                    "caption": {"text": "Another detailed astronomical description."},
                    "frontier_evaluation": {
                        "prediction": "C",
                        "raw_response": "FINAL ANSWER: C",
                        "is_correct": False,
                        "forced_fallback": True,
                    },
                },
            ]
            with open(preds_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            task = CaptionDistanceTask(scheme="3-group")
            metrics = compute_caption_metrics(tmpdir, task)

            self.assertEqual(metrics["total_samples"], 2)
            self.assertEqual(metrics["task_metrics"]["accuracy"], 0.5)
            self.assertEqual(metrics["caption_diagnostics"]["frontier_fallback_rate"], 0.5)
            self.assertEqual(metrics["caption_diagnostics"]["frontier_parse_success_rate"], 1.0)

            # Check report.md was generated
            report_path = os.path.join(tmpdir, "report.md")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r") as rf:
                content = rf.read()
                self.assertIn("# Evaluation Report: caption_distance", content)
                self.assertIn("Confusion Matrix", content)
                self.assertIn("Qualitative Spotlights", content)

    def test_compute_caption_metrics_multilabel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preds_file = os.path.join(tmpdir, "predictions.jsonl")
            records = [
                {
                    "sample_id": "1",
                    "ground_truth": {"Hα": 25.0, "Hβ": 10.0},
                    "caption": {"text": "A spectrum with strong Balmer emission."},
                    "frontier_evaluation": {
                        "prediction": ["Hα", "Hβ"],
                        "raw_response": "EMISSION LINES: Hα, Hβ",
                        "is_correct": True,
                        "forced_fallback": False,
                    },
                },
                {
                    "sample_id": "2",
                    "ground_truth": {"Hα": 15.0},
                    "caption": {"text": "A faint continuum with weak features."},
                    "frontier_evaluation": {
                        "prediction": ["Hβ"],
                        "raw_response": "EMISSION LINES: Hβ",
                        "is_correct": False,
                        "forced_fallback": False,
                    },
                },
            ]
            with open(preds_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            task = CaptionEmissionLineTask()
            metrics = compute_caption_metrics(tmpdir, task)

            self.assertEqual(metrics["total_samples"], 2)
            self.assertIn("sample_precision", metrics["task_metrics"])
            self.assertIn("sample_recall", metrics["task_metrics"])
            self.assertIn("sample_f1", metrics["task_metrics"])
            self.assertIn("snr_weighted_f1", metrics["task_metrics"])
            self.assertIn("per_line", metrics["task_metrics"])

            report_path = os.path.join(tmpdir, "report.md")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r") as rf:
                content = rf.read()
                self.assertIn("# Evaluation Report: caption_emission_lines", content)
                self.assertIn("Per-Line Detection Statistics", content)

    def test_run_caption_evaluation_pipeline(self):
        synthetic_df = pd.DataFrame([
            {
                "wiki_entity_id": f"id_{i}",
                "survey": "sdss",
                "Z": 0.05 * (i + 1),
                "spectrum": {
                    "flux": [1.0, 2.0, 1.5],
                    "lambda": [4000, 5000, 6000],
                    "mask": [False, False, False],
                }
            }
            for i in range(3)
        ])

        with patch("eval_scripts.run_caption_eval_local.load_test_spectra", return_value=synthetic_df):
            with tempfile.TemporaryDirectory() as tmpdir:
                task_config = {"name": "caption_distance", "kwargs": {"scheme": "3-group"}}
                frontier_config = {"frontier_type": "mock", "mock_answer": "A"}
                output_dir = os.path.join(tmpdir, "eval_run")
                responder_config = {"responder_type": "mock", "caption_prompt": "Describe spectrum"}

                metrics = run_caption_evaluation(
                    task_config=task_config,
                    frontier_config=frontier_config,
                    responder_config=responder_config,
                    captions_file=None,
                    output_dir=output_dir,
                    limit=3,
                )

                self.assertTrue(os.path.exists(os.path.join(output_dir, "captions.jsonl")))
                self.assertTrue(os.path.exists(os.path.join(output_dir, "predictions.jsonl")))
                self.assertTrue(os.path.exists(os.path.join(output_dir, "metrics.json")))
                self.assertTrue(os.path.exists(os.path.join(output_dir, "report.md")))
                self.assertTrue(os.path.exists(os.path.join(output_dir, "run_config.json")))
                self.assertEqual(metrics["total_samples"], 3)


if __name__ == "__main__":
    unittest.main()

