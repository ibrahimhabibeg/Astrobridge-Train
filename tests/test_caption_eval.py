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
from evals.caption_responders.hf_vision import clean_and_extract_caption
from evals.caption_tasks import (
    get_caption_task,
    CaptionDistanceTask,
    CaptionEmissionLineTask,
    CategoricalCaptionTask,
)
from evals.data import (
    BenchmarkDataManager,
    BenchmarkSpec,
    get_default_data_manager,
    load_benchmark_dataset,
    load_all_benchmark_spectra,
)
from evals.config import (
    DEFAULT_HF_DATA_REPO,
    DEFAULT_HF_EVALS_SUBDIR,
    get_default_cache_dir,
    get_hf_benchmark_subpath,
    get_hf_data_repo,
    get_hf_evals_subdir,
    get_repo_root,
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

    def test_clean_and_extract_caption(self):
        # 1. Plain caption without tags
        raw1 = "This optical spectrum exhibits a flat continuum with strong H-alpha emission."
        self.assertEqual(clean_and_extract_caption(raw1), raw1)

        # 2. Caption with CAPTION: prefix
        raw2 = "<think>Analyzing plot...</think>\nCAPTION: Galaxy spectrum with narrow [O III] lines."
        self.assertEqual(clean_and_extract_caption(raw2), "Galaxy spectrum with narrow [O III] lines.")

        # 3. Caption with Draft and Final polish note
        raw3 = "**Draft:** CAPTION: Quasar spectrum showing broad Balmer lines.\n\nNote: Red line is continuum."
        self.assertEqual(clean_and_extract_caption(raw3), "Quasar spectrum showing broad Balmer lines.")

        # 4. Caption with reasoning preamble and final answer
        raw4 = "The user wants a caption.\n\n1. Analyze image: x-axis 4000 to 9000.\n\nCAPTION: Low continuum spectrum with prominent emission lines."
        self.assertEqual(clean_and_extract_caption(raw4), "Low continuum spectrum with prominent emission lines.")

    def test_frontier_model_mock(self):
        config = {"frontier_type": "mock", "mock_answer": "C"}
        model = get_frontier_model(config)

        # Distance task test
        resp = model.predict("Classify distance... FINAL ANSWER: [Letter]", parse_fn=lambda x: "C")
        self.assertEqual(resp.parsed, "C")
        self.assertIn("FINAL ANSWER: C", resp.raw_text)

        # Emission line task test
        def parse_lines(text):
            return ["HALPHA", "HBETA"] if "Hα" in text or "HALPHA" in text else []

        resp_lines = model.predict("EMISSION LINES: candidate list...", parse_fn=parse_lines)
        self.assertIn("HALPHA", resp_lines.parsed)

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

        # Ground truth extraction with benchmark row schema
        gt_row = {"ground_truth": {"z": 0.05, "redshift_bin": "<0.1"}}
        self.assertEqual(task.extract_ground_truth(gt_row), "A")

        gt_row_high = {"ground_truth": {"z": 1.2, "redshift_bin": ">0.5"}}
        self.assertEqual(task.extract_ground_truth(gt_row_high), "C")

        # Fallback legacy extraction
        self.assertEqual(task.extract_ground_truth({"Z": 0.25}), "B")

    def test_caption_emission_line_task(self):
        task = get_caption_task("caption_emission_lines")
        candidates = ["HALPHA", "OIII_5007", "SII_6716"]

        prompt = task.build_frontier_prompt(
            "Strong Balmer lines with H-alpha and [O III] detected.",
            item={"candidate_query_lines": candidates},
        )
        self.assertIn("Allowed candidate lines:", prompt)
        self.assertIn("HALPHA", prompt)
        self.assertIn("OIII_5007", prompt)
        self.assertIn("EMISSION LINES: line1, line2, ...", prompt)

        # Parsing aliases and canonical mappings
        parsed = task.default_parse("The spectrum has features.\nEMISSION LINES: Hα, [O III] 5007", candidate_lines=candidates)
        self.assertIn("HALPHA", parsed)
        self.assertIn("OIII_5007", parsed)

        parsed_none = task.default_parse("EMISSION LINES: NONE", candidate_lines=candidates)
        self.assertEqual(parsed_none, [])

        # Ground truth extraction from benchmark format
        row_gt = {
            "ground_truth": {
                "detected_lines": ["HALPHA", "OIII_5007"],
                "absent_lines": ["SII_6716"],
                "line_details": {
                    "HALPHA": {"snr": 25.0, "detected": True},
                    "OIII_5007": {"snr": 12.0, "detected": True},
                    "SII_6716": {"snr": 0.5, "detected": False},
                }
            }
        }
        gt = task.extract_ground_truth(row_gt)
        self.assertEqual(gt, {"HALPHA": 25.0, "OIII_5007": 12.0})

    def test_caption_categorical_tasks(self):
        source_task = get_caption_task(
            "caption_source",
            active_classes={"GALAXY": "Galaxy", "QUASAR": "Quasar", "QSO": "Quasar"},
        )
        prompt = source_task.build_frontier_prompt("Broad emission lines and blue continuum.")
        self.assertIn("Galaxy, Quasar", prompt)
        self.assertIn("FINAL ANSWER: [Category]", prompt)

        parsed = source_task.default_parse("FINAL ANSWER: Quasar")
        self.assertEqual(parsed, "Quasar")

        # Test benchmark ground truth extraction
        gt = source_task.extract_ground_truth({"ground_truth": {"source_class": "QUASAR", "subclass": ""}})
        self.assertEqual(gt, "Quasar")

        subclass_task = get_caption_task(
            "caption_subclass",
            active_classes={"AGN": "AGN", "STARBURST": "Starburst", "BROADLINE": "Broadline"},
        )
        gt_sub = subclass_task.extract_ground_truth({"ground_truth": {"subclass": "BROADLINE", "source_class": "QSO"}})
        self.assertEqual(gt_sub, "Broadline")

    def test_load_benchmark_datasets(self):
        # Verify local benchmarks load properly
        for name in ["redshift", "source_class", "subclass", "emission_lines"]:
            df = load_benchmark_dataset(name)
            self.assertGreater(len(df), 0)
            self.assertIn("sample_id", df.columns)
            self.assertIn("spectrum", df.columns)
            self.assertIn("ground_truth", df.columns)

        all_df = load_all_benchmark_spectra()
        self.assertEqual(len(all_df), 717)
        self.assertIn("sample_id", all_df.columns)

    def test_validate_and_normalize_strict(self):
        mgr = get_default_data_manager()
        spec = mgr.resolve_spec("redshift")

        # 1. Missing both sample_id and object_id raises ValueError
        df_no_id = pd.DataFrame({
            "ra": [1, 2],
            "survey": ["sdss", "sdss"],
            "spectrum": [[1.0], [2.0]],
            "ground_truth": [0.1, 0.2],
        })
        with self.assertRaises(ValueError) as ctx:
            mgr._validate_and_normalize(df_no_id, spec)
        self.assertIn("missing required identifier column", str(ctx.exception))

        # 2. Missing survey raises ValueError (never assume SDSS)
        df_no_survey = pd.DataFrame({
            "sample_id": ["1", "2"],
            "spectrum": [[1.0], [2.0]],
            "ground_truth": [0.1, 0.2],
        })
        with self.assertRaises(ValueError) as ctx:
            mgr._validate_and_normalize(df_no_survey, spec)
        self.assertIn("missing required 'survey' column", str(ctx.exception))

        # 3. Missing spectrum raises ValueError
        df_no_spec = pd.DataFrame({
            "sample_id": ["1", "2"],
            "survey": ["desi", "desi"],
            "ground_truth": [0.1, 0.2],
        })
        with self.assertRaises(ValueError) as ctx:
            mgr._validate_and_normalize(df_no_spec, spec)
        self.assertIn("missing the 'spectrum' column", str(ctx.exception))

        # 4. Missing ground_truth raises ValueError
        df_no_gt = pd.DataFrame({
            "sample_id": ["1", "2"],
            "survey": ["desi", "desi"],
            "spectrum": [[1.0], [2.0]],
        })
        with self.assertRaises(ValueError) as ctx:
            mgr._validate_and_normalize(df_no_gt, spec)
        self.assertIn("missing the 'ground_truth' column", str(ctx.exception))

        # 5. Standardizes sample_id from sample_id column
        df_valid_sample = pd.DataFrame({
            "sample_id": [12345, 67890],
            "survey": ["desi", "sdss"],
            "spectrum": [[1.0], [2.0]],
            "ground_truth": [0.1, 0.2],
        })
        normalized_sample = mgr._validate_and_normalize(df_valid_sample, spec)
        self.assertEqual(normalized_sample["sample_id"].tolist(), ["12345", "67890"])
        self.assertEqual(normalized_sample["survey"].tolist(), ["desi", "sdss"])

        # 6. Standardizes sample_id from object_id column
        df_valid_object = pd.DataFrame({
            "object_id": [12345, 67890],
            "survey": ["desi", "sdss"],
            "spectrum": [[1.0], [2.0]],
            "ground_truth": [0.1, 0.2],
        })
        normalized_object = mgr._validate_and_normalize(df_valid_object, spec)
        self.assertEqual(normalized_object["sample_id"].tolist(), ["12345", "67890"])
        self.assertEqual(normalized_object["survey"].tolist(), ["desi", "sdss"])

        # 7. Missing expected ground truth keys in dict raises ValueError
        df_invalid_gt = pd.DataFrame({
            "sample_id": ["1", "2"],
            "survey": ["desi", "sdss"],
            "spectrum": [[1.0], [2.0]],
            "ground_truth": [{"z": 0.1}, {"z": 0.2}],
        })
        with self.assertRaises(ValueError) as ctx:
            mgr._validate_and_normalize(df_invalid_gt, spec)
        self.assertIn("missing expected ground truth key 'redshift_bin'", str(ctx.exception))

    def test_benchmark_data_manager_canonical_specs(self):
        mgr = get_default_data_manager()
        self.assertIn("redshift", mgr.list_available_benchmarks())
        self.assertIn("emission_lines", mgr.list_available_benchmarks())

        # Test canonical resolution
        spec_dist = mgr.resolve_spec("redshift")
        self.assertEqual(spec_dist.name, "redshift")
        self.assertEqual(spec_dist.filename, "redshift.parquet")

        spec_src = mgr.resolve_spec("source_class")
        self.assertEqual(spec_src.name, "source_class")

        spec_em = mgr.resolve_spec("emission_lines")
        self.assertEqual(spec_em.name, "emission_lines")
        self.assertEqual(
            spec_dist.required_columns,
            ("sample_id", "survey", "spectrum", "ground_truth"),
        )

        # Test non-canonical name raises clean ValueError
        with self.assertRaises(ValueError) as ctx:
            mgr.resolve_spec("distance")
        self.assertIn("Unknown benchmark 'distance'", str(ctx.exception))
        self.assertIn("Valid benchmarks", str(ctx.exception))

    def test_eval_config(self):
        root = get_repo_root()
        self.assertTrue(root.is_dir())
        self.assertEqual(get_hf_data_repo(), DEFAULT_HF_DATA_REPO)
        self.assertEqual(get_hf_evals_subdir(), DEFAULT_HF_EVALS_SUBDIR)
        self.assertEqual(
            get_hf_benchmark_subpath("redshift.parquet"),
            "evals/spectra/redshift.parquet",
        )
        cache_dir = get_default_cache_dir()
        self.assertTrue(str(cache_dir).endswith("data/benchmarks"))

        # Test environment variable overrides
        with patch.dict(os.environ, {"ASTROBRIDGE_HF_DATA_REPO": "Custom/Repo", "ASTROBRIDGE_HF_EVALS_SUBDIR": "custom/path"}):
            self.assertEqual(get_hf_data_repo(), "Custom/Repo")
            self.assertEqual(get_hf_evals_subdir(), "custom/path")
            self.assertEqual(get_hf_benchmark_subpath("test.parquet"), "custom/path/test.parquet")



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
                    "regime": "high_snr_positive",
                    "ground_truth": {"HALPHA": 25.0, "HBETA": 10.0},
                    "caption": {"text": "A spectrum with strong Balmer emission."},
                    "frontier_evaluation": {
                        "prediction": ["HALPHA", "HBETA"],
                        "raw_response": "EMISSION LINES: HALPHA, HBETA",
                        "is_correct": True,
                        "forced_fallback": False,
                    },
                },
                {
                    "sample_id": "2",
                    "regime": "pure_negative",
                    "ground_truth": {},
                    "caption": {"text": "A flat continuum without any detectable features."},
                    "frontier_evaluation": {
                        "prediction": [],
                        "raw_response": "EMISSION LINES: NONE",
                        "is_correct": True,
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
            self.assertIn("sample_jaccard", metrics["task_metrics"])
            self.assertIn("dataset_precision", metrics["task_metrics"])
            self.assertIn("dataset_recall", metrics["task_metrics"])
            self.assertIn("dataset_f1", metrics["task_metrics"])
            self.assertIn("dataset_jaccard", metrics["task_metrics"])
            self.assertIn("exact_match_rate", metrics["task_metrics"])
            self.assertIn("hamming_loss", metrics["task_metrics"])
            self.assertIn("regime_breakdown", metrics["task_metrics"])
            self.assertIn("pure_negative", metrics["task_metrics"]["regime_breakdown"])
            self.assertEqual(metrics["task_metrics"]["regime_breakdown"]["pure_negative"]["clean_negative_rate"], 1.0)

            report_path = os.path.join(tmpdir, "report.md")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r") as rf:
                content = rf.read()
                self.assertIn("# Evaluation Report: caption_emission_lines", content)
                self.assertIn("Sample-Mean Jaccard (IoU)", content)
                self.assertIn("Emission Line Regime Breakdown", content)

    def test_run_caption_evaluation_pipeline(self):
        synthetic_df = pd.DataFrame([
            {
                "object_id": f"id_{i}",
                "sample_id": f"id_{i}",
                "survey": "sdss",
                "ground_truth": {"z": 0.05 * (i + 1), "redshift_bin": "<0.1"},
                "spectrum": {
                    "flux": [1.0, 2.0, 1.5],
                    "lambda": [4000, 5000, 6000],
                    "mask": [False, False, False],
                }
            }
            for i in range(3)
        ])

        with patch("eval_scripts.run_caption_eval_local.load_benchmark_dataset", return_value=synthetic_df):
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
