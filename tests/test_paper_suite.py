from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from eval_scripts.run_paper_captions import (
    execute_local_model,
    execute_modal_model,
    load_suite_config,
    resolve_models_to_run,
)
from evals.hub import (
    check_and_download_existing,
    get_hf_token,
    upload_caption_file,
    write_run_metadata,
)


class TestPaperSuite(unittest.TestCase):
    def test_load_suite_config_canonical(self):
        config = load_suite_config("paper_controls")
        self.assertIn("models", config)
        self.assertEqual(config.get("suite_name"), "paper_controls")
        model_ids = [m["id"] for m in config["models"]]
        self.assertIn("astrobridge", model_ids)
        self.assertIn("qwen_3.5_9b_vision", model_ids)
        self.assertIn("qwen_3.5_9b_text", model_ids)
        self.assertIn("gemma_4_12b_vision", model_ids)
        self.assertIn("gemma_4_31b_text", model_ids)
        self.assertIn("qwen_3.8_27b_vision", model_ids)
        self.assertIn("meta_muse_glimmer_text", model_ids)
        self.assertEqual(len(model_ids), 11)

        for m in config["models"]:
            self.assertIn("batch_size", m)
            self.assertIsInstance(m["batch_size"], int)
            self.assertGreater(m["batch_size"], 0)
            self.assertIn("responder_type", m)

    def test_load_suite_config_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_suite_config("non_existent_suite_xyz")

    def test_resolve_models_to_run_all(self):
        models = [
            {"id": "m1", "name": "Model 1"},
            {"id": "m2", "name": "Model 2"},
        ]
        res = resolve_models_to_run(models, None)
        self.assertEqual(len(res), 2)

    def test_resolve_models_to_run_filtered_by_id(self):
        models = [
            {"id": "astrobridge", "name": "AstroBridge"},
            {"id": "qwen_vision", "name": "Vision"},
            {"id": "qwen_text", "name": "Text"},
        ]
        res = resolve_models_to_run(models, ["astrobridge", "qwen_text"])
        self.assertEqual([m["id"] for m in res], ["astrobridge", "qwen_text"])

    def test_resolve_models_to_run_filtered_by_responder_type(self):
        models = [
            {"id": "qwen_vision", "responder_type": "hf_vision"},
            {"id": "gemma_vision", "responder_type": "hf_vision"},
            {"id": "qwen_text", "responder_type": "hf_text"},
        ]
        res = resolve_models_to_run(models, responder_type="hf_vision")
        self.assertEqual([m["id"] for m in res], ["qwen_vision", "gemma_vision"])

    def test_resolve_models_to_run_invalid_id(self):
        models = [{"id": "astrobridge"}, {"id": "qwen_vision"}]
        with self.assertRaises(ValueError) as ctx:
            resolve_models_to_run(models, ["invalid_model"])
        self.assertIn("invalid_model", str(ctx.exception))

    def test_default_output_filename(self):
        m1 = {"id": "my_model"}
        m2 = {"id": "my_model", "output_filename": "custom_name.jsonl"}
        self.assertEqual(m1.get("output_filename") or f"{m1['id']}.jsonl", "my_model.jsonl")
        self.assertEqual(m2.get("output_filename") or f"{m2['id']}.jsonl", "custom_name.jsonl")

    def test_auto_config_path_resolution(self):
        m = {"id": "m1", "responder_type": "hf_vision"}
        cfg_rel_path = m.get("config") or f"eval_configs/responders/{m['responder_type']}.yaml"
        self.assertEqual(cfg_rel_path, "eval_configs/responders/hf_vision.yaml")
        self.assertTrue((_REPO_ROOT / cfg_rel_path).is_file())

    @patch("subprocess.run")
    def test_execute_local_model_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        execute_local_model(
            model_config_path="/path/to/config.yaml",
            output_path="/path/to/out.jsonl",
            benchmark="source",
            limit=50,
            batch_size=16,
            devices="0,1",
            num_gpus=2,
        )
        self.assertTrue(mock_run.called)
        cmd = mock_run.call_args[0][0]
        self.assertIn("generate_captions.py", cmd[1])
        self.assertIn("--responder", cmd)
        self.assertIn("/path/to/config.yaml", cmd)
        self.assertIn("--output", cmd)
        self.assertIn("/path/to/out.jsonl", cmd)
        self.assertIn("--benchmark", cmd)
        self.assertIn("source", cmd)
        self.assertIn("--limit", cmd)
        self.assertIn("50", cmd)
        self.assertIn("--batch-size", cmd)
        self.assertIn("16", cmd)
        self.assertIn("--devices", cmd)
        self.assertIn("0,1", cmd)
        self.assertIn("--num-gpus", cmd)
        self.assertIn("2", cmd)

    @patch("subprocess.run")
    def test_execute_modal_model_env_and_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        cfg_path = _REPO_ROOT / "eval_configs" / "responders" / "astrobridge.yaml"
        execute_modal_model(
            model_config_path=cfg_path,
            output_path="/tmp/out.jsonl",
            benchmark="all",
            limit=10,
            batch_size=8,
            modal_gpu="A100-80GB",
        )
        self.assertTrue(mock_run.called)
        args_tuple, kwargs = mock_run.call_args
        cmd = args_tuple[0]
        self.assertEqual(cmd[0], "modal")
        self.assertEqual(cmd[1], "run")
        self.assertIn("run_modal.py", cmd[2])
        self.assertIn("--output", cmd)
        self.assertIn("/tmp/out.jsonl", cmd)

        passed_env = kwargs.get("env", {})
        self.assertEqual(passed_env.get("MODAL_GPU"), "A100-80GB")

    def test_get_hf_token(self):
        with patch.dict(os.environ, {"HF_TOKEN": "test_token_123"}, clear=False):
            self.assertEqual(get_hf_token(), "test_token_123")

    def test_upload_caption_file_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            upload_caption_file("/non/existent/file.jsonl", token="dummy")

    def test_upload_caption_file_missing_token(self):
        with tempfile.NamedTemporaryFile() as tmp:
            with patch.dict(os.environ, {}, clear=True), patch("evals.hub.get_hf_token", return_value=None):
                with self.assertRaises(ValueError):
                    upload_caption_file(tmp.name)

    @patch("evals.hub.HfApi")
    def test_upload_caption_file_success(self, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.upload_file.return_value = "https://huggingface.co/commit/123"

        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(b'{"sample_id": "test"}\n')
            tmp.flush()

            res = upload_caption_file(
                local_path=tmp.name,
                hf_repo="UniverseTBD/AstroBridge-Data",
                hf_path="evals/captions/test.jsonl",
                token="test_token",
            )
            self.assertIn("123", res)
            self.assertTrue(mock_api.upload_file.called)

    @patch("evals.hub.HfApi")
    @patch("evals.hub.hf_hub_download")
    def test_check_and_download_existing(self, mock_download, mock_hf_api_cls):
        mock_api = MagicMock()
        mock_hf_api_cls.return_value = mock_api
        mock_api.file_exists.return_value = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = Path(tmp_dir) / "downloaded.jsonl"
            src_file.write_text('{"sample_id": "1"}')
            mock_download.return_value = str(src_file)

            dest_file = Path(tmp_dir) / "target.jsonl"
            downloaded = check_and_download_existing(
                hf_repo="UniverseTBD/AstroBridge-Data",
                hf_path="evals/captions/target.jsonl",
                local_path=dest_file,
                token="test_token",
            )
            self.assertTrue(downloaded)
            self.assertTrue(dest_file.exists())
            self.assertEqual(dest_file.read_text(), '{"sample_id": "1"}')

    def test_write_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            suite_cfg = {
                "suite_name": "test_suite",
                "benchmark": "all",
                "hf_repo": "UniverseTBD/AstroBridge-Data",
                "hf_subdir": "evals/captions",
            }
            meta_path = write_run_metadata(tmp_dir, suite_cfg, ["astrobridge", "hf_vision"])
            self.assertTrue(meta_path.exists())
            data = json.loads(meta_path.read_text())
            self.assertEqual(data["suite_name"], "test_suite")
            self.assertEqual(data["models_run"], ["astrobridge", "hf_vision"])
            self.assertEqual(data["hf_subdir"], "evals/captions")

    @patch("subprocess.run")
    def test_execute_local_model_with_model_id_override(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        execute_local_model(
            model_config_path="/path/to/config.yaml",
            output_path="/path/to/out.jsonl",
            model_id="google/paligemma2-10b",
        )
        self.assertTrue(mock_run.called)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--model-id", cmd)
        self.assertIn("google/paligemma2-10b", cmd)

    @patch("subprocess.run")
    def test_execute_modal_model_with_model_id_override(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        cfg_path = _REPO_ROOT / "eval_configs" / "responders" / "hf_vision.yaml"
        execute_modal_model(
            model_config_path=cfg_path,
            output_path="/tmp/out.jsonl",
            model_id="google/paligemma2-10b",
        )
        self.assertTrue(mock_run.called)
        args_tuple, _ = mock_run.call_args
        cmd = args_tuple[0]
        args_str_idx = cmd.index("--args") + 1
        self.assertIn("--model-id google/paligemma2-10b", cmd[args_str_idx])

    @patch("subprocess.run")
    def test_execute_local_model_with_astrobridge_id_override(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        execute_local_model(
            model_config_path="/path/to/astrobridge.yaml",
            output_path="/path/to/astrobridge.jsonl",
            model_id="UniverseTBD/astrobridge-model-v8",
        )
        self.assertTrue(mock_run.called)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--model-id", cmd)
        self.assertIn("UniverseTBD/astrobridge-model-v8", cmd)


if __name__ == "__main__":
    unittest.main()

