from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from eval_scripts.generate_captions import resolve_devices, run_caption_generation


def _make_dummy_dataframe(n: int = 7) -> pd.DataFrame:
    records = []
    for i in range(n):
        records.append(
            {
                "sample_id": f"dummy_{i:03d}",
                "survey": "desi" if i % 2 == 0 else "sdss",
                "spectrum": {
                    "lambda": [4000.0, 5000.0, 6000.0],
                    "flux": [float(i), float(i + 1), float(i + 2)],
                    "mask": [0, 0, 0],
                    "ivar": [1.0, 1.0, 1.0],
                },
            }
        )
    return pd.DataFrame(records)


class TestMultiDeviceGeneration(unittest.TestCase):
    def test_resolve_devices_explicit_strings(self):
        self.assertEqual(resolve_devices(devices="0,1"), ["cuda:0", "cuda:1"])
        self.assertEqual(resolve_devices(devices="cuda:2,cuda:3"), ["cuda:2", "cuda:3"])
        self.assertEqual(resolve_devices(devices=["0", "1"]), ["cuda:0", "cuda:1"])
        self.assertEqual(resolve_devices(devices="cpu,cpu"), ["cpu", "cpu"])

    def test_resolve_devices_num_gpus(self):
        self.assertEqual(resolve_devices(num_gpus=3), ["cuda:0", "cuda:1", "cuda:2"])
        with self.assertRaises(ValueError):
            resolve_devices(num_gpus=0)

    def test_resolve_devices_single_device(self):
        self.assertEqual(resolve_devices(device="cuda:2"), ["cuda:2"])
        self.assertEqual(resolve_devices(device="cpu"), ["cpu"])

    def test_resolve_devices_default(self):
        devices = resolve_devices()
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0)
        for d in devices:
            self.assertIsInstance(d, str)

    def test_single_device_generation(self):
        df = _make_dummy_dataframe(5)
        config = {
            "responder_type": "mock",
            "mock_caption": "Mock caption test single.",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "captions.jsonl"
            total = run_caption_generation(
                responder_config=config,
                output_path=out_file,
                batch_size=2,
                device="cpu",
                df=df,
            )
            self.assertEqual(total, 5)
            self.assertTrue(out_file.exists())

            lines = [json.loads(line) for line in out_file.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), 5)
            for i, item in enumerate(lines):
                self.assertEqual(item["sample_id"], f"dummy_{i:03d}")
                self.assertEqual(item["caption"], "Mock caption test single.")

    def test_multi_worker_batch_routing_and_ordering(self):
        # 7 samples with batch_size=2 results in 4 batches: [2, 2, 2, 1]
        # Two workers will asynchronously claim and process batches
        df = _make_dummy_dataframe(7)
        config = {
            "responder_type": "mock",
            "mock_caption": "Mock caption test multi-worker.",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "captions.jsonl"
            total = run_caption_generation(
                responder_config=config,
                output_path=out_file,
                batch_size=2,
                devices=["cpu", "cpu"],
                df=df,
            )
            self.assertEqual(total, 7)
            self.assertTrue(out_file.exists())

            lines = [json.loads(line) for line in out_file.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), 7)
            # Ensure deterministic order matching original dataframe
            for i, item in enumerate(lines):
                self.assertEqual(item["sample_id"], f"dummy_{i:03d}")
                self.assertEqual(item["caption"], "Mock caption test multi-worker.")

    def test_multi_worker_error_handling(self):
        df = _make_dummy_dataframe(4)
        config = {
            "responder_type": "unknown_nonexistent_responder",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "captions.jsonl"
            with self.assertRaises(RuntimeError):
                run_caption_generation(
                    responder_config=config,
                    output_path=out_file,
                    batch_size=2,
                    devices=["cpu", "cpu"],
                    df=df,
                )


if __name__ == "__main__":
    unittest.main()

