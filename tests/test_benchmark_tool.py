"""Kiểm thử tools/benchmark.py: thống kê mean/p50/p95/p99/fps + các stage đo được thuần."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.detector as detector_module  # noqa: E402
import tools.benchmark as benchmark  # noqa: E402
from modules.detector import WasteDetector  # noqa: E402
from modules.transformer import CoordinateTransformer  # noqa: E402


class FakeInterpreter:
    def __call__(self, model_path):
        return self

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"shape": [1, 320, 320, 3], "dtype": np.float32, "index": 0, "quantization": (0.0, 0)}]

    def get_output_details(self):
        return [{"shape": [1, 7, 20], "dtype": np.float32, "index": 1, "quantization": (0.0, 0)}]

    def set_tensor(self, index, tensor):
        pass

    def invoke(self):
        pass

    def get_tensor(self, index):
        return np.zeros((1, 7, 20), dtype=np.float32)


def test_stats_computes_mean_p50_p95_p99_fps():
    samples = [0.010, 0.020, 0.030, 0.040, 0.050]  # secondes
    stats = benchmark._stats(samples)
    assert stats["mean_ms"] == 30.0
    assert stats["p50_ms"] == 30.0
    assert stats["fps"] == round(1000.0 / 30.0, 2)


def test_stats_empty_samples_returns_none_fields():
    stats = benchmark._stats([])
    assert stats["mean_ms"] is None
    assert stats["fps"] is None


def test_benchmark_scheduler_returns_positive_samples():
    samples = benchmark.benchmark_scheduler(iterations=5, n_targets=8)
    assert len(samples) == 5
    assert all(s >= 0 for s in samples)


def test_benchmark_transform_returns_samples_matching_iterations(tmp_path):
    import json

    calib_path = tmp_path / "calib.json"
    calib_path.write_text(json.dumps({"homography_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}), encoding="utf-8")
    tf = CoordinateTransformer(str(calib_path), allow_identity=True)
    samples = benchmark.benchmark_transform(tf, iterations=10)
    assert len(samples) == 10


def test_benchmark_inference_runs_against_fake_detector(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    monkeypatch.setattr(detector_module, "_load_interpreter", lambda: FakeInterpreter())
    detector = WasteDetector(str(model_file), num_classes=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    samples = benchmark.benchmark_inference(detector, frame, iterations=3)
    assert len(samples) == 3
