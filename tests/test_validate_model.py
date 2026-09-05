"""Kiểm thử tools/validate_model.py bằng fake TFLite interpreter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.detector as detector_module  # noqa: E402
import tools.validate_model as validate_model  # noqa: E402


class FakeInterpreter:
    def __init__(self, output_shapes):
        self._output_shapes = output_shapes

    def __call__(self, model_path):
        return self

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"shape": [1, 320, 320, 3], "dtype": np.float32, "index": 0, "quantization": (0.0, 0)}]

    def get_output_details(self):
        return [
            {"shape": shape, "dtype": np.float32, "index": i + 1, "quantization": (0.0, 0)}
            for i, shape in enumerate(self._output_shapes)
        ]

    def set_tensor(self, index, tensor):
        pass

    def invoke(self):
        pass

    def get_tensor(self, index):
        return np.zeros(self._output_shapes[0], dtype=np.float32)


def _write_json(path: Path, data) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_missing_model_file_fails_fast(tmp_path):
    report = validate_model.validate(
        str(tmp_path / "missing.tflite"),
        str(tmp_path / "mapping.json"),
        str(tmp_path / "classes.json"),
        0.5,
        0.45,
    )
    assert report.ok is False
    assert report.checks[0][0] == "model file exists"


def test_full_valid_pipeline_passes(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    classes_file = _write_json(tmp_path / "classes.json", ["can", "cardboard", "pet_bottle"])
    mapping_file = _write_json(
        tmp_path / "mapping.json", {"can": 1, "cardboard": 2, "pet_bottle": 3}
    )
    monkeypatch.setattr(
        detector_module, "_load_interpreter", lambda: FakeInterpreter([[1, 7, 100]])
    )

    report = validate_model.validate(str(model_file), mapping_file, classes_file, 0.5, 0.45)
    assert report.ok is True


def test_missing_class_mapping_entry_fails(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    classes_file = _write_json(tmp_path / "classes.json", ["can", "unmapped_class"])
    mapping_file = _write_json(tmp_path / "mapping.json", {"can": 1})
    monkeypatch.setattr(
        detector_module, "_load_interpreter", lambda: FakeInterpreter([[1, 6, 100]])
    )

    report = validate_model.validate(str(model_file), mapping_file, classes_file, 0.5, 0.45)
    mapping_check = next(c for c in report.checks if c[0] == "every model class has a group mapping")
    assert mapping_check[1] is False


def test_invalid_confidence_threshold_fails(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    classes_file = _write_json(tmp_path / "classes.json", ["can"])
    mapping_file = _write_json(tmp_path / "mapping.json", {"can": 1})
    monkeypatch.setattr(
        detector_module, "_load_interpreter", lambda: FakeInterpreter([[1, 5, 100]])
    )

    report = validate_model.validate(str(model_file), mapping_file, classes_file, 1.5, 0.45)
    assert report.ok is False
