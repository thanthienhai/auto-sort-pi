"""Kiểm thử validate model khi khởi tạo WasteDetector (§5/§6 task.md).

Model không tương thích phải raise ``ModelLoadError`` tường minh ngay ở
``__init__`` thay vì âm thầm decode sai lúc infer. Dùng một fake TFLite
Interpreter (qua monkeypatch ``modules.detector._load_interpreter``) để không
phụ thuộc file .tflite thật.
"""

from __future__ import annotations

import numpy as np
import pytest

import modules.detector as detector_module
from modules.detector import WasteDetector
from modules.exceptions import ModelInferenceError, ModelLoadError


class FakeInterpreter:
    def __init__(self, input_shape, input_dtype, output_shapes, fail_invoke=False):
        self._input_shape = input_shape
        self._input_dtype = input_dtype
        self._output_shapes = output_shapes
        self._fail_invoke = fail_invoke
        self._tensors: dict[int, np.ndarray] = {}

    # signature compatible with tflite_runtime.interpreter.Interpreter
    def __call__(self, model_path):
        return self

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"shape": self._input_shape, "dtype": self._input_dtype, "index": 0, "quantization": (0.0, 0)}]

    def get_output_details(self):
        return [
            {"shape": shape, "dtype": np.float32, "index": i + 1, "quantization": (0.0, 0)}
            for i, shape in enumerate(self._output_shapes)
        ]

    def set_tensor(self, index, tensor):
        self._tensors[index] = tensor

    def invoke(self):
        if self._fail_invoke:
            raise RuntimeError("simulated hardware failure")

    def get_tensor(self, index):
        return np.zeros((1, 7, 1), dtype=np.float32)


def _patch_interpreter(monkeypatch, fake_factory):
    monkeypatch.setattr(detector_module, "_load_interpreter", lambda: fake_factory)


def test_missing_model_file_raises_model_load_error(tmp_path):
    with pytest.raises(ModelLoadError):
        WasteDetector(str(tmp_path / "missing.tflite"))


def test_invalid_confidence_threshold_raises(tmp_path):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    with pytest.raises(ModelLoadError):
        WasteDetector(str(model_file), confidence_threshold=1.5)


def test_wrong_input_shape_raises(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    fake = FakeInterpreter(
        input_shape=[1, 320, 320],  # thiếu kênh màu -> rank 3 thay vì 4
        input_dtype=np.float32,
        output_shapes=[[1, 7, 100]],
    )
    _patch_interpreter(monkeypatch, fake)
    with pytest.raises(ModelLoadError, match="Input shape"):
        WasteDetector(str(model_file), num_classes=3)


def test_wrong_input_dtype_raises(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    fake = FakeInterpreter(
        input_shape=[1, 320, 320, 3],
        input_dtype=np.int16,
        output_shapes=[[1, 7, 100]],
    )
    _patch_interpreter(monkeypatch, fake)
    with pytest.raises(ModelLoadError, match="dtype"):
        WasteDetector(str(model_file), num_classes=3)


def test_output_shape_mismatched_num_classes_raises(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    # num_classes=3 -> kỳ vọng feature dim 7, nhưng output là 9 (5 lớp) và != 6.
    fake = FakeInterpreter(
        input_shape=[1, 320, 320, 3],
        input_dtype=np.float32,
        output_shapes=[[1, 9, 100]],
    )
    _patch_interpreter(monkeypatch, fake)
    with pytest.raises(ModelLoadError, match="num_classes"):
        WasteDetector(str(model_file), num_classes=3)


def test_valid_model_spec_loads_successfully(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    fake = FakeInterpreter(
        input_shape=[1, 320, 320, 3],
        input_dtype=np.float32,
        output_shapes=[[1, 7, 100]],
    )
    _patch_interpreter(monkeypatch, fake)
    det = WasteDetector(str(model_file), num_classes=3)
    assert det.input_width == 320
    assert det.input_height == 320


def test_inference_exception_wrapped_as_model_inference_error(tmp_path, monkeypatch):
    model_file = tmp_path / "m.tflite"
    model_file.write_bytes(b"0")
    fake = FakeInterpreter(
        input_shape=[1, 320, 320, 3],
        input_dtype=np.float32,
        output_shapes=[[1, 7, 100]],
        fail_invoke=True,
    )
    _patch_interpreter(monkeypatch, fake)
    det = WasteDetector(str(model_file), num_classes=3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ModelInferenceError):
        det.infer(frame)
