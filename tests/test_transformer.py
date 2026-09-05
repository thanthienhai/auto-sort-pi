"""Kiểm thử phép chiếu pixel → world qua homography + validate calibration (§7/§8)."""

import json

import pytest

from modules.exceptions import CalibrationError
from modules.transformer import CoordinateTransformer


def _write_calib(tmp_path, matrix, **extra):
    path = tmp_path / "calib.json"
    payload = {"homography_matrix": matrix, **extra}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _real_calib_extra():
    """Các field bắt buộc để một calibration được coi là "thật" (không placeholder)."""
    return dict(
        source_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        destination_points_mm=[[0, 0], [100, 0], [100, 100], [0, 100]],
        reprojection_error_mm=0.8,
        image_width=1920,
        image_height=1080,
        created_at="2026-01-01T00:00:00Z",
    )


def test_identity_homography_is_passthrough_when_allowed(tmp_path):
    calib = _write_calib(tmp_path, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tf = CoordinateTransformer(calib, allow_identity=True)
    assert tf.pixel_to_world(123.4, 56.7) == (123.4, 56.7)


def test_scale_and_translation(tmp_path):
    # X = 2u + 10, Y = 3v - 5
    calib = _write_calib(tmp_path, [[2, 0, 10], [0, 3, -5], [0, 0, 1]], **_real_calib_extra())
    tf = CoordinateTransformer(calib)
    assert tf.pixel_to_world(100, 100) == (210.0, 295.0)


def test_perspective_division(tmp_path):
    # w' = 0.5 -> toạ độ được chia đôi mẫu số
    calib = _write_calib(tmp_path, [[1, 0, 0], [0, 1, 0], [0, 0, 0.5]], **_real_calib_extra())
    tf = CoordinateTransformer(calib)
    assert tf.pixel_to_world(50, 80) == (100.0, 160.0)


def test_rejects_non_3x3_matrix(tmp_path):
    calib = _write_calib(tmp_path, [[1, 0], [0, 1]], **_real_calib_extra())
    with pytest.raises(CalibrationError):
        CoordinateTransformer(calib)


def test_rejects_missing_file():
    with pytest.raises(CalibrationError):
        CoordinateTransformer("does/not/exist.json")


def test_rejects_identity_matrix_as_placeholder(tmp_path):
    calib = _write_calib(tmp_path, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], **_real_calib_extra())
    with pytest.raises(CalibrationError, match="identity"):
        CoordinateTransformer(calib)


def test_rejects_missing_calibration_points(tmp_path):
    calib = _write_calib(tmp_path, [[2, 0, 10], [0, 3, -5], [0, 0, 1]])
    with pytest.raises(CalibrationError, match="source_points"):
        CoordinateTransformer(calib)


def test_rejects_reprojection_error_over_threshold(tmp_path):
    extra = _real_calib_extra()
    extra["reprojection_error_mm"] = 5.0
    calib = _write_calib(tmp_path, [[2, 0, 10], [0, 3, -5], [0, 0, 1]], **extra)
    with pytest.raises(CalibrationError, match="Reprojection error"):
        CoordinateTransformer(calib, max_reprojection_error_mm=3.0)


def test_accepts_reprojection_error_within_threshold(tmp_path):
    extra = _real_calib_extra()
    extra["reprojection_error_mm"] = 1.2
    calib = _write_calib(tmp_path, [[2, 0, 10], [0, 3, -5], [0, 0, 1]], **extra)
    tf = CoordinateTransformer(calib, max_reprojection_error_mm=3.0)
    assert tf.pixel_to_world(0, 0) == (10.0, -5.0)
