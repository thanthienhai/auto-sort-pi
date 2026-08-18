"""Kiểm thử phép chiếu pixel → world qua homography."""

import json

from modules.transformer import CoordinateTransformer


def _write_calib(tmp_path, matrix):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({"homography_matrix": matrix}), encoding="utf-8")
    return str(path)


def test_identity_homography_is_passthrough(tmp_path):
    calib = _write_calib(tmp_path, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tf = CoordinateTransformer(calib)
    assert tf.pixel_to_world(123.4, 56.7) == (123.4, 56.7)


def test_scale_and_translation(tmp_path):
    # X = 2u + 10, Y = 3v - 5
    calib = _write_calib(tmp_path, [[2, 0, 10], [0, 3, -5], [0, 0, 1]])
    tf = CoordinateTransformer(calib)
    assert tf.pixel_to_world(100, 100) == (210.0, 295.0)


def test_perspective_division(tmp_path):
    # w' = 0.5 -> toạ độ được chia đôi mẫu số
    calib = _write_calib(tmp_path, [[1, 0, 0], [0, 1, 0], [0, 0, 0.5]])
    tf = CoordinateTransformer(calib)
    assert tf.pixel_to_world(50, 80) == (100.0, 160.0)


def test_rejects_non_3x3_matrix(tmp_path):
    calib = _write_calib(tmp_path, [[1, 0], [0, 1]])
    try:
        CoordinateTransformer(calib)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for non-3x3 matrix")
