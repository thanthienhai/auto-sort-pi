"""Kiểm thử tools/test_calibration.py: mean/max/RMSE error + reject theo ngưỡng."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import tools.test_calibration as test_calibration  # noqa: E402
from modules.transformer import CoordinateTransformer  # noqa: E402


def _write_calib(tmp_path, matrix, **extra):
    path = tmp_path / "calib.json"
    payload = {"homography_matrix": matrix, **extra}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _real_calib_extra():
    return dict(
        source_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        destination_points_mm=[[0, 0], [100, 0], [100, 100], [0, 100]],
        reprojection_error_mm=0.5,
        image_width=640,
        image_height=480,
        created_at="2026-01-01T00:00:00Z",
    )


def test_evaluate_perfect_calibration_has_zero_error(tmp_path):
    calib = _write_calib(tmp_path, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], **_real_calib_extra())
    tf = CoordinateTransformer(calib, allow_identity=True)
    points = [
        test_calibration.CheckPoint(pixel=(10.0, 20.0), world_mm=(10.0, 20.0)),
        test_calibration.CheckPoint(pixel=(50.0, 60.0), world_mm=(50.0, 60.0)),
    ]
    errors, mean_error, max_error, rmse = test_calibration.evaluate(tf, points)
    assert mean_error < 1e-6
    assert max_error < 1e-6
    assert rmse < 1e-6


def test_evaluate_reports_nonzero_error_for_offset_calibration(tmp_path):
    calib = _write_calib(tmp_path, [[1, 0, 5], [0, 1, 5], [0, 0, 1]], **_real_calib_extra())
    tf = CoordinateTransformer(calib)
    points = [test_calibration.CheckPoint(pixel=(0.0, 0.0), world_mm=(0.0, 0.0))]
    errors, mean_error, max_error, rmse = test_calibration.evaluate(tf, points)
    # H chiếu (0,0) -> (5,5) nhưng expected (0,0) -> error = hypot(5,5)
    assert abs(mean_error - (5.0 * 2 ** 0.5)) < 1e-6


def test_load_points_file(tmp_path):
    points_file = tmp_path / "points.json"
    points_file.write_text(
        json.dumps({"points": [{"pixel": [1, 2], "world_mm": [3, 4]}]}), encoding="utf-8"
    )
    points = test_calibration.load_points_file(str(points_file))
    assert points == [test_calibration.CheckPoint(pixel=(1, 2), world_mm=(3, 4))]


def test_main_rejects_when_error_exceeds_threshold(tmp_path, monkeypatch):
    calib = _write_calib(tmp_path, [[1, 0, 50], [0, 1, 50], [0, 0, 1]], **_real_calib_extra())
    points_file = tmp_path / "points.json"
    points_file.write_text(
        json.dumps({"points": [{"pixel": [0, 0], "world_mm": [0, 0]}]}), encoding="utf-8"
    )

    argv = sys.argv
    sys.argv = [
        "test_calibration.py",
        "--calib",
        calib,
        "--points-file",
        str(points_file),
        "--max-error-mm",
        "3.0",
        "--settings",
        str(tmp_path / "nonexistent.yaml"),
    ]
    try:
        with pytest.raises(SystemExit) as exc_info:
            test_calibration.main()
        assert exc_info.value.code == 1
    finally:
        sys.argv = argv


def test_main_accepts_when_error_within_threshold(tmp_path, capsys):
    calib = _write_calib(tmp_path, [[1, 0, 5], [0, 1, 5], [0, 0, 1]], **_real_calib_extra())
    points_file = tmp_path / "points.json"
    points_file.write_text(
        json.dumps({"points": [{"pixel": [0, 0], "world_mm": [5, 5]}]}), encoding="utf-8"
    )

    argv = sys.argv
    sys.argv = [
        "test_calibration.py",
        "--calib",
        calib,
        "--points-file",
        str(points_file),
        "--max-error-mm",
        "3.0",
        "--settings",
        str(tmp_path / "nonexistent.yaml"),
    ]
    try:
        test_calibration.main()  # không raise SystemExit
    finally:
        sys.argv = argv
    assert "RESULT: OK" in capsys.readouterr().out
