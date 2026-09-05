"""Kiểm thử tools/health.py: các phần không cần phần cứng thật (§18 task.md)."""

from __future__ import annotations

import json
import time

import tools.health as health


def _write_calib(tmp_path, matrix, **extra):
    path = tmp_path / "calib.json"
    payload = {"homography_matrix": matrix, **extra}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_check_calibration_ok_for_real_calibration(tmp_path):
    calib = _write_calib(
        tmp_path,
        [[2, 0, 10], [0, 3, -5], [0, 0, 1]],
        source_points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        destination_points_mm=[[0, 0], [100, 0], [100, 100], [0, 100]],
        reprojection_error_mm=0.5,
    )
    settings = {"calibration": {"file": calib, "max_reprojection_error_mm": 3.0}}
    ok, detail = health.check_calibration(settings)
    assert ok is True
    assert "OK" in detail


def test_check_calibration_fails_for_identity_placeholder(tmp_path):
    calib = _write_calib(tmp_path, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    settings = {"calibration": {"file": calib, "max_reprojection_error_mm": 3.0}}
    ok, detail = health.check_calibration(settings)
    assert ok is False
    assert "identity" in detail


def test_read_live_snapshot_missing_file_returns_none(tmp_path):
    assert health.read_live_snapshot(tmp_path / "missing.json") is None


def test_read_live_snapshot_reads_valid_json(tmp_path):
    snap_path = tmp_path / "snapshot.json"
    snap_path.write_text(json.dumps({"state": "SCANNING", "last_error": None}), encoding="utf-8")
    data = health.read_live_snapshot(snap_path)
    assert data == {"state": "SCANNING", "last_error": None}


def test_read_live_snapshot_handles_corrupt_json(tmp_path):
    snap_path = tmp_path / "bad.json"
    snap_path.write_text("{not json", encoding="utf-8")
    assert health.read_live_snapshot(snap_path) is None


def test_age_formats_missing_and_present_timestamps():
    assert health._age(None) == "chưa có"
    formatted = health._age(time.time() - 5)
    assert formatted.endswith("s trước")
