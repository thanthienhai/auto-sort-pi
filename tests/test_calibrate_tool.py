"""Kiểm thử tools/calibrate.py: tính homography, reprojection error, lưu file.

Chạy non-interactive qua ``--points-file`` để không cần GUI/click chuột hay
``input()`` — pipeline toán học (``cv2.findHomography`` + tính RMSE) mới là
phần cần test, không phải UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.calibrate as calibrate  # noqa: E402


def test_compute_homography_perfect_square_zero_error():
    source_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    destination_points_mm = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
    H, rmse = calibrate.compute_homography(source_points, destination_points_mm)
    assert rmse < 1e-6
    # Scale factor 2x, kiểm tra một điểm không phải mốc.
    vec = H @ np.array([50.0, 50.0, 1.0])
    x, y = vec[0] / vec[2], vec[1] / vec[2]
    assert abs(x - 100.0) < 1e-6 and abs(y - 100.0) < 1e-6


def test_save_and_reload_calibration(tmp_path):
    source_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    destination_points_mm = [(0.0, 0.0), (150.0, 0.0), (150.0, 150.0), (0.0, 150.0)]
    H, rmse = calibrate.compute_homography(source_points, destination_points_mm)
    out = tmp_path / "calib.json"
    calibrate.save_calibration(
        str(out), H, source_points, destination_points_mm, rmse, image_width=640, image_height=480
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["homography_matrix"]) == 3
    assert len(data["source_points"]) == 4
    assert data["image_width"] == 640
    assert "created_at" in data

    from modules.transformer import CoordinateTransformer

    tf = CoordinateTransformer(str(out))
    x, y = tf.pixel_to_world(100.0, 100.0)
    assert abs(x - 150.0) < 0.5 and abs(y - 150.0) < 0.5


def test_main_end_to_end_with_points_file(tmp_path):
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((480, 640, 3), dtype=np.uint8))

    points_file = tmp_path / "points.json"
    points_file.write_text(
        json.dumps(
            {
                "source_points": [[10, 10], [600, 10], [600, 460], [10, 460]],
                "destination_points_mm": [[0, 0], [300, 0], [300, 200], [0, 200]],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "out_calib.json"

    argv = sys.argv
    sys.argv = [
        "calibrate.py",
        "--image",
        str(image_path),
        "--points-file",
        str(points_file),
        "--out",
        str(out_path),
    ]
    try:
        calibrate.main()
    finally:
        sys.argv = argv

    assert out_path.is_file()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["reprojection_error_mm"] < 1.0


def test_collinear_points_raise_system_exit():
    source_points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    destination_points_mm = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0), (300.0, 0.0)]
    with pytest.raises(SystemExit):
        calibrate.compute_homography(source_points, destination_points_mm)
