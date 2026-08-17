"""Pixel-to-world coordinate transformation using a calibrated homography."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class CoordinateTransformer:
    def __init__(self, calib_file: str = "config/calibration_matrix.json") -> None:
        data = json.loads(Path(calib_file).read_text(encoding="utf-8"))
        matrix = np.asarray(data["homography_matrix"], dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {matrix.shape}")
        self.H = matrix

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        pixel_pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pixel_pt, self.H)
        x_w = round(float(world_pt[0, 0, 0]), 1)
        y_w = round(float(world_pt[0, 0, 1]), 1)
        return x_w, y_w
