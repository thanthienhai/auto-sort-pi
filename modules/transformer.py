"""Pixel-to-world coordinate transformation using a calibrated homography.

Ma trận Homography ``H`` được calibration offline (``cv2.findHomography`` từ 4 điểm
mốc) và lưu trong ``config/calibration_matrix.json``. Ở runtime, việc chiếu một điểm
chỉ là một phép nhân ma trận 3×3, nên module này dùng numpy thuần — không cần OpenCV
trong hot path, và unit-test được mà không phụ thuộc phần cứng.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CoordinateTransformer:
    def __init__(self, calib_file: str = "config/calibration_matrix.json") -> None:
        data = json.loads(Path(calib_file).read_text(encoding="utf-8"))
        matrix = np.asarray(data["homography_matrix"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {matrix.shape}")
        self.H = matrix

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        """Chuyển tâm bounding box ``(u, v)`` pixel sang tọa độ thực ``(Xw, Yw)`` mm.

        Tương đương ``cv2.perspectiveTransform`` cho một điểm:
        ``[x', y', w']ᵀ = H · [u, v, 1]ᵀ`` rồi chia cho ``w'``.
        """
        vec = np.array([float(u), float(v), 1.0], dtype=np.float64)
        x_p, y_p, w_p = self.H @ vec
        if abs(w_p) < 1e-9:
            raise ValueError(
                f"Degenerate homography projection at pixel ({u}, {v}): w'≈0"
            )
        x_w = round(float(x_p / w_p), 1)
        y_w = round(float(y_p / w_p), 1)
        return x_w, y_w
