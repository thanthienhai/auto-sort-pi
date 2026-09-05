"""Pixel-to-world coordinate transformation using a calibrated homography.

Ma trận Homography ``H`` được calibration offline (``tools/calibrate.py``,
``cv2.findHomography`` từ 4 điểm mốc) và lưu trong
``config/calibration_matrix.json``. Ở runtime, việc chiếu một điểm chỉ là một
phép nhân ma trận 3×3, nên module này dùng numpy thuần — không cần OpenCV
trong hot path, và unit-test được mà không phụ thuộc phần cứng.

§7 task.md: **không được** coi identity matrix là calibration thật. Runtime
phải reject file placeholder (không có ``source_points``/``destination_points_mm``
hoặc reprojection error vượt ngưỡng) trước khi hệ thống được vào ``READY``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .exceptions import CalibrationError


class CoordinateTransformer:
    def __init__(
        self,
        calib_file: str = "config/calibration_matrix.json",
        max_reprojection_error_mm: float | None = None,
        allow_identity: bool = False,
    ) -> None:
        """Load + validate calibration.

        ``allow_identity=True`` chỉ nên dùng trong dev/test khi chưa calibration
        phần cứng thật (vd tools/dev_detect.py mà không truyền ``--calib``); mặc
        định là ``False`` để main.py không thể lỡ chạy production với identity.
        """
        path = Path(calib_file)
        if not path.is_file():
            raise CalibrationError(f"Calibration file không tồn tại: {calib_file}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"Calibration file '{calib_file}' không phải JSON hợp lệ: {exc}") from exc

        matrix = np.asarray(data.get("homography_matrix"), dtype=np.float64)
        if matrix.shape != (3, 3):
            raise CalibrationError(f"Homography matrix must be 3x3, got {matrix.shape}")

        source_points = data.get("source_points") or []
        destination_points_mm = data.get("destination_points_mm") or []
        reprojection_error_mm = data.get("reprojection_error_mm")

        is_identity = np.allclose(matrix, np.eye(3))
        if is_identity and not allow_identity:
            raise CalibrationError(
                f"Calibration file '{calib_file}' là identity matrix — chưa calibration "
                "thật. Chạy `python tools/calibrate.py` trước khi vận hành."
            )
        if not allow_identity:
            if len(source_points) < 4 or len(destination_points_mm) < 4:
                raise CalibrationError(
                    f"Calibration file '{calib_file}' thiếu source_points/"
                    "destination_points_mm (cần >= 4 điểm mốc). Chạy "
                    "`python tools/calibrate.py`."
                )
            if reprojection_error_mm is None:
                raise CalibrationError(
                    f"Calibration file '{calib_file}' thiếu 'reprojection_error_mm'; "
                    "không thể xác nhận độ chính xác calibration."
                )
            if max_reprojection_error_mm is not None and reprojection_error_mm > max_reprojection_error_mm:
                raise CalibrationError(
                    f"Reprojection error {reprojection_error_mm}mm vượt ngưỡng cho phép "
                    f"{max_reprojection_error_mm}mm. Calibration lại bàn thao tác."
                )

        self.H = matrix
        self.source_points = source_points
        self.destination_points_mm = destination_points_mm
        self.reprojection_error_mm = reprojection_error_mm
        self.image_width = data.get("image_width")
        self.image_height = data.get("image_height")
        self.created_at = data.get("created_at")

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        """Chuyển tâm bounding box ``(u, v)`` pixel sang tọa độ thực ``(Xw, Yw)`` mm.

        Tương đương ``cv2.perspectiveTransform`` cho một điểm:
        ``[x', y', w']ᵀ = H · [u, v, 1]ᵀ`` rồi chia cho ``w'``.
        """
        vec = np.array([float(u), float(v), 1.0], dtype=np.float64)
        x_p, y_p, w_p = self.H @ vec
        if abs(w_p) < 1e-9:
            raise CalibrationError(
                f"Degenerate homography projection at pixel ({u}, {v}): w'≈0"
            )
        x_w = round(float(x_p / w_p), 1)
        y_w = round(float(y_p / w_p), 1)
        return x_w, y_w
