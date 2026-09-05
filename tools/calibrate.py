"""Interactive homography calibration tool (§7 task.md).

Quy trình:

1. Mở camera (hoặc dùng ``--image`` cho một ảnh tĩnh đã chụp sẵn — hữu ích khi
   test logic mà không có camera thật gắn vào máy).
2. Hiển thị frame, cho phép click chọn 4 điểm pixel trên bàn (thứ tự tuỳ ý,
   miễn nhất quán với thứ tự nhập toạ độ world ở bước sau).
3. Nhập 4 điểm world tương ứng (mm) qua CLI.
4. Tính Homography bằng ``cv2.findHomography``.
5. Tính reprojection error (mm) của chính 4 điểm mốc.
6. Hiển thị kết quả (ma trận + sai số).
7. Yêu cầu xác nhận trước khi ghi đè.
8. Lưu ``config/calibration_matrix.json`` đúng schema task.md.

Chạy:
    python tools/calibrate.py --source 0
    python tools/calibrate.py --image samples/waste_topdown2.jpg   # không cần camera thật
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WINDOW_NAME = "Auto Sort Pi - Calibration (click 4 points, ESC to cancel)"


class PointPicker:
    def __init__(self, image) -> None:
        self.image = image
        self.points: list[tuple[int, int]] = []

    def _on_mouse(self, event, x, y, flags, param) -> None:  # noqa: D401 - cv2 callback signature
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def run(self) -> list[tuple[int, int]]:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        while len(self.points) < 4:
            display = self.image.copy()
            for i, (x, y) in enumerate(self.points):
                cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(
                display,
                f"Click point {len(self.points) + 1}/4",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(WINDOW_NAME)
                raise SystemExit("Calibration cancelled by user (ESC).")
        cv2.destroyWindow(WINDOW_NAME)
        return self.points


def capture_source(source: str, image_path: str | None) -> np.ndarray:
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise SystemExit(f"Không đọc được ảnh: {image_path}")
        return frame

    try:
        cam_index: int | str = int(source)
    except ValueError:
        cam_index = source
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera: {source}")
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    if frame is None:
        raise SystemExit("Camera không trả về frame nào.")
    return frame


def prompt_world_points() -> list[tuple[float, float]]:
    print("Nhập 4 điểm world tương ứng (mm), theo đúng thứ tự đã click:")
    points = []
    for i in range(4):
        while True:
            raw = input(f"  Điểm {i + 1} (X,Y mm): ").strip()
            try:
                x_str, y_str = raw.split(",")
                points.append((float(x_str), float(y_str)))
                break
            except ValueError:
                print("    Định dạng sai, ví dụ hợp lệ: 150.0,320.0")
    return points


def compute_homography(
    source_points: list[tuple[float, float]],
    destination_points_mm: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    src = np.array(source_points, dtype=np.float64)
    dst = np.array(destination_points_mm, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise SystemExit("cv2.findHomography thất bại — 4 điểm có thể thẳng hàng.")

    # Reprojection error: chiếu lại 4 điểm pixel qua H và so với world thật.
    errors = []
    for (u, v), (xw, yw) in zip(source_points, destination_points_mm):
        vec = H @ np.array([u, v, 1.0])
        x_p, y_p, w_p = vec
        x_proj, y_proj = x_p / w_p, y_p / w_p
        errors.append(float(np.hypot(x_proj - xw, y_proj - yw)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return H, rmse


def save_calibration(
    out_path: str,
    H: np.ndarray,
    source_points: list[tuple[float, float]],
    destination_points_mm: list[tuple[float, float]],
    reprojection_error_mm: float,
    image_width: int,
    image_height: int,
) -> None:
    payload = {
        "homography_matrix": H.tolist(),
        "source_points": [list(p) for p in source_points],
        "destination_points_mm": [list(p) for p in destination_points_mm],
        "reprojection_error_mm": round(reprojection_error_mm, 3),
        "image_width": image_width,
        "image_height": image_height,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate pixel->world homography")
    ap.add_argument("--source", default="0", help="camera index hoặc device path")
    ap.add_argument("--image", default=None, help="dùng ảnh tĩnh thay vì mở camera")
    ap.add_argument("--out", default="config/calibration_matrix.json")
    ap.add_argument(
        "--max-reprojection-error-mm",
        type=float,
        default=3.0,
        help="Cảnh báo (không tự chặn) nếu RMSE vượt ngưỡng này",
    )
    ap.add_argument(
        "--points-file",
        default=None,
        help="JSON {\"source_points\": [...], \"destination_points_mm\": [...]}"
        " để chạy non-interactive (test / scripting) thay vì click chuột + input().",
    )
    args = ap.parse_args()

    frame = capture_source(args.source, args.image)
    h, w = frame.shape[:2]

    if args.points_file:
        data = json.loads(Path(args.points_file).read_text(encoding="utf-8"))
        source_points = [tuple(p) for p in data["source_points"]]
        destination_points_mm = [tuple(p) for p in data["destination_points_mm"]]
    else:
        source_points = PointPicker(frame).run()
        destination_points_mm = prompt_world_points()

    H, rmse = compute_homography(source_points, destination_points_mm)

    print("\nHomography matrix:")
    for row in H:
        print("  " + "  ".join(f"{v: .6f}" for v in row))
    print(f"\nReprojection RMSE: {rmse:.3f} mm (threshold: {args.max_reprojection_error_mm} mm)")
    if rmse > args.max_reprojection_error_mm:
        print("CẢNH BÁO: sai số vượt ngưỡng — cân nhắc chọn lại điểm mốc chính xác hơn.")

    if not args.points_file:
        answer = input("\nGhi đè config/calibration_matrix.json? [y/N]: ").strip().lower()
        if answer != "y":
            print("Huỷ — không ghi file.")
            return

    save_calibration(
        args.out, H, source_points, destination_points_mm, rmse, image_width=w, image_height=h
    )
    print(f"Đã lưu calibration: {args.out}")


if __name__ == "__main__":
    main()
