"""Validate một calibration đã lưu bằng các điểm kiểm tra độc lập (§8 task.md).

Cho phép click một số điểm trên bàn (hoặc nạp từ file), nhập toạ độ world thật
đo bằng thước, rồi so sánh với kết quả ``pixel_to_world`` để tính mean/max/RMSE
error. Reject (exit code 1) nếu vượt ``calibration.max_reprojection_error_mm``
trong ``config/settings.yaml``.

Chạy:
    python tools/test_calibration.py --image samples/waste_topdown2.jpg
    python tools/test_calibration.py --points-file check_points.json  # non-interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from modules.transformer import CoordinateTransformer  # noqa: E402


class CheckPoint(NamedTuple):
    pixel: tuple[float, float]
    world_mm: tuple[float, float]


def collect_points_interactive(image_path: str) -> list[CheckPoint]:
    import cv2

    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Không đọc được ảnh: {image_path}")

    picked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):  # noqa: D401 - cv2 callback signature
        if event == cv2.EVENT_LBUTTONDOWN:
            picked.append((x, y))

    window = "Auto Sort Pi - Calibration check (click points, ENTER to stop)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    print("Click các điểm cần kiểm tra trên bàn, nhấn ENTER khi xong.")
    while True:
        display = frame.copy()
        for i, (x, y) in enumerate(picked):
            cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(display, str(i + 1), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27:  # ESC
            cv2.destroyWindow(window)
            raise SystemExit("Huỷ bởi người dùng.")
    cv2.destroyWindow(window)

    points: list[CheckPoint] = []
    for i, (x, y) in enumerate(picked):
        raw = input(f"  World mm thật của điểm {i + 1} tại pixel ({x},{y}) [X,Y]: ").strip()
        wx_str, wy_str = raw.split(",")
        points.append(CheckPoint(pixel=(x, y), world_mm=(float(wx_str), float(wy_str))))
    return points


def load_points_file(path: str) -> list[CheckPoint]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CheckPoint(pixel=tuple(p["pixel"]), world_mm=tuple(p["world_mm"])) for p in data["points"]]


def evaluate(
    transformer: CoordinateTransformer, points: list[CheckPoint]
) -> tuple[list[float], float, float, float]:
    errors = []
    for point in points:
        xw, yw = transformer.pixel_to_world(*point.pixel)
        err = float(np.hypot(xw - point.world_mm[0], yw - point.world_mm[1]))
        errors.append(err)
    mean_error = float(np.mean(errors)) if errors else 0.0
    max_error = float(np.max(errors)) if errors else 0.0
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
    return errors, mean_error, max_error, rmse


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a saved homography calibration")
    ap.add_argument("--calib", default="config/calibration_matrix.json")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--image", default=None, help="ảnh để click điểm kiểm tra (interactive)")
    ap.add_argument("--points-file", default=None, help="JSON {\"points\": [{pixel, world_mm}, ...]}")
    ap.add_argument("--max-error-mm", type=float, default=None, help="Override ngưỡng cấu hình")
    args = ap.parse_args()

    settings = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8")) if Path(args.settings).is_file() else {}
    threshold = args.max_error_mm
    if threshold is None:
        threshold = float(
            settings.get("calibration", {}).get("max_reprojection_error_mm", 3.0)
        )

    transformer = CoordinateTransformer(args.calib, max_reprojection_error_mm=threshold)

    if args.points_file:
        points = load_points_file(args.points_file)
    elif args.image:
        points = collect_points_interactive(args.image)
    else:
        raise SystemExit("Cần --points-file hoặc --image để lấy điểm kiểm tra.")

    if not points:
        raise SystemExit("Không có điểm kiểm tra nào.")

    errors, mean_error, max_error, rmse = evaluate(transformer, points)

    print("Calibration check")
    print("=" * 40)
    for i, (point, err) in enumerate(zip(points, errors)):
        xw, yw = transformer.pixel_to_world(*point.pixel)
        print(
            f"  point {i + 1}: pixel={point.pixel} -> world=({xw},{yw}) "
            f"expected={point.world_mm} error={err:.2f}mm"
        )
    print("-" * 40)
    print(f"mean error : {mean_error:.3f} mm")
    print(f"max error  : {max_error:.3f} mm")
    print(f"RMSE       : {rmse:.3f} mm")
    print(f"threshold  : {threshold:.3f} mm")
    print("=" * 40)

    if max_error > threshold:
        print("RESULT: REJECTED — sai số vượt ngưỡng cho phép.")
        sys.exit(1)
    print("RESULT: OK — calibration đạt yêu cầu.")


if __name__ == "__main__":
    main()
