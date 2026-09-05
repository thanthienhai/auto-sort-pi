"""Runtime health check CLI (§18 task.md).

Hai loại kiểm tra:

1. **Pre-flight / standalone** (không cần ``main.py`` đang chạy): thử mở
   camera, validate model (tái dùng ``tools/validate_model.py``), validate
   calibration, thử mở cổng UART. Dùng để chẩn đoán trước khi khởi động
   pipeline, hoặc khi pipeline đang KHÔNG chạy.
2. **Live snapshot**: nếu ``logs/health_snapshot.json`` tồn tại (được
   ``main.py`` ghi định kỳ khi đang chạy — xem ``main.HealthStatus``), đọc và
   hiển thị ``last_frame`` / ``last_inference`` / ``last_pick`` / ``last_error``
   / trạng thái FSM hiện tại.

Chạy:
    python tools/health.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from modules.exceptions import CalibrationError  # noqa: E402
from modules.transformer import CoordinateTransformer  # noqa: E402
from tools import validate_model  # noqa: E402


def check_camera(source, width: int, height: int) -> tuple[bool, str]:
    try:
        import cv2

        cap = cv2.VideoCapture(source)
        try:
            if not cap.isOpened():
                return False, f"không mở được camera source={source!r}"
            ok, _ = cap.read()
            if not ok:
                return False, "mở được camera nhưng không đọc được frame"
            return True, f"OK ({width}x{height})"
        finally:
            cap.release()
    except Exception as exc:  # noqa: BLE001
        return False, f"lỗi khi kiểm tra camera: {exc}"


def check_model(settings: dict) -> tuple[bool, str]:
    infer_cfg = settings["inference"]
    report = validate_model.validate(
        infer_cfg["model_path"],
        "config/class_mapping.json",
        "config/model_classes.json",
        infer_cfg["confidence_threshold"],
        infer_cfg["nms_iou_threshold"],
    )
    if report.ok:
        return True, "OK"
    failed = [name for name, ok, _ in report.checks if not ok]
    return False, f"FAILED: {', '.join(failed)}"


def check_calibration(settings: dict) -> tuple[bool, str]:
    calib_cfg = settings["calibration"]
    try:
        tf = CoordinateTransformer(
            calib_cfg["file"],
            max_reprojection_error_mm=calib_cfg["max_reprojection_error_mm"],
            allow_identity=False,
        )
        return True, f"OK (reprojection_error={tf.reprojection_error_mm}mm)"
    except CalibrationError as exc:
        return False, str(exc)


def check_uart(settings: dict) -> tuple[bool, str]:
    uart_cfg = settings["uart"]
    try:
        import serial

        ser = serial.Serial(port=uart_cfg["port"], baudrate=uart_cfg["baudrate"], timeout=0.2)
        ser.close()
        return True, f"OK ({uart_cfg['port']} @ {uart_cfg['baudrate']}bps)"
    except ImportError:
        return False, "pyserial chưa cài"
    except Exception as exc:  # noqa: BLE001
        return False, f"không mở được {uart_cfg['port']}: {exc}"


def read_live_snapshot(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _age(ts: float | None) -> str:
    if ts is None:
        return "chưa có"
    return f"{time.time() - ts:.1f}s trước"


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto Sort Pi health check")
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--snapshot", default="logs/health_snapshot.json")
    ap.add_argument("--skip-camera", action="store_true", help="bỏ qua kiểm tra camera thật")
    ap.add_argument("--skip-uart", action="store_true", help="bỏ qua kiểm tra UART thật")
    args = ap.parse_args()

    settings = yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))
    camera_cfg = settings["camera"]

    results: list[tuple[str, bool, str]] = []

    if args.skip_camera:
        results.append(("Camera", True, "bỏ qua (--skip-camera)"))
    else:
        results.append(("Camera", *check_camera(camera_cfg["source"], camera_cfg["width"], camera_cfg["height"])))

    results.append(("Model", *check_model(settings)))
    results.append(("Calibration", *check_calibration(settings)))

    if args.skip_uart:
        results.append(("UART", True, "bỏ qua (--skip-uart)"))
    else:
        results.append(("UART", *check_uart(settings)))

    print("Auto Sort Pi — Health Check")
    print("=" * 40)
    for name, ok, detail in results:
        print(f"{name:12s}: {'OK' if ok else 'FAIL'}  ({detail})")

    snapshot = read_live_snapshot(Path(args.snapshot))
    print("-" * 40)
    if snapshot is None:
        print("Live status: không tìm thấy health snapshot — pipeline có thể chưa chạy.")
    else:
        print(f"FSM state        : {snapshot.get('state')}")
        print(f"Last frame       : {_age(snapshot.get('last_frame_at'))}")
        print(f"Last inference   : {_age(snapshot.get('last_inference_at'))}")
        print(f"Last successful pick : {_age(snapshot.get('last_pick_at'))}")
        print(f"Last error       : {snapshot.get('last_error') or 'none'}")
    print("=" * 40)

    system_ready = all(ok for _, ok, _ in results)
    print(f"System: {'READY' if system_ready else 'NOT READY'}")
    sys.exit(0 if system_ready else 1)


if __name__ == "__main__":
    main()
