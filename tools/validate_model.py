"""Validate một model TFLite trước khi deploy (§6 task.md).

Kiểm tra: file tồn tại, load được, input shape/dtype, output shapes, số class,
class mapping (mọi class trong ``model_classes.json`` có trong
``class_mapping.json``), khoảng confidence/NMS cấu hình, và decoder có
compatible với output layout hay không (chạy thử một forward pass giả gồm
toàn zero để chắc chắn ``_decode_outputs`` không raise).

Không sửa runtime; ``main.py`` gọi lại đúng logic validate này (qua
``WasteDetector.__init__``) trước khi vào ``READY`` — tool này chỉ để chạy
độc lập, in report dễ đọc, dùng trong CI/CD hoặc trước khi flash một model mới.

Ví dụ:
    python tools/validate_model.py --model models/yolov8n_waste_int8.tflite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from modules.detector import WasteDetector  # noqa: E402
from modules.exceptions import ModelError  # noqa: E402


class ValidationReport:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def render(self) -> str:
        lines = ["Model validation report", "=" * 40]
        for name, ok, detail in self.checks:
            status = "PASS" if ok else "FAIL"
            line = f"[{status}] {name}"
            if detail:
                line += f" — {detail}"
            lines.append(line)
        lines.append("=" * 40)
        lines.append("RESULT: " + ("OK — model hợp lệ để deploy" if self.ok else "FAILED — không được deploy"))
        return "\n".join(lines)


def validate(
    model_path: str,
    class_mapping_path: str,
    model_classes_path: str,
    confidence_threshold: float,
    nms_iou_threshold: float,
) -> ValidationReport:
    report = ValidationReport()

    model_file = Path(model_path)
    report.add("model file exists", model_file.is_file(), model_path)
    if not model_file.is_file():
        return report

    class_names: list[str] = []
    try:
        class_names = json.loads(Path(model_classes_path).read_text(encoding="utf-8"))
        report.add("model_classes.json readable", True, f"{len(class_names)} classes")
    except Exception as exc:  # noqa: BLE001
        report.add("model_classes.json readable", False, str(exc))

    mapping: dict[str, int] = {}
    try:
        mapping = json.loads(Path(class_mapping_path).read_text(encoding="utf-8"))
        report.add("class_mapping.json readable", True, f"{len(mapping)} entries")
    except Exception as exc:  # noqa: BLE001
        report.add("class_mapping.json readable", False, str(exc))

    if class_names and mapping:
        missing = [name for name in class_names if name not in mapping]
        report.add(
            "every model class has a group mapping",
            not missing,
            f"missing: {missing}" if missing else f"{len(class_names)}/{len(class_names)} mapped",
        )

    report.add(
        "confidence_threshold in [0,1]",
        0.0 <= confidence_threshold <= 1.0,
        str(confidence_threshold),
    )
    report.add(
        "nms_iou_threshold in [0,1]",
        0.0 <= nms_iou_threshold <= 1.0,
        str(nms_iou_threshold),
    )

    try:
        detector = WasteDetector(
            model_path,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            num_classes=len(class_names) or None,
        )
        report.add("model loads (input/output shape valid)", True)
    except ModelError as exc:
        report.add("model loads (input/output shape valid)", False, str(exc))
        return report
    except Exception as exc:  # noqa: BLE001
        report.add("model loads (input/output shape valid)", False, f"unexpected error: {exc}")
        return report

    report.add(
        "input dtype",
        True,
        str(detector.input_details[0]["dtype"]),
    )
    report.add(
        "input size",
        True,
        f"{detector.input_width}x{detector.input_height}",
    )
    for i, out in enumerate(detector.output_details):
        report.add(f"output[{i}] shape", True, str(tuple(out["shape"])))

    # Decoder compatibility: chạy thử decode một output toàn-zero (mọi score = 0,
    # dưới ngưỡng) để chắc chắn không raise ModelDecodingError với layout này.
    try:
        fake_outputs = [np.zeros(tuple(out["shape"]), dtype=np.float32) for out in detector.output_details]
        detections = detector._decode_outputs(fake_outputs, image_w=detector.input_width, image_h=detector.input_height)
        report.add("decoder compatible with output layout", True, f"{len(detections)} dets on zero input")
    except Exception as exc:  # noqa: BLE001
        report.add("decoder compatible with output layout", False, str(exc))

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a TFLite model before deployment")
    ap.add_argument("--model", required=True)
    ap.add_argument("--class-mapping", default="config/class_mapping.json")
    ap.add_argument("--model-classes", default="config/model_classes.json")
    ap.add_argument("--confidence-threshold", type=float, default=0.50)
    ap.add_argument("--nms-iou-threshold", type=float, default=0.45)
    args = ap.parse_args()

    report = validate(
        args.model,
        args.class_mapping,
        args.model_classes,
        args.confidence_threshold,
        args.nms_iou_threshold,
    )
    print(report.render())
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
