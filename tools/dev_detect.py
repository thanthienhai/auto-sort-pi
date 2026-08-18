"""Dev-machine detection harness.

Chạy qua đúng logic decode/NMS trong ``modules.detector`` trên ảnh tĩnh hoặc webcam,
vẽ bbox và in tọa độ — để kiểm tra pipeline trước khi lên phần cứng.

Hai backend (tự chọn theo đuôi file, hoặc ép bằng ``--backend``):

- ``tflite`` — dùng nguyên ``WasteDetector`` (đúng code deploy lên Pi). Cần một
  TFLite runtime (tflite-runtime / ai-edge-litert / tensorflow) và model ``.tflite``.
- ``onnx``  — dùng ``onnxruntime`` cho phần inference (tiện trên Windows/dev, nơi
  không export được ``.tflite``) nhưng **tái sử dụng nguyên hàm ``_decode_outputs``**
  của ``WasteDetector`` để validate đúng phần logic của ta.

Ví dụ:
    python tools/dev_detect.py --model yolov8n.onnx --source samples/bus.jpg \\
        --labels coco --out samples/bus_out.jpg
    python tools/dev_detect.py --model models/yolov8n_waste_int8.tflite --source 0 \\
        --labels config/model_classes.json --calib config/calibration_matrix.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Cho phép chạy trực tiếp `python tools/dev_detect.py` từ repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Console Windows mặc định cp1252 không in được tiếng Việt -> ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from modules.detector import WasteDetector  # noqa: E402
from modules.transformer import CoordinateTransformer  # noqa: E402

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def make_decoder(conf: float, iou: float, num_classes: int | None, in_wh: int) -> WasteDetector:
    """Tạo WasteDetector chỉ để dùng phần decode (không load model tflite)."""
    det = WasteDetector.__new__(WasteDetector)
    det.confidence_threshold = conf
    det.nms_iou_threshold = iou
    det.num_classes = num_classes
    det.input_width = in_wh
    det.input_height = in_wh
    return det


class OnnxBackend:
    """Inference bằng onnxruntime + decode bằng chính logic của WasteDetector."""

    def __init__(self, model: str, conf: float, iou: float, num_classes: int | None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.inp = self.session.get_inputs()[0]
        shape = self.inp.shape  # thường [1, 3, H, W]
        self.ih = shape[2] if isinstance(shape[2], int) else 320
        self.iw = shape[3] if isinstance(shape[3], int) else 320
        self.decoder = make_decoder(conf, iou, num_classes, self.iw)

    def infer(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.iw, self.ih)).astype(np.float32) / 255.0
        tensor = np.transpose(img, (2, 0, 1))[None]  # NCHW
        outputs = self.session.run(None, {self.inp.name: tensor})
        return self.decoder._decode_outputs(outputs, w, h)


def load_labels(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    if spec == "coco":
        return COCO80
    return json.loads(Path(spec).read_text(encoding="utf-8"))


def read_source(source: str) -> np.ndarray:
    try:
        cam_index = int(source)
    except ValueError:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Không đọc được ảnh: {source}")
        return frame

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera index {cam_index}")
    frame = None
    for _ in range(10):  # bỏ vài frame đầu để camera ổn định phơi sáng
        ok, frame = cap.read()
        time.sleep(0.03)
    cap.release()
    if frame is None:
        raise SystemExit("Camera không trả về frame")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Dev detection harness for auto-sort-pi")
    ap.add_argument("--model", required=True, help="đường dẫn .tflite hoặc .onnx")
    ap.add_argument("--source", required=True, help="đường dẫn ảnh hoặc index camera")
    ap.add_argument("--backend", choices=["auto", "tflite", "onnx"], default="auto")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--num-classes", default="auto", help="'auto' hoặc số nguyên")
    ap.add_argument("--labels", default=None, help="'coco' hoặc file json list tên lớp")
    ap.add_argument("--calib", default=None, help="calibration_matrix.json (tuỳ chọn)")
    ap.add_argument("--out", default="samples/out.jpg")
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "onnx" if args.model.lower().endswith(".onnx") else "tflite"

    num_classes = None if args.num_classes == "auto" else int(args.num_classes)
    if backend == "onnx":
        detector: object = OnnxBackend(args.model, args.conf, args.iou, num_classes)
    else:
        detector = WasteDetector(
            args.model,
            confidence_threshold=args.conf,
            nms_iou_threshold=args.iou,
            num_classes=num_classes,
        )

    labels = load_labels(args.labels)
    transformer = CoordinateTransformer(args.calib) if args.calib else None

    frame = read_source(args.source)
    h, w = frame.shape[:2]

    started = time.perf_counter()
    detections = detector.infer(frame)
    latency = (time.perf_counter() - started) * 1000.0
    print(f"[{backend}] Ảnh {w}x{h} | infer {latency:.1f} ms | {len(detections)} detections")

    for d in sorted(detections, key=lambda x: -x["conf"]):
        cid = d["class_id"]
        name = labels[cid] if labels and 0 <= cid < len(labels) else str(cid)
        cx, cy = d["center"]
        line = f"  {name:16s} conf={d['conf']:.2f} center=({cx:.0f},{cy:.0f})"
        if transformer is not None:
            xw, yw = transformer.pixel_to_world(cx, cy)
            line += f" -> world=({xw},{yw})mm"
        print(line)

        x, y, bw, bh = (int(v) for v in d["box"])
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        cv2.putText(
            frame,
            f"{name} {d['conf']:.2f}",
            (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, frame)
    print(f"Đã lưu ảnh kết quả: {args.out}")


if __name__ == "__main__":
    main()
