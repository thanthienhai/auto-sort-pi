"""TensorFlow Lite waste detector adapter.

Hỗ trợ hai layout output phổ biến và tự nhận diện:

- **YOLOv8 / YOLOv11 raw** ``[1, 4+num_classes, num_anchors]`` (export Ultralytics
  mặc định, ``nms=False``): mỗi anchor gồm ``(cx, cy, w, h)`` + điểm số từng lớp,
  không có ô objectness riêng. Cần transpose, argmax lớp, đổi ``xywh → xyxy`` rồi NMS
  ngay trên Pi.
- **NMS-embedded** ``[N, 6]`` = ``(x1, y1, x2, y2, score, class_id)`` (export kèm
  ``TFLite_Detection_PostProcess``).

Phần giải mã + NMS dùng numpy thuần nên unit-test được mà không cần OpenCV; OpenCV
chỉ dùng cho tiền xử lý ảnh (resize / đổi màu) trong ``infer``.

§5/§6 task.md: model không tương thích (input shape/dtype, output shape, số class)
phải ``raise`` lỗi tường minh (``ModelLoadError``) ngay khi khởi tạo, không được
"silently accept" rồi decode sai ở runtime. Lỗi loading / inference / decoding
được phân biệt bằng ba exception riêng.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import ModelDecodingError, ModelInferenceError, ModelLoadError
from .logging_setup import get_logger

logger = get_logger("autosort.detector")

# Các dtype input mà pipeline tiền xử lý (float [0,1] hoặc uint8 quantized) hỗ trợ.
SUPPORTED_INPUT_DTYPES = (np.float32, np.uint8, np.int8)


def _load_interpreter():
    """Import TFLite Interpreter lazily từ runtime khả dụng.

    Thứ tự thử: ``tflite-runtime`` (Raspberry Pi) → ``ai-edge-litert`` (runtime kế
    nhiệm, có wheel cho Windows/macOS nên tiện test trên máy dev) → ``tensorflow``.
    """
    errors = []
    for module_path in (
        "tflite_runtime.interpreter",
        "ai_edge_litert.interpreter",
        "tensorflow.lite.python.interpreter",
    ):
        try:
            module = __import__(module_path, fromlist=["Interpreter"])
            return module.Interpreter
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            errors.append(f"{module_path}: {exc}")
    raise ModelLoadError(
        "Không tìm thấy TFLite runtime. Cài một trong: tflite-runtime (Pi), "
        "ai-edge-litert (dev), hoặc tensorflow.\n" + "\n".join(errors)
    )


class WasteDetector:
    """Run a TFLite object detector and return normalized detection dictionaries."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.50,
        nms_iou_threshold: float = 0.45,
        num_classes: int | None = None,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ModelLoadError(
                f"confidence_threshold phải trong [0,1], nhận {confidence_threshold}"
            )
        if not (0.0 <= nms_iou_threshold <= 1.0):
            raise ModelLoadError(
                f"nms_iou_threshold phải trong [0,1], nhận {nms_iou_threshold}"
            )
        if not Path(model_path).is_file():
            raise ModelLoadError(f"Model file không tồn tại: {model_path}")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.num_classes = num_classes

        Interpreter = _load_interpreter()
        try:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - runtime lỗi khi parse file model
            raise ModelLoadError(f"Không load được model '{model_path}': {exc}") from exc

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self._validate_input_spec()
        self._validate_output_spec()

        _, self.input_height, self.input_width, _ = self.input_details[0]["shape"]

    # ------------------------------------------------------------ validation

    def _validate_input_spec(self) -> None:
        if not self.input_details:
            raise ModelLoadError(f"Model '{self.model_path}' không có input tensor nào.")
        detail = self.input_details[0]
        shape = tuple(int(v) for v in detail["shape"])
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ModelLoadError(
                f"Input shape không hợp lệ {shape}; kỳ vọng [1, H, W, 3] "
                f"(model: '{self.model_path}')."
            )
        if detail["dtype"] not in SUPPORTED_INPUT_DTYPES:
            raise ModelLoadError(
                f"Input dtype không hỗ trợ: {detail['dtype']}; kỳ vọng một trong "
                f"{[dt.__name__ for dt in SUPPORTED_INPUT_DTYPES]} "
                f"(model: '{self.model_path}')."
            )

    def _validate_output_spec(self) -> None:
        if not self.output_details:
            raise ModelLoadError(f"Model '{self.model_path}' không có output tensor nào.")
        for detail in self.output_details:
            shape = tuple(int(v) for v in detail["shape"])
            # Layout hợp lệ: [1, 4+num_classes, N] / [1, N, 6] / [N, 6] (batch tuỳ chọn).
            dims = [d for d in shape if d not in (1,)] or list(shape)
            if len(shape) not in (2, 3):
                raise ModelLoadError(
                    f"Output shape không hợp lệ {shape}; kỳ vọng rank 2 hoặc 3 "
                    f"(model: '{self.model_path}')."
                )
            feature_candidates = {d for d in shape if d not in (1,)}
            if self.num_classes is not None:
                expected_feat = 4 + self.num_classes
                if expected_feat not in shape and 6 not in shape:
                    raise ModelLoadError(
                        f"Output shape {shape} không khớp num_classes={self.num_classes} "
                        f"(kỳ vọng feature dim = {expected_feat} hoặc 6 cho NMS-embedded); "
                        f"model: '{self.model_path}'."
                    )
            del dims, feature_candidates  # chỉ dùng để early-fail rõ ràng ở trên

    # ------------------------------------------------------------------ infer

    def infer(self, frame: np.ndarray) -> list[dict[str, Any]]:
        import cv2  # lazy: OpenCV chỉ cần cho tiền xử lý ảnh khi chạy thật.

        started = time.perf_counter()
        original_h, original_w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (self.input_width, self.input_height))
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        detail = self.input_details[0]
        if detail["dtype"] == np.uint8:
            scale, zero_point = detail["quantization"]
            tensor = np.clip(tensor / scale + zero_point, 0, 255).astype(np.uint8)
        else:
            tensor = tensor.astype(detail["dtype"])

        try:
            self.interpreter.set_tensor(detail["index"], tensor)
            self.interpreter.invoke()
            outputs = []
            for item in self.output_details:
                raw = self.interpreter.get_tensor(item["index"])
                quant = item.get("quantization", (0.0, 0))
                scale, zero_point = quant[0], quant[1]
                if raw.dtype in (np.uint8, np.int8) and scale:
                    raw = (raw.astype(np.float32) - zero_point) * scale
                outputs.append(raw)
        except Exception as exc:  # noqa: BLE001 - lỗi runtime khi invoke interpreter
            raise ModelInferenceError(f"Lỗi khi chạy inference: {exc}") from exc

        detections = self._decode_outputs(outputs, original_w, original_h)
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "step.infer.done",
            n_detections=len(detections),
            latency_ms=round(latency_ms, 1),
        )
        return detections

    # ------------------------------------------------------------------ decode

    def _decode_outputs(
        self,
        outputs: list[np.ndarray],
        image_w: int,
        image_h: int,
    ) -> list[dict[str, Any]]:
        if not outputs:
            return []

        try:
            raw = np.asarray(outputs[0], dtype=np.float32)
            # Chỉ bỏ trục batch dẫn đầu, KHÔNG dùng squeeze (sẽ xoá nhầm trục anchor=1).
            if raw.ndim == 3 and raw.shape[0] == 1:
                raw = raw[0]
            if raw.ndim != 2:
                raise ModelDecodingError(
                    f"Unsupported detector output shape {tuple(np.asarray(outputs[0]).shape)}; "
                    "expected [N, 6] or [4+num_classes, N] (with optional leading batch axis)."
                )

            raw, mode = self._orient(raw)
            if mode == "nms":
                return self._decode_nms_embedded(raw, image_w, image_h)
            return self._decode_yolo(raw, image_w, image_h)
        except ModelDecodingError:
            raise
        except Exception as exc:  # noqa: BLE001 - bất kỳ lỗi decode bất ngờ nào khác
            raise ModelDecodingError(f"Lỗi khi decode output model: {exc}") from exc

    def _orient(self, raw: np.ndarray) -> tuple[np.ndarray, str]:
        """Đưa mảng về dạng ``[num_predictions, feature]`` và xác định layout."""
        rows, cols = raw.shape
        yolo_feat = (4 + self.num_classes) if self.num_classes else None

        # Ưu tiên khớp theo số feature đã biết (num_classes cấu hình từ class map).
        if yolo_feat is not None and yolo_feat in (rows, cols):
            if cols != yolo_feat:  # feature đang là trục 0 -> transpose
                raw = raw.T
            return raw, "yolo"
        if 6 in (rows, cols):
            if cols != 6:
                raw = raw.T
            return raw, "nms"

        # Không biết num_classes: coi trục nhỏ hơn là feature (anchors >> features).
        if rows < cols:  # [feature, anchors] -> transpose
            raw = raw.T
        feat = raw.shape[1]
        return raw, ("nms" if feat == 6 else "yolo")

    def _decode_yolo(
        self,
        raw: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> list[dict[str, Any]]:
        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4:]
        if self.num_classes is not None:
            class_scores = class_scores[:, : self.num_classes]

        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        keep = scores >= self.confidence_threshold
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        cx, cy, w, h = (
            boxes_xywh[:, 0],
            boxes_xywh[:, 1],
            boxes_xywh[:, 2],
            boxes_xywh[:, 3],
        )
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        boxes_xyxy = self._to_image_pixels(boxes_xyxy, image_w, image_h)
        return self._nms_and_pack(boxes_xyxy, scores, class_ids, image_w, image_h)

    def _decode_nms_embedded(
        self,
        raw: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> list[dict[str, Any]]:
        scores = raw[:, 4].astype(np.float32)
        keep = scores >= self.confidence_threshold
        if not np.any(keep):
            return []
        raw = raw[keep]
        boxes_xyxy = raw[:, :4].astype(np.float32)
        boxes_xyxy = self._to_image_pixels(boxes_xyxy, image_w, image_h)
        class_ids = raw[:, 5].astype(int)
        return self._nms_and_pack(boxes_xyxy, raw[:, 4], class_ids, image_w, image_h)

    def _to_image_pixels(
        self,
        boxes_xyxy: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> np.ndarray:
        """Scale bbox về pixel của khung hình gốc.

        Toạ độ raw có thể normalized ``[0, 1]`` (YOLOv8 tflete mặc định) hoặc theo
        pixel của input model (``input_width`` × ``input_height``).
        """
        boxes = boxes_xyxy.astype(np.float32).copy()
        if boxes.size and float(np.nanmax(np.abs(boxes))) <= 1.5:
            boxes[:, [0, 2]] *= image_w
            boxes[:, [1, 3]] *= image_h
        else:
            boxes[:, [0, 2]] *= image_w / self.input_width
            boxes[:, [1, 3]] *= image_h / self.input_height
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_h - 1)
        return boxes

    def _nms_and_pack(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for class_id in np.unique(class_ids):
            idx = np.where(class_ids == class_id)[0]
            kept = self._nms(boxes_xyxy[idx], scores[idx], self.nms_iou_threshold)
            for k in kept:
                gi = int(idx[k])
                x1, y1, x2, y2 = boxes_xyxy[gi]
                if x2 <= x1 or y2 <= y1:
                    continue
                w = float(x2 - x1)
                h = float(y2 - y1)
                detections.append(
                    {
                        "box": (float(x1), float(y1), w, h),
                        "center": (float(x1) + w / 2.0, float(y1) + h / 2.0),
                        "conf": float(scores[gi]),
                        "class_id": int(class_id),
                        "class": str(int(class_id)),
                    }
                )
        return detections

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Greedy non-maximum suppression thuần numpy. Trả về chỉ số giữ lại."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[np.where(iou <= iou_threshold)[0]]
        return keep
