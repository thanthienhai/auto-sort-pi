"""TensorFlow Lite waste detector adapter."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # pragma: no cover - development machines may use tensorflow
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


class WasteDetector:
    """Run a TFLite object detector and return normalized detection dictionaries."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.50,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.input_height, self.input_width, _ = self.input_details[0]["shape"]

    def infer(self, frame: np.ndarray) -> list[dict[str, Any]]:
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

        self.interpreter.set_tensor(detail["index"], tensor)
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(item["index"]) for item in self.output_details]
        return self._decode_outputs(outputs, original_w, original_h)

    def _decode_outputs(
        self,
        outputs: list[np.ndarray],
        image_w: int,
        image_h: int,
    ) -> list[dict[str, Any]]:
        """Decode common YOLO-style TFLite output layouts.

        The exact tensor layout depends on the exported model. This method supports
        [1, N, 6] / [N, 6] style detections and deliberately fails clearly for
        unsupported layouts instead of silently returning incorrect coordinates.
        """
        if not outputs:
            return []

        raw = np.asarray(outputs[0])
        raw = np.squeeze(raw)
        if raw.ndim != 2 or raw.shape[1] < 6:
            raise ValueError(
                f"Unsupported detector output shape {tuple(raw.shape)}; "
                "expected [N, >=6]."
            )

        detections: list[dict[str, Any]] = []
        boxes: list[list[int]] = []
        scores: list[float] = []
        classes: list[int] = []

        for row in raw:
            x1, y1, x2, y2 = map(float, row[:4])
            score = float(row[4])
            class_id = int(row[5])
            if score < self.confidence_threshold:
                continue

            # Treat normalized coordinates as [0, 1], otherwise as model pixels.
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * image_w, x2 * image_w
                y1, y2 = y1 * image_h, y2 * image_h
            else:
                x1 *= image_w / self.input_width
                x2 *= image_w / self.input_width
                y1 *= image_h / self.input_height
                y2 *= image_h / self.input_height

            x1 = max(0, min(image_w - 1, x1))
            y1 = max(0, min(image_h - 1, y1))
            x2 = max(0, min(image_w - 1, x2))
            y2 = max(0, min(image_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(score)
            classes.append(class_id)

        keep = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence_threshold,
            self.nms_iou_threshold,
        )
        for index in np.asarray(keep).reshape(-1):
            x, y, w, h = boxes[int(index)]
            detections.append(
                {
                    "box": (x, y, w, h),
                    "center": (x + w / 2.0, y + h / 2.0),
                    "conf": scores[int(index)],
                    "class_id": classes[int(index)],
                    "class": str(classes[int(index)]),
                }
            )
        return detections
