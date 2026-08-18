"""Kiểm thử giải mã output detector (thuần numpy, không cần cv2/tflite)."""

import numpy as np

from modules.detector import WasteDetector


def _make_detector(num_classes):
    """Tạo detector không load model — chỉ set thuộc tính cho phần decode."""
    det = WasteDetector.__new__(WasteDetector)
    det.confidence_threshold = 0.50
    det.nms_iou_threshold = 0.45
    det.num_classes = num_classes
    det.input_width = 320
    det.input_height = 320
    return det


def test_decode_yolov8_raw_layout():
    # Layout [1, 4+nc, N] = [1, 7, 2] với toạ độ normalized.
    det = _make_detector(num_classes=3)
    raw = np.zeros((1, 7, 2), dtype=np.float32)
    # anchor 0: mạnh, lớp 1, tâm giữa ảnh, box 0.2 x 0.2
    raw[0, :, 0] = [0.5, 0.5, 0.2, 0.2, 0.10, 0.90, 0.20]
    # anchor 1: yếu -> bị loại
    raw[0, :, 1] = [0.1, 0.1, 0.1, 0.1, 0.10, 0.10, 0.10]

    dets = det._decode_outputs([raw], image_w=100, image_h=100)
    assert len(dets) == 1
    d = dets[0]
    assert d["class_id"] == 1
    assert abs(d["conf"] - 0.90) < 1e-6
    cx, cy = d["center"]
    assert abs(cx - 50.0) < 1.0 and abs(cy - 50.0) < 1.0
    w = d["box"][2]
    assert abs(w - 20.0) < 1.0  # 0.2 * 100


def test_nms_suppresses_overlapping_same_class():
    # Layout [N, 6] = (x1,y1,x2,y2,score,class), normalized.
    det = _make_detector(num_classes=3)  # 6 != 4+3 -> nhận diện nhánh nms-embedded
    raw = np.array(
        [
            [0.10, 0.10, 0.30, 0.30, 0.90, 1],  # giữ (score cao nhất)
            [0.12, 0.12, 0.32, 0.32, 0.80, 1],  # trùng lớp 1, IoU>0.45 -> loại
            [0.60, 0.60, 0.80, 0.80, 0.70, 2],  # lớp khác -> giữ
        ],
        dtype=np.float32,
    )
    dets = det._decode_outputs([raw], image_w=100, image_h=100)
    classes = sorted(d["class_id"] for d in dets)
    assert classes == [1, 2]
    top1 = next(d for d in dets if d["class_id"] == 1)
    assert abs(top1["conf"] - 0.90) < 1e-6


def test_below_threshold_returns_empty():
    det = _make_detector(num_classes=3)
    raw = np.zeros((1, 7, 1), dtype=np.float32)
    raw[0, :, 0] = [0.5, 0.5, 0.2, 0.2, 0.1, 0.2, 0.1]  # max score 0.2 < 0.5
    assert det._decode_outputs([raw], 100, 100) == []


def test_pixel_scale_layout():
    # Toạ độ theo pixel model (>1.5) phải được scale theo input_width/height.
    det = _make_detector(num_classes=3)
    raw = np.zeros((1, 7, 1), dtype=np.float32)
    # tâm (160,160) trên 320 -> giữa ảnh; w=64 -> 0.2 ảnh
    raw[0, :, 0] = [160.0, 160.0, 64.0, 64.0, 0.1, 0.9, 0.1]
    dets = det._decode_outputs([raw], image_w=100, image_h=100)
    assert len(dets) == 1
    cx, cy = dets[0]["center"]
    assert abs(cx - 50.0) < 1.0 and abs(cy - 50.0) < 1.0
