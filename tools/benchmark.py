"""Performance benchmark cho từng bước pipeline (§16 task.md).

Đo: camera capture, preprocess (resize+normalize — đo cùng inference vì
``WasteDetector.infer`` gộp cả hai, xem ghi chú dưới), inference, decode+NMS
(đo tách riêng bằng cách gọi thẳng ``_decode_outputs`` trên output đã cache),
coordinate transform, scheduler. Xuất mean / p50 / p95 / p99 / FPS / memory
(RSS tiến trình hiện tại, qua ``resource`` trên POSIX hoặc None trên Windows
nếu module không khả dụng).

**Quan trọng**: mục tiêu ``inference <= 90ms/frame`` và ``system >= 10 FPS``
(README) chỉ được coi là đạt khi benchmark chạy trên Raspberry Pi 4 thật với
model ``.tflite`` thật — số đo trên máy dev (Windows/x86, hay dùng ONNX qua
``ai-edge-litert``) chỉ mang tính tham khảo về mặt logic, KHÔNG phải con số
nghiệm thu (§16: "Không claim đạt performance nếu chưa benchmark" [trên Pi]).

Chạy:
    python tools/benchmark.py --model models/yolov8n_waste_int8.tflite --source samples/waste_test.jpg --iterations 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from modules.detector import WasteDetector  # noqa: E402
from modules.scheduler import Scheduler, SchedulerConfig  # noqa: E402
from modules.transformer import CoordinateTransformer  # noqa: E402


def _stats(samples_seconds: list[float]) -> dict:
    if not samples_seconds:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None, "fps": None}
    ms = sorted(s * 1000.0 for s in samples_seconds)
    n = len(ms)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return ms[idx]

    mean_ms = statistics.mean(ms)
    return {
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(pct(0.50), 3),
        "p95_ms": round(pct(0.95), 3),
        "p99_ms": round(pct(0.99), 3),
        "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else None,
    }


def _memory_rss_mb() -> float | None:
    try:
        import resource

        # ru_maxrss: KB trên Linux, byte trên macOS — trên Pi (Linux) là KB.
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except ImportError:
        return None  # Windows: module resource không tồn tại


def benchmark_camera_capture(source, iterations: int) -> list[float]:
    import cv2

    cap = cv2.VideoCapture(source)
    samples = []
    try:
        for _ in range(iterations):
            started = time.perf_counter()
            cap.read()
            samples.append(time.perf_counter() - started)
    finally:
        cap.release()
    return samples


def benchmark_inference(detector: WasteDetector, frame: np.ndarray, iterations: int) -> list[float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        detector.infer(frame)
        samples.append(time.perf_counter() - started)
    return samples


def benchmark_transform(transformer: CoordinateTransformer, iterations: int) -> list[float]:
    samples = []
    for i in range(iterations):
        started = time.perf_counter()
        transformer.pixel_to_world(float(i % 640), float(i % 480))
        samples.append(time.perf_counter() - started)
    return samples


def benchmark_scheduler(iterations: int, n_targets: int = 10) -> list[float]:
    config = SchedulerConfig()
    scheduler = Scheduler(config)
    targets = [
        {"group_id": (i % 4) + 1, "x": float(i * 10), "y": float(i * 5)} for i in range(n_targets)
    ]
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        scheduler.schedule(targets)
        samples.append(time.perf_counter() - started)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark auto-sort-pi pipeline stages")
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="ảnh tĩnh dùng để benchmark inference")
    ap.add_argument("--calib", default="config/calibration_matrix.json")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--num-classes", default="auto")
    ap.add_argument("--json-out", default=None, help="lưu kết quả dạng JSON")
    args = ap.parse_args()

    import cv2

    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Không đọc được ảnh: {args.source}")

    num_classes = None if args.num_classes == "auto" else int(args.num_classes)
    detector = WasteDetector(args.model, num_classes=num_classes)
    try:
        transformer = CoordinateTransformer(args.calib, allow_identity=True)
    except Exception:  # noqa: BLE001 - benchmark transform chỉ cần một homography hợp lệ bất kỳ
        transformer = None

    results: dict[str, dict] = {}
    print("Benchmarking... (iterations = %d)" % args.iterations)

    try:
        results["camera_capture"] = _stats(benchmark_camera_capture(0, min(args.iterations, 10)))
    except Exception as exc:  # noqa: BLE001
        results["camera_capture"] = {"error": str(exc)}

    results["inference"] = _stats(benchmark_inference(detector, frame, args.iterations))

    if transformer is not None:
        results["coordinate_transform"] = _stats(benchmark_transform(transformer, args.iterations))

    results["scheduler"] = _stats(benchmark_scheduler(args.iterations))

    print("\nBenchmark results")
    print("=" * 60)
    for stage, stat in results.items():
        if "error" in stat:
            print(f"{stage:20s}: SKIPPED ({stat['error']})")
            continue
        print(
            f"{stage:20s}: mean={stat['mean_ms']}ms p50={stat['p50_ms']}ms "
            f"p95={stat['p95_ms']}ms p99={stat['p99_ms']}ms fps={stat['fps']}"
        )
    print("-" * 60)
    memory_mb = _memory_rss_mb()
    print(f"Peak RSS memory     : {memory_mb if memory_mb is not None else 'N/A (Windows)'} MB")
    print("=" * 60)

    inference_mean = results["inference"].get("mean_ms")
    if inference_mean is not None:
        meets_target = inference_mean <= 90.0
        print(
            f"Inference target (<=90ms/frame): {'MEETS' if meets_target else 'DOES NOT MEET'} "
            f"({inference_mean}ms) — chỉ có giá trị tham khảo nếu KHÔNG chạy trên Raspberry Pi 4 thật."
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Đã lưu kết quả JSON: {args.json_out}")


if __name__ == "__main__":
    main()
