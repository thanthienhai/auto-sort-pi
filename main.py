"""Main state machine for the Auto Sort Pi pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from modules import (
    CameraStream,
    CoordinateTransformer,
    SerialCommunicator,
    WasteDetector,
    optimize_trajectory,
)


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    settings = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    mapping = load_json("config/class_mapping.json")
    class_names = load_json("config/model_classes.json")

    camera_cfg = settings["camera"]
    infer_cfg = settings["inference"]
    uart_cfg = settings["uart"]
    scheduler_cfg = settings["scheduler"]

    camera = CameraStream(
        src=camera_cfg["source"],
        width=camera_cfg["width"],
        height=camera_cfg["height"],
    ).start()
    detector = WasteDetector(
        infer_cfg["model_path"],
        confidence_threshold=infer_cfg["confidence_threshold"],
        nms_iou_threshold=infer_cfg["nms_iou_threshold"],
    )
    transformer = CoordinateTransformer("config/calibration_matrix.json")
    comm = SerialCommunicator(
        port=uart_cfg["port"],
        baudrate=uart_cfg["baudrate"],
        timeout=uart_cfg["timeout"],
        ack_timeout=uart_cfg["ack_timeout"],
        action_timeout=uart_cfg["action_timeout"],
        max_retries=uart_cfg["max_retries"],
    )

    print("Auto Sort Pi: vision pipeline ready.")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(settings["runtime"]["loop_delay_seconds"])
                continue

            detections = detector.infer(frame)
            targets = []
            for detection in detections:
                class_id = int(detection["class_id"])
                if class_id < 0 or class_id >= len(class_names):
                    continue

                class_name = class_names[class_id]
                group_id = mapping.get(class_name)
                if group_id is None:
                    continue

                u_center, v_center = detection["center"]
                x_world, y_world = transformer.pixel_to_world(u_center, v_center)
                targets.append(
                    {
                        "group_id": int(group_id),
                        "x": x_world,
                        "y": y_world,
                        "conf": float(detection["conf"]),
                        "class": class_name,
                    }
                )

            if not targets:
                time.sleep(settings["runtime"]["loop_delay_seconds"])
                continue

            scheduled = optimize_trajectory(
                targets,
                start_position=tuple(scheduler_cfg["start_position_mm"]),
                fragile_group_id=int(scheduler_cfg["fragile_group_id"]),
            )

            for target in scheduled:
                ok = comm.send_pick_command(
                    group_id=target["group_id"],
                    x=target["x"],
                    y=target["y"],
                )
                if not ok:
                    print(
                        "Pick failed at "
                        f"({target['x']:.1f}, {target['y']:.1f}) "
                        f"for group {target['group_id']}"
                    )
                    break

    except KeyboardInterrupt:
        print("Stopping Auto Sort Pi...")
    finally:
        camera.stop()
        comm.close()


if __name__ == "__main__":
    main()
