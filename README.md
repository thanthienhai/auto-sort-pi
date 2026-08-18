# Auto Sort Pi

Hệ thống xử lý hình ảnh và điều phối phân loại rác tự động chạy trên **Raspberry Pi 4**, sử dụng camera cố định, mô hình Object Detection và giao tiếp UART với MCU/robot để xác định vị trí, lập lịch và thực hiện thao tác gắp rác.

> **Status:** Technical specification / implementation blueprint. Các thông số và tiêu chí dưới đây được lấy từ đặc tả kỹ thuật của module xử lý ảnh và định vị; README không mặc định rằng toàn bộ module đã được triển khai hoàn chỉnh.

## Mục tiêu

`auto-sort-pi` chịu trách nhiệm cho pipeline chính:

1. Thu nhận video thời gian thực từ camera.
2. Phát hiện vật thể bằng YOLO tối ưu cho Raspberry Pi.
3. Ánh xạ class chi tiết của model về 4 nhóm rác chính.
4. Chuyển tọa độ tâm bounding box từ pixel sang tọa độ thực tế `(X, Y)` tính bằng mm.
5. Lập lịch thứ tự gắp dựa trên khoảng cách và ưu tiên vật liệu dễ vỡ.
6. Gửi lệnh gắp tuần tự tới MCU qua UART.
7. Chờ `ACK` / `DONE` / `ERR` để bảo đảm quá trình điều khiển có handshake.

## Kiến trúc tổng thể

```text
┌──────────────┐
│ Camera 1080p │
└──────┬───────┘
       │
       ▼
┌────────────────────────────┐
│ Frame Preprocessing        │
│ Resize 320×320 + Normalize │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│ AI Inference               │
│ YOLOv8n / YOLOv11n - INT8  │
└────────────┬───────────────┘
             │ detections
             ▼
┌────────────────────────────┐
│ Post-processing             │
│ NMS + Class Mapping         │
└────────────┬───────────────┘
             │ center (u, v)
             ▼
┌────────────────────────────┐
│ Homography Transformation   │
│ Pixel → World (X, Y) in mm │
└────────────┬───────────────┘
             │ targets
             ▼
┌────────────────────────────┐
│ Scheduler                  │
│ Nearest Neighbor + Batching│
└────────────┬───────────────┘
             │ target sequence
             ▼
┌────────────────────────────┐
│ UART Handshake Handler      │
└────────────┬───────────────┘
             │
             ▼
       ┌─────────────┐
       │ MCU / Robot │
       └─────────────┘
```

## Phần cứng & môi trường

| Thành phần | Yêu cầu / khuyến nghị |
|---|---|
| CPU | Raspberry Pi 4 Model B, khuyến nghị 4GB / 8GB RAM |
| OS | Raspberry Pi OS 64-bit, ưu tiên bản Lite |
| Camera | Full HD 1920×1080, CSI-2 Pi Camera v2/v3 hoặc USB 3.0 UVC |
| UART | `/dev/ttyAMA0` hoặc `/dev/ttyUSB0` |
| Baudrate | `115200 bps` |
| AI Runtime | `tflite-runtime` hoặc `onnxruntime` trên ARM Cortex-A72 |

## AI Detection

### Model

- Kiến trúc: **YOLOv8n (Nano)** hoặc **YOLOv11n**.
- Input: `320 × 320 × 3 RGB`.
- Deployment format: TensorFlow Lite Quantized **INT8** (`.tflite`).
- Confidence threshold: `≥ 0.50`.
- NMS IoU threshold: `≤ 0.45`.

`modules/detector.py` tự nhận diện và hỗ trợ hai layout output:

- **YOLOv8/v11 raw** `[1, 4+num_classes, num_anchors]` (export Ultralytics mặc định,
  `nms=False`) — decode `(cx,cy,w,h)` + argmax lớp, đổi `xywh → xyxy`, rồi NMS
  (per-class) ngay trên Pi.
- **NMS-embedded** `[N, 6]` = `(x1,y1,x2,y2,score,class_id)` (export kèm
  `TFLite_Detection_PostProcess`).

Toạ độ raw được tự suy luận là normalized `[0,1]` hay theo pixel của input model.
Số lớp lấy từ `config/model_classes.json`. Nếu model có layout khác, hàm decode
báo lỗi rõ ràng thay vì trả về toạ độ sai.

### Phân loại 4 nhóm

| Group ID | Nhóm | Màu thùng | Model classes |
|---:|---|---|---|
| 1 | **KIM LOẠI** | Đỏ | `can`, `tin_can`, `aluminum_foil`, `metal_cap` |
| 2 | **TÁI CHẾ KHÁC** | Vàng | `cardboard`, `paper_cup`, `tetra_pak`, `paper_box`, `newspaper` |
| 3 | **RÁC THẢI NHỰA** | Xanh lá | `pet_bottle`, `plastic_cup`, `detergent_bottle`, `nylon_bag` |
| 4 | **THỦY TINH / TÁI CHẾ RIÊNG** | Xanh dương | `glass_bottle`, `glass_jar`, `broken_glass` |

## Pixel → World Coordinates

Camera được cố định phía trên mặt bàn phân loại, vì vậy hệ thống sử dụng **Homography Transformation** để chuyển tọa độ pixel `(u, v)` sang tọa độ cơ khí thực tế `(Xw, Yw)` theo mm.

```text
[x']   [h11 h12 h13] [u]
[y'] = [h21 h22 h23] [v]
[w']   [h31 h32 h33] [1]

Xw = x' / w'
Yw = y' / w'
```

Ma trận `H` kích thước `3×3` được calibration trước từ 4 điểm mốc cố định trên mặt bàn bằng `cv2.findHomography()`.

Điểm đại diện của vật thể là tâm bounding box:

```text
u_center = x_min + width / 2
v_center = y_min + height / 2
```

## Sorting & Scheduling

Khi có nhiều vật thể cùng lúc, scheduler sử dụng **Nearest Neighbor** kết hợp **Batching**.

Quy trình:

1. Khởi tạo vị trí hiện tại của đầu gắp `P0 = (Xcur, Ycur)`.
2. Ưu tiên **Nhóm 4 - Thủy tinh** nếu tồn tại vì đây là nhóm dễ vỡ.
3. Tính khoảng cách Euclid từ vị trí hiện tại tới các vật còn lại.
4. Chọn vật có khoảng cách nhỏ nhất.
5. Cập nhật vị trí hiện tại thành vị trí thùng chứa sau khi robot thả vật.
6. Lặp cho tới khi toàn bộ target được xử lý.

## UART Protocol

### Serial configuration

```text
Baudrate   : 115200
Data bits  : 8
Parity     : None
Stop bits  : 1
Flow       : None
```

### Command frame

Pi gửi MCU theo format:

```text
$CMD,GROUP_ID,POS_X,POS_Y,ANGLE,CHECKSUM\n
```

Trong đó:

- `CMD`: `PICK`, `SCAN` hoặc `PURGE`.
- `GROUP_ID`: `1..4`.
- `POS_X`, `POS_Y`: tọa độ thực tế, mm, 1 chữ số thập phân.
- `ANGLE`: góc xoay theo trục dọc, đơn vị độ.
- `CHECKSUM`: XOR checksum dạng Hex 2 ký tự, tính từ sau `$` đến trước `*`.

Ví dụ:

```text
$PICK,1,150.2,320.0,0.0*09\n
```

> Checksum `09` là XOR thực tế của payload `PICK,1,150.2,320.0,0.0` (từ sau `$` đến
> trước `*`), khớp với `SerialCommunicator.calc_checksum`.

### Response frame

MCU phản hồi:

```text
!ACK\n
!DONE\n
!ERR,CODE\n
```

- `!ACK`: MCU đã nhận lệnh và bắt đầu thực thi.
- `!DONE`: hoàn thành gắp và thả.
- `!ERR,CODE`: lỗi phần cứng hoặc lỗi thực thi.

Pi gửi lệnh tiếp theo chỉ sau khi nhận `!DONE`; nếu nhận lỗi hoặc timeout thì dừng xử lý target hiện tại.

## Cấu trúc mã nguồn đề xuất

```text
auto-sort-pi/
├── config/
│   ├── calibration_matrix.json   # Ma trận Homography H
│   ├── class_mapping.json        # AI class name → Group ID
│   ├── model_classes.json        # class_id (thứ tự) → class name
│   └── settings.yaml             # Camera, UART, threshold
├── models/
│   └── yolov8n_waste_int8.tflite # AI model
├── modules/
│   ├── __init__.py
│   ├── camera_stream.py           # Camera stream / FPS optimization
│   ├── detector.py                # TFLite inference + decode + NMS
│   ├── transformer.py             # Pixel → World (numpy)
│   ├── scheduler.py               # Sorting / trajectory scheduling
│   ├── serial_comm.py             # UART + handshake
│   └── logging_setup.py           # structlog: log per-cycle ra file JSON
├── tests/                         # Unit tests (pytest, không cần phần cứng)
├── main.py                        # Main state machine (FSM)
└── requirements.txt
```

Đây là cấu trúc được đề xuất trong SRS và có thể được dùng làm baseline cho implementation.

## Core flow

Logic điều phối chính có dạng:

```python
frame = camera.read()
detections = detector.infer(frame)

for detection in detections:
    group_id = mapping.get(detection["class"])
    x, y = transformer.pixel_to_world(*detection["center"])
    targets.append({
        "group_id": group_id,
        "x": x,
        "y": y,
        "conf": detection["conf"],
    })

scheduled_targets = optimize_trajectory(targets)

for target in scheduled_targets:
    comm.send_pick_command(
        group_id=target["group_id"],
        x=target["x"],
        y=target["y"],
    )
```

Luồng xử lý: detect → mapping → transform → schedule → UART command.

## Calibration

File `config/calibration_matrix.json` lưu ma trận Homography đã calibration. Việc calibration phải xác định 4 điểm mốc cố định trên mặt bàn để tính `H` trước khi vận hành hệ thống.

Ví dụ cấu trúc:

```json
{
  "homography_matrix": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ]
}
```

> Ma trận trên chỉ là ví dụ cấu trúc, **không phải calibration thực tế**.

## Reliability & Error Handling

UART command cần bảo đảm:

- Kiểm tra checksum.
- Chờ `!ACK` sau khi gửi command.
- Chờ `!DONE` để xác nhận hoàn thành.
- Xử lý `!ERR,CODE`.
- Timeout cho một hành trình gắp.
- Retry tối đa **3 lần** khi mất gói hoặc checksum sai theo tiêu chí nghiệm thu.

## Performance Targets

| Hạng mục | Tiêu chí nghiệm thu |
|---|---|
| Inference latency | `≤ 90 ms/frame` trên Raspberry Pi 4 |
| Inference throughput | `≥ 10 FPS` |
| Position accuracy | Sai số `≤ ±3.0 mm` |
| UART resilience | Retry tối đa `3` lần khi mất gói / checksum lỗi |

## Logging

`modules/logging_setup.py` cấu hình structlog theo chuẩn dự án: mỗi *chu kỳ* xử lý
(một vòng `capture → detect → transform → schedule → uart`) mang một `request_id`
có tiền tố timestamp, được bind bằng contextvars ở đầu chu kỳ trong `main.py`. Mọi
log của các module con tự đính kèm `request_id` và được ghi:

- Ra file `logs/requests/<request_id>.log` (JSON lines) — mỗi chu kỳ một file.
- Ra console để theo dõi khi vận hành.

Tên event theo dạng `<domain>.<action>.<status>` (ví dụ `step.infer.done`,
`step.uart.tx`, `step.pick.failed`, `cycle.success`).

## Testing

```bash
pip install pytest
pytest -q
```

Unit test trong `tests/` phủ scheduler, transformer, decode detector (numpy thuần)
và handshake UART (serial giả) — **không cần** camera, MCU hay model `.tflite`, nên
chạy được trên máy dev bất kỳ.

## Development Notes

README này mô tả kiến trúc và yêu cầu kỹ thuật của hệ thống. Khi triển khai thực tế, các thành phần sau cần được cấu hình theo phần cứng cụ thể:

- Camera source và resolution.
- UART device (`/dev/ttyAMA0` hoặc `/dev/ttyUSB0`).
- Ma trận Homography đã calibration.
- Model `.tflite` thực tế.
- Class mapping tương ứng với model được huấn luyện.
- Vị trí các thùng chứa trong hệ tọa độ world.
- Timeout / retry policy của MCU.
