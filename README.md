# Auto Sort Pi

Hệ thống xử lý hình ảnh và điều phối phân loại rác tự động chạy trên **Raspberry Pi 4**, sử dụng camera cố định, mô hình Object Detection (YOLOv8/v11 TFLite) và giao tiếp UART với MCU/robot để xác định vị trí, lập lịch và thực hiện thao tác gắp rác.

> **Status:** Software implementation với FSM, validation, safety layer, dedup, và test suite đầy đủ (`pytest -q` — 124 test, tất cả pass). **Chưa được chạy/kiểm chứng trên Raspberry Pi 4 thật, camera thật, model `.tflite` đã huấn luyện thật, hay MCU thật** — xem mục [Implemented / Experimental / Hardware-dependent / TODO](#implemented--experimental--hardware-dependent--todo) để biết chính xác phần nào đã validate ở mức nào.

## Mục tiêu

`auto-sort-pi` chịu trách nhiệm cho pipeline chính:

1. Thu nhận video thời gian thực từ camera (threaded, latest-frame buffer, reconnect).
2. Phát hiện vật thể bằng YOLO TFLite tối ưu cho Raspberry Pi.
3. Validate detection, ánh xạ class chi tiết của model về 4 nhóm rác chính.
4. Chuyển tọa độ tâm bounding box từ pixel sang tọa độ thực tế `(X, Y)` mm qua homography đã calibration.
5. Validate target (workspace, confidence, staleness, đã xử lý chưa) và loại trùng lặp giữa các scan cycle (dedup).
6. Lập lịch thứ tự gắp dựa trên khoảng cách và ưu tiên vật liệu dễ vỡ.
7. Safety-check lần cuối trước khi gửi lệnh gắp tuần tự tới MCU qua UART.
8. Chờ `ACK` / `DONE` / `ERR` (handshake, retry, reconnect) rồi xác nhận kết quả trước khi chuyển sang target kế tiếp.

## Architecture

Pipeline điều phối bằng một FSM tường minh trong `main.py` (`class AutoSortPi`):

```text
Camera → Frame Capture → Preprocessing → Object Detection → Detection Validation
  → Class Mapping → Pixel→World → Target Validation → Object Dedup/Tracking
  → Scheduler → Safety Check → UART Command → ACK → DONE/ERR
  → Post-action verification (bookkeeping) → Next target
```

FSM states:

```text
BOOT → SELF_CHECK → INITIALIZING → READY → SCANNING → SCHEDULING
  → VALIDATING_TARGET → EXECUTING → WAITING_ACK → WAITING_DONE
  → VERIFYING_RESULT → (VALIDATING_TARGET | SCANNING)

ERROR / RECOVERING / SAFE_STOP: xử lý lỗi phân loại theo nhóm (camera / model /
calibration / target / UART / MCU / timeout / cấu hình), mỗi nhóm có recovery
policy riêng (xem main.py::_state_error / _state_recovering).
```

- Lỗi **cấu hình / model không tương thích / calibration placeholder** phát hiện ở `BOOT`/`SELF_CHECK`/`INITIALIZING` → **fatal**, đi thẳng `SAFE_STOP` (cần con người sửa rồi khởi động lại).
- Lỗi **camera tạm thời / UART mất kết nối / MCU báo lỗi / timeout** → **recoverable**, qua `RECOVERING` với backoff, không làm chết tiến trình.
- Lỗi decode/inference của **một frame** hoặc lỗi UART/MCU của **một target** không làm hỏng cả chu kỳ — bị bỏ qua (log rõ lý do), pipeline tiếp tục với frame/target kế tiếp.

### Cấu trúc mã nguồn

```text
auto-sort-pi/
├── config/
│   ├── calibration_matrix.json   # Homography H — hiện là PLACEHOLDER identity, chưa calibration thật
│   ├── class_mapping.json        # AI class name → Group ID
│   ├── model_classes.json        # class_id (thứ tự) → class name
│   └── settings.yaml             # Toàn bộ tham số vận hành (§20 — không hard-code)
├── deploy/
│   ├── auto-sort-pi.service      # systemd unit
│   ├── auto-sort-pi.env.example  # biến môi trường triển khai
│   └── install.sh                # script cài đặt hạ tầng lên Pi
├── models/                       # đặt model .tflite thật ở đây (chưa có sẵn trong repo)
├── modules/
│   ├── camera_stream.py          # threaded capture, frame age, reconnect
│   ├── detector.py                # TFLite inference + decode + NMS + validate model spec
│   ├── transformer.py             # Pixel → World (numpy) + validate calibration
│   ├── scheduler.py               # scheduling config-driven (drop positions, workspace, batch)
│   ├── target_validator.py        # validate target trước scheduler/UART
│   ├── safety_validator.py        # gate cuối trước khi gửi UART
│   ├── dedup.py                    # object dedup: spatial exclusion + cooldown + disappearance confirm
│   ├── serial_comm.py             # UART transport + handshake + reconnect
│   ├── uart_protocol.py           # framing/parsing UART thuần (tách khỏi transport)
│   ├── config_validation.py       # validate settings.yaml khi startup
│   ├── exceptions.py               # exception hierarchy phân loại lỗi
│   └── logging_setup.py           # structlog: log per-cycle ra file JSON + rotation theo tuổi
├── tools/
│   ├── dev_detect.py               # chạy decode/NMS trên ảnh tĩnh/webcam (dev, không cần Pi)
│   ├── validate_model.py           # validate model .tflite trước khi deploy
│   ├── calibrate.py                # tool calibration homography (interactive + --points-file)
│   ├── test_calibration.py         # validate calibration đã lưu (mean/max/RMSE error)
│   ├── benchmark.py                 # benchmark từng stage pipeline
│   └── health.py                    # health check CLI
├── tests/                          # 124 test, pytest, không cần phần cứng
├── main.py                         # FSM chính
└── requirements.txt
```

## Requirements

| Thành phần | Yêu cầu / khuyến nghị |
|---|---|
| CPU | Raspberry Pi 4 Model B, khuyến nghị 4GB/8GB RAM |
| OS | Raspberry Pi OS 64-bit, ưu tiên bản Lite |
| Camera | Full HD 1920×1080, CSI-2 Pi Camera v2/v3 hoặc USB 3.0 UVC |
| UART | `/dev/ttyAMA0` hoặc `/dev/ttyUSB0`, `115200 bps` |
| AI Runtime | `tflite-runtime` (Pi) / `ai-edge-litert` hoặc `tensorflow` (dev) |
| Python | 3.10+ |

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/Pi: source .venv/bin/activate
pip install -r requirements.txt
```

Trên máy dev không có camera/UART thật, `pytest -q` vẫn chạy đầy đủ (mọi phần
phụ thuộc phần cứng dùng lazy import + fake injection).

## Configuration

Toàn bộ tham số vận hành nằm trong `config/settings.yaml` (§20 task.md — không
hard-code drop position / workspace / retry / timeout / threshold / FPS / model
path trong Python). `main.py` gọi `modules.config_validation.assert_valid_settings`
ở state `SELF_CHECK`; cấu hình sai → hệ thống đi thẳng `SAFE_STOP`, không bao giờ
vào `READY`. Các section bắt buộc: `camera`, `inference`, `calibration`,
`workspace`, `scheduler`, `robot`, `dedup`, `uart`, `runtime`, `logging`.

## Model preparation

Đặt model TFLite (YOLOv8n/YOLOv11n, INT8 hoặc float32/float16) vào `models/` và
trỏ `inference.model_path` trong `settings.yaml`. `modules/detector.py` tự nhận
diện 2 layout output (`[1, 4+num_classes, N]` raw hoặc `[N, 6]` NMS-embedded).
Xem `docs`/memory nội bộ về cách export/convert model & test detection trên máy
dev Windows (không có Pi) nếu cần — không lặp lại chi tiết ở đây.

## Model validation

**Bắt buộc** trước khi đổi model trên thiết bị thật:

```bash
python tools/validate_model.py --model models/yolov8n_waste_int8.tflite
```

Kiểm tra: file tồn tại, load được, input shape/dtype, output shape, số class
khớp `model_classes.json`, mọi class có mapping trong `class_mapping.json`,
confidence/NMS threshold hợp lệ, decoder tương thích layout output (chạy thử
decode). `WasteDetector.__init__` chạy lại đúng các kiểm tra input/output shape
này ngay khi `main.py` khởi động — model không tương thích không bao giờ vào
`READY` được (raise `ModelLoadError` tường minh).

## Calibration

**Bắt buộc phải chạy trước khi vận hành thật** — `config/calibration_matrix.json`
hiện tại trong repo vẫn là **placeholder identity matrix**, và runtime **chủ
động reject** nó (`CalibrationError`) trước khi vào `READY` (§7 task.md).

```bash
python tools/calibrate.py --source 0          # mở camera thật, click 4 điểm, nhập world mm
python tools/calibrate.py --image frame.jpg   # dùng ảnh tĩnh thay vì camera
python tools/calibrate.py --image frame.jpg --points-file points.json --out config/calibration_matrix.json  # non-interactive
```

Tool tính `cv2.findHomography`, reprojection error (RMSE), yêu cầu xác nhận
trước khi ghi đè, và lưu đúng schema:

```json
{
  "homography_matrix": [...],
  "source_points": [...],
  "destination_points_mm": [...],
  "reprojection_error_mm": 0.8,
  "image_width": 1920,
  "image_height": 1080,
  "created_at": "2026-01-01T00:00:00Z"
}
```

Sau khi calibration, validate độc lập bằng các điểm kiểm tra khác:

```bash
python tools/test_calibration.py --image frame.jpg
python tools/test_calibration.py --points-file check_points.json  # non-interactive
```

In mean/max/RMSE error (mm) và **reject** (`exit 1`) nếu vượt
`calibration.max_reprojection_error_mm` trong `settings.yaml` (mặc định 3.0mm —
đúng tiêu chí nghiệm thu "sai số ≤ ±3.0mm").

> Chế độ interactive (click chuột) của cả 2 tool cần `opencv-python` (không
> phải `opencv-python-headless`) — xem ghi chú trong `requirements.txt`.

## Development mode

`tools/dev_detect.py` chạy đúng logic decode/NMS của `WasteDetector` trên ảnh
tĩnh hoặc webcam, không cần phần cứng Pi:

```bash
python tools/dev_detect.py --model yolov8n.onnx --source samples/bus.jpg --labels coco --out samples/bus_out.jpg
python tools/dev_detect.py --model models/yolov8n_waste_int8.tflite --source 0 --labels config/model_classes.json
```

Hardware abstraction (§14): mọi thành phần chính của `main.AutoSortPi`
(`camera`, `detector`, `transformer`, `comm`) có thể dependency-inject —
`tests/test_integration_pipeline.py` chạy toàn bộ FSM với `FakeCamera`,
`FakeDetector`, `FakeTransformer` và `SerialCommunicator(ser=FakeSerial(...))`
(đóng vai "FakeRobot" phía MCU), không cần camera/model/UART thật.

## Testing

```bash
pip install pytest
pytest -q
```

124 test hiện tại, tất cả pass, phủ:

- decode detector (numpy thuần) + validate model spec (input/output shape, dtype, num_classes).
- transformer + validate calibration (reject identity/placeholder, reject vượt reprojection error threshold).
- scheduler (fragile priority, nearest-neighbor, tie-break xác định, workspace filter, batching).
- target validator, safety validator, object dedup (cooldown + disappearance confirmation).
- UART protocol (framing/checksum), handshake, retry, reconnect, phân loại lỗi (MCU/timeout/disconnect).
- camera stream (frame age, reconnect) bằng fake `cv2.VideoCapture`.
- config validation, log rotation theo tuổi file.
- `tools/validate_model.py`, `tools/calibrate.py`, `tools/test_calibration.py`, `tools/health.py`, `tools/benchmark.py`.
- **Integration test end-to-end** (`tests/test_integration_pipeline.py`): happy path
  fake frame → detect → mapping → calibration → schedule → target validation →
  fake UART → ACK → DONE → cycle success; và 8 failure mode: camera failure,
  detector failure, invalid calibration, invalid target, UART ACK timeout +
  retry, UART timeout hết retry, MCU error, DONE timeout.

## Benchmark

```bash
python tools/benchmark.py --model models/yolov8n_waste_int8.tflite --source samples/waste_test.jpg --iterations 50
```

Đo mean/p50/p95/p99/FPS cho camera capture, inference, coordinate transform,
scheduler, và peak RSS memory (Linux). **Chưa chạy trên Raspberry Pi 4 thật** —
số đo trên máy dev chỉ tham khảo về mặt logic đo lường, không phải con số
nghiệm thu cho mục tiêu `inference ≤ 90ms/frame`, `system ≥ 10 FPS`.

## Raspberry Pi deployment

```bash
sudo ./deploy/install.sh          # copy code, tạo venv, cài systemd unit
sudo systemctl enable --now auto-sort-pi
sudo journalctl -u auto-sort-pi -f
```

`deploy/auto-sort-pi.service`: auto start khi boot, `Restart=on-failure` với
backoff + `StartLimitBurst` (tránh restart-loop vô tận), graceful shutdown qua
`SIGTERM` (`main.py` bắt signal này và tự đóng camera/UART), working directory
+ environment file riêng, group `dialout`/`video` cho quyền UART/camera.

**Chưa test trên phần cứng Pi thật** — cấu hình được viết theo best-practice
systemd chuẩn nhưng chưa có lần triển khai thực tế nào để xác nhận.

## UART protocol

```text
Baudrate: 115200, 8N1, không flow control.

Command:  $CMD,GROUP_ID,X,Y,ANGLE*CHECKSUM\n   (CMD ∈ {PICK, SCAN, PURGE})
Response: !ACK\n | !DONE\n | !ERR,CODE\n
```

`CHECKSUM`: XOR (Hex 2 ký tự) của phần payload từ sau `$` đến trước `*`. Ví dụ
`$PICK,1,150.2,320.0,0.0*09\n` — checksum `09` khớp
`modules.uart_protocol.calc_checksum`.

Chính sách handshake (`modules/serial_comm.py`):

- Chờ `!ACK` (`uart.ack_timeout`); timeout → retry tối đa `uart.max_retries` lần (nghi mất gói).
- `!ERR` ở bất kỳ pha nào, hoặc timeout chờ `!DONE` (`uart.action_timeout`) → **không retry**, bỏ target hiện tại, tiếp tục target kế tiếp trong hàng đợi.
- Response không nhận dạng được (malformed/nhiễu) bị bỏ qua, tiếp tục chờ tới deadline.
- Mất kết nối serial khi ghi → tự thử mở lại cổng (`uart.max_reconnect_attempts`) trước khi báo lỗi.

## Troubleshooting

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| `SAFE_STOP` ngay khi khởi động, log `CalibrationError` | Chưa chạy `tools/calibrate.py` — file vẫn là identity placeholder |
| `SAFE_STOP`, log `ConfigError` | `settings.yaml` thiếu section/giá trị sai — xem chi tiết trong log (liệt kê từng lỗi) |
| `SAFE_STOP`, log `ModelLoadError` | Model input/output shape không khớp `num_classes`, hoặc `model_path` sai — chạy `tools/validate_model.py` để xem chi tiết |
| Không detect được gì dù model đúng | Kiểm tra `confidence_threshold` trong `settings.yaml`, hoặc dùng `tools/dev_detect.py` để debug decode trực tiếp |
| Robot không nhận lệnh | Kiểm tra `uart.port`, quyền truy cập serial (group `dialout`), dùng `tools/health.py` để test mở cổng |
| Cùng một vật bị gắp 2 lần | Kiểm tra `dedup.suppression_radius_mm`/`cooldown_seconds` — có thể vật di chuyển ra ngoài bán kính suppress |

## Production checklist

Theo acceptance criteria (§25 task.md) — trạng thái thực tế:

- [ ] Model TFLite thực tế đã huấn luyện + `validate_model.py` xác nhận hợp lệ (**chưa có model thật trong repo**)
- [ ] Calibration thực tế, sai số trong giới hạn (**chưa chạy — file hiện là placeholder**)
- [x] Workspace safety validation (`modules/safety_validator.py`, `modules/target_validator.py`)
- [x] Camera có reconnect (`modules/camera_stream.py`, test bằng fake `cv2.VideoCapture`)
- [x] UART có handshake + retry + recovery (`modules/serial_comm.py`)
- [x] Object dedup chống pick trùng (`modules/dedup.py`)
- [x] End-to-end integration test (`tests/test_integration_pipeline.py`, chạy bằng fake, **chưa chạy bằng hardware thật**)
- [ ] Benchmark trên Raspberry Pi 4 thật (**tool có sẵn (`tools/benchmark.py`), chưa chạy trên Pi**)
- [x] systemd deployment (`deploy/`, **chưa test trên Pi thật**)
- [x] Health check (`tools/health.py`)
- [x] Logging end-to-end theo `cycle_id`/`target_id`/... (`modules/logging_setup.py`, `main.py`)
- [x] `pytest -q` pass (124/124)
- [x] README phản ánh đúng trạng thái thực tế (tài liệu này)
- [x] Không còn dependency/config không dùng (đã rà soát — xem `requirements.txt`)
- [x] Không hard-code giá trị quan trọng trong source (toàn bộ qua `config/settings.yaml`)

## Implemented / Experimental / Hardware-dependent / TODO

**Implemented (software) + unit/integration-tested (fake/simulation):**
FSM đầy đủ 14 state, exception hierarchy phân loại lỗi, camera reconnect logic,
detector model validation, calibration validation, scheduler config-driven,
target validator, safety validator, object dedup, UART protocol/handshake/
reconnect, config validation, log rotation theo tuổi file, health check CLI,
benchmark CLI, validate_model CLI, calibrate/test_calibration CLI (chế độ
non-interactive `--points-file` đã test; chế độ interactive click-chuột chưa
test tự động được vì cần GUI thật).

**Hardware-dependent (chưa validate được vì thiếu phần cứng):**
Toàn bộ pipeline chạy với camera thật, model `.tflite` đã huấn luyện thật,
calibration thật (4 điểm mốc đo bằng thước), MCU/robot thật qua UART thật;
benchmark số đo trên Raspberry Pi 4; systemd service chạy thật trên Pi;
quyền truy cập `/dev/ttyAMA0`/camera qua group `dialout`/`video`.

**Experimental / cần theo dõi thêm khi có phần cứng:**
Giá trị mặc định `dedup.suppression_radius_mm`/`cooldown_seconds`,
`workspace`/`robot.safety` bounds, `scheduler.drop_positions_mm` trong
`settings.yaml` — đều là giá trị khởi điểm hợp lý, cần tinh chỉnh theo kích
thước bàn/robot thật.

**TODO:**
Post-action verification hiện chỉ là bookkeeping phần mềm (đánh dấu dedup
cooldown + `mark_processed`) — **chưa** capture lại camera để xác nhận trực
quan vật đã biến mất khỏi vị trí pick sau `DONE` (yêu cầu thêm 1 lần
capture+detect sau mỗi pick, chưa triển khai). `tools/calibrate.py`/
`test_calibration.py` interactive mode chưa có test tự động (giới hạn công cụ
test trong môi trường này, không có GUI).
