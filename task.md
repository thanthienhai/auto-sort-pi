Hãy làm việc trực tiếp trên repository `auto-sort-pi`.

Mục tiêu: tiếp tục phát triển project từ trạng thái software prototype hiện tại thành một hệ thống Auto Sort Pi có khả năng triển khai thực tế trên Raspberry Pi 4, camera cố định, mô hình YOLO/TFLite và MCU/robot qua UART.

## 1. Bắt buộc đọc và audit code hiện tại trước khi sửa

Đọc toàn bộ:

* `README.md`
* `main.py`
* `modules/camera_stream.py`
* `modules/detector.py`
* `modules/transformer.py`
* `modules/scheduler.py`
* `modules/serial_comm.py`
* `modules/logging_setup.py`
* `config/settings.yaml`
* `config/class_mapping.json`
* `config/model_classes.json`
* `config/calibration_matrix.json`
* `tests/*`
* `tools/dev_detect.py`

Không được rewrite toàn bộ project nếu không cần thiết.

Giữ lại các thành phần đang hoạt động tốt và chỉ refactor khi có lý do kỹ thuật rõ ràng.

## 2. Mục tiêu kiến trúc cuối

Thiết kế pipeline rõ ràng:

```text
Camera
  ↓
Frame Capture
  ↓
Preprocessing
  ↓
Object Detection
  ↓
Detection Validation
  ↓
Class Mapping
  ↓
Pixel → World
  ↓
Target Validation
  ↓
Object Deduplication / Tracking
  ↓
Scheduler
  ↓
Safety Check
  ↓
UART Command
  ↓
ACK
  ↓
DONE / ERR
  ↓
Post-action verification
  ↓
Next target
```

Main application phải điều phối pipeline này bằng FSM rõ ràng.

## 3. Hoàn thiện FSM

Refactor FSM thành một state machine production-oriented.

Tối thiểu cần:

```text
BOOT
SELF_CHECK
INITIALIZING
READY
SCANNING
SCHEDULING
VALIDATING_TARGET
EXECUTING
WAITING_ACK
WAITING_DONE
VERIFYING_RESULT
RECOVERING
ERROR
SAFE_STOP
SHUTDOWN
```

Không để lỗi một frame hoặc một target làm process chết.

Phân biệt rõ:

* lỗi camera;
* lỗi model;
* lỗi calibration;
* lỗi target;
* lỗi UART;
* lỗi MCU;
* timeout;
* lỗi cấu hình.

Có recovery policy riêng cho từng nhóm lỗi.

## 4. Camera subsystem

Hoàn thiện `CameraStream`.

Yêu cầu:

* threaded capture;
* latest-frame buffer;
* FPS configuration thực sự được áp dụng;
* timestamp cho frame;
* frame age validation;
* phát hiện camera disconnected;
* tự reconnect;
* giới hạn retry;
* graceful shutdown;
* logging.

Không cho inference xử lý frame quá cũ.

Mỗi frame phải có:

```text
timestamp
frame_id
capture_latency
age
```

## 5. Detector subsystem

Giữ decoder hiện tại nhưng nâng cấp thành adapter ổn định.

Yêu cầu:

* hỗ trợ TFLite;
* giữ lazy runtime;
* validate model input shape;
* validate dtype;
* validate output shape;
* validate number of classes;
* validate model class mapping;
* confidence threshold configurable;
* NMS threshold configurable;
* input preprocessing configurable;
* latency measurement.

Không được silently accept output shape không đúng.

Nếu model không tương thích:

```text
raise explicit error
```

với thông tin đầy đủ.

Phải phân biệt:

```text
model loading error
model inference error
model decoding error
```

## 6. Model validation

Tạo một module/tool riêng để kiểm tra model trước khi deploy.

Ví dụ:

```bash
python tools/validate_model.py --model ...
```

Tool phải kiểm tra:

* file tồn tại;
* TFLite model load được;
* input shape;
* input dtype;
* output shapes;
* số class;
* class mapping;
* confidence range;
* decoder compatibility.

Xuất report dễ đọc.

Không cho `main.py` chạy model sai configuration.

## 7. Calibration subsystem

Đây là phần bắt buộc phải hoàn thiện.

Không sử dụng identity matrix như calibration thật.

Xây dựng tool:

```bash
python tools/calibrate.py
```

Tool cần cho phép:

1. mở camera;
2. hiển thị frame;
3. chọn 4 điểm pixel;
4. nhập 4 điểm world tương ứng tính bằng mm;
5. tính Homography bằng OpenCV;
6. tính reprojection error;
7. hiển thị kết quả;
8. yêu cầu xác nhận;
9. lưu `config/calibration_matrix.json`.

File JSON nên chứa:

```json
{
  "homography_matrix": [...],
  "source_points": [...],
  "destination_points_mm": [...],
  "reprojection_error_mm": ...,
  "image_width": ...,
  "image_height": ...,
  "created_at": ...
}
```

Runtime phải kiểm tra calibration hợp lệ trước khi READY.

## 8. Calibration validation

Tạo:

```bash
python tools/test_calibration.py
```

Cho phép click một số điểm trên bàn và kiểm tra:

```text
pixel
→ world
```

Hiển thị tọa độ.

Tính:

```text
mean error
max error
RMSE
```

Reject calibration nếu sai số vượt threshold cấu hình.

## 9. Scheduler

Refactor scheduler thành module có configuration rõ ràng.

Không hard-code drop positions trong Python.

Đưa toàn bộ vào:

```yaml
scheduler:
  start_position_mm: [...]
  fragile_group_id: 4

  drop_positions_mm:
    1: [...]
    2: [...]
    3: [...]
    4: [...]
```

Scheduler phải hỗ trợ:

* priority;
* fragile priority;
* nearest neighbor;
* drop-position transition;
* deterministic ordering;
* tie breaking;
* maximum workspace;
* optional batching.

Phải giải thích rõ trong code:

```text
robot position hiện tại là pick position
hay drop position
```

không để semantics mơ hồ.

Viết test đầy đủ.

## 10. Object deduplication

Đây là chức năng bắt buộc.

Tránh tình trạng cùng một object bị detect lại trong nhiều scan cycle.

Có thể dùng:

* spatial exclusion;
* object tracking;
* cooldown;
* disappearance confirmation.

Thiết kế đơn giản, nhẹ, phù hợp Raspberry Pi.

Ví dụ:

```text
target A
pick
↓
robot DONE
↓
scan
↓
object A vẫn còn
↓
không lập tức pick lại
```

Cần có logic xác nhận object đã biến mất hoặc target đã được xử lý.

## 11. Target validation layer

Trước khi scheduler hoặc UART sử dụng target, phải validate:

```text
group_id hợp lệ
confidence hợp lệ
x/y finite
x/y trong workspace
x/y không phải NaN
x/y không vượt boundary
target chưa processed
target không stale
```

Tạo module riêng, ví dụ:

```text
modules/target_validator.py
```

## 12. Robot safety layer

Trước khi gửi PICK:

```text
SafetyValidator
```

phải kiểm tra:

```text
X_MIN <= X <= X_MAX
Y_MIN <= Y <= Y_MAX
```

và các giới hạn khác.

Nếu vi phạm:

```text
không được gửi UART
```

Phải log lý do.

Không cho AI model trực tiếp điều khiển robot nếu target chưa qua safety validation.

## 13. UART subsystem

Giữ protocol hiện tại nhưng refactor cho rõ ràng hơn.

Phải hỗ trợ:

```text
PICK
SCAN
PURGE
```

Frame:

```text
$PICK,GROUP_ID,X,Y,ANGLE*CHECKSUM
```

Response:

```text
!ACK
!DONE
!ERR,CODE
```

Yêu cầu:

* retry ACK timeout;
* không retry khi MCU trả ERR;
* timeout;
* malformed response;
* unexpected response;
* UART disconnect;
* reconnect;
* flush/reset buffer;
* logging đầy đủ.

Tạo protocol parser riêng nếu cần.

Không để business logic phụ thuộc trực tiếp vào `serial.Serial`.

## 14. Hardware abstraction

Cho phép:

```text
FakeCamera
FakeDetector
FakeSerial
FakeRobot
```

để integration test không cần phần cứng.

Main controller phải dependency-inject được các module chính.

## 15. Integration test

Viết test mô phỏng toàn bộ pipeline:

```text
fake frame
→ fake detection
→ mapping
→ calibration
→ scheduling
→ target validation
→ fake UART
→ ACK
→ DONE
→ cycle success
```

Test cả:

```text
camera failure
detector failure
invalid calibration
invalid target
UART timeout
UART retry
MCU error
DONE timeout
```

## 16. Performance benchmark

Tạo:

```bash
python tools/benchmark.py
```

Benchmark:

```text
camera capture
preprocess
inference
decode
NMS
coordinate transform
scheduler
```

Xuất:

```text
mean
p50
p95
p99
FPS
memory
```

Mục tiêu ban đầu:

```text
inference <= 90 ms/frame
system >= 10 FPS
```

nhưng phải đo trên Raspberry Pi thực tế.

Không claim đạt performance nếu chưa benchmark.

## 17. Raspberry Pi deployment

Chuẩn bị deployment production.

Tạo:

```text
deploy/
```

và hỗ trợ:

```text
systemd service
```

Ví dụ:

```text
auto-sort-pi.service
```

Yêu cầu:

* auto start;
* restart on failure;
* graceful shutdown;
* logs;
* working directory;
* environment;
* model/config path rõ ràng.

## 18. Health check

Thêm runtime health monitor:

```text
camera
model
calibration
UART
last frame
last inference
last successful pick
last error
```

Có thể expose CLI:

```bash
python tools/health.py
```

Output:

```text
Camera: OK
Model: OK
Calibration: OK
UART: OK
System: READY
```

## 19. Logging

Giữ `request_id` hiện tại.

Bổ sung các field có cấu trúc:

```text
cycle_id
frame_id
target_id
group_id
class
confidence
pixel_x
pixel_y
world_x
world_y
scheduler_distance
uart_attempt
uart_status
latency_ms
error_code
```

Mọi cycle phải truy vết được end-to-end.

Thêm log rotation nếu cần để tránh disk đầy.

## 20. Configuration

Đưa các giá trị hard-code ra configuration:

```yaml
camera:
inference:
calibration:
workspace:
scheduler:
robot:
uart:
runtime:
logging:
```

Không hard-code:

* drop position;
* workspace;
* retry;
* timeout;
* thresholds;
* FPS;
* model path.

Validate configuration khi startup.

Nếu configuration invalid → system không được vào READY.

## 21. Documentation

Cập nhật README theo trạng thái thực tế.

Không mô tả feature chưa triển khai như feature đã hoàn thành.

README cần có:

```text
Architecture
Requirements
Installation
Configuration
Model preparation
Model validation
Calibration
Development mode
Testing
Benchmark
Raspberry Pi deployment
UART protocol
Troubleshooting
Production checklist
```

Tách rõ:

```text
Implemented
Experimental
Hardware-dependent
TODO
```

## 22. Code quality

Sau khi thực hiện:

```bash
pytest -q
```

phải pass toàn bộ test.

Kiểm tra:

* type hints;
* error handling;
* imports;
* dead code;
* unused config;
* duplicated logic;
* resource leaks;
* thread shutdown;
* UART cleanup.

Không thêm dependency nếu không cần.

Ưu tiên thư viện nhẹ vì target là Raspberry Pi 4.

## 23. Không được phá những phần đang tốt

Đặc biệt giữ lại và cải thiện:

* NumPy-based decoder;
* NMS hiện tại;
* lazy hardware imports;
* UART fake injection;
* structured logging;
* unit-test architecture;
* `tools/dev_detect.py`.

Không thay detector bằng framework nặng hơn nếu không có lợi ích rõ ràng.

## 24. Thứ tự triển khai

Thực hiện theo thứ tự:

### Phase 1 — Audit + cleanup

* audit code;
* sửa configuration;
* sửa những bug rõ ràng;
* đảm bảo test hiện tại pass.

### Phase 2 — Runtime robustness

* FSM;
* validation;
* recovery;
* camera reconnect;
* UART reconnect;
* model validation.

### Phase 3 — Calibration

* calibration tool;
* calibration validation;
* workspace validation.

### Phase 4 — Scheduler + object lifecycle

* drop positions config;
* safety;
* deduplication;
* processed-target lifecycle.

### Phase 5 — Integration test

* end-to-end fake pipeline;
* failure injection.

### Phase 6 — Hardware deployment

* Raspberry Pi;
* real camera;
* real model;
* real UART;
* real MCU.

### Phase 7 — Benchmark + reliability

* FPS;
* latency;
* memory;
* long-running test;
* error recovery.

### Phase 8 — Productionization

* systemd;
* watchdog;
* log rotation;
* health check;
* final documentation.

## 25. Acceptance criteria

Project chỉ được coi là hoàn thành khi:

1. Có model TFLite thực tế và `validate_model.py` xác nhận hợp lệ.
2. Có calibration thực tế và sai số nằm trong giới hạn cho phép.
3. Có workspace safety validation.
4. Camera có reconnect.
5. UART có handshake + retry + recovery.
6. Không xảy ra duplicate pick do detection lặp.
7. Có end-to-end integration test.
8. Có benchmark trên Raspberry Pi 4.
9. Có systemd deployment.
10. Có health check.
11. Có logging end-to-end.
12. `pytest -q` pass.
13. README phản ánh đúng trạng thái thực tế.
14. Không còn dependency/configuration không được sử dụng.
15. Không có giá trị quan trọng bị hard-code trong source.

## 26. Quy tắc làm việc

Trước mỗi nhóm thay đổi:

1. Đọc code hiện tại.
2. Xác định vấn đề.
3. Sửa tối thiểu nhưng đúng kiến trúc.
4. Viết test trước hoặc cùng lúc.
5. Chạy test.
6. Kiểm tra regression.
7. Cập nhật documentation.

Không được chỉ viết code mà không test.

Không được đánh dấu feature "done" chỉ vì code đã compile.

Phải phân biệt rõ:

```text
implemented in software
validated with simulation
validated with real hardware
validated in production
```

Cuối cùng tạo một báo cáo:

```text
DONE
PARTIAL
TODO
BLOCKED BY HARDWARE
RISKS
NEXT STEPS
```

kèm danh sách file đã sửa và lý do sửa từng file.
