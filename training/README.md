# Training / Dataset Pipeline — auto-sort-pi

Pipeline độc lập với runtime (`modules/`, `main.py`, FSM, UART, safety,
calibration) để: thu thập public dataset waste/recycling, chuẩn hóa annotation
về YOLO, phân tích/ánh xạ class an toàn vào taxonomy 16 class chính thức, xây
review queue cho nhãn mơ hồ, dedup, group-aware split, build dataset version
reproducible, và train/evaluate baseline detector (YOLOv8n/YOLO11n).

Yêu cầu gốc: xem `training_online_dataset.md` ở repo root.

## 1. Taxonomy

Nguồn sự thật LÀ repository — **không duplicate** ở đây:

- `config/model_classes.json` — 16 tên class, thứ tự = class_id (0..15).
- `config/class_mapping.json` — class name -> sorting group id (1=METAL,
  2=PAPER, 3=PLASTIC, 4=GLASS).
- `training/configs/taxonomy.yaml` — chỉ phần *không* có trong runtime config:
  tên nhóm, ngưỡng coverage, "hard class" priority.

Load bằng `training.lib.taxonomy.load_taxonomy()`. `training/lib/taxonomy.py`
raise `TaxonomyError` nếu 2 file config không nhất quán (sai số lượng class,
thiếu group mapping, ...) — không bao giờ tự ý đổi class ID.

## 2. Source datasets

7 dataset Roboflow trong `training_online_dataset.md` §3 đã có manifest sẵn
trong `training/datasets/sources/<name>/manifest.yaml`:

| name | phase | status |
|---|---|---|
| roboflow_trash_detection | 1 | not_downloaded |
| roboflow_new_trash_detection | 1 | not_downloaded |
| roboflow_yolov7_trash_v7 | 1 | not_downloaded |
| roboflow_general_waste_data | 1 | not_downloaded |
| roboflow_solidwaste | 1 | not_downloaded |
| roboflow_waste_detection_tetrapak | 1 | not_downloaded |
| roboflow_waste_detection_suthida | 2 | not_downloaded |
| bootstrap_manual | - | imported |

**Chưa dataset public nào được tải về.** Roboflow universe yêu cầu đăng nhập
hoặc `ROBOFLOW_API_KEY` để export (§8) — pipeline không tự động tải (không
hard-code API key, không đăng nhập hộ). Muốn dùng dataset thật:

1. Export dataset đó thủ công từ trang Roboflow (chọn format **YOLOv8**), tải
   ZIP về máy.
2. `python -m training.cli import --source <zip> --name <name>`
   (manifest.yaml đã có sẵn `target_mapping` — script sẽ tự dùng).
3. Thêm `<name>` vào `training/configs/dataset_sources.yaml` rồi chạy
   `build_dataset.py`.

`bootstrap_manual` là dataset nội bộ (không phải public) dùng để smoke-test
toàn bộ pipeline + chạy thử training GPU khi chưa có dataset thật — xem mục 12.

Thêm dataset mới bất kỳ (không giới hạn 7 dataset trên, §3.H): tạo
`training/datasets/sources/<name>/manifest.yaml` theo schema ở mục 4, rồi
`import_dataset.py`.

## 3. Dataset format hỗ trợ (§7)

`training/lib/formats.py` tự phát hiện format (`detect_format`), không giả
định dataset nào cũng là YOLO:

- **yolo** — `images/` + `labels/*.txt` + `data.yaml` hoặc `classes.txt`.
- **coco** — JSON với `images`/`annotations`/`categories`.
- **voc** — Pascal VOC XML (`<annotation>`).
- **generic_json** — format nội bộ: `<ảnh>.jpg` cạnh `<ảnh>.json` chứa
  `[{"label": str, "bbox_xyxy": [x0,y0,x1,y1]}]`. Đây cũng là format dùng cho
  **ảnh camera thực tế sau này** (§1.10) — chỉ cần export cùng format này là
  vào chung pipeline, không cần code riêng.
- ZIP tự động được giải nén trước khi detect.

## 4. Class mapping — 3 trạng thái (§4)

`training/lib/mapping.py` (`MappingEngine`) + `training/configs/dataset_mapping.yaml`
(bảng global) + `target_mapping` riêng trong từng `manifest.yaml` (ưu tiên cao
hơn bảng global).

| Trạng thái | Ý nghĩa | Vào training set? |
|---|---|---|
| `EXACT` | label trùng đúng tên 1 class taxonomy | Có |
| `SAFE_ALIAS` | label khác tên nhưng đã xác nhận an toàn | Có |
| `REVIEW_REQUIRED` | mơ hồ, không đủ thông tin | **Không** — vào review queue |

Thứ tự ưu tiên khi resolve 1 label: `manifest.target_mapping` của dataset đó
→ `dataset_mapping.yaml` (global) → auto-EXACT (trùng tên class) → mặc định
`REVIEW_REQUIRED` (không rõ thì không tự merge).

Bảng mapping đầy đủ: xem `training/configs/dataset_mapping.yaml`. Trích một số
quyết định quan trọng:

| Source label | Target class | Type | Confidence | Lý do |
|---|---|---|---|---|
| Pet Bottle / PET Bottle / PET_Bottle | pet_bottle | SAFE_ALIAS | 0.95 | tên khác của pet_bottle |
| Plastic Bottle | pet_bottle | SAFE_ALIAS | 0.85 | giả định phổ biến nhất |
| Tetra Pack/Pak/-Pak/TetraPak | tetra_pak | SAFE_ALIAS | 0.95 | chính tả khác nhau |
| Glass Bottle | glass_bottle | EXACT/SAFE_ALIAS | 0.95-1.0 | |
| Glass Jar | glass_jar | EXACT/SAFE_ALIAS | 0.95-1.0 | |
| Plastic Bag | nylon_bag | SAFE_ALIAS | 0.9 | taxonomy dùng tên nylon_bag |
| Aluminium Foil | aluminum_foil | SAFE_ALIAS | 0.95 | chính tả Anh-Anh |
| **Can / Metal Can / Drink Can / Food Can / Aluminium Can / Tin Can** | — | **REVIEW_REQUIRED** | 0.5 | có thể là nhôm hoặc thiếc, không đủ chắc chắn — **kể cả "Tin Can"** dù gần tên `tin_can`, theo đúng yêu cầu §4 |
| **Glass** (bare) | — | **REVIEW_REQUIRED** | 0.3 | quá chung, phải tách bottle/jar/broken |
| **Paper / Paperboard / Paper Packaging** | — | **REVIEW_REQUIRED** | 0.3-0.4 | không được tự động coi cardboard/paper_box/newspaper là 1 |
| PAPER/PLASTIC/GLASS/METAL (tên nhóm) | — | REVIEW_REQUIRED | 0.2 | tên nhóm sorting, không phải class cụ thể |

## 5. Review queue (§11)

Instance bị `REVIEW_REQUIRED` **không** vào normalized dataset. Chúng được:

- Ghi 1 dòng vào `training/datasets/review/review_manifest.jsonl` (schema:
  `image, source_dataset, source_class, candidate_targets, reason, status,
  bbox`).
- Copy ảnh gốc vào `training/datasets/review/<candidate_class_đầu_tiên>/`.

Người dùng review thủ công, cập nhật `status` (`pending` → `accepted`/
`rejected`), rồi có thể bổ sung `target_mapping` cụ thể cho dataset đó và
import lại.

## 6. Import pipeline (§7)

```
python -m training.cli import --source <path_or_zip> --name <name> [--source-type auto|yolo|coco|voc|generic_json] [--dry-run]
```

`source -> detect format -> đọc annotation -> resolve mapping (per box) ->
normalize ảnh (EXIF, decode-check, sha256/phash) -> ghi accepted box vào
`training/datasets/normalized/<name>/{images,labels,metadata.jsonl}` -> ghi
box REVIEW_REQUIRED vào review queue -> cập nhật `manifest.yaml` (image_count,
original_classes thực tế quan sát được, status=imported)`.

Ảnh lỗi/corrupt/zero-byte bị loại và ghi vào `warnings` trong summary, KHÔNG
raise lỗi toàn bộ import.

## 7. Validation (§9, §10)

```
python -m training.cli normalize --input training/datasets/normalized/<name> [--dry-run]
python -m training.cli validate  --input training/datasets/normalized/<name>
```

`validate_dataset` kiểm tra ảnh (decode, zero-byte) + annotation (class_id
range, NaN/inf, x/y/w/h range, bbox vượt ảnh, bbox quá nhỏ, duplicate bbox,
malformed line) và exit code != 0 nếu có lỗi (dùng được trong CI).

## 8. Deduplication (§15)

```
python -m training.cli dedup --input training/datasets/normalized/<name> [--phash-threshold 4] [--dry-run]
```

SHA256 (exact) bắt buộc + perceptual hash (`imagehash.phash`, near-duplicate,
Hamming distance <= threshold) nếu thư viện có sẵn. Union-find gộp group,
`keep` = tên nhỏ nhất theo alphabet (deterministic). **Không xóa file** — chỉ
ghi `duplicates.json` (schema `group_id/members/keep/duplicates`).

## 9. Split (§16, §17)

```
python -m training.cli split --input training/datasets/normalized/<name> --seed 42 [--ratios 0.7,0.2,0.1] [--dry-run]
```

Group-aware: nếu metadata của ảnh có `sequence_id`/`scene_id`/`group_key`/
`capture_session` (frame từ cùng 1 video/scene — dành cho ảnh camera thực tế
sau này), mọi ảnh cùng group chỉ rơi vào **1** split. Ảnh public dataset rời
rạc (không có metadata này) mặc định mỗi ảnh là 1 group riêng. Deterministic:
group -> bucket bằng `sha256(seed:group_key)`, không phụ thuộc thời gian chạy.
`verify_no_leakage` raise lỗi nếu phát hiện leakage.

## 10. Build dataset version (§24, §25, §26)

```
python -m training.cli build \
    --config training/configs/dataset_sources.yaml \
    --output training/datasets/versions/v0.1-public-baseline \
    [--seed 42] [--ratios 0.7,0.2,0.1] [--dry-run]
```

Gộp mọi dataset trong `dataset_sources.yaml` → dedup toàn cục → validate
annotation → group-aware split → copy vào `images/{train,val,test}` +
`labels/{train,val,test}` → ghi `data.yaml` (YOLO-ready, names 0..15 đúng
taxonomy) + `metadata/{manifest,stats,source_manifest,class_mapping,
duplicates,review,validation_report}.json` + `README.md` per-version.

**Deterministic**: cùng input + cùng config + cùng seed ⇒ cùng composition
(test: `tests/test_training_build_dataset.py::test_build_dataset_is_deterministic`).
Không phụ thuộc timestamp (chỉ ghi `created_at_utc` làm thông tin).

`--dry-run` in kế hoạch đầy đủ (sources, classes, image count, duplicate
count, ambiguous label count, warnings) mà không ghi file nào.

## 11. Dataset report (§12, §13, §14)

```
python -m training.cli report --input training/datasets/versions/v0.1-public-baseline
```

Sinh `dataset_report.{json,csv,md}`: total images/instances, bbox area
distribution, resolution/source/license distribution, **class coverage đầy đủ
16 class** (images/instances/%/status MISSING|VERY_LOW|LOW|MODERATE|GOOD theo
ngưỡng configurable trong `taxonomy.yaml`), và cờ `hard_class_priority` cho 10
class khó (§14: aluminum_foil, metal_cap, paper_cup, tetra_pak, paper_box,
newspaper, detergent_bottle, nylon_bag, glass_jar, broken_glass).

## 12. Milestone M0 hiện tại — smoke test, KHÔNG PHẢI baseline thật

**Chưa có public dataset nào được tải về** (giới hạn của môi trường: Roboflow
universe yêu cầu đăng nhập/API key để export, xem mục 2). Để verify toàn bộ
pipeline hoạt động đúng đầu-cuối (import → mapping → review → normalize →
validate → dedup → split → build → **train trên GPU** → evaluate) ngay trong
milestone này, dataset `bootstrap_manual` được tạo từ 2 ảnh có sẵn trong
`samples/` (không phải camera thật, không phải public dataset):

- `waste_composite_topdown.jpg` + ground truth có sẵn trong repo — 4 object
  gốc, trong đó 2 object (`nhua_pet_bottle` → pet_bottle, `thuytinh_glass_bottle`
  → glass_bottle) được accept, 2 object (`kim_loai_can`, `giay_cardboard_box`)
  bị đưa vào review vì mơ hồ — **đúng theo §4/§11**.
- 1 ảnh negative (`zidane.jpg`, không có object thuộc taxonomy).

`training/datasets/versions/v0.1-public-baseline/` build từ nguồn này (lệnh ở
mục 10, với `--seed 1 --ratios 0.5,0.5,0` để dataset 2 ảnh có cả train lẫn
val) **chỉ để chứng minh pipeline chạy được** — 2 ảnh, 2 instance, không đủ để
huấn luyện model có ý nghĩa. `training.cli train` đã chạy thử thành công
trên GPU (GTX 1060, xem mục 13) với dataset này — loss/mAP ≈ 0 là kỳ vọng
đúng với 1 ảnh train (ngẫu nhiên rơi vào group này lại là ảnh negative).

**Baseline thật cần PHASE 1 public dataset** (§39): YoloV7 Trash Dataset V7,
Trash Detection, General waste data, solidwaste, Waste Detection (tetra pak).
Sau khi export thủ công 5 dataset này (mục 2) và `import_dataset.py` từng cái,
thêm tên vào `dataset_sources.yaml` rồi `build_dataset.py` lại — không cần
sửa code.

## 13. Training baseline (§27) — đã verify chạy được trên GPU

Cấu hình: `training/configs/training.yaml` (model, imgsz, epochs, batch,
device, workers, seed — **không hard-code** trong `train.py`).

```
python -m training.cli train --config training/configs/training.yaml
```

Yêu cầu `torch` + `ultralytics` (không nằm trong `requirements.txt` runtime —
xem `requirements-training.txt`, chỉ cần trên máy dev/training).

**Lưu ý GPU trên máy dev này (GTX 1060, compute capability 6.1/Pascal)**: bản
PyTorch mới nhất (`cu128`) đã **bỏ** kernel cho `sm_61` — `torch.cuda.is_available()`
vẫn trả `True` nhưng mọi phép tính CUDA sẽ lỗi "no kernel image is available".
Phải dùng build `cu118` (driver NVIDIA tương thích ngược với runtime CUDA cũ
hơn):

```
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1+cu118 torchvision==0.22.1+cu118
```

Đã verify: `torch.cuda.is_available() == True`, matmul trên GPU chạy được, và
`train.py`/`evaluate.py` chạy full vòng train + validate trên GPU 0 (GTX 1060)
với dataset `v0.1-public-baseline`, sinh checkpoint `best.pt`/`last.pt` trong
`training/runs/<name>/weights/`.

`ultralytics` kéo theo `opencv-python` (bản GUI) làm dependency — nó ghi đè
cùng namespace `cv2` với `opencv-python-headless` mà runtime dùng. Xem cảnh
báo + cách xử lý trong `requirements-training.txt`.

## 14. Evaluation (§28)

```
python -m training.cli evaluate --weights training/runs/<name>/weights/best.pt --data <data.yaml> --split test
```

Output: mAP50, mAP50-95, precision/recall/F1 trung bình, **per-class**
precision/recall/AP50/AP50-95 (map đúng qua `ap_class_index` — không giả định
mọi class đều xuất hiện trong split), `low_recall_classes` (5 class recall
thấp nhất — hệ thống sorting cần ưu tiên recall) và tham chiếu confusion
matrix theo nhóm (`confusion_matrix.png/.csv` trong thư mục run của
ultralytics).

## 15. Export & Active learning (§29-§33) — interface, chưa bắt buộc milestone này

- `tools/validate_model.py` (đã có, validate `.tflite` trước deploy) là đích
  cuối sau khi export — training pipeline này KHÔNG tự export `.tflite` trong
  milestone M0, chỉ chuẩn bị interface train/evaluate/export/validate.
- `training/cli.py` (subcommand `auto-label`) — pseudo-label ảnh mới bằng model đã train,
  3 ngưỡng configurable (`accept_threshold`, `review_threshold` trong
  `training.yaml`): `>= accept` → auto_labels/, `[review, accept)` → review/,
  `< review` → uncertain/ (đây cũng chính là hạt giống của active-learning
  queue ở §31 — `uncertain/` tương ứng). Không tự động đưa pseudo-label vào
  final training set.
- Hard negative mining (§33) và representative dataset cho INT8 (§30): chưa
  có script riêng ở milestone này — chuẩn bị bằng metadata `domain` +
  `source_weight` đã có sẵn trong mọi manifest/metadata record, sẵn sàng cho
  bước lọc sau.

## 16. Tests (§36)

```
python -m pytest tests/test_training_*.py -q
```

Bao phủ: taxonomy loading + reject sai class count/thiếu mapping, class ID
0-15 giữ nguyên, exact/safe_alias/review_required mapping (kể cả "Tin Can"
buộc REVIEW_REQUIRED), YOLO parser, COCO→YOLO, VOC→YOLO, bbox validation
(NaN/inf/range/quá nhỏ/vượt ảnh), duplicate bbox, SHA256+phash dedup
(deterministic keep), group-aware split (không leakage), manifest/review
manifest roundtrip, **deterministic build** + **data.yaml generation**
(`test_training_build_dataset.py`).

## 17. Không phá runtime (§37)

Package `training/` độc lập hoàn toàn — không import từ `modules/`, không sửa
`main.py`/FSM/scheduler/UART/safety/calibration/`config/settings.yaml`. Toàn
bộ 124 test runtime hiện có (`tests/test_detector.py`, ...) vẫn pass không
thay đổi sau khi thêm pipeline này. Duy nhất `.gitignore` được bổ sung thêm
entry cho `training/datasets/{raw,normalized,review,final,versions,representative}/*`
và `training/runs/` (không commit ảnh/annotation/checkpoint nhị phân).

## 18. Xung đột phát hiện trong spec gốc (§42)

`training_online_dataset.md` §24 liệt kê tên file metadata cho mỗi version
(`manifest.json, dataset_stats.json, classes.yaml, sources.json, mapping.json,
README.md`) khác với §26 (`manifest.json, stats.json, source_manifest.json,
class_mapping.json, duplicates.json, review.json`) — không nhất quán về tên
file. **Quyết định**: dùng đúng bộ tên trong §26 (được liệt kê cụ thể hơn,
gắn trực tiếp với cấu trúc `training/datasets/versions/<v>/` mà §26 mô tả) và
bổ sung thêm `validation_report.json` (theo yêu cầu milestone M0 §40: "Dataset
phải có ... validation report"). `README.md` per-version (§24) vẫn được tạo
tại gốc thư mục version (không phải trong `metadata/`).
