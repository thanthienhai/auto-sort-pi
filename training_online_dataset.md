# TASK: Xây dựng Public Dataset Acquisition + Dataset Normalization + Training Preparation cho auto-sort-pi

## 1. Bối cảnh project

Repository:
    thanthienhai/auto-sort-pi

Mục tiêu hiện tại:
- Chưa cần đưa ảnh thực tế từ camera vào ngay.
- Trước mắt xây dựng pipeline để:
  1. tìm/thu thập các public dataset về waste/recycling;
  2. tải về/localize dataset;
  3. phân tích class;
  4. mapping class của dataset nguồn vào taxonomy chính thức của project;
  5. lọc ảnh/annotation không phù hợp;
  6. deduplicate;
  7. chuẩn hóa annotation về YOLO;
  8. tạo dataset version;
  9. chuẩn bị dataset baseline để training;
  10. sau này chỉ cần đưa ảnh thực tế vào là có thể nhập vào cùng pipeline.

Không triển khai việc thu thập ảnh thực tế ở giai đoạn đầu, nhưng architecture phải mở rộng được cho việc đó.

---

# 2. Taxonomy chính thức của auto-sort-pi

PHẢI sử dụng đúng thứ tự class ID hiện tại của repository.

16 classes:

0  can
1  tin_can
2  aluminum_foil
3  metal_cap
4  cardboard
5  paper_cup
6  tetra_pak
7  paper_box
8  newspaper
9  pet_bottle
10 plastic_cup
11 detergent_bottle
12 nylon_bag
13 glass_bottle
14 glass_jar
15 broken_glass

Không được tự ý đổi ID.

Mapping 4 nhóm sorting:

GROUP 1 - METAL
    can
    tin_can
    aluminum_foil
    metal_cap

GROUP 2 - PAPER
    cardboard
    paper_cup
    tetra_pak
    paper_box
    newspaper

GROUP 3 - PLASTIC
    pet_bottle
    plastic_cup
    detergent_bottle
    nylon_bag

GROUP 4 - GLASS
    glass_bottle
    glass_jar
    broken_glass

Nguồn sự thật:
- config/model_classes.json
- config/class_mapping.json

Không duplicate taxonomy ở nhiều file nếu không cần thiết.
Hãy load taxonomy từ config hiện tại khi có thể.

---

# 3. Public datasets cần hỗ trợ

Thiết kế pipeline generic, nhưng ban đầu phải hỗ trợ các dataset/candidate sau:

### A. Roboflow Trash Detection
URL:
https://universe.roboflow.com/search?q=class%3Aaluminium-foil

Dataset có các class đáng chú ý:
- Aluminium foil
- Broken glass
- Clear plastic bottle
- Corrugated carton
- Disposable plastic cup
- Drink can
- Drink carton
- Food Can
- Glass/Glass bottle
...

### B. Roboflow new_trash_detection
Dataset khoảng 3k ảnh, nhiều class waste.
Các class có thể tận dụng:
- Aluminium foil
- Broken glass
- Bottle
- Can
- Corrugated carton
- Disposable plastic cup
- Drink can
- Drink carton
...

### C. Roboflow YoloV7 Trash Dataset V7
URL:
https://universe.roboflow.com/yolov7-trash-detection-group/yolov7-trash-dataset-v7-05-06-2023

Các class hữu ích:
- Cardboard
- Plastic Bottle
- Plastic Bag
- Glass Bottle
- Paper Cup
- Plastic Cup
- Paperboard
- Plastic Wrapper
- Cans
- Tetra Pack / Tetra Pak / Tetra-Pak / TetraPak
- Colored Glass Bottles

License được ghi trên dataset page: CC BY 4.0.

### D. Roboflow General waste data
URL:
https://universe.roboflow.com/mcl2/general-waste-data

Có nhiều class gần taxonomy:
- PAPER
- PLASTIC
- GLASS
- METAL
- CARDBOARD
- PLASTIC bottle
- PLASTIC bag
- GLASS bottle
- PAPER cup
- GLASS jar
- PLASTIC cup
- METAL cap
- GLASS broken

### E. Roboflow solidwaste
URL:
https://universe.roboflow.com/solidwaste-dc21q/solidwaste-cbyxk

Các class hữu ích:
- cardboard
- foil
- glass bottle jar
- paper loose
- paper packaging
- single use cup
- tetrapak
- plastic soft flexible
- aluminium can

License: CC BY 4.0.

### F. Roboflow Waste Detection
URL:
https://universe.roboflow.com/plastic-imjtq/waste-detection-ot1ha

Khoảng 6.6k ảnh.
Classes:
- beverage bottle
- express package
- tableware box
- tetra pak
- wash supplies bottle

License: CC BY 4.0.

### G. Roboflow Waste Detection - suthida
Khoảng 950 ảnh:
- paper
- cardboard
- plastic bottle
- plastic bag
- glass bottle
- pet bottle
- plastic container
- uht carton

License: CC BY 4.0.

### H. Các dataset khác
Cho phép thêm dataset qua config/manifest.
Không hard-code việc chỉ hỗ trợ 7 dataset trên.

Pipeline cần có cơ chế:
    register dataset
    inspect dataset
    map classes
    import dataset

---

# 4. Quan trọng: Không merge class một cách mù quáng

Phải xây dựng class mapping có 3 trạng thái:

1. EXACT
2. SAFE_ALIAS
3. REVIEW_REQUIRED

Ví dụ:

SAFE_ALIAS:

"Pet Bottle"                -> pet_bottle
"PET Bottle"                -> pet_bottle
"Plastic Bottle"            -> pet_bottle
"PET_Bottle"                 -> pet_bottle

"Tetra Pack"                -> tetra_pak
"Tetra Pak"                 -> tetra_pak
"Tetra-Pak"                 -> tetra_pak
"Tetrapak"                  -> tetra_pak

"Glass Bottle"              -> glass_bottle

"Glass Jar"                 -> glass_jar

"Plastic Cup"               -> plastic_cup

"Paper Cup"                 -> paper_cup

"Plastic Bag"               -> nylon_bag

"Metal Cap"                 -> metal_cap

"Aluminium Foil"            -> aluminum_foil

Các mapping nguy hiểm phải REVIEW_REQUIRED:

"Can"
"Metal Can"
"Drink Can"
"Food Can"
"Aluminium Can"
"Tin Can"

Không tự động gộp tất cả thành can/tin_can nếu không đủ thông tin.

Tương tự:

"Glass"
"Broken Glass"
"Glass Bottle"
"Glass Jar"

phải tách rõ.

"Paper"
"Paperboard"
"Cardboard"
"Paper Packaging"
"Paper Box"

không được tự động coi là cùng class.

---

# 5. Dataset manifest

Mỗi dataset phải có metadata manifest.

Ví dụ:

dataset_sources/
    roboflow_yolov7_trash/
        manifest.yaml

Schema tối thiểu:

name:
source:
url:
source_type:
license:
author:
version:
download_date:
task:
image_count:
annotation_format:
original_classes:
target_mapping:
notes:
quality_score:

target_mapping:

    "Plastic Bottle":
        target_class: pet_bottle
        mapping_type: SAFE_ALIAS
        confidence: 0.98
        note: "Assume standard PET bottle"

    "Can":
        target_class: null
        mapping_type: REVIEW_REQUIRED
        confidence: 0.50
        note: "May represent aluminum or tin"

Phải lưu nguyên class gốc.

---

# 6. Dataset directory structure

Đề xuất:

training/
    datasets/
        raw/
        normalized/
        review/
        final/

        sources/
            <dataset_name>/

        versions/
            v0.1/
            v0.2/

    configs/
        taxonomy.yaml
        dataset_sources.yaml
        dataset_mapping.yaml
        training.yaml

    scripts/
        inspect_dataset.py
        import_dataset.py
        normalize_dataset.py
        validate_dataset.py
        deduplicate_dataset.py
        build_dataset.py
        split_dataset.py
        dataset_report.py

    README.md

tools/
    dataset/

Có thể thay đổi path nếu repository hiện tại có cấu trúc khác, nhưng phải giữ separation giữa:
- raw
- processed
- final training dataset

---

# 7. Dataset import

Viết CLI:

python -m training.scripts.import_dataset \
    --source <path_or_url> \
    --name <dataset_name>

Hỗ trợ:
- Roboflow export
- YOLO dataset
- COCO detection
- Pascal VOC nếu cần
- ZIP dataset

Không giả định dataset nào cũng là YOLO.

Pipeline:

source
  ↓
detect format
  ↓
read metadata
  ↓
inspect classes
  ↓
apply mapping
  ↓
copy/normalize images
  ↓
convert annotations
  ↓
validate
  ↓
deduplicate
  ↓
produce normalized dataset

---

# 8. Roboflow handling

Không commit dataset binary vào Git.

Nếu Roboflow cần API key:
- KHÔNG hard-code API key
- đọc ROBOFLOW_API_KEY từ environment variable
- hoặc yêu cầu user export/download dataset trước.

Ưu tiên pipeline làm việc được với local ZIP/export.

Có thể cung cấp helper:

python -m training.scripts.import_dataset \
    --source datasets/downloads/yolov7-trash.zip \
    --source-type roboflow

Nếu có API integration:
    ROBOFLOW_API_KEY

Nhưng API integration là optional.

---

# 9. Image normalization

Chuẩn hóa:

- JPG/PNG/WebP đều đọc được
- output chuẩn JPG hoặc PNG
- normalize EXIF orientation
- kiểm tra image corruption
- loại ảnh zero-byte
- loại ảnh không thể decode
- giữ nguyên aspect ratio nếu không cần resize
- không resize dataset source chỉ để import

Lưu metadata:

width
height
channels
format
source
source_image_id
sha256
phash

---

# 10. Annotation normalization

Output YOLO:

<class_id> <x_center> <y_center> <width> <height>

Tất cả tọa độ normalized [0,1].

Validate:

0 <= class_id < 16
0 <= x <= 1
0 <= y <= 1
0 < width <= 1
0 < height <= 1

Phải phát hiện:

- negative values
- NaN
- inf
- width/height <= 0
- bbox vượt ảnh
- bbox quá nhỏ
- duplicate bbox
- malformed line

Có report riêng cho annotation errors.

---

# 11. Ambiguous label review queue

Đây là phần quan trọng.

Nếu mapping:

REVIEW_REQUIRED

thì KHÔNG đưa ngay vào training dataset.

Đưa vào:

training/datasets/review/

Ví dụ:

review/
    can/
    glass/
    paper/
    plastic_bag/

Tạo file:

review_manifest.jsonl

Ví dụ:

{
  "image": "...",
  "source_dataset": "...",
  "source_class": "Can",
  "candidate_targets": [
      "can",
      "tin_can"
  ],
  "reason": "Ambiguous metal container",
  "status": "pending"
}

Sau này người dùng có thể review thủ công.

---

# 12. Target class coverage analysis

Viết script:

python -m training.scripts.dataset_report \
    --dataset <path>

Sinh report:

dataset_report.json
dataset_report.csv
dataset_report.md

Phải thống kê:

- total images
- total instances
- instances/class
- images/class
- bbox/class
- bbox area distribution
- image resolution distribution
- source dataset distribution
- license distribution
- mapped class distribution
- unmapped classes
- review-required classes
- ignored classes
- duplicates
- corrupted images

---

# 13. Class coverage

Report rõ 16 class:

0 can
1 tin_can
2 aluminum_foil
3 metal_cap
4 cardboard
5 paper_cup
6 tetra_pak
7 paper_box
8 newspaper
9 pet_bottle
10 plastic_cup
11 detergent_bottle
12 nylon_bag
13 glass_bottle
14 glass_jar
15 broken_glass

Mỗi class:

images:
instances:
sources:
mapping_confidence:
percentage:

Phải xác định:

UNDERREPRESENTED
HEALTHY
OVERREPRESENTED
MISSING

Ngưỡng configurable.

Ví dụ:

< 100 instances:
    MISSING/VERY LOW

100-300:
    LOW

300-1000:
    MODERATE

>1000:
    GOOD

Không hard-code nếu dễ tránh; đưa vào config.

---

# 14. Critical class priority

Không tối ưu dataset chỉ theo tổng số ảnh.

Các class khó phải ưu tiên:

- aluminum_foil
- metal_cap
- paper_cup
- tetra_pak
- paper_box
- newspaper
- detergent_bottle
- nylon_bag
- glass_jar
- broken_glass

Đặc biệt:
- aluminum foil
- nylon bag
- broken glass
- glass bottle
- transparent plastic bottle

thường có shape/texture/reflection khó.

Report phải có "hard class priority".

---

# 15. Deduplication

Phải implement:

1. exact duplicate bằng SHA256
2. near duplicate bằng perceptual hash
3. optional image embedding similarity nếu dependency cho phép

Ban đầu:
- SHA256 bắt buộc
- pHash/dHash bắt buộc nếu thư viện ổn định

Không xóa file source.
Chỉ đánh dấu duplicate.

Ví dụ:

duplicate_groups.json

{
  "group_001": [
      "img001.jpg",
      "img002.jpg"
  ],
  "keep": "img001.jpg",
  "duplicates": ["img002.jpg"]
}

---

# 16. Dataset leakage prevention

Đây là yêu cầu bắt buộc.

Không random split đơn thuần nếu ảnh đến từ:

- cùng video
- cùng scene
- cùng sequence
- cùng source batch

Phải có group/session-aware split khi metadata cho phép.

Một source sequence chỉ nên thuộc:
- train
HOẶC
- val
HOẶC
- test

Không được xuất hiện ở nhiều split.

Điều này đặc biệt quan trọng sau khi có ảnh camera thực tế.

---

# 17. Train/Val/Test split

Mục tiêu:

70/20/10 hoặc 80/10/10 configurable.

Split theo:

source group / capture session / sequence

Ưu tiên stratified theo class nhưng phải bảo toàn group isolation.

Sinh:

images/train
images/val
images/test

labels/train
labels/val
labels/test

---

# 18. Negative samples

Pipeline phải hỗ trợ negative images:

Ảnh không có bất kỳ object target nào.

Ví dụ:
- nền băng tải
- bàn trống
- bóng
- phản xạ ánh sáng
- tay người
- vật không thuộc taxonomy

Negative images không có label file hoặc có empty label file tùy convention thống nhất.

Phải thống kê ratio negative/positive.

---

# 19. Không over-augment public dataset ở giai đoạn import

Import dataset phải giữ annotation gốc.

Augmentation chỉ áp dụng ở training stage.

Không tạo hàng triệu ảnh augmented trong dataset final.

---

# 20. Augmentation strategy

Training augmentation phải phù hợp với object sorting.

Cho phép:

- horizontal flip nếu không phá semantics
- small rotation
- scale
- translation
- brightness
- contrast
- hue/saturation nhỏ
- blur
- noise
- JPEG compression
- shadow
- random occlusion
- mosaic có kiểm soát

Cẩn trọng với:
- aggressive perspective
- extreme rotation
- extreme crop

vì sau này detector sẽ phục vụ bài toán mapping tọa độ object -> workspace/cơ cấu gắp.

Không augmentation quá mạnh khiến hình học object trở nên phi thực tế.

---

# 21. Public dataset quality filtering

Mỗi image phải có quality score.

Các tiêu chí:

resolution
annotation validity
object visibility
bbox size
blur
occlusion
truncation
duplicate
class ambiguity

Loại bỏ hoặc đưa vào review:

- object quá nhỏ
- object không nhận diện được
- bbox sai
- label không phù hợp
- ảnh quá mờ
- annotation lệch object
- screenshot/meme/cartoon
- ảnh render nếu mục tiêu là ảnh thực tế
- ảnh không giống domain waste sorting

Không được tự động xóa raw data.

---

# 22. Domain quality

Đánh dấu:

domain:
    natural
    studio
    conveyor
    floor
    street
    indoor
    outdoor
    synthetic
    unknown

Mục tiêu cuối cùng là training cho hệ thống camera cố định.

Public images vẫn hữu ích để học appearance/object shape,
nhưng phải đánh dấu domain mismatch.

---

# 23. Dataset source weighting

Mỗi source có:

source_weight

Ví dụ:

high:
    conveyor / real sorting scene

medium:
    realistic object images

low:
    studio/white-background

Không nhất thiết dùng ngay trong YOLO training,
nhưng metadata phải tồn tại để sau này weighted sampling.

---

# 24. Dataset versioning

Mỗi dataset build tạo version:

v0.1-public-baseline

v0.2-public-cleaned

v0.3-public-plus-real

v0.4-active-learning

Mỗi version phải có:

manifest.json
dataset_stats.json
classes.yaml
sources.json
mapping.json
README.md

Ghi:

- source datasets
- source versions
- number of images
- number of instances
- class distribution
- preprocessing
- filtering
- dedup result
- split seed
- creation time
- git commit if available

---

# 25. Dataset reproducibility

Phải có deterministic build.

Cùng:
- source
- mapping config
- filter config
- split seed

=> phải tạo ra cùng dataset composition.

Không phụ thuộc timestamp random.

---

# 26. Training dataset output

Final output phải tương thích YOLO.

Ví dụ:

training/datasets/versions/v0.1-public-baseline/

    images/
        train/
        val/
        test/

    labels/
        train/
        val/
        test/

    data.yaml

    metadata/
        manifest.json
        stats.json
        source_manifest.json
        class_mapping.json
        duplicates.json
        review.json

data.yaml:

path: ...
train: images/train
val: images/val
test: images/test

names:
    0: can
    1: tin_can
    2: aluminum_foil
    3: metal_cap
    4: cardboard
    5: paper_cup
    6: tetra_pak
    7: paper_box
    8: newspaper
    9: pet_bottle
    10: plastic_cup
    11: detergent_bottle
    12: nylon_bag
    13: glass_bottle
    14: glass_jar
    15: broken_glass

---

# 27. Baseline training preparation

Chưa cần optimize Raspberry Pi ở bước này.

Mục tiêu:

train a small detector baseline.

Support:

YOLOv8n
hoặc
YOLO11n

Thông qua config.

Ví dụ:

training/configs/training.yaml

model: yolov8n.pt
imgsz: 320
epochs: 100
batch: auto
device: auto
workers: 8
seed: 42

Các hyperparameters phải configurable.

---

# 28. Evaluation

Sau training phải có:

mAP50
mAP50-95
precision
recall
F1

per-class:

precision
recall
AP50
AP50-95

Và:

confusion matrix

Đặc biệt report riêng:

- metal confusion
- paper confusion
- plastic confusion
- glass confusion

Sorting system cần ưu tiên recall cho các vật cần gắp,
nên report phải highlight low-recall classes.

---

# 29. Export preparation

Training pipeline cần chuẩn bị integration với:

tools/validate_model.py

Cuối cùng target:

models/yolov8n_waste_int8.tflite

Nhưng KHÔNG bắt buộc training pipeline phải tự export ngay trong milestone đầu.

Thiết kế interface:

train
evaluate
export
validate

---

# 30. Representative dataset cho INT8

Pipeline phải có khả năng tạo:

training/datasets/representative/

Tập representative phải lấy từ ảnh training thực tế,
và sau này ưu tiên thêm ảnh camera deployment.

Không dùng một tập ảnh quá sạch chỉ có white background.

---

# 31. Active learning architecture

Chuẩn bị từ đầu cho giai đoạn sau.

Input:

camera images
model predictions

Sinh queue:

uncertain/
false_positive/
false_negative/
wrong_class/
bad_bbox/
out_of_distribution/

Metadata:

confidence
predicted_class
candidate_classes
bbox
image_path
model_version
dataset_version

Sau này human review sẽ đưa các ảnh này trở lại training dataset.

---

# 32. Auto-labeling

Pipeline phải thiết kế interface cho pseudo-label.

Ví dụ:

python -m training.scripts.auto_label \
    --model models/baseline.pt \
    --input incoming/

Rule ban đầu:

confidence >= 0.90
    -> auto accepted

0.50 <= confidence < 0.90
    -> review

confidence < 0.50
    -> uncertain

Threshold phải configurable.

Không đưa pseudo-label chất lượng thấp trực tiếp vào final training set.

---

# 33. Hard negative mining

Sau khi có model:

chạy model trên ảnh không thuộc target.

Tìm:
- false positive bottle
- false positive cup
- reflection mistaken as glass
- shadow mistaken as object
- label/logo mistaken as object
- human hand detection
- background waste not belonging taxonomy

Đưa vào hard-negative pool.

---

# 34. CLI requirements

Implement các CLI sau:

1.
python -m training.scripts.inspect_dataset \
    --input <dataset>

2.
python -m training.scripts.import_dataset \
    --source <dataset>

3.
python -m training.scripts.normalize_dataset \
    --input <dataset>

4.
python -m training.scripts.validate_dataset \
    --input <dataset>

5.
python -m training.scripts.deduplicate_dataset \
    --input <dataset>

6.
python -m training.scripts.build_dataset \
    --config training/configs/dataset_sources.yaml \
    --output training/datasets/versions/v0.1-public-baseline

7.
python -m training.scripts.dataset_report \
    --input <dataset>

8.
python -m training.scripts.split_dataset \
    --input <dataset> \
    --seed 42

Các command phải có:
--help

---

# 35. Dry-run

Tất cả command destructive phải hỗ trợ:

--dry-run

Ví dụ:

python -m training.scripts.build_dataset \
    --config ... \
    --dry-run

Output:

- sources
- classes
- mapping
- image count
- expected output
- duplicate count
- ambiguous labels
- warnings

Không thay đổi file khi dry-run.

---

# 36. Tests

Viết test cho:

- taxonomy loading
- class mapping
- exact alias mapping
- ambiguous mapping
- YOLO parser
- COCO -> YOLO conversion
- bbox normalization
- bbox validation
- duplicate detection
- dataset split
- group leakage
- manifest generation
- deterministic build
- data.yaml generation

Test đặc biệt:

class ID phải luôn giữ đúng 0-15.

---

# 37. Không phá runtime hiện tại

Không thay đổi behavior của:

- modules/
- FSM
- scheduler
- UART
- safety
- calibration
- runtime detector

trừ khi cần thêm integration rất rõ ràng.

Training/data pipeline phải độc lập với runtime.

---

# 38. Documentation

Tạo:

training/README.md

Nội dung phải giải thích:

1. taxonomy
2. source datasets
3. dataset import
4. class mapping
5. review workflow
6. validation
7. dedup
8. split
9. build baseline dataset
10. train
11. evaluate
12. export
13. active learning

Thêm một bảng:

Source class
-> Target class
-> Mapping type
-> Confidence
-> Reason

---

# 39. Dataset acquisition priority

Không cần tải tất cả dataset ngay lập tức.

Ưu tiên:

PHASE 1:
- YoloV7 Trash Dataset V7
- Trash Detection
- General waste data
- solidwaste
- Waste Detection (tetra pak / wash supplies)

PHASE 2:
- bổ sung dataset chuyên biệt cho class thiếu

PHASE 3:
- ảnh thực tế từ camera của user

---

# 40. Milestone hiện tại

Milestone M0:

Mục tiêu:
Tạo được public baseline dataset sạch, reproducible, không có mapping mơ hồ đi thẳng vào training.

Expected:

16 classes được định nghĩa chính xác.

Dataset report cho biết:
- class nào có dữ liệu
- class nào thiếu
- source nào đóng góp dữ liệu
- class nào mapping ambiguous
- số lượng duplicate
- số lượng image/instance
- train/val/test distribution

Output:

training/datasets/versions/v0.1-public-baseline/

Dataset phải có:
- YOLO annotation
- data.yaml
- metadata
- source manifest
- mapping manifest
- validation report

Chưa cần ảnh camera thực tế.

---

# 41. Acceptance criteria

Task chỉ được xem là hoàn thành khi:

[ ] Có pipeline import public dataset.

[ ] Có taxonomy 16 class đúng repository.

[ ] Có source manifest.

[ ] Có class mapping rõ ràng.

[ ] Không tự ý merge ambiguous class.

[ ] Có review queue.

[ ] Có YOLO conversion.

[ ] Có annotation validation.

[ ] Có image validation.

[ ] Có SHA256 + perceptual dedup.

[ ] Có dataset report.

[ ] Có class distribution report.

[ ] Có group-aware split.

[ ] Không leakage giữa train/val/test.

[ ] Có deterministic build.

[ ] Có data.yaml.

[ ] Có tests.

[ ] Có README.

[ ] Có thể chạy từ command line.

[ ] Không làm hỏng runtime hiện tại.

---

# 42. Quan trọng về thứ tự thực hiện

Không viết toàn bộ code một cách mù quáng.

Trước tiên:

1. audit repository hiện tại
2. đọc:
   - config/model_classes.json
   - config/class_mapping.json
   - settings.yaml
   - README.md
   - tools/validate_model.py
   - tools/dev_detect.py
3. xác nhận taxonomy
4. thiết kế dataset pipeline
5. inspect public datasets
6. implement import/normalize
7. validate
8. build v0.1
9. report
10. mới bổ sung training code

Sau mỗi phase:
- chạy tests
- chạy CLI thử
- sinh report
- không phá runtime.

Nếu phát hiện bất kỳ điểm nào trong yêu cầu này mâu thuẫn với code hiện tại của repository:
- ưu tiên tính nhất quán với repository
- ghi rõ conflict
- đề xuất phương án
- không tự ý thay đổi taxonomy.
