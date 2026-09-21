"""Annotation normalization (§10) + dataset format detection/readers (§7).

Parse/validate YOLO, convert COCO/VOC -> YOLO, và phát hiện/đọc format dataset:
YOLO, COCO, Pascal VOC, generic_json. Không giả định dataset nào cũng là
YOLO — mỗi source dataset phải qua bước ``detect_format`` trước khi đọc.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

MIN_BBOX_NORM_AREA = 1e-5  # bbox nhỏ hơn ngưỡng này (theo tỉ lệ ảnh) bị coi là "quá nhỏ"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x: float
    y: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}"


class AnnotationError(str):
    """Một lỗi annotation, dùng str để dễ serialize trong report."""


def parse_yolo_line(line: str) -> tuple[YoloBox | None, list[str]]:
    """Parse 1 dòng YOLO. Trả về (box hoặc None, list lỗi)."""

    errors: list[str] = []
    parts = line.strip().split()
    if len(parts) != 5:
        return None, [f"malformed line (expected 5 fields, got {len(parts)}): {line!r}"]

    try:
        class_id = int(parts[0])
    except ValueError:
        return None, [f"malformed class_id: {parts[0]!r}"]

    try:
        x, y, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return None, [f"malformed coordinate in line: {line!r}"]

    return YoloBox(class_id, x, y, w, h), errors


def validate_bbox(box: YoloBox, num_classes: int) -> list[str]:
    """Validate 1 bbox YOLO theo §10. Trả về list lỗi (rỗng nếu hợp lệ)."""

    errors: list[str] = []

    for name, val in (("x", box.x), ("y", box.y), ("width", box.w), ("height", box.h)):
        if math.isnan(val):
            errors.append(f"{name} is NaN")
        elif math.isinf(val):
            errors.append(f"{name} is inf")

    if errors:
        return errors  # NaN/inf làm mọi so sánh dưới đây vô nghĩa

    if not (0 <= box.class_id < num_classes):
        errors.append(f"class_id {box.class_id} out of range [0,{num_classes})")

    if box.x < 0 or box.x > 1:
        errors.append(f"x_center {box.x} out of [0,1]")
    if box.y < 0 or box.y > 1:
        errors.append(f"y_center {box.y} out of [0,1]")
    if box.w <= 0:
        errors.append(f"width {box.w} <= 0")
    elif box.w > 1:
        errors.append(f"width {box.w} > 1")
    if box.h <= 0:
        errors.append(f"height {box.h} <= 0")
    elif box.h > 1:
        errors.append(f"height {box.h} > 1")

    if not errors:
        x0, x1 = box.x - box.w / 2, box.x + box.w / 2
        y0, y1 = box.y - box.h / 2, box.y + box.h / 2
        if x0 < -1e-6 or x1 > 1 + 1e-6 or y0 < -1e-6 or y1 > 1 + 1e-6:
            errors.append(f"bbox exceeds image bounds: x=[{x0:.4f},{x1:.4f}] y=[{y0:.4f},{y1:.4f}]")
        if box.w * box.h < MIN_BBOX_NORM_AREA:
            errors.append(f"bbox too small: area={box.w * box.h:.8f} < {MIN_BBOX_NORM_AREA}")

    return errors


def find_duplicate_boxes(boxes: list[YoloBox], iou_threshold: float = 0.98) -> list[tuple[int, int]]:
    """Trả về list các cặp (i, j) bbox trùng nhau (cùng class, IoU cao)."""

    def iou(a: YoloBox, b: YoloBox) -> float:
        ax0, ax1 = a.x - a.w / 2, a.x + a.w / 2
        ay0, ay1 = a.y - a.h / 2, a.y + a.h / 2
        bx0, bx1 = b.x - b.w / 2, b.x + b.w / 2
        by0, by1 = b.y - b.h / 2, b.y + b.h / 2
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        union = a.w * a.h + b.w * b.h - inter
        return inter / union if union > 0 else 0.0

    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].class_id == boxes[j].class_id and iou(boxes[i], boxes[j]) >= iou_threshold:
                pairs.append((i, j))
    return pairs


def validate_label_file(lines: list[str], num_classes: int) -> dict:
    """Validate toàn bộ 1 file label YOLO. Trả về report dict."""

    boxes: list[YoloBox] = []
    errors: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        box, line_errors = parse_yolo_line(line)
        if line_errors:
            errors.extend(f"line {i}: {e}" for e in line_errors)
            continue
        box_errors = validate_bbox(box, num_classes)
        if box_errors:
            errors.extend(f"line {i}: {e}" for e in box_errors)
        else:
            boxes.append(box)

    dup_pairs = find_duplicate_boxes(boxes)
    for i, j in dup_pairs:
        errors.append(f"duplicate bbox: line index {i} and {j} (class {boxes[i].class_id})")

    return {"boxes": boxes, "errors": errors, "num_duplicates": len(dup_pairs)}


def coco_bbox_to_yolo(bbox_xywh_abs: tuple[float, float, float, float], img_w: int, img_h: int) -> YoloBox | None:
    """COCO bbox = [x_min, y_min, width, height] tuyệt đối (pixel) -> YOLO normalized."""

    x_min, y_min, w, h = bbox_xywh_abs
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    cx = (x_min + w / 2) / img_w
    cy = (y_min + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return YoloBox(class_id=-1, x=cx, y=cy, w=nw, h=nh)


def voc_bbox_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> YoloBox | None:
    """Pascal VOC bbox = [xmin, ymin, xmax, ymax] tuyệt đối -> YOLO normalized."""

    w = xmax - xmin
    h = ymax - ymin
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    cx = (xmin + w / 2) / img_w
    cy = (ymin + h / 2) / img_h
    return YoloBox(class_id=-1, x=cx, y=cy, w=w / img_w, h=h / img_h)


class UnknownDatasetFormatError(ValueError):
    pass


@dataclass
class RawBox:
    label: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class RawAnnotatedImage:
    image_path: Path
    width: int
    height: int
    boxes: list[RawBox] = field(default_factory=list)


def maybe_extract_zip(path: str | Path, extract_dir: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        return path
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _find_yolo_names(root: Path) -> list[str] | None:
    for data_yaml in root.rglob("data.yaml"):
        import yaml

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict):
            return [names[i] for i in sorted(names, key=int)]
        if isinstance(names, list):
            return names
    for classes_txt in root.rglob("classes.txt"):
        return [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def detect_format(root: str | Path) -> str:
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".zip":
        return "zip"

    if any(root.rglob("*.coco.json")):
        return "coco"
    for json_file in root.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys():
            return "coco"

    if any(root.rglob("*.xml")):
        for xml_file in root.rglob("*.xml"):
            try:
                root_tag = ET.parse(xml_file).getroot().tag
            except ET.ParseError:
                continue
            if root_tag == "annotation":
                return "voc"

    has_images_labels = any(root.rglob("images/**/*")) and any(root.rglob("labels/**/*.txt"))
    if has_images_labels or _find_yolo_names(root) is not None:
        return "yolo"

    # generic_json bootstrap format: <image>.jpg cạnh <image>.json (list of
    # {"label":..., "bbox_xyxy":[...]})
    for json_file in root.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, list) and data and all("label" in d and "bbox_xyxy" in d for d in data):
            return "generic_json"

    raise UnknownDatasetFormatError(f"Không nhận diện được format dataset tại {root}")


def read_yolo_dataset(root: Path) -> list[RawAnnotatedImage]:
    names = _find_yolo_names(root)
    if names is None:
        raise UnknownDatasetFormatError(f"YOLO dataset tại {root} thiếu data.yaml/classes.txt")

    results = []
    label_files = list(root.rglob("labels/**/*.txt"))
    if not label_files:
        label_files = [p.with_suffix(".txt") for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    for label_file in label_files:
        image_path = None
        for ext in IMAGE_EXTS:
            candidate = Path(str(label_file).replace("labels", "images", 1)).with_suffix(ext)
            if candidate.is_file():
                image_path = candidate
                break
        if image_path is None:
            candidate = label_file.with_suffix("")
            for ext in IMAGE_EXTS:
                if candidate.with_suffix(ext).is_file():
                    image_path = candidate.with_suffix(ext)
                    break
        if image_path is None or not label_file.is_file():
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            continue

        boxes = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            box, errors = parse_yolo_line(line)
            if errors or box is None:
                continue
            if 0 <= box.class_id < len(names):
                boxes.append(RawBox(names[box.class_id], box.x, box.y, box.w, box.h))
        results.append(RawAnnotatedImage(image_path, width, height, boxes))
    return results


def read_coco_dataset(root: Path) -> list[RawAnnotatedImage]:
    coco_files = list(root.rglob("*.coco.json")) or [
        f
        for f in root.rglob("*.json")
        if _is_coco_json(f)
    ]
    results: list[RawAnnotatedImage] = []
    for coco_file in coco_files:
        data = json.loads(coco_file.read_text(encoding="utf-8"))
        cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
        images_by_id = {img["id"]: img for img in data.get("images", [])}
        anns_by_image: dict[int, list] = {}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        image_dir = coco_file.parent
        for image_id, img_meta in images_by_id.items():
            image_path = image_dir / img_meta["file_name"]
            if not image_path.is_file():
                candidates = list(root.rglob(img_meta["file_name"]))
                if not candidates:
                    continue
                image_path = candidates[0]
            width, height = img_meta.get("width", 0), img_meta.get("height", 0)
            boxes = []
            for ann in anns_by_image.get(image_id, []):
                yolo = coco_bbox_to_yolo(tuple(ann["bbox"]), width, height)
                if yolo is None:
                    continue
                label = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
                boxes.append(RawBox(label, yolo.x, yolo.y, yolo.w, yolo.h))
            results.append(RawAnnotatedImage(image_path, width, height, boxes))
    return results


def _is_coco_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(data, dict) and {"images", "annotations", "categories"} <= data.keys()


def read_voc_dataset(root: Path) -> list[RawAnnotatedImage]:
    results = []
    for xml_file in root.rglob("*.xml"):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue
        xml_root = tree.getroot()
        if xml_root.tag != "annotation":
            continue

        filename_el = xml_root.find("filename")
        image_path = None
        if filename_el is not None and filename_el.text:
            candidates = list(root.rglob(filename_el.text))
            if candidates:
                image_path = candidates[0]
        if image_path is None:
            for ext in IMAGE_EXTS:
                candidate = xml_file.with_suffix(ext)
                if candidate.is_file():
                    image_path = candidate
                    break
        if image_path is None:
            continue

        size_el = xml_root.find("size")
        if size_el is not None:
            width = int(size_el.findtext("width", "0"))
            height = int(size_el.findtext("height", "0"))
        else:
            with Image.open(image_path) as img:
                width, height = img.size

        boxes = []
        for obj in xml_root.findall("object"):
            name = obj.findtext("name", "")
            bnd = obj.find("bndbox")
            if bnd is None or not name:
                continue
            xmin, ymin = float(bnd.findtext("xmin", "0")), float(bnd.findtext("ymin", "0"))
            xmax, ymax = float(bnd.findtext("xmax", "0")), float(bnd.findtext("ymax", "0"))
            yolo = voc_bbox_to_yolo(xmin, ymin, xmax, ymax, width, height)
            if yolo is not None:
                boxes.append(RawBox(name, yolo.x, yolo.y, yolo.w, yolo.h))
        results.append(RawAnnotatedImage(image_path, width, height, boxes))
    return results


def read_generic_json_dataset(root: Path) -> list[RawAnnotatedImage]:
    """Format bootstrap tự định nghĩa cho ảnh chưa có trong 1 dataset chuẩn:

    <name>.jpg cạnh <name>.json chứa list [{"label": str, "bbox_xyxy": [x0,y0,x1,y1]}].
    Đây cũng là format dùng cho ảnh camera thực tế sau này (§1.10, §31).
    """

    results = []
    for image_path in root.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            continue
        try:
            annotations = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        boxes = []
        for ann in annotations:
            x0, y0, x1, y1 = ann["bbox_xyxy"]
            yolo = voc_bbox_to_yolo(x0, y0, x1, y1, width, height)
            if yolo is not None:
                boxes.append(RawBox(ann["label"], yolo.x, yolo.y, yolo.w, yolo.h))
        results.append(RawAnnotatedImage(image_path, width, height, boxes))
    return results


READERS = {
    "yolo": read_yolo_dataset,
    "coco": read_coco_dataset,
    "voc": read_voc_dataset,
    "generic_json": read_generic_json_dataset,
}


def read_dataset(root: str | Path, fmt: str | None = None) -> list[RawAnnotatedImage]:
    root = Path(root)
    fmt = fmt or detect_format(root)
    if fmt not in READERS:
        raise UnknownDatasetFormatError(f"Format không hỗ trợ: {fmt}")
    return READERS[fmt](root)
