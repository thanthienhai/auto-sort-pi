import json
import math

from PIL import Image

from training.lib.formats import (
    YoloBox,
    coco_bbox_to_yolo,
    detect_format,
    find_duplicate_boxes,
    parse_yolo_line,
    read_dataset,
    validate_bbox,
    validate_label_file,
    voc_bbox_to_yolo,
)


def _make_image(path, size=(100, 80), color=(200, 50, 50)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_detect_and_read_yolo_dataset(tmp_path):
    root = tmp_path / "yolo_ds"
    _make_image(root / "images" / "img1.jpg")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text("names:\n  0: pet_bottle\n", encoding="utf-8")

    assert detect_format(root) == "yolo"
    images = read_dataset(root, "yolo")
    assert len(images) == 1
    assert images[0].boxes[0].label == "pet_bottle"


def test_detect_and_read_coco_dataset(tmp_path):
    root = tmp_path / "coco_ds"
    _make_image(root / "img1.jpg", size=(1000, 500))
    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 1000, "height": 500}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 10, "bbox": [100, 50, 200, 100]}],
        "categories": [{"id": 10, "name": "Glass Bottle"}],
    }
    (root / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    assert detect_format(root) == "coco"
    images = read_dataset(root, "coco")
    assert len(images) == 1
    assert images[0].boxes[0].label == "Glass Bottle"


def test_detect_and_read_voc_dataset(tmp_path):
    root = tmp_path / "voc_ds"
    _make_image(root / "img1.jpg", size=(400, 300))
    xml = """<annotation>
        <filename>img1.jpg</filename>
        <size><width>400</width><height>300</height></size>
        <object>
            <name>Cardboard</name>
            <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>110</xmax><ymax>110</ymax></bndbox>
        </object>
    </annotation>"""
    (root / "img1.xml").write_text(xml, encoding="utf-8")

    assert detect_format(root) == "voc"
    images = read_dataset(root, "voc")
    assert len(images) == 1
    assert images[0].boxes[0].label == "Cardboard"


def test_detect_and_read_generic_json_dataset(tmp_path):
    root = tmp_path / "generic_ds"
    _make_image(root / "img1.jpg", size=(500, 400))
    annotations = [{"label": "nhua_pet_bottle", "bbox_xyxy": [50, 50, 150, 150]}]
    (root / "img1.json").write_text(json.dumps(annotations), encoding="utf-8")

    assert detect_format(root) == "generic_json"
    images = read_dataset(root, "generic_json")
    assert len(images) == 1
    assert images[0].boxes[0].label == "nhua_pet_bottle"


def test_parse_yolo_line_valid():
    box, errors = parse_yolo_line("3 0.5 0.5 0.2 0.3")
    assert not errors
    assert box == YoloBox(3, 0.5, 0.5, 0.2, 0.3)


def test_parse_yolo_line_malformed():
    box, errors = parse_yolo_line("3 0.5 0.5")
    assert box is None
    assert errors

    box2, errors2 = parse_yolo_line("abc 0.5 0.5 0.2 0.3")
    assert box2 is None
    assert errors2


def test_validate_bbox_happy_path():
    box = YoloBox(0, 0.5, 0.5, 0.2, 0.2)
    assert validate_bbox(box, num_classes=16) == []


def test_validate_bbox_class_id_out_of_range():
    box = YoloBox(99, 0.5, 0.5, 0.2, 0.2)
    errors = validate_bbox(box, num_classes=16)
    assert any("out of range" in e for e in errors)


def test_validate_bbox_negative_and_out_of_bounds():
    box = YoloBox(0, -0.1, 0.5, 0.2, 0.2)
    errors = validate_bbox(box, num_classes=16)
    assert any("out of [0,1]" in e for e in errors)


def test_validate_bbox_nan_and_inf():
    box_nan = YoloBox(0, math.nan, 0.5, 0.2, 0.2)
    assert any("NaN" in e for e in validate_bbox(box_nan, 16))

    box_inf = YoloBox(0, math.inf, 0.5, 0.2, 0.2)
    assert any("inf" in e for e in validate_bbox(box_inf, 16))


def test_validate_bbox_zero_or_negative_size():
    box = YoloBox(0, 0.5, 0.5, 0.0, -0.1)
    errors = validate_bbox(box, 16)
    assert any("width" in e for e in errors)
    assert any("height" in e for e in errors)


def test_validate_bbox_exceeds_image_bounds():
    box = YoloBox(0, 0.95, 0.5, 0.5, 0.2)  # x + w/2 = 1.2 > 1
    errors = validate_bbox(box, 16)
    assert any("exceeds image bounds" in e for e in errors)


def test_validate_bbox_too_small():
    box = YoloBox(0, 0.5, 0.5, 0.001, 0.001)
    errors = validate_bbox(box, 16)
    assert any("too small" in e for e in errors)


def test_find_duplicate_boxes():
    boxes = [
        YoloBox(0, 0.5, 0.5, 0.2, 0.2),
        YoloBox(0, 0.5001, 0.5001, 0.2, 0.2),  # near-identical, same class
        YoloBox(1, 0.5, 0.5, 0.2, 0.2),  # different class, not a duplicate
    ]
    pairs = find_duplicate_boxes(boxes, iou_threshold=0.98)
    assert pairs == [(0, 1)]


def test_validate_label_file_reports_duplicate_and_malformed():
    lines = [
        "0 0.5 0.5 0.2 0.2",
        "0 0.5001 0.5001 0.2 0.2",
        "bad line",
        "",
    ]
    result = validate_label_file(lines, num_classes=16)
    assert result["num_duplicates"] == 1
    assert any("malformed" in e for e in result["errors"])
    assert any("duplicate bbox" in e for e in result["errors"])


def test_coco_bbox_to_yolo_conversion():
    # COCO bbox: x_min=100, y_min=50, w=200, h=100 in a 1000x500 image
    yolo = coco_bbox_to_yolo((100, 50, 200, 100), img_w=1000, img_h=500)
    assert yolo is not None
    assert math.isclose(yolo.x, (100 + 100) / 1000)
    assert math.isclose(yolo.y, (50 + 50) / 500)
    assert math.isclose(yolo.w, 200 / 1000)
    assert math.isclose(yolo.h, 100 / 500)


def test_coco_bbox_to_yolo_invalid_returns_none():
    assert coco_bbox_to_yolo((0, 0, 0, 100), img_w=1000, img_h=500) is None
    assert coco_bbox_to_yolo((0, 0, 100, 100), img_w=0, img_h=500) is None


def test_voc_bbox_to_yolo_conversion():
    yolo = voc_bbox_to_yolo(xmin=100, ymin=50, xmax=300, ymax=150, img_w=1000, img_h=500)
    assert yolo is not None
    assert math.isclose(yolo.x, 200 / 1000)
    assert math.isclose(yolo.y, 100 / 500)
    assert math.isclose(yolo.w, 200 / 1000)
    assert math.isclose(yolo.h, 100 / 500)
