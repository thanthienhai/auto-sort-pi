"""Taxonomy 16-class chính thức của auto-sort-pi + class mapping engine.

## Taxonomy

Nguồn sự thật LÀ repository, không duplicate ở đây:

- ``config/model_classes.json``  — danh sách 16 class, thứ tự = class_id (0..15).
- ``config/class_mapping.json``  — class name -> sorting group id (1..4).

Module này chỉ load hai file đó và cộng thêm phần cấu hình *không* có sẵn trong
runtime config (tên nhóm, ngưỡng coverage, danh sách "hard class") từ
``training/configs/taxonomy.yaml``.

## Mapping engine

Class mapping engine: source dataset label -> taxonomy class.

3 trạng thái bắt buộc (§4 training_online_dataset.md):

- EXACT           : label trùng khớp tên class taxonomy (sau khi chuẩn hóa).
- SAFE_ALIAS       : label khác tên nhưng đã được con người xác nhận là an toàn
                     để tự động map (vd "Plastic Bottle" -> pet_bottle).
- REVIEW_REQUIRED  : label mơ hồ, KHÔNG được tự động đưa vào training dataset.

Thứ tự ưu tiên khi resolve một label:

1. ``target_mapping`` khai báo trong manifest riêng của dataset (con người đã
   review class list cụ thể của dataset đó) — cao nhất.
2. Bảng global ``training/configs/dataset_mapping.yaml`` (áp dụng chung).
3. Auto EXACT nếu label chuẩn hóa trùng đúng tên một class taxonomy.
4. Mặc định REVIEW_REQUIRED (không rõ -> không tự merge).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CLASSES = REPO_ROOT / "config" / "model_classes.json"
DEFAULT_CLASS_MAPPING = REPO_ROOT / "config" / "class_mapping.json"
DEFAULT_TAXONOMY_EXTRA = REPO_ROOT / "training" / "configs" / "taxonomy.yaml"
DEFAULT_DATASET_MAPPING = REPO_ROOT / "training" / "configs" / "dataset_mapping.yaml"

NUM_CLASSES = 16

_SEP_RE = re.compile(r"[\s\-_/]+")


class TaxonomyError(ValueError):
    """Taxonomy trong repository không hợp lệ hoặc không nhất quán."""


@dataclass(frozen=True)
class Taxonomy:
    classes: tuple[str, ...]
    """16 tên class, index = class_id."""

    class_group: dict[str, int]
    """class name -> sorting group id (1=METAL, 2=PAPER, 3=PLASTIC, 4=GLASS)."""

    group_names: dict[int, str]
    coverage_thresholds: dict[str, int]
    hard_classes: tuple[str, ...]

    name_to_id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name_to_id", {name: i for i, name in enumerate(self.classes)})

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def id_for(self, name: str) -> int:
        return self.name_to_id[name]

    def name_for(self, class_id: int) -> str:
        return self.classes[class_id]

    def group_for(self, name: str) -> int:
        return self.class_group[name]

    def group_name_for(self, name: str) -> str:
        return self.group_names[self.class_group[name]]

    def is_valid_class_id(self, class_id: int) -> bool:
        return 0 <= class_id < self.num_classes

    def classes_in_group(self, group_id: int) -> list[str]:
        return [c for c in self.classes if self.class_group[c] == group_id]


def load_taxonomy(
    model_classes_path: str | Path = DEFAULT_MODEL_CLASSES,
    class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING,
    taxonomy_extra_path: str | Path = DEFAULT_TAXONOMY_EXTRA,
) -> Taxonomy:
    """Load taxonomy từ config runtime của repository (không duplicate)."""

    model_classes_path = Path(model_classes_path)
    class_mapping_path = Path(class_mapping_path)

    try:
        classes = json.loads(model_classes_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaxonomyError(f"Không tìm thấy {model_classes_path}") from exc
    except json.JSONDecodeError as exc:
        raise TaxonomyError(f"{model_classes_path} không phải JSON hợp lệ: {exc}") from exc

    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise TaxonomyError(f"{model_classes_path} phải là JSON list các string")
    if len(classes) != NUM_CLASSES:
        raise TaxonomyError(
            f"{model_classes_path} có {len(classes)} class, taxonomy chính thức yêu cầu {NUM_CLASSES}"
        )

    try:
        class_group_raw = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TaxonomyError(f"Không tìm thấy {class_mapping_path}") from exc
    except json.JSONDecodeError as exc:
        raise TaxonomyError(f"{class_mapping_path} không phải JSON hợp lệ: {exc}") from exc

    if not isinstance(class_group_raw, dict):
        raise TaxonomyError(f"{class_mapping_path} phải là JSON object")

    missing = [c for c in classes if c not in class_group_raw]
    if missing:
        raise TaxonomyError(f"{class_mapping_path} thiếu mapping cho class: {missing}")

    class_group = {name: int(class_group_raw[name]) for name in classes}

    extra: dict = {}
    extra_path = Path(taxonomy_extra_path)
    if extra_path.is_file():
        extra = yaml.safe_load(extra_path.read_text(encoding="utf-8")) or {}

    group_names = {int(k): v for k, v in (extra.get("group_names") or {}).items()}
    default_group_names = {1: "METAL", 2: "PAPER", 3: "PLASTIC", 4: "GLASS"}
    for gid, name in default_group_names.items():
        group_names.setdefault(gid, name)

    coverage_thresholds = extra.get("coverage_thresholds") or {
        "missing_max": 100,
        "low_max": 300,
        "moderate_max": 1000,
    }
    hard_classes = tuple(extra.get("hard_classes") or ())

    return Taxonomy(
        classes=tuple(classes),
        class_group=class_group,
        group_names=group_names,
        coverage_thresholds=coverage_thresholds,
        hard_classes=hard_classes,
    )


def coverage_status(instance_count: int, thresholds: dict[str, int]) -> str:
    if instance_count < thresholds.get("missing_max", 100):
        return "MISSING" if instance_count == 0 else "VERY_LOW"
    if instance_count < thresholds.get("low_max", 300):
        return "LOW"
    if instance_count < thresholds.get("moderate_max", 1000):
        return "MODERATE"
    return "GOOD"


class MappingType(str, Enum):
    EXACT = "EXACT"
    SAFE_ALIAS = "SAFE_ALIAS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


def normalize_label(label: str) -> str:
    """Chuẩn hóa label: lowercase, gộp khoảng trắng/-/_//  thành 1 dấu space."""

    return _SEP_RE.sub(" ", label.strip().lower()).strip()


@dataclass(frozen=True)
class MappingResult:
    source_label: str
    normalized_label: str
    target_class: str | None
    mapping_type: MappingType
    confidence: float
    candidate_targets: tuple[str, ...] = ()
    note: str = ""
    rule_source: str = "default"

    def to_dict(self) -> dict:
        return {
            "source_label": self.source_label,
            "normalized_label": self.normalized_label,
            "target_class": self.target_class,
            "mapping_type": self.mapping_type.value,
            "confidence": self.confidence,
            "candidate_targets": list(self.candidate_targets),
            "note": self.note,
            "rule_source": self.rule_source,
        }


@dataclass
class _Rule:
    target_class: str | None
    mapping_type: MappingType
    confidence: float
    candidate_targets: tuple[str, ...] = ()
    note: str = ""


def _parse_rule(norm_key: str, entry: dict) -> _Rule:
    mapping_type = MappingType(entry["mapping_type"])
    target_class = entry.get("target_class")
    if mapping_type != MappingType.REVIEW_REQUIRED and not target_class:
        raise ValueError(f"mapping rule '{norm_key}': target_class required for {mapping_type}")
    return _Rule(
        target_class=target_class,
        mapping_type=mapping_type,
        confidence=float(entry.get("confidence", 1.0 if mapping_type == MappingType.EXACT else 0.5)),
        candidate_targets=tuple(entry.get("candidate_targets") or ()),
        note=entry.get("note", ""),
    )


class MappingEngine:
    def __init__(self, taxonomy: Taxonomy, global_rules: dict[str, _Rule] | None = None) -> None:
        self.taxonomy = taxonomy
        self._global_rules = global_rules or {}
        self._class_name_set = set(taxonomy.classes)

    @classmethod
    def load(
        cls,
        taxonomy: Taxonomy,
        dataset_mapping_path: str | Path = DEFAULT_DATASET_MAPPING,
    ) -> "MappingEngine":
        path = Path(dataset_mapping_path)
        raw: dict = {}
        if path.is_file():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        rules: dict[str, _Rule] = {}
        for source_label, entry in (raw.get("labels") or {}).items():
            norm_key = normalize_label(source_label)
            rules[norm_key] = _parse_rule(norm_key, entry)

        engine = cls(taxonomy, rules)
        engine._validate_rules()
        return engine

    def _validate_rules(self) -> None:
        for norm_key, rule in self._global_rules.items():
            if rule.target_class and rule.target_class not in self._class_name_set:
                raise ValueError(
                    f"dataset_mapping rule '{norm_key}' points to unknown class '{rule.target_class}'"
                )
            for cand in rule.candidate_targets:
                if cand not in self._class_name_set:
                    raise ValueError(
                        f"dataset_mapping rule '{norm_key}' has unknown candidate_target '{cand}'"
                    )

    def resolve(
        self,
        source_label: str,
        dataset_override: dict[str, dict] | None = None,
    ) -> MappingResult:
        norm = normalize_label(source_label)

        # 1. per-dataset override (manifest.yaml target_mapping) — cao nhất.
        if dataset_override:
            for key, entry in dataset_override.items():
                if normalize_label(key) == norm:
                    rule = _parse_rule(norm, entry)
                    return MappingResult(
                        source_label=source_label,
                        normalized_label=norm,
                        target_class=rule.target_class,
                        mapping_type=rule.mapping_type,
                        confidence=rule.confidence,
                        candidate_targets=rule.candidate_targets,
                        note=rule.note,
                        rule_source="dataset_manifest",
                    )

        # 2. global rule table (bao gồm cả SAFE_ALIAS và các trường hợp nguy hiểm
        #    bị ép REVIEW_REQUIRED dù trùng tên class, vd "Tin Can").
        rule = self._global_rules.get(norm)
        if rule is not None:
            return MappingResult(
                source_label=source_label,
                normalized_label=norm,
                target_class=rule.target_class,
                mapping_type=rule.mapping_type,
                confidence=rule.confidence,
                candidate_targets=rule.candidate_targets,
                note=rule.note,
                rule_source="global_mapping",
            )

        # 3. auto EXACT: label chuẩn hóa trùng đúng tên 1 class.
        collapsed = norm.replace(" ", "_")
        if collapsed in self._class_name_set:
            return MappingResult(
                source_label=source_label,
                normalized_label=norm,
                target_class=collapsed,
                mapping_type=MappingType.EXACT,
                confidence=1.0,
                rule_source="auto_exact",
            )

        # 4. mặc định: không rõ -> REVIEW_REQUIRED, không tự merge.
        return MappingResult(
            source_label=source_label,
            normalized_label=norm,
            target_class=None,
            mapping_type=MappingType.REVIEW_REQUIRED,
            confidence=0.0,
            candidate_targets=(),
            note="Unknown label, not present in taxonomy or mapping tables — cần review thủ công",
            rule_source="default_unknown",
        )
