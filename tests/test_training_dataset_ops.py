import pytest

from training.lib.dataset_ops import (
    DatasetManifest,
    ReviewRecord,
    SplitRatios,
    assign_group_splits,
    read_review_manifest,
    split_items,
    verify_no_leakage,
    write_review_manifest,
)


def test_manifest_roundtrip(tmp_path):
    manifest = DatasetManifest(
        name="test_ds",
        source="roboflow",
        license="CC BY 4.0",
        original_classes=["Can", "Cardboard"],
        target_mapping={
            "Can": {"target_class": None, "mapping_type": "REVIEW_REQUIRED", "confidence": 0.5, "candidate_targets": ["can", "tin_can"]},
        },
    )
    path = tmp_path / "manifest.yaml"
    manifest.save(path)

    loaded = DatasetManifest.load(path)
    assert loaded.name == "test_ds"
    assert loaded.license == "CC BY 4.0"
    assert loaded.original_classes == ["Can", "Cardboard"]
    assert loaded.target_mapping["Can"]["mapping_type"] == "REVIEW_REQUIRED"


def test_review_manifest_write_and_read(tmp_path):
    records = [
        ReviewRecord(
            image="img1.jpg",
            source_dataset="ds1",
            source_class="Can",
            candidate_targets=["can", "tin_can"],
            reason="Ambiguous metal container",
        )
    ]
    path = tmp_path / "review_manifest.jsonl"
    write_review_manifest(records, path)

    loaded = read_review_manifest(path)
    assert len(loaded) == 1
    assert loaded[0]["source_class"] == "Can"
    assert loaded[0]["status"] == "pending"
    assert loaded[0]["candidate_targets"] == ["can", "tin_can"]


def test_read_review_manifest_missing_file_returns_empty(tmp_path):
    assert read_review_manifest(tmp_path / "nope.jsonl") == []


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitRatios(train=0.5, val=0.5, test=0.5)


def test_split_is_deterministic_for_same_seed():
    groups = [f"group_{i}" for i in range(200)]
    result1 = assign_group_splits(groups, seed=42)
    result2 = assign_group_splits(groups, seed=42)
    assert result1 == result2


def test_different_seed_can_change_assignment():
    groups = [f"group_{i}" for i in range(200)]
    result_a = assign_group_splits(groups, seed=1)
    result_b = assign_group_splits(groups, seed=2)
    assert result_a != result_b


def test_split_ratio_approximate_over_many_groups():
    groups = [f"group_{i}" for i in range(2000)]
    result = assign_group_splits(groups, SplitRatios(0.7, 0.2, 0.1), seed=42)
    counts = {"train": 0, "val": 0, "test": 0}
    for split in result.values():
        counts[split] += 1
    total = len(groups)
    assert abs(counts["train"] / total - 0.7) < 0.05
    assert abs(counts["val"] / total - 0.2) < 0.05
    assert abs(counts["test"] / total - 0.1) < 0.05


def test_group_never_leaks_across_splits():
    # 3 images per group; if leakage existed they could land in different splits.
    items = []
    item_to_group = {}
    for g in range(50):
        for i in range(3):
            item_id = f"g{g}_img{i}"
            items.append((item_id, f"group_{g}"))
            item_to_group[item_id] = f"group_{g}"

    result = split_items(items, seed=42)
    errors = verify_no_leakage(result, item_to_group)
    assert errors == []


def test_verify_no_leakage_detects_a_real_leak():
    result = {"train": ["img1"], "val": ["img2"], "test": []}
    item_to_group = {"img1": "same_group", "img2": "same_group"}
    errors = verify_no_leakage(result, item_to_group)
    assert len(errors) == 1
    assert "same_group" in errors[0]
