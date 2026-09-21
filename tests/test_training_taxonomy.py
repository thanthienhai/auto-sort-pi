import pytest

from training.lib.taxonomy import (
    MappingEngine,
    MappingType,
    TaxonomyError,
    coverage_status,
    load_taxonomy,
    normalize_label,
)


def test_taxonomy_loads_16_classes_in_order():
    tax = load_taxonomy()
    assert tax.num_classes == 16
    assert tax.classes[0] == "can"
    assert tax.classes[15] == "broken_glass"
    assert tax.id_for("pet_bottle") == 9


def test_taxonomy_class_ids_are_0_to_15():
    tax = load_taxonomy()
    for i, name in enumerate(tax.classes):
        assert tax.id_for(name) == i
        assert tax.name_for(i) == name
        assert tax.is_valid_class_id(i)
    assert not tax.is_valid_class_id(16)
    assert not tax.is_valid_class_id(-1)


def test_taxonomy_group_mapping_matches_repo_config():
    tax = load_taxonomy()
    assert tax.group_for("can") == 1
    assert tax.group_for("cardboard") == 2
    assert tax.group_for("pet_bottle") == 3
    assert tax.group_for("glass_bottle") == 4
    assert tax.group_name_for("can") == "METAL"
    assert set(tax.classes_in_group(4)) == {"glass_bottle", "glass_jar", "broken_glass"}


def test_taxonomy_rejects_wrong_class_count(tmp_path):
    bad = tmp_path / "model_classes.json"
    bad.write_text('["a", "b"]', encoding="utf-8")
    mapping = tmp_path / "class_mapping.json"
    mapping.write_text('{"a": 1, "b": 1}', encoding="utf-8")
    with pytest.raises(TaxonomyError):
        load_taxonomy(bad, mapping)


def test_taxonomy_rejects_missing_group_mapping(tmp_path):
    classes = tmp_path / "model_classes.json"
    classes.write_text(
        '["can","tin_can","aluminum_foil","metal_cap","cardboard","paper_cup",'
        '"tetra_pak","paper_box","newspaper","pet_bottle","plastic_cup",'
        '"detergent_bottle","nylon_bag","glass_bottle","glass_jar","broken_glass"]',
        encoding="utf-8",
    )
    mapping = tmp_path / "class_mapping.json"
    mapping.write_text('{"can": 1}', encoding="utf-8")
    with pytest.raises(TaxonomyError):
        load_taxonomy(classes, mapping)


@pytest.mark.parametrize(
    "count,expected",
    [(0, "MISSING"), (50, "VERY_LOW"), (150, "LOW"), (500, "MODERATE"), (2000, "GOOD")],
)
def test_coverage_status_thresholds(count, expected):
    thresholds = {"missing_max": 100, "low_max": 300, "moderate_max": 1000}
    assert coverage_status(count, thresholds) == expected


@pytest.fixture(scope="module")
def engine():
    return MappingEngine.load(load_taxonomy())


def test_normalize_label_collapses_separators():
    assert normalize_label("PET_Bottle") == "pet bottle"
    assert normalize_label("Tetra-Pak") == "tetra pak"
    assert normalize_label("  Glass   Jar ") == "glass jar"


def test_exact_mapping_auto_matches_taxonomy_class_name(engine):
    result = engine.resolve("Cardboard")
    assert result.mapping_type == MappingType.EXACT
    assert result.target_class == "cardboard"
    assert result.confidence == 1.0


def test_safe_alias_mapping_from_global_table(engine):
    result = engine.resolve("Plastic Bottle")
    assert result.mapping_type == MappingType.SAFE_ALIAS
    assert result.target_class == "pet_bottle"

    result2 = engine.resolve("Tetra-Pak")
    assert result2.mapping_type == MappingType.SAFE_ALIAS
    assert result2.target_class == "tetra_pak"


@pytest.mark.parametrize(
    "label",
    ["Can", "Metal Can", "Drink Can", "Food Can", "Aluminium Can", "Tin Can"],
)
def test_dangerous_can_labels_forced_review_required(engine, label):
    result = engine.resolve(label)
    assert result.mapping_type == MappingType.REVIEW_REQUIRED
    assert result.target_class is None
    assert "can" in result.candidate_targets or "tin_can" in result.candidate_targets


def test_bare_glass_and_paper_are_review_required(engine):
    for label in ("Glass", "Paper", "Paperboard"):
        result = engine.resolve(label)
        assert result.mapping_type == MappingType.REVIEW_REQUIRED
        assert result.target_class is None


def test_glass_bottle_and_glass_jar_are_kept_distinct(engine):
    bottle = engine.resolve("Glass Bottle")
    jar = engine.resolve("Glass Jar")
    assert bottle.target_class == "glass_bottle"
    assert jar.target_class == "glass_jar"
    assert bottle.target_class != jar.target_class


def test_unknown_label_defaults_to_review_required(engine):
    result = engine.resolve("some totally unknown gizmo")
    assert result.mapping_type == MappingType.REVIEW_REQUIRED
    assert result.target_class is None
    assert result.rule_source == "default_unknown"


def test_dataset_override_takes_precedence_over_global(engine):
    override = {
        "Can": {
            "target_class": "tin_can",
            "mapping_type": "SAFE_ALIAS",
            "confidence": 0.9,
            "note": "This specific dataset only has tin cans",
        }
    }
    result = engine.resolve("Can", dataset_override=override)
    assert result.mapping_type == MappingType.SAFE_ALIAS
    assert result.target_class == "tin_can"
    assert result.rule_source == "dataset_manifest"


def test_camelcase_label_without_separator_is_handled(engine):
    # "PlasticCups" (real Roboflow label, no space/hyphen/underscore) can't be
    # split by normalize_label — it must be an explicit table entry rather
    # than relying on auto-exact matching, otherwise it would silently fall
    # through to the default REVIEW_REQUIRED bucket with no candidates.
    result = engine.resolve("PlasticCups")
    assert result.mapping_type == MappingType.SAFE_ALIAS
    assert result.target_class == "plastic_cup"


def test_out_of_taxonomy_labels_are_review_required_not_dropped(engine):
    # Labels genuinely outside the 16-class taxonomy (face mask, organic
    # waste, styrofoam, ...) must still surface via REVIEW_REQUIRED rather
    # than being silently ignored/dropped by the mapping engine.
    for label in ("Face Mask", "Styrofoam", "Peel", "Rags", "Pile of Leaves"):
        result = engine.resolve(label)
        assert result.mapping_type == MappingType.REVIEW_REQUIRED
        assert result.target_class is None


def test_engine_validates_rule_targets_exist():
    taxonomy = load_taxonomy()
    with pytest.raises(ValueError):
        from training.lib.taxonomy import _Rule

        bad_engine = MappingEngine(taxonomy, {"bogus label": _Rule(target_class="not_a_class", mapping_type=MappingType.EXACT, confidence=1.0)})
        bad_engine._validate_rules()
