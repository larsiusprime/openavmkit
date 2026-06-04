"""collapse_sparse_categories output_field variants must inherit their source field's
categorical classification, so tree models (LightGBM/XGBoost/CatBoost) receive them as
`category` rather than raw strings."""
from openavmkit.utilities.settings import get_fields_categorical


def _settings(collapse):
    return {
        "field_classification": {
            "land": {"categorical": ["subdivision", "neighborhood"], "numeric": ["land_area_sqft"]},
        },
        "data": {"process": {"collapse_sparse_categories": collapse}},
    }


def test_output_field_inherits_categorical():
    s = _settings({"subdivision": {"sales_min": 5, "univ_min": 25, "output_field": "subdivision_collapsed"}})
    cats = get_fields_categorical(s)
    assert "subdivision" in cats
    assert "subdivision_collapsed" in cats  # inherited


def test_no_output_field_no_extra():
    s = _settings({"subdivision": {"sales_min": 5, "univ_min": 25}})  # in-place collapse
    cats = get_fields_categorical(s)
    assert "subdivision" in cats
    assert "subdivision_collapsed" not in cats


def test_output_field_of_non_categorical_source_not_added():
    # Collapsing a field that isn't classified categorical shouldn't promote its variant.
    s = _settings({"land_area_sqft": {"sales_min": 5, "univ_min": 25, "output_field": "land_area_sqft_collapsed"}})
    cats = get_fields_categorical(s)
    assert "land_area_sqft_collapsed" not in cats


def test_comment_keys_ignored():
    s = _settings({
        "__comment": "ignore me",
        "subdivision": {"sales_min": 5, "univ_min": 25, "output_field": "subdivision_collapsed"},
    })
    cats = get_fields_categorical(s)
    assert "subdivision_collapsed" in cats
