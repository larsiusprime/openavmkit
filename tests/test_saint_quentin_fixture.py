from pathlib import Path

from openavmkit.utilities.settings import load_settings


def test_saint_quentin_fixture_loads_with_metric_model_config():
    settings_path = (
        Path(__file__).parent / "data" / "fr-02-saint_quentin" / "settings.json"
    )

    settings = load_settings(settings_file=str(settings_path))

    assert settings["locality"]["slug"] == "fr-02-saint_quentin"
    assert settings["locality"]["units"] == "metric"

    model_groups = settings["modeling"]["model_groups"]
    assert model_groups["all"]["filter"] == [">=", "land_area_sqm", 0]

    instructions = settings["modeling"]["instructions"]
    assert instructions["dep_var"] == "sale_price_time_adj"
    assert instructions["dep_var_test"] == "sale_price_time_adj"
    assert instructions["time_adjustment"]["period"] == "M"
    assert instructions["main"]["run"] == ["naive_area", "local_area", "lightgbm"]
    assert instructions["vacant"]["run"] == ["naive_area", "local_area", "lightgbm"]

    naive_area = settings["modeling"]["models"]["main"]["naive_area"]
    assert naive_area["model"] == "naive_area"
    assert "land_area_sqm" in naive_area["ind_vars"]
    assert "bldg_area_finished_sqm" in naive_area["ind_vars"]

    lightgbm = settings["modeling"]["models"]["main"]["lightgbm"]
    assert lightgbm["model"] == "lightgbm"
    assert lightgbm["engine"] == "lightgbm"
    assert lightgbm["n_trials"] == 10
    assert "neighborhood" in lightgbm["ind_vars"]

    vacant_lightgbm = settings["modeling"]["models"]["vacant"]["lightgbm"]
    assert "land_area_sqm" in vacant_lightgbm["ind_vars"]
    assert "bldg_area_finished_sqm" not in vacant_lightgbm["ind_vars"]

    sales_load = settings["data"]["load"]["sales"]["load"]
    assert sales_load["valid_for_ratio_study"] == [
        "valid_for_ratio_study",
        "boolean",
        "na_false",
    ]
    assert sales_load["valid_for_land_ratio_study"] == [
        "valid_for_land_ratio_study",
        "boolean",
        "na_false",
    ]
