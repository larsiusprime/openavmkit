"""The settings template's default field classification."""

import json
from importlib import resources


def _template() -> dict:
    path = resources.files("openavmkit.resources.settings").joinpath("settings.template.json")
    return json.loads(path.read_text())


def test_dates_are_not_improvement_categories():
    """Improvement categories get "UNKNOWN" where blank, which turns a date column into text. The permit
    dates sit with the other dates instead."""
    fc = _template()["+field_classification"]
    impr = fc["+impr"]["+categorical"]
    other = fc["+other"]["+categorical"]
    for date in ("reno_date", "demo_date", "sale_date"):
        assert date not in impr
        assert date in other
