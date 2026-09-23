"""Tests for the second-pass invalid-sales scrub (``data.process.invalid_sales``).

Covers the absolute price floors and the relative ``invalid_sales.calc`` hook that lets a
filter reference a derived field (e.g. ``sale_to_assr_ratio``) computed on the hydrated sales
frame — enabling relative rules like "sale_price < 0.5 * assr_market_value" that the filter
DSL cannot express inline.
"""
import pandas as pd

from openavmkit.cleaning import filter_invalid_sales
from openavmkit.data import SalesUniversePair


def _sup():
    # assr_market_value lives on the universe (a parcel/CAMA field) and is merged onto sales
    # during hydration inside filter_invalid_sales.
    universe = pd.DataFrame({
        "key": ["0", "1", "2", "3", "4"],
        "assr_market_value": [200000.0, 100000.0, 0.0, 300000.0, 50000.0],
    })
    sales = pd.DataFrame({
        "key": ["0", "1", "2", "3", "4"],
        "key_sale": ["0-s", "1-s", "2-s", "3-s", "4-s"],
        "sale_price": [40000.0, 95000.0, 30000.0, 500.0, 20000.0],
        "valid_sale": [True, True, True, True, True],
        "vacant_sale": [False, False, False, False, True],
    })
    return SalesUniversePair(sales=sales, universe=universe)


def _settings():
    return {
        "data": {
            "process": {
                "invalid_sales": {
                    "enabled": True,
                    "calc": {
                        "sale_to_assr_ratio": ["/0", "sale_price", "assr_market_value"],
                    },
                    "filter": [
                        "or",
                        ["<", "sale_price", 1000],
                        ["and", ["==", "vacant_sale", False], ["<", "sale_price", 5000]],
                        ["and", ["==", "vacant_sale", False], [">", "assr_market_value", 0],
                         ["<", "sale_to_assr_ratio", 0.5]],
                    ],
                }
            }
        }
    }


def test_invalid_sales_relative_and_absolute_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # write_cache writes under cwd; keep it in the tmp dir
    sup = filter_invalid_sales(_sup(), _settings())
    survivors = set(sup["sales"]["key_sale"].values)

    # key 0: ratio 0.20 (40k / 200k), improved -> relative rule -> INVALID
    # key 1: ratio 0.95 (95k / 100k) -> valid
    # key 2: assr_market_value == 0 -> relative rule guarded off; price 30k > 5k -> valid
    # key 3: sale_price 500 < 1000 -> absolute rule -> INVALID
    # key 4: vacant sale (ratio 0.40) -> both vacant-guarded rules skip it -> valid
    assert survivors == {"1-s", "2-s", "4-s"}


def test_invalid_sales_calc_absent_is_noop_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Without the relative clause, only the absolute floors apply: key 3 drops, key 0 survives.
    s = _settings()
    s["data"]["process"]["invalid_sales"]["filter"] = [
        "or",
        ["<", "sale_price", 1000],
        ["and", ["==", "vacant_sale", False], ["<", "sale_price", 5000]],
    ]
    del s["data"]["process"]["invalid_sales"]["calc"]
    sup = filter_invalid_sales(_sup(), s)
    survivors = set(sup["sales"]["key_sale"].values)
    assert survivors == {"0-s", "1-s", "2-s", "4-s"}
