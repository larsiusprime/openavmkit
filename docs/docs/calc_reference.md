# The `calc` expression language

OpenAVMKit's `settings.json` lets you derive new columns, normalize messy values, and define filter expressions without writing Python. This is done through `calc` blocks, which use a small Lisp-style expression language: every expression is a JSON list whose first element is an operator and whose remaining elements are operands.

This page is the authoritative reference for the operator set. For where `calc` is *used* in the broader settings tree, see [Advanced settings reference](advanced_settings.md). For an introduction to the calc system in context, see [the tutorial § B.4](tutorial.md#b4-author-a-minimum-viable-settingsjson).

---

## 1. Where `calc` is used

`calc` blocks appear in several places in `settings.json`. They all use the same expression language documented here, but the timing and the data they operate on differ:

| Location | Runs | Operates on | Typical use |
| --- | --- | --- | --- |
| `data.load.<id>.calc` | At file-load time, immediately after column renames and dtype coercion | The just-loaded file's columns (renamed) | Convert units (acres → sqft), build composite keys, normalize categorical values, mark valid sales |
| `data.process.enrich.calc.universe` / `.sales` | After all enrichments are complete | The fully-enriched universe / sales DataFrames | Derive features that depend on enriched columns (Census fields, distance fields, GIS area, etc.) |

> **Key point:** `data.load.<id>.calc` only sees columns that exist in the file you just loaded. `data.process.enrich.calc.*` sees the full enriched picture. If a calc you wrote in `data.load` says "field not found," it's almost always because the field comes from enrichment — move the calc to `data.process.enrich.calc`.

A sibling system, **`tweak`**, handles targeted per-row overrides — see [§ 5 Tweaks](#5-tweaks) below.

---

## 2. Expression syntax

Every `calc` entry is a JSON list of the form:

```json
["<operator>", <operand1>, <operand2>, ...]
```

A `calc` block is a dictionary mapping new column names to expressions:

```json
{
    "calc": {
        "land_area_sqft": ["*", "TOTAL_ACRES", 43560],
        "valid_sale": ["?", ["==", "SALE_STATUS", "str:QUALIFIED"]]
    }
}
```

Each entry produces a column with the given name, evaluated row-by-row.

### Operand types

How each operand is interpreted:

| Operand | Meaning | Example |
| --- | --- | --- |
| Bare string (no prefix) | Column name — resolves to the named column's values for each row | `"TOTAL_ACRES"` |
| `"str:..."` | String literal (everything after the prefix) | `"str:QUALIFIED"` |
| Number (int or float) | Numeric literal | `43560` |
| Nested list | Recursive sub-expression — the result of evaluating it | `["+", "BLDG_AREA_1", "BLDG_AREA_2"]` |
| `"$$path.to.value"` | Settings reference resolved by the preprocessor before `calc` runs (see [advanced_settings.md § 1.2](advanced_settings.md#12-variable-references-via-pathtovalue)) | `"$$ref.calc.qual_to_perc_map"` |

> **Why the `str:` prefix exists.** A bare string is interpreted as a column name. If you want a literal string, you must say so explicitly: `"str:QUALIFIED"`, not `"QUALIFIED"`. Without the prefix the calc engine would look for a column called `QUALIFIED` and raise an error.

---

## 3. Operator reference

### 3.1 Control flow

#### `?` — selection filter

```json
["?", <filter_expression>]
```

Returns a boolean Series — `True` for rows where the filter expression matches, `False` otherwise. Used to derive boolean fields like `valid_sale`.

```json
"valid_sale": ["?", ["==", "SALE_STATUS", "str:QUALIFIED"]]
```

#### `where` — conditional

```json
["where", <filter_expression>, <value_if_true>, <value_if_false>]
```

Returns a Series whose values come from `<value_if_true>` for matching rows and `<value_if_false>` for non-matching rows.

```json
"sale_price_clean": ["where",
    ["==", "DEED_TYPE", "str:WARRANTY"],
    "SALE_PRICE",
    0
]
```

#### `values` — multi-column packing

```json
["values", <col1>, <col2>, ...]
```

Returns a sub-DataFrame containing the specified columns. Mostly used as input to operators that take multiple columns at once (notably `join`).

### 3.2 Type coercion (unary)

| Operator | Effect | Example |
| --- | --- | --- |
| `asint` | Cast to nullable int64 | `["asint", "RAW_COUNT"]` |
| `asfloat` | Cast to float | `["asfloat", "RAW_PRICE"]` |
| `asstr` | Cast to string | `["asstr", "PARCEL_ID"]` |

### 3.3 Math (unary)

| Operator | Effect | Example |
| --- | --- | --- |
| `floor` | `np.floor` (round down) | `["floor", "AREA"]` |
| `ceil` | `np.ceil` (round up) | `["ceil", "AREA"]` |
| `round` | `np.round` (round half-to-even) | `["round", "AREA"]` |
| `abs` | `np.abs` (absolute value) | `["abs", "DELTA"]` |
| `set` | Pass-through (returns the operand unchanged) | `["set", "OTHER_FIELD"]` |

### 3.4 Logic (unary)

| Operator | Effect | Example |
| --- | --- | --- |
| `not` | Bitwise NOT (flip booleans) | `["not", "is_vacant"]` |

### 3.5 String (unary)

| Operator | Effect | Example |
| --- | --- | --- |
| `strip` | Remove leading and trailing whitespace | `["strip", "ADDRESS"]` |
| `striplzero` | Remove leading zeros | `["striplzero", "PARCEL_ID"]` |
| `stripkey` | Remove all whitespace and leading zeros — for normalizing parcel keys | `["stripkey", "PARCEL_ID"]` |

### 3.6 Comparison (binary)

| Operator | Effect | Notes |
| --- | --- | --- |
| `==` | Equals | When the right side is a string, both sides are stripped before comparison |
| `!=` | Not equals | Same string-stripping behavior |

```json
["==", "SALE_STATUS", "str:QUALIFIED"]
["!=", "ZONING", "str:EXEMPT"]
```

### 3.7 Arithmetic (binary)

| Operator | Effect | Example |
| --- | --- | --- |
| `+` | Add | `["+", "BLDG_AREA_1", "BLDG_AREA_2"]` |
| `-` | Subtract | `["-", "TOTAL", "PORTION"]` |
| `*` | Multiply | `["*", "TOTAL_ACRES", 43560]` |
| `/` | Divide (raises on divide-by-zero) | `["/", "PRICE", "AREA"]` |
| `//` | Integer divide (cast to int) | `["//", "AGE_YEARS", 10]` |
| `/0` | Z-safe divide (returns `NaN` on divide-by-zero instead of erroring) | `["/0", "PRICE", "AREA"]` |
| `min` | Element-wise minimum | `["min", "VALUE_A", "VALUE_B"]` |
| `max` | Element-wise maximum | `["max", "VALUE_A", "VALUE_B"]` |
| `round_nearest` | Round LHS to the nearest multiple of RHS | `["round_nearest", "bldg_condition_num", 5]` |

### 3.8 Logic (binary)

| Operator | Effect | Example |
| --- | --- | --- |
| `and` | Bitwise AND on boolean Series | `["and", "is_residential", "is_owner_occupied"]` |
| `or` | Bitwise OR on boolean Series | `["or", "is_townhome", "is_condo"]` |

### 3.9 Set membership

| Operator | Effect | Example |
| --- | --- | --- |
| `isin` | True for rows whose LHS appears in the RHS list | `["isin", "ZONING", ["str:R1", "str:R2", "str:R3"]]` |

### 3.10 Mapping and gap-filling

#### `map` — dictionary lookup

```json
["map", <column_or_expr>, <dict>]
```

Looks each value up in the dict and substitutes. Values not in the dict are passed through unchanged.

```json
["map", "GRADE", {
    "A+": "Excellent",
    "A": "Excellent",
    "B": "Good",
    "C": "Fair"
}]
```

Dictionaries are commonly defined once in a `ref` block and referenced via `$$ref.calc.<name>` to keep `settings.json` DRY.

#### `fillempty` — replace blanks

```json
["fillempty", <column>, <fill_value>]
```

Treats `NaN`, empty strings, and the case-insensitive strings `"none"`, `"null"`, `"n/a"`, `"na"`, `"nan"`, `"<na>"` as missing, and replaces them.

#### `fillna` — replace pandas NaN

```json
["fillna", <column>, <fill_value>]
```

Stricter than `fillempty` — only replaces actual `NaN`. Use `fillempty` for messy real-world data.

### 3.11 String replace and search

| Operator | Effect | Example |
| --- | --- | --- |
| `replace` | Apply a dict of literal find/replace pairs to a string column | `["replace", "GRADE", {"+": "", "-": ""}]` |
| `replace_regex` | Same as `replace`, but the keys are regex patterns | `["replace_regex", "ADDRESS", {"\\s+": " "}]` |
| `contains` | True if LHS string contains RHS substring | `["contains", "DESC", "str:CONDO"]` |
| `contains_case_insensitive` | Case-insensitive variant of `contains`. RHS may also be a list of needles, in which case the result is true if LHS contains *any* of them. | Single: `["contains_case_insensitive", "DESC", "str:condo"]`<br>List: `["contains_case_insensitive", "DESC", ["str:condo", "str:townhome", "str:apartment"]]` |

### 3.12 String split, join, and slice

| Operator | Effect | Example |
| --- | --- | --- |
| `split_before` | Take everything before the first occurrence of the separator | `["split_before", "GRADE", "str:-"]` |
| `split_after` | Take everything after the first occurrence of the separator | `["split_after", "GRADE", "str:  "]` |
| `join` | Concatenate the columns from a `values` expression with a separator | see below |
| `substr` | Slice a string by index | `["substr", "PARCEL_ID", {"left": 0, "right": 6}]` |

`join` operates on multiple columns at once, so the LHS must be a `values` expression:

```json
["join",
    ["values", "REID", ["asstr", ["datetimestr", "SALE_DATE", "str:%m/%d/%Y"]]],
    "str:---"
]
```

This produces a sale key like `"123456---2024-03-15"`.

`substr` accepts a dict for the right-hand side: `{"left": <start>, "right": <end>}`. Either bound is optional.

### 3.13 Date

| Operator | Effect | Example |
| --- | --- | --- |
| `datetime` | Parse a string column as a datetime using the given format | `["datetime", "SALE_DATE", "str:%m/%d/%Y"]` |
| `datetimestr` | Parse and reformat as ISO date string (`YYYY-MM-DD`) | `["datetimestr", "SALE_DATE", "str:%m/%d/%Y %H:%M"]` |

### 3.14 Geometry

#### `geo_area` — compute area from geometry

```json
["geo_area", <unit>]
```

Computes the area of the row's `geometry` in the specified unit. Requires a GeoDataFrame with a `geometry` column. Valid units: `sqft`, `sqm`, `acres`, `sqkm`, `hectares`. The DataFrame is reprojected to an equal-area CRS before computing area.

```json
"land_area_sqft": ["geo_area", "str:sqft"]
```

---

## 4. Worked examples

### Convert acres to square feet

```json
"land_area_sqft": ["*", "TOTAL_ACRES", 43560]
```

### Build a composite sale key

```json
"key_sale": ["join",
    ["values",
        "REID",
        ["asstr", ["datetimestr", "SALE_DATE", "str:%m/%d/%Y %H:%M"]]
    ],
    "str:---"
]
```

### Normalize building grade text

```json
"bldg_quality_txt": ["split_before",
    ["split_before",
        ["split_before", ["map", "GRADE", "$$ref.calc.qual_to_perc_map"], "str:  "],
        "str:-"
    ],
    "str:+"
]
```

### Mark valid sales

```json
"valid_sale": ["?", ["==", "SALE_STATUS", "str:QUALIFIED"]]
```

### Compose multiple selection criteria

```json
"valid_residential_sale": ["?",
    ["and",
        ["==", "SALE_STATUS", "str:QUALIFIED"],
        ["isin", "ZONING", ["str:R1", "str:R2", "str:R3"]]
    ]
]
```

### Convert grade letter to a numeric quality score

```json
"bldg_quality_num": ["/",
    ["asfloat",
        ["replace",
            ["split_after", ["map", "GRADE", "$$ref.calc.qual_to_perc_map"], "str:  "],
            {"%": ""}
        ]
    ],
    100
]
```

This example, drawn from the [us-nc-guilford settings](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-nc-guilford/in/settings.json), shows nested operations: take a string like `"B+  85%"`, look it up in a reference map, split off the percentage, strip the `%`, cast to float, divide by 100.

### Compute GIS area at load time

```json
"land_area_sqft": ["geo_area", "str:sqft"]
```

(For most use cases, the basic-geo enrichment step in `data.process.enrich.basic` already produces `land_area_gis_<unit>` automatically — see [advanced_settings.md § 4.1](advanced_settings.md#41-basic-geometric-enrichment-dataprocessenrichbasic). Use `geo_area` in a `calc` only when you need it before enrichment runs.)

---

## 5. Tweaks

`tweak` is a sibling system that handles targeted per-row overrides. It's simpler than `calc` — no expression language, just key-value substitutions.

A `tweak` block is a list of entries, each with `field`, `key`, and `values`:

```json
{
    "tweak": [
        {
            "field": "bldg_quality_num",
            "key": "PARCEL_ID",
            "values": {
                "12345": 0.85,
                "67890": 0.90
            }
        }
    ]
}
```

This says: "for the row where `PARCEL_ID == 12345`, set `bldg_quality_num` to `0.85`; for the row where `PARCEL_ID == 67890`, set it to `0.90`." All other rows are unchanged.

**Use tweaks for**: jurisdiction-specific manual overrides — known data errors on specific parcels, hand-corrected values that don't fit a general rule.

**Don't use tweaks for**: rule-based transformations. Those belong in `calc` or `data.process.fill`.

---

## 6. Common pitfalls

- **"Field not found" errors.** A bare string operand is interpreted as a column name. If you meant a string literal, prefix it with `str:`. If you meant a reference to a value defined elsewhere in `settings.json`, prefix it with `$$`.
- **Calc references an enriched field.** `data.load.<id>.calc` only sees the just-loaded file. If your calc references a Census or distance column, move it to `data.process.enrich.calc`.
- **Type errors in arithmetic.** A column loaded as a string (because of a non-numeric character somewhere in the data) won't multiply or divide cleanly. Use `asfloat` or `asint` first, or use `replace` to scrub the offending characters.
- **Division by zero.** `/` raises an error on zeros. Use `/0` for z-safe division that returns `NaN` instead.
- **Map values that don't match.** `map` passes through unmatched values unchanged. If you need to error or fall back to a default, add a follow-up `fillna` or `fillempty`.

---

## 7. See also

- [The tutorial § B.4](tutorial.md#b4-author-a-minimum-viable-settingsjson) — calc in the context of authoring your first `settings.json`
- [Advanced settings reference](advanced_settings.md) — settings.json structure, preprocessor, enrichment, fill rules, caching
- [openavmkit/calculations.py](https://github.com/larsiusprime/openavmkit/blob/master/openavmkit/calculations.py) — implementation
- Canonical examples that use calc heavily:
    - [us-nc-guilford](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-nc-guilford/in/settings.json)
    - [us-va-petersburgcity](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-va-petersburgcity/in/settings.json)
    - [us-pa-philadelphia](https://github.com/larsiusprime/openavmkit/blob/master/notebooks/pipeline/data/us-pa-philadelphia/in/settings.json)
