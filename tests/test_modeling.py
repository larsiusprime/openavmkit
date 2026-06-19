import tempfile

import numpy as np
import pandas as pd

from openavmkit.modeling import simple_ols, _greedy_nn_limited, simple_mra
from openavmkit.utilities.assertions import lists_are_equal
from openavmkit.utilities.stats import calc_vif
import os

from openavmkit.modeling import (
    DataSplit,
    run_mra,
    run_multi_mra,
    run_ngboost,
    SingleModelResults,
    write_model_parameters,
)

import warnings

def test_vif():
	
	data = {
		"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	}
	df = pd.DataFrame(data)
	
	data2 = {
		"a": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
		"b": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
		"c": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
	}
	df2 = pd.DataFrame(data2)
	
	vif = calc_vif(df)
	vif2 = calc_vif(df2)


def test_simple_ols():

	data = {
		"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
		"b": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
	}
	df = pd.DataFrame(data)

	results = simple_ols(df, "a", "b")

	assert results["slope"] - 2.0 < 1e-6
	assert results["intercept"] - 4.0 < 1e-6
	assert results["r2"] - 1.0 < 1e-6
	assert results["adj_r2"] - 1.0 < 1e-6


def test_nearest_neighbor():

	def make_snake(lat_base=29.749907, lon_base=-95.358421,
			lat_size=0.001, lon_size=0.010,
			n_cols=100, n_rows=10):

		lats, lons, expected = [], [], []
		for col in range(n_cols):
			lon = lon_base + col * lon_size

			# even columns go bottom→top, odd go top→bottom
			if col % 2 == 0:
				rows = range(n_rows)
			else:
				rows = range(n_rows - 1, -1, -1)

			for row in rows:
				lat = lat_base + (row + 1) * lat_size
				lats.append(lat)
				lons.append(lon)
				expected.append(len(expected))

		return lats, lons, expected

	# in your test
	lats, lons, expected = make_snake()
	order = _greedy_nn_limited(lats, lons, start_idx=0, k=16)
	assert list(order) == expected


def test_mra_constant():
    
    data = {
        "key": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
        "key_sale": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"],
        "bldg_area_finished_sqft": [
            1000, 1000, 1000, 1000, 
            1500, 1500, 1500, 1500, 
            2000, 2000, 2000, 2000,
            2500, 2500, 2500, 2500,
            3000, 3000, 3000, 3000
        ],
        "land_area_sqft": [
            10000, 10000, 10000, 10000,
            20000, 20000, 20000, 20000,
            30000, 30000, 30000, 30000,
            40000, 40000, 40000, 40000,
            50000, 50000, 50000, 50000
        ],
        "bldg_type_A": [
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0
        ],
        "bldg_type_B": [
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1
        ],
        "location_A": [
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0
        ],
        "location_B": [
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1
        ],
        "flat": [
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, 1, 0  #putting a zero here will make this field a constant in the test set but not the train set, which is an edge case we need to guard against
        ],
        "model_group": [
            "a", "a", "a", "a",
            "a", "a", "a", "a",
            "a", "a", "a", "a",
            "a", "a", "a", "a",
            "a", "a", "a", "a"
        ]
    }
    
    df = pd.DataFrame(data)
    df["sale_price"] = df["bldg_area_finished_sqft"] * 20
    df["sale_price"] += df["bldg_type_A"] * df["bldg_area_finished_sqft"] * 2
    df["sale_price"] += df["bldg_type_B"] * df["bldg_area_finished_sqft"] * 5
    df["sale_price"] += df["location_A"] * df["land_area_sqft"] * 1
    df["sale_price"] += df["location_B"] * df["land_area_sqft"] * 2
    df["valid_sale"] = True
    df["vacant_sale"] = False
    df["sale_date"] = "2025-01-01"
    df["sale_date"] = pd.to_datetime(df["sale_date"], format="%Y-%m-%d")
    df["is_vacant"] = False
    df["valid_for_ratio_study"] = True
    df["sale_age_days"] = 0
    
    ind_vars = ["bldg_area_finished_sqft", "land_area_sqft", "bldg_type_A", "bldg_type_B", "location_A", "location_B", "flat"]
    df_sales = df.copy()
    df_universe = df[(["key","is_vacant"]+ind_vars)].copy()
    test_keys = ["0", "1", "2", "3", "4", "5"]
    train_keys = ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
    
    ds = DataSplit(
        "",
        df_sales,
        df_universe,
        "a",
        {},
        "sale_price",
        "sale_price",
        ind_vars,
        [],
        {},
        test_keys,
        train_keys
    )
    
    
    run_mra(ds, intercept=True)


def test_ngboost_smoke():
    # Smoke test: NGBoost runs end-to-end with a categorical ind_var, returns a
    # SingleModelResults, and surfaces a non-negative per-parcel prediction_std.
    n = 24
    keys = [str(i) for i in range(n)]
    rng = np.arange(n)
    data = {
        "key": keys,
        "key_sale": keys,
        "bldg_area_finished_sqft": (1000 + (rng % 6) * 250).astype(float),
        "land_area_sqft": (10000 + (rng % 6) * 5000).astype(float),
        # categorical predictor — exercises the numeric-encoding path
        "neighborhood": [["north", "south", "east"][i % 3] for i in range(n)],
        "is_vacant": False,
        "model_group": ["a"] * n,
    }
    df = pd.DataFrame(data)
    df["sale_price"] = (
        df["bldg_area_finished_sqft"] * 20
        + df["land_area_sqft"] * 1.5
        + df["neighborhood"].map({"north": 50000, "south": 0, "east": 25000})
    ).astype(float)
    df["valid_sale"] = True
    df["vacant_sale"] = False
    df["sale_date"] = pd.to_datetime("2025-01-01", format="%Y-%m-%d")
    df["valid_for_ratio_study"] = True
    df["sale_age_days"] = 0

    ind_vars = ["bldg_area_finished_sqft", "land_area_sqft", "neighborhood"]
    fields_cat = ["neighborhood"]
    df_sales = df.copy()
    df_universe = df[["key", "is_vacant"] + ind_vars].copy()
    test_keys = keys[:6]
    train_keys = keys[6:]

    ds = DataSplit(
        "",
        df_sales,
        df_universe,
        "a",
        {},
        "sale_price",
        "sale_price",
        ind_vars,
        fields_cat,
        {},
        test_keys,
        train_keys,
    )

    with tempfile.TemporaryDirectory() as outpath:
        results = run_ngboost(ds, outpath, n_trials=1)

        assert isinstance(results, SingleModelResults)

        preds = results.df_universe["prediction"].to_numpy()
        assert len(preds) == len(df_universe)
        assert np.all(np.isfinite(preds))

        assert "prediction_std" in results.df_universe.columns
        std = results.df_universe["prediction_std"].to_numpy()
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)

        # --- SHAP: exact tree decomposition for mean (loc) and uncertainty (logscale) ---
        model_dir = os.path.join(outpath, results.model_name)
        write_model_parameters(results.model, results, None, model_dir)

        # Both the standard (mean) and the std (uncertainty) SHAP files must exist.
        for fname in [
            "params_universe.csv",
            "contributions_universe.csv",
            "params_std_universe.csv",
            "contributions_std_universe.csv",
        ]:
            assert os.path.exists(os.path.join(model_dir, fname)), f"missing {fname}"

        # Mean additivity: base + Σ contributions reconstructs the prediction (check_delta ~ 0).
        contrib = pd.read_csv(os.path.join(model_dir, "contributions_universe.csv"))
        assert np.max(np.abs(contrib["check_delta"])) < 1e-3 * float(np.mean(preds))

        # Uncertainty additivity: contribution_sum reconstructs log(predictive std).
        contrib_std = pd.read_csv(os.path.join(model_dir, "contributions_std_universe.csv"))
        merged = results.df_universe[["key", "prediction_std"]].merge(
            contrib_std[["key", "contribution_sum"]].assign(key=lambda d: d["key"].astype(str)),
            on="key",
            how="inner",
        )
        assert len(merged) == len(df_universe)
        recon_logstd = merged["contribution_sum"].to_numpy()
        true_logstd = np.log(merged["prediction_std"].to_numpy())
        assert np.max(np.abs(recon_logstd - true_logstd)) < 1e-6


def test_simple_mra():

	data = {
		"key": [0, 1, 2, 3, 4, 5],
		"bldg_area_finished_sqft": [1000, 1500, 2000, 2500, 3000, 3500],
		"land_area_sqft": [10000, 20000, 30000, 40000, 50000, 60000],
		"bldg_type_A": [1, 0, 0, 1, 0, 1],
		"bldg_type_B": [0, 1, 1, 0, 1, 0],
		"location_A": [1, 0, 0, 1, 0, 1],
		"location_B": [0, 1, 1, 0, 1, 0],
	}
	df = pd.DataFrame(data)
	df["location_A"] = df["location_A"] * df["land_area_sqft"]
	df["location_B"] = df["location_B"] * df["land_area_sqft"]
	df["bldg_type_A"] = df["bldg_type_A"] * df["bldg_area_finished_sqft"]
	df["bldg_type_B"] = df["bldg_type_B"] * df["bldg_area_finished_sqft"]

	true_coefs = {
		"bldg_type_A": 10.0,
		"bldg_type_B": 25.0,
		"location_A": 5,
		"location_B": 1
	}

	df["sale_price"] = 0.0

	for coef in true_coefs:
		df["sale_price"] += df[coef] * true_coefs[coef]

	results = simple_mra(df, ["bldg_type_A", "bldg_type_B", "location_A", "location_B"], "sale_price")
	coefs = results["coefs"]

	print(coefs)

	for coef in coefs:
		true_value = true_coefs[coef]
		mra_value = coefs[coef]
		assert abs(true_value - mra_value) < 1e-2, f"Coefficient for {coef} does not match: expected {true_value}, got {mra_value}"


def _log_target_dataset():
    """20-row synthetic SF dataset with strictly positive prices, for log-target tests."""
    n = 20
    keys = [str(i) for i in range(n)]
    sqft = np.array([1000, 1500, 2000, 2500, 3000] * 4, dtype=float)
    land = np.array([10000, 20000, 30000, 40000, 50000] * 4, dtype=float)
    loc_a = np.array(([1, 0] * 10), dtype=float)
    data = {
        "key": keys,
        "key_sale": keys,
        "bldg_area_finished_sqft": sqft,
        "land_area_sqft": land,
        "location_A": loc_a,
        "model_group": ["a"] * n,
    }
    df = pd.DataFrame(data)
    # price is linear & strictly positive (so log is well-defined)
    df["sale_price"] = 20.0 * df["bldg_area_finished_sqft"] + 1.5 * df["land_area_sqft"] \
        + df["location_A"] * 5000.0 + 10000.0
    df["valid_sale"] = True
    df["vacant_sale"] = False
    df["is_vacant"] = False
    df["valid_for_ratio_study"] = True
    df["sale_date"] = pd.to_datetime("2025-01-01", format="%Y-%m-%d")
    df["sale_age_days"] = 0

    ind_vars = ["bldg_area_finished_sqft", "land_area_sqft", "location_A"]
    df_sales = df.copy()
    df_universe = df[["key", "is_vacant"] + ind_vars].copy()
    test_keys = ["0", "1", "2", "3", "4", "5"]
    train_keys = [k for k in keys if k not in test_keys]
    return df_sales, df_universe, ind_vars, test_keys, train_keys


def test_mra_log_param_predictions_price_space_and_positive():
    # run_mra(log=True) fits on log(price) and exponentiates predictions back internally, so the
    # model's output is price-space (not the ~10 log range), strictly positive, and in the same
    # ballpark as an equivalent price-target fit. dep_var stays "sale_price" — no dep_var games.
    df_sales, df_universe, ind_vars, test_keys, train_keys = _log_target_dataset()

    ds_log = DataSplit("", df_sales.copy(), df_universe.copy(), "a", {},
                       "sale_price", "sale_price", ind_vars, [], {}, test_keys, train_keys)
    res_log = run_mra(ds_log, intercept=True, log=True)

    ds_price = DataSplit("", df_sales.copy(), df_universe.copy(), "a", {},
                         "sale_price", "sale_price", ind_vars, [], {}, test_keys, train_keys)
    res_price = run_mra(ds_price, intercept=True, log=False)

    assert res_log.pred_test.y_pred.mean() > 1000
    assert res_log.pred_test.y_pred.mean() < 1e7
    assert (np.asarray(res_log.pred_univ, dtype="float64") > 0).all()
    ratio = res_log.pred_test.y_pred.mean() / res_price.pred_test.y_pred.mean()
    assert 0.33 < ratio < 3.0


def test_mra_log_param_handles_nullable_float_predictions():
    # Pipeline frames use pandas nullable dtypes, so statsmodels returns a Float64 Series for
    # y_pred, which np.exp can't consume ("'float' object has no callable exp method"). The
    # internal exp must coerce to plain float64 first. Cast features to Float64 to reproduce it.
    df_sales, df_universe, ind_vars, test_keys, train_keys = _log_target_dataset()
    for c in ind_vars:
        df_universe[c] = df_universe[c].astype("Float64")
        df_sales[c] = df_sales[c].astype("Float64")

    ds = DataSplit("", df_sales, df_universe, "a", {}, "sale_price", "sale_price",
                   ind_vars, [], {}, test_keys, train_keys)
    res = run_mra(ds, intercept=True, log=True)  # would TypeError on the Float64 Series without coerce

    assert np.isfinite(np.asarray(res.pred_univ, dtype="float64")).all()
    assert (np.asarray(res.pred_univ, dtype="float64") > 0).all()


def test_mra_log_params_written_with_log_prefix(tmp_path):
    # A log model's params/contributions are log-space; they must be written under a "log_" prefix
    # (so they aren't read as dollars AND are auto-excluded from the price-space ensemble/ map
    # consumers, which key off the bare filenames). The log contributions reconcile in log space.
    import os
    from openavmkit.modeling import write_mra_params

    df_sales, df_universe, ind_vars, test_keys, train_keys = _log_target_dataset()
    ds = DataSplit("", df_sales, df_universe, "a", {}, "sale_price", "sale_price",
                   ind_vars, [], {}, test_keys, train_keys)
    res = run_mra(ds, intercept=True, log=True)

    xs = {"test": res.ds.X_test, "sales": res.ds.X_sales, "universe": res.ds.X_univ}
    dfs = {"test": res.df_test, "train": res.df_train,
           "universe": res.df_universe, "sales": res.df_sales}
    write_mra_params(res.model, str(tmp_path), xs, dfs)

    # log_ artifacts exist; bare (price-space) names do NOT (so consumers skip them)
    assert os.path.exists(tmp_path / "log_params.csv")
    assert os.path.exists(tmp_path / "log_contributions_test.csv")
    assert not os.path.exists(tmp_path / "params.csv")
    assert not os.path.exists(tmp_path / "contributions_test.csv")

    # The log contributions file is self-consistent in log space (sum ~ log_prediction)
    con = pd.read_csv(tmp_path / "log_contributions_test.csv")
    assert "log_prediction" in con.columns
    assert "prediction" not in con.columns
    assert np.max(np.abs(con["check_delta"].to_numpy())) < 1e-6


def test_mra_non_log_params_have_no_prefix(tmp_path):
    # A plain (price-space) model keeps the bare filenames — no behavior change.
    import os
    from openavmkit.modeling import write_mra_params

    df_sales, df_universe, ind_vars, test_keys, train_keys = _log_target_dataset()
    ds = DataSplit("", df_sales, df_universe, "a", {}, "sale_price", "sale_price",
                   ind_vars, [], {}, test_keys, train_keys)
    res = run_mra(ds, intercept=True)  # log defaults False

    xs = {"test": res.ds.X_test, "sales": res.ds.X_sales, "universe": res.ds.X_univ}
    dfs = {"test": res.df_test, "train": res.df_train,
           "universe": res.df_universe, "sales": res.df_sales}
    write_mra_params(res.model, str(tmp_path), xs, dfs)

    assert os.path.exists(tmp_path / "params.csv")
    assert os.path.exists(tmp_path / "contributions_test.csv")
    assert not os.path.exists(tmp_path / "log_params.csv")
    con = pd.read_csv(tmp_path / "contributions_test.csv")
    assert "prediction" in con.columns  # price-space reconciliation, unchanged


def test_multi_mra_log_param_price_space(tmp_path):
    # Multi-MRA has its own train/predict path; log=True must likewise produce positive,
    # price-space predictions (it log-transforms the target once and exps at the end).
    df_sales, df_universe, ind_vars, test_keys, train_keys = _log_target_dataset()
    nbhd = ["n1", "n2"] * 10
    df_sales["neighborhood"] = nbhd
    df_universe = df_universe.copy()
    df_universe["neighborhood"] = nbhd

    ds = DataSplit("", df_sales, df_universe, "a", {}, "sale_price", "sale_price",
                   ind_vars, ["neighborhood"], {}, test_keys, train_keys)
    res = run_multi_mra(ds, str(tmp_path), ["neighborhood"],
                        optimize_vars=False, intercept=True, log=True)

    assert np.isfinite(np.asarray(res.pred_univ, dtype="float64")).all()
    assert (np.asarray(res.pred_univ, dtype="float64") > 0).all()
    assert res.pred_test.y_pred.mean() > 1000


def test_lcomp_reconstruct_with_falloffs_matches_full_fit():
    # Safety proof for the lcomp weight_falloff cache: rebuilding the ensemble with the saved
    # per-tree falloffs (skipping the minimize_scalar search) must reproduce a normal fit's
    # predictions bit-for-bit. If this ever diverges (e.g. layeredcompmodel changes its fit),
    # the version guard in run_layeredcomp should fall back to a full fit instead.
    from layeredcompmodel import LayeredCompBaggingModel
    from openavmkit.modeling import (
        _reconstruct_lcomp_with_falloffs, _LCOMP_TREE_COUNT, _LCOMP_SAMPLE_PCT,
        _LCOMP_SPLIT_METRIC, _LCOMP_N_JOBS,
    )
    rng = np.random.default_rng(0)
    nn = 300
    X = pd.DataFrame({
        "land_area_sqft": rng.uniform(3000, 20000, nn),
        "bldg_area_finished_sqft": rng.uniform(800, 4000, nn),
        "bldg_age_years": rng.integers(0, 120, nn).astype(float),
        "neighborhood": rng.integers(0, 8, nn).astype(str).astype(object),
    })
    y = 50000 + 40 * X["bldg_area_finished_sqft"] + rng.normal(0, 15000, nn)
    seed = 42

    full = LayeredCompBaggingModel(
        tree_count=_LCOMP_TREE_COUNT, sample_pct=_LCOMP_SAMPLE_PCT,
        random_state=seed, split_metric=_LCOMP_SPLIT_METRIC, n_jobs=_LCOMP_N_JOBS,
    )
    full.fit(X, y)
    falloffs = [float(est.weight_falloff) for est in full.estimators_]

    recon = _reconstruct_lcomp_with_falloffs(X, y, falloffs, seed)

    Xq = X.head(100)
    np.testing.assert_allclose(full.predict(Xq), recon.predict(Xq), rtol=1e-9, atol=1e-6)


def test_lcomp_save_and_reuse_falloffs_roundtrip(tmp_path):
    # End-to-end: first run saves lcomp_falloffs.json; second run (use_saved_params) reloads it,
    # skips the search, and produces identical universe predictions.
    import os
    from openavmkit.modeling import run_layeredcomp

    rng = np.random.default_rng(1)
    nn = 200
    keys = [str(i) for i in range(nn)]
    df = pd.DataFrame({
        "key": keys, "key_sale": keys,
        "land_area_sqft": rng.uniform(3000, 20000, nn),
        "bldg_area_finished_sqft": rng.uniform(800, 4000, nn),
        "bldg_age_years": rng.integers(0, 120, nn).astype(float),
        "neighborhood": rng.integers(0, 8, nn).astype(str),
        "model_group": ["a"] * nn,
    })
    df["sale_price"] = 50000 + 40 * df["bldg_area_finished_sqft"] + rng.normal(0, 15000, nn)
    df["valid_sale"] = True; df["vacant_sale"] = False; df["is_vacant"] = False
    df["valid_for_ratio_study"] = True
    df["sale_date"] = pd.to_datetime("2025-01-01"); df["sale_age_days"] = 0
    ind_vars = ["land_area_sqft", "bldg_area_finished_sqft", "bldg_age_years", "neighborhood"]
    df_universe = df[["key", "is_vacant"] + ind_vars].copy()
    test_keys = keys[:40]; train_keys = keys[40:]

    def fresh_ds():
        return DataSplit("", df.copy(), df_universe.copy(), "a", {}, "sale_price", "sale_price",
                         ind_vars, ["neighborhood"], {}, test_keys, train_keys)

    r1 = run_layeredcomp(fresh_ds(), str(tmp_path), save_params=True, use_saved_params=False)
    assert os.path.exists(tmp_path / "lcomp_falloffs.json")

    r2 = run_layeredcomp(fresh_ds(), str(tmp_path), save_params=False, use_saved_params=True)
    np.testing.assert_allclose(
        np.asarray(r1.pred_univ, dtype="float64"),
        np.asarray(r2.pred_univ, dtype="float64"),
        rtol=1e-9, atol=1e-6,
    )


def test_mra_log_param_off_is_unchanged():
    # log defaults to False -> identical to a plain run_mra (no behavior change for everyone else).
    df_sales, df_universe, ind_vars, test_keys, train_keys = _log_target_dataset()
    ds_a = DataSplit("", df_sales.copy(), df_universe.copy(), "a", {}, "sale_price", "sale_price",
                     ind_vars, [], {}, test_keys, train_keys)
    ds_b = DataSplit("", df_sales.copy(), df_universe.copy(), "a", {}, "sale_price", "sale_price",
                     ind_vars, [], {}, test_keys, train_keys)
    res_default = run_mra(ds_a, intercept=True)
    res_explicit = run_mra(ds_b, intercept=True, log=False)
    np.testing.assert_allclose(
        np.asarray(res_default.pred_test.y_pred, dtype="float64"),
        np.asarray(res_explicit.pred_test.y_pred, dtype="float64"),
        rtol=1e-9,
    )