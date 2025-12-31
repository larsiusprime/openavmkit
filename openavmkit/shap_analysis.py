import textwrap

import numpy as np
import pandas as pd
import shap

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt

from openavmkit.utilities.modeling import (
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    TreeBasedCategoricalData
)


def get_full_model_shaps(
    model: XGBoostModel | LightGBMModel | CatBoostModel,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_sales: pd.DataFrame,
    X_univ: pd.DataFrame,
    verbose: bool = False
):
    """
    Calculates shaps for all subsets (test, train, sales, universe) of one model run
    
    Parameters
    ----------
    model: XGBoostModel | LightGBMModel | CatBoostModel
        A trained prediction model
    X_train: pd.DataFrame
        2D array of independent variables' values from the training set
    X_test: pd.DataFrame
        2D array of independent variables' values from the testing set
    X_sales: pd.DataFrame
        2D array of independent variables' values from the sales set
    X_univ: pd.DataFrame
        2D array of independent variables' values from the universe set
    verbose: bool
        Whether to print verbose information. Defaults to False.
    
    Returns
    -------
    dict
        A dict containing shap.Explanation objects keyed to "train", "test", "sales", and "univ"
    
    """

    tree_explainer: shap.TreeExplainer
    
    approximate = True
    cat_data = model.cat_data
    
    model_type = ""
    if isinstance(model, XGBoostModel):
        model_type = "xgboost"
        tree_explainer = _xgboost_shap(model, X_train)
    elif isinstance(model, LightGBMModel):
        model_type = "lightgbm"
        approximate = False # approx. not supported for LightGBM
        tree_explainer = _lightgbm_shap(model, X_train)
    elif isinstance(model, CatBoostModel):
        model_type = "catboost"
        tree_explainer = _catboost_shap(model, X_train)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    if verbose:
        print(f"Generating SHAPs...")
    
    shap_sales = _shap_explain(model_type, tree_explainer, X_sales, cat_data=cat_data, approximate=approximate, verbose=verbose, label="sales")
    shap_train = _shap_explain(model_type, tree_explainer, X_train, cat_data=cat_data, approximate=approximate, verbose=verbose, label="train")
    shap_test  = _shap_explain(model_type, tree_explainer, X_test,  cat_data=cat_data, approximate=approximate, verbose=verbose, label="test")
    shap_univ  = _shap_explain(model_type, tree_explainer, X_univ,  cat_data=cat_data, approximate=approximate, verbose=verbose, label="universe")

    return {
        "train": shap_train,
        "test":  shap_test,
        "sales": shap_sales,
        "univ":  shap_univ,
    }


def plot_full_beeswarm(
    explanation: shap.Explanation, 
    title: str = "SHAP Beeswarm", 
    save_path: str | None = None,
    save_kwargs: dict | None = None,
    wrap_width: int = 20
) -> None:
    """
    Plot a full SHAP beeswarm for a tree-based model with wrapped feature names.

    This function wraps long feature names, auto-scales figure size to the number of
    features, and renders a beeswarm plot with rotated, smaller y-axis labels.

    Parameters
    ----------
    explanation : shap.Explanation
        SHAP Explanation object with `values`, `base_values`, `data`, and `feature_names`.
    title : str, optional
        Title of the plot. Defaults to "SHAP Beeswarm".
    wrap_width : int, optional
        Maximum character width for feature name wrapping. Defaults to 20.
    save_path : str, optional
        If provided, save the figure to this path (format inferred from extension).
        e.g., 'beeswarm.png', 'beeswarm.pdf', 'figs/beeswarm.svg'.
    save_kwargs : dict, optional
        Extra kwargs passed to `fig.savefig` (e.g., {'dpi': 300, 'bbox_inches': 'tight',
        'transparent': True}).
    """
    if save_kwargs is None:
        save_kwargs = {"dpi": 300, "bbox_inches": "tight"}
    
    # Wrap feature names
    wrapped_names = [
        "\n".join(textwrap.wrap(fn, width=wrap_width))
        for fn in explanation.feature_names
    ]
    expl_wrapped = shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=wrapped_names,
    )

    # Determine figure size based on # features
    n_feats = len(wrapped_names)
    width = max(12, 0.3 * n_feats)
    height = max(6, 0.3 * n_feats)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    # Draw the beeswarm (max_display defaults to all features here)
    shap.plots.beeswarm(expl_wrapped, max_display=n_feats, show=False)

    # Title + tweak y-labels
    ax.set_title(title)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=8)

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, **save_kwargs)

    plt.show()


def make_shap_table(
    expl: shap.Explanation,
    list_keys: list[str],
    list_vars: list[str],
    list_keys_sale: list[str] = None,
    include_pred: bool = True
) -> pd.DataFrame:
    """
    Convert a shap explanation into a dataframe breaking down the full contribution to value
    
    Parameters
    ----------
    expl : shap.Explanation
        Output of your _xgboost_shap (values: (n,m), base_values: scalar or (n,)).
    list_keys : list[str]
        Primary keys in the same row order as X_to_explain
    list_vars : list[str]
        Feature names in the order used for training (your canonical order).
    list_keys_sale : list[str] | None
        Optional. Transaction keys in the same row order as X_to_explain. Default is None.
    include_pred : bool
        Optional. Add a column that reconstructs the model output on the explained scale:
        base_value + sum(shap_values across features). Default is True.

    Returns
    -------
    pd.DataFrame
    """
    # 1) Validate / normalize SHAP values shape (expect regression/binary: (n, m))
    vals = expl.values
    if isinstance(vals, list):
        raise ValueError("Got a list of SHAP arrays (likely multiclass). Handle per-class tables separately.")
    vals = np.asarray(vals)
    if vals.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values (n_samples, n_features), got shape {vals.shape}.")

    n, m = vals.shape

    # 2) Base values: scalar or per-row
    base = expl.base_values
    if np.isscalar(base):
        base_arr = np.full((n,), float(base))
    else:
        base = np.asarray(base)
        if base.ndim == 0:
            base_arr = np.full((n,), float(base))
        elif base.ndim == 1 and base.shape[0] == n:
            base_arr = base.astype(float)
        else:
            raise ValueError(f"Unexpected base_values shape {base.shape}. For multiclass, build per-class tables.")

    # 3) Build feature DF in the *training* column order
    # expl.feature_names comes from X_to_explain; align to canonical list_vars
    if expl.feature_names is None:
        # assume expl.values columns already match list_vars
        feature_cols = list_vars
    else:
        # ensure all requested vars exist
        existing = list(expl.feature_names)
        missing = [c for c in list_vars if c not in existing]
        if missing:
            raise ValueError(f"These list_vars are missing from explanation features: {missing}")
        feature_cols = list_vars  # enforce this order

    df_features = pd.DataFrame(vals, columns=expl.feature_names)
    df_features = df_features[feature_cols]  # reorder
    
    # 4) Keys up front (robust expansion)
    if len(list_keys) != n:
        raise ValueError(f"list_keys length {len(list_keys)} != number of rows {n}")
    if list_keys_sale is not None and len(list_keys_sale) != n:
        raise ValueError(f"list_keys_sale length {len(list_keys_sale)} != number of rows {n}")

    if list_keys_sale is not None:
        df_keys = pd.DataFrame({"key": list_keys, "key_sale": list_keys_sale})
    else:
        df_keys = pd.DataFrame({"key": list_keys})

    # 5) Base value column (between keys and features)
    df_base = pd.DataFrame({"base_value": base_arr})

    # 6) Optional reconstructed prediction on the explained scale
    # (raw margin for classifiers unless you used model_output="probability")
    if include_pred:
        pred = base_arr + df_features.sum(axis=1).to_numpy()
        df_pred = pd.DataFrame({"contribution_sum": pred})
        df = pd.concat([df_keys, df_base, df_features, df_pred], axis=1)
    else:
        df = pd.concat([df_keys, df_base, df_features], axis=1)

    return df


def _calc_shap(
    model, X_train: pd.DataFrame, X_to_explain: pd.DataFrame, background_size: int = 100
) -> shap.Explanation:
    
    # 1) XGBoost
    if isinstance(model, XGBoostModel):
        xgb_explainer = _xgboost_shap(
            model,
            X_train
        )
        return _xgboost_explain(X_to_explain)

    # 2) LightGBM
    if isinstance(model, LightGBMModel):
        lgbm_explainer = _lightgbm_shap(
            model,
            X_train
        )
        return _lightgbm_explain(X_to_explain)

    # 3) CatBoost
    if isinstance(model, CatBoostModel):
        return _catboost_shap(
            model,
            X_to_explain
        )

    # 4) Everything else
    explainer = shap.Explainer(model, X_train)
    return explainer(X_to_explain)


def _xgboost_shap(
    model: XGBoostModel,
    X_train: pd.DataFrame,
    background_size: int = 100,
    approximate: bool = True,
    check_additivity: bool = False
)-> shap.TreeExplainer :
    # For categorical splits, tree_path_dependent is the safe choice.
    te = shap.TreeExplainer(
        model.regressor,
        feature_perturbation="tree_path_dependent",
    )

    # Make SHAP create DMatrix with categorical support when input has pandas category dtype
    if hasattr(te, "model"):
        te.model._xgb_dmatrix_props = {"enable_categorical": True}

    return te


def _lightgbm_shap(
    model: LightGBMModel,
    X_train: pd.DataFrame,
    background_size: int = 100,
    approximate: bool = True,
    check_additivity: bool = False,
) -> shap.TreeExplainer:
    booster = model.booster

    has_cats = bool(getattr(model, "cat_data", None)) and len(model.cat_data.categorical_cols) > 0

    if has_cats:
        # SHAP limitation: categorical splits only supported with tree_path_dependent
        # and *no* background data passed.
        te = shap.TreeExplainer(
            booster,
            feature_perturbation="tree_path_dependent",
        )

        # Optional: set a background-based expected value for nicer baselines
        # (SHAP will otherwise use the model's internal baseline).
        bg_df = X_train.sample(min(background_size, len(X_train)), random_state=0)
        bg_df = model.cat_data.apply(bg_df)
        try:
            preds = booster.predict(bg_df)
            te.expected_value = preds.mean(axis=0)
        except Exception:
            # If prediction fails for some reason, fall back to SHAP's default baseline.
            pass

        return te

    # No categoricals: interventional + numeric background is fine
    bg_df = X_train.sample(min(background_size, len(X_train)), random_state=0)
    bg_arr = bg_df.to_numpy(dtype=np.float64)

    return shap.TreeExplainer(
        booster,
        data=bg_arr,
        feature_perturbation="interventional",
    )


def _catboost_shap(
    model: CatBoostModel,
    X_train: pd.DataFrame,
    background_size: int = 100,
    approximate: bool = True,
    check_additivity: bool = False,
) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for a CatBoostRegressor.

    IMPORTANT: For CatBoost models with categorical splits, SHAP currently
    only supports TreeExplainer with feature_perturbation="tree_path_dependent"
    and *no* background data passed to the constructor. So X_train /
    background_size are accepted only for signature compatibility and are
    not used here.

    Parameters
    ----------
    model : CatBoostModel
        Trained CatBoostModel instance.
    X_train : pd.DataFrame
        Training data (unused here, kept for API symmetry).
    background_size : int, default=100
        Unused for CatBoost because SHAP cannot accept background data
        when categorical splits are present.
    approximate : bool, default=True
        Kept for interface compatibility with other backends.
    check_additivity : bool, default=False
        Kept for interface compatibility with other backends.

    Returns
    -------
    shap.TreeExplainer
        Configured TreeExplainer for the CatBoost model.
    """

    regressor = model.regressor
    
    explainer = shap.TreeExplainer(
        regressor,
        feature_perturbation="tree_path_dependent",
    )

    # Tag this explainer so _shap_explain knows it's CatBoost
    explainer._cb_model = regressor   # type: ignore[attr-defined]

    return explainer


def _lgb_pred_contrib_chunked(booster, X: pd.DataFrame, chunk_size: int, verbose: bool, label: str):
    import time
    n = len(X)
    out = []
    t_all = time.time()
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        t0 = time.time()
        out.append(booster.predict(X.iloc[i:j], pred_contrib=True))
        if verbose:
            dt = time.time() - t0
            print(f"[{label}] LightGBM contrib rows {i}:{j}/{n} ({100*j/n:.1f}%) in {dt:.2f}s", flush=True)
    if verbose:
        print(f"[{label}] LightGBM contrib total {time.time()-t_all:.2f}s", flush=True)
    return np.vstack(out)


def _shap_explain(
    model_type: str,
    te: shap.TreeExplainer,
    X_to_explain: pd.DataFrame,
    approximate: bool = True,
    check_additivity: bool = False,
    cat_data: TreeBasedCategoricalData | None = None,
    verbose: bool = False,
    label: str = ""
) -> shap.Explanation:
    """
    Use a TreeExplainer to compute SHAP values for X_to_explain and wrap
    them in a shap.Explanation for downstream plotting.

    For CatBoost explainers (tagged with `_cb_model`), we bypass
    TreeExplainer.shap_values and use CatBoost's native
    `get_feature_importance(..., type="ShapValues")` instead, which
    supports a fast "Approximate" mode.
    
    Parameters
    ----------
    model_type: str
        "xgboost", "catboost", or "lightgbm"
    te : shap.TreeExplainer
        TreeExplainer instance built on the trained model and background data.
    X_to_explain : pd.DataFrame
        Data to explain (same feature order as used for training / background).
    approximate : bool, default=True
        Passed through to TreeExplainer.shap_values (where supported).
    check_additivity : bool, default=False
        Passed through to TreeExplainer.shap_values.
    cat_data: TreeBasedCategoricalData, default=None
        Categorical encoding data, if any
    verbose: bool
        Whether to print verbose output. Defaults to False.
    label: str
        Informative label

    Returns
    -------
    shap.Explanation
    """

    # --- CatBoost fast path -------------------------------------------------
    if model_type == "catboost":
        cb_model = getattr(te, "_cb_model", None)
        # This is a CatBoostRegressor explainer; use CatBoost-native SHAP.

        # Discover categorical feature indices from the model if available.
        try:
            cat_idx = cb_model.get_cat_feature_indices()
        except AttributeError:
            cat_idx = None

        pool = cb.Pool(
            X_to_explain,
            cat_features=cat_idx if cat_idx else None,
        )

        shap_type = "Approximate" if approximate else "Regular"

        shap_vals = cb_model.get_feature_importance(
            data=pool,
            type="ShapValues",
            shap_calc_type=shap_type,
            thread_count=-1,   # use all threads
        )
        # shap_vals shape: (n_samples, n_features + 1)
        base_values = shap_vals[:, -1]
        values = shap_vals[:, :-1]

        if cat_data is not None:
            # numeric matrix: categories -> codes, bool-like -> 0/1, stable column order
            X_arr = cat_data.to_numeric_matrix(X_to_explain)
            feature_names = list(cat_data.feature_names)
        else:
            # fallback, but do NOT force float because it can contain strings
            X_arr = X_to_explain.to_numpy(copy=False)
            feature_names = list(X_to_explain.columns)

        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=X_arr,
            feature_names=feature_names,
        )

    # --- LightGBM native fast path -----------------------------------------
    
    # TreeExplainer stores a TreeEnsemble wrapper in te.model, and often keeps the
    # original model at te.model.original_model.
    if model_type == "lightgbm":
        lgb_booster = getattr(getattr(te, "model", None), "original_model", None)
    
        # Apply categorical structure if provided
        X_used = cat_data.apply(X_to_explain) if (cat_data is not None and len(cat_data.categorical_cols) > 0) else X_to_explain

        # LightGBM native contributions are fast and avoid SHAP's slow path.
        if verbose:
            print(f"LightGBM.predict()...")
        #contrib = lgb_booster.predict(X_used, pred_contrib=True)
        contrib = _lgb_pred_contrib_chunked(lgb_booster, X_used, chunk_size=10_000, verbose=verbose, label=label)
        if verbose:
            print(f"LightGBM.predict() finished.")

        # contrib: (n_samples, n_features + 1)
        base_values = contrib[:, -1]
        values = contrib[:, :-1]

        # Keep data in a reasonably numpy-ish form for Explanation.
        # Don't force float if categoricals exist.
        X_arr = X_used.to_numpy(copy=False)

        feature_names = list(X_used.columns)
        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=X_arr,
            feature_names=feature_names,
        )

    # --- Default path: XGBoost / LightGBM / generic trees -------------------
    # If it's not tagged as CatBoost, fall back to TreeExplainer directly.
    
    if model_type == "xgboost":
        X_used = cat_data.apply(X_to_explain)
        try:
            vals = te.shap_values(
                X_used,
                approximate=approximate,
                check_additivity=check_additivity,
            )
        except NotImplementedError as e:
            feature_names = cat_data.feature_names
            # If TreeExplainer refuses due to categorical splits, rebuild explainer in tree_path_dependent mode
            if "Categorical split is not yet supported" in str(e):
                te = shap.TreeExplainer(te.model.original_model, feature_perturbation="tree_path_dependent")
                te.model._xgb_dmatrix_props = {"enable_categorical": True}
                vals = te.shap_values(
                    X_used,
                    approximate=approximate,
                    check_additivity=check_additivity,
                )
            else:
                raise

        X_arr = X_used.to_numpy(copy=False)
        return shap.Explanation(
            values=vals,
            base_values=te.expected_value,
            data=X_arr,
            feature_names=list(X_used.columns),
        )
    else:
        raise ValueError(f"Unsupported model type \"{model_type}\"")