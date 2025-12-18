import textwrap

import numpy as np
import pandas as pd
import shap

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt


def get_model_shaps(
    model: xgb.XGBRegressor | lgb.Booster | cb.CatBoostRegressor,
    X_bkg: pd.DataFrame,
    X_to_explain: pd.DataFrame
)-> shap.Explanation:
    """
    Calculates shaps for a single background/explanation set
    
    Parameters
    ----------
    model: xgboost.XGBRegressor | lightgbm.Booster | catboost.CatBoostRegressor
        A trained prediction model
    X_bkg: pd.DataFrame
        2D array of independent variables' values from the background set
    X_to_explain: pd.DataFrame
        2D array of independent variables' values you wish to explain
    
    Returns
    -------
    shap.Explanation
    """

    tree_explainer: shap.TreeExplainer

    if isinstance(model, (xgb.core.Booster, xgb.XGBRegressor)):
        tree_explainer = _xgboost_shap(model, X_bkg)
    elif isinstance(model, lgb.basic.Booster):
        tree_explainer = _lightgbm_shap(model, X_bkg)
    elif isinstance(model, cb.CatBoostRegressor):
        tree_explainer = _catboost_shap(model, X_bkg)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    return _shap_explain(tree_explainer, X_to_explain)


def get_full_model_shaps(
    model: xgb.XGBRegressor | lgb.Booster | cb.CatBoostRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_sales: pd.DataFrame,
    X_univ: pd.DataFrame
):
    """
    Calculates shaps for all subsets (test, train, sales, universe) of one model run
    
    Parameters
    ----------
    model: xgboost.XGBRegressor | lightgbm.Booster | catboost.CatBoostRegressor
        A trained prediction model
    X_train: pd.DataFrame
        2D array of independent variables' values from the training set
    X_test: pd.DataFrame
        2D array of independent variables' values from the testing set
    X_sales: pd.DataFrame
        2D array of independent variables' values from the sales set
    X_univ: pd.DataFrame
        2D array of independent variables' values from the universe set
    
    Returns
    -------
    dict
        A dict containing shap.Explanation objects keyed to "train", "test", "sales", and "univ"
    
    """

    tree_explainer: shap.TreeExplainer

    if isinstance(model, (xgb.core.Booster, xgb.XGBRegressor)):
        tree_explainer = _xgboost_shap(model, X_train)
    elif isinstance(model, lgb.basic.Booster):
        tree_explainer = _lightgbm_shap(model, X_train)
    elif isinstance(model, cb.CatBoostRegressor):
        tree_explainer = _catboost_shap(model, X_train)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    shap_train = _shap_explain(tree_explainer, X_train)
    shap_test  = _shap_explain(tree_explainer, X_test)
    shap_sales = _shap_explain(tree_explainer, X_sales)
    shap_univ  = _shap_explain(tree_explainer, X_univ)

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
    if isinstance(model, (xgb.core.Booster, xgb.XGBRegressor)):
        xgb_explainer = _xgboost_shap(
            model,
            X_train
        )
        return _xgboost_explain(X_to_explain)

    # 2) LightGBM
    if isinstance(model, lgb.basic.Booster):
        lgbm_explainer = _lightgbm_shap(
            model,
            X_train
        )
        return _lightgbm_explain(X_to_explain)

    # 3) CatBoost
    if isinstance(model, cb.CatBoostRegressor):
        return _catboost_shap(
            model,
            X_to_explain
        )

    # 4) Everything else
    explainer = shap.Explainer(model, X_train)
    return explainer(X_to_explain)


def _xgboost_shap(
    model: xgb.core.Booster | xgb.XGBRegressor,
    X_train: pd.DataFrame,
    background_size: int = 100,
    approximate: bool = True,
    check_additivity: bool = False
)-> shap.TreeExplainer :
    # a) sample & convert to float32 array
    bg_df = X_train.sample(min(background_size, len(X_train)), random_state=0)
    bg_arr = bg_df.to_numpy(dtype=np.float32)

    # b) build the TreeExplainer on the Booster itself
    booster = model.get_booster() if isinstance(model, xgb.XGBRegressor) else model
    return shap.TreeExplainer(
        booster,
        data=bg_arr,
        feature_perturbation="interventional"
    )


def _lightgbm_shap(
    model: lgb.basic.Booster,
    X_train: pd.DataFrame,
    background_size: int = 100,
    approximate: bool = True,
    check_additivity: bool = False
)-> shap.TreeExplainer:
    bg_df  = X_train.sample(min(background_size, len(X_train)), random_state=0)
    bg_arr = bg_df.to_numpy(dtype=np.float64)
    return shap.TreeExplainer(
        model,
        data=bg_arr,
        feature_perturbation="interventional"
    )


def _catboost_shap(
    model: cb.CatBoostRegressor,
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
    model : catboost.CatBoostRegressor
        Trained CatBoostRegressor instance.
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

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )

    # Tag this explainer so _shap_explain knows it's CatBoost
    explainer._cb_model = model   # type: ignore[attr-defined]

    return explainer



def _shap_explain(
    te: shap.TreeExplainer,
    X_to_explain: pd.DataFrame,
    approximate: bool = True,
    check_additivity: bool = False,
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
    te : shap.TreeExplainer
        TreeExplainer instance built on the trained model and background data.
    X_to_explain : pd.DataFrame
        Data to explain (same feature order as used for training / background).
    approximate : bool, default=True
        Passed through to TreeExplainer.shap_values (where supported).
    check_additivity : bool, default=False
        Passed through to TreeExplainer.shap_values.

    Returns
    -------
    shap.Explanation
    """

    # --- CatBoost fast path -------------------------------------------------
    cb_model = getattr(te, "_cb_model", None)
    if cb_model is not None:
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

        X_arr = X_to_explain.to_numpy(dtype=np.float64, copy=False)

        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=X_arr,
            feature_names=list(X_to_explain.columns),
        )

    # --- Default path: XGBoost / LightGBM / generic trees -------------------
    # If it's not tagged as CatBoost, fall back to TreeExplainer directly.
    vals = te.shap_values(
        X_to_explain,
        approximate=approximate,
        check_additivity=check_additivity,
    )

    X_arr = X_to_explain.to_numpy(dtype=np.float64, copy=False)

    return shap.Explanation(
        values=vals,
        base_values=te.expected_value,
        data=X_arr,
        feature_names=list(X_to_explain.columns),
    )