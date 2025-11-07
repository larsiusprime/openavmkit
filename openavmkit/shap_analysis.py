import textwrap

import numpy as np
import pandas as pd
import shap

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt

from openavmkit.modeling import SingleModelResults


def quick_shap(smr: SingleModelResults, plot: bool = False, title: str = ""):
    """
    Compute SHAP values for a given model and dataset and optionally plot it.

    Parameters
    ----------
    smr : SingleModelResults
        The SingleModelResults object containing the fitted model and data splits.
    plot : bool, optional
        If True, generate and display a SHAP summary plot. Defaults to False.
    title : str, optional
        Title to use for the SHAP plot if `plot` is True. Defaults to an empty string.

    Returns
    -------
    np.ndarray
        SHAP values array for the evaluation dataset.
    """

    if smr.type not in ["xgboost", "catboost", "lightgbm"]:
        # SHAP is not supported for this model type
        return

    X_train = smr.ds.X_train

    shaps = _calc_shap(smr.model, X_train, X_train)

    if plot:
        plot_full_beeswarm(shaps, title=title)


def get_model_shaps(smr: SingleModelResults):
    
    X_train = smr.ds.X_train
    X_test = smr.ds.X_test
    X_univ = smr.ds.X_univ
    
    #shaps = _calc_shap(smr.model, X_train, X_train)


def plot_full_beeswarm(
    explanation: shap.Explanation, title: str = "SHAP Beeswarm", wrap_width: int = 20
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
    """

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

    plt.show()


def _make_shap_table(
    expl: shap.Explanation,
    list_keys: list[str],
    list_vars: list[str],
    list_keys_sale: list[str] = None,
    include_pred: bool = True
) -> pd.DataFrame:
    """
    Build a wide SHAP table:
      [<key cols> ... , base_value, <feature_1> ... <feature_n>, (optional) pred_reconstructed]

    Parameters
    ----------
    expl : shap.Explanation
        Output of your _xgboost_shap (values: (n,m), base_values: scalar or (n,)).
    list_keys : list[str]
        Primary keys in the same row order as X_to_explain
    list_vars : list[str]
        Feature names in the order used for training (your canonical order).
    list_keys_sale : list[str]
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
    if list_keys_sale and len(list_keys_sale) != n:
        raise ValueError(f"list_keys_sale length {len(list_keys_sale)} != number of rows {n}")

    if list_keys_sale:
        df_keys = pd.DataFrame({"key": list_keys, "key_sale": list_keys_sale})
    else:
        df_keys = pd.DataFrame({"key": list_keys})

    # 5) Base value column (between keys and features)
    df_base = pd.DataFrame({"base_value": base_arr})

    # 6) Optional reconstructed prediction on the explained scale
    # (raw margin for classifiers unless you used model_output="probability")
    if include_pred:
        pred = base_arr + df_features.sum(axis=1).to_numpy()
        df_pred = pd.DataFrame({"pred_reconstructed": pred})
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
    X_to_explain: pd.DataFrame,
    background_size: int,
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


def _xgboost_explain(
    te : shap.TreeExplainer,
    X_to_explain: pd.DataFrame,
)-> shap.Explanation:
    # explain your rows, again as a float32 array
    X_arr = X_to_explain.to_numpy(dtype=np.float32)
    vals = te.shap_values(X_arr, approximate=approximate, check_additivity=check_additivity)  # shape (n_samples, n_features)

    # wrap into an Explanation for downstream plotting
    return shap.Explanation(
        values=vals,
        base_values=te.expected_value,
        data=X_arr,
        feature_names=list(X_to_explain.columns),
    )


def _lightgbm_shap(
    model: lgb.basic.Booster,
    X_train: pd.DataFrame,
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


def _lightgbm_explain(
    te: shap.TreeExplainer,
    X_to_explain: pd.DataFrame
)-> shap.Explanation:
    
    vals = te.shap_values(X_to_explain, approximate=approximate, check_additivity=check_additivity)

    return shap.Explanation(
        values=vals,
        base_values=te.expected_value,
        data=X_to_explain.to_numpy(),
        feature_names=list(X_to_explain.columns),
    )


def _catboost_shap(
    model: cb.CatBoost,
    X: pd.DataFrame,
    shap_type: str = "Approximate",
    n_threads: int | None = None
) -> shap.Explanation:
    # 1. Ask the model for categorical columns (fallback = None)
    try:
        cat_idx = model.get_cat_feature_indices()
    except AttributeError:
        cat_idx = None                 # very old CatBoost versions

    # 2. Wrap the data in a Pool; only pass cat_features if we actually have them
    pool = cb.Pool(
        X,
        cat_features=cat_idx if cat_idx else None
    )

    # 3. Fast SHAP via CatBoost’s native call
    shap_vals = model.get_feature_importance(
        data=pool,
        type="ShapValues",
        shap_calc_type=shap_type,       # "Approximate" --> 10-100x faster
        thread_count=n_threads or -1,
    )                                   # (n_samples, n_features+1)

    base_values = shap_vals[:, -1]
    values = shap_vals[:, :-1]

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=X.to_numpy(),
        feature_names=X.columns.tolist()
    )