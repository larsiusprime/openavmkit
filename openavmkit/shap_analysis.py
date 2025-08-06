import textwrap

import numpy as np
import pandas as pd
import shap

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt

from openavmkit.modeling import SingleModelResults


def compute_shap(smr: SingleModelResults, plot: bool = False, title: str = ""):
    """
    Compute SHAP values for a given model and dataset.

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

    shaps = _compute_shap(smr.model, X_train, X_train)

    if plot:
        plot_full_beeswarm(shaps, title=title)


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


def _compute_shap(
    model, X_train: pd.DataFrame, X_to_explain: pd.DataFrame, background_size: int = 100
) -> shap.Explanation:
    """
    Handles:
      • raw xgboost.core.Booster
      • raw lightgbm.basic.Booster
      • any sklearn‐style wrapper (XGBRegressor, LGBMRegressor, CatBoostRegressor, etc.)
    """

    # 1) Raw XGBoost Booster or XGBRegressor --> force legacy TreeExplainer + numpy arrays
    if isinstance(model, (xgb.core.Booster, xgb.XGBRegressor)):
        # a) sample & convert to float64 array
        bg_df = X_train.sample(min(background_size, len(X_train)), random_state=0)
        bg_arr = bg_df.to_numpy(dtype=np.float64)

        # b) build the TreeExplainer on the Booster itself
        booster = model.get_booster() if isinstance(model, xgb.XGBRegressor) else model
        te = shap.TreeExplainer(
            booster,
            data=bg_arr,
            feature_perturbation="interventional"
        )

        # c) explain your rows, again as a float64 array
        X_arr = X_to_explain.to_numpy(dtype=np.float64)
        vals = te.shap_values(X_to_explain, approximate=True, check_additivity=False)  # shape (n_samples, n_features)

        # d) wrap into an Explanation for downstream plotting
        return shap.Explanation(
            values=vals,
            base_values=te.expected_value,
            data=X_arr,
            feature_names=list(X_to_explain.columns),
        )

    # 2) Raw LightGBM Booster --> *no* data arg, default interventional
    if isinstance(model, lgb.basic.Booster):
        bg_df  = X_train.sample(min(background_size, len(X_train)), random_state=0)
        bg_arr = bg_df.to_numpy(dtype=np.float64)
        te = shap.TreeExplainer(
            model,
            data=bg_arr,
            feature_perturbation="interventional"
        )
        vals = te.shap_values(X_to_explain, approximate=True, check_additivity=False)

        return shap.Explanation(
            values=vals,
            base_values=te.expected_value,
            data=X_to_explain.to_numpy(),
            feature_names=list(X_to_explain.columns),
        )

    # 3) CatBoostRegressor — path_dependent explainer, NO background
    if isinstance(model, cb.CatBoostRegressor):
        return _fast_catboost_shap(
            model,
            X_to_explain,
            shap_type="Approximate"
        )
    
    # if isinstance(model, cb.CatBoostRegressor):
    #     te = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    #     # CatBoost will accept the DataFrame directly here
    #     vals = te.shap_values(X_to_explain, check_additivity=False)
    #     return shap.Explanation(
    #         values=np.array(vals),  # ensure ndarray
    #         base_values=te.expected_value,
    #         data=X_to_explain.to_numpy(),
    #         feature_names=list(X_to_explain.columns),
    #     )

    # 4) Everything else (sklearn wrappers, CatBoostRegressor, etc.) — unified API ———
    explainer = shap.Explainer(model, X_train)
    return explainer(X_to_explain)


def _fast_catboost_shap(
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
        shap_calc_type=shap_type,       # "Approximate" → 10-100 × faster
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