import textwrap

import numpy as np
import pandas as pd
import shap

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt

from openavmkit.modeling import SingleModelResults


def compute_shap(
    smr: SingleModelResults,
    plot: bool = False,
    title: str = ""
):
  """
  Compute SHAP values for a given model and dataset.
  :param smr: The SingleModelResults object containing the model and datasets.
  :param plot: Whether to plot the SHAP values.
  :return: SHAP values for the evaluation dataset.
  """

  if smr.type not in ["xgboost", "catboost", "lightgbm"]:
    # SHAP is not supported for this model type
    return

  X_train = smr.ds.X_train
  X_univ = smr.ds.X_univ

  shaps = _compute_shap(
      smr.model,
      X_train,
      X_train
  )

  if plot:
    plot_full_beeswarm(shaps, title=title)

    # plt.close("all")
    # fig, ax = plt.subplots(figsize=(10,6))
    #
    # # draw into your fig/ax but don't let SHAP call show()
    # shap.plots.beeswarm(shaps, show=False, max_display=30)
    #
    # ax.set_title(title)
    # plt.tight_layout()
    # plt.show()


def plot_full_beeswarm(explanation, title="SHAP Beeswarm", wrap_width=20):
  """
  1) Compute a single-output SHAP Explanation for any tree model
  2) Wrap long feature names
  3) Auto-scale the figure size
  4) Plot a full beeswarm with rotated, smaller y-ticks
  """

  # Wrap feature names
  wrapped_names = [
    "\n".join(textwrap.wrap(fn, width=wrap_width))
    for fn in explanation.feature_names
  ]
  expl_wrapped = shap.Explanation(
    values=        explanation.values,
    base_values=   explanation.base_values,
    data=          explanation.data,
    feature_names= wrapped_names
  )

  # Determine figure size based on # features
  n_feats = len(wrapped_names)
  width  = max(12, 0.3 * n_feats)
  height = max(6,  0.3 * n_feats)
  fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

  # Draw the beeswarm (max_display defaults to all features here)
  shap.plots.beeswarm(
    expl_wrapped,
    max_display=n_feats,
    show=False
  )

  # Title + tweak y-labels
  ax.set_title(title)
  plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=8)

  plt.show()


def _compute_shap(
    model,
    X_train: pd.DataFrame,
    X_to_explain: pd.DataFrame,
    background_size: int = 100
) -> shap.Explanation:
  """
  Handles:
    • raw xgboost.core.Booster
    • raw lightgbm.basic.Booster
    • any sklearn‐style wrapper (XGBRegressor, LGBMRegressor, CatBoostRegressor, etc.)
  """

  print(f"model name = {model.__class__.__name__}")
  print(f"model type = {type(model)}")

  # 1) Raw XGBoost Booster or XGBRegressor → force legacy TreeExplainer + numpy arrays
  if isinstance(model, (xgb.core.Booster, xgb.XGBRegressor)):
    # a) sample & convert to float64 array
    bg_df  = X_train.sample(min(background_size, len(X_train)), random_state=0)
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
    vals  = te.shap_values(X_arr)   # shape (n_samples, n_features)

    # d) wrap into an Explanation for downstream plotting
    return shap.Explanation(
      values=      vals,
      base_values= te.expected_value,
      data=        X_arr,
      feature_names=list(X_to_explain.columns)
    )

  # ——— 2) Raw LightGBM Booster — *no* data arg, default interventional ———
  if isinstance(model, lgb.basic.Booster):
    te = shap.TreeExplainer(
      model,
      feature_perturbation="interventional"
    )
    vals = te.shap_values(X_to_explain)

    return shap.Explanation(
      values=vals,
      base_values=te.expected_value,
      data=X_to_explain.to_numpy(),
      feature_names=list(X_to_explain.columns)
    )

  # ——— 3) CatBoostRegressor — path_dependent explainer, NO background ———
  if isinstance(model, cb.CatBoostRegressor):
    te = shap.TreeExplainer(
      model,
      feature_perturbation="tree_path_dependent"
    )
    # CatBoost will accept the DataFrame directly here
    vals = te.shap_values(X_to_explain)
    return shap.Explanation(
      values=      np.array(vals),  # ensure ndarray
      base_values= te.expected_value,
      data=        X_to_explain.to_numpy(),
      feature_names=list(X_to_explain.columns)
    )

  # ——— 3) Everything else (sklearn wrappers, CatBoostRegressor, etc.) — unified API ———
  explainer = shap.Explainer(model, X_train)
  return explainer(X_to_explain)