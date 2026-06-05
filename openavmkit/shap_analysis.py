"""
SHAP-value computation and reporting for tree-based models.

Computes SHAP (Shapley) values across all subsets (test, train, sales,
universe) for XGBoost, LightGBM, and CatBoost models. Used by
:mod:`openavmkit.modeling` to produce the per-feature ``params`` and
``contributions`` outputs that mirror what linear models produce as
coefficients and coefficient×value contributions.

See Also
--------
openavmkit.modeling : Calls into this module to build params/contribs
    files for tree-based models.
"""
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
    NGBoostModel,
    LayeredCompModel,
    TreeBasedCategoricalData
)


_NGBOOST_REQUIRED_ATTRS = ("base_models", "scalings", "col_idxs", "learning_rate", "init_params")


def ngboost_internals_ok(regressor) -> bool:
    """Return True if an NGBRegressor exposes the additive-ensemble internals SHAP needs.

    NGBoost's exact tree-SHAP decomposition relies on per-stage ``base_models``,
    ``scalings``, ``col_idxs``, ``learning_rate`` and ``init_params``. This guards
    against future NGBoost layout changes — callers should skip SHAP (rather than
    crash) when it returns False.
    """
    if not all(hasattr(regressor, a) for a in _NGBOOST_REQUIRED_ATTRS):
        return False
    bms = getattr(regressor, "base_models", None)
    if not bms or len(bms) == 0:
        return False
    # Each stage holds one fitted learner per distribution parameter (loc, logscale).
    if len(bms[0]) < 2:
        return False
    return True


class _NGBoostShapExplainer:
    """Exact additive SHAP for one NGBoost distribution parameter.

    NGBoost predicts each distribution parameter as an additive ensemble of
    per-stage trees::

        param_p(X) = init_params[p] - lr * Σ_i scaling_i * tree_{i,p}.predict(X[:, col_idx_i])

    so its SHAP is the weighted sum of per-stage ``TreeExplainer`` values. The
    base value folds in each tree's expected value so that
    ``base_value + Σ_features shap == param_p(X)`` exactly.

    ``param_index`` selects the distribution parameter: 0 = loc (the mean / point
    estimate), 1 = logscale (= log of the predictive std, for a Normal Dist).
    """

    def __init__(self, model: NGBoostModel, param_index: int = 0):
        regressor = model.regressor
        self.feature_names = list(model.cat_data.feature_names)
        self.n_features = len(self.feature_names)
        self.param_index = param_index

        lr = float(regressor.learning_rate)
        base = float(np.asarray(regressor.init_params, dtype=float)[param_index])
        self.stages = []  # (weight, col_idx, TreeExplainer)
        for stage_models, scaling, col_idx in zip(
            regressor.base_models, regressor.scalings, regressor.col_idxs
        ):
            tree = stage_models[param_index]
            weight = -lr * float(scaling)
            te = shap.TreeExplainer(tree, feature_perturbation="tree_path_dependent")
            base += weight * float(np.ravel(te.expected_value)[0])
            self.stages.append((weight, np.asarray(col_idx, dtype=int), te))
        self.expected_value = base

    def shap_values(self, Xn: np.ndarray) -> np.ndarray:
        """Compute exact SHAP values for the numeric matrix ``Xn`` (n_rows, n_features)."""
        out = np.zeros((Xn.shape[0], self.n_features), dtype=np.float64)
        for weight, col_idx, te in self.stages:
            sv = np.asarray(te.shap_values(Xn[:, col_idx], check_additivity=False))
            out[:, col_idx] += weight * sv
        return out


def _ngboost_shap(model: NGBoostModel, param_index: int = 0) -> _NGBoostShapExplainer:
    """Build an exact additive SHAP explainer for one NGBoost distribution parameter."""
    return _NGBoostShapExplainer(model, param_index=param_index)


def get_full_ngboost_shaps(
    model: NGBoostModel,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_sales: pd.DataFrame,
    X_univ: pd.DataFrame,
    param_index: int = 0,
    verbose: bool = False,
):
    """Compute exact SHAP Explanations for all subsets for one NGBoost distribution parameter.

    Mirrors :func:`get_full_model_shaps` but for NGBoost's additive tree
    decomposition. ``param_index`` selects loc (0, the mean) or logscale
    (1, log predictive std).
    """
    cat_data = model.cat_data
    explainer = _ngboost_shap(model, param_index=param_index)

    def explain(X, label):
        return _shap_explain(
            "ngboost", explainer, X, cat_data=cat_data, verbose=verbose, label=label
        )

    if verbose:
        which = "loc (mean)" if param_index == 0 else "logscale (uncertainty)"
        print(f"Generating NGBoost SHAPs for {which}...")

    return {
        "train": explain(X_train, "train"),
        "test": explain(X_test, "test"),
        "sales": explain(X_sales, "sales"),
        "universe": explain(X_univ, "universe"),
    }


# ---------------------------------------------------------------------------
# LayeredComp exact tree-SHAP
# ---------------------------------------------------------------------------
#
# A LayeredComp tree is an ordinary binary decision tree, but its prediction is
# NOT the value at the leaf a row reaches -- it is a ``weight_falloff``-weighted
# average of the Wilson-trimmed means of every node on the root->terminal path
# (see ``layeredcompmodel.model.LayeredCompModel._predict_row``). Because that
# blend is a deterministic function of *which* terminal node a row reaches, we
# FOLD it into a single value per terminal node. Folding turns each tree into a
# plain regression tree, to which exact path-dependent tree-SHAP (Lundberg et
# al. 2019, "Consistent Individualized Feature Attribution for Tree Ensembles",
# Algorithm 2) applies directly.
#
# Two LayeredComp-specific wrinkles are handled natively in the recursion --
# rather than by translating the tree into SHAP's numeric-threshold-only tree
# format (which can't express one-vs-rest categorical splits without blowing up
# the tree, and can't reproduce the early-stop semantics at all):
#
#   * Categorical one-vs-rest splits (string equality). The hot/cold child is
#     chosen by equality; everything else in the recursion is feature-identity
#     bookkeeping that does not care how the split was decided.
#   * Early stopping. A NaN at a numeric split -- or a routed-to child that was
#     never created because its partition was empty -- makes the model stop and
#     emit the *current* node's folded value (it does not descend to a leaf). We
#     model this with a synthetic "stop" leaf per node whose cover is the number
#     of training rows that stopped there (parent.count - sum(child.count)), so
#     both the prediction and the cover-weighted base value stay exact.
#
# The ensemble SHAP is the mean of the per-tree SHAP values, mirroring
# ``LayeredCompBaggingModel.predict`` (a plain mean over trees), with a
# correspondingly averaged base value.


class _FoldedNode:
    """One node of a folded LayeredComp tree (see module-level note)."""

    __slots__ = (
        "feat", "is_numeric", "val", "cover", "value",
        "children", "n_real", "stop_index",
    )

    def __init__(self, feat, is_numeric, val, cover, value, children, n_real, stop_index):
        self.feat = feat            # feature index, or -1 for a leaf / stop leaf
        self.is_numeric = is_numeric
        self.val = val              # split threshold (numeric) or category value
        self.cover = cover          # node_sample_weight = training rows reaching here
        self.value = value          # folded prediction if a row terminates here
        self.children = children    # tuple of _FoldedNode; real children first, stop last
        self.n_real = n_real        # number of *real* (non-stop) children
        self.stop_index = stop_index  # index of the stop child in ``children``, or None

    @property
    def is_leaf(self) -> bool:
        return self.feat < 0


def _fold_path_value(path_means: list, falloff: float) -> float:
    """Collapse a root->node path of Wilson means into the model's predicted value.

    Replicates ``LayeredCompModel._predict_row``'s weighting exactly: for a path
    of ``n`` nodes, node ``i`` (root=0, leaf=n-1) gets weight ``(1 - x)**falloff``
    with ``x = (n - 1 - i) / (n - 1)``, then the weighted mean is normalized.
    """
    n = len(path_means)
    if n == 1:
        return float(path_means[0])
    weighted_sum = 0.0
    total_w = 0.0
    for i, m in enumerate(path_means):
        x = (n - 1 - i) / (n - 1)
        w = (1.0 - x) ** falloff
        weighted_sum += m * w
        total_w += w
    return float(weighted_sum / total_w)


def _expected_value(node: _FoldedNode) -> float:
    """Cover-weighted mean of folded leaf values below ``node`` (E[f] at root)."""
    if node.is_leaf or node.cover <= 0:
        return node.value
    acc = 0.0
    for child in node.children:
        acc += (child.cover / node.cover) * _expected_value(child)
    return acc


class _LayeredCompShapExplainer:
    """Exact path-dependent tree-SHAP for a LayeredComp bagging ensemble.

    Accepts either an :class:`openavmkit.utilities.modeling.LayeredCompModel`
    wrapper or a raw ``LayeredCompBaggingModel``. ``shap_values(X)`` returns an
    ``(n_rows, n_features)`` array; ``base_value + row.sum()`` reconstructs
    ``model.predict(X)`` exactly (up to floating point).
    """

    def __init__(self, model):
        bag = getattr(model, "model", model)
        self.feature_names = list(bag.feature_names_in_)
        self.n_features = len(self.feature_names)
        self._feat_index = {f: i for i, f in enumerate(self.feature_names)}
        self.trees = []
        tree_bases = []
        for est in bag.estimators_:
            root = self._fold(est.tree_, float(est.weight_falloff), [])
            self.trees.append(root)
            tree_bases.append(_expected_value(root))
        self.expected_value = float(np.mean(tree_bases)) if tree_bases else 0.0

    def _fold(self, node, falloff: float, path_means: list) -> _FoldedNode:
        pm = path_means + [float(node.wilson_mean)]
        folded = _fold_path_value(pm, falloff)
        children = node.children if node.children else []

        if not children or node.filter_col is None:
            return _FoldedNode(
                feat=-1, is_numeric=False, val=None, cover=float(node.count),
                value=folded, children=(), n_real=0, stop_index=None,
            )

        kids = [self._fold(ch, falloff, pm) for ch in children]
        n_real = len(kids)
        sum_cov = sum(k.cover for k in kids)
        stop_cov = float(node.count) - sum_cov

        # A row can land on the stop leaf when a numeric split sees a NaN, or
        # when it routes to a child that does not exist (empty partition). Create
        # one whenever that is reachable; cover may be 0 (it then contributes
        # nothing to the base value, but still gives such a row a value to read).
        is_numeric = bool(node.is_numeric)
        need_stop = stop_cov > 1e-9 or is_numeric or n_real < 2
        stop_index = None
        if need_stop:
            kids.append(_FoldedNode(
                feat=-1, is_numeric=False, val=None, cover=max(stop_cov, 0.0),
                value=folded, children=(), n_real=0, stop_index=None,
            ))
            stop_index = len(kids) - 1

        return _FoldedNode(
            feat=self._feat_index[node.filter_col],
            is_numeric=is_numeric,
            val=node.filter_val,
            cover=float(node.count),
            value=folded,
            children=tuple(kids),
            n_real=n_real,
            stop_index=stop_index,
        )

    def _hot_index(self, node: _FoldedNode, xvals) -> int:
        """Which child the instance routes to -- mirrors ``_predict_row`` exactly."""
        rv = xvals[node.feat]
        if node.is_numeric:
            if pd.isna(rv):
                return node.stop_index
            rvn = pd.to_numeric(rv, errors="coerce")
            valn = pd.to_numeric(node.val, errors="coerce")
            if pd.isna(rvn) or pd.isna(valn):
                return node.stop_index
            if rvn <= valn:
                return 0
            return 1 if node.n_real >= 2 else node.stop_index
        # categorical one-vs-rest
        rs = "NaN" if pd.isna(rv) else str(rv)
        if rs == str(node.val):
            return 0
        return 1 if node.n_real >= 2 else node.stop_index

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        out = np.zeros((n, self.n_features), dtype=np.float64)

        # Align inputs to the training feature order; missing columns -> NaN.
        xarr = np.empty((n, self.n_features), dtype=object)
        for j, f in enumerate(self.feature_names):
            if f in X.columns:
                xarr[:, j] = X[f].to_numpy(dtype=object)
            else:
                xarr[:, j] = np.nan

        # SHAP values depend only on the row's feature vector, so identical rows
        # share a result -- a big win on assessment data with repeated profiles.
        sig_to_rows: dict = {}
        for r in range(n):
            sig_to_rows.setdefault(tuple(xarr[r]), []).append(r)

        n_trees = len(self.trees) or 1
        for sig, rows in sig_to_rows.items():
            xvals = list(sig)
            phi = np.zeros(self.n_features, dtype=np.float64)
            for root in self.trees:
                _tree_shap(root, xvals, phi, self._hot_index)
            phi /= n_trees
            out[rows] = phi
        return out


def _tree_shap(root: _FoldedNode, xvals, phi: np.ndarray, hot_index) -> None:
    """Accumulate path-dependent SHAP for one folded tree into ``phi`` in place.

    Direct port of Lundberg et al. (2019) Algorithm 2 (EXTEND/UNWIND on the
    "unique path"), generalized so a node may have more than two children (the
    real children plus the synthetic stop leaf). The first path element is a
    sentinel at index 0 and is skipped when attributing.
    """

    def extend(d, z, o, w, pz, po, pi):
        d = d + [pi]; z = z + [pz]; o = o + [po]; w = w + [0.0]
        l = len(d) - 1
        w[l] = 1.0 if l == 0 else 0.0
        for i in range(l - 1, -1, -1):
            w[i + 1] += po * w[i] * (i + 1) / (l + 1)
            w[i] = pz * w[i] * (l - i) / (l + 1)
        return d, z, o, w

    def unwind(d, z, o, w, idx):
        l = len(d) - 1
        oi, zi = o[idx], z[idx]
        w = list(w)
        n = w[l]
        for j in range(l - 1, -1, -1):
            if oi != 0:
                t = w[j]
                w[j] = n * (l + 1) / ((j + 1) * oi)
                n = t - w[j] * zi * (l - j) / (l + 1)
            elif zi != 0:
                w[j] = w[j] * (l + 1) / (zi * (l - j))
            else:
                # Degenerate z=0,o=0 element (cold zero-cover); contributes no
                # weight. Cold zero-cover branches are skipped before they reach
                # the path, so this is only a defensive guard.
                w[j] = 0.0
        d = d[:idx] + d[idx + 1:]
        z = z[:idx] + z[idx + 1:]
        o = o[:idx] + o[idx + 1:]
        return d, z, o, w[:l]

    def recurse(node, d, z, o, w, pz, po, pi):
        d, z, o, w = extend(d, z, o, w, pz, po, pi)
        if node.is_leaf:
            v = node.value
            for i in range(1, len(d)):
                _, _, _, w2 = unwind(d, z, o, w, i)
                phi[d[i]] += sum(w2) * (o[i] - z[i]) * v
            return
        hot = hot_index(node, xvals)
        # If this feature is already on the path, splice it out and carry its
        # incoming fractions forward (handles a feature reused down the tree).
        iz = io = 1.0
        k = None
        for idx in range(1, len(d)):
            if d[idx] == node.feat:
                k = idx
                break
        if k is not None:
            iz, io = z[k], o[k]
            d, z, o, w = unwind(d, z, o, w, k)
        cov = node.cover
        for c, child in enumerate(node.children):
            # A zero-cover branch carries no training mass, so it contributes
            # nothing when its feature is marginalized out (cold). Skipping it
            # also keeps the path free of degenerate z=0,o=0 elements that would
            # divide-by-zero in UNWIND. The synthetic stop leaf is cover-0
            # whenever no training row stopped there (the norm once NaNs are
            # filled); it still gets visited when it is the *hot* child (a NaN
            # row routing to it), where o>0 keeps UNWIND well defined.
            if c != hot and child.cover == 0:
                continue
            cz = iz * (child.cover / cov) if cov > 0 else 0.0
            cp = io if c == hot else 0.0
            recurse(child, d, z, o, w, cz, cp, node.feat)

    recurse(root, [], [], [], [], 1.0, 1.0, -1)


def _layeredcomp_shap(model) -> _LayeredCompShapExplainer:
    """Build an exact path-dependent SHAP explainer for a LayeredComp model."""
    return _LayeredCompShapExplainer(model)


def _to_numeric_data(X: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Numeric matrix aligned to ``feature_names`` for SHAP plotting (categories -> codes)."""
    n = len(X)
    arr = np.full((n, len(feature_names)), np.nan, dtype=np.float64)
    for j, f in enumerate(feature_names):
        if f not in X.columns:
            continue
        col = X[f]
        if pd.api.types.is_numeric_dtype(col):
            arr[:, j] = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
        else:
            arr[:, j] = pd.factorize(col)[0].astype(np.float64)
    return arr


def get_full_layeredcomp_shaps(
    model: LayeredCompModel,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_sales: pd.DataFrame,
    X_univ: pd.DataFrame,
    verbose: bool = False,
):
    """Compute exact SHAP Explanations for all subsets of a LayeredComp model.

    Mirrors :func:`get_full_model_shaps` / :func:`get_full_ngboost_shaps` for the
    LayeredComp engine's folded-tree decomposition.
    """
    explainer = _layeredcomp_shap(model)

    def explain(X, label):
        return _shap_explain("layeredcomp", explainer, X, verbose=verbose, label=label)

    if verbose:
        print("Generating LayeredComp SHAPs...")

    return {
        "train": explain(X_train, "train"),
        "test": explain(X_test, "test"),
        "sales": explain(X_sales, "sales"),
        "universe": explain(X_univ, "universe"),
    }


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
        A dict containing shap.Explanation objects keyed to "train", "test", "sales", and "universe"

    """

    # NGBoost uses an exact additive tree decomposition rather than a single
    # TreeExplainer; delegate to its dedicated path (param_index=0 -> mean).
    if isinstance(model, NGBoostModel):
        return get_full_ngboost_shaps(
            model, X_train, X_test, X_sales, X_univ, param_index=0, verbose=verbose
        )

    # LayeredComp folds its path-weighted prediction into per-leaf values and
    # gets exact path-dependent tree-SHAP via its own hand-rolled explainer.
    if isinstance(model, LayeredCompModel):
        return get_full_layeredcomp_shaps(
            model, X_train, X_test, X_sales, X_univ, verbose=verbose
        )

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
        "universe":  shap_univ,
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
    
    feature_names = explanation.feature_names
    if not feature_names:
        feature_names = []
    
    # Wrap feature names
    wrapped_names = [
        "\n".join(textwrap.wrap(fn, width=wrap_width))
        for fn in feature_names
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
    if isinstance(model, NGBoostModel):
        # NGBoost uses an exact additive tree decomposition (param 0 = mean / loc).
        if not ngboost_internals_ok(model.regressor):
            return None
        return _shap_explain(
            "ngboost", _ngboost_shap(model, 0), X_to_explain, cat_data=model.cat_data
        )
    if isinstance(model, LayeredCompModel):
        return _shap_explain("layeredcomp", _layeredcomp_shap(model), X_to_explain)
    if isinstance(model, XGBoostModel):
        explainer = _xgboost_shap(model, X_train, background_size=background_size)
    elif isinstance(model, LightGBMModel):
        explainer = _lightgbm_shap(model, X_train, background_size=background_size)
    elif isinstance(model, CatBoostModel):
        explainer = _catboost_shap(model, X_train, background_size=background_size)
    else:
        explainer = shap.Explainer(model, X_train)
    explanation = explainer(X_to_explain)
    if not hasattr(explanation, "values"):
        raise TypeError(f"Expected shap.Explanation, got {type(explanation)}")
    return explanation


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
    
    if (X_to_explain is None) or len(X_to_explain) == 0:
        return None

    # --- LayeredComp folded-tree exact path-dependent SHAP -----------------
    if model_type == "layeredcomp":
        explainer = te  # a _LayeredCompShapExplainer
        values = explainer.shap_values(X_to_explain)
        data = _to_numeric_data(X_to_explain, explainer.feature_names)
        return shap.Explanation(
            values=values,
            base_values=np.full(len(X_to_explain), explainer.expected_value, dtype=np.float64),
            data=data,
            feature_names=list(explainer.feature_names),
        )

    # --- NGBoost exact additive decomposition ------------------------------
    if model_type == "ngboost":
        # NGBoost's base learners are numeric-only; encode via the same numeric
        # matrix used at training time (categories -> codes, missing -> NaN).
        Xn = cat_data.to_numeric_matrix(X_to_explain)
        values = te.shap_values(Xn)
        return shap.Explanation(
            values=values,
            base_values=np.full(Xn.shape[0], te.expected_value, dtype=np.float64),
            data=Xn,
            feature_names=list(cat_data.feature_names),
        )

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