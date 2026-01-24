import xgboost as xgb
import lightgbm as lgb
import gpboost as gpb
import numpy as np
import optuna
import pandas as pd
import time
from catboost import Pool, CatBoostRegressor, cv
import os
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from optuna.integration import CatBoostPruningCallback

from openavmkit.utilities.modeling import _to_gpboost_label

#######################################
# PRIVATE
#######################################


def _tune_xgboost(
    X,
    y,
    sizes,
    he_ids,
    n_trials=50,
    n_splits=5,
    random_state=42,
    cat_vars=None,
    verbose=False,
):
    """Tunes XGBoost hyperparameters using Optuna and rolling-origin cross-validation.
    Uses the xgboost.train API for training. Includes logging for progress monitoring.
    """

    def objective(trial):
        """Objective function for Optuna to optimize XGBoost hyperparameters."""
        params = {
            "objective": "reg:squarederror",  # Regression objective
            "eval_metric": "mape",  # Mean Absolute Percentage Error
            "tree_method": "hist",  # Use 'hist' for performance; use 'gpu_hist' for GPUs
            "enable_categorical": True,
            "max_cat_to_onehot": 1,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 10, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, log=False),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.4, 1.0, log=False
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.4, 1.0, log=False
            ),
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode", 0.4, 1.0, log=False
            ),
            "gamma": trial.suggest_float("gamma", 0.1, 10, log=True),  # min_split_loss
            "lambda": trial.suggest_float("lambda", 1e-4, 10, log=True),  # reg_lambda
            "alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),  # reg_alpha
            "max_bin": trial.suggest_int(
                "max_bin", 64, 512
            ),  # Relevant for 'hist' tree_method
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
        }
        num_boost_round = trial.suggest_int("num_boost_round", 100, 3000)

        mape = _xgb_rolling_origin_cv(
            X,
            y,
            params,
            num_boost_round,
            n_splits,
            random_state,
            verbose_eval=False,
            sizes=sizes,
            he_ids=he_ids,
            custom_alpha=0.1,
        )
        if verbose:
            print(
                f"-->trial # {trial.number}/{n_trials}, MAPE: {mape:0.4f}"
            )  # , params: {params}")
        return mape  # Optuna minimizes, so return the MAPE directly

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective, n_trials=n_trials, n_jobs=-1, callbacks=[_plateau_callback]
    )
    if verbose:
        print(
            f"Best trial: {study.best_trial.number} with MAPE: {study.best_trial.value:0.4f} and params: {study.best_trial.params}"
        )
    return study.best_params


def _tune_lightgbm(
    X,
    y,
    sizes,
    he_ids,
    n_trials=50,
    n_splits=5,
    random_state=42,
    cat_vars=None,
    verbose=False,
):
    """Tunes LightGBM hyperparameters using Optuna and rolling-origin cross-validation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        sizes (array-like): Array of size values (land or building size)
        he_ids (array-like): Array of horizontal equity cluster ID's
        n_trials (int): Number of optimization trials for Optuna. Default is 100.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.
        verbose (bool): Whether to print Optuna progress.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """

    def objective(trial):
        """Objective function for Optuna to optimize LightGBM hyperparameters."""
        params = {
            "objective": "regression",
            "metric": "mape",
            "boosting_type": "gbdt",
            "num_iterations": trial.suggest_int("num_iterations", 300, 5000),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0001, 0.1, log=True
            ),
            "max_bin": trial.suggest_int("max_bin", 64, 1024),
            "num_leaves": trial.suggest_int("num_leaves", 64, 2048),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split", 1e-4, 50, log=True
            ),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.4, 0.9, log=False
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8, log=False),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10, log=True),
            "cat_smooth": trial.suggest_int("cat_smooth", 5, 200),
            "verbosity": -1,
            "early_stopping_round": 50,
        }

        # Use rolling-origin cross-validation
        mape = _lightgbm_rolling_origin_cv(
            X, y, params, n_splits=n_splits, random_state=random_state, cat_vars=cat_vars
        )
        if verbose:
            print(
                f"-->trial # {trial.number}/{n_trials}, MAPE: {mape:0.4f}"
            )  # , params: {params}")
        return mape  # Optuna minimizes, so return the MAPE directly

    # Run Bayesian Optimization with Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        objective, n_trials=n_trials, n_jobs=-1, callbacks=[_plateau_callback]
    )  # Use parallelism if available

    if verbose:
        print(
            f"Best trial: {study.best_trial.number} with MAPE: {study.best_trial.value:0.4f} and params: {study.best_trial.params}"
        )
    return study.best_params


def _tune_gpboost(
    X,
    y,
    sizes,
    he_ids,
    n_trials=50,
    n_splits=5,
    random_state=42,
    policy=None,
    cat_vars=None,
    verbose=False,
    gp_coords=None,
    group_data=None,
    cluster_ids=None,
    gp_model_params=None,
    tune_gp_model=False,
    cv_strategy: str = 'kfold',
):
    """Tunes GPBoost hyperparameters using Optuna and rolling-origin cross-validation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        sizes (array-like): Array of size values (land or building size)
        he_ids (array-like): Array of horizontal equity cluster ID's
        n_trials (int): Number of optimization trials for Optuna. Default is 100.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.
        verbose (bool): Whether to print Optuna progress.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """

    def objective(trial):
        """Objective function for Optuna to optimize GPBoost hyperparameters."""
        params = {
            "objective": "regression_l2",
            "metric": "mape",
            "boosting_type": "gbdt",
            "num_iterations": trial.suggest_int("num_iterations", 300, 5000),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.0001, 0.1, log=True
            ),
            "max_bin": trial.suggest_int("max_bin", 64, 1024),
            "num_leaves": trial.suggest_int("num_leaves", 64, 2048),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split", 1e-4, 50, log=True
            ),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.4, 0.9, log=False
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8, log=False),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10, log=True),
            "cat_smooth": trial.suggest_int("cat_smooth", 5, 200),
            "verbosity": -1,
        }

        # Optionally tune GP / random-effects hyperparameters as well
        gp_params = dict(gp_model_params or {})
        if tune_gp_model:
            gp_params.setdefault('likelihood', 'gaussian')
            gp_params['cov_function'] = trial.suggest_categorical('cov_function', ['matern', 'exponential', 'gaussian'])
            gp_params['gp_approx'] = trial.suggest_categorical('gp_approx', ['none', 'vecchia'])
            if gp_params['gp_approx'] == 'vecchia':
                gp_params['num_neighbors'] = trial.suggest_int('num_neighbors', 10, 60)

        # Cross-validation (must include GP model if gp_coords / group_data provided)
        
        mape = _gpboost_cv(
            X,
            y,
            params,
            n_splits=n_splits,
            random_state=random_state,
            policy=policy,
            cat_vars=cat_vars,
            gp_coords=gp_coords,
            group_data=group_data,
            cluster_ids=cluster_ids,
            gp_model_params=gp_params,
            cv_strategy=cv_strategy,
        )
        if verbose:
            print(
                f"-->trial # {trial.number}/{n_trials}, MAPE: {mape:0.4f}"
            )  # , params: {params}")
        return mape  # Optuna minimizes, so return the MAPE directly

    # Run Bayesian Optimization with Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    
    progress_cb = OptunaProgress(total_trials=n_trials, print_every=1)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        callbacks=[_plateau_callback, progress_cb],
        show_progress_bar=True,   # <-- add this
    )

    if verbose:
        print(
            f"Best trial: {study.best_trial.number} with MAPE: {study.best_trial.value:0.4f} and params: {study.best_trial.params}"
        )
    return study.best_params


def _tune_catboost(
    X,
    y,
    sizes,
    he_ids,
    verbose=False,
    cat_vars=None,
    n_trials=50,
    n_splits=5,
    random_state=42,
    use_gpu=True
):

    # Pre-build a single Pool for CV
    cat_feats = [c for c in (cat_vars or []) if c in X.columns]
    full_pool = Pool(X, y, cat_features=cat_feats)
    
    #task_type = "GPU" if use_gpu else "CPU"
    task_type = "CPU" # GPU is too unreliable for now, so default catboost to CPU
    
    if verbose:
        print(f"Tuning Catboost. n_trials={n_trials}, n_splits={n_splits}, use_gpu={use_gpu}")
    
    def objective(trial):
        params = {
            "loss_function": "MAPE",
            "eval_metric": "MAPE",
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "border_count": trial.suggest_int("border_count", 32, 64),
            "random_strength": trial.suggest_float("random_strength", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "bootstrap_type": "Bayesian",
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
            "boosting_type": "Plain",
            "task_type": task_type,
            "random_seed": random_state,
            "verbose": False,
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
            ),
        }

        # Additional param only for Lossguide
        if params["grow_policy"] == "Lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 31, 128)

        # Use CatBoost's built-in CV (MUCH faster)
        cv_results = cv(
            full_pool,
            params,
            fold_count=n_splits,
            partition_random_seed=random_state,
            early_stopping_rounds=100,
            verbose=False,
        )

        # Optuna Pruner: report learning curve as it trains
        # Extract the test MAPE curve
        mape_curve = cv_results["test-MAPE-mean"]
        for i, v in enumerate(mape_curve):
            trial.report(v, step=i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Objective = final CV MAPE
        return mape_curve.iloc[-1]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15, n_warmup_steps=100, interval_steps=10
        ),
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    if verbose:
        print(
            f"Best trial #{study.best_trial.number} → MAPE={study.best_trial.value:.4f}"
        )
        print("Params:", study.best_trial.params)

    return study.best_params


def _plateau_callback(study, trial):
    """Stops the study if no significant improvement (>= 1% over the current best value)
    is observed over the last 10 trials."""
    plateau_trials = 10
    improvement_threshold = 0.01  # require at least 1% improvement

    # Only check if we've completed enough trials.
    if trial.number < plateau_trials:
        return

    # Get the last plateau_trials trials.
    recent_trials = study.trials[-plateau_trials:]
    best_value = study.best_trial.value

    # If none of the recent trials improved the best value by more than the threshold, stop the study.

    # guard against null values in best_value:
    if best_value is None:
        return

    if all(
        t.value is not None and t.value >= best_value * (1 + improvement_threshold)
        for t in recent_trials
    ):
        print(
            "Plateau detected: no significant improvement in the last "
            f"{plateau_trials} trials. Stopping study early."
        )
        study.stop()


def _xgb_custom_obj_variance_factory(size, cluster, alpha=0.1):
    """Returns a custom objective function for XGBoost that adds a variance-based reward
    term on the normalized predictions (prediction/size) within each cluster.

    Parameters:
      size   : numpy array of "size" values (one per training instance)
      cluster: numpy array of "cluster_id" (one per instance)
      alpha  : weighting factor for the custom reward term relative to MSE.
    """

    def custom_obj(preds, dtrain):
        labels = dtrain.get_label()

        # Standard MSE gradient and hessian
        grad_mse = preds - labels
        hess_mse = np.ones_like(preds)

        # Prepare arrays for custom variance gradient and hessian
        grad_custom = np.zeros_like(preds)
        hess_custom = np.zeros_like(preds)

        # Process each cluster separately
        unique_clusters = np.unique(cluster)
        for cl in unique_clusters:
            idx = np.where(cluster == cl)[0]
            if len(idx) == 0:
                continue

            n = len(idx)
            # Compute A = prediction/size for each row in this cluster
            A = preds[idx] / size[idx]
            m = np.mean(A)

            # Compute gradient for the variance term:
            # dV/dA_i = (2/n)*(A_i - m)
            # Then by chain rule: dV/dp_i = dV/dA_i * (1/size)
            grad_custom[idx] = (2.0 / n) * (A - m) * (1.0 / size[idx])

            # Approximate Hessian: 2/n * (1/size^2)
            hess_custom[idx] = (2.0 / n) * (1.0 / (size[idx] ** 2))

        # Combine the standard MSE with the custom variance reward term
        grad = grad_mse + alpha * grad_custom
        hess = hess_mse + alpha * hess_custom

        return grad, hess

    return custom_obj


def _xgb_rolling_origin_cv(
    X,
    y,
    params,
    num_boost_round,
    n_splits=5,
    random_state=42,
    verbose_eval=50,
    sizes=None,
    he_ids=None,
    custom_alpha=0.1,
):
    """Performs rolling-origin cross-validation for XGBoost model evaluation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        params (dict): XGBoost hyperparameters.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.
        verbose_eval (int|bool): Logging interval for XGBoost. Default is 50.

    Returns:
        float: Mean MAPE score across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mape_scores = []

    for train_idx, val_idx in kf.split(X):
        if hasattr(X, 'iloc'):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        val_data = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        evals = [(val_data, "validation")]

        # If custom arrays are provided, subset them for training data and build custom objective
        custom_obj = None
        # TODO: enable this later
        # if sizes is not None and he_ids is not None:
        #     custom_obj = _xgb_custom_obj_variance_factory(size=sizes, cluster=he_ids, alpha=custom_alpha)

        # Train XGBoost
        model = xgb.train(
            params=params,
            dtrain=train_data,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=verbose_eval,  # Ensure verbose_eval is enabled
            obj=custom_obj,
        )

        # Predict and evaluate
        y_pred = model.predict(val_data, iteration_range=(0, model.best_iteration))
        mape = mean_absolute_percentage_error(y_val, y_pred)
        mape_scores.append(mape)

    return np.mean(mape_scores)


def _catboost_rolling_origin_cv(
    X, y, params, n_splits=5, random_state=42, cat_vars=None, verbose=False
):
    """Performs rolling-origin cross-validation for CatBoost model evaluation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        params (dict): CatBoost hyperparameters.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.
        cat_vars (list): List of categorical variables. Default is None.
        verbose (bool): Whether to print CatBoost training logs.

    Returns:
        float: Mean MAPE score across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mape_scores = []

    for train_idx, val_idx in kf.split(X):
        # Use .iloc for DataFrame-like objects
        if hasattr(X, 'iloc'):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        _cat_vars_train = [var for var in cat_vars if var in X_train.columns.values]
        _cat_vars_val = [var for var in cat_vars if var in X_val.columns.values]

        # scan categorical variables, look for any that contain NaN or floating-point values:
        for var in _cat_vars_train:
            dtype = X_train[var].dtype
            if dtype == "float64" or dtype == "float32":
                raise ValueError(
                    f"Categorical variable '{var}' contains floating-point values. Please convert to integer or string."
                )
            if X_train[var].isnull().any():
                raise ValueError(
                    f"Categorical variable '{var}' contains NaN values. Please handle them before training."
                )
            if X_val[var].isnull().any():
                raise ValueError(
                    f"Categorical variable '{var}' contains NaN values in validation set. Please handle them before training."
                )
            if dtype == "object":
                # check if any values in this field are non-integer (real) numbers:
                if not X_train[var].apply(lambda x: isinstance(x, (int, str))).all():
                    raise ValueError(
                        f"Categorical variable '{var}' contains non-integer values. Please convert to integer or string."
                    )
                if not X_val[var].apply(lambda x: isinstance(x, (int, str))).all():
                    raise ValueError(
                        f"Categorical variable '{var}' contains non-integer values in validation set. Please convert to integer or string."
                    )

        train_pool = Pool(X_train, y_train, cat_features=_cat_vars_train)
        val_pool = Pool(X_val, y_val, cat_features=_cat_vars_val)

        # Train CatBoost
        model = CatBoostRegressor(**params)
        model.fit(
            train_pool, eval_set=val_pool, verbose=verbose, early_stopping_rounds=50
        )

        # Predict and evaluate
        y_pred = model.predict(val_pool)
        mape_scores.append(mean_absolute_percentage_error(y_val, y_pred))

    return np.mean(mape_scores)


def _lightgbm_rolling_origin_cv(X, y, params, n_splits=5, random_state=42, cat_vars=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mape_scores = []

    for train_idx, val_idx in kf.split(X):
        if hasattr(X, "iloc"):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        # Determine categorical features present in this fold
        cat_feats = [c for c in (cat_vars or []) if hasattr(X_train, "columns") and c in X_train.columns]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feats, reference=train_data)

        # Work on a fold-local copy to avoid cross-fold mutation
        fold_params = dict(params)
        fold_params["verbosity"] = -1

        num_boost_round = 1000
        if "num_iterations" in fold_params:
            num_boost_round = fold_params.pop("num_iterations")

        model = lgb.train(
            fold_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=5, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        mape_scores.append(mean_absolute_percentage_error(y_val, y_pred))

    return np.mean(mape_scores)


def _tlog(t0: float, msg: str):
    print(f"[GPB-CV {time.time() - t0:7.1f}s] {msg}")


def _gpboost_cv(
    X,
    y,
    params,
    n_splits=5,
    random_state=42,
    policy=None,
    cat_vars=None,
    gp_coords=None,
    group_data=None,
    cluster_ids=None,
    gp_model_params=None,
    cv_strategy: str = "kfold",
):
    """Cross-validation for GPBoost.

    IMPORTANT: If gp_coords and/or group_data are provided, this CV trains and validates with a fold-local GPModel so
    the tuned tree hyperparameters reflect the *combined* tree + GP/RE model (not just the LightGBM-like part).

    Parameters
    ----------
    gp_coords : np.ndarray | None
        Array of shape (n, 2) aligned to X row order.
    group_data : np.ndarray | None
        Array of shape (n,) or (n, k) aligned to X row order (supports multiple grouping factors).
    cluster_ids : np.ndarray | None
        Array of shape (n,) aligned to X row order, for independent GP realizations (e.g. multiple markets).
    gp_model_params : dict | None
        Extra kwargs passed to gpb.GPModel(...).
    cv_strategy : {'kfold','timeseries'}
        If 'timeseries', uses TimeSeriesSplit (X must already be sorted by time).
    """
    
    DEBUG_DISABLE_GP = False
    
    if cv_strategy not in {"kfold", "timeseries"}:
        raise ValueError(f"cv_strategy must be 'kfold' or 'timeseries', got {cv_strategy!r}")

    splitter = (
        TimeSeriesSplit(n_splits=n_splits)
        if cv_strategy == "timeseries"
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )

    mape_scores = []
    gp_model_params = gp_model_params or {}
    gp_model_params.setdefault("gp_approx", "vecchia")
    gp_model_params.setdefault("num_neighbors", 30)  # 20–60 typical

    # Require numpy arrays if effects are used
    if gp_coords is not None:
        gp_coords = np.asarray(gp_coords)
    if group_data is not None:
        group_data = np.asarray(group_data)
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
    
    t0 = time.time()
    _tlog(t0, f"CV start | n={len(X)} | folds={n_splits} | "
          f"gp={gp_coords is not None} group={group_data is not None} "
          f"cluster={cluster_ids is not None}")

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        _tlog(t0, f"Fold {fold_i}/{n_splits}: slice data")
        if hasattr(X, "iloc"):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        # Determine categorical features present in this fold
        cat_feats = [
            c for c in (cat_vars or [])
            if hasattr(X_train, "columns") and c in X_train.columns
        ]
        
        # Labels must be numeric arrays
        y_train_arr = _to_gpboost_label(y_train)
        y_val_arr   = _to_gpboost_label(y_val)
        
        _tlog(t0, f"Fold {fold_i}: build Dataset objects (train={len(train_idx)} val={len(val_idx)})")
        
        # Build datasets using fold-specific categorical list
        train_data = gpb.Dataset(
            X_train,
            y_train_arr,
            categorical_feature=cat_feats,
            free_raw_data=False,
        )
        
        # IMPORTANT: reference=train_data reduces overrides & ensures consistent metadata
        val_data = gpb.Dataset(
            X_val,
            y_val_arr,
            categorical_feature=cat_feats,
            reference=train_data,
            free_raw_data=False,
        )
        
        _tlog(t0, f"Fold {fold_i}: Dataset.construct()")
        train_data.construct()
        val_data.construct()

        fold_params = dict(params)
        fold_params["verbosity"] = -1

        num_boost_round = fold_params.pop("num_iterations", 1000)
        # debug
        num_boost_round = min(num_boost_round, 200)
        
        _tlog(t0, f"Fold {fold_i}: GPModel init")
        
        if policy is None:
            policy = GPBoostEffectsPolicy() # defaults afe auto
        
        # Build fold-local GPModel if effects are provided
        gp_model = None
        use_gp_val = False
        
        has_coords = gp_coords is not None
        has_groups = group_data is not None
        
        use_coords, use_groups, gp_patch, reason = choose_gpboost_effects(
            n_train=len(train_idx),
            has_coords=has_coords,
            has_groups=has_groups,
            policy=policy,
        )
        
        if policy.verbose_choices and (fold_i == 1):
            _tlog(t0, f"Effects choice: {reason}")
        
        if use_coords or use_groups:
            fold_gp_params = dict(gp_model_params)
            fold_gp_params.update(gp_patch)
            fold_gp_params.setdefault("likelihood", "gaussian")
        
            gp_model = gpb.GPModel(
                gp_coords=(gp_coords[train_idx] if (use_coords and gp_coords is not None) else None),
                group_data=(group_data[train_idx] if (use_groups and group_data is not None) else None),
                cluster_ids=(cluster_ids[train_idx] if cluster_ids is not None else None),
                likelihood=fold_gp_params.get("likelihood", "gaussian"),
                **{k: v for k, v in fold_gp_params.items() if k != "likelihood"},
            )
        
            gp_model.set_prediction_data(
                gp_coords_pred=(gp_coords[val_idx] if (use_coords and gp_coords is not None) else None),
                group_data_pred=(group_data[val_idx] if (use_groups and group_data is not None) else None),
                cluster_ids_pred=(cluster_ids[val_idx] if cluster_ids is not None else None),
            )
            use_gp_val = True

        
        fold_params.setdefault("num_threads", os.cpu_count() or 4)
        _tlog(t0, f"Fold {fold_i}: train() num_boost_round={num_boost_round} num_threads={fold_params.get('num_threads')}")
        train_start = time.time()
        
        fold_params.setdefault("num_threads", -1)
        fold_params.setdefault("n_jobs", -1)
        
        model = gpb.train(
            fold_params,
            train_data,
            num_boost_round=num_boost_round,
            gp_model=gp_model,
            use_gp_model_for_validation=use_gp_val,
            valid_sets=[val_data],
            early_stopping_rounds=5,
            verbose_eval=1,
        )
        
        _tlog(t0, f"Fold {fold_i}: train finished in {time.time() - train_start:.1f}s")
        
        _tlog(t0, f"Fold {fold_i}: train done (best_iter={getattr(model, 'best_iteration', None)})")

        # Predict with effects if they exist
        pred_kwargs = {"num_iteration": model.best_iteration}
        if gp_coords is not None:
            pred_kwargs["gp_coords_pred"] = gp_coords[val_idx]
        if group_data is not None:
            pred_kwargs["group_data_pred"] = group_data[val_idx]
        if cluster_ids is not None:
            pred_kwargs["cluster_ids_pred"] = cluster_ids[val_idx]

        y_pred = model.predict(X_val, **pred_kwargs)
        mape_scores.append(mean_absolute_percentage_error(y_val, y_pred))

    return float(np.mean(mape_scores))


class OptunaProgress:
    def __init__(self, total_trials: int, print_every: int = 1):
        self.total = total_trials
        self.print_every = print_every
        self.t0 = None
        self.last_print_n = 0

    def __call__(self, study, trial):
        # trial is finished (COMPLETE/PRUNED/FAIL) when callback runs
        if self.t0 is None:
            self.t0 = time.time()

        n_done = len(study.trials)
        # Print only every N finished trials
        if n_done - self.last_print_n < self.print_every:
            return
        self.last_print_n = n_done

        elapsed = time.time() - self.t0
        # Count only completed trials for "rate" stability
        completed = [t for t in study.trials if t.value is not None]
        n_complete = len(completed)

        if n_complete == 0:
            rate = None
        else:
            rate = elapsed / n_complete  # sec per completed trial

        remaining = self.total - n_done
        if rate is None or remaining <= 0:
            eta_str = "ETA: ?"
        else:
            eta = remaining * rate
            eta_str = f"ETA: {int(eta//60)}m {int(eta%60)}s"

        best = study.best_value if study.best_trial is not None else None
        best_str = f"best={best:.5f}" if best is not None else "best=?"

        print(f"[Optuna] {n_done}/{self.total} trials done | {best_str} | {eta_str}")
