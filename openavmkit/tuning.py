import xgboost as xgb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import Pool, CatBoostRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from optuna.integration import CatBoostPruningCallback

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
            X, y, params, n_splits=n_splits, random_state=random_state
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
):
    # 1) Pre‐split once, build Pools
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_pools = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        cat_feats = [c for c in cat_vars if c in X.columns]
        cv_pools.append(
            (
                Pool(X_tr, y_tr, cat_features=cat_feats),
                Pool(X_va, y_va, cat_features=cat_feats),
            )
        )

    def objective(trial):
        # 2) Only valid constructor params here:
        params = {
            "loss_function": "MAPE", "eval_metric": "MAPE",
              "iterations": trial.suggest_int("iterations", 300, 1000),
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
              "depth": trial.suggest_int("depth", 4, 10), "border_count": trial.suggest_int("border_count", 32, 64),
              "random_strength": trial.suggest_float("random_strength", 0, 10),
              "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True), "bootstrap_type": "Bayesian",
              "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10), "boosting_type": "Plain",
              "task_type": "CPU", "thread_count": -1, "random_seed": random_state, "verbose": False,
              "grow_policy": trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
              )
        }
        if params["grow_policy"] == "Lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 31, 256)

        pruning_cb = CatBoostPruningCallback(trial, "MAPE")
        mapes = []
        # 3) Loop folds, pass early_stopping & pruning into fit()
        for train_pool, val_pool in cv_pools:
            model = CatBoostRegressor(**params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=100,
                callbacks=[pruning_cb],
                verbose=False,
            )
            pruning_cb.check_pruned()
            mapes.append(mean_absolute_percentage_error(val_pool.get_label(), model.predict(val_pool)))

        return sum(mapes) / len(mapes)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15,
            n_warmup_steps=100,
            interval_steps=10)
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

        train_data = xgb.DMatrix(X_train, label=y_train)
        val_data = xgb.DMatrix(X_val, label=y_val)

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
        y_pred = model.predict(X_val)
        mape_scores.append(mean_absolute_percentage_error(y_val, y_pred))

    return np.mean(mape_scores)


def _lightgbm_rolling_origin_cv(X, y, params, n_splits=5, random_state=42):
    """Performs rolling-origin cross-validation for LightGBM model evaluation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        params (dict): LightGBM hyperparameters.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.

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

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params["verbosity"] = -1

        num_boost_round = 1000
        if "num_iterations" in params:
            num_boost_round = params.pop("num_iterations")

        # Train LightGBM
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=5, verbose=False
                ),  # Early stopping after 50 rounds
                lgb.log_evaluation(period=0),  # Disable evaluation logs
            ],
        )

        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        mape_scores.append(mean_absolute_percentage_error(y_val, y_pred))

    return np.mean(mape_scores)
