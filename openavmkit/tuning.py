import logging

import xgboost as xgb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import Pool, CatBoostRegressor
from optuna import Trial
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def tune_xgboost(X, y, n_trials=100, n_splits=5, random_state=42, verbose=False):
    """
    Tunes XGBoost hyperparameters using Optuna and rolling-origin cross-validation.
    Uses the xgboost.train API for training. Includes logging for progress monitoring.
    """

    def objective(trial):
        """
        Objective function for Optuna to optimize XGBoost hyperparameters.
        """
        params = {
            "objective": "reg:squarederror",  # Regression objective
            "eval_metric": "mae",  # Mean Absolute Error
            "tree_method": "hist",  # Use 'hist' for performance; use 'gpu_hist' for GPUs
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, log=False),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0, log=False),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0, log=False),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0, log=False),
            "gamma": trial.suggest_float("gamma", 0.1, 10, log=True),  # min_split_loss
            "lambda": trial.suggest_float("lambda", 1e-4, 10, log=True),  # reg_lambda
            "alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),  # reg_alpha
            "max_bin": trial.suggest_int("max_bin", 64, 512),  # Relevant for 'hist' tree_method
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        }
        num_boost_round = trial.suggest_int("num_boost_round", 100, 3000)

        mae = _xgb_rolling_origin_cv(
            X, y, params, num_boost_round, n_splits, random_state, verbose_eval=False
        )
        if verbose:
            print(f"-->trial # {trial.number}/{n_trials}, MAE: {mae:10.0f}, params: {params}")
        return mae  # Optuna minimizes, so return the MAE directly

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    if verbose:
        print(f"Best trial: {study.best_trial.number} with MAE: {study.best_trial.value:10.0f} and params: {study.best_trial.params}")
    return study.best_params


def tune_lightgbm(X, y, n_trials=100, n_splits=5, random_state=42, verbose=False):
    """
    Tunes LightGBM hyperparameters using Optuna and rolling-origin cross-validation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        n_trials (int): Number of optimization trials for Optuna. Default is 100.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """

    def objective(trial):
        """
        Objective function for Optuna to optimize LightGBM hyperparameters.
        """
        params = {
            "objective": "regression",
            "metric": "mae",  # Mean Absolute Error for regression
            "boosting_type": "gbdt",
            "num_iterations": trial.suggest_int("num_iterations", 300, 5000),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1, log=True),
            "max_bin": trial.suggest_int("max_bin", 64, 1024),
            "num_leaves": trial.suggest_int("num_leaves", 64, 2048),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-4, 50, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9, log=False),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8, log=False),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10, log=True),
            "cat_smooth": trial.suggest_int("cat_smooth", 5, 200),
            "verbosity": -1,
            "early_stopping_round": 50
        }

        # Use rolling-origin cross-validation
        mae = _lightgbm_rolling_origin_cv(X, y, params, n_splits=n_splits, random_state=random_state)
        if verbose:
            print(f"-->trial # {trial.number}/{n_trials}, MAE: {mae:10.0f}, params: {params}")
        return mae  # Optuna minimizes, so return the MAE directly

    # Run Bayesian Optimization with Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # Use parallelism if available

    if verbose:
        print(f"Best trial: {study.best_trial.number} with MAE: {study.best_trial.value:10.0f} and params: {study.best_trial.params}")
    return study.best_params


def tune_catboost(X, y, n_trials=100, n_splits=5, random_state=42, verbose=False):
    """
    Tunes CatBoost hyperparameters using Optuna and rolling-origin cross-validation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        n_trials (int): Number of optimization trials for Optuna. Default is 100.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        dict: Best hyperparameters found by Optuna.
        float: Best MAE score achieved.
    """
    def objective(trial: Trial):
        """
        Objective function for Optuna to optimize CatBoost hyperparameters.
        """
        params = {
            "loss_function": "MAE",  # Mean Absolute Error
            "eval_metric": "MAE",
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "random_strength": trial.suggest_float("random_strength", 0, 10, log=False),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1, log=False),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, log=False),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "random_seed": random_state,
            "verbose": 0  # Suppresses CatBoost output
        }
        if params["grow_policy"] == "Lossguide":
            params["max_leaves"] = trial.suggest_int("max_leaves", 31, 128)

        # Perform rolling-origin cross-validation
        mae = _catboost_rolling_origin_cv(X, y, params, n_splits=n_splits, random_state=random_state)
        if verbose:
            print(f"-->trial # {trial.number}/{n_trials}, MAE: {mae:10.0f}, params: {params}")
        return mae  # Optuna minimizes, so return the MAE directly

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run Bayesian Optimization with Optuna
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # Use parallelism if available

    if verbose:
        print(f"Best trial: {study.best_trial.number} with MAE: {study.best_trial.value:10.0f} and params: {study.best_trial.params}")
    return study.best_params


def tune_krig(X_Train, y_train, verbose=False):
    """
    Tune the hyperparameters for a Kriging model using Optuna.
    :param X_Train: pd.DataFrame of independent variables
    :param y_train: pd.Series of dependent variables
    :param verbose: bool, whether to enable verbose output
    :return: dict: the best hyperparameters found by Optuna
    """
    def objective(trial):
        # Suggest variogram model
        variogram = trial.suggest_categorical("variogram", ["linear", "power", "gaussian", "spherical", "exponential"])

        # Suggest grid size
        grid_size_str = trial.suggest_categorical("grid_size", ["coarse", "medium", "fine"])

        # Use these hyperparameters to run regression kriging
        try:
            # Simulate the process with a sample DataSplit (mock `run_krig` logic here)
            krig_results = run_krig(ds, variogram, grid_size_str, verbose=False)

            # Evaluate using RMSE on validation data (mock example)
            y_pred = krig_results.y_pred_test
            y_true = ds.y_test

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            return rmse
        except Exception as e:
            # Penalize failure cases
            if verbose:
                print(f"Trial failed with error: {e}")
            return float("inf")

    # Create and optimize the study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=verbose)  # Adjust n_trials as needed

    # Return the best parameters
    if verbose:
        print(f"Best hyperparameters: {study.best_params}")
    return study.best_params



#### PRIVATE:


def _xgb_rolling_origin_cv(X, y, params, num_boost_round, n_splits=5, random_state=42, verbose_eval=50):
    """
    Performs rolling-origin cross-validation for XGBoost model evaluation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        params (dict): XGBoost hyperparameters.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.
        verbose_eval (int|bool): Logging interval for XGBoost. Default is 50.

    Returns:
        float: Mean MAE score across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mae_scores = []

    for train_idx, val_idx in kf.split(X):
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        train_data = xgb.DMatrix(X_train, label=y_train)
        val_data = xgb.DMatrix(X_val, label=y_val)

        evals = [(val_data, "validation")]

        # Train XGBoost
        model = xgb.train(
            params=params,
            dtrain=train_data,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=verbose_eval  # Ensure verbose_eval is enabled
        )

        # Predict and evaluate
        y_pred = model.predict(val_data, iteration_range=(0, model.best_iteration))
        mae = mean_absolute_error(y_val, y_pred)
        mae_scores.append(mae)

    mean_mae = np.mean(mae_scores)
    return mean_mae


def _catboost_rolling_origin_cv(X, y, params, n_splits=5, random_state=42, verbose=False):
    """
    Performs rolling-origin cross-validation for CatBoost model evaluation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        params (dict): CatBoost hyperparameters.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.
        verbose (bool): Whether to print CatBoost training logs.

    Returns:
        float: Mean MAE score across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mae_scores = []

    for train_idx, val_idx in kf.split(X):
        # Use .iloc for Pandas DataFrames
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        # Train CatBoost
        model = CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=verbose,
            early_stopping_rounds=50
        )

        # Predict and evaluate
        y_pred = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, y_pred))

    return np.mean(mae_scores)


def _lightgbm_rolling_origin_cv(X, y, params, n_splits=5, random_state=42):
    """
    Performs rolling-origin cross-validation for LightGBM model evaluation.

    Args:
        X (array-like): Feature matrix.
        y (array-like): Target vector.
        params (dict): LightGBM hyperparameters.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        float: Mean MAE score across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mae_scores = []

    for train_idx, val_idx in kf.split(X):
        # Use .iloc for Pandas DataFrames
        if isinstance(X, pd.DataFrame):
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
            num_boost_round = num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=5, verbose=False),  # Early stopping after 50 rounds
                lgb.log_evaluation(period=0)  # Disable evaluation logs
            ]
        )

        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        mae_scores.append(mean_absolute_error(y_val, y_pred))

    return np.mean(mae_scores)
