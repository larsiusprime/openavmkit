import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def _rolling_origin_cv(X, y, params, n_splits=5, random_state=42):
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
    warnings.filterwarnings("ignore", category=UserWarning)

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

        warnings.filterwarnings("ignore", category=UserWarning)
        # Train LightGBM
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=5, verbose=False),  # Early stopping after 50 rounds
                lgb.log_evaluation(period=0)  # Disable evaluation logs
            ]
        )

        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        mae_scores.append(mean_absolute_error(y_val, y_pred))

    # restore warnings to default:
    #warnings.filterwarnings("default")
    return np.mean(mae_scores)


def tune_lightgbm(X, y, n_trials=100, n_splits=5, random_state=42):
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
        float: Best MAE score achieved.
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
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.1),
            "max_bin": trial.suggest_int("max_bin", 64, 1024),
            "num_leaves": trial.suggest_int("num_leaves", 64, 2048),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_gain_to_split": trial.suggest_loguniform("min_gain_to_split", 1e-4, 50),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 0.9),
            "subsample": trial.suggest_uniform("subsample", 0.5, 0.8),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.1, 10),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 0.1, 10),
            "cat_smooth": trial.suggest_int("cat_smooth", 5, 200),
            "verbosity": -1
        }

        # Use rolling-origin cross-validation
        mae = _rolling_origin_cv(X, y, params, n_splits=n_splits, random_state=random_state)
        return mae  # Optuna minimizes, so return the MAE directly

    # Run Bayesian Optimization with Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Return the best parameters and best MAE score
    return study.best_params
