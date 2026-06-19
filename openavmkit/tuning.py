"""
Hyperparameter tuning for tree-based models.

Uses Optuna to search hyperparameter spaces for XGBoost, LightGBM, and
CatBoost models with k-fold cross-validation, optimizing for mean absolute
percentage error. Called from :mod:`openavmkit.modeling` when a model entry
requests tuning rather than fixed parameters.

All public APIs are private (underscore-prefixed) — tuning is invoked
indirectly through model setup, not as a user-facing operation.
"""
import glob
import hashlib
import os

import xgboost as xgb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import Pool, CatBoostRegressor, cv
from ngboost import NGBRegressor
from ngboost.distns import Normal

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from optuna.integration import CatBoostPruningCallback

from openavmkit.utilities.modeling import TreeBasedCategoricalData

#######################################
# PRIVATE
#######################################

# Trials per generation in the batched (deterministic-parallel) XGBoost/LightGBM search.
# Fixed (not core count) so the TPE adaptivity cadence — and thus the result — is
# machine-independent; only how many of a batch run concurrently scales with cores.
_TUNING_BATCH_SIZE = 8

# Bump this whenever ANY tuner's search space changes (param ranges, bounds, added/removed
# hyperparameters). It is folded into _study_fingerprint, so a bump invalidates both resume
# journals and saved params.json files — forcing a re-tune against the new space instead of
# silently reusing params/trials scored against the old one. v2 = LightGBM/XGBoost lr-floor
# raised to 0.01 + iteration/leaf caps tightened (2026-06-18).
_SEARCH_SPACE_VERSION = 2


def _resumable_study(
    direction, study_name=None, storage_path=None, pruner=None, sampler=None, verbose=False
):
    """Create an Optuna study, optionally backed by a persistent journal file.

    When ``storage_path`` is ``None`` this returns a plain in-memory study, exactly
    matching the historical behavior. When a path is given, the study is backed by a
    :class:`optuna.storages.JournalStorage` so trials persist to disk as they complete;
    ``load_if_exists=True`` means an interrupted run reattaches to whatever trials are
    already on disk instead of starting over.

    The journal (file) backend is used rather than SQLite because the XGBoost/LightGBM
    tuners run trials with ``n_jobs=-1``; the journal backend is built for concurrent
    access, whereas SQLite raises "database is locked" under parallel writes.

    Parameters
    ----------
    direction : str
        Optimization direction passed to :func:`optuna.create_study`.
    study_name : str, optional
        Stable study name; required for resume to find the prior study in the journal.
    storage_path : str, optional
        Path to the journal file. ``None`` (default) keeps the study in memory.
    pruner : optuna.pruners.BasePruner, optional
        Pruner to attach (CatBoost passes a ``MedianPruner``; others pass ``None``).
    sampler : optuna.samplers.BaseSampler, optional
        Sampler to attach. Pass a seeded sampler (e.g. ``TPESampler(seed=...)``) for a
        reproducible search; ``None`` uses Optuna's default entropy-seeded sampler.
    verbose : bool, optional
        If True, print how many trials were resumed from disk.

    Returns
    -------
    optuna.study.Study
    """
    if storage_path is None:
        return optuna.create_study(direction=direction, pruner=pruner, sampler=sampler)

    # Use the open()-based lock rather than Optuna's default symlink lock: on Windows
    # os.symlink requires elevated privilege (WinError 1314), so the symlink lock fails
    # for ordinary users. The open lock is cross-platform.
    journal = optuna.storages.journal
    storage = optuna.storages.JournalStorage(
        journal.JournalFileBackend(
            storage_path, lock_obj=journal.JournalFileOpenLock(storage_path)
        )
    )
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )
    if verbose and len(study.trials):
        print(
            f"--> resuming study '{study_name}': {len(study.trials)} trial(s) already on disk"
        )
    return study


def _seeded_sampler(random_state, constant_liar=False):
    """A seeded TPE sampler when ``random_state`` is set, else ``None`` (default sampler).

    Optuna's default sampler is seeded from OS entropy, so the hyperparameter search path
    differs every run even on identical data. Seeding it is the main lever for making the
    tree-based models reproducible.

    ``constant_liar=True`` is used by the batched XGBoost/LightGBM tuners: it accounts for
    trials that have been asked but not yet told (the in-flight batch), so the parallel
    proposals within a generation spread out instead of clustering.
    """
    if random_state is None:
        return None
    return optuna.samplers.TPESampler(seed=random_state, constant_liar=constant_liar)


def _is_plateaued(study, plateau_trials=10, improvement_threshold=0.01):
    """True if the best value hasn't improved by >= ``improvement_threshold`` over the
    last ``plateau_trials`` completed trials. Generation-boundary equivalent of
    :func:`_plateau_callback`, used by :func:`_run_batched`.
    """
    completed = [t for t in study.trials if t.value is not None]
    if len(completed) < plateau_trials:
        return False
    best_value = study.best_trial.value
    if best_value is None:
        return False
    recent = completed[-plateau_trials:]
    return all(t.value >= best_value * (1 + improvement_threshold) for t in recent)


def _run_batched(study, suggest, evaluate, n_trials, storage_path, verbose, label="", batch_size=None):
    """Deterministic *and* parallel hyperparameter search via synchronous batched ask-and-tell.

    The non-determinism in plain ``study.optimize(n_jobs>1)`` comes from telling trial
    results back in *completion* order, which races. This avoids that:

    1. **Ask a batch sequentially** (main thread): ``suggest(trial)`` samples each trial's
       params in a deterministic order, so the proposed params are a pure function of
       ``(seed, told-set)`` — no RNG race. (Sampling must NOT happen inside the parallel
       step.)
    2. **Evaluate the batch in parallel**: ``evaluate`` runs the (single-threaded, hence
       deterministic) CV for each trial concurrently; completion order is irrelevant
       because each evaluation is independent.
    3. **Tell in ask order**, so the study's state after each generation is identical
       run-to-run regardless of who finished first.

    Net: reproducible results with up to ``batch_size``-way parallelism. ``batch_size`` is
    a fixed constant (not core count) so the TPE adaptivity cadence — and therefore the
    result — does not depend on the machine; only how many evaluations run at once does.

    Resumes cleanly: only ``COMPLETE`` trials count toward the target, so a journal-backed
    study continues from where it left off.
    """
    from concurrent.futures import ThreadPoolExecutor
    from optuna.trial import TrialState

    if batch_size is None:
        batch_size = _TUNING_BATCH_SIZE
    workers = max(1, min(batch_size, (os.cpu_count() or 2) - 2))

    n_done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    while n_done < n_trials:
        b = min(batch_size, n_trials - n_done)
        asked = []
        for _ in range(b):
            trial = study.ask()
            asked.append((trial, suggest(trial)))  # sequential -> deterministic sampling

        with ThreadPoolExecutor(max_workers=workers) as ex:
            values = list(ex.map(lambda item: evaluate(item[1]), asked))

        for (trial, _), value in zip(asked, values):  # tell in ask order, not finish order
            study.tell(trial, value)
        n_done += b

        if verbose:
            print(f"-->{label} tuning: {n_done}/{n_trials} trials (best MAPE {study.best_value:0.4f})")
        if _is_plateaued(study):
            if verbose:
                print(f"Plateau detected: stopping {label} search early at {n_done} trials.")
            break


def _remaining_trials(study, n_trials, storage_path):
    """Number of trials still needed to reach the ``n_trials`` target.

    In-memory studies (``storage_path is None``) always run the full ``n_trials``.
    Resumable studies subtract the trials already persisted on disk so the *total*
    across runs converges to ``n_trials`` rather than running ``n_trials`` afresh each
    time the run is restarted.
    """
    if storage_path is None:
        return n_trials
    return max(0, n_trials - len(study.trials))


def _study_fingerprint(columns, n_rows, n_trials, seed=None):
    """Short stable hash identifying a tuning study's search context.

    A resumed study (or reused ``params.json``) is only valid if it is searching the *same*
    objective: same feature set, same number of training rows, same trial budget, same seed,
    and same search-space version (``_SEARCH_SPACE_VERSION``). Baking this fingerprint into the
    journal filename / saved params means they are reused only on an exact-context match; a
    changed ``ind_vars`` list, sales window, seed, or tuner search space yields a different
    fingerprint (stale artifacts are discarded) rather than silently mixing trials/params
    scored against a different objective.
    """
    key = (
        "|".join(sorted(str(c) for c in columns))
        + f"||rows={n_rows}||trials={n_trials}||seed={seed}||space=v{_SEARCH_SPACE_VERSION}"
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def _discard_stale_studies(outpath, slug, keep, verbose=False):
    """Delete any ``{slug}_study_*.journal`` files whose fingerprint != ``keep``.

    Removes journals (and their lock sidecars) left over from a prior tuning run whose
    search context no longer matches, so a stale study is never resumed.
    """
    keep_name = f"{slug}_study_{keep}.journal"
    for path in glob.glob(f"{outpath}/{slug}_study_*.journal*"):
        # Keep the matching journal and its lock sidecars; drop everything else.
        if not os.path.basename(path).startswith(keep_name):
            try:
                os.remove(path)
                if verbose:
                    print(f"--> discarded stale tuning journal: {path}")
            except OSError:
                pass


def _cleanup_study_files(storage_path):
    """Remove a journal file and any lock sidecars; tolerate already-gone files."""
    if storage_path is None:
        return
    for path in glob.glob(f"{storage_path}*"):
        try:
            os.remove(path)
        except OSError:
            pass


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
    storage_path=None,
    study_name=None,
):
    """Tunes XGBoost hyperparameters using Optuna and shuffled k-fold cross-validation.
    Uses the xgboost.train API for training. Includes logging for progress monitoring.

    When ``storage_path`` is set the study is journal-backed so an interrupted run
    resumes from the trials already on disk (see :func:`_resumable_study`).
    """

    # Split into a sequential `suggest` (samples params from the trial) and a parallel
    # `evaluate` (runs CV). The suggest step MUST run sequentially in the ask-phase so the
    # sampler RNG is advanced in a deterministic order; only the CV evaluation is
    # parallelized across trials. nthread=1 keeps each fit deterministic and avoids
    # oversubscription when many trials evaluate concurrently. See `_run_batched`.
    def suggest(trial):
        params = {
            "objective": "reg:squarederror",  # Regression objective
            "eval_metric": "mape",  # Mean Absolute Percentage Error
            "tree_method": "hist",  # Use 'hist' for performance; use 'gpu_hist' for GPUs
            "enable_categorical": True,
            "max_cat_to_onehot": 1,
            "nthread": 1,
            "seed": random_state if random_state is not None else 0,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
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
        num_boost_round = trial.suggest_int("num_boost_round", 100, 1500)
        return params, num_boost_round

    def evaluate(suggested):
        params, num_boost_round = suggested
        return _xgb_kfold_cv(
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _resumable_study(
        "minimize",
        study_name=study_name,
        storage_path=storage_path,
        sampler=_seeded_sampler(random_state, constant_liar=True),
        verbose=verbose,
    )
    _run_batched(study, suggest, evaluate, n_trials, storage_path, verbose, label="XGBoost")
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
    storage_path=None,
    study_name=None,
):
    """Tunes LightGBM hyperparameters using Optuna and shuffled k-fold cross-validation.

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

    # Bound search space by training-fold size to prevent memorisation on thin datasets.
    # Each CV fold trains on roughly (n_splits-1)/n_splits of the data.
    n_train_per_fold = int(len(X) * (n_splits - 1) / n_splits)
    # num_leaves: cap at min(256, n_train_per_fold // 8). 256 leaves is already very expressive
    # for a few dozen features; the old n//4 cap (e.g. ~1371 leaves on 5.5k rows) bloated per-round
    # cost with no accuracy benefit and overfit thin folds. //8 keeps ~8+ samples/leaf.
    max_num_leaves = max(8, min(256, n_train_per_fold // 8))
    # min_data_in_leaf: upper bound must not exceed training fold size or every split is illegal.
    max_min_data_in_leaf = max(2, min(500, n_train_per_fold // 4))
    if verbose and max_num_leaves < 64:
        print(
            f"  [tune_lightgbm] thin dataset (n_train_per_fold={n_train_per_fold}): "
            f"num_leaves capped at {max_num_leaves}, min_data_in_leaf capped at {max_min_data_in_leaf}"
        )

    # Sequential `suggest` (deterministic sampling) + parallel `evaluate` (CV). The
    # deterministic/force_row_wise/num_threads=1 trio makes each fit bit-reproducible and
    # keeps concurrent trials from oversubscribing cores. See `_run_batched`.
    def suggest(trial):
        return {
            "objective": "regression",
            "metric": "mape",
            "boosting_type": "gbdt",
            "num_threads": 1,
            "deterministic": True,
            "force_row_wise": True,
            "seed": random_state if random_state is not None else 0,
            "num_iterations": trial.suggest_int("num_iterations", 300, 1500),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "max_bin": trial.suggest_int("max_bin", 64, 1024),
            "num_leaves": trial.suggest_int("num_leaves", min(64, max_num_leaves), max_num_leaves),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split", 1e-4, 50, log=True
            ),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", min(20, max_min_data_in_leaf), max_min_data_in_leaf),
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

    def evaluate(params):
        return _lightgbm_kfold_cv(
            X, y, params, n_splits=n_splits, random_state=random_state, cat_vars=cat_vars
        )

    # Run Bayesian Optimization with Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _resumable_study(
        "minimize",
        study_name=study_name,
        storage_path=storage_path,
        pruner=optuna.pruners.MedianPruner(),
        sampler=_seeded_sampler(random_state, constant_liar=True),
        verbose=verbose,
    )
    _run_batched(study, suggest, evaluate, n_trials, storage_path, verbose, label="LightGBM")

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
    use_gpu=True,
    storage_path=None,
    study_name=None,
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
    study = _resumable_study(
        "minimize",
        study_name=study_name,
        storage_path=storage_path,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=15, n_warmup_steps=100, interval_steps=10
        ),
        sampler=_seeded_sampler(random_state),
        verbose=verbose,
    )

    remaining = _remaining_trials(study, n_trials, storage_path)
    if remaining > 0:
        # Study-level plateau stop, consistent with NGBoost (and the batched _is_plateaued check
        # in LightGBM/XGBoost): bail out of the trial budget once the best value stops improving
        # rather than grinding all n_trials x n_splits folds. CatBoost uses serial optimize, so the
        # callback applies cleanly. (The MedianPruner above can't save time here — the objective
        # reports its curve only after CatBoost's built-in cv() has already run all folds.)
        study.optimize(objective, n_trials=remaining, n_jobs=1, callbacks=[_plateau_callback])

    if verbose:
        print(
            f"Best trial #{study.best_trial.number} -> MAPE={study.best_trial.value:.4f}"
        )
        print("Params:", study.best_trial.params)

    return study.best_params


def _tune_ngboost(
    X,
    y,
    sizes,
    he_ids,
    n_trials=50,
    n_splits=5,
    random_state=42,
    cat_vars=None,
    verbose=False,
    storage_path=None,
    study_name=None,
):
    """Tunes NGBoost hyperparameters using Optuna and k-fold cross-validation.

    NGBoost is a probabilistic gradient booster; here we tune for point-estimate
    accuracy (MAPE on the predicted mean), consistent with the other tree tuners.
    Its natural-gradient boosting is markedly slower than XGBoost/LightGBM, so
    callers should keep ``n_trials`` small.

    Args:
        X (pd.DataFrame): Feature matrix (categoricals as 'category' dtype).
        y (array-like): Target vector.
        sizes (array-like): Array of size values (land or building size).
        he_ids (array-like): Array of horizontal equity cluster ID's.
        n_trials (int): Number of optimization trials for Optuna. Default is 50.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.
        cat_vars (list): Categorical feature names to encode numerically.
        verbose (bool): Whether to print Optuna progress.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    y = np.asarray(y, dtype=np.float64)

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        minibatch_frac = trial.suggest_float("minibatch_frac", 0.5, 1.0)
        max_depth = trial.suggest_int("max_depth", 3, 8)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        mape_scores = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # cat_data is fit on the training fold so unseen val categories map to NaN
            cat_data = TreeBasedCategoricalData.from_training_data(
                X_tr, categorical_cols=[c for c in (cat_vars or []) if c in X_tr.columns]
            )
            Xn_tr = cat_data.to_numeric_matrix(X_tr)
            Xn_val = cat_data.to_numeric_matrix(X_val)

            base = DecisionTreeRegressor(
                max_depth=max_depth, criterion="friedman_mse", random_state=random_state
            )
            model = NGBRegressor(
                Dist=Normal,
                Base=base,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                minibatch_frac=minibatch_frac,
                random_state=random_state,
                verbose=False,
            )
            model.fit(Xn_tr, y_tr)
            preds = model.predict(Xn_val)
            mape_scores.append(mean_absolute_percentage_error(y_val, preds))

        mape = float(np.mean(mape_scores))
        if verbose:
            print(f"-->trial # {trial.number}/{n_trials}, MAPE: {mape:0.4f}")
        return mape

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _resumable_study(
        "minimize",
        study_name=study_name,
        storage_path=storage_path,
        sampler=_seeded_sampler(random_state),
        verbose=verbose,
    )
    # NGBoost training is not thread-safe enough for n_jobs=-1; tune serially.
    remaining = _remaining_trials(study, n_trials, storage_path)
    if remaining > 0:
        study.optimize(
            objective, n_trials=remaining, n_jobs=1, callbacks=[_plateau_callback]
        )

    if verbose:
        print(
            f"Best trial: {study.best_trial.number} with MAPE: {study.best_trial.value:0.4f} "
            f"and params: {study.best_trial.params}"
        )
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


def _xgb_kfold_cv(
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
    """Shuffled (random) K-fold CV for XGBoost hyperparameter selection.

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


def _catboost_kfold_cv(
    X, y, params, n_splits=5, random_state=42, cat_vars=None, verbose=False
):
    """Shuffled (random) K-fold CV for CatBoost. CURRENTLY UNUSED.

    The live CatBoost tuner (`_tune_catboost`) uses CatBoost's built-in `cv()`; this
    helper is kept for reference/parity with the XGBoost/LightGBM paths.

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


def _lightgbm_kfold_cv(X, y, params, n_splits=5, random_state=42, cat_vars=None):
    """Shuffled (random) K-fold CV for LightGBM hyperparameter selection.
    """
    n_samples = len(X)
    n_splits = min(n_splits, n_samples)
    if n_splits < 2:
        import warnings
        warnings.warn(
            f"Not enough samples ({n_samples}) for cross-validation with n_splits={n_splits}. "
            "Returning penalty MAPE of 1.0.",
            UserWarning,
        )
        return 1.0
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

