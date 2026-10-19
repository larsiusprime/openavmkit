"""Run the OpenAVMKit pipeline stages without Jupyter.

The notebooks in ``notebooks/pipeline`` are the reference workflow. This module
runs the same sequence of :mod:`openavmkit.pipeline` calls from plain Python, so
scripts, CI jobs and desktop front ends can drive a locality end to end and
receive structured progress (see :mod:`openavmkit.progress`).

Stages mirror the notebooks:

* ``assemble`` -- 01-assemble: load, merge, enrich, tag model groups.
* ``clean`` -- 02-clean: fills, equity clusters, sales processing, scrutiny.
* ``model`` -- 03-model: splits, spatial lag, area stats, try/finalize models,
  ratio study report.

Like the notebooks, a stage works inside its locality folder (``in/``, ``out/``,
``cache/``). :func:`run_stage` changes into that folder for the duration of the
call and restores the previous working directory afterwards; it is meant to run
one stage at a time per process.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from openavmkit import progress
from openavmkit.progress import CancelCheck, ProgressCallback

STAGES = ("assemble", "clean", "model")


@contextmanager
def _in_locality(project_dir: Path) -> Iterator[None]:
    previous = os.getcwd()
    os.makedirs(project_dir / "in", exist_ok=True)
    os.makedirs(project_dir / "out", exist_ok=True)
    os.chdir(project_dir)
    try:
        yield
    finally:
        os.chdir(previous)


def run_stage(
    project_dir: str | os.PathLike,
    stage: str,
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelCheck] = None,
    *,
    verbose: bool = False,
    run_vacant: bool = True,
    drop_scrutiny_outliers: bool = True,
    do_shaps: bool = False,
    use_saved_params: bool = True,
) -> None:
    """Run one pipeline stage for a locality.

    Parameters
    ----------
    project_dir : path-like
        The locality folder (the one containing ``in/`` and ``out/``).
    stage : str
        One of ``"assemble"``, ``"clean"`` or ``"model"``.
    on_progress : callable, optional
        Receives a :class:`openavmkit.progress.ProgressEvent` at each step.
    cancel : callable, optional
        Returns True to request cancellation; the stage then raises
        :class:`openavmkit.progress.Cancelled` at the next step boundary.
    verbose : bool, optional
        Passed through to the pipeline functions. Defaults to False.
    run_vacant : bool, optional
        Whether ``finalize_models`` also runs vacant-only models. Defaults to True,
        matching 03-model.
    drop_scrutiny_outliers : bool, optional
        Whether sales scrutiny drops flagged sales (cluster and heuristic outliers).
        Defaults to True, matching 02-clean.
    do_shaps : bool, optional
        Whether ``try_models`` computes SHAP values. Defaults to False.
    use_saved_params : bool, optional
        Whether ``finalize_models`` reuses saved tuned hyperparameters.
        Defaults to True, matching 03-model.

    Raises
    ------
    ValueError
        If ``stage`` is not a known stage.
    openavmkit.progress.Cancelled
        If ``cancel`` returned True during the run.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of: {', '.join(STAGES)}")
    project_dir = Path(project_dir).resolve()
    with progress.reporting(on_progress, cancel, stage), _in_locality(project_dir):
        if stage == "assemble":
            _assemble(verbose)
        elif stage == "clean":
            _clean(verbose, drop_scrutiny_outliers)
        else:
            _model(verbose, run_vacant, do_shaps, use_saved_params)


def _assemble(verbose: bool) -> None:
    from openavmkit import pipeline as p

    p.delete_checkpoints("1-assemble")
    progress.report("load_dataframes", 0.0, "Loading data files")
    dataframes = p.load_dataframes(settings=p.load_settings(), verbose=verbose)
    progress.report("process_dataframes", 0.15, "Merging and enriching data")
    sup = p.process_dataframes(dataframes=dataframes, settings=p.load_settings(), verbose=verbose)
    progress.report("enrich_streets", 0.8, "Street frontage (if enabled)")
    sup = p.enrich_sup_streets(sup=sup, settings=p.load_settings(), verbose=verbose)
    progress.report("tag_model_groups", 0.9, "Assigning model groups")
    sup = p.tag_model_groups_sup(sup=sup, settings=p.load_settings(), verbose=verbose)
    progress.report("write_output", 0.95, "Saving results")
    p.write_notebook_output_sup(sup, "1-assemble", parquet=True, gpkg=False, shp=False)
    progress.report("done", 1.0, f"{len(sup.universe):,} parcels, {len(sup.sales):,} sales")


def _clean(verbose: bool, drop_scrutiny_outliers: bool) -> None:
    from openavmkit import pipeline as p

    p.delete_checkpoints("2-clean")
    progress.report("load", 0.0, "Loading assembled data")
    sup = p.read_pickle("out/1-assemble-sup")
    progress.report("fill", 0.05, "Filling blanks")
    sup = p.fill_unknown_values_sup(sup, p.load_settings())
    progress.report("horizontal_equity_clusters", 0.1, "Grouping similar parcels")
    sup = p.mark_horizontal_equity_clusters_per_model_group_sup(
        sup=sup, settings=p.load_settings(), verbose=verbose, do_land_clusters=True, do_impr_clusters=True
    )
    progress.report("process_sales", 0.45, "Checking sale validity and time trends")
    sup = p.process_sales(sup=sup, settings=p.load_settings(), verbose=verbose)
    progress.report("sales_scrutiny", 0.55, "Scrutinizing sales")
    sup = p.run_sales_scrutiny(
        sup=sup,
        settings=p.load_settings(),
        drop_cluster_outliers=drop_scrutiny_outliers,
        drop_heuristic_outliers=drop_scrutiny_outliers,
        verbose=verbose,
    )
    progress.report("collapse_sparse_categories", 0.9, "Tidying rare categories")
    sup = p.collapse_sparse_categories_sup(sup, p.load_settings())
    progress.report("write_output", 0.95, "Saving results")
    p.write_notebook_output_sup(sup, "2-clean")
    progress.report("done", 1.0, f"{len(sup.sales):,} sales after cleaning")


def _model(verbose: bool, run_vacant: bool, do_shaps: bool, use_saved_params: bool) -> None:
    from openavmkit import pipeline as p

    p.delete_checkpoints("3-model")
    settings = p.load_settings()
    progress.report("load", 0.0, "Loading cleaned data")
    sup = p.load_cleaned_data_for_modeling(settings)
    progress.report("splits", 0.02, "Splitting sales into training and test sets")
    p.write_canonical_splits(sup, settings, verbose=verbose)
    progress.report("spatial_lag", 0.04, "Computing nearby-sales features")
    sup = p.enrich_sup_spatial_lag(sup=sup, settings=settings, verbose=verbose)
    progress.report("area_stats", 0.08, "Computing neighbourhood statistics")
    sup = p.enrich_sup_area_stats(sup=sup, settings=settings, verbose=verbose)
    p.write_parquet(sup.universe, "out/look/3-spatial-lag-universe.parquet")
    p.write_parquet(sup.sales, "out/look/3-spatial-lag-sales.parquet")
    progress.report("try_models", 0.1, "Trying models")
    p.try_models(
        sup=sup, settings=settings, save_params=True, verbose=verbose, run_main=True,
        run_vacant=False, run_ensemble=True, do_shaps=do_shaps, do_plots=False,
    )
    progress.report("finalize_models", 0.5, "Finalizing models")
    p.finalize_models(
        sup=sup, settings=settings, save_params=True, use_saved_params=use_saved_params,
        verbose=verbose, run_main=True, run_vacant=run_vacant,
    )
    progress.report("ratio_study_report", 0.95, "Writing ratio study reports")
    p.run_and_write_ratio_study_breakdowns(settings)
    progress.report("done", 1.0, "Models finalized")
