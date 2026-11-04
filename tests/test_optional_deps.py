"""The cloud SDKs (Azure, Hugging Face, SFTP), NGBoost, XGBoost and CatBoost are optional extras: importing OpenAVMKit's pipeline and
models doesn't load them, and using one that isn't installed says how to install it."""

import subprocess
import sys

import pytest

OPTIONAL = ("azure", "huggingface_hub", "paramiko", "ngboost", "xgboost", "catboost")


def test_importing_the_pipeline_and_models_loads_none_of_them():
    code = (
        "import sys, openavmkit.pipeline, openavmkit.modeling, openavmkit.tuning, openavmkit.cloud.cloud, "
        "openavmkit.model_runner, openavmkit.inference, openavmkit.shap_analysis; "
        f"print([m for m in {OPTIONAL!r} if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_ngboost_missing_says_how_to_install(monkeypatch):
    from openavmkit.utilities.optional import ngboost_classes

    monkeypatch.setitem(sys.modules, "ngboost", None)  # as if not installed
    with pytest.raises(ImportError, match=r'pip install "openavmkit\[ngboost\]"'):
        ngboost_classes()


@pytest.mark.parametrize(
    "cloud_type, sdk", [("azure", "azure.storage.blob"), ("huggingface", "huggingface_hub"), ("sftp", "paramiko")]
)
def test_a_cloud_sdk_missing_says_how_to_install(monkeypatch, cloud_type, sdk):
    from openavmkit.cloud import cloud

    monkeypatch.setitem(sys.modules, sdk, None)
    with pytest.raises(ImportError, match=r'pip install "openavmkit\[cloud\]"'):
        cloud._backend(cloud_type)


def test_installed_ones_still_load():
    from openavmkit.cloud import cloud
    from openavmkit.utilities.optional import ngboost_classes

    regressor, normal = ngboost_classes()
    assert regressor.__name__ == "NGBRegressor" and normal.__name__ == "Normal"
    for cloud_type in ("azure", "huggingface", "sftp"):
        assert cloud._backend(cloud_type).__name__ == f"openavmkit.cloud.{cloud_type}"
    with pytest.raises(ValueError, match="Unsupported cloud type"):
        cloud._backend("dropbox")


@pytest.mark.parametrize(
    "module, getter, extra",
    [("xgboost", "xgboost_module", "xgboost"), ("catboost", "catboost_module", "catboost")],
)
def test_xgboost_or_catboost_missing_says_how_to_install(monkeypatch, module, getter, extra):
    from openavmkit.utilities import optional

    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=rf'pip install "openavmkit\[{extra}\]"'):
        getattr(optional, getter)()


def test_inference_without_xgboost_tries_the_other_models(monkeypatch, capsys):
    from openavmkit import inference

    assert inference._experiment_models() == ["random_forest", "lightgbm", "xgboost", "ensemble"]
    monkeypatch.setattr(inference, "installed", lambda m: m != "xgboost")
    assert inference._experiment_models() == ["random_forest", "lightgbm"]
    assert 'openavmkit[xgboost]' in capsys.readouterr().out
    # Asking for it by name says how to install it.
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ImportError, match=r'openavmkit\[xgboost\]'):
        inference._get_inference_model("xgboost")


def test_engines_to_run_reads_settings_as_the_run_does():
    from openavmkit.model_runner import _engines_to_run

    settings = {
        "modeling": {
            "instructions": {
                "main": {"run": ["lightgbm", "trees", "cat"], "skip": {"condo": ["cat"]}},
                "vacant": {"skip": {"sf": ["all"]}},
            },
            "models": {
                "main": {"default": {}, "trees": {"model": "xgboost"}, "cat": {"model": "catboost"}},
                # vacant: no "run" list, so every model it has an entry for
                "vacant": {"condo": {"ngboost": {}, "mra": {}}},
            },
        }
    }
    assert _engines_to_run(settings, ["condo"], True, False) == {"lightgbm", "xgboost"}
    assert _engines_to_run(settings, ["sf"], True, False) == {"lightgbm", "xgboost", "catboost"}
    assert _engines_to_run(settings, ["condo"], False, True) == {"ngboost", "mra"}
    assert _engines_to_run(settings, ["sf"], False, True) == set()  # skips "all"


def test_a_run_stops_before_training_when_an_engine_is_missing(monkeypatch):
    """The run names every missing engine up front, and trains nothing: not the models listed before them."""
    import pandas as pd

    import openavmkit.model_runner as mr
    from openavmkit.utilities import optional

    trained = []
    monkeypatch.setattr(mr, "_run_models", lambda *a, **k: trained.append(a[1]))
    monkeypatch.setattr(optional, "installed", lambda m: m not in ("xgboost", "catboost"))
    settings = {
        "modeling": {
            "instructions": {"model_groups": ["sf"], "main": {"run": ["lightgbm", "xgboost", "cat"]}},
            "models": {"main": {"default": {}, "cat": {"model": "catboost"}}},
        }
    }
    sup = {"universe": pd.DataFrame({"key": ["1"]})}
    with pytest.raises(ImportError, match=r'pip install "openavmkit\[catboost,xgboost\]"'):
        mr.run_models(sup, settings, run_vacant=False)
    assert trained == []

    monkeypatch.setattr(optional, "installed", lambda m: True)
    mr.run_models(sup, settings, run_vacant=False)
    assert trained == ["sf"]
