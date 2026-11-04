"""The cloud SDKs (Azure, Hugging Face, SFTP) and NGBoost are optional extras: importing OpenAVMKit's pipeline and
models doesn't load them, and using one that isn't installed says how to install it."""

import subprocess
import sys

import pytest

OPTIONAL = ("azure", "huggingface_hub", "paramiko", "ngboost")


def test_importing_the_pipeline_and_models_loads_none_of_them():
    code = (
        "import sys, openavmkit.pipeline, openavmkit.modeling, openavmkit.tuning, openavmkit.cloud.cloud; "
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
