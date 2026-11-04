"""Optional dependencies: imported only where they're used, with an error that says how to install them.

OpenAVMKit's cloud storage (Azure, Hugging Face, SFTP) and its NGBoost model need packages that most uses never
touch, so they are extras rather than requirements:

    pip install "openavmkit[cloud]"     # cloud_sync: Azure, Hugging Face, SFTP
    pip install "openavmkit[ngboost]"   # the "ngboost" model
    pip install "openavmkit[xgboost]"   # the "xgboost" model (and the xgboost/ensemble fill-in models)
    pip install "openavmkit[catboost]"  # the "catboost" model
"""

import importlib
import importlib.util
from types import ModuleType


def require(module: str, extra: str, what: str) -> ModuleType:
    """Import ``module``, or raise an ImportError that names the extra that installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        package = module.split(".")[0]
        raise ImportError(
            f"{what} needs the optional package '{package}', which isn't installed. "
            f'Install it with: pip install "openavmkit[{extra}]"'
        ) from e


def ngboost_classes():
    """NGBoost's regressor and its Normal distribution (the "ngboost" extra)."""
    ngboost = require("ngboost", "ngboost", "The NGBoost model")
    distns = require("ngboost.distns", "ngboost", "The NGBoost model")
    return ngboost.NGBRegressor, distns.Normal


def installed(module: str) -> bool:
    """Whether an optional package can be imported, without importing it."""
    return importlib.util.find_spec(module) is not None


def xgboost_module() -> ModuleType:
    """XGBoost (the "xgboost" extra)."""
    return require("xgboost", "xgboost", "The XGBoost model")


def catboost_module() -> ModuleType:
    """CatBoost (the "catboost" extra)."""
    return require("catboost", "catboost", "The CatBoost model")
