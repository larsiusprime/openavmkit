"""Optional dependencies: imported only where they're used, with an error that says how to install them.

OpenAVMKit's cloud storage (Azure, Hugging Face, SFTP) and its NGBoost model need packages that most uses never
touch, so they are extras rather than requirements:

    pip install "openavmkit[cloud]"     # cloud_sync: Azure, Hugging Face, SFTP
    pip install "openavmkit[ngboost]"   # the "ngboost" model
"""

import importlib
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
