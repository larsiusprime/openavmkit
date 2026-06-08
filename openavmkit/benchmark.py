"""DEPRECATED — ``openavmkit.benchmark`` was renamed to ``openavmkit.model_runner``.

The module orchestrates the whole model run (not only the benchmark comparison),
and the old name collided with the research ``benchmark/`` harness. This shim
re-exports everything from :mod:`openavmkit.model_runner` and emits a
``DeprecationWarning`` on import.

==============================================================================
REMOVE THIS SHIM BEFORE THE 0.7.0 RELEASE. (See AGENTS.md §"Pending removals".)
==============================================================================
"""
import warnings

from openavmkit import model_runner as _model_runner
from openavmkit.model_runner import *  # noqa: F401,F403  (re-export public API)

warnings.warn(
    "openavmkit.benchmark has been renamed to openavmkit.model_runner; "
    "update your imports. This compatibility shim will be removed before 0.7.0.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    # Transparently forward attribute access for names that `import *` skips
    # (underscore-prefixed module privates) so existing
    # `from openavmkit.benchmark import _foo` / `openavmkit.benchmark._foo` keep working.
    return getattr(_model_runner, name)
