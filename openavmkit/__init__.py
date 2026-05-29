"""
OpenAVMKit — a Python library for real estate mass appraisal.

Provides modules for data assembly, cleaning, enrichment, predictive modeling,
ratio studies, equity studies, and report generation. Intended to be driven
either through the bundled Jupyter notebooks (see ``notebooks/`` in the
repository) or imported as a library directly into user code.

The notebook-facing public API lives in :mod:`openavmkit.pipeline`. Library users
may import any module directly.

See Also
--------
openavmkit.pipeline : Public functions called by the bundled notebooks.
openavmkit.data : Core data structures, including ``SalesUniversePair``.
openavmkit.modeling : Predictive modeling engines.
openavmkit.utilities.settings : ``settings.json`` loader and accessors.
"""

# Optional convenience: expose the installed version at runtime
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("openavmkit")
except Exception:
    __version__ = "0+unknown"