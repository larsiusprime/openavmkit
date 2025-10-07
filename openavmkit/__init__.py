# Optional convenience: expose the installed version at runtime
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("openavmkit")
except Exception:
    __version__ = "0+unknown"