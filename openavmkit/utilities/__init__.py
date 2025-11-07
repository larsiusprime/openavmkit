"""
The purpose of the utilities folder is to solve issues with circular imports.

Modules in the utilities folder:

- May NOT import each other.
- May NOT import any other module from the openavmkit package.
- MAY import modules NOT from the openavmkit package.

Functions that should be accessed from all utilities go in _utils.py

"""

from ._utils import to_parquet_safe  # re-export at package top-level
__all__ = ["to_parquet_safe"]