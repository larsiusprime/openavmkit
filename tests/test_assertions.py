"""Tests for the comparison helpers in openavmkit.utilities.assertions.

These back dicts_are_equal and a number of test assertions across the suite, so a
false "equal" here silently weakens everything built on top of them.
"""
import numpy as np
import pandas as pd

from openavmkit.utilities.assertions import lists_are_equal, dicts_are_equal


def test_lists_are_equal_detects_a_mismatch_anywhere():
    """Every element decides the result, not just the last one.

    The loop used to overwrite `result` on each pass, so only the final element
    mattered and lists_are_equal([2, 2], [3, 2]) returned True.
    """
    assert lists_are_equal([1, 2, 3], [1, 2, 3]) is True

    # mismatch in the FIRST element, match in the last -- the regression case
    assert lists_are_equal([2, 2], [3, 2]) is False
    # mismatch in the middle
    assert lists_are_equal([1, 2, 3], [1, 9, 3]) is False
    # mismatch in the last
    assert lists_are_equal([1, 2, 3], [1, 2, 9]) is False
    # several mismatches, last element matches
    assert lists_are_equal([1, 2, 3], [9, 9, 3]) is False


def test_lists_are_equal_edge_cases():
    assert lists_are_equal([], []) is True
    assert lists_are_equal([1], [1, 2]) is False
    assert lists_are_equal([1, 2], [1]) is False
    assert lists_are_equal([1.0, 2.0], [1.0, 2.0]) is True


def test_dicts_are_equal_inherits_the_fix():
    """dicts_are_equal delegates list values to lists_are_equal."""
    assert dicts_are_equal({"k": [1, 2]}, {"k": [1, 2]}) is True
    assert dicts_are_equal({"k": [2, 2]}, {"k": [3, 2]}) is False
