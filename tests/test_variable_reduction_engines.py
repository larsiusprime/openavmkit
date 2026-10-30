from openavmkit.model_runner import _first_reducing_engine


def test_the_assessor_needs_no_variable_reduction():
    assert _first_reducing_engine(["lightgbm", "assessor"], [], {}) is None
    assert _first_reducing_engine(["lightgbm", "assessor", "mra"], [], {}) == "mra"
    # A skipped model doesn't count; an aliased entry counts by its engine.
    assert _first_reducing_engine(["mra", "lightgbm"], ["mra"], {}) is None
    assert _first_reducing_engine(["roll"], [], {"roll": {"model": "assessor"}}) is None
    assert _first_reducing_engine(["lin"], [], {"lin": {"model": "mra"}}) == "mra"
