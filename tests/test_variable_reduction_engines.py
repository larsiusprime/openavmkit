from openavmkit.model_runner import _first_reducing_engine, _takes_reduced_variables


def test_the_assessor_needs_no_variable_reduction():
    assert _first_reducing_engine(["lightgbm", "assessor"], [], {}) is None
    assert _first_reducing_engine(["lightgbm", "assessor", "mra"], [], {}) == "mra"
    # A skipped model doesn't count; an aliased entry counts by its engine.
    assert _first_reducing_engine(["mra", "lightgbm"], ["mra"], {}) is None
    assert _first_reducing_engine(["roll"], [], {"roll": {"model": "assessor"}}) is None
    assert _first_reducing_engine(["lin"], [], {"lin": {"model": "mra"}}) == "mra"


def test_layered_comps_is_an_ml_model_for_variable_reduction():
    """Like the trees, lcomp neither starts the variable recommendations nor takes the reduced list."""
    assert _first_reducing_engine(["lightgbm", "lcomp"], [], {}) is None
    assert _first_reducing_engine(["comps"], [], {"comps": {"model": "lcomp"}}) is None
    assert _first_reducing_engine(["lcomp", "mra"], [], {}) == "mra"
    for engine in ("lcomp", "lightgbm", "xgboost", "catboost"):
        assert not _takes_reduced_variables(engine)
    for engine in ("mra", "gwr", "local_area"):
        assert _takes_reduced_variables(engine)
