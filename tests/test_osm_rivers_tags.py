from openavmkit.utilities.openstreetmap import OpenStreetMapService


def test_rivers_query_includes_waterway_linestrings():
    tags = OpenStreetMapService()._get_tags("rivers")
    # waterway linestrings (the added coverage) — how rivers/streams are
    # commonly mapped in OSM
    assert "waterway" in tags
    assert "river" in tags["waterway"]
    assert "stream" in tags["waterway"]
    # original polygon "water" coverage retained
    assert "water" in tags
