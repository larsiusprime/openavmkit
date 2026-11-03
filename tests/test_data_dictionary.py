"""The data dictionary says, for every field, what it is and where its values come from."""

import json
from importlib import resources

TYPES = {"string", "number", "boolean", "date", "percent", "geometry"}
SOURCES = {"input", "derived", "either", "reserved"}


def test_every_entry_is_complete():
	text = resources.files("openavmkit.resources.settings").joinpath("data_dictionary.json").read_text()
	entries = json.loads(text)
	for key, entry in entries.items():
		assert entry.get("name"), key
		assert entry.get("type") in TYPES, key
		# input: only from the user's data; derived: always computed by OpenAVMKit;
		# either: computed when missing, kept when given; reserved: a name kept for a field
		# OpenAVMKit means to compute but doesn't yet.
		assert entry.get("source") in SOURCES, key
