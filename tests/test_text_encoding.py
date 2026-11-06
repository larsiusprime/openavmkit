"""Text encodings (openavmkit.utilities.text): a CSV's encoding worked out from the file, a data.load entry's
`encoding`, and reading back what an earlier OpenAVMKit wrote in the Windows code page."""

import codecs

import pandas as pd
import pytest

from openavmkit.data import load_dataframe
from openavmkit.utilities.text import FALLBACK, detect_encoding, read_text

ROWS = "PIN,OWNER\n1,José Peña\n2,Zoë Müller\n"


def _write(path, text, encoding):
	path.write_bytes(text.encode(encoding))
	return path


def test_detect_encoding(tmp_path):
	assert detect_encoding(_write(tmp_path / "a.csv", ROWS, "utf-8")) == "utf-8"
	assert detect_encoding(_write(tmp_path / "b.csv", ROWS, "utf-8-sig")) == "utf-8-sig"  # Excel's "CSV UTF-8"
	assert detect_encoding(_write(tmp_path / "c.csv", ROWS, "utf-16")) == "utf-16"
	assert detect_encoding(_write(tmp_path / "d.csv", ROWS, "cp1252")) == FALLBACK == "cp1252"  # Excel's plain CSV
	assert detect_encoding(_write(tmp_path / "e.csv", "PIN\n1\n", "ascii")) == "utf-8"  # plain ASCII is UTF-8


def test_a_non_utf8_letter_far_down_the_file_is_found(tmp_path):
	"""The whole file is checked, not a sample: a county export's one accented name can be on row 500,000."""
	text = "PIN,OWNER\n" + "".join(f"{i},SMITH\n" for i in range(300_000)) + "300000,Peña\n"
	assert detect_encoding(_write(tmp_path / "long.csv", text, "cp1252")) == "cp1252"
	assert detect_encoding(_write(tmp_path / "long8.csv", text, "utf-8")) == "utf-8"


@pytest.mark.parametrize("written", ["utf-8", "utf-8-sig", "utf-16", "cp1252"])
def test_a_load_entry_with_encoding_auto_reads_the_file_as_it_is(tmp_path, monkeypatch, written):
	monkeypatch.chdir(tmp_path)
	(tmp_path / "in").mkdir()
	_write(tmp_path / "in" / "cama.csv", ROWS, written)
	entry = {"filename": "cama.csv", "encoding": "auto", "load": {"key": ["PIN", "string"], "owner": ["OWNER", "string"]}}
	df = load_dataframe(entry, {})
	assert df["owner"].tolist() == ["José Peña", "Zoë Müller"]


def test_a_load_entry_names_its_encoding(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	(tmp_path / "in").mkdir()
	_write(tmp_path / "in" / "cama.csv", ROWS, "cp1252")
	load = {"key": ["PIN", "string"], "owner": ["OWNER", "string"]}
	assert load_dataframe({"filename": "cama.csv", "encoding": "cp1252", "load": load}, {})["owner"][0] == "José Peña"
	# Without the setting, as before it existed: pandas' UTF-8, which can't read it.
	with pytest.raises(UnicodeDecodeError):
		load_dataframe({"filename": "cama.csv", "load": load}, {})


def test_read_text_reads_what_an_earlier_version_wrote_on_windows(tmp_path):
	assert read_text(_write(tmp_path / "new.json", '{"name": "José"}', "utf-8")) == '{"name": "José"}'
	assert read_text(_write(tmp_path / "old.json", '{"name": "José"}', "cp1252")) == '{"name": "José"}'
	assert codecs.lookup(FALLBACK).name == "cp1252"


def test_read_text_drops_a_byte_order_mark(tmp_path):
	"""A settings file saved by Notepad as "UTF-8 with BOM": JSON can't start with the mark."""
	import json

	assert json.loads(read_text(_write(tmp_path / "settings.json", '{"a": "é"}', "utf-8-sig"))) == {"a": "é"}


def test_every_text_read_and_write_names_its_encoding():
	"""On Windows, Python reads and writes text in the computer's code page (cp1252) unless told otherwise, so a
	"é" written on one computer misreads on another. OpenAVMKit's own reads go through utilities.text.read_text,
	its writes say encoding="utf-8"; this finds any that don't."""
	import ast
	from pathlib import Path

	root = Path(__file__).resolve().parents[1] / "openavmkit"
	missing = []
	for path in root.rglob("*.py"):
		for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
			if not isinstance(node, ast.Call):
				continue
			f, kw = node.func, {k.arg for k in node.keywords}
			bare = isinstance(f, ast.Name) and f.id == "open"
			method = isinstance(f, ast.Attribute) and f.attr in ("read_text", "write_text", "TextIOWrapper")
			if not (bare or method) or "encoding" in kw:
				continue
			modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
			modes += [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
			if any("b" in str(m) for m in modes):
				continue  # bytes
			missing.append(f"{path.relative_to(root.parent)}:{node.lineno}: {ast.unparse(node)[:80]}")
	assert not missing, 'Say encoding="utf-8" (or read with utilities.text.read_text):\n' + "\n".join(missing)
