"""
Text encodings: which one a file the user brings is in, and reading back what OpenAVMKit wrote itself.

Assessors' CSVs come from many places. Excel's "CSV UTF-8" writes UTF-8 with a byte-order mark; its plain
"CSV (Comma delimited)" writes the Windows code page (cp1252 on US and Western European Windows); some exports are
UTF-16. Few users know which they have, so a ``data.load`` entry can say ``"encoding": "auto"`` and
:func:`detect_encoding` works it out.
"""

import codecs
from pathlib import Path

# What a file is read as when it isn't UTF-8 (or UTF-16, with its byte-order mark): Excel's plain CSV on US and
# Western European Windows.
FALLBACK = "cp1252"

_CHUNK = 1 << 20


def detect_encoding(path: str | Path) -> str:
    """Work out a text file's encoding.

    A byte-order mark says UTF-8 (``"utf-8-sig"``) or UTF-16 (``"utf-16"``). Otherwise the whole file is checked
    as UTF-8, a chunk at a time: UTF-8 checks itself, and text in a Windows code page with even one accented letter
    is almost never valid UTF-8, so a file that reads as UTF-8 is (``"utf-8"``). Anything else is read as
    :data:`FALLBACK`.

    Parameters
    ----------
    path : str or Path
        The file.

    Returns
    -------
    str
        An encoding name Python and pandas accept.
    """
    with open(path, "rb") as f:
        head = f.read(4)
        if head.startswith(codecs.BOM_UTF8):
            return "utf-8-sig"
        if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return "utf-16"
        f.seek(0)
        decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            while chunk := f.read(_CHUNK):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            return FALLBACK
    return "utf-8"


def resolve_encoding(path: str | Path, encoding: str | None) -> str | None:
    """The encoding to read a file with, as a ``data.load`` entry gives it.

    Parameters
    ----------
    path : str or Path
        The file.
    encoding : str or None
        The entry's ``encoding``: a name Python knows, ``"auto"`` (worked out by :func:`detect_encoding`), or
        None (pandas' default, UTF-8, as before this setting existed).

    Returns
    -------
    str or None
        The encoding, or None for pandas' default.
    """
    return detect_encoding(path) if encoding == "auto" else encoding


def read_text(path: str | Path) -> str:
    """Read a text file OpenAVMKit wrote itself.

    It writes UTF-8. Earlier versions wrote in the computer's code page, which on Windows is not UTF-8, so a file
    that doesn't read as UTF-8 is read as :data:`FALLBACK`. A byte-order mark (Notepad's "UTF-8 with BOM", say,
    on a settings file edited by hand) is dropped: JSON can't start with one.

    Parameters
    ----------
    path : str or Path
        The file.

    Returns
    -------
    str
        Its text.
    """
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode(FALLBACK, errors="replace")
