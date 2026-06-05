"""Build shim for openavmkit.

Almost all packaging metadata lives in pyproject.toml. The one thing we compute here is
``long_description`` (PyPI's rendered README): we rewrite the README's repo-relative Markdown
links to absolute GitHub URLs so they work on PyPI, which renders the README with no repo
context (relative links there 404). The README file in the repo is left untouched, so GitHub
and the docs site keep using relative links. This runs automatically on ``python -m build``.
"""
import re
from pathlib import Path

from setuptools import setup

_REPO = "https://github.com/larsiusprime/openavmkit"
_BRANCH = "master"
_RAW = "https://raw.githubusercontent.com/larsiusprime/openavmkit"

# Match a Markdown link or image whose target is REPO-RELATIVE. We skip targets that are
# already absolute (http/https), mailto, in-page anchors (#...), or site-absolute (/...).
_LINK_RE = re.compile(
    r"(!?)\[([^\]]*)\]\(\s*(?!https?://|mailto:|#|/)([^)\s]+?)\s*\)"
)


def _absolutize_links(md: str) -> str:
    """Rewrite repo-relative Markdown links/images to absolute GitHub URLs."""

    def _repl(m: "re.Match") -> str:
        bang, text, target = m.group(1), m.group(2), m.group(3)
        if bang:  # image -> raw content URL so it renders inline
            return f"{bang}[{text}]({_RAW}/{_BRANCH}/{target})"
        # directory link (trailing slash) -> tree view; file -> blob view. Anchors ride along.
        base = "tree" if target.endswith("/") else "blob"
        return f"[{text}]({_REPO}/{base}/{_BRANCH}/{target})"

    return _LINK_RE.sub(_repl, md)


def _long_description() -> str:
    readme = Path(__file__).parent / "README.md"
    return _absolutize_links(readme.read_text(encoding="utf-8"))


if __name__ == "__main__":
    setup(
        long_description=_long_description(),
        long_description_content_type="text/markdown",
    )
