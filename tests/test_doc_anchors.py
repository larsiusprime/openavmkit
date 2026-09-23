"""Guard against broken in-document anchor links in the Markdown docs.

Our Markdown is rendered by two engines that slugify headings *differently*, and a
link that is correct in one is broken in the other:

- **GitHub** (README, and anyone browsing the repo) strips punctuation and turns each
  remaining space into a hyphen, but does **not** collapse runs. So
  ``## Part A — Smoke test`` becomes ``#part-a--smoke-test`` (two hyphens, because the
  em dash vanished from between two spaces).
- **Python-Markdown / MkDocs** (www.openavmkit.com) does the same, then collapses runs
  of hyphens and whitespace, giving ``#part-a-smoke-test`` (one hyphen).

Every heading containing ``—`` or ``&`` therefore has two different valid anchors, and
you cannot satisfy both at once. The rule we follow is to validate each file against the
engine that actually renders it:

- ``README.md`` and the other repo-root docs are only ever rendered by GitHub and PyPI
  (they are outside ``docs_dir`` and are not part of the MkDocs site), so they use
  GitHub anchors.
- ``docs/docs/*.md`` are the published site, so they use MkDocs anchors.

``mkdocs build --strict`` already catches the MkDocs half. Nothing catches the GitHub
half, which is why this test exists — six anchors in README were silently broken.
"""

import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files rendered by MkDocs (the published site) -> collapsed anchors.
SITE_DOCS = sorted((REPO_ROOT / "docs" / "docs").glob("*.md"))
# Files rendered only by GitHub/PyPI -> uncollapsed anchors.
GITHUB_DOCS = [
    p
    for p in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "changelog.md",
    )
    if p.exists()
]

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+?)\s*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
_ID_COUNT = re.compile(r"^(.*)_(\d+)$")


def _heading_text(raw: str) -> str:
    """Reduce a raw heading line to the plain text both engines slugify.

    Both slugify the *rendered* heading, so ``## [Docs](docs.md)`` slugifies "Docs",
    not the URL. Backticks, ``*`` and other punctuation need no special handling —
    they are stripped by the slug regexes below.

    HTML tags are stripped, but *not* inside inline code spans: a heading like
    ``## 2. Data load: `data.load.<id>` `` has a literal ``<id>`` that both engines
    keep (yielding ``...dataloadid``), and treating it as a tag silently changes the
    expected anchor.
    """
    text = _MD_LINK.sub(r"\1", raw)
    out, pos = [], 0
    for m in _CODE_SPAN.finditer(text):
        out.append(_HTML_TAG.sub("", text[pos : m.start()]))
        out.append(m.group(2))
        pos = m.end()
    out.append(_HTML_TAG.sub("", text[pos:]))
    return "".join(out)


def _slug_mkdocs(text: str) -> str:
    """Python-Markdown's default ``toc`` slugify: ASCII-fold, strip, collapse runs."""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _slug_github(text: str) -> str:
    """GitHub's heading slugger. Same as above but *without* collapsing runs."""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return value.replace(" ", "-")


def _anchors(path: Path, slugify, dupe_suffix: str) -> set[str]:
    """Every anchor `path` exposes under `slugify`, including duplicate-heading suffixes.

    Both engines de-duplicate repeated headings, but with different separators:
    Python-Markdown appends ``_1`` and GitHub appends ``-1``.
    """
    found: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING.match(line)
        if not m:
            continue
        slug = slugify(_heading_text(m.group(2)))
        if not slug:
            continue
        if slug not in found:
            found.add(slug)
            continue
        n = 1
        while f"{slug}{dupe_suffix}{n}" in found:
            n += 1
        found.add(f"{slug}{dupe_suffix}{n}")
    return found


def _links(path: Path):
    """Yield ``(line_no, link_text, target)`` for every Markdown link outside a fence."""
    in_fence = False
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for text, target in _MD_LINK.findall(line):
            yield i, text, target


def _check(path: Path, slugify, dupe_suffix: str) -> list[str]:
    """Return a human-readable problem per anchor link in `path` that resolves nowhere.

    Only links whose target file we can find and parse are checked; external URLs and
    links into non-Markdown files are skipped rather than guessed at.
    """
    problems = []
    for line_no, text, target in _links(path):
        if target.startswith(("http://", "https://", "mailto:", "<")):
            continue
        rel, _, frag = target.partition("#")
        if not frag:
            continue
        if rel:
            dest = (path.parent / rel).resolve()
            if dest.suffix.lower() != ".md" or not dest.is_file():
                continue
        else:
            dest = path
        if frag not in _anchors(dest, slugify, dupe_suffix):
            problems.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}:{line_no}: "
                f"[{text}] -> {dest.relative_to(REPO_ROOT).as_posix()}#{frag} "
                f"(no such heading)"
            )
    return problems


@pytest.mark.parametrize("path", SITE_DOCS, ids=lambda p: p.name)
def test_site_doc_anchors_resolve(path):
    """docs/docs/*.md are published by MkDocs, so they must use collapsed anchors."""
    problems = _check(path, _slug_mkdocs, "_")
    assert not problems, "Broken anchor(s) as rendered by MkDocs:\n" + "\n".join(problems)


@pytest.mark.parametrize("path", GITHUB_DOCS, ids=lambda p: p.name)
def test_github_doc_anchors_resolve(path):
    """Repo-root docs are rendered only by GitHub/PyPI, so they must use GitHub anchors."""
    problems = _check(path, _slug_github, "-")
    assert not problems, "Broken anchor(s) as rendered by GitHub:\n" + "\n".join(problems)
