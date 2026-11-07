"""Progress reporting and cooperative cancellation for long-running work.

Notebook users see OpenAVMKit's printed output. Other front ends (scripts, CI,
desktop apps) need structured progress instead. Code that does long-running work
calls :func:`report` at meaningful points; a caller that wants the events wraps
the work in :func:`reporting`. With no listener installed, :func:`report` is a
no-op, so notebooks behave exactly as before.

Cancellation is cooperative: a caller passes a ``cancel`` callable to
:func:`reporting`, and every :func:`report` call checks it and raises
:class:`Cancelled` if it returns True.

Skipping is the same, for one optional part of the work (an enrichment, say): a
:func:`report` that names its ``part`` raises :class:`Skipped` when the caller's
``skip`` callable says that part is to be skipped. The code running the part
catches it, leaves the part's results out, and carries on.
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass(frozen=True)
class ProgressEvent:
    """One progress update.

    Attributes
    ----------
    stage : str or None
        The pipeline stage being run (e.g. ``"assemble"``), if known.
    step : str
        Short machine-friendly name of the current step (e.g. ``"load_dataframes"``).
    fraction : float or None
        Overall completion of the stage in ``[0, 1]``, if known.
    message : str
        Human-readable description of what is happening.
    sub : float or None
        Completion of the current step itself in ``[0, 1]``, if known.
    part : str or None
        The optional part this is in (``"distances"``), which can be skipped.
    """

    stage: Optional[str]
    step: str
    fraction: Optional[float]
    message: str
    sub: Optional[float] = None
    part: Optional[str] = None


class Cancelled(Exception):
    """Raised by :func:`report` when the caller has requested cancellation."""


class Skipped(BaseException):
    """Raised by :func:`report` when the caller has asked to skip the part it's in.

    A ``BaseException``, like ``KeyboardInterrupt``: the part's own ``except Exception``
    handlers (a failed download, a bad batch) must not swallow it and carry on."""

    def __init__(self, part: str):
        super().__init__(f"{part} skipped")
        self.part = part


ProgressCallback = Callable[[ProgressEvent], None]
CancelCheck = Callable[[], bool]
SkipCheck = Callable[[str], bool]

_listener: contextvars.ContextVar[Optional[ProgressCallback]] = contextvars.ContextVar("_listener", default=None)
_cancel: contextvars.ContextVar[Optional[CancelCheck]] = contextvars.ContextVar("_cancel", default=None)
_stage: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_stage", default=None)
_skip: contextvars.ContextVar[Optional[SkipCheck]] = contextvars.ContextVar("_skip", default=None)
# The part of the work running now (in_part), for check_part() deep inside it: a model's tuning, say.
_part_now: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_part_now", default=None)

# A file naming parts to skip, one per line: how a caller asks to skip a part that's running in worker processes
# (folds tuned in parallel), which its skip check can't reach. Workers inherit the variable; the file is read as they
# check, so a part named after they started is still skipped.
SKIP_FILE_ENV = "OPENAVMKIT_SKIP_FILE"

# The last step reported: (step, fraction, message), for detail() to add to.
_current: contextvars.ContextVar[Optional[tuple[str, Optional[float], str]]] = contextvars.ContextVar(
    "_current", default=None
)


@contextmanager
def reporting(
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelCheck] = None,
    stage: Optional[str] = None,
    skip: Optional[SkipCheck] = None,
) -> Iterator[None]:
    """Install a progress listener and/or cancel check for the enclosed block.

    Parameters
    ----------
    on_progress : callable, optional
        Called with a :class:`ProgressEvent` for every :func:`report`.
    cancel : callable, optional
        Returns True when the work should stop; checked at every :func:`report`.
    stage : str, optional
        Stage name attached to events reported inside the block.
    skip : callable, optional
        Given a part's name, returns True when that part should be skipped;
        checked at every :func:`report` that names a part.
    """
    tokens = [
        _listener.set(on_progress),
        _cancel.set(cancel),
        _stage.set(stage),
    ]
    skip_token = _skip.set(skip)
    current_token = _current.set(None)
    try:
        yield
    finally:
        _current.reset(current_token)
        _skip.reset(skip_token)
        _stage.reset(tokens[2])
        _cancel.reset(tokens[1])
        _listener.reset(tokens[0])


def report(
    step: str,
    fraction: Optional[float] = None,
    message: str = "",
    sub: Optional[float] = None,
    part: Optional[str] = None,
) -> None:
    """Report progress, then raise :class:`Cancelled` if cancellation was requested, or
    :class:`Skipped` if the part it names is to be skipped.

    Parameters
    ----------
    step : str
        Short machine-friendly name of the current step.
    fraction : float, optional
        Overall completion of the current stage in ``[0, 1]``.
    message : str, optional
        Human-readable description. Defaults to ``step``.
    sub : float, optional
        Completion of this step itself in ``[0, 1]``.
    part : str, optional
        The optional part this is in: the caller may skip it here.
    """
    listener = _listener.get()
    if listener is not None:
        _current.set((step, fraction, message or step))
        listener(ProgressEvent(_stage.get(), step, fraction, message or step, sub, part))
    check_cancelled()
    if part is not None:
        check_skipped(part)


def detail(text: str, sub: Optional[float] = None) -> None:
    """Say what the current step is doing now (which model group, say), without starting a new step.

    Reports the last step again, its message followed by ``text``, with ``sub`` as the step's own
    completion. Generic helpers that don't know which step they're in (``do_per_model_group``) call
    this. A no-op with no listener, or before any step has been reported.
    """
    listener = _listener.get()
    current = _current.get()
    if listener is not None and current is not None:
        step, fraction, message = current
        listener(ProgressEvent(_stage.get(), step, fraction, f"{message}: {text}", sub))
    check_cancelled()


def check_skipped(part: str) -> None:
    """Raise :class:`Skipped` if the active skip check says to skip ``part``."""
    skip = _skip.get()
    if skip is not None and skip(part):
        raise Skipped(part)


@contextmanager
def in_part(part: str) -> Iterator[None]:
    """Mark the enclosed work as ``part`` (``"model:<group>:<model>"``), for :func:`check_part` inside it."""
    token = _part_now.set(part)
    try:
        yield
    finally:
        _part_now.reset(token)


def current_part() -> Optional[str]:
    """The part marked by :func:`in_part` in this thread, if any (for passing to worker threads)."""
    return _part_now.get()


def skip_file_names(part: str) -> bool:
    """Whether the skip file (``OPENAVMKIT_SKIP_FILE``) names ``part``."""
    path = os.environ.get(SKIP_FILE_ENV)
    if not path:
        return False
    try:
        with open(path, encoding="utf-8") as f:
            return part in {line.strip() for line in f}
    except OSError:
        return False


def check_part(part: Optional[str] = None) -> None:
    """Raise :class:`Skipped` if the part running now (``part``, or the one :func:`in_part` marked) is to be skipped:
    by the caller's skip check, or by the skip file (which reaches worker processes). Long work calls this at safe
    points: between tuning trials, between folds."""
    part = part if part is not None else _part_now.get()
    if part is None:
        return
    check_skipped(part)
    if skip_file_names(part):
        raise Skipped(part)


def check_cancelled() -> None:
    """Raise :class:`Cancelled` if the active cancel check returns True."""
    cancel = _cancel.get()
    if cancel is not None and cancel():
        raise Cancelled()
