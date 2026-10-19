"""Progress reporting and cooperative cancellation for long-running work.

Notebook users see OpenAVMKit's printed output. Other front ends (scripts, CI,
desktop apps) need structured progress instead. Code that does long-running work
calls :func:`report` at meaningful points; a caller that wants the events wraps
the work in :func:`reporting`. With no listener installed, :func:`report` is a
no-op, so notebooks behave exactly as before.

Cancellation is cooperative: a caller passes a ``cancel`` callable to
:func:`reporting`, and every :func:`report` call checks it and raises
:class:`Cancelled` if it returns True.
"""

from __future__ import annotations

import contextvars
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
    """

    stage: Optional[str]
    step: str
    fraction: Optional[float]
    message: str


class Cancelled(Exception):
    """Raised by :func:`report` when the caller has requested cancellation."""


ProgressCallback = Callable[[ProgressEvent], None]
CancelCheck = Callable[[], bool]

_listener: contextvars.ContextVar[Optional[ProgressCallback]] = contextvars.ContextVar("_listener", default=None)
_cancel: contextvars.ContextVar[Optional[CancelCheck]] = contextvars.ContextVar("_cancel", default=None)
_stage: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_stage", default=None)


@contextmanager
def reporting(
    on_progress: Optional[ProgressCallback] = None,
    cancel: Optional[CancelCheck] = None,
    stage: Optional[str] = None,
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
    """
    tokens = [
        _listener.set(on_progress),
        _cancel.set(cancel),
        _stage.set(stage),
    ]
    try:
        yield
    finally:
        _stage.reset(tokens[2])
        _cancel.reset(tokens[1])
        _listener.reset(tokens[0])


def report(step: str, fraction: Optional[float] = None, message: str = "") -> None:
    """Report progress, then raise :class:`Cancelled` if cancellation was requested.

    Parameters
    ----------
    step : str
        Short machine-friendly name of the current step.
    fraction : float, optional
        Overall completion of the current stage in ``[0, 1]``.
    message : str, optional
        Human-readable description. Defaults to ``step``.
    """
    listener = _listener.get()
    if listener is not None:
        listener(ProgressEvent(_stage.get(), step, fraction, message or step))
    check_cancelled()


def check_cancelled() -> None:
    """Raise :class:`Cancelled` if the active cancel check returns True."""
    cancel = _cancel.get()
    if cancel is not None and cancel():
        raise Cancelled()
