"""Rich output that works both inside Jupyter and in plain Python processes.

OpenAVMKit is usually driven from notebooks, but it is also run headless (scripts,
CI, desktop front ends). These helpers use IPython's rich display when a kernel is
running and fall back to plain printing otherwise, so IPython is not a hard
runtime dependency.
"""


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def display(obj) -> None:
    """Show ``obj`` with IPython's rich display if available, else print it."""
    if _in_notebook():
        from IPython.display import display as _display

        _display(obj)
    else:
        print(obj)


def HTML(html: str):
    """Wrap ``html`` for rich display if IPython is available, else return it as-is."""
    if _in_notebook():
        from IPython.display import HTML as _HTML

        return _HTML(html)
    return html
