"""Knowledge Retrieval A-Z Package"""

__version__ = "0.1.0"

from knowledge.timing import (
    CellTimer,
    disable_cell_timer,
    enable_cell_timer,
    format_duration,
    format_execution_time_badge,
    parse_execution_metadata,
)

# Auto-enable cell timer if imported inside an interactive IPython/Jupyter kernel
try:
    import IPython
    _ip = IPython.get_ipython()
    if _ip is not None:
        enable_cell_timer(verbose=False)
except Exception:
    pass

__all__ = [
    "__version__",
    "CellTimer",
    "enable_cell_timer",
    "disable_cell_timer",
    "format_duration",
    "format_execution_time_badge",
    "parse_execution_metadata",
]
