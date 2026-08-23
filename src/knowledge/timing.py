"""IPython / Jupyter Cell Execution Timing Module.

Provides automatic execution time tracking and formatting for code cells
running in interactive IPython kernels and Jupyter sessions without altering
plain Python syntax or causing compatibility issues.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional


def format_duration(seconds: float) -> str:
    """Format duration in seconds into a clean, human-readable string."""
    if seconds < 0:
        return "0.0ms"
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def parse_execution_metadata(metadata: Optional[dict[str, Any]]) -> Optional[float]:
    """Extract elapsed execution time in seconds from cell execution metadata.

    Supports Jupyter/ExecutePreprocessor execution metadata containing ISO timestamps.
    """
    if not metadata or not isinstance(metadata, dict):
        return None
    exec_info = metadata.get("execution")
    if not exec_info or not isinstance(exec_info, dict):
        return None

    start_str = exec_info.get("iopub.execute_input") or exec_info.get("iopub.status.busy")
    end_str = exec_info.get("iopub.status.idle")
    if not start_str or not end_str:
        return None

    try:
        from datetime import datetime
        start_t = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        end_t = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        delta = (end_t - start_t).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None


def format_execution_time_badge(metadata: Optional[dict[str, Any]]) -> str:
    """Format execution metadata into a formatted string badge if available."""
    duration = parse_execution_metadata(metadata)
    if duration is None:
        return ""
    return format_duration(duration)


class CellTimer:
    """Singleton helper managing IPython pre_run_cell and post_run_cell event hooks."""

    def __init__(self) -> None:
        self._start_wall: Optional[float] = None
        self._start_cpu: Optional[float] = None
        self._is_enabled: bool = False
        self._ipython_instance: Any = None

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    def pre_run_cell(self, *args: Any, **kwargs: Any) -> None:
        """Capture cell start timestamps."""
        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()

    def post_run_cell(self, *args: Any, **kwargs: Any) -> None:
        """Calculate and print cell elapsed time."""
        if self._start_wall is None or self._start_cpu is None:
            return
        wall_elapsed = time.perf_counter() - self._start_wall
        cpu_elapsed = time.process_time() - self._start_cpu
        self._start_wall = None
        self._start_cpu = None

        wall_str = format_duration(wall_elapsed)
        cpu_str = format_duration(cpu_elapsed)
        print(f"\033[90m⏱️  Cell execution: {wall_str} (CPU: {cpu_str})\033[0m")

    def enable(self, ipython: Optional[Any] = None) -> bool:
        """Register hooks in IPython kernel if available."""
        if self._is_enabled:
            return True

        if ipython is None:
            try:
                import IPython
                ipython = IPython.get_ipython()
            except Exception:
                ipython = None

        if ipython is None or not hasattr(ipython, "events"):
            return False

        try:
            ipython.events.register("pre_run_cell", self.pre_run_cell)
            ipython.events.register("post_run_cell", self.post_run_cell)
            self._ipython_instance = ipython
            self._is_enabled = True
            return True
        except Exception:
            return False

    def disable(self) -> bool:
        """Deregister hooks from IPython kernel."""
        if not self._is_enabled or self._ipython_instance is None:
            self._is_enabled = False
            return True

        try:
            self._ipython_instance.events.unregister("pre_run_cell", self.pre_run_cell)
            self._ipython_instance.events.unregister("post_run_cell", self.post_run_cell)
        except Exception:
            pass
        finally:
            self._is_enabled = False
            self._ipython_instance = None
        return True


# Global default instance
_cell_timer = CellTimer()


def enable_cell_timer(verbose: bool = True) -> bool:
    """Enable automatic cell execution timing in the current IPython/Jupyter session."""
    success = _cell_timer.enable()
    if success and verbose:
        print("\033[32m[OK] Automatic cell execution timer enabled for IPython/Jupyter.\033[0m")
    return success


def disable_cell_timer() -> bool:
    """Disable automatic cell execution timing."""
    return _cell_timer.disable()


def load_ipython_extension(ipython: Any) -> None:
    """Standard IPython extension entrypoint: `%load_ext knowledge.timing`."""
    _cell_timer.enable(ipython)


def unload_ipython_extension(ipython: Any) -> None:
    """Standard IPython extension unload entrypoint: `%unload_ext knowledge.timing`."""
    _cell_timer.disable()
