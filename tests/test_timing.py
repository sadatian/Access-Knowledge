import pytest
from datetime import datetime, timezone
import knowledge.timing as timing
from knowledge.timing import (
    CellTimer,
    enable_cell_timer,
    disable_cell_timer,
    format_duration,
    format_execution_time_badge,
    parse_execution_metadata,
)


class MockEvents:
    def __init__(self):
        self.callbacks = {}

    def register(self, event_name, callback):
        self.callbacks[event_name] = callback

    def unregister(self, event_name, callback):
        if event_name in self.callbacks and self.callbacks[event_name] == callback:
            del self.callbacks[event_name]


class MockIPython:
    def __init__(self):
        self.events = MockEvents()


def test_format_duration():
    assert format_duration(-1.0) == "0.0ms"
    assert format_duration(0.0000123) == "12.3µs"
    assert format_duration(0.00095) == "950.0µs"
    assert format_duration(0.0154) == "15.4ms"
    assert format_duration(0.852) == "852.0ms"
    assert format_duration(1.234) == "1.23s"
    assert format_duration(59.99) == "59.99s"
    assert format_duration(65.4) == "1m 5.4s"
    assert format_duration(125.0) == "2m 5.0s"


def test_parse_execution_metadata_valid():
    metadata = {
        "execution": {
            "iopub.execute_input": "2026-08-21T12:00:00.000000Z",
            "iopub.status.idle": "2026-08-21T12:00:01.500000Z",
        }
    }
    duration = parse_execution_metadata(metadata)
    assert duration is not None
    assert pytest.approx(duration, 0.001) == 1.5

    badge = format_execution_time_badge(metadata)
    assert badge == "1.50s"


def test_parse_execution_metadata_busy_fallback():
    metadata = {
        "execution": {
            "iopub.status.busy": "2026-08-21T12:00:00.000000Z",
            "iopub.status.idle": "2026-08-21T12:00:00.025000Z",
        }
    }
    duration = parse_execution_metadata(metadata)
    assert duration is not None
    assert pytest.approx(duration, 0.001) == 0.025

    badge = format_execution_time_badge(metadata)
    assert badge == "25.0ms"


def test_parse_execution_metadata_invalid_or_missing():
    assert parse_execution_metadata(None) is None
    assert parse_execution_metadata({}) is None
    assert parse_execution_metadata({"execution": {}}) is None
    assert parse_execution_metadata({"execution": {"iopub.status.idle": "invalid"}}) is None
    assert format_execution_time_badge(None) == ""
    assert format_execution_time_badge({}) == ""


def test_cell_timer_lifecycle():
    timer = CellTimer()
    mock_ip = MockIPython()

    assert not timer.is_enabled
    assert timer.enable(mock_ip) is True
    assert timer.is_enabled is True
    assert "pre_run_cell" in mock_ip.events.callbacks
    assert "post_run_cell" in mock_ip.events.callbacks

    # Test pre/post hooks
    timer.pre_run_cell()
    assert timer._start_wall is not None
    timer.post_run_cell()
    assert timer._start_wall is None

    # Disable timer
    assert timer.disable() is True
    assert not timer.is_enabled
    assert "pre_run_cell" not in mock_ip.events.callbacks


def test_timer_global_helpers():
    # Calling in non-interactive environment should not fail
    disable_cell_timer()
    assert timing._cell_timer.is_enabled is False
