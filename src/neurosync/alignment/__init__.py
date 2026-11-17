"""Timestamp alignment between simulated task events and synthetic EEG."""

from neurosync.alignment.events import (
    annotate_raw_with_task_events,
    epoch_task_events,
)

__all__ = ["annotate_raw_with_task_events", "epoch_task_events"]
