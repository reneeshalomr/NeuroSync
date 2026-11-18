"""Data contracts and calculations for the NeuroSync research dashboard."""

from neurosync.dashboard.data import (
    DashboardData,
    DashboardDataError,
    load_dashboard_data,
)

__all__ = ["DashboardData", "DashboardDataError", "load_dashboard_data"]
