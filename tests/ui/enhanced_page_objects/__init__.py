"""Enhanced page objects with robust UI interaction patterns."""

from .base_page import BasePage, retry_on_failure
from .dashboard_page import DashboardPage

__all__ = ["BasePage", "DashboardPage", "retry_on_failure"]
