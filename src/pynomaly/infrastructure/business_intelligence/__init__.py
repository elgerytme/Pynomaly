"""Business Intelligence integration module.

Provides comprehensive business intelligence capabilities including performance
reporting, algorithm recommendations, and executive dashboards as part of
Phase 2 strategic enhancement.
"""

from .reporting_service import ReportingService, ExecutiveReport, TechnicalReport

__all__ = [
    'ReportingService',
    'ExecutiveReport', 
    'TechnicalReport'
]