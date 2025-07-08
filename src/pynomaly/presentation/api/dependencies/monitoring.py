"""Monitoring service dependencies for FastAPI."""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from pynomaly.infrastructure.monitoring.external_monitoring_service import ExternalMonitoringService
from pynomaly.infrastructure.monitoring.dual_metrics_service import DualMetricsService


def get_monitoring_service(request: Request) -> Optional[ExternalMonitoringService]:
    """Get the external monitoring service from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ExternalMonitoringService instance or None if not available
    """
    return getattr(request.app.state, 'monitoring_service', None)


def get_dual_metrics_service(request: Request) -> Optional[DualMetricsService]:
    """Get the dual metrics service from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        DualMetricsService instance or None if not available
    """
    return getattr(request.app.state, 'dual_metrics_service', None)
