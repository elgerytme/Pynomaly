"""Schema definitions for pynomaly analytics package.

This module provides comprehensive schema definitions for anomaly detection,
system health monitoring, financial impact analysis, and ROI tracking.
"""

from .versioning import SchemaVersion, compare_versions, migrate_schema, is_compatible_version
from .validation import (
    validate_schema_compatibility,
    ensure_backward_compatibility,
    SchemaCompatibilityError,
    BackwardCompatibilityValidator,
)
from .analytics.base import MetricFrame, RealTimeMetricFrame
from .analytics.anomaly_kpis import AnomalyKPIFrame
from .analytics.system_health import SystemHealthFrame
from .analytics.financial_impact import FinancialImpactFrame
from .analytics.roi import ROIFrame

# Current schema version
SCHEMA_VERSION = "1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "SchemaVersion",
    "compare_versions",
    "migrate_schema",
    "is_compatible_version",
    "validate_schema_compatibility",
    "ensure_backward_compatibility",
    "SchemaCompatibilityError",
    "BackwardCompatibilityValidator",
    "MetricFrame",
    "RealTimeMetricFrame",
    "AnomalyKPIFrame",
    "SystemHealthFrame",
    "FinancialImpactFrame",
    "ROIFrame",
]
