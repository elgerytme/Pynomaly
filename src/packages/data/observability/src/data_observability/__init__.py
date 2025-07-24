"""
Data Observability Package - Data Monitoring and Quality Assurance

This package provides comprehensive data observability capabilities including:
- Data catalog management
- Data lineage tracking
- Pipeline health monitoring
- Predictive quality assessment
- Data quality metrics
- Anomaly detection for data pipelines
"""

__version__ = "0.1.0"
__author__ = "Anomaly Detection Team"
__email__ = "support@anomaly_detection.com"

from .application.facades.observability_facade import DataObservabilityFacade

__all__ = [
    "DataObservabilityFacade",
    "__version__",
    "__author__",
    "__email__",
]