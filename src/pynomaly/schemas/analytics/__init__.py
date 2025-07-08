"""Analytics schema definitions for pynomaly.

This module provides specialized schema definitions for various analytics
including anomaly detection, system health, financial impact, and ROI analysis.
"""

from .base import MetricFrame, RealTimeMetricFrame
from .anomaly_kpis import (
    AnomalyKPIFrame,
    AnomalyDetectionMetrics,
    AnomalyClassificationMetrics,
    AnomalySeverity,
    AnomalyCategory,
)
from .system_health import (
    SystemHealthFrame,
    SystemResourceMetrics,
    SystemPerformanceMetrics,
    SystemStatusMetrics,
    SystemStatus,
)
from .financial_impact import (
    FinancialImpactFrame,
    CostMetrics,
    SavingsMetrics,
    RevenueMetrics,
)
from .roi import (
    ROIFrame,
    InvestmentMetrics,
    CostBenefitAnalysis,
)

__all__ = [
    "MetricFrame",
    "RealTimeMetricFrame",
    "AnomalyKPIFrame",
    "AnomalyDetectionMetrics",
    "AnomalyClassificationMetrics",
    "AnomalySeverity",
    "AnomalyCategory",
    "SystemHealthFrame",
    "SystemResourceMetrics",
    "SystemPerformanceMetrics",
    "SystemStatusMetrics",
    "SystemStatus",
    "FinancialImpactFrame",
    "CostMetrics",
    "SavingsMetrics",
    "RevenueMetrics",
    "ROIFrame",
    "InvestmentMetrics",
    "CostBenefitAnalysis",
]
