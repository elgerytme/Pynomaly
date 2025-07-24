"""Domain interfaces for machine learning operations."""

from .automl_operations import (
    AutoMLOptimizationPort,
    ModelSelectionPort,
    HyperparameterOptimizationPort,
    OptimizationConfig,
    AlgorithmConfig,
    EnsembleConfig,
    OptimizationResult,
    OptimizationTrial,
    OptimizationMetric,
    SearchStrategy,
    AlgorithmType,
    EnsembleMethod,
)

from .explainability_operations import (
    ExplainabilityPort,
    ModelInterpretabilityPort,
    ExplanationRequest,
    ExplanationResult,
    GlobalExplanationResult,
    FeatureContribution,
    ExplanationMethod,
    ExplanationScope,
)

from .monitoring_operations import (
    MonitoringPort,
    DistributedTracingPort,
    AlertingPort,
    HealthCheckPort,
    MetricValue,
    TraceSpan,
    PerformanceMetrics,
    MetricType,
    TraceLevel,
)

__all__ = [
    # AutoML operations
    "AutoMLOptimizationPort",
    "ModelSelectionPort", 
    "HyperparameterOptimizationPort",
    "OptimizationConfig",
    "AlgorithmConfig",
    "EnsembleConfig",
    "OptimizationResult",
    "OptimizationTrial",
    "OptimizationMetric",
    "SearchStrategy",
    "AlgorithmType",
    "EnsembleMethod",
    
    # Explainability operations
    "ExplainabilityPort",
    "ModelInterpretabilityPort",
    "ExplanationRequest",
    "ExplanationResult",
    "GlobalExplanationResult",
    "FeatureContribution",
    "ExplanationMethod",
    "ExplanationScope",
    
    # Monitoring operations
    "MonitoringPort",
    "DistributedTracingPort",
    "AlertingPort",
    "HealthCheckPort",
    "MetricValue",
    "TraceSpan",
    "PerformanceMetrics",
    "MetricType",
    "TraceLevel",
]