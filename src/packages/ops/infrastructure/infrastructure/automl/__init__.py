"""Advanced AutoML infrastructure for Pynomaly.

This module provides state-of-the-art automated machine learning capabilities including:
- Advanced hyperparameter optimization
- Multi-objective optimization
- Meta-learning and warm starts
- Bayesian optimization with advanced acquisition functions
- Population-based training
- Automated early stopping
"""

from .advanced_optimizer import (
    AcquisitionFunction,
    AdvancedHyperparameterOptimizer,
    AdvancedOptimizationConfig,
    EarlyStoppingConfig,
    MetaLearningConfig,
    MetaLearningStrategy,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStrategy,
    OptimizationTrial,
)

__all__ = [
    "AdvancedHyperparameterOptimizer",
    "AdvancedOptimizationConfig",
    "OptimizationStrategy",
    "AcquisitionFunction",
    "MetaLearningStrategy",
    "OptimizationObjective",
    "OptimizationConstraint",
    "EarlyStoppingConfig",
    "MetaLearningConfig",
    "OptimizationTrial",
    "OptimizationResult",
]
