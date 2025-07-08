"""Factories for domain value objects using factory-boy."""

from __future__ import annotations

from typing import Any

import factory
from factory import fuzzy

from pynomaly.domain.value_objects import (
    AnomalyScore,
    ContaminationRate,
    PerformanceMetrics,
    ThresholdConfig,
)


class AnomalyScoreFactory(factory.Factory):
    """Factory for creating AnomalyScore instances."""

    class Meta:
        model = AnomalyScore

    value = factory.Faker("pyfloat", min_value=0.0, max_value=1.0)

    @classmethod
    def high_score(cls) -> AnomalyScore:
        """Create high anomaly score (> 0.8)."""
        return cls(value=factory.Faker("pyfloat", min_value=0.8, max_value=1.0).generate())

    @classmethod
    def medium_score(cls) -> AnomalyScore:
        """Create medium anomaly score (0.3 - 0.8)."""
        return cls(value=factory.Faker("pyfloat", min_value=0.3, max_value=0.8).generate())

    @classmethod
    def low_score(cls) -> AnomalyScore:
        """Create low anomaly score (< 0.3)."""
        return cls(value=factory.Faker("pyfloat", min_value=0.0, max_value=0.3).generate())


class ContaminationRateFactory(factory.Factory):
    """Factory for creating ContaminationRate instances."""

    class Meta:
        model = ContaminationRate

    value = factory.Faker("pyfloat", min_value=0.05, max_value=0.3)

    @classmethod
    def auto(cls) -> ContaminationRate:
        """Create auto contamination rate."""
        return ContaminationRate.auto()

    @classmethod
    def low_contamination(cls) -> ContaminationRate:
        """Create low contamination rate (< 0.1)."""
        return cls(value=factory.Faker("pyfloat", min_value=0.01, max_value=0.1).generate())

    @classmethod
    def high_contamination(cls) -> ContaminationRate:
        """Create high contamination rate (> 0.2)."""
        return cls(value=factory.Faker("pyfloat", min_value=0.2, max_value=0.5).generate())


class PerformanceMetricsFactory(factory.Factory):
    """Factory for creating PerformanceMetrics instances."""

    class Meta:
        model = PerformanceMetrics

    @factory.lazy_attribute
    def precision(self) -> float:
        """Generate precision score."""
        return factory.Faker("pyfloat", min_value=0.0, max_value=1.0).generate()

    @factory.lazy_attribute
    def recall(self) -> float:
        """Generate recall score."""
        return factory.Faker("pyfloat", min_value=0.0, max_value=1.0).generate()

    @factory.lazy_attribute
    def f1_score(self) -> float:
        """Generate F1 score."""
        # F1 is harmonic mean of precision and recall
        if hasattr(self, 'precision') and hasattr(self, 'recall') and self.precision + self.recall > 0:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        return factory.Faker("pyfloat", min_value=0.0, max_value=1.0).generate()

    @factory.lazy_attribute
    def accuracy(self) -> float:
        """Generate accuracy score."""
        return factory.Faker("pyfloat", min_value=0.0, max_value=1.0).generate()

    @factory.lazy_attribute
    def roc_auc(self) -> float:
        """Generate ROC AUC score."""
        return factory.Faker("pyfloat", min_value=0.5, max_value=1.0).generate()

    @factory.lazy_attribute
    def pr_auc(self) -> float:
        """Generate Precision-Recall AUC score."""
        return factory.Faker("pyfloat", min_value=0.0, max_value=1.0).generate()

    @classmethod
    def excellent_performance(cls) -> PerformanceMetrics:
        """Create metrics with excellent performance (> 0.9)."""
        return cls(
            precision=factory.Faker("pyfloat", min_value=0.9, max_value=1.0).generate(),
            recall=factory.Faker("pyfloat", min_value=0.9, max_value=1.0).generate(),
            accuracy=factory.Faker("pyfloat", min_value=0.9, max_value=1.0).generate(),
            roc_auc=factory.Faker("pyfloat", min_value=0.9, max_value=1.0).generate(),
        )

    @classmethod
    def poor_performance(cls) -> PerformanceMetrics:
        """Create metrics with poor performance (< 0.6)."""
        return cls(
            precision=factory.Faker("pyfloat", min_value=0.1, max_value=0.6).generate(),
            recall=factory.Faker("pyfloat", min_value=0.1, max_value=0.6).generate(),
            accuracy=factory.Faker("pyfloat", min_value=0.1, max_value=0.6).generate(),
            roc_auc=factory.Faker("pyfloat", min_value=0.5, max_value=0.6).generate(),
        )


class ThresholdConfigFactory(factory.Factory):
    """Factory for creating ThresholdConfig instances."""

    class Meta:
        model = ThresholdConfig

    value = factory.Faker("pyfloat", min_value=0.1, max_value=0.9)
    method = fuzzy.FuzzyChoice(["percentile", "std_deviation", "fixed", "adaptive"])
    
    @factory.lazy_attribute
    def parameters(self) -> dict[str, Any]:
        """Generate threshold parameters based on method."""
        if self.method == "percentile":
            return {
                "percentile": factory.Faker("pyfloat", min_value=80, max_value=99).generate(),
            }
        elif self.method == "std_deviation":
            return {
                "n_std": factory.Faker("pyfloat", min_value=1.0, max_value=3.0).generate(),
            }
        elif self.method == "adaptive":
            return {
                "window_size": factory.Faker("pyint", min_value=10, max_value=100).generate(),
                "update_rate": factory.Faker("pyfloat", min_value=0.01, max_value=0.1).generate(),
            }
        else:  # fixed
            return {}

    @classmethod
    def percentile_threshold(cls, percentile: float = 95.0) -> ThresholdConfig:
        """Create percentile-based threshold."""
        return cls(
            method="percentile",
            parameters={"percentile": percentile}
        )

    @classmethod
    def std_threshold(cls, n_std: float = 2.0) -> ThresholdConfig:
        """Create standard deviation-based threshold."""
        return cls(
            method="std_deviation",
            parameters={"n_std": n_std}
        )

    @classmethod
    def fixed_threshold(cls, value: float = 0.5) -> ThresholdConfig:
        """Create fixed threshold."""
        return cls(
            method="fixed",
            value=value,
            parameters={}
        )

    @classmethod
    def adaptive_threshold(cls) -> ThresholdConfig:
        """Create adaptive threshold."""
        return cls(
            method="adaptive",
            parameters={
                "window_size": factory.Faker("pyint", min_value=50, max_value=200).generate(),
                "update_rate": factory.Faker("pyfloat", min_value=0.01, max_value=0.05).generate(),
            }
        )
