"""Factories for domain entities using factory-boy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import factory
import pandas as pd
from factory import fuzzy
from tests.factories.value_object_factories import (
    AnomalyScoreFactory,
    ContaminationRateFactory,
)

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.training_result import TrainingResult
from pynomaly.domain.entities.user import User
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate


class AnomalyFactory(factory.Factory):
    """Factory for creating Anomaly instances."""

    class Meta:
        model = Anomaly

    id = factory.LazyFunction(uuid4)
    score = factory.SubFactory(AnomalyScoreFactory)
    data_point = factory.LazyFunction(
        lambda: {
            "feature_1": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
            "feature_2": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
            "feature_3": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
        }
    )
    detector_name = factory.Faker("word")
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    metadata = factory.LazyFunction(
        lambda: {
            "detection_method": factory.Faker("word").generate(),
            "confidence": factory.Faker("pyfloat", min_value=0.5, max_value=1.0).generate(),
        }
    )
    explanation = factory.Faker("text", max_nb_chars=200)

    @factory.lazy_attribute
    def severity(self) -> str:
        """Generate severity based on score."""
        score_value = self.score.value if hasattr(self.score, 'value') else self.score
        if score_value > 0.9:
            return "critical"
        elif score_value > 0.7:
            return "high"
        elif score_value > 0.5:
            return "medium"
        else:
            return "low"


class DatasetFactory(factory.Factory):
    """Factory for creating Dataset instances."""

    class Meta:
        model = Dataset

    id = factory.LazyFunction(uuid4)
    name = factory.Faker("word")
    description = factory.Faker("text", max_nb_chars=100)
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    metadata = factory.LazyFunction(
        lambda: {
            "source": factory.Faker("word").generate(),
            "version": factory.Faker("pystr", max_chars=10).generate(),
        }
    )

    @factory.lazy_attribute
    def data(self) -> pd.DataFrame:
        """Generate sample DataFrame for testing."""
        return pd.DataFrame({
            "feature_1": factory.Faker("pyfloat", min_value=0, max_value=100).generate({}) for _ in range(100),
            "feature_2": factory.Faker("pyfloat", min_value=0, max_value=100).generate({}) for _ in range(100),
            "feature_3": factory.Faker("pyfloat", min_value=0, max_value=100).generate({}) for _ in range(100),
        })

    @classmethod
    def with_size(cls, n_samples: int = 100, n_features: int = 3) -> Dataset:
        """Create dataset with specific dimensions."""
        data = pd.DataFrame({
            f"feature_{i}": [
                factory.Faker("pyfloat", min_value=0, max_value=100).generate({}) 
                for _ in range(n_samples)
            ]
            for i in range(n_features)
        })
        return cls(data=data)

    @classmethod
    def anomaly_dataset(cls, contamination_rate: float = 0.1) -> Dataset:
        """Create dataset with known anomalies."""
        n_samples = 1000
        n_anomalies = int(n_samples * contamination_rate)
        
        # Normal data
        normal_data = pd.DataFrame({
            "feature_1": factory.Faker("pyfloat", min_value=0, max_value=10).generate({}) for _ in range(n_samples - n_anomalies),
            "feature_2": factory.Faker("pyfloat", min_value=0, max_value=10).generate({}) for _ in range(n_samples - n_anomalies),
        })
        
        # Anomalous data (outliers)
        anomaly_data = pd.DataFrame({
            "feature_1": factory.Faker("pyfloat", min_value=50, max_value=100).generate({}) for _ in range(n_anomalies),
            "feature_2": factory.Faker("pyfloat", min_value=50, max_value=100).generate({}) for _ in range(n_anomalies),
        })
        
        # Combine and shuffle
        combined_data = pd.concat([normal_data, anomaly_data], ignore_index=True)
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)
        
        return cls(data=combined_data, name="anomaly_test_dataset")


class DetectorFactory(factory.Factory):
    """Factory for creating Detector instances."""

    class Meta:
        model = Detector

    id = factory.LazyFunction(uuid4)
    name = factory.Faker("word")
    algorithm_name = fuzzy.FuzzyChoice([
        "IsolationForest", "LocalOutlierFactor", "OneClassSVM", 
        "EllipticEnvelope", "DBSCAN", "AutoEncoder"
    ])
    contamination_rate = factory.SubFactory(ContaminationRateFactory)
    parameters = factory.LazyFunction(
        lambda: {
            "n_estimators": factory.Faker("pyint", min_value=50, max_value=200).generate(),
            "max_samples": factory.Faker("pyfloat", min_value=0.1, max_value=1.0).generate(),
            "random_state": 42,
        }
    )
    metadata = factory.LazyFunction(
        lambda: {
            "supports_streaming": factory.Faker("pybool").generate(),
            "supports_multivariate": True,
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
        }
    )
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    trained_at = None
    is_fitted = False

    @classmethod
    def fitted(cls, **kwargs) -> Detector:
        """Create a fitted detector."""
        return cls(
            is_fitted=True,
            trained_at=datetime.now(timezone.utc),
            **kwargs
        )

    @classmethod
    def with_algorithm(cls, algorithm_name: str, **kwargs) -> Detector:
        """Create detector with specific algorithm."""
        return cls(algorithm_name=algorithm_name, **kwargs)


class DetectionResultFactory(factory.Factory):
    """Factory for creating DetectionResult instances."""

    class Meta:
        model = DetectionResult

    id = factory.LazyFunction(uuid4)
    detector_id = factory.LazyFunction(uuid4)
    dataset_id = factory.LazyFunction(uuid4)
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    threshold = factory.Faker("pyfloat", min_value=0.1, max_value=0.9)
    execution_time_ms = factory.Faker("pyfloat", min_value=100, max_value=5000)
    metadata = factory.LazyFunction(
        lambda: {
            "algorithm": factory.Faker("word").generate(),
            "n_samples": factory.Faker("pyint", min_value=100, max_value=1000).generate(),
            "n_features": factory.Faker("pyint", min_value=2, max_value=10).generate(),
        }
    )

    @factory.lazy_attribute
    def anomalies(self) -> list[Anomaly]:
        """Generate list of anomalies."""
        n_anomalies = factory.Faker("pyint", min_value=1, max_value=10).generate()
        return AnomalyFactory.create_batch(n_anomalies)

    @factory.lazy_attribute
    def scores(self) -> list[AnomalyScore]:
        """Generate anomaly scores."""
        n_scores = factory.Faker("pyint", min_value=100, max_value=1000).generate()
        return AnomalyScoreFactory.create_batch(n_scores)

    @factory.lazy_attribute
    def labels(self) -> list[int]:
        """Generate binary labels (0=normal, 1=anomaly)."""
        n_labels = len(self.scores) if hasattr(self, 'scores') else 100
        return [factory.Faker("pyint", min_value=0, max_value=1).generate() for _ in range(n_labels)]

    @classmethod
    def with_anomalies(cls, n_anomalies: int = 5) -> DetectionResult:
        """Create result with specific number of anomalies."""
        anomalies = AnomalyFactory.create_batch(n_anomalies)
        return cls(anomalies=anomalies)

    @classmethod
    def no_anomalies(cls) -> DetectionResult:
        """Create result with no anomalies detected."""
        return cls(anomalies=[], scores=[], labels=[])


class TrainingResultFactory(factory.Factory):
    """Factory for creating TrainingResult instances."""

    class Meta:
        model = TrainingResult

    id = factory.LazyFunction(uuid4)
    detector_id = factory.LazyFunction(uuid4)
    dataset_id = factory.LazyFunction(uuid4)
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    training_duration = factory.Faker("pyfloat", min_value=1.0, max_value=300.0)
    is_successful = True
    error_message = None
    
    @factory.lazy_attribute
    def metrics(self) -> dict[str, Any]:
        """Generate training metrics."""
        return {
            "n_samples": factory.Faker("pyint", min_value=100, max_value=1000).generate(),
            "n_features": factory.Faker("pyint", min_value=2, max_value=10).generate(),
            "training_time": self.training_duration,
            "mean_score": factory.Faker("pyfloat", min_value=0.0, max_value=1.0).generate(),
            "max_score": factory.Faker("pyfloat", min_value=0.5, max_value=1.0).generate(),
            "min_score": factory.Faker("pyfloat", min_value=0.0, max_value=0.5).generate(),
        }

    algorithm = factory.Faker("word")
    contamination_rate = factory.Faker("pyfloat", min_value=0.05, max_value=0.3)

    @classmethod
    def success_result(cls, **kwargs) -> TrainingResult:
        """Create successful training result."""
        return cls(is_successful=True, error_message=None, **kwargs)

    @classmethod
    def failure_result(cls, **kwargs) -> TrainingResult:
        """Create failed training result."""
        return cls(
            is_successful=False,
            error_message=factory.Faker("text", max_nb_chars=100).generate(),
            **kwargs
        )


class UserFactory(factory.Factory):
    """Factory for creating User instances."""

    class Meta:
        model = User

    id = factory.LazyFunction(uuid4)
    username = factory.Faker("user_name")
    email = factory.Faker("email")
    full_name = factory.Faker("name")
    is_active = True
    is_admin = False
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    metadata = factory.LazyFunction(
        lambda: {
            "last_login": factory.Faker("date_time").generate().isoformat(),
            "preferences": {
                "theme": factory.Faker("word").generate(),
                "notifications": factory.Faker("pybool").generate(),
            },
        }
    )

    @classmethod
    def admin(cls, **kwargs) -> User:
        """Create admin user."""
        return cls(is_admin=True, **kwargs)

    @classmethod
    def inactive(cls, **kwargs) -> User:
        """Create inactive user."""
        return cls(is_active=False, **kwargs)
