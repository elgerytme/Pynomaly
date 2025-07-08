"""Factories for DTOs using factory-boy."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import factory
from factory import fuzzy

from pynomaly.application.dto.detection_dto import (
    DetectionRequestDTO,
    DetectionResultDTO,
    ExplanationRequestDTO,
    TrainingRequestDTO,
    TrainingResultDTO,
)


class DetectionRequestDTOFactory(factory.Factory):
    """Factory for creating DetectionRequestDTO instances."""

    class Meta:
        model = DetectionRequestDTO

    detector_id = factory.LazyFunction(uuid4)
    dataset_id = factory.LazyFunction(uuid4)
    data = factory.LazyFunction(
        lambda: [
            {
                "feature_1": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
                "feature_2": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
                "feature_3": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
            }
            for _ in range(factory.Faker("pyint", min_value=10, max_value=100).generate())
        ]
    )
    threshold = factory.Faker("pyfloat", min_value=0, max_value=1)
    return_scores = True
    return_explanations = False
    validate_features = True
    save_results = True

    @classmethod
    def from_data(cls, data: list[dict[str, Any]], **kwargs) - DetectionRequestDTO:
        """Create DTO from inline data rather than dataset_id."""
        return cls(data=data, dataset_id=None, **kwargs)


class TrainingRequestDTOFactory(factory.Factory):
    """Factory for creating TrainingRequestDTO instances."""

    class Meta:
        model = TrainingRequestDTO

    detector_id = factory.LazyFunction(uuid4)
    dataset_id = factory.LazyFunction(uuid4)
    validation_split = factory.Faker("pyfloat", min_value=0, max_value=0.5)
    cross_validation = False
    save_model = True
    parameters = factory.LazyFunction(
        lambda: {
            "n_estimators": factory.Faker("pyint", min_value=50, max_value=200).generate(),
            "max_samples": "auto",
            "random_state": 42,
        }
    )


class DetectionResultDTOFactory(factory.Factory):
    """Factory for creating DetectionResultDTO instances."""

    class Meta:
        model = DetectionResultDTO

    id = factory.LazyFunction(uuid4)
    detector_id = factory.LazyFunction(uuid4)
    dataset_id = factory.LazyFunction(uuid4)
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    n_samples = factory.Faker("pyint", min_value=100, max_value=1000)
    n_anomalies = factory.Faker("pyint", min_value=1, max_value=50)
    anomaly_rate = factory.Faker("pyfloat", min_value=0, max_value=1)
    threshold = factory.Faker("pyfloat", min_value=0, max_value=1)
    execution_time_ms = factory.Faker("pyfloat", min_value=100, max_value=5000)
    metadata = factory.LazyFunction(
        lambda: {
            "algorithm": factory.Faker("word").generate(),
            "model_version": factory.Faker("pystr", max_chars=10).generate(),
        }
    )
    anomalies = factory.LazyFunction(lambda: [])
    predictions = factory.LazyFunction(lambda: [])
    scores = factory.LazyFunction(lambda: [])
    score_statistics = factory.LazyFunction(
        lambda: {
            "min": factory.Faker("pyfloat", min_value=0, max_value=0.3).generate(),
            "max": factory.Faker("pyfloat", min_value=0.7, max_value=1.0).generate(),
            "mean": factory.Faker("pyfloat", min_value=0.3, max_value=0.7).generate(),
        }
    )

    @classmethod
    def with_anomalies(cls, n_anomalies: int = 5) - DetectionResultDTO:
        """Create DTO with specific number of anomaly DTOs."""
        return cls(n_anomalies=n_anomalies, anomalies=[AnomalyDTOFactory() for _ in range(n_anomalies)])


class TrainingResultDTOFactory(factory.Factory):
    """Factory for creating TrainingResultDTO instances."""

    class Meta:
        model = TrainingResultDTO

    detector_id = factory.LazyFunction(uuid4)
    dataset_id = factory.LazyFunction(uuid4)
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    training_time_ms = factory.Faker("pyfloat", min_value=1000, max_value=10000)
    model_path = None
    training_warnings = factory.LazyFunction(lambda: [])
    training_metrics = factory.LazyFunction(
        lambda: {
            "accuracy": factory.Faker("pyfloat", min_value=0.7, max_value=1.0).generate(),
            "loss": factory.Faker("pyfloat", min_value=0.0, max_value=0.3).generate(),
        }
    )


class ExplanationRequestDTOFactory(factory.Factory):
    """Factory for creating ExplanationRequestDTO instances."""

    class Meta:
        model = ExplanationRequestDTO

    detector_id = factory.LazyFunction(uuid4)
    instance = factory.LazyFunction(
        lambda: {
            "feature_1": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
            "feature_2": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
            "feature_3": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
        }
    )
    method = fuzzy.FuzzyChoice(["shap", "lime"])
    feature_names = factory.LazyFunction(lambda: [f"feature_{i}" for i in range(1, 4)])
    n_features = 10


class AnomalyDTOFactory(factory.Factory):
    """Factory for creating AnomalyDTO instances."""

    class Meta:
        model = AnomalyDTO

    id = factory.LazyFunction(uuid4)
    score = factory.Faker("pyfloat", min_value=0, max_value=1)
    detector_name = factory.Faker("word")
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    data_point = factory.LazyFunction(
        lambda: {
            "feature_1": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
            "feature_2": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
            "feature_3": factory.Faker("pyfloat", min_value=0, max_value=100).generate(),
        }
    )
    metadata = factory.LazyFunction(
        lambda: {
            "detection_method": factory.Faker("word").generate(),
            "detection_time_ms": factory.Faker("pyfloat", min_value=10, max_value=100).generate(),
        }
    )
    explanation = factory.Faker("text", max_nb_chars=200)
    severity = fuzzy.FuzzyChoice(["low", "medium", "high", "critical"])
