"""Unit tests for classification services."""

import pytest
from unittest.mock import MagicMock

from pynomaly.domain.services.classification import (
    ISeverityClassifier,
    ITypeClassifier,
    ClassificationRegistry,
    ClassifierFactory,
)
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.severity_level import SeverityLevel
from pynomaly.domain.value_objects.anomaly_type import AnomalyType


class MockSeverityClassifier(ISeverityClassifier):
    """Mock implementation of a Severity Classifier."""
    
    def classify(self, anomaly: Anomaly) -> SeverityLevel:
        return SeverityLevel.LOW

    def get_confidence(self, anomaly: Anomaly) -> float:
        return 0.9

    @property
    def name(self) -> str:
        return "MockSeverityClassifier"

    @property
    def description(self) -> str:
        return "A mock severity classifier."


class MockTypeClassifier(ITypeClassifier):
    """Mock implementation of a Type Classifier."""
    
    def categorize(self, anomaly: Anomaly) -> AnomalyType:
        return AnomalyType.UNKNOWN

    def get_confidence(self, anomaly: Anomaly) -> float:
        return 0.8

    @property
    def name(self) -> str:
        return "MockTypeClassifier"

    @property
    def description(self) -> str:
        return "A mock type classifier."


@pytest.fixture
def registry():
    return ClassificationRegistry()


@pytest.fixture
def factory(registry):
    return ClassifierFactory(registry)


def test_register_severity_classifier(registry):
    registry.register_severity_classifier("mock", MockSeverityClassifier)
    assert "mock" in registry.list_severity_classifiers()


def test_register_type_classifier(registry):
    registry.register_type_classifier("mock", MockTypeClassifier)
    assert "mock" in registry.list_type_classifiers()


def test_create_severity_classifier(factory):
    factory.registry.register_severity_classifier("mock", MockSeverityClassifier)
    classifier = factory.create_severity_classifier("mock")
    assert isinstance(classifier, MockSeverityClassifier)


def test_create_type_classifier(factory):
    factory.registry.register_type_classifier("mock", MockTypeClassifier)
    classifier = factory.create_type_classifier("mock")
    assert isinstance(classifier, MockTypeClassifier)


def test_severity_classifier_not_registered(factory):
    with pytest.raises(ValueError, match="Severity classifier 'unknown' is not registered"):
        factory.create_severity_classifier("unknown")


def test_type_classifier_not_registered(factory):
    with pytest.raises(ValueError, match="Type classifier 'unknown' is not registered"):
        factory.create_type_classifier("unknown")


def test_global_registry():
    from pynomaly.domain.services.classification.registry import get_global_registry
    global_registry = get_global_registry()
    global_registry.register_severity_classifier("mock", MockSeverityClassifier)
    assert "mock" in global_registry.list_severity_classifiers()


def test_global_factory():
    from pynomaly.domain.services.classification.registry import get_global_factory
    global_factory = get_global_factory()
    global_factory.registry.register_type_classifier("mock", MockTypeClassifier)
    classifier = global_factory.create_type_classifier("mock")
    assert isinstance(classifier, MockTypeClassifier)


def test_validate_classifier_config(factory):
    factory.registry.register_severity_classifier("mock", MockSeverityClassifier)
    assert factory.validate_classifier_config("severity", "mock", {}) is True


def test_invalid_config_validation(factory):
    assert factory.validate_classifier_config("severity", "unknown", {}) is False


def test_registry_info(registry):
    registry.register_severity_classifier("mock", MockSeverityClassifier)
    info = registry.get_classifier_info("mock")
    assert info is not None
    assert info["name"] == "mock"
    assert info["type"] == "severity"
