"""Unit tests for classification interfaces."""

import pytest
from unittest.mock import MagicMock
from datetime import datetime
from uuid import uuid4

from pynomaly.domain.services.classification.interfaces import (
    ISeverityClassifier,
    ITypeClassifier,
)
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.severity_level import SeverityLevel
from pynomaly.domain.value_objects.anomaly_type import AnomalyType
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore


class TestSeverityClassifier(ISeverityClassifier):
    """Test implementation of ISeverityClassifier for testing."""
    
    def __init__(self, fixed_severity=SeverityLevel.MEDIUM, fixed_confidence=0.8):
        self.fixed_severity = fixed_severity
        self.fixed_confidence = fixed_confidence
        self.parameters = {}
    
    def classify(self, anomaly: Anomaly) -> SeverityLevel:
        """Return a fixed severity level."""
        if anomaly.score.value > 0.8:
            return SeverityLevel.HIGH
        elif anomaly.score.value > 0.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def get_confidence(self, anomaly: Anomaly) -> float:
        """Return a fixed confidence."""
        return self.fixed_confidence
    
    @property
    def name(self) -> str:
        return "TestSeverityClassifier"
    
    @property
    def description(self) -> str:
        return "A test severity classifier implementation."
    
    def get_parameters(self):
        return self.parameters.copy()
    
    def set_parameters(self, params):
        self.parameters.update(params)
    
    def validate_anomaly(self, anomaly: Anomaly) -> bool:
        return isinstance(anomaly, Anomaly)


class TestTypeClassifier(ITypeClassifier):
    """Test implementation of ITypeClassifier for testing."""
    
    def __init__(self, fixed_type=AnomalyType.UNKNOWN, fixed_confidence=0.7):
        self.fixed_type = fixed_type
        self.fixed_confidence = fixed_confidence
        self.parameters = {}
    
    def categorize(self, anomaly: Anomaly) -> AnomalyType:
        """Return a fixed anomaly type."""
        if "security" in anomaly.metadata:
            return AnomalyType.SECURITY
        elif "performance" in anomaly.metadata:
            return AnomalyType.PERFORMANCE
        else:
            return self.fixed_type
    
    def get_confidence(self, anomaly: Anomaly) -> float:
        """Return a fixed confidence."""
        return self.fixed_confidence
    
    @property
    def name(self) -> str:
        return "TestTypeClassifier"
    
    @property
    def description(self) -> str:
        return "A test type classifier implementation."
    
    def get_parameters(self):
        return self.parameters.copy()
    
    def set_parameters(self, params):
        self.parameters.update(params)
    
    def validate_anomaly(self, anomaly: Anomaly) -> bool:
        return isinstance(anomaly, Anomaly)
    
    def get_supported_types(self):
        return [AnomalyType.UNKNOWN, AnomalyType.SECURITY, AnomalyType.PERFORMANCE]


@pytest.fixture
def sample_anomaly():
    """Create a sample anomaly for testing."""
    return Anomaly(
        id=uuid4(),
        score=AnomalyScore(value=0.75),
        data_point={"feature1": 1.0, "feature2": 2.0},
        detector_name="test_detector",
        timestamp=datetime.now(),
        metadata={"source": "test"},
        anomaly_type=AnomalyType.UNKNOWN,
        severity_level=SeverityLevel.MEDIUM
    )


@pytest.fixture
def severity_classifier():
    """Create a test severity classifier."""
    return TestSeverityClassifier()


@pytest.fixture
def type_classifier():
    """Create a test type classifier."""
    return TestTypeClassifier()


class TestISeverityClassifier:
    """Test cases for ISeverityClassifier interface."""
    
    def test_classify_returns_severity_level(self, severity_classifier, sample_anomaly):
        """Test that classify returns a SeverityLevel."""
        result = severity_classifier.classify(sample_anomaly)
        assert isinstance(result, SeverityLevel)
        assert result == SeverityLevel.MEDIUM  # Based on score 0.75
    
    def test_classify_high_score(self, severity_classifier, sample_anomaly):
        """Test classification with high score."""
        sample_anomaly.score = AnomalyScore(value=0.9)
        result = severity_classifier.classify(sample_anomaly)
        assert result == SeverityLevel.HIGH
    
    def test_classify_low_score(self, severity_classifier, sample_anomaly):
        """Test classification with low score."""
        sample_anomaly.score = AnomalyScore(value=0.3)
        result = severity_classifier.classify(sample_anomaly)
        assert result == SeverityLevel.LOW
    
    def test_get_confidence_returns_float(self, severity_classifier, sample_anomaly):
        """Test that get_confidence returns a float."""
        result = severity_classifier.get_confidence(sample_anomaly)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_name_property(self, severity_classifier):
        """Test that name property returns string."""
        assert isinstance(severity_classifier.name, str)
        assert len(severity_classifier.name) > 0
    
    def test_description_property(self, severity_classifier):
        """Test that description property returns string."""
        assert isinstance(severity_classifier.description, str)
        assert len(severity_classifier.description) > 0
    
    def test_get_parameters_returns_dict(self, severity_classifier):
        """Test that get_parameters returns a dictionary."""
        result = severity_classifier.get_parameters()
        assert isinstance(result, dict)
    
    def test_set_parameters_updates_configuration(self, severity_classifier):
        """Test that set_parameters updates the configuration."""
        params = {"threshold": 0.5, "method": "score_based"}
        severity_classifier.set_parameters(params)
        
        result = severity_classifier.get_parameters()
        assert result["threshold"] == 0.5
        assert result["method"] == "score_based"
    
    def test_validate_anomaly_returns_bool(self, severity_classifier, sample_anomaly):
        """Test that validate_anomaly returns boolean."""
        result = severity_classifier.validate_anomaly(sample_anomaly)
        assert isinstance(result, bool)
        assert result is True
    
    def test_validate_anomaly_with_invalid_input(self, severity_classifier):
        """Test validate_anomaly with invalid input."""
        result = severity_classifier.validate_anomaly("not_an_anomaly")
        assert result is False


class TestITypeClassifier:
    """Test cases for ITypeClassifier interface."""
    
    def test_categorize_returns_anomaly_type(self, type_classifier, sample_anomaly):
        """Test that categorize returns an AnomalyType."""
        result = type_classifier.categorize(sample_anomaly)
        assert isinstance(result, AnomalyType)
        assert result == AnomalyType.UNKNOWN
    
    def test_categorize_with_security_metadata(self, type_classifier, sample_anomaly):
        """Test categorization with security metadata."""
        sample_anomaly.metadata["security"] = True
        result = type_classifier.categorize(sample_anomaly)
        assert result == AnomalyType.SECURITY
    
    def test_categorize_with_performance_metadata(self, type_classifier, sample_anomaly):
        """Test categorization with performance metadata."""
        sample_anomaly.metadata["performance"] = True
        result = type_classifier.categorize(sample_anomaly)
        assert result == AnomalyType.PERFORMANCE
    
    def test_get_confidence_returns_float(self, type_classifier, sample_anomaly):
        """Test that get_confidence returns a float."""
        result = type_classifier.get_confidence(sample_anomaly)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_name_property(self, type_classifier):
        """Test that name property returns string."""
        assert isinstance(type_classifier.name, str)
        assert len(type_classifier.name) > 0
    
    def test_description_property(self, type_classifier):
        """Test that description property returns string."""
        assert isinstance(type_classifier.description, str)
        assert len(type_classifier.description) > 0
    
    def test_get_parameters_returns_dict(self, type_classifier):
        """Test that get_parameters returns a dictionary."""
        result = type_classifier.get_parameters()
        assert isinstance(result, dict)
    
    def test_set_parameters_updates_configuration(self, type_classifier):
        """Test that set_parameters updates the configuration."""
        params = {"confidence_threshold": 0.8, "use_metadata": True}
        type_classifier.set_parameters(params)
        
        result = type_classifier.get_parameters()
        assert result["confidence_threshold"] == 0.8
        assert result["use_metadata"] is True
    
    def test_validate_anomaly_returns_bool(self, type_classifier, sample_anomaly):
        """Test that validate_anomaly returns boolean."""
        result = type_classifier.validate_anomaly(sample_anomaly)
        assert isinstance(result, bool)
        assert result is True
    
    def test_validate_anomaly_with_invalid_input(self, type_classifier):
        """Test validate_anomaly with invalid input."""
        result = type_classifier.validate_anomaly("not_an_anomaly")
        assert result is False
    
    def test_get_supported_types_returns_list(self, type_classifier):
        """Test that get_supported_types returns a list."""
        result = type_classifier.get_supported_types()
        assert isinstance(result, list)
        assert all(isinstance(item, AnomalyType) for item in result)
        assert len(result) > 0
    
    def test_get_supported_types_contains_expected_types(self, type_classifier):
        """Test that get_supported_types contains expected types."""
        result = type_classifier.get_supported_types()
        assert AnomalyType.UNKNOWN in result
        assert AnomalyType.SECURITY in result
        assert AnomalyType.PERFORMANCE in result


class TestAbstractClassBehavior:
    """Test that abstract methods are properly enforced."""
    
    def test_cannot_instantiate_severity_classifier_directly(self):
        """Test that ISeverityClassifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ISeverityClassifier()
    
    def test_cannot_instantiate_type_classifier_directly(self):
        """Test that ITypeClassifier cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ITypeClassifier()
    
    def test_incomplete_severity_classifier_implementation(self):
        """Test that incomplete implementations raise TypeError."""
        class IncompleteSeverityClassifier(ISeverityClassifier):
            def classify(self, anomaly: Anomaly) -> SeverityLevel:
                return SeverityLevel.LOW
            # Missing other abstract methods
        
        with pytest.raises(TypeError):
            IncompleteSeverityClassifier()
    
    def test_incomplete_type_classifier_implementation(self):
        """Test that incomplete implementations raise TypeError."""
        class IncompleteTypeClassifier(ITypeClassifier):
            def categorize(self, anomaly: Anomaly) -> AnomalyType:
                return AnomalyType.UNKNOWN
            # Missing other abstract methods
        
        with pytest.raises(TypeError):
            IncompleteTypeClassifier()


class TestClassifierIntegration:
    """Integration tests for classifiers working together."""
    
    def test_severity_and_type_classification_together(self, severity_classifier, type_classifier, sample_anomaly):
        """Test that severity and type classification work together."""
        # Add metadata to influence type classification
        sample_anomaly.metadata["security"] = True
        sample_anomaly.score = AnomalyScore(value=0.9)
        
        severity = severity_classifier.classify(sample_anomaly)
        anomaly_type = type_classifier.categorize(sample_anomaly)
        
        assert severity == SeverityLevel.HIGH
        assert anomaly_type == AnomalyType.SECURITY
    
    def test_classifier_confidence_scores(self, severity_classifier, type_classifier, sample_anomaly):
        """Test that both classifiers return valid confidence scores."""
        severity_confidence = severity_classifier.get_confidence(sample_anomaly)
        type_confidence = type_classifier.get_confidence(sample_anomaly)
        
        assert 0.0 <= severity_confidence <= 1.0
        assert 0.0 <= type_confidence <= 1.0
    
    def test_classifier_parameter_independence(self, severity_classifier, type_classifier):
        """Test that classifier parameters are independent."""
        severity_classifier.set_parameters({"param1": "value1"})
        type_classifier.set_parameters({"param2": "value2"})
        
        severity_params = severity_classifier.get_parameters()
        type_params = type_classifier.get_parameters()
        
        assert "param1" in severity_params
        assert "param2" in type_params
        assert "param1" not in type_params
        assert "param2" not in severity_params
