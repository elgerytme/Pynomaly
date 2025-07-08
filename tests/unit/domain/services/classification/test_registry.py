"""Unit tests for classification registry and factory system."""

import pytest
from threading import Thread
from unittest.mock import patch, MagicMock

from pynomaly.domain.services.classification.registry import (
    ClassificationRegistry,
    ClassifierFactory,
    get_global_registry,
    get_global_factory,
    register_severity_classifier,
    register_type_classifier,
)
from pynomaly.domain.services.classification.interfaces import (
    ISeverityClassifier,
    ITypeClassifier,
)
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.severity_level import SeverityLevel
from pynomaly.domain.value_objects.anomaly_type import AnomalyType


class MockSeverityClassifier(ISeverityClassifier):
    """Mock severity classifier for testing."""
    
    def __init__(self, name="MockSeverity", confidence=0.8):
        self._name = name
        self._confidence = confidence
        self.parameters = {}
    
    def classify(self, anomaly: Anomaly) -> SeverityLevel:
        return SeverityLevel.MEDIUM
    
    def get_confidence(self, anomaly: Anomaly) -> float:
        return self._confidence
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return f"Mock severity classifier: {self._name}"
    
    def get_parameters(self):
        return self.parameters.copy()
    
    def set_parameters(self, params):
        self.parameters.update(params)


class MockTypeClassifier(ITypeClassifier):
    """Mock type classifier for testing."""
    
    def __init__(self, name="MockType", confidence=0.7):
        self._name = name
        self._confidence = confidence
        self.parameters = {}
    
    def categorize(self, anomaly: Anomaly) -> AnomalyType:
        return AnomalyType.UNKNOWN
    
    def get_confidence(self, anomaly: Anomaly) -> float:
        return self._confidence
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return f"Mock type classifier: {self._name}"
    
    def get_parameters(self):
        return self.parameters.copy()
    
    def set_parameters(self, params):
        self.parameters.update(params)


class InvalidClassifier:
    """Invalid classifier that doesn't implement the interface."""
    pass


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return ClassificationRegistry()


@pytest.fixture
def factory():
    """Create a factory with fresh registry for each test."""
    return ClassifierFactory()


class TestClassificationRegistry:
    """Test cases for ClassificationRegistry."""
    
    def test_register_severity_classifier_success(self, registry):
        """Test successful registration of severity classifier."""
        registry.register_severity_classifier("mock", MockSeverityClassifier)
        
        assert "mock" in registry.list_severity_classifiers()
        assert registry.get_severity_classifier("mock") == MockSeverityClassifier
    
    def test_register_type_classifier_success(self, registry):
        """Test successful registration of type classifier."""
        registry.register_type_classifier("mock", MockTypeClassifier)
        
        assert "mock" in registry.list_type_classifiers()
        assert registry.get_type_classifier("mock") == MockTypeClassifier
    
    def test_register_severity_classifier_invalid_class(self, registry):
        """Test registration with invalid classifier class."""
        with pytest.raises(ValueError, match="must implement ISeverityClassifier"):
            registry.register_severity_classifier("invalid", InvalidClassifier)
    
    def test_register_type_classifier_invalid_class(self, registry):
        """Test registration with invalid classifier class."""
        with pytest.raises(ValueError, match="must implement ITypeClassifier"):
            registry.register_type_classifier("invalid", InvalidClassifier)
    
    def test_register_duplicate_severity_classifier(self, registry):
        """Test registering duplicate severity classifier name."""
        registry.register_severity_classifier("mock", MockSeverityClassifier)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register_severity_classifier("mock", MockSeverityClassifier)
    
    def test_register_duplicate_type_classifier(self, registry):
        """Test registering duplicate type classifier name."""
        registry.register_type_classifier("mock", MockTypeClassifier)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register_type_classifier("mock", MockTypeClassifier)
    
    def test_get_nonexistent_severity_classifier(self, registry):
        """Test getting non-existent severity classifier."""
        result = registry.get_severity_classifier("nonexistent")
        assert result is None
    
    def test_get_nonexistent_type_classifier(self, registry):
        """Test getting non-existent type classifier."""
        result = registry.get_type_classifier("nonexistent")
        assert result is None
    
    def test_list_empty_classifiers(self, registry):
        """Test listing classifiers when registry is empty."""
        assert registry.list_severity_classifiers() == []
        assert registry.list_type_classifiers() == []
    
    def test_list_multiple_classifiers(self, registry):
        """Test listing multiple classifiers."""
        registry.register_severity_classifier("mock1", MockSeverityClassifier)
        registry.register_severity_classifier("mock2", MockSeverityClassifier)
        registry.register_type_classifier("type1", MockTypeClassifier)
        
        severity_list = registry.list_severity_classifiers()
        type_list = registry.list_type_classifiers()
        
        assert len(severity_list) == 2
        assert "mock1" in severity_list
        assert "mock2" in severity_list
        assert len(type_list) == 1
        assert "type1" in type_list
    
    def test_unregister_severity_classifier(self, registry):
        """Test unregistering severity classifier."""
        registry.register_severity_classifier("mock", MockSeverityClassifier)
        
        result = registry.unregister_severity_classifier("mock")
        assert result is True
        assert "mock" not in registry.list_severity_classifiers()
    
    def test_unregister_nonexistent_severity_classifier(self, registry):
        """Test unregistering non-existent severity classifier."""
        result = registry.unregister_severity_classifier("nonexistent")
        assert result is False
    
    def test_unregister_type_classifier(self, registry):
        """Test unregistering type classifier."""
        registry.register_type_classifier("mock", MockTypeClassifier)
        
        result = registry.unregister_type_classifier("mock")
        assert result is True
        assert "mock" not in registry.list_type_classifiers()
    
    def test_unregister_nonexistent_type_classifier(self, registry):
        """Test unregistering non-existent type classifier."""
        result = registry.unregister_type_classifier("nonexistent")
        assert result is False
    
    def test_clear_all_classifiers(self, registry):
        """Test clearing all classifiers."""
        registry.register_severity_classifier("mock1", MockSeverityClassifier)
        registry.register_type_classifier("mock2", MockTypeClassifier)
        
        registry.clear_all()
        
        assert registry.list_severity_classifiers() == []
        assert registry.list_type_classifiers() == []
    
    def test_get_classifier_info_severity(self, registry):
        """Test getting classifier info for severity classifier."""
        registry.register_severity_classifier("mock", MockSeverityClassifier)
        
        info = registry.get_classifier_info("mock")
        assert info is not None
        assert info["name"] == "mock"
        assert info["type"] == "severity"
        assert info["class"] == MockSeverityClassifier
    
    def test_get_classifier_info_type(self, registry):
        """Test getting classifier info for type classifier."""
        registry.register_type_classifier("mock", MockTypeClassifier)
        
        info = registry.get_classifier_info("mock")
        assert info is not None
        assert info["name"] == "mock"
        assert info["type"] == "type"
        assert info["class"] == MockTypeClassifier
    
    def test_get_classifier_info_nonexistent(self, registry):
        """Test getting classifier info for non-existent classifier."""
        info = registry.get_classifier_info("nonexistent")
        assert info is None
    
    def test_thread_safety(self, registry):
        """Test thread-safe operations."""
        results = []
        
        def register_classifier(name):
            try:
                registry.register_severity_classifier(f"mock_{name}", MockSeverityClassifier)
                results.append(f"success_{name}")
            except Exception as e:
                results.append(f"error_{name}")
        
        threads = []
        for i in range(10):
            thread = Thread(target=register_classifier, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All registrations should succeed
        success_count = sum(1 for r in results if r.startswith("success"))
        assert success_count == 10
        assert len(registry.list_severity_classifiers()) == 10


class TestClassifierFactory:
    """Test cases for ClassifierFactory."""
    
    def test_factory_initialization_with_registry(self):
        """Test factory initialization with custom registry."""
        custom_registry = ClassificationRegistry()
        factory = ClassifierFactory(custom_registry)
        
        assert factory.registry is custom_registry
    
    def test_factory_initialization_without_registry(self):
        """Test factory initialization without registry."""
        factory = ClassifierFactory()
        
        assert isinstance(factory.registry, ClassificationRegistry)
    
    def test_create_severity_classifier_success(self, factory):
        """Test successful creation of severity classifier."""
        factory.registry.register_severity_classifier("mock", MockSeverityClassifier)
        
        classifier = factory.create_severity_classifier("mock")
        
        assert isinstance(classifier, MockSeverityClassifier)
        assert classifier.name == "MockSeverity"
    
    def test_create_type_classifier_success(self, factory):
        """Test successful creation of type classifier."""
        factory.registry.register_type_classifier("mock", MockTypeClassifier)
        
        classifier = factory.create_type_classifier("mock")
        
        assert isinstance(classifier, MockTypeClassifier)
        assert classifier.name == "MockType"
    
    def test_create_severity_classifier_with_kwargs(self, factory):
        """Test creating severity classifier with custom arguments."""
        factory.registry.register_severity_classifier("mock", MockSeverityClassifier)
        
        classifier = factory.create_severity_classifier("mock", name="CustomName", confidence=0.9)
        
        assert classifier.name == "CustomName"
        assert classifier._confidence == 0.9
    
    def test_create_type_classifier_with_kwargs(self, factory):
        """Test creating type classifier with custom arguments."""
        factory.registry.register_type_classifier("mock", MockTypeClassifier)
        
        classifier = factory.create_type_classifier("mock", name="CustomType", confidence=0.6)
        
        assert classifier.name == "CustomType"
        assert classifier._confidence == 0.6
    
    def test_create_severity_classifier_not_registered(self, factory):
        """Test creating non-registered severity classifier."""
        with pytest.raises(ValueError, match="not registered"):
            factory.create_severity_classifier("nonexistent")
    
    def test_create_type_classifier_not_registered(self, factory):
        """Test creating non-registered type classifier."""
        with pytest.raises(ValueError, match="not registered"):
            factory.create_type_classifier("nonexistent")
    
    def test_create_severity_classifier_construction_error(self, factory):
        """Test creation failure due to constructor error."""
        class FailingClassifier(ISeverityClassifier):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Constructor failed")
            
            def classify(self, anomaly): pass
            def get_confidence(self, anomaly): pass
            @property
            def name(self): pass
            @property
            def description(self): pass
        
        factory.registry.register_severity_classifier("failing", FailingClassifier)
        
        with pytest.raises(Exception, match="Failed to create severity classifier"):
            factory.create_severity_classifier("failing")
    
    def test_create_classifier_with_config_severity(self, factory):
        """Test creating classifier with configuration."""
        factory.registry.register_severity_classifier("mock", MockSeverityClassifier)
        
        config = {"threshold": 0.8, "method": "test"}
        classifier = factory.create_classifier_with_config("severity", "mock", config)
        
        assert isinstance(classifier, MockSeverityClassifier)
        assert classifier.get_parameters() == config
    
    def test_create_classifier_with_config_type(self, factory):
        """Test creating type classifier with configuration."""
        factory.registry.register_type_classifier("mock", MockTypeClassifier)
        
        config = {"use_metadata": True, "confidence_threshold": 0.7}
        classifier = factory.create_classifier_with_config("type", "mock", config)
        
        assert isinstance(classifier, MockTypeClassifier)
        assert classifier.get_parameters() == config
    
    def test_create_classifier_with_config_invalid_type(self, factory):
        """Test creating classifier with invalid type."""
        with pytest.raises(ValueError, match="Invalid classifier type"):
            factory.create_classifier_with_config("invalid", "mock", {})
    
    def test_validate_classifier_config_valid(self, factory):
        """Test validating valid classifier configuration."""
        factory.registry.register_severity_classifier("mock", MockSeverityClassifier)
        
        result = factory.validate_classifier_config("severity", "mock", {})
        assert result is True
    
    def test_validate_classifier_config_invalid(self, factory):
        """Test validating invalid classifier configuration."""
        result = factory.validate_classifier_config("severity", "nonexistent", {})
        assert result is False
    
    def test_get_available_classifiers(self, factory):
        """Test getting available classifiers."""
        factory.registry.register_severity_classifier("severity1", MockSeverityClassifier)
        factory.registry.register_severity_classifier("severity2", MockSeverityClassifier)
        factory.registry.register_type_classifier("type1", MockTypeClassifier)
        
        available = factory.get_available_classifiers()
        
        assert "severity" in available
        assert "type" in available
        assert len(available["severity"]) == 2
        assert len(available["type"]) == 1
        assert "severity1" in available["severity"]
        assert "severity2" in available["severity"]
        assert "type1" in available["type"]


class TestGlobalRegistryAndFactory:
    """Test cases for global registry and factory functions."""
    
    def test_get_global_registry(self):
        """Test getting global registry."""
        registry = get_global_registry()
        
        assert isinstance(registry, ClassificationRegistry)
        
        # Should return the same instance
        registry2 = get_global_registry()
        assert registry is registry2
    
    def test_get_global_factory(self):
        """Test getting global factory."""
        factory = get_global_factory()
        
        assert isinstance(factory, ClassifierFactory)
        
        # Should return the same instance
        factory2 = get_global_factory()
        assert factory is factory2
    
    def test_global_registry_and_factory_connection(self):
        """Test that global factory uses global registry."""
        factory = get_global_factory()
        registry = get_global_registry()
        
        assert factory.registry is registry
    
    def test_register_severity_classifier_global(self):
        """Test registering severity classifier globally."""
        register_severity_classifier("global_mock", MockSeverityClassifier)
        
        registry = get_global_registry()
        assert "global_mock" in registry.list_severity_classifiers()
    
    def test_register_type_classifier_global(self):
        """Test registering type classifier globally."""
        register_type_classifier("global_mock", MockTypeClassifier)
        
        registry = get_global_registry()
        assert "global_mock" in registry.list_type_classifiers()
    
    def test_global_factory_can_create_globally_registered_classifiers(self):
        """Test that global factory can create globally registered classifiers."""
        register_severity_classifier("global_severity", MockSeverityClassifier)
        register_type_classifier("global_type", MockTypeClassifier)
        
        factory = get_global_factory()
        
        severity_classifier = factory.create_severity_classifier("global_severity")
        type_classifier = factory.create_type_classifier("global_type")
        
        assert isinstance(severity_classifier, MockSeverityClassifier)
        assert isinstance(type_classifier, MockTypeClassifier)


class TestRegistryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_classifier_name(self, registry):
        """Test registering classifier with empty name."""
        # This should work, but might not be desirable
        registry.register_severity_classifier("", MockSeverityClassifier)
        assert "" in registry.list_severity_classifiers()
    
    def test_none_classifier_name(self, registry):
        """Test registering classifier with None name."""
        with pytest.raises(TypeError):
            registry.register_severity_classifier(None, MockSeverityClassifier)
    
    def test_none_classifier_class(self, registry):
        """Test registering None as classifier class."""
        with pytest.raises(TypeError):
            registry.register_severity_classifier("mock", None)
    
    def test_classifier_with_same_name_different_types(self, registry):
        """Test registering classifiers with same name but different types."""
        registry.register_severity_classifier("same_name", MockSeverityClassifier)
        registry.register_type_classifier("same_name", MockTypeClassifier)
        
        # Should work fine as they're in different registries
        assert "same_name" in registry.list_severity_classifiers()
        assert "same_name" in registry.list_type_classifiers()
    
    def test_classifier_info_with_same_name_different_types(self, registry):
        """Test getting classifier info when same name exists in both registries."""
        registry.register_severity_classifier("same_name", MockSeverityClassifier)
        registry.register_type_classifier("same_name", MockTypeClassifier)
        
        # Should return the severity classifier info first
        info = registry.get_classifier_info("same_name")
        assert info["type"] == "severity"
    
    def test_factory_with_broken_set_parameters(self, factory):
        """Test factory with classifier that has broken set_parameters."""
        class BrokenParameterClassifier(ISeverityClassifier):
            def classify(self, anomaly): return SeverityLevel.LOW
            def get_confidence(self, anomaly): return 0.5
            @property
            def name(self): return "broken"
            @property
            def description(self): return "broken"
            
            def set_parameters(self, params):
                raise RuntimeError("Broken set_parameters")
        
        factory.registry.register_severity_classifier("broken", BrokenParameterClassifier)
        
        # Should still work if set_parameters fails
        classifier = factory.create_classifier_with_config("severity", "broken", {})
        assert isinstance(classifier, BrokenParameterClassifier)
