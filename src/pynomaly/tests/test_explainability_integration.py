"""Integration tests for completed explainability service."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from uuid import uuid4

from pynomaly.domain.services.explainability_service import (
    ExplainabilityService,
    ExplanationMethod,
    LocalExplanation,
    GlobalExplanation
)
from pynomaly.infrastructure.config import Settings
from pynomaly.infrastructure.explainers.explainer_factory import (
    ExplainerFactory,
    ExplainerManager,
    ExplainerConfiguration
)
from pynomaly.infrastructure.explainers.service_initialization import (
    initialize_explainability_service,
    create_explainer_for_model,
    get_explainability_service
)


class MockModel:
    """Mock model for testing."""
    
    def predict(self, X):
        """Mock predict method."""
        return np.random.random(len(X))
    
    def predict_proba(self, X):
        """Mock predict_proba method."""
        probs = np.random.random((len(X), 2))
        return probs / probs.sum(axis=1, keepdims=True)


class TestExplainerFactory:
    """Test explainer factory functionality."""
    
    def test_explainer_configuration_creation(self):
        """Test explainer configuration with custom settings."""
        settings = Settings()
        config = ExplainerConfiguration(settings)
        
        # Verify configuration loaded correctly
        assert isinstance(config.shap_config, dict)
        assert isinstance(config.lime_config, dict)
        assert isinstance(config.performance_config, dict)
        
        # Check specific configuration values
        assert config.shap_config["algorithm"] == settings.explainability.shap_algorithm
        assert config.lime_config["num_features"] == settings.explainability.max_features
        assert config.performance_config["enable_caching"] == settings.explainability.enable_caching
    
    def test_explainer_factory_creation(self):
        """Test explainer factory creation."""
        factory = ExplainerFactory()
        
        # Check factory initialization
        assert factory.config is not None
        assert len(factory._explainer_classes) > 0
        assert ExplanationMethod.SHAP in factory._explainer_classes
        assert ExplanationMethod.LIME in factory._explainer_classes
    
    def test_explainer_factory_with_custom_config(self):
        """Test explainer factory with custom configuration."""
        settings = Settings()
        settings.explainability.max_features = 5
        settings.explainability.background_samples = 50
        
        config = ExplainerConfiguration(settings)
        factory = ExplainerFactory(config)
        
        # Verify custom configuration is applied
        assert config.shap_config["n_background_samples"] == 50
        assert config.lime_config["num_features"] == 5
    
    @patch('pynomaly.infrastructure.explainers.shap_explainer.SHAPExplainer')
    def test_explainer_creation_shap(self, mock_shap_class):
        """Test SHAP explainer creation."""
        mock_explainer = Mock()
        mock_shap_class.return_value = mock_explainer
        
        factory = ExplainerFactory()
        explainer = factory.create_explainer(ExplanationMethod.SHAP)
        
        # Verify explainer was created
        assert explainer == mock_explainer
        mock_shap_class.assert_called_once()
    
    @patch('pynomaly.infrastructure.explainers.lime_explainer.LIMEExplainer')
    def test_explainer_creation_lime(self, mock_lime_class):
        """Test LIME explainer creation."""
        mock_explainer = Mock()
        mock_lime_class.return_value = mock_explainer
        
        factory = ExplainerFactory()
        explainer = factory.create_explainer(ExplanationMethod.LIME)
        
        # Verify explainer was created
        assert explainer == mock_explainer
        mock_lime_class.assert_called_once()
    
    def test_explainer_creation_unsupported_method(self):
        """Test error handling for unsupported explanation method."""
        factory = ExplainerFactory()
        
        # Should raise error for unsupported method
        with pytest.raises(ValueError, match="Unsupported explanation method"):
            factory.create_explainer("unsupported_method")
    
    @patch('pynomaly.infrastructure.explainers.shap_explainer.SHAPExplainer')
    def test_explainer_caching(self, mock_shap_class):
        """Test explainer caching functionality."""
        mock_explainer = Mock()
        mock_shap_class.return_value = mock_explainer
        
        factory = ExplainerFactory()
        
        # Create explainer twice
        explainer1 = factory.create_explainer(ExplanationMethod.SHAP)
        explainer2 = factory.create_explainer(ExplanationMethod.SHAP)
        
        # Should return same cached instance
        assert explainer1 == explainer2
        mock_shap_class.assert_called_once()  # Only called once due to caching


class TestExplainerManager:
    """Test explainer manager functionality."""
    
    @patch('pynomaly.infrastructure.explainers.explainer_factory.ExplainerFactory')
    def test_explainer_manager_initialization(self, mock_factory_class):
        """Test explainer manager initialization."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        manager = ExplainerManager()
        
        # Verify manager initialized correctly
        assert manager.factory == mock_factory
        assert manager.settings is not None
        assert isinstance(manager._active_explainers, dict)
    
    @patch('pynomaly.infrastructure.explainers.explainer_factory.ExplainerFactory')
    def test_get_explainer_for_model(self, mock_factory_class):
        """Test getting explainer for specific model."""
        mock_factory = Mock()
        mock_explainer = Mock()
        mock_factory.create_explainer.return_value = mock_explainer
        mock_factory_class.return_value = mock_factory
        
        manager = ExplainerManager()
        model_id = "test_model_123"
        
        explainer = manager.get_explainer_for_model(
            model_id=model_id,
            method=ExplanationMethod.SHAP
        )
        
        # Verify explainer was created and cached
        assert explainer == mock_explainer
        mock_factory.create_explainer.assert_called_once_with(
            method=ExplanationMethod.SHAP,
            model_type=None,
            custom_config=None
        )
        assert model_id in manager._active_explainers
        assert ExplanationMethod.SHAP in manager._active_explainers[model_id]
    
    @patch('pynomaly.infrastructure.explainers.explainer_factory.ExplainerFactory')
    def test_initialize_explainers_for_model(self, mock_factory_class):
        """Test initializing all explainers for a model."""
        mock_factory = Mock()
        mock_factory.get_supported_methods.return_value = [
            ExplanationMethod.SHAP,
            ExplanationMethod.LIME
        ]
        mock_factory.create_explainer.return_value = Mock()
        mock_factory_class.return_value = mock_factory
        
        manager = ExplainerManager()
        model_id = "test_model_456"
        
        explainers = manager.initialize_explainers_for_model(model_id)
        
        # Verify all explainers were created
        assert len(explainers) == 2
        assert ExplanationMethod.SHAP in explainers
        assert ExplanationMethod.LIME in explainers
        assert mock_factory.create_explainer.call_count == 2
    
    @patch('pynomaly.infrastructure.explainers.explainer_factory.ExplainerFactory')
    def test_cleanup_model_explainers(self, mock_factory_class):
        """Test cleaning up explainers for a model."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        manager = ExplainerManager()
        model_id = "test_model_789"
        
        # Add some explainers
        manager._active_explainers[model_id] = {
            ExplanationMethod.SHAP: Mock(),
            ExplanationMethod.LIME: Mock()
        }
        
        # Cleanup
        manager.cleanup_model_explainers(model_id)
        
        # Verify cleanup
        assert model_id not in manager._active_explainers
    
    @patch('pynomaly.infrastructure.explainers.explainer_factory.ExplainerFactory')
    def test_get_explainer_stats(self, mock_factory_class):
        """Test getting explainer statistics."""
        mock_factory = Mock()
        mock_factory.get_supported_methods.return_value = [
            ExplanationMethod.SHAP,
            ExplanationMethod.LIME
        ]
        mock_factory_class.return_value = mock_factory
        
        manager = ExplainerManager()
        
        # Add some test data
        manager._active_explainers = {
            "model_1": {ExplanationMethod.SHAP: Mock()},
            "model_2": {ExplanationMethod.SHAP: Mock(), ExplanationMethod.LIME: Mock()}
        }
        
        stats = manager.get_explainer_stats()
        
        # Verify stats
        assert stats["total_models"] == 2
        assert stats["total_explainers"] == 3
        assert stats["method_distribution"]["shap"] == 2
        assert stats["method_distribution"]["lime"] == 1


class TestServiceInitialization:
    """Test service initialization functionality."""
    
    @patch('pynomaly.infrastructure.explainers.service_initialization.configure_explainers')
    def test_initialize_explainability_service(self, mock_configure):
        """Test explainability service initialization."""
        mock_manager = Mock()
        mock_configure.return_value = mock_manager
        
        settings = Settings()
        service = initialize_explainability_service(settings)
        
        # Verify service was initialized
        assert isinstance(service, ExplainabilityService)
        mock_configure.assert_called_once_with(settings)
    
    @patch('pynomaly.infrastructure.explainers.service_initialization.configure_explainers')
    def test_create_explainer_for_model(self, mock_configure):
        """Test creating explainer for specific model."""
        mock_manager = Mock()
        mock_explainer = Mock()
        mock_manager.get_explainer_for_model.return_value = mock_explainer
        mock_configure.return_value = mock_manager
        
        model = MockModel()
        explainer = create_explainer_for_model(
            model=model,
            method=ExplanationMethod.SHAP,
            model_type="tree"
        )
        
        # Verify explainer was created
        assert explainer == mock_explainer
        mock_manager.get_explainer_for_model.assert_called_once()
    
    def test_create_explainer_for_model_disabled(self):
        """Test error when explainability is disabled."""
        settings = Settings()
        settings.explainability.enable_explainability = False
        
        model = MockModel()
        
        with pytest.raises(ValueError, match="Explainability is disabled"):
            create_explainer_for_model(model=model, settings=settings)


class TestExplainabilityServiceIntegration:
    """Test complete explainability service integration."""
    
    def test_explainability_service_domain_methods(self):
        """Test domain service methods."""
        service = ExplainabilityService()
        
        # Test initial state
        assert len(service.get_available_methods()) == 0
        
        # Register mock explainer
        mock_explainer = Mock()
        service.register_explainer(ExplanationMethod.SHAP, mock_explainer)
        
        # Verify registration
        available_methods = service.get_available_methods()
        assert len(available_methods) == 1
        assert ExplanationMethod.SHAP in available_methods
    
    def test_explainability_service_explain_instance(self):
        """Test instance explanation functionality."""
        service = ExplainabilityService()
        
        # Create mock explainer
        mock_explainer = Mock()
        mock_explanation = LocalExplanation(
            instance_id="test_instance",
            method=ExplanationMethod.SHAP,
            feature_contributions=[],
            prediction_confidence=0.85,
            explanation_confidence=0.90
        )
        mock_explainer.explain_local.return_value = mock_explanation
        
        # Register explainer
        service.register_explainer(ExplanationMethod.SHAP, mock_explainer)
        
        # Test explanation
        instance = np.array([1, 2, 3, 4, 5])
        model = MockModel()
        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        
        explanation = service.explain_instance(
            instance=instance,
            model=model,
            feature_names=feature_names,
            method=ExplanationMethod.SHAP
        )
        
        # Verify explanation
        assert explanation == mock_explanation
        mock_explainer.explain_local.assert_called_once()
    
    def test_explainability_service_explain_model(self):
        """Test global model explanation functionality."""
        service = ExplainabilityService()
        
        # Create mock explainer
        mock_explainer = Mock()
        mock_explanation = GlobalExplanation(
            model_id="test_model",
            method=ExplanationMethod.SHAP,
            feature_importance={},
            model_performance={},
            explanation_metadata={}
        )
        mock_explainer.explain_global.return_value = mock_explanation
        
        # Register explainer
        service.register_explainer(ExplanationMethod.SHAP, mock_explainer)
        
        # Test explanation
        data = np.random.random((100, 5))
        model = MockModel()
        feature_names = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        
        explanation = service.explain_model(
            data=data,
            model=model,
            feature_names=feature_names,
            method=ExplanationMethod.SHAP
        )
        
        # Verify explanation
        assert explanation == mock_explanation
        mock_explainer.explain_global.assert_called_once()
    
    def test_explainability_service_unsupported_method(self):
        """Test error handling for unsupported explanation method."""
        service = ExplainabilityService()
        
        instance = np.array([1, 2, 3])
        model = MockModel()
        feature_names = ["feature_1", "feature_2", "feature_3"]
        
        # Should raise error for unregistered method
        with pytest.raises(ValueError, match="Explainer for method"):
            service.explain_instance(
                instance=instance,
                model=model,
                feature_names=feature_names,
                method=ExplanationMethod.LIME  # Not registered
            )


class TestExplainabilitySettings:
    """Test explainability settings functionality."""
    
    def test_explainability_settings_defaults(self):
        """Test default explainability settings."""
        settings = Settings()
        
        # Verify default values
        assert settings.explainability.enable_explainability is True
        assert settings.explainability.max_features == 10
        assert settings.explainability.background_samples == 100
        assert settings.explainability.enable_caching is True
        assert settings.explainability.n_jobs == -1
    
    def test_explainability_settings_customization(self):
        """Test customizing explainability settings."""
        settings = Settings()
        
        # Modify settings
        settings.explainability.max_features = 15
        settings.explainability.background_samples = 200
        settings.explainability.enable_caching = False
        
        # Verify changes
        assert settings.explainability.max_features == 15
        assert settings.explainability.background_samples == 200
        assert settings.explainability.enable_caching is False


class TestExplainabilityPerformance:
    """Test explainability service performance."""
    
    @patch('pynomaly.infrastructure.explainers.service_initialization.get_explainability_service')
    def test_explanation_generation_performance(self, mock_get_service):
        """Test that explanation generation completes in reasonable time."""
        # Mock service
        mock_service = Mock()
        mock_explanation = LocalExplanation(
            instance_id="perf_test",
            method=ExplanationMethod.SHAP,
            feature_contributions=[],
            prediction_confidence=0.8,
            explanation_confidence=0.85
        )
        mock_service.explain_instance.return_value = mock_explanation
        mock_get_service.return_value = mock_service
        
        # Test performance (should complete quickly with mocked service)
        import time
        start_time = time.time()
        
        service = get_explainability_service()
        result = service.explain_instance(
            instance=np.array([1, 2, 3]),
            model=MockModel(),
            feature_names=["f1", "f2", "f3"]
        )
        
        execution_time = time.time() - start_time
        
        # Verify performance (should be very fast with mocks)
        assert execution_time < 1.0  # Should complete within 1 second
        assert result == mock_explanation