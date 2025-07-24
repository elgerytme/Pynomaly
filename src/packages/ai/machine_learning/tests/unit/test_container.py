"""Unit tests for dependency injection container."""

import pytest
from unittest.mock import Mock, patch
from typing import Type, Any

from machine_learning.infrastructure.container.container import (
    Container,
    ContainerConfig,
    get_container,
    reset_container
)
from machine_learning.domain.interfaces.automl_operations import (
    AutoMLOptimizationPort,
    ModelSelectionPort,
    HyperparameterOptimizationPort
)
from machine_learning.domain.interfaces.monitoring_operations import (
    MonitoringPort,
    DistributedTracingPort
)
from machine_learning.domain.services.refactored_automl_service import AutoMLService


class TestContainerConfig:
    """Test container configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ContainerConfig()
        
        assert config.enable_sklearn_automl is True
        assert config.enable_optuna_optimization is True
        assert config.enable_distributed_tracing is True
        assert config.tracing_backend == "local"
        assert config.environment == "development"
        assert config.log_level == "INFO"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ContainerConfig(
            enable_sklearn_automl=False,
            tracing_backend="jaeger",
            environment="production",
            log_level="ERROR"
        )
        
        assert config.enable_sklearn_automl is False
        assert config.tracing_backend == "jaeger"
        assert config.environment == "production"
        assert config.log_level == "ERROR"


class TestContainer:
    """Test dependency injection container."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ContainerConfig(
            enable_sklearn_automl=False,  # Use stubs for testing
            enable_distributed_tracing=False,
            log_level="WARNING"  # Reduce test noise
        )
    
    def test_container_initialization(self):
        """Test container initializes correctly."""
        container = Container(self.config)
        
        assert container._config == self.config
        assert len(container._singletons) > 0
    
    def test_singleton_registration_and_retrieval(self):
        """Test singleton service registration and retrieval."""
        container = Container(self.config)
        
        # Test that AutoML interfaces are registered
        automl_port = container.get(AutoMLOptimizationPort)
        model_selection_port = container.get(ModelSelectionPort)
        hyperopt_port = container.get(HyperparameterOptimizationPort)
        
        assert automl_port is not None
        assert model_selection_port is not None
        assert hyperopt_port is not None
        
        # Test singleton behavior - same instance returned
        automl_port_2 = container.get(AutoMLOptimizationPort)
        assert automl_port is automl_port_2
    
    def test_service_not_registered_error(self):
        """Test error when requesting unregistered service."""
        container = Container(self.config)
        
        class UnregisteredInterface:
            pass
        
        with pytest.raises(ValueError, match="Service not registered"):
            container.get(UnregisteredInterface)
    
    def test_is_registered(self):
        """Test checking if service is registered."""
        container = Container(self.config)
        
        assert container.is_registered(AutoMLOptimizationPort) is True
        assert container.is_registered(ModelSelectionPort) is True
        
        class UnregisteredInterface:
            pass
        
        assert container.is_registered(UnregisteredInterface) is False
    
    def test_domain_service_configuration(self):
        """Test that domain services are properly configured."""
        container = Container(self.config)
        
        automl_service = container.get(AutoMLService)
        
        assert automl_service is not None
        assert automl_service._automl_port is not None
        assert automl_service._model_selection_port is not None
    
    def test_configuration_summary(self):
        """Test configuration summary generation."""
        container = Container(self.config)
        summary = container.get_configuration_summary()
        
        assert "environment" in summary
        assert "automl" in summary
        assert "monitoring" in summary
        assert "registered_services" in summary
        
        assert summary["environment"] == "development"
        assert "sklearn_enabled" in summary["automl"]
        assert "singletons" in summary["registered_services"]
        assert len(summary["registered_services"]["singletons"]) > 0
    
    @patch('machine_learning.infrastructure.container.container.SklearnAutoMLAdapter')
    def test_sklearn_adapter_configuration(self, mock_adapter_class):
        """Test scikit-learn adapter configuration."""
        # Enable sklearn in config
        config = ContainerConfig(enable_sklearn_automl=True)
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        with patch.object(Container, '_check_sklearn_availability', return_value=True):
            container = Container(config)
        
        # Verify adapter was created and registered
        mock_adapter_class.assert_called_once()
        
        # Verify same adapter instance is used for multiple interfaces
        automl_port = container.get(AutoMLOptimizationPort)
        model_selection_port = container.get(ModelSelectionPort)
        hyperopt_port = container.get(HyperparameterOptimizationPort)
        
        assert automl_port is mock_adapter
        assert model_selection_port is mock_adapter
        assert hyperopt_port is mock_adapter
    
    def test_stub_fallback_configuration(self):
        """Test fallback to stubs when external libraries unavailable."""
        config = ContainerConfig(enable_sklearn_automl=False)
        container = Container(config)
        
        automl_port = container.get(AutoMLOptimizationPort)
        
        # Should be a stub implementation
        assert "Stub" in type(automl_port).__name__
    
    def test_ml_integration_reconfiguration(self):
        """Test ML integration reconfiguration."""
        container = Container(self.config)
        
        # Get initial configuration
        initial_summary = container.get_configuration_summary()
        initial_sklearn_enabled = initial_summary["automl"]["sklearn_enabled"]
        
        # Reconfigure
        container.configure_ml_integration(
            enable_sklearn=not initial_sklearn_enabled,
            enable_optuna=True
        )
        
        # Verify configuration changed
        new_summary = container.get_configuration_summary()
        new_sklearn_enabled = new_summary["automl"]["sklearn_enabled"]
        
        assert new_sklearn_enabled != initial_sklearn_enabled
    
    def test_monitoring_integration_reconfiguration(self):
        """Test monitoring integration reconfiguration.""" 
        container = Container(self.config)
        
        # Reconfigure monitoring
        container.configure_monitoring_integration(
            enable_tracing=True,
            tracing_backend="jaeger",
            enable_monitoring=True
        )
        
        # Verify configuration applied
        summary = container.get_configuration_summary()
        assert summary["monitoring"]["tracing_enabled"] is True
        assert summary["monitoring"]["tracing_backend"] == "jaeger"


class TestGlobalContainer:
    """Test global container functionality."""
    
    def setup_method(self):
        """Reset global container before each test."""
        reset_container()
    
    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()
    
    def test_get_container_singleton(self):
        """Test global container is singleton."""
        container1 = get_container()
        container2 = get_container()
        
        assert container1 is container2
    
    def test_get_container_with_config(self):
        """Test global container with custom config."""
        config = ContainerConfig(environment="test")
        container = get_container(config)
        
        assert container._config.environment == "test"
    
    def test_reset_container(self):
        """Test resetting global container."""
        container1 = get_container()
        reset_container()
        container2 = get_container()
        
        assert container1 is not container2


class TestContainerIntegration:
    """Integration tests for container with real components."""
    
    def test_complete_service_wiring(self):
        """Test complete service dependency wiring."""
        config = ContainerConfig(
            enable_sklearn_automl=False,  # Use stubs
            enable_distributed_tracing=False,
            log_level="ERROR"
        )
        container = Container(config)
        
        # Get service and verify all dependencies are wired
        automl_service = container.get(AutoMLService)
        
        assert automl_service._automl_port is not None
        assert automl_service._model_selection_port is not None
        # Monitoring and tracing ports may be None if disabled
    
    @pytest.mark.asyncio
    async def test_service_functionality_with_stubs(self):
        """Test that services work with stub implementations."""
        config = ContainerConfig(
            enable_sklearn_automl=False,  # Force stubs
            log_level="ERROR"
        )
        container = Container(config)
        
        automl_service = container.get(AutoMLService)
        
        # Create mock dataset
        class MockDataset:
            def __init__(self):
                self.data = [[1, 2, 3] for _ in range(100)]
            
            def __len__(self):
                return 100
        
        dataset = MockDataset()
        
        # This should work with stubs
        from machine_learning.domain.interfaces.automl_operations import OptimizationConfig
        config = OptimizationConfig(max_trials=5)
        
        result = await automl_service.optimize_prediction(dataset, config)
        
        assert result is not None
        assert result.best_algorithm_type is not None
        assert result.best_score >= 0
        assert result.total_trials >= 0
    
    def test_different_environment_configurations(self):
        """Test container behavior in different environments."""
        environments = ["development", "staging", "production"]
        
        for env in environments:
            config = ContainerConfig(environment=env)
            container = Container(config)
            
            summary = container.get_configuration_summary()
            assert summary["environment"] == env
            
            # All environments should have basic services
            assert container.is_registered(AutoMLOptimizationPort)
            assert container.is_registered(AutoMLService)


@pytest.mark.performance
class TestContainerPerformance:
    """Performance tests for container operations."""
    
    def test_container_initialization_performance(self):
        """Test container initialization is fast."""
        import time
        
        config = ContainerConfig(log_level="ERROR")
        
        start_time = time.time()
        container = Container(config)
        initialization_time = time.time() - start_time
        
        # Container should initialize quickly (< 100ms)
        assert initialization_time < 0.1
        assert len(container._singletons) > 0
    
    def test_service_retrieval_performance(self):
        """Test service retrieval is fast."""
        import time
        
        config = ContainerConfig(log_level="ERROR")
        container = Container(config)
        
        # Warm up
        container.get(AutoMLOptimizationPort)
        
        # Measure retrieval time
        start_time = time.time()
        for _ in range(1000):
            service = container.get(AutoMLOptimizationPort)
        retrieval_time = time.time() - start_time
        
        # 1000 retrievals should be very fast (< 10ms)
        assert retrieval_time < 0.01
        assert service is not None
    
    def test_memory_usage_reasonable(self):
        """Test container doesn't use excessive memory."""
        import sys
        
        config = ContainerConfig(log_level="ERROR")
        
        # Measure memory before
        initial_size = sys.getsizeof(config)
        
        container = Container(config)
        
        # Get all services to fully initialize
        container.get(AutoMLOptimizationPort)
        container.get(ModelSelectionPort)
        container.get(AutoMLService)
        
        # Container should not use excessive memory
        container_size = sys.getsizeof(container._singletons)
        
        # Should be reasonable (< 10KB for container itself)
        assert container_size < 10000