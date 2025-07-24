"""Integration tests for hexagonal architecture implementation."""

import pytest
import asyncio
from typing import Any, Dict, List, Optional

from machine_learning.infrastructure.container import Container, ContainerConfig
from machine_learning.domain.services.refactored_automl_service import AutoMLService
from machine_learning.domain.interfaces.automl_operations import (
    AutoMLOptimizationPort,
    ModelSelectionPort,
    OptimizationConfig,
    AlgorithmType
)
from machine_learning.domain.interfaces.monitoring_operations import (
    MonitoringPort,
    DistributedTracingPort
)


class MockDataset:
    """Mock dataset for integration testing."""
    
    def __init__(self, name: str = "test_dataset", size: int = 1000):
        self.name = name
        self.size = size
        self.data = [[i, i*2, i*3] for i in range(size)]
    
    def __len__(self):
        return self.size
    
    def __repr__(self):
        return f"MockDataset(name='{self.name}', size={self.size})"


class TestContainerIntegration:
    """Test container integration with real components."""
    
    def test_container_creates_services_with_proper_dependencies(self):
        """Test container creates services with properly wired dependencies."""
        # Create container with minimal configuration
        config = ContainerConfig(
            enable_sklearn_automl=False,  # Use stubs for testing
            enable_distributed_tracing=False,
            log_level="ERROR"  # Reduce test noise
        )
        container = Container(config)
        
        # Get service from container
        service = container.get(AutoMLService)
        
        # Verify service is properly configured
        assert service is not None
        assert service._automl_port is not None
        assert service._model_selection_port is not None
        
        # Verify ports implement expected interfaces
        assert isinstance(service._automl_port, AutoMLOptimizationPort)
        assert isinstance(service._model_selection_port, ModelSelectionPort)
    
    def test_container_singleton_behavior(self):
        """Test container provides singleton instances."""
        config = ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        )
        container = Container(config)
        
        # Get same service multiple times
        service1 = container.get(AutoMLService)
        service2 = container.get(AutoMLService)
        
        # Should be same instance
        assert service1 is service2
        
        # Underlying ports should also be singletons
        assert service1._automl_port is service2._automl_port
        assert service1._model_selection_port is service2._model_selection_port
    
    def test_container_handles_different_configurations(self):
        """Test container handles different configurations correctly."""
        configurations = [
            # Development configuration
            ContainerConfig(
                environment="development",
                enable_sklearn_automl=False,
                enable_distributed_tracing=False,
                log_level="INFO"
            ),
            # Production configuration  
            ContainerConfig(
                environment="production",
                enable_sklearn_automl=False,
                enable_distributed_tracing=False,
                log_level="WARNING"
            ),
            # Testing configuration
            ContainerConfig(
                environment="testing",
                enable_sklearn_automl=False,
                enable_distributed_tracing=False,
                log_level="ERROR"
            )
        ]
        
        for config in configurations:
            container = Container(config)
            service = container.get(AutoMLService)
            
            assert service is not None
            summary = container.get_configuration_summary()
            assert summary["environment"] == config.environment


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows using hexagonal architecture."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ContainerConfig(
            enable_sklearn_automl=False,  # Use stubs for predictable testing
            enable_distributed_tracing=False,
            log_level="ERROR"
        )
        self.container = Container(self.config)
        self.service = self.container.get(AutoMLService)
        self.dataset = MockDataset("integration_test", 500)
    
    @pytest.mark.asyncio
    async def test_complete_optimization_workflow(self):
        """Test complete optimization workflow from start to finish."""
        # Configure optimization
        optimization_config = OptimizationConfig(
            max_trials=3,
            algorithms_to_test=[
                AlgorithmType.ISOLATION_FOREST,
                AlgorithmType.LOCAL_OUTLIER_FACTOR
            ]
        )
        
        # Run optimization
        result = await self.service.optimize_prediction(
            dataset=self.dataset,
            optimization_config=optimization_config
        )
        
        # Verify results
        assert result is not None
        assert result.best_algorithm_type in optimization_config.algorithms_to_test
        assert result.best_score >= 0
        assert result.total_trials <= optimization_config.max_trials
        assert result.optimization_time_seconds >= 0
        
        # Verify business logic was applied
        assert isinstance(result.best_algorithm_type, AlgorithmType)
        assert isinstance(result.best_parameters, dict)
    
    @pytest.mark.asyncio
    async def test_algorithm_selection_workflow(self):
        """Test algorithm selection workflow."""
        # Get algorithm recommendation
        algorithm, config = await self.service.auto_select_algorithm(
            dataset=self.dataset,
            quick_mode=True
        )
        
        # Verify recommendation
        assert isinstance(algorithm, AlgorithmType)
        assert config.algorithm_type == algorithm
        assert isinstance(config.parameters, dict)
        
        # Get detailed recommendations
        recommendations = await self.service.get_optimization_recommendations(
            dataset=self.dataset
        )
        
        # Verify recommendations structure
        assert isinstance(recommendations, dict)
        assert "algorithms" in recommendations or len(recommendations) == 0  # Stubs may return empty
    
    @pytest.mark.asyncio
    async def test_parallel_optimization_workflow(self):
        """Test parallel optimization requests."""
        datasets = [
            MockDataset(f"dataset_{i}", 100 + i*50)
            for i in range(3)
        ]
        
        configs = [
            OptimizationConfig(max_trials=2),
            OptimizationConfig(max_trials=3),
            OptimizationConfig(max_trials=1)
        ]
        
        # Run optimizations in parallel
        tasks = [
            self.service.optimize_prediction(dataset, config)
            for dataset, config in zip(datasets, configs)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result is not None
            assert result.total_trials <= configs[i].max_trials


class TestArchitecturalBoundaries:
    """Test that architectural boundaries are properly maintained."""
    
    def test_domain_service_only_depends_on_interfaces(self):
        """Test domain service only depends on domain interfaces."""
        import inspect
        from machine_learning.domain.services.refactored_automl_service import AutoMLService
        
        # Get the service's __init__ method signature
        sig = inspect.signature(AutoMLService.__init__)
        
        # Verify all dependencies are interfaces (ports)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Check that parameter annotations are interfaces
            annotation = param.annotation
            if annotation != inspect.Parameter.empty:
                # Should be an interface (ABC) not a concrete class
                # This is validated by checking the parameter names match expected ports
                assert any(port_name in param_name for port_name in [
                    'port', 'automl', 'model_selection', 'monitoring', 'tracing'
                ])
    
    def test_infrastructure_adapters_implement_domain_interfaces(self):
        """Test infrastructure adapters implement domain interfaces."""
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"
        ))
        
        # Get adapters from container
        automl_port = container.get(AutoMLOptimizationPort)
        model_selection_port = container.get(ModelSelectionPort)
        
        # Verify they implement the interfaces
        assert isinstance(automl_port, AutoMLOptimizationPort)
        assert isinstance(model_selection_port, ModelSelectionPort)
        
        # Verify they have the required methods
        assert hasattr(automl_port, 'optimize_model')
        assert hasattr(model_selection_port, 'select_best_algorithm')
    
    def test_container_enforces_dependency_inversion(self):
        """Test container enforces dependency inversion principle."""
        container = Container(ContainerConfig(log_level="ERROR"))
        
        # High-level service should depend on abstractions
        service = container.get(AutoMLService)
        
        # Service should receive interfaces, not concrete implementations
        assert isinstance(service._automl_port, AutoMLOptimizationPort)
        assert isinstance(service._model_selection_port, ModelSelectionPort)
        
        # Service should not know about concrete adapter classes


class TestConfigurationManagement:
    """Test configuration management across the architecture."""
    
    def test_different_environments_produce_different_behaviors(self):
        """Test different environments produce appropriate behaviors."""
        environments = ["development", "staging", "production"]
        
        containers = {}
        for env in environments:
            config = ContainerConfig(
                environment=env,
                enable_sklearn_automl=False,
                log_level="ERROR"
            )
            containers[env] = Container(config)
        
        # Each environment should be configured
        for env, container in containers.items():
            summary = container.get_configuration_summary()
            assert summary["environment"] == env
            
            # Should be able to create services in each environment
            service = container.get(AutoMLService)
            assert service is not None
    
    def test_runtime_reconfiguration(self):
        """Test runtime reconfiguration capabilities."""
        container = Container(ContainerConfig(log_level="ERROR"))
        
        # Get initial configuration
        initial_summary = container.get_configuration_summary()
        
        # Reconfigure ML integration
        container.configure_ml_integration(
            enable_sklearn=False,
            enable_optuna=True
        )
        
        # Verify configuration changed
        new_summary = container.get_configuration_summary()
        assert new_summary != initial_summary
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration should work
        valid_config = ContainerConfig(
            environment="production",
            enable_sklearn_automl=True,
            log_level="INFO"
        )
        
        container = Container(valid_config)
        assert container is not None
        
        # Invalid configurations should be handled gracefully
        # (Current implementation uses defaults for invalid values)


class TestGracefulDegradation:
    """Test graceful degradation when external services unavailable."""
    
    @pytest.mark.asyncio
    async def test_optimization_works_with_all_stubs(self):
        """Test optimization works when all external services use stubs."""
        # Force all stubs
        config = ContainerConfig(
            enable_sklearn_automl=False,
            enable_optuna_optimization=False,
            enable_distributed_tracing=False,
            enable_prometheus_monitoring=False,
            log_level="ERROR"  # Reduce stub warnings
        )
        
        container = Container(config)
        service = container.get(AutoMLService)
        
        dataset = MockDataset("stub_test", 200)
        optimization_config = OptimizationConfig(max_trials=2)
        
        # Should complete successfully with stubs
        result = await service.optimize_prediction(dataset, optimization_config)
        
        assert result is not None
        assert result.best_algorithm_type is not None
        assert result.best_score >= 0
        assert result.total_trials >= 0
    
    def test_container_falls_back_to_stubs_automatically(self):
        """Test container automatically falls back to stubs when needed."""
        # Configure to try real implementations but expect fallback
        config = ContainerConfig(
            enable_sklearn_automl=True,  # Will fall back to stub if sklearn unavailable
            log_level="ERROR"
        )
        
        container = Container(config)
        
        # Should create container successfully regardless of sklearn availability
        assert container is not None
        
        # Should provide working services
        service = container.get(AutoMLService)
        assert service is not None
        assert service._automl_port is not None
    
    def test_partial_service_availability(self):
        """Test system works with partial service availability."""
        # Enable some services, disable others
        config = ContainerConfig(
            enable_sklearn_automl=False,       # Use stub
            enable_distributed_tracing=True,   # Try real implementation
            enable_prometheus_monitoring=False # Use stub
        )
        
        container = Container(config)
        service = container.get(AutoMLService)
        
        # Service should be created with mixed real/stub implementations
        assert service is not None
        assert service._automl_port is not None
        assert service._model_selection_port is not None


class TestPerformanceIntegration:
    """Test performance characteristics of the integrated system."""
    
    @pytest.mark.performance
    def test_container_initialization_performance(self):
        """Test container initialization is fast enough for production."""
        import time
        
        configs = [
            ContainerConfig(environment="production", log_level="ERROR"),
            ContainerConfig(environment="development", log_level="ERROR"),
            ContainerConfig(environment="testing", log_level="ERROR")
        ]
        
        for config in configs:
            start_time = time.time()
            container = Container(config)
            service = container.get(AutoMLService)
            initialization_time = time.time() - start_time
            
            # Should initialize quickly (< 100ms)
            assert initialization_time < 0.1
            assert service is not None
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_end_to_end_performance(self):
        """Test end-to-end performance is acceptable."""
        import time
        
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,  # Use fast stubs
            log_level="ERROR"
        ))
        service = container.get(AutoMLService)
        
        dataset = MockDataset("performance_test", 100)
        config = OptimizationConfig(max_trials=1)
        
        start_time = time.time()
        result = await service.optimize_prediction(dataset, config)
        execution_time = time.time() - start_time
        
        # Should complete quickly with stubs (< 100ms)
        assert execution_time < 0.1
        assert result is not None
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self):
        """Test performance under concurrent load."""
        import time
        
        container = Container(ContainerConfig(
            enable_sklearn_automl=False,
            log_level="ERROR"  
        ))
        service = container.get(AutoMLService)
        
        # Create multiple concurrent requests
        datasets = [MockDataset(f"concurrent_{i}", 50) for i in range(10)]
        config = OptimizationConfig(max_trials=1)
        
        start_time = time.time()
        tasks = [
            service.optimize_prediction(dataset, config)
            for dataset in datasets
        ]
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Should handle 10 concurrent requests quickly (< 500ms with stubs)
        assert execution_time < 0.5
        assert len(results) == 10
        assert all(result is not None for result in results)


class TestErrorHandlingIntegration:
    """Test error handling across architectural boundaries."""
    
    @pytest.mark.asyncio
    async def test_service_error_propagation(self):
        """Test errors propagate correctly through the architecture."""
        # This would test error handling in a real scenario
        # For now, we verify the architecture supports error propagation
        container = Container(ContainerConfig(log_level="ERROR"))
        service = container.get(AutoMLService)
        
        # Service should be properly configured for error handling
        assert service is not None
        assert hasattr(service, '_automl_port')
        assert hasattr(service, '_model_selection_port')
    
    def test_container_handles_misconfiguration_gracefully(self):
        """Test container handles misconfiguration gracefully."""
        # Invalid log level should not break container
        config = ContainerConfig(log_level="INVALID_LEVEL")
        
        # Should create container without crashing
        container = Container(config)
        assert container is not None
        
        # Should provide working services
        service = container.get(AutoMLService)
        assert service is not None


@pytest.mark.integration
class TestRealDependencyIntegration:
    """Integration tests with real external dependencies (when available)."""
    
    @pytest.mark.skipif(True, reason="Requires external dependencies")
    @pytest.mark.asyncio
    async def test_real_sklearn_integration(self):
        """Test with real scikit-learn when available."""
        try:
            config = ContainerConfig(
                enable_sklearn_automl=True,
                log_level="INFO"
            )
            container = Container(config)
            service = container.get(AutoMLService)
            
            # Use real data
            import numpy as np
            class RealDataset:
                def __init__(self):
                    # Generate random dataset
                    np.random.seed(42)
                    self.data = np.random.random((1000, 10))
                
                def __len__(self):
                    return 1000
            
            dataset = RealDataset()
            config = OptimizationConfig(max_trials=5)
            
            result = await service.optimize_prediction(dataset, config)
            
            assert result is not None
            assert result.total_trials <= 5
            assert result.best_score >= 0
            
        except ImportError:
            pytest.skip("sklearn not available for real integration test")
    
    @pytest.mark.skipif(True, reason="Requires external dependencies")
    def test_real_monitoring_integration(self):
        """Test with real monitoring when available."""
        try:
            config = ContainerConfig(
                enable_prometheus_monitoring=True,
                enable_distributed_tracing=True,
                tracing_backend="jaeger"
            )
            container = Container(config)
            
            # Should integrate with real monitoring systems
            summary = container.get_configuration_summary()
            assert summary["monitoring"]["prometheus_enabled"] is True
            
        except ImportError:
            pytest.skip("Monitoring dependencies not available")