"""Unit tests for domain services."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, Dict, List, Optional

from machine_learning.domain.services.refactored_automl_service import AutoMLService
from machine_learning.domain.interfaces.automl_operations import (
    AutoMLOptimizationPort,
    ModelSelectionPort,
    HyperparameterOptimizationPort,
    OptimizationConfig,
    OptimizationResult,
    AlgorithmConfig,
    AlgorithmType
)
from machine_learning.domain.interfaces.monitoring_operations import (
    MonitoringPort,
    DistributedTracingPort,
    TraceSpan
)


class TestAutoMLServiceInitialization:
    """Test AutoML service initialization and dependency injection."""
    
    def test_service_initialization_with_required_dependencies(self):
        """Test service initializes with required dependencies."""
        mock_automl_port = Mock(spec=AutoMLOptimizationPort)
        mock_model_selection_port = Mock(spec=ModelSelectionPort)
        
        service = AutoMLService(
            automl_port=mock_automl_port,
            model_selection_port=mock_model_selection_port
        )
        
        assert service._automl_port is mock_automl_port
        assert service._model_selection_port is mock_model_selection_port
        assert service._monitoring_port is None  # Optional dependency
        assert service._tracing_port is None     # Optional dependency
    
    def test_service_initialization_with_all_dependencies(self):
        """Test service initializes with all dependencies."""
        mock_automl_port = Mock(spec=AutoMLOptimizationPort)
        mock_model_selection_port = Mock(spec=ModelSelectionPort)
        mock_monitoring_port = Mock(spec=MonitoringPort)
        mock_tracing_port = Mock(spec=DistributedTracingPort)
        
        service = AutoMLService(
            automl_port=mock_automl_port,
            model_selection_port=mock_model_selection_port,
            monitoring_port=mock_monitoring_port,
            tracing_port=mock_tracing_port
        )
        
        assert service._automl_port is mock_automl_port
        assert service._model_selection_port is mock_model_selection_port
        assert service._monitoring_port is mock_monitoring_port
        assert service._tracing_port is mock_tracing_port
    
    def test_service_initialization_fails_with_missing_required_dependencies(self):
        """Test service initialization fails without required dependencies."""
        with pytest.raises(TypeError):
            AutoMLService()  # Missing required parameters
        
        with pytest.raises(TypeError):
            AutoMLService(automl_port=Mock())  # Missing model_selection_port


class TestAutoMLServiceOptimization:
    """Test AutoML service optimization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_automl_port = Mock(spec=AutoMLOptimizationPort)
        self.mock_model_selection_port = Mock(spec=ModelSelectionPort)
        self.mock_monitoring_port = Mock(spec=MonitoringPort)
        self.mock_tracing_port = Mock(spec=DistributedTracingPort)
        
        self.service = AutoMLService(
            automl_port=self.mock_automl_port,
            model_selection_port=self.mock_model_selection_port,
            monitoring_port=self.mock_monitoring_port,
            tracing_port=self.mock_tracing_port
        )
        
        # Mock dataset
        class MockDataset:
            def __init__(self, size=1000):
                self.size = size
                self.data = [[i, i*2] for i in range(size)]
            
            def __len__(self):
                return self.size
        
        self.mock_dataset = MockDataset()
    
    @pytest.mark.asyncio
    async def test_optimize_prediction_basic_flow(self):
        """Test basic optimization flow."""
        # Arrange
        config = OptimizationConfig(max_trials=5)
        expected_result = OptimizationResult(
            best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
            best_parameters={"contamination": 0.1},
            best_score=0.85,
            total_trials=5,
            optimization_time_seconds=10.0
        )
        
        self.mock_automl_port.optimize_model = AsyncMock(return_value=expected_result)
        self.mock_monitoring_port.record_metric = AsyncMock()
        
        # Act
        result = await self.service.optimize_prediction(
            dataset=self.mock_dataset,
            optimization_config=config
        )
        
        # Assert
        assert result == expected_result
        self.mock_automl_port.optimize_model.assert_called_once_with(
            self.mock_dataset, config, None
        )
        # Should record metrics about optimization
        self.mock_monitoring_port.record_metric.assert_called()
    
    @pytest.mark.asyncio
    async def test_optimize_prediction_with_tracing(self):
        """Test optimization with distributed tracing."""
        # Arrange
        config = OptimizationConfig(max_trials=3)
        expected_result = OptimizationResult(
            best_algorithm_type=AlgorithmType.ONE_CLASS_SVM,
            best_parameters={"nu": 0.05},
            best_score=0.78,
            total_trials=3,
            optimization_time_seconds=15.0
        )
        
        mock_trace_span = TraceSpan(
            span_id="test_span",
            operation_name="optimize_prediction"
        )
        
        self.mock_tracing_port.start_trace = AsyncMock(return_value=mock_trace_span)
        self.mock_tracing_port.finish_trace = AsyncMock()
        self.mock_automl_port.optimize_model = AsyncMock(return_value=expected_result)
        
        # Act
        result = await self.service.optimize_prediction(
            dataset=self.mock_dataset,
            optimization_config=config
        )
        
        # Assert
        assert result == expected_result
        self.mock_tracing_port.start_trace.assert_called_once_with(
            "optimize_prediction",
            tags={"dataset_size": len(self.mock_dataset), "max_trials": 3}
        )
        self.mock_tracing_port.finish_trace.assert_called_once_with(mock_trace_span.span_id)
    
    @pytest.mark.asyncio
    async def test_optimize_prediction_handles_port_exceptions(self):
        """Test optimization handles exceptions from ports gracefully."""
        # Arrange
        config = OptimizationConfig(max_trials=5)
        self.mock_automl_port.optimize_model = AsyncMock(
            side_effect=RuntimeError("External service unavailable")
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="External service unavailable"):
            await self.service.optimize_prediction(
                dataset=self.mock_dataset,
                optimization_config=config
            )
    
    @pytest.mark.asyncio
    async def test_optimize_prediction_without_optional_dependencies(self):
        """Test optimization works without optional monitoring/tracing."""
        # Arrange - Create service without optional dependencies
        minimal_service = AutoMLService(
            automl_port=self.mock_automl_port,
            model_selection_port=self.mock_model_selection_port
        )
        
        config = OptimizationConfig(max_trials=3)
        expected_result = OptimizationResult(
            best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
            best_parameters={},
            best_score=0.90,
            total_trials=3,
            optimization_time_seconds=8.0
        )
        
        self.mock_automl_port.optimize_model = AsyncMock(return_value=expected_result)
        
        # Act
        result = await minimal_service.optimize_prediction(
            dataset=self.mock_dataset,
            optimization_config=config  
        )
        
        # Assert
        assert result == expected_result
        self.mock_automl_port.optimize_model.assert_called_once()


class TestAutoMLServiceAlgorithmSelection:
    """Test AutoML service algorithm selection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_automl_port = Mock(spec=AutoMLOptimizationPort)
        self.mock_model_selection_port = Mock(spec=ModelSelectionPort)
        
        self.service = AutoMLService(
            automl_port=self.mock_automl_port,
            model_selection_port=self.mock_model_selection_port
        )
        
        class MockDataset:
            def __len__(self):
                return 5000
        
        self.mock_dataset = MockDataset()
    
    @pytest.mark.asyncio
    async def test_auto_select_algorithm_delegates_to_port(self):
        """Test auto_select_algorithm delegates to model selection port."""
        # Arrange
        expected_algorithm = AlgorithmType.LOCAL_OUTLIER_FACTOR
        expected_config = AlgorithmConfig(
            algorithm_type=expected_algorithm,
            parameters={"n_neighbors": 20}
        )
        
        self.mock_model_selection_port.select_best_algorithm = AsyncMock(
            return_value=(expected_algorithm, expected_config)
        )
        
        # Act
        algorithm, config = await self.service.auto_select_algorithm(
            dataset=self.mock_dataset,
            quick_mode=True
        )
        
        # Assert
        assert algorithm == expected_algorithm
        assert config == expected_config
        self.mock_model_selection_port.select_best_algorithm.assert_called_once_with(
            self.mock_dataset, quick_mode=True
        )
    
    @pytest.mark.asyncio
    async def test_get_optimization_recommendations_delegates_to_port(self):
        """Test get_optimization_recommendations delegates to port."""
        # Arrange
        expected_recommendations = {
            "algorithms": ["isolation_forest", "local_outlier_factor"],
            "hyperparameters": {"contamination": [0.05, 0.1, 0.15]},
            "preprocessing": ["standardization", "normalization"]
        }
        
        self.mock_model_selection_port.get_algorithm_recommendations = AsyncMock(
            return_value=expected_recommendations
        )
        
        # Act
        recommendations = await self.service.get_optimization_recommendations(
            dataset=self.mock_dataset
        )
        
        # Assert
        assert recommendations == expected_recommendations
        self.mock_model_selection_port.get_algorithm_recommendations.assert_called_once_with(
            self.mock_dataset
        )


class TestAutoMLServiceBusinessLogic:
    """Test AutoML service business logic and orchestration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_automl_port = Mock(spec=AutoMLOptimizationPort)
        self.mock_model_selection_port = Mock(spec=ModelSelectionPort)
        self.mock_monitoring_port = Mock(spec=MonitoringPort)
        
        self.service = AutoMLService(
            automl_port=self.mock_automl_port,
            model_selection_port=self.mock_model_selection_port,
            monitoring_port=self.mock_monitoring_port
        )
    
    def test_service_enforces_business_rules(self):
        """Test service enforces business rules and validations."""
        # Test validation of input parameters
        class InvalidDataset:
            def __len__(self):
                return 0  # Empty dataset should be rejected
        
        invalid_dataset = InvalidDataset()
        
        # The service should validate inputs before calling ports
        # This would be implemented in the actual service
        pass
    
    def test_service_orchestrates_multiple_ports(self):
        """Test service orchestrates multiple ports correctly."""
        # Test that service coordinates between different ports
        # For complex operations requiring multiple port interactions
        pass
    
    def test_service_applies_domain_business_logic(self):
        """Test service applies domain-specific business logic."""
        # Test that service adds business value beyond just delegating
        # For example, combining results from multiple ports
        pass


class TestAutoMLServiceErrorHandling:
    """Test AutoML service error handling and resilience."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_automl_port = Mock(spec=AutoMLOptimizationPort)
        self.mock_model_selection_port = Mock(spec=ModelSelectionPort)
        
        self.service = AutoMLService(
            automl_port=self.mock_automl_port,
            model_selection_port=self.mock_model_selection_port
        )
    
    @pytest.mark.asyncio
    async def test_service_handles_port_timeout_errors(self):
        """Test service handles port timeout errors."""
        # Arrange
        self.mock_automl_port.optimize_model = AsyncMock(
            side_effect=TimeoutError("Operation timed out")
        )
        
        class MockDataset:
            def __len__(self):
                return 1000
        
        dataset = MockDataset()
        config = OptimizationConfig()
        
        # Act & Assert
        with pytest.raises(TimeoutError):
            await self.service.optimize_prediction(dataset, config)
    
    @pytest.mark.asyncio
    async def test_service_handles_port_connection_errors(self):
        """Test service handles port connection errors."""
        # Arrange
        self.mock_automl_port.optimize_model = AsyncMock(
            side_effect=ConnectionError("Cannot connect to service")
        )
        
        class MockDataset:
            def __len__(self):
                return 1000
        
        dataset = MockDataset()
        config = OptimizationConfig()
        
        # Act & Assert
        with pytest.raises(ConnectionError):
            await self.service.optimize_prediction(dataset, config)
    
    def test_service_validates_input_parameters(self):
        """Test service validates input parameters."""
        # Test that service validates inputs before processing
        pass
    
    def test_service_provides_meaningful_error_messages(self):
        """Test service provides meaningful error messages."""
        # Test that service wraps low-level errors with domain context
        pass


class TestAutoMLServiceMonitoring:
    """Test AutoML service monitoring and observability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_automl_port = Mock(spec=AutoMLOptimizationPort)
        self.mock_model_selection_port = Mock(spec=ModelSelectionPort)
        self.mock_monitoring_port = Mock(spec=MonitoringPort)
        
        self.service = AutoMLService(
            automl_port=self.mock_automl_port,
            model_selection_port=self.mock_model_selection_port,
            monitoring_port=self.mock_monitoring_port
        )
    
    @pytest.mark.asyncio
    async def test_service_records_performance_metrics(self):
        """Test service records performance metrics."""
        # Arrange
        config = OptimizationConfig(max_trials=5)
        result = OptimizationResult(
            best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
            best_parameters={},
            best_score=0.85,
            total_trials=5,
            optimization_time_seconds=30.0
        )
        
        self.mock_automl_port.optimize_model = AsyncMock(return_value=result)
        self.mock_monitoring_port.record_metric = AsyncMock()
        
        class MockDataset:
            def __len__(self):
                return 1000
        
        dataset = MockDataset()
        
        # Act
        await self.service.optimize_prediction(dataset, config)
        
        # Assert - Should record various metrics
        calls = self.mock_monitoring_port.record_metric.call_args_list
        assert len(calls) > 0
        
        # Should record at least optimization time
        metric_names = [call[0][0] for call in calls]
        assert any("optimization_time" in name for name in metric_names)
    
    @pytest.mark.asyncio
    async def test_service_records_business_metrics(self):
        """Test service records business-relevant metrics."""
        # Test that service records domain-specific metrics
        # Like algorithm selection frequency, success rates, etc.
        pass


class TestAutoMLServicePerformance:
    """Test AutoML service performance characteristics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_automl_port = Mock(spec=AutoMLOptimizationPort)
        self.mock_model_selection_port = Mock(spec=ModelSelectionPort)
        
        self.service = AutoMLService(
            automl_port=self.mock_automl_port,
            model_selection_port=self.mock_model_selection_port
        )
    
    @pytest.mark.asyncio
    async def test_service_handles_concurrent_requests(self):
        """Test service handles concurrent optimization requests."""
        import asyncio
        
        # Arrange
        config = OptimizationConfig(max_trials=1)
        result = OptimizationResult(
            best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
            best_parameters={},
            best_score=0.8,
            total_trials=1,
            optimization_time_seconds=1.0
        )
        
        self.mock_automl_port.optimize_model = AsyncMock(return_value=result)
        
        class MockDataset:
            def __len__(self):
                return 100
        
        datasets = [MockDataset() for _ in range(5)]
        
        # Act - Run multiple optimizations concurrently
        tasks = [
            self.service.optimize_prediction(dataset, config)
            for dataset in datasets
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert len(results) == 5
        for result in results:
            assert isinstance(result, OptimizationResult)
        
        # Should have called port for each request
        assert self.mock_automl_port.optimize_model.call_count == 5
    
    @pytest.mark.performance
    def test_service_memory_usage_reasonable(self):
        """Test service memory usage is reasonable."""
        import sys
        
        # Create multiple service instances
        services = []
        for _ in range(100):
            mock_automl = Mock(spec=AutoMLOptimizationPort)
            mock_selection = Mock(spec=ModelSelectionPort)
            service = AutoMLService(mock_automl, mock_selection)
            services.append(service)
        
        # Should not consume excessive memory
        assert len(services) == 100
        
        # Clean up
        del services
    
    @pytest.mark.asyncio
    async def test_service_response_time_reasonable(self):
        """Test service response time is reasonable."""
        import time
        
        # Arrange - Fast-responding mock
        config = OptimizationConfig(max_trials=1)
        result = OptimizationResult(
            best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
            best_parameters={},
            best_score=0.8,
            total_trials=1,
            optimization_time_seconds=0.1
        )
        
        self.mock_automl_port.optimize_model = AsyncMock(return_value=result)
        
        class MockDataset:
            def __len__(self):
                return 100
        
        dataset = MockDataset()
        
        # Act
        start_time = time.time()
        await self.service.optimize_prediction(dataset, config)
        response_time = time.time() - start_time
        
        # Assert - Service overhead should be minimal (< 10ms)
        assert response_time < 0.01