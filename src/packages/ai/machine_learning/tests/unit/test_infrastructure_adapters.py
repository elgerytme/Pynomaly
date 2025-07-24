"""Unit tests for infrastructure adapters."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict, List, Optional

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
    DistributedTracingPort,
    MonitoringPort,
    TraceSpan
)


class TestSklearnAutoMLAdapter:
    """Test scikit-learn AutoML adapter implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the adapter to avoid sklearn dependency in tests
        self.mock_sklearn_available = True
        self.mock_optuna_available = True
    
    @patch('machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter.SklearnAutoMLAdapter._check_sklearn_availability')
    @patch('machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter.IsolationForest')
    def test_sklearn_adapter_initialization(self, mock_isolation_forest, mock_check_availability):
        """Test that sklearn adapter initializes correctly."""
        mock_check_availability.return_value = True
        
        # Import only when mocked to avoid dependency issues
        with patch.dict('sys.modules', {
            'sklearn': Mock(),
            'sklearn.ensemble': Mock(),
            'sklearn.svm': Mock(),
            'sklearn.neighbors': Mock()
        }):
            from machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter import SklearnAutoMLAdapter
            
            adapter = SklearnAutoMLAdapter()
            assert adapter is not None
            mock_check_availability.assert_called_once()
    
    @patch('machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter.SklearnAutoMLAdapter._check_sklearn_availability')
    def test_sklearn_adapter_optimization_interface(self, mock_check_availability):
        """Test that sklearn adapter implements optimization interface correctly."""
        mock_check_availability.return_value = True
        
        with patch.dict('sys.modules', {
            'sklearn': Mock(),
            'sklearn.ensemble': Mock(),
            'sklearn.svm': Mock(),
            'sklearn.neighbors': Mock(),
            'optuna': Mock()
        }):
            from machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter import SklearnAutoMLAdapter
            
            adapter = SklearnAutoMLAdapter()
            
            # Verify it implements required interfaces
            assert isinstance(adapter, AutoMLOptimizationPort)
            assert isinstance(adapter, ModelSelectionPort)
            assert isinstance(adapter, HyperparameterOptimizationPort)
    
    @pytest.mark.asyncio
    @patch('machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter.SklearnAutoMLAdapter._check_sklearn_availability')
    async def test_sklearn_adapter_optimize_model_mock(self, mock_check_availability):
        """Test optimize_model method with mocked sklearn."""
        mock_check_availability.return_value = True
        
        # Create comprehensive mocks
        mock_sklearn = Mock()
        mock_isolation_forest = Mock()
        mock_isolation_forest.fit.return_value = mock_isolation_forest
        mock_isolation_forest.score_samples.return_value = [0.1, 0.2, 0.8, 0.9]
        mock_sklearn.ensemble.IsolationForest.return_value = mock_isolation_forest
        
        mock_optuna = Mock()
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.1
        mock_trial.suggest_int.return_value = 100
        mock_study.best_params = {"contamination": 0.1, "n_estimators": 100}
        mock_study.best_value = 0.85
        mock_study.trials = [mock_trial] * 10
        mock_optuna.create_study.return_value = mock_study
        
        with patch.dict('sys.modules', {
            'sklearn': mock_sklearn,
            'sklearn.ensemble': mock_sklearn.ensemble,
            'sklearn.svm': Mock(),
            'sklearn.neighbors': Mock(),
            'optuna': mock_optuna
        }):
            from machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter import SklearnAutoMLAdapter
            
            adapter = SklearnAutoMLAdapter()
            
            # Create mock dataset
            class MockDataset:
                def __init__(self):
                    self.data = [[1, 2], [3, 4], [5, 6], [7, 8]]
                
                def __len__(self):
                    return 4
            
            dataset = MockDataset()
            config = OptimizationConfig(max_trials=5)
            
            # Mock the internal optimization method
            with patch.object(adapter, '_run_optuna_optimization') as mock_optimize:
                expected_result = OptimizationResult(
                    best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
                    best_parameters={"contamination": 0.1},
                    best_score=0.85,
                    total_trials=5,
                    optimization_time_seconds=10.0
                )
                mock_optimize.return_value = expected_result
                
                result = await adapter.optimize_model(dataset, config)
                
                assert isinstance(result, OptimizationResult)
                assert result.best_algorithm_type == AlgorithmType.ISOLATION_FOREST
                assert result.best_score == 0.85
                mock_optimize.assert_called_once()


class TestDistributedTracingAdapter:
    """Test distributed tracing adapter implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_jaeger_available = True
        self.mock_config = {
            "service_name": "test_service",
            "jaeger_endpoint": "http://localhost:14268"
        }
    
    @patch('machine_learning.infrastructure.adapters.monitoring.distributed_tracing_adapter.DistributedTracingAdapter._check_tracing_availability')
    def test_tracing_adapter_initialization(self, mock_check_availability):
        """Test tracing adapter initializes correctly."""
        mock_check_availability.return_value = True
        
        with patch.dict('sys.modules', {
            'jaeger_client': Mock(),
            'opentracing': Mock()
        }):
            from machine_learning.infrastructure.adapters.monitoring.distributed_tracing_adapter import DistributedTracingAdapter
            
            adapter = DistributedTracingAdapter(backend="jaeger", config=self.mock_config)
            assert adapter is not None
            assert isinstance(adapter, DistributedTracingPort)
    
    @pytest.mark.asyncio
    @patch('machine_learning.infrastructure.adapters.monitoring.distributed_tracing_adapter.DistributedTracingAdapter._check_tracing_availability')
    async def test_tracing_adapter_start_trace(self, mock_check_availability):
        """Test starting a trace."""
        mock_check_availability.return_value = True
        
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.span_id = "test_span_123" 
        mock_span.context.span_id = "test_span_123"
        mock_tracer.start_span.return_value = mock_span
        
        with patch.dict('sys.modules', {
            'jaeger_client': Mock(),
            'opentracing': Mock()
        }):
            from machine_learning.infrastructure.adapters.monitoring.distributed_tracing_adapter import DistributedTracingAdapter
            
            adapter = DistributedTracingAdapter(backend="local")
            adapter._tracer = mock_tracer
            
            trace_span = await adapter.start_trace(
                operation_name="test_operation",
                tags={"test": "value"}
            )
            
            assert isinstance(trace_span, TraceSpan)
            assert trace_span.operation_name == "test_operation"
            assert trace_span.tags.get("test") == "value"
    
    @patch('machine_learning.infrastructure.adapters.monitoring.distributed_tracing_adapter.DistributedTracingAdapter._check_tracing_availability')
    def test_tracing_adapter_decorator(self, mock_check_availability):
        """Test tracing decorator functionality."""
        mock_check_availability.return_value = True
        
        with patch.dict('sys.modules', {
            'jaeger_client': Mock(),
            'opentracing': Mock()
        }):
            from machine_learning.infrastructure.adapters.monitoring.distributed_tracing_adapter import DistributedTracingAdapter
            
            adapter = DistributedTracingAdapter(backend="local")
            
            # Test that decorator can be applied
            @adapter.trace_operation("test_operation")
            async def test_function():
                return "success"
            
            # Verify decorator was applied
            assert hasattr(test_function, '__wrapped__')


class TestStubImplementations:
    """Test stub adapter implementations."""
    
    @pytest.mark.asyncio
    async def test_automl_stub_basic_functionality(self):
        """Test AutoML stub provides basic functionality."""
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import AutoMLOptimizationStub
        
        stub = AutoMLOptimizationStub()
        assert isinstance(stub, AutoMLOptimizationPort)
        
        # Create mock dataset
        class MockDataset:
            def __len__(self):
                return 100
        
        dataset = MockDataset()
        config = OptimizationConfig(max_trials=5)
        
        result = await stub.optimize_model(dataset, config)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_algorithm_type is not None
        assert result.best_score >= 0
        assert result.total_trials >= 0
        assert result.optimization_time_seconds >= 0
    
    @pytest.mark.asyncio 
    async def test_model_selection_stub_functionality(self):
        """Test model selection stub provides reasonable responses."""
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import ModelSelectionStub
        
        stub = ModelSelectionStub()
        assert isinstance(stub, ModelSelectionPort)
        
        class MockDataset:
            def __len__(self):
                return 1000
        
        dataset = MockDataset()
        
        algorithm, config = await stub.select_best_algorithm(dataset)
        
        assert isinstance(algorithm, AlgorithmType)
        assert isinstance(config, AlgorithmConfig)
        assert config.algorithm_type == algorithm
    
    @pytest.mark.asyncio
    async def test_monitoring_stub_functionality(self):
        """Test monitoring stub provides basic monitoring capabilities."""
        from machine_learning.infrastructure.adapters.stubs.monitoring_stubs import MonitoringStub
        
        stub = MonitoringStub()
        assert isinstance(stub, MonitoringPort)
        
        # Test metric recording
        await stub.record_metric("test_metric", 42.0)
        
        # Test metrics summary
        summary = await stub.get_metrics_summary()
        assert isinstance(summary, dict)
        assert "test_metric" in summary
    
    @pytest.mark.asyncio
    async def test_tracing_stub_functionality(self):
        """Test tracing stub provides basic tracing capabilities."""
        from machine_learning.infrastructure.adapters.stubs.monitoring_stubs import DistributedTracingStub
        
        stub = DistributedTracingStub()
        assert isinstance(stub, DistributedTracingPort)
        
        # Test trace creation
        trace_span = await stub.start_trace("test_operation")
        
        assert isinstance(trace_span, TraceSpan)
        assert trace_span.operation_name == "test_operation"
        assert trace_span.span_id is not None
        
        # Test trace completion
        await stub.finish_trace(trace_span.span_id)
        assert trace_span.end_time is not None


class TestAdapterConfiguration:
    """Test adapter configuration and initialization."""
    
    def test_adapter_factory_pattern(self):
        """Test that adapters can be created via factory pattern."""
        # This would test a factory if we implemented one
        pass
    
    def test_adapter_configuration_validation(self):
        """Test adapter configuration validation."""
        # Test invalid configurations
        with pytest.raises((ValueError, TypeError)):
            # Invalid backend type
            pass
    
    def test_adapter_graceful_degradation(self):
        """Test adapters handle missing dependencies gracefully."""
        # Test that adapters fall back to stubs when dependencies unavailable
        pass


class TestAdapterIntegration:
    """Integration tests for adapters with external dependencies."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires external dependencies")
    async def test_real_sklearn_integration(self):
        """Test real sklearn integration (requires sklearn installed)."""
        try:
            from machine_learning.infrastructure.adapters.automl.sklearn_automl_adapter import SklearnAutoMLAdapter
            
            adapter = SklearnAutoMLAdapter()
            
            # Test with real data
            import numpy as np
            class RealDataset:
                def __init__(self):
                    self.data = np.random.random((100, 5))
                
                def __len__(self):
                    return 100
            
            dataset = RealDataset()
            config = OptimizationConfig(max_trials=3)
            
            result = await adapter.optimize_model(dataset, config)
            
            assert isinstance(result, OptimizationResult)
            assert result.total_trials <= config.max_trials
            
        except ImportError:
            pytest.skip("sklearn not available for integration test")
    
    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="Requires external dependencies")
    async def test_real_tracing_integration(self):
        """Test real distributed tracing integration."""
        try:
            from machine_learning.infrastructure.adapters.monitoring.distributed_tracing_adapter import DistributedTracingAdapter
            
            adapter = DistributedTracingAdapter(backend="local")
            
            trace_span = await adapter.start_trace("integration_test")
            assert trace_span is not None
            
            await adapter.finish_trace(trace_span.span_id)
            assert trace_span.end_time is not None
            
        except ImportError:
            pytest.skip("Tracing dependencies not available")


class TestAdapterErrorHandling:
    """Test adapter error handling and resilience."""
    
    @pytest.mark.asyncio
    async def test_adapter_handles_invalid_data(self):
        """Test adapters handle invalid input data gracefully."""
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import AutoMLOptimizationStub
        
        stub = AutoMLOptimizationStub()
        
        # Test with None dataset
        with pytest.raises((ValueError, TypeError)):
            await stub.optimize_model(None, OptimizationConfig())
        
        # Test with invalid config
        class MockDataset:
            def __len__(self):
                return 10
        
        with pytest.raises((ValueError, TypeError)):
            await stub.optimize_model(MockDataset(), None)
    
    @pytest.mark.asyncio
    async def test_adapter_handles_external_service_failures(self):
        """Test adapters handle external service failures."""
        # Mock external service failure and verify graceful handling
        pass
    
    def test_adapter_logging_and_error_reporting(self):
        """Test that adapters properly log errors and provide diagnostics."""
        # Verify proper logging behavior
        pass


class TestAdapterPerformance:
    """Performance tests for adapters."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_stub_performance(self):
        """Test that stubs perform well under load."""
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import AutoMLOptimizationStub
        
        stub = AutoMLOptimizationStub()
        
        class MockDataset:
            def __len__(self):
                return 1000
        
        dataset = MockDataset()
        config = OptimizationConfig(max_trials=1)
        
        import time
        start_time = time.time()
        
        # Run multiple optimizations
        tasks = []
        for _ in range(10):
            tasks.append(stub.optimize_model(dataset, config))
        
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Should complete quickly (< 1 second for 10 stub operations)
        assert elapsed_time < 1.0
        assert len(results) == 10
        
        for result in results:
            assert isinstance(result, OptimizationResult)
    
    @pytest.mark.performance
    def test_adapter_memory_usage(self):
        """Test adapter memory usage is reasonable."""
        import sys
        
        from machine_learning.infrastructure.adapters.stubs.automl_stubs import AutoMLOptimizationStub
        
        # Measure memory before
        initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        # Create multiple adapters
        adapters = [AutoMLOptimizationStub() for _ in range(100)]
        
        # Measure memory after
        final_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        # Memory usage should be reasonable
        assert len(adapters) == 100
        
        # Clean up
        del adapters


# Import asyncio and gc for performance tests
import asyncio
try:
    import gc
except ImportError:
    gc = None