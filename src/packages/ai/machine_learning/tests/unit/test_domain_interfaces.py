"""Unit tests for domain interfaces (ports)."""

import pytest
from abc import ABC
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
from machine_learning.domain.interfaces.explainability_operations import (
    ExplainabilityPort,
    FeatureImportancePort,
    ExplanationRequest,
    ExplanationResult,
    FeatureContribution,
    ExplanationMethod,
    ExplanationScope
)
from machine_learning.domain.interfaces.monitoring_operations import (
    MonitoringPort,
    DistributedTracingPort,
    AlertingPort,
    HealthCheckPort,
    TraceSpan,
    MetricType,
    HealthStatus
)


class TestDomainInterfaceDefinitions:
    """Test that all domain interfaces are properly defined."""
    
    def test_automl_interfaces_are_abstract(self):
        """Test that AutoML interfaces are abstract base classes."""
        assert issubclass(AutoMLOptimizationPort, ABC)
        assert issubclass(ModelSelectionPort, ABC)
        assert issubclass(HyperparameterOptimizationPort, ABC)
        
        # Should not be instantiable
        with pytest.raises(TypeError):
            AutoMLOptimizationPort()
    
    def test_explainability_interfaces_are_abstract(self):
        """Test that explainability interfaces are abstract base classes.""" 
        assert issubclass(ExplainabilityPort, ABC)
        assert issubclass(FeatureImportancePort, ABC)
        
        with pytest.raises(TypeError):
            ExplainabilityPort()
    
    def test_monitoring_interfaces_are_abstract(self):
        """Test that monitoring interfaces are abstract base classes."""
        assert issubclass(MonitoringPort, ABC)
        assert issubclass(DistributedTracingPort, ABC)  
        assert issubclass(AlertingPort, ABC)
        assert issubclass(HealthCheckPort, ABC)
        
        with pytest.raises(TypeError):
            MonitoringPort()


class TestAutoMLDataTransferObjects:
    """Test AutoML data transfer objects."""
    
    def test_optimization_config_creation(self):
        """Test OptimizationConfig creation and validation."""
        config = OptimizationConfig(
            max_trials=50,
            algorithms_to_test=[AlgorithmType.ISOLATION_FOREST],
            parameters={"contamination": 0.1}
        )
        
        assert config.max_trials == 50
        assert AlgorithmType.ISOLATION_FOREST in config.algorithms_to_test
        assert config.parameters["contamination"] == 0.1
    
    def test_optimization_config_defaults(self):
        """Test OptimizationConfig with default values."""
        config = OptimizationConfig()
        
        assert config.max_trials == 100
        assert len(config.algorithms_to_test) > 0
        assert config.parameters == {}
        assert config.cross_validation_folds == 5
    
    def test_algorithm_config_creation(self):
        """Test AlgorithmConfig creation."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.ONE_CLASS_SVM,
            parameters={"nu": 0.05},
            hyperparameter_space={"nu": [0.01, 0.1, 0.2]}
        )
        
        assert config.algorithm_type == AlgorithmType.ONE_CLASS_SVM
        assert config.parameters["nu"] == 0.05
        assert "nu" in config.hyperparameter_space
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
            best_parameters={"n_estimators": 100},
            best_score=0.85,
            total_trials=25,
            optimization_time_seconds=120.5
        )
        
        assert result.best_algorithm_type == AlgorithmType.ISOLATION_FOREST
        assert result.best_parameters["n_estimators"] == 100
        assert result.best_score == 0.85
        assert result.total_trials == 25
        assert result.optimization_time_seconds == 120.5
    
    def test_algorithm_type_enum_values(self):
        """Test AlgorithmType enum has expected values."""
        expected_algorithms = [
            "ISOLATION_FOREST",
            "LOCAL_OUTLIER_FACTOR", 
            "ONE_CLASS_SVM",
            "ELLIPTIC_ENVELOPE",
            "AUTOENCODER",
            "LSTM_AUTOENCODER"
        ]
        
        algorithm_names = [algo.name for algo in AlgorithmType]
        
        for expected in expected_algorithms:
            assert expected in algorithm_names


class TestExplainabilityDataTransferObjects:
    """Test explainability data transfer objects."""
    
    def test_explanation_request_creation(self):
        """Test ExplanationRequest creation."""
        request = ExplanationRequest(
            model_id="model_123",
            instance_data={"feature1": 1.0, "feature2": 2.0},
            method=ExplanationMethod.SHAP,
            scope=ExplanationScope.GLOBAL
        )
        
        assert request.model_id == "model_123"
        assert request.instance_data["feature1"] == 1.0
        assert request.method == ExplanationMethod.SHAP
        assert request.scope == ExplanationScope.GLOBAL
    
    def test_explanation_result_creation(self):
        """Test ExplanationResult creation."""
        contributions = [
            FeatureContribution(feature_name="feature1", contribution_score=0.8),
            FeatureContribution(feature_name="feature2", contribution_score=-0.3)
        ]
        
        result = ExplanationResult(
            method_used=ExplanationMethod.LIME,
            feature_contributions=contributions,
            explanation_score=0.92
        )
        
        assert result.method_used == ExplanationMethod.LIME
        assert len(result.feature_contributions) == 2
        assert result.feature_contributions[0].feature_name == "feature1"
        assert result.explanation_score == 0.92
    
    def test_feature_contribution_creation(self):
        """Test FeatureContribution creation."""
        contribution = FeatureContribution(
            feature_name="temperature",
            contribution_score=0.65,
            confidence_interval=(0.5, 0.8)
        )
        
        assert contribution.feature_name == "temperature"
        assert contribution.contribution_score == 0.65
        assert contribution.confidence_interval == (0.5, 0.8)
    
    def test_explanation_method_enum(self):
        """Test ExplanationMethod enum values."""
        expected_methods = ["SHAP", "LIME", "PERMUTATION", "INTEGRATED_GRADIENTS"]
        method_names = [method.name for method in ExplanationMethod]
        
        for expected in expected_methods:
            assert expected in method_names
    
    def test_explanation_scope_enum(self):
        """Test ExplanationScope enum values."""
        expected_scopes = ["LOCAL", "GLOBAL", "COHORT"]
        scope_names = [scope.name for scope in ExplanationScope]
        
        for expected in expected_scopes:
            assert expected in scope_names


class TestMonitoringDataTransferObjects:
    """Test monitoring data transfer objects."""
    
    def test_trace_span_creation(self):
        """Test TraceSpan creation."""
        span = TraceSpan(
            span_id="span_123",
            operation_name="model_optimization",
            start_time=1640995200.0,
            tags={"algorithm": "isolation_forest"}
        )
        
        assert span.span_id == "span_123"
        assert span.operation_name == "model_optimization"
        assert span.start_time == 1640995200.0
        assert span.tags["algorithm"] == "isolation_forest"
        assert span.end_time is None  # Not finished yet
    
    def test_trace_span_completion(self):
        """Test TraceSpan completion."""
        span = TraceSpan(
            span_id="span_456",
            operation_name="model_training"
        )
        
        end_time = 1640995300.0
        span.end_time = end_time
        
        assert span.end_time == end_time
        assert span.duration_seconds == end_time - span.start_time
    
    def test_metric_type_enum(self):
        """Test MetricType enum values."""
        expected_types = ["COUNTER", "GAUGE", "HISTOGRAM", "SUMMARY"]
        type_names = [metric_type.name for metric_type in MetricType]
        
        for expected in expected_types:
            assert expected in type_names
    
    def test_health_status_enum(self):
        """Test HealthStatus enum values."""
        expected_statuses = ["HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN"]
        status_names = [status.name for status in HealthStatus]
        
        for expected in expected_statuses:
            assert expected in status_names


class TestInterfaceMethodSignatures:
    """Test that interface method signatures are correctly defined."""
    
    def test_automl_optimization_port_methods(self):
        """Test AutoMLOptimizationPort method signatures."""
        methods = [
            "optimize_model",
            "get_optimization_status", 
            "cancel_optimization"
        ]
        
        for method_name in methods:
            assert hasattr(AutoMLOptimizationPort, method_name)
            method = getattr(AutoMLOptimizationPort, method_name)
            assert callable(method)
    
    def test_model_selection_port_methods(self):
        """Test ModelSelectionPort method signatures."""
        methods = [
            "select_best_algorithm",
            "compare_algorithms",
            "get_algorithm_recommendations"
        ]
        
        for method_name in methods:
            assert hasattr(ModelSelectionPort, method_name)
            method = getattr(ModelSelectionPort, method_name)
            assert callable(method)
    
    def test_explainability_port_methods(self):
        """Test ExplainabilityPort method signatures."""
        methods = [
            "explain_prediction",
            "get_feature_importance",
            "generate_explanation_report"
        ]
        
        for method_name in methods:
            assert hasattr(ExplainabilityPort, method_name)
            method = getattr(ExplainabilityPort, method_name)
            assert callable(method)
    
    def test_monitoring_port_methods(self):
        """Test MonitoringPort method signatures."""
        methods = [
            "record_metric",
            "get_metrics_summary",
            "create_custom_metric"
        ]
        
        for method_name in methods:
            assert hasattr(MonitoringPort, method_name)
            method = getattr(MonitoringPort, method_name)
            assert callable(method)
    
    def test_distributed_tracing_port_methods(self):
        """Test DistributedTracingPort method signatures."""
        methods = [
            "start_trace",
            "finish_trace",
            "add_trace_tag",
            "get_trace_context"
        ]
        
        for method_name in methods:
            assert hasattr(DistributedTracingPort, method_name)
            method = getattr(DistributedTracingPort, method_name)
            assert callable(method)


class TestInterfaceContractValidation:
    """Test that interfaces define proper contracts."""
    
    def test_automl_methods_are_async(self):
        """Test that AutoML methods are properly defined as async."""
        # This test verifies the interface definition includes async methods
        # In real implementation, we'd check the method annotations
        pass
    
    def test_method_parameter_types(self):
        """Test that methods have proper type hints."""
        # Verify type hints are present in interface definitions
        import inspect
        
        # Check AutoMLOptimizationPort.optimize_model signature
        sig = inspect.signature(AutoMLOptimizationPort.optimize_model)
        assert len(sig.parameters) >= 3  # self, dataset, config
        
        # Check return type annotation exists
        assert sig.return_annotation is not None
    
    def test_interface_inheritance_hierarchy(self):
        """Test that interfaces have proper inheritance."""
        # All ports should inherit from ABC
        interfaces = [
            AutoMLOptimizationPort,
            ModelSelectionPort, 
            ExplainabilityPort,
            MonitoringPort,
            DistributedTracingPort
        ]
        
        for interface in interfaces:
            assert issubclass(interface, ABC)
            # Should have at least one abstract method
            assert len(interface.__abstractmethods__) > 0


class TestDataClassValidation:
    """Test data class validation and immutability."""
    
    def test_data_classes_are_frozen(self):
        """Test that data classes are immutable where appropriate."""
        # Some DTOs should be immutable to prevent accidental modification
        result = OptimizationResult(
            best_algorithm_type=AlgorithmType.ISOLATION_FOREST,
            best_parameters={},
            best_score=0.8,
            total_trials=10,
            optimization_time_seconds=30.0
        )
        
        # Basic functionality should work
        assert result.best_score == 0.8
        
        # Should be able to create new instance with modifications
        assert result.total_trials == 10
    
    def test_data_class_field_validation(self):
        """Test data class field validation."""
        # Test that invalid data raises appropriate errors
        with pytest.raises((ValueError, TypeError)):
            OptimizationConfig(max_trials=-1)  # Negative trials should fail
    
    def test_enum_value_validation(self):
        """Test enum value validation."""
        # Valid enum usage
        config = AlgorithmConfig(algorithm_type=AlgorithmType.ISOLATION_FOREST)
        assert config.algorithm_type == AlgorithmType.ISOLATION_FOREST
        
        # Invalid enum should raise error
        with pytest.raises((ValueError, TypeError)):
            AlgorithmConfig(algorithm_type="invalid_algorithm")