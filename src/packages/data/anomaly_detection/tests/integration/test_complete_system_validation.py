"""Complete system validation tests - end-to-end testing of the entire anomaly detection system."""

import pytest
import numpy as np
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.infrastructure.monitoring import get_metrics_collector, get_health_checker
from anomaly_detection.infrastructure.logging.error_handler import ErrorHandler, InputValidationError
from anomaly_detection.infrastructure.validation.comprehensive_validators import ComprehensiveValidator
from anomaly_detection.infrastructure.api.response_utilities import ResponseBuilder
from anomaly_detection.infrastructure.middleware.comprehensive_error_middleware import (
    ComprehensiveErrorMiddleware, RateLimiter, SecurityValidator
)


@pytest.fixture
def detection_service():
    """Create detection service for testing."""
    return DetectionService()


@pytest.fixture
def ensemble_service():
    """Create ensemble service for testing."""
    return EnsembleService()


@pytest.fixture
def streaming_service():
    """Create streaming service for testing."""
    return StreamingService(window_size=100, update_frequency=50)


@pytest.fixture
def model_repository():
    """Create model repository for testing."""
    return ModelRepository()


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 5))
    anomaly_data = np.random.normal(3, 0.5, (10, 5))
    return np.vstack([normal_data, anomaly_data])


@pytest.fixture
def streaming_data():
    """Create streaming dataset generator."""
    def generate_stream(n_samples=50):
        np.random.seed(123)
        for i in range(n_samples):
            if i % 10 == 0:  # Inject anomalies
                yield np.random.normal(4, 0.3, 5)
            else:
                yield np.random.normal(0, 1, 5)
    return generate_stream


class TestCompleteSystemValidation:
    """Complete system validation tests."""
    
    def test_end_to_end_detection_workflow(self, detection_service, sample_data):
        """Test complete detection workflow from data input to results."""
        # Test basic detection
        result = detection_service.detect_anomalies(
            data=sample_data,
            algorithm="iforest",
            contamination=0.1
        )
        
        # Validate result structure
        assert isinstance(result, DetectionResult)
        assert result.success is True
        assert result.total_samples == 110
        assert result.anomaly_count > 0
        assert result.anomaly_rate > 0
        assert len(result.predictions) == 110
        assert result.algorithm == "iforest"
        
        # Validate anomaly detection quality
        # Anomalies should be in the last 10 samples (indices 100-109)
        detected_anomalies = set(result.anomalies)
        expected_anomaly_range = set(range(100, 110))
        
        # Should detect at least 50% of injected anomalies
        overlap = detected_anomalies.intersection(expected_anomaly_range)
        precision = len(overlap) / len(detected_anomalies) if detected_anomalies else 0
        recall = len(overlap) / len(expected_anomaly_range)
        
        assert precision > 0.3, f"Low precision: {precision}"
        assert recall > 0.3, f"Low recall: {recall}"
        
        # Test with confidence scores
        if result.confidence_scores is not None:
            assert len(result.confidence_scores) == 110
            assert all(0 <= score <= 1 for score in result.confidence_scores)
    
    def test_ensemble_detection_workflow(self, ensemble_service, sample_data):
        """Test complete ensemble detection workflow."""
        algorithms = ["iforest", "lof"]
        
        result = ensemble_service.detect_ensemble(
            data=sample_data,
            algorithms=algorithms,
            method="majority",
            contamination=0.1
        )
        
        # Validate ensemble result
        assert isinstance(result, DetectionResult)
        assert result.success is True
        assert result.algorithm == "ensemble_majority"
        assert result.total_samples == 110
        assert result.anomaly_count > 0
        
        # Test weighted ensemble
        weights = [0.6, 0.4]
        weighted_result = ensemble_service.detect_ensemble(
            data=sample_data,
            algorithms=algorithms,
            method="weighted_average",
            weights=weights,
            contamination=0.1
        )
        
        assert isinstance(weighted_result, DetectionResult)
        assert weighted_result.success is True
        assert weighted_result.algorithm == "ensemble_weighted_average"
    
    @pytest.mark.asyncio
    async def test_streaming_detection_workflow(self, streaming_service, streaming_data):
        """Test complete streaming detection workflow."""
        results = []
        
        # Process streaming data
        for sample in streaming_data(30):
            result = streaming_service.process_sample(sample)
            results.append(result)
        
        # Validate streaming results
        assert len(results) == 30
        
        # Check that all results are valid
        for result in results:
            assert isinstance(result, DetectionResult)
            assert result.total_samples >= 1
        
        # Get streaming statistics
        stats = streaming_service.get_streaming_stats()
        
        assert stats["total_samples"] == 30
        assert stats["buffer_size"] <= 100  # Window size limit
        assert stats["update_frequency"] == 50
        
        # Test concept drift detection
        drift_result = streaming_service.detect_concept_drift()
        
        assert "drift_detected" in drift_result
        assert "buffer_size" in drift_result
        assert isinstance(drift_result["drift_detected"], bool)
    
    def test_model_persistence_workflow(self, detection_service, model_repository, sample_data):
        """Test complete model training and persistence workflow."""
        # Train and save model
        model_id = "test_model_123"
        
        # Fit the model
        detection_service.fit(
            data=sample_data,
            algorithm="iforest",
            contamination=0.1
        )
        
        # Create a model instance to save
        from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus
        
        metadata = ModelMetadata(
            algorithm="iforest",
            contamination=0.1,
            training_samples=110,
            feature_count=5,
            performance_metrics={"precision": 0.8, "recall": 0.7},
            created_at=datetime.utcnow(),
            version="1.0",
            status=ModelStatus.TRAINED
        )
        
        model = Model(
            id=model_id,
            metadata=metadata,
            model_data={"fitted": True}  # Simplified model data
        )
        
        # Save model
        saved_path = model_repository.save(model)
        assert saved_path is not None
        
        # Load and validate model
        loaded_model = model_repository.load(model_id)
        assert loaded_model.id == model_id
        assert loaded_model.metadata.algorithm == "iforest"
        assert loaded_model.metadata.training_samples == 110
        
        # Test model listing
        models = model_repository.list_models()
        model_ids = [m["id"] for m in models]
        assert model_id in model_ids
        
        # Cleanup
        model_repository.delete(model_id)
    
    def test_error_handling_workflow(self, detection_service):
        """Test complete error handling workflow."""
        error_handler = ErrorHandler()
        
        # Test input validation error
        try:
            detection_service.detect_anomalies(
                data=np.array([]),  # Empty data
                algorithm="iforest",
                contamination=0.1
            )
            pytest.fail("Should have raised an error for empty data")
        except Exception as e:
            handled_error = error_handler.handle_error(
                error=e,
                context={"operation": "test_detection"},
                operation="test_workflow",
                reraise=False
            )
            
            assert handled_error is not None
            assert handled_error.recoverable is True or handled_error.recoverable is False
            
            # Test error response creation
            error_response = error_handler.create_error_response(handled_error)
            assert error_response["success"] is False
            assert "error" in error_response
            assert "message" in error_response["error"]
    
    def test_comprehensive_validation_workflow(self, sample_data):
        """Test complete validation workflow."""
        validator = ComprehensiveValidator()
        
        # Test valid detection request
        result = validator.validate_detection_request(
            data=sample_data,
            algorithm="iforest",
            contamination=0.1,
            parameters={"n_estimators": 100, "random_state": 42}
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Test invalid detection request
        invalid_result = validator.validate_detection_request(
            data=np.random.rand(5, 3),  # Too few samples
            algorithm="unknown_algo",
            contamination=0.8,  # Too high
            parameters={"invalid_param": "value"}
        )
        
        assert invalid_result.is_valid is False
        assert len(invalid_result.errors) > 0
        
        # Test ensemble validation
        ensemble_result = validator.validate_ensemble_request(
            data=sample_data,
            algorithms=["iforest", "lof"],
            method="majority",
            contamination=0.1
        )
        
        assert ensemble_result.is_valid is True
        assert len(ensemble_result.errors) == 0
    
    def test_api_response_workflow(self):
        """Test complete API response workflow."""
        builder = ResponseBuilder(request_id="test-123")
        
        # Test success response
        success_response = builder.success(
            data={"anomalies": [1, 5, 10], "total": 100},
            message="Detection completed successfully"
        )
        
        assert success_response.success is True
        assert success_response.status == "success"
        assert success_response.request_id == "test-123"
        assert success_response.data["anomalies"] == [1, 5, 10]
        
        # Test error response
        error_response = builder.error(
            error="Data validation failed",
            message="Invalid input data"
        )
        
        assert error_response.success is False
        assert error_response.status == "error"
        assert len(error_response.errors) > 0
        
        # Test validation error response
        from anomaly_detection.infrastructure.api.response_utilities import ValidationError
        
        validation_errors = [
            ValidationError(field="data", value=[], message="Data cannot be empty"),
            ValidationError(field="algorithm", value="unknown", message="Unknown algorithm")
        ]
        
        validation_response = builder.validation_error(validation_errors)
        
        assert validation_response.success is False
        assert len(validation_response.errors) == 2
        assert "validation_errors" in validation_response.metadata
    
    @pytest.mark.asyncio
    async def test_middleware_workflow(self):
        """Test complete middleware workflow."""
        from fastapi import Request, Response
        from unittest.mock import AsyncMock
        
        # Create middleware with test configuration
        middleware = ComprehensiveErrorMiddleware(
            app=Mock(),
            enable_detailed_errors=True
        )
        
        # Create mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/api/v1/detect"
        request.headers = {"content-type": "application/json", "content-length": "100"}
        request.query_params = {}
        request.state = Mock()
        request.client = Mock()
        request.client.host = "192.168.1.100"
        
        # Create mock response
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        
        # Create mock call_next
        async def call_next(req):
            return mock_response
        
        # Test successful request
        result = await middleware.dispatch(request, call_next)
        
        assert result == mock_response
        assert "X-Request-ID" in mock_response.headers
        
        # Test middleware statistics
        stats = middleware.get_middleware_stats()
        assert "total_requests" in stats
        assert "total_errors" in stats
        assert stats["total_requests"] >= 1
    
    @pytest.mark.asyncio 
    async def test_rate_limiting_workflow(self):
        """Test complete rate limiting workflow."""
        rate_limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        client_id = "test_client"
        
        # Test allowed requests
        for i in range(3):
            allowed, info = await rate_limiter.is_allowed(client_id)
            assert allowed is True
            assert info["remaining"] == 2 - i
        
        # Test blocked request
        blocked, info = await rate_limiter.is_allowed(client_id)
        
        assert blocked is False
        assert info["remaining"] == 0
        assert info["current_count"] == 3
    
    def test_monitoring_integration_workflow(self):
        """Test complete monitoring integration workflow."""
        metrics_collector = get_metrics_collector()
        health_checker = get_health_checker()
        
        # Test metrics collection
        operation_id = metrics_collector.start_operation("test_detection")
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        duration = metrics_collector.end_operation(
            operation_id, 
            success=True,
            context={"algorithm": "iforest", "samples": 100}
        )
        
        assert duration > 0
        
        # Test metrics recording
        metrics_collector.record_metric("test_metric", 42.0, {"unit": "count"})
        metrics_collector.increment_counter("test_counter", 5)
        metrics_collector.set_gauge("test_gauge", 3.14)
        
        # Get summary stats
        stats = metrics_collector.get_summary_stats()
        
        assert "total_metrics" in stats
        assert "counters" in stats
        assert "gauges" in stats
        
        # Test cleanup
        cleanup_stats = metrics_collector.cleanup_old_metrics()
        assert "total_removed" in cleanup_stats
    
    def test_complete_data_pipeline_workflow(self, sample_data):
        """Test complete data pipeline from input to output."""
        # Step 1: Data validation
        validator = ComprehensiveValidator()
        
        validation_result = validator.validate_detection_request(
            data=sample_data,
            algorithm="iforest",
            contamination=0.1
        )
        
        assert validation_result.is_valid is True
        
        # Step 2: Detection
        detection_service = DetectionService()
        
        detection_result = detection_service.detect_anomalies(
            data=sample_data,
            algorithm="iforest",
            contamination=0.1
        )
        
        assert detection_result.success is True
        
        # Step 3: Response formatting
        builder = ResponseBuilder(request_id="pipeline-test")
        
        api_response = builder.success(
            data={
                "anomalies": detection_result.anomalies,
                "total_samples": detection_result.total_samples,
                "anomaly_count": detection_result.anomaly_count,
                "anomaly_rate": detection_result.anomaly_rate,
                "algorithm": detection_result.algorithm
            },
            message="Detection completed successfully"
        )
        
        assert api_response.success is True
        assert api_response.data["total_samples"] == 110
        assert api_response.data["anomaly_count"] > 0
        
        # Step 4: Validation of output format
        response_dict = api_response.dict()
        
        required_fields = ["status", "success", "data", "timestamp", "request_id"]
        for field in required_fields:
            assert field in response_dict
        
        # Step 5: Error handling validation
        try:
            # Intentionally cause an error
            detection_service.detect_anomalies(
                data=np.array([]),  # Empty data
                algorithm="iforest",
                contamination=0.1
            )
        except Exception as e:
            error_response = builder.error(
                error=e,
                message="Detection failed due to invalid input"
            )
            
            assert error_response.success is False
            assert len(error_response.errors) > 0


class TestSecurityValidation:
    """Security validation tests."""
    
    @pytest.mark.asyncio
    async def test_security_validation_workflow(self):
        """Test complete security validation workflow."""
        security_validator = SecurityValidator()
        
        # Test valid request
        valid_request = Mock()
        valid_request.headers = {"content-length": "100"}
        valid_request.client = Mock()
        valid_request.client.host = "192.168.1.1"
        valid_request.url = Mock()
        valid_request.url.__str__ = Mock(return_value="https://api.example.com/detect")
        
        result = await security_validator.validate_request(valid_request)
        assert result is None
        
        # Test blocked IP
        blocked_request = Mock()
        blocked_request.headers = {}
        blocked_request.client = Mock()
        blocked_request.client.host = "192.168.1.100"
        
        security_validator.block_ip("192.168.1.100")
        
        result = await security_validator.validate_request(blocked_request)
        assert result is not None
        assert result["error"] == "blocked_ip"
        
        # Test payload too large
        large_payload_request = Mock()
        large_payload_request.headers = {"content-length": str(20 * 1024 * 1024)}  # 20MB
        large_payload_request.client = Mock()
        large_payload_request.client.host = "192.168.1.1"
        
        result = await security_validator.validate_request(large_payload_request)
        assert result is not None
        assert result["error"] == "payload_too_large"


class TestPerformanceValidation:
    """Performance validation tests."""
    
    def test_algorithm_performance_comparison(self, sample_data):
        """Test performance comparison between algorithms."""
        detection_service = DetectionService()
        algorithms = ["iforest", "lof"]
        
        performance_results = {}
        
        for algorithm in algorithms:
            start_time = datetime.utcnow()
            
            result = detection_service.detect_anomalies(
                data=sample_data,
                algorithm=algorithm,
                contamination=0.1
            )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            performance_results[algorithm] = {
                "duration_seconds": duration,
                "anomaly_count": result.anomaly_count,
                "success": result.success
            }
        
        # Validate all algorithms completed successfully
        for algorithm, perf in performance_results.items():
            assert perf["success"] is True
            assert perf["anomaly_count"] > 0
            assert perf["duration_seconds"] < 10.0  # Should complete within 10 seconds
        
        # Isolation Forest should generally be faster than LOF for this data size
        assert performance_results["iforest"]["duration_seconds"] <= performance_results["lof"]["duration_seconds"] * 2
    
    def test_streaming_performance_validation(self, streaming_service, streaming_data):
        """Test streaming performance validation."""
        processing_times = []
        
        for sample in streaming_data(20):
            start_time = datetime.utcnow()
            result = streaming_service.process_sample(sample)
            end_time = datetime.utcnow()
            
            processing_time = (end_time - start_time).total_seconds()
            processing_times.append(processing_time)
            
            assert result is not None
            assert processing_time < 1.0  # Each sample should process within 1 second
        
        # Validate average processing time
        avg_processing_time = sum(processing_times) / len(processing_times)
        assert avg_processing_time < 0.1  # Average should be under 100ms
        
        # Get memory statistics
        memory_stats = streaming_service.get_memory_stats()
        
        assert memory_stats["current_memory_mb"] > 0
        assert memory_stats["memory_usage_ratio"] < 1.0  # Should not exceed memory limit


@pytest.mark.integration
class TestRealWorldScenarios:
    """Real-world scenario validation tests."""
    
    def test_financial_fraud_detection_scenario(self):
        """Test financial fraud detection scenario."""
        # Create realistic financial transaction data
        np.random.seed(42)
        
        # Normal transactions: amount, merchant_category, time_of_day, location_risk
        normal_transactions = np.random.multivariate_normal(
            mean=[50, 1, 12, 0.1],
            cov=[[100, 0, 0, 0], [0, 1, 0, 0], [0, 0, 36, 0], [0, 0, 0, 0.01]],
            size=1000
        )
        
        # Fraudulent transactions: higher amounts, unusual times, higher risk
        fraud_transactions = np.random.multivariate_normal(
            mean=[500, 3, 3, 0.8],
            cov=[[10000, 0, 0, 0], [0, 1, 0, 0], [0, 0, 4, 0], [0, 0, 0, 0.04]],
            size=50
        )
        
        # Combine datasets
        all_transactions = np.vstack([normal_transactions, fraud_transactions])
        true_labels = np.concatenate([np.ones(1000), -np.ones(50)])  # 1=normal, -1=fraud
        
        # Run detection
        detection_service = DetectionService()
        
        result = detection_service.detect_anomalies(
            data=all_transactions,
            algorithm="iforest",
            contamination=0.05  # Expect 5% fraud
        )
        
        # Validate results
        assert result.success is True
        assert result.total_samples == 1050
        
        # Calculate detection metrics
        detected_anomalies = set(result.anomalies)
        true_fraud_indices = set(range(1000, 1050))
        
        true_positives = len(detected_anomalies.intersection(true_fraud_indices))
        false_positives = len(detected_anomalies - true_fraud_indices)
        false_negatives = len(true_fraud_indices - detected_anomalies)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Should achieve reasonable performance on this synthetic data
        assert precision > 0.2, f"Low precision: {precision}"
        assert recall > 0.2, f"Low recall: {recall}"
    
    def test_network_intrusion_detection_scenario(self):
        """Test network intrusion detection scenario."""
        # Create network traffic data: packet_size, duration, protocol, port
        np.random.seed(123)
        
        # Normal traffic
        normal_traffic = np.random.multivariate_normal(
            mean=[1500, 0.1, 80, 80],
            cov=[[250000, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 100, 0], [0, 0, 0, 400]],
            size=800
        )
        
        # Intrusion attempts: large packets, long duration, unusual ports
        intrusion_traffic = np.random.multivariate_normal(
            mean=[8000, 2.0, 443, 8080],
            cov=[[1000000, 0, 0, 0], [0, 1, 0, 0], [0, 0, 100, 0], [0, 0, 0, 10000]],
            size=40
        )
        
        all_traffic = np.vstack([normal_traffic, intrusion_traffic])
        
        # Use ensemble detection for better coverage
        ensemble_service = EnsembleService()
        
        result = ensemble_service.detect_ensemble(
            data=all_traffic,
            algorithms=["iforest", "lof"],
            method="majority",
            contamination=0.05
        )
        
        assert result.success is True
        assert result.total_samples == 840
        assert result.anomaly_count > 0
        
        # Should detect some intrusions
        detected_anomalies = set(result.anomalies)
        true_intrusion_indices = set(range(800, 840))
        
        overlap = detected_anomalies.intersection(true_intrusion_indices)
        detection_rate = len(overlap) / len(true_intrusion_indices)
        
        assert detection_rate > 0.1, f"Low intrusion detection rate: {detection_rate}"
    
    def test_iot_sensor_anomaly_scenario(self, streaming_service):
        """Test IoT sensor anomaly detection scenario."""
        # Simulate IoT sensor data stream with periodic anomalies
        def iot_sensor_stream():
            np.random.seed(456)
            
            for i in range(100):
                if i % 20 == 0:  # Periodic sensor malfunction
                    # Sensor reading spikes
                    yield np.array([25.0, 85.0, 1013.0, 60.0, 500.0])  # temp, humidity, pressure, light, vibration
                elif i % 50 == 0:  # Rare complete sensor failure
                    # All readings go to zero
                    yield np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    # Normal sensor readings with some noise
                    base_reading = np.array([20.0, 50.0, 1015.0, 300.0, 10.0])
                    noise = np.random.normal(0, 0.1, 5)
                    yield base_reading + noise
        
        # Process IoT stream
        anomaly_count = 0
        results = []
        
        for sample in iot_sensor_stream():
            result = streaming_service.process_sample(sample, algorithm="lof")
            results.append(result)
            
            if result.predictions[0] == -1:  # Anomaly detected
                anomaly_count += 1
        
        # Validate streaming results
        assert len(results) == 100
        assert anomaly_count > 0  # Should detect some anomalies
        
        # Check streaming statistics
        stats = streaming_service.get_streaming_stats()
        assert stats["total_samples"] == 100
        assert stats["avg_processing_time_ms"] >= 0
        
        # Test concept drift detection
        drift_result = streaming_service.detect_concept_drift()
        assert "drift_detected" in drift_result
        assert isinstance(drift_result["drift_detected"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])