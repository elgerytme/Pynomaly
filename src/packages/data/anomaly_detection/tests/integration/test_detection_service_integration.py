"""Integration tests for DetectionService."""

import pytest
import numpy as np
from typing import Dict, Any

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.infrastructure.logging.error_handler import InputValidationError, AlgorithmError
from .conftest import assert_detection_result_valid


class TestDetectionServiceIntegration:
    """Integration tests for DetectionService with real algorithms."""
    
    def test_isolation_forest_detection(self, detection_service: DetectionService, test_data: Dict[str, Any]):
        """Test Isolation Forest detection with real data."""
        data = test_data['data_only']
        
        result = detection_service.detect_anomalies(
            data=data,
            algorithm="iforest",
            contamination=0.1
        )
        
        assert_detection_result_valid(result, test_data['n_samples'], "iforest")
        
        # Should detect some anomalies
        assert result.anomaly_count > 0
        assert result.anomaly_rate > 0
        
        # Anomaly rate should be close to contamination rate
        assert 0.05 <= result.anomaly_rate <= 0.15  # Allow some variance
    
    def test_local_outlier_factor_detection(self, detection_service: DetectionService, test_data: Dict[str, Any]):
        """Test Local Outlier Factor detection with real data."""
        data = test_data['data_only']
        
        result = detection_service.detect_anomalies(
            data=data,
            algorithm="lof",
            contamination=0.1
        )
        
        assert_detection_result_valid(result, test_data['n_samples'], "lof")
        
        # Should detect some anomalies
        assert result.anomaly_count > 0
    
    def test_multiple_algorithms_consistency(self, detection_service: DetectionService, test_data: Dict[str, Any]):
        """Test that different algorithms produce reasonable results on same data."""
        data = test_data['data_only']
        algorithms = ["iforest", "lof"]
        
        results = {}
        for algorithm in algorithms:
            results[algorithm] = detection_service.detect_anomalies(
                data=data,
                algorithm=algorithm,
                contamination=0.1
            )
        
        # All algorithms should succeed
        for algorithm, result in results.items():
            assert_detection_result_valid(result, test_data['n_samples'], algorithm)
            assert result.anomaly_count > 0
        
        # Results should be somewhat consistent (within reasonable range)
        anomaly_counts = [result.anomaly_count for result in results.values()]
        min_count, max_count = min(anomaly_counts), max(anomaly_counts)
        
        # Allow up to 50% variation between algorithms
        assert max_count <= min_count * 1.5
    
    def test_fit_and_predict_workflow(self, detection_service: DetectionService, test_data: Dict[str, Any]):
        """Test the fit and predict workflow."""
        train_data = test_data['normal_data']  # Train on normal data only
        test_data_array = test_data['data_only']  # Test on mixed data
        
        # Fit the model
        detection_service.fit(train_data, algorithm="iforest")
        
        # Make predictions
        result = detection_service.predict(test_data_array, algorithm="iforest")
        
        assert_detection_result_valid(result, test_data['n_samples'], "iforest")
        
        # Should detect the injected anomalies
        assert result.anomaly_count > 0
    
    def test_different_contamination_rates(self, detection_service: DetectionService, test_data: Dict[str, Any]):
        """Test detection with different contamination rates."""
        data = test_data['data_only']
        contamination_rates = [0.05, 0.1, 0.2]
        
        results = []
        for contamination in contamination_rates:
            result = detection_service.detect_anomalies(
                data=data,
                algorithm="iforest",
                contamination=contamination
            )
            results.append(result)
        
        # Higher contamination should generally detect more anomalies
        anomaly_counts = [r.anomaly_count for r in results]
        
        # Check that anomaly counts generally increase with contamination
        # (allowing for some variance in algorithm behavior)
        assert anomaly_counts[2] >= anomaly_counts[0]  # 0.2 >= 0.05
    
    def test_algorithm_parameters(self, detection_service: DetectionService, test_data: Dict[str, Any]):
        """Test detection with different algorithm parameters."""
        data = test_data['data_only']
        
        # Test Isolation Forest with different parameters
        result1 = detection_service.detect_anomalies(
            data=data,
            algorithm="iforest",
            contamination=0.1,
            n_estimators=50,
            random_state=42
        )
        
        result2 = detection_service.detect_anomalies(
            data=data,
            algorithm="iforest",
            contamination=0.1,
            n_estimators=200,
            random_state=42
        )
        
        # Both should succeed
        assert_detection_result_valid(result1, test_data['n_samples'], "iforest")
        assert_detection_result_valid(result2, test_data['n_samples'], "iforest")
        
        # With same random state, results should be similar but not necessarily identical
        # due to different n_estimators
        assert result1.anomaly_count > 0
        assert result2.anomaly_count > 0
    
    def test_edge_cases(self, detection_service: DetectionService):
        """Test detection with edge cases."""
        # Very small dataset
        small_data = np.array([[1, 2], [2, 3]], dtype=np.float64)
        
        result = detection_service.detect_anomalies(
            data=small_data,
            algorithm="iforest",
            contamination=0.5
        )
        
        assert_detection_result_valid(result, 2, "iforest")
    
    def test_error_handling(self, detection_service: DetectionService):
        """Test error handling for invalid inputs."""
        # Empty data
        with pytest.raises(InputValidationError):
            detection_service.detect_anomalies(
                data=np.array([]).reshape(0, 2),
                algorithm="iforest"
            )
        
        # Invalid contamination rate
        with pytest.raises(InputValidationError):
            detection_service.detect_anomalies(
                data=np.array([[1, 2], [3, 4]], dtype=np.float64),
                algorithm="iforest",
                contamination=1.5  # Invalid: > 1
            )
        
        # Unknown algorithm
        with pytest.raises(AlgorithmError):
            detection_service.detect_anomalies(
                data=np.array([[1, 2], [3, 4]], dtype=np.float64),
                algorithm="nonexistent_algorithm"
            )
        
        # 1D data (should be 2D)
        with pytest.raises(InputValidationError):
            detection_service.detect_anomalies(
                data=np.array([1, 2, 3, 4]),
                algorithm="iforest"
            )
    
    def test_large_dataset_performance(self, detection_service: DetectionService, large_dataset: np.ndarray):
        """Test detection on larger dataset for performance."""
        result = detection_service.detect_anomalies(
            data=large_dataset,
            algorithm="iforest",
            contamination=0.1
        )
        
        assert_detection_result_valid(result, 1000, "iforest")
        
        # Should detect reasonable number of anomalies
        assert 50 <= result.anomaly_count <= 150  # Allow range around 10%
    
    def test_reproducibility(self, detection_service: DetectionService, test_data: Dict[str, Any]):
        """Test that detection results are reproducible with same random state."""
        data = test_data['data_only']
        
        result1 = detection_service.detect_anomalies(
            data=data,
            algorithm="iforest",
            contamination=0.1,
            random_state=42
        )
        
        result2 = detection_service.detect_anomalies(
            data=data,
            algorithm="iforest",
            contamination=0.1,
            random_state=42
        )
        
        # Results should be identical with same random state
        assert np.array_equal(result1.predictions, result2.predictions)
        assert result1.anomaly_count == result2.anomaly_count
    
    def test_algorithm_availability(self, detection_service: DetectionService):
        """Test algorithm availability and information."""
        available_algorithms = detection_service.list_available_algorithms()
        
        # Should have built-in algorithms
        assert "iforest" in available_algorithms
        assert "lof" in available_algorithms
        
        # Test algorithm info
        iforest_info = detection_service.get_algorithm_info("iforest")
        assert iforest_info["name"] == "iforest"
        assert iforest_info["type"] == "builtin"
        assert "scikit-learn" in iforest_info["requires"]
    
    def test_service_state_management(self, detection_service: DetectionService, test_data: Dict[str, Any]):
        """Test that service manages fitted models correctly."""
        data = test_data['normal_data']
        
        # Initially no fitted models
        assert len(detection_service._fitted_models) == 0
        
        # Fit a model
        detection_service.fit(data, algorithm="iforest")
        
        # Should now have fitted model
        assert len(detection_service._fitted_models) == 1
        assert "iforest" in detection_service._fitted_models
        
        # Fit another algorithm
        detection_service.fit(data, algorithm="lof")
        
        # Should have both models
        assert len(detection_service._fitted_models) == 2
        assert "lof" in detection_service._fitted_models