"""Unit tests for EnsembleService."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List

from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.entities.detection_result import DetectionResult


class TestEnsembleService:
    """Test suite for EnsembleService."""
    
    @pytest.fixture
    def mock_detection_service(self):
        """Create mock detection service."""
        mock_service = Mock(spec=DetectionService)
        mock_service.list_available_algorithms.return_value = ["iforest", "lof", "ocsvm"]
        return mock_service
    
    @pytest.fixture
    def ensemble_service(self, mock_detection_service):
        """Create ensemble service with mock detection service."""
        return EnsembleService(detection_service=mock_detection_service)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.randn(100, 5).astype(np.float64)
    
    @pytest.fixture
    def mock_detection_results(self):
        """Create mock detection results."""
        # Algorithm 1: 20% anomalies (predictions: 1=normal, -1=anomaly)
        result1 = Mock(spec=DetectionResult)
        result1.predictions = np.array([1 if i % 5 != 0 else -1 for i in range(100)])
        result1.scores = np.random.rand(100).astype(np.float64)
        result1.algorithm = "iforest"
        
        # Algorithm 2: 30% anomalies  
        result2 = Mock(spec=DetectionResult)
        result2.predictions = np.array([1 if i % 3 != 0 else -1 for i in range(100)])
        result2.scores = np.random.rand(100).astype(np.float64)  
        result2.algorithm = "lof"
        
        return [result1, result2]
    
    def test_initialization_with_detection_service(self, mock_detection_service):
        """Test ensemble service initialization with provided detection service."""
        ensemble = EnsembleService(detection_service=mock_detection_service)
        assert ensemble.detection_service is mock_detection_service
    
    def test_initialization_without_detection_service(self):
        """Test ensemble service initialization without detection service."""
        ensemble = EnsembleService()
        assert isinstance(ensemble.detection_service, DetectionService)
    
    def test_detect_with_ensemble_default_algorithms(self, ensemble_service, sample_data, mock_detection_results):
        """Test ensemble detection with default algorithms."""
        # Setup mock to return our mock results
        ensemble_service.detection_service.detect_anomalies.side_effect = mock_detection_results
        
        result = ensemble_service.detect_with_ensemble(data=sample_data)
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "ensemble(iforest,lof)"
        assert len(result.predictions) == len(sample_data)
        assert result.metadata["algorithms"] == ["iforest", "lof"]
        assert result.metadata["combination_method"] == "majority"
        assert result.metadata["individual_results"] == 2
        
        # Verify detection service was called for each algorithm
        assert ensemble_service.detection_service.detect_anomalies.call_count == 2
    
    def test_detect_with_ensemble_custom_algorithms(self, ensemble_service, sample_data, mock_detection_results):
        """Test ensemble detection with custom algorithms."""
        algorithms = ["iforest", "lof", "ocsvm"]
        
        # Add third mock result
        result3 = Mock(spec=DetectionResult)
        result3.predictions = np.array([1 if i % 4 != 0 else -1 for i in range(100)])
        result3.scores = np.random.rand(100).astype(np.float64)
        result3.algorithm = "ocsvm"
        
        all_results = mock_detection_results + [result3]
        ensemble_service.detection_service.detect_anomalies.side_effect = all_results
        
        result = ensemble_service.detect_with_ensemble(
            data=sample_data,
            algorithms=algorithms
        )
        
        assert result.algorithm == "ensemble(iforest,lof,ocsvm)"
        assert result.metadata["algorithms"] == algorithms
        assert result.metadata["individual_results"] == 3
        assert ensemble_service.detection_service.detect_anomalies.call_count == 3
    
    def test_detect_with_ensemble_insufficient_algorithms(self, ensemble_service, sample_data):
        """Test ensemble detection with insufficient algorithms."""
        with pytest.raises(ValueError) as exc_info:
            ensemble_service.detect_with_ensemble(
                data=sample_data,
                algorithms=["iforest"]  # Only one algorithm
            )
        
        assert "Ensemble requires at least 2 algorithms" in str(exc_info.value)
    
    def test_detect_with_ensemble_all_algorithms_fail(self, ensemble_service, sample_data):
        """Test ensemble detection when all algorithms fail."""
        # Make all algorithm calls fail
        ensemble_service.detection_service.detect_anomalies.side_effect = Exception("Algorithm failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            ensemble_service.detect_with_ensemble(data=sample_data)
        
        assert "All algorithms failed in ensemble" in str(exc_info.value)
    
    def test_detect_with_ensemble_some_algorithms_fail(self, ensemble_service, sample_data, mock_detection_results):
        """Test ensemble detection when some algorithms fail."""
        # First algorithm succeeds, second fails
        def side_effect(*args, **kwargs):
            algorithm = kwargs.get('algorithm', args[1] if len(args) > 1 else 'unknown')
            if algorithm == "iforest":
                return mock_detection_results[0]
            else:
                raise Exception("Algorithm failed")
        
        ensemble_service.detection_service.detect_anomalies.side_effect = side_effect
        
        result = ensemble_service.detect_with_ensemble(data=sample_data)
        
        # Should still work with one successful algorithm
        assert isinstance(result, DetectionResult)
        assert result.metadata["individual_results"] == 1
    
    def test_combine_predictions_majority(self, ensemble_service):
        """Test majority voting prediction combination."""
        predictions_list = [
            np.array([1, -1, 1, -1, 1]),  # Algorithm 1
            np.array([1, 1, -1, -1, 1]),  # Algorithm 2  
            np.array([-1, 1, 1, -1, -1])  # Algorithm 3
        ]
        
        combined = ensemble_service._combine_predictions(
            predictions_list, method="majority"
        )
        
        # Expected: majority of [1,1,-1], [-1,1,1], [1,-1,1], [-1,-1,-1], [1,1,-1]
        # Results:   1,               1,           1,            -1,            1
        expected = np.array([1, 1, 1, 0, 1])  # Note: majority voting uses > 0.5 threshold
        np.testing.assert_array_equal(combined, expected)
    
    def test_combine_predictions_average(self, ensemble_service):
        """Test average prediction combination."""
        predictions_list = [
            np.array([1, -1, 1, -1]),  
            np.array([1, 1, -1, -1])
        ]
        
        combined = ensemble_service._combine_predictions(
            predictions_list, method="average"
        )
        
        # Average of predictions, then threshold at 0.5
        # [1,1] -> 1.0 > 0.5 -> 1
        # [-1,1] -> 0.0 = 0.5 -> 0  
        # [1,-1] -> 0.0 = 0.5 -> 0
        # [-1,-1] -> -1.0 < 0.5 -> 0
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_array_equal(combined, expected)
    
    def test_combine_predictions_max(self, ensemble_service):
        """Test max prediction combination."""
        predictions_list = [
            np.array([1, -1, 1, -1]),
            np.array([-1, 1, -1, -1])
        ]
        
        combined = ensemble_service._combine_predictions(
            predictions_list, method="max"
        )
        
        # Maximum of each position
        expected = np.array([1, 1, 1, -1])
        np.testing.assert_array_equal(combined, expected)
    
    def test_combine_predictions_weighted(self, ensemble_service):
        """Test weighted prediction combination."""
        predictions_list = [
            np.array([1, -1, 1, -1]),
            np.array([-1, 1, -1, 1])
        ]
        weights = [0.7, 0.3]
        
        combined = ensemble_service._combine_predictions(
            predictions_list, method="weighted", weights=weights
        )
        
        # Weighted combination: 0.7 * pred1 + 0.3 * pred2, then threshold at 0.5
        # Position 0: 0.7*1 + 0.3*(-1) = 0.4 < 0.5 -> 0
        # Position 1: 0.7*(-1) + 0.3*1 = -0.4 < 0.5 -> 0
        # Position 2: 0.7*1 + 0.3*(-1) = 0.4 < 0.5 -> 0  
        # Position 3: 0.7*(-1) + 0.3*1 = -0.4 < 0.5 -> 0
        expected = np.array([0, 0, 0, 0])
        np.testing.assert_array_equal(combined, expected)
    
    def test_combine_predictions_weighted_no_weights(self, ensemble_service):
        """Test weighted prediction combination with default weights."""
        predictions_list = [
            np.array([1, -1]),
            np.array([-1, 1])
        ]
        
        combined = ensemble_service._combine_predictions(
            predictions_list, method="weighted", weights=None
        )
        
        # Should use equal weights [1.0, 1.0] -> normalized to [0.5, 0.5]
        # Same as average method
        expected = np.array([0, 0])
        np.testing.assert_array_equal(combined, expected)
    
    def test_combine_predictions_weighted_wrong_weight_count(self, ensemble_service):
        """Test weighted prediction with wrong number of weights."""
        predictions_list = [
            np.array([1, -1]),
            np.array([-1, 1])
        ]
        weights = [0.7]  # Only one weight for two algorithms
        
        with pytest.raises(ValueError) as exc_info:
            ensemble_service._combine_predictions(
                predictions_list, method="weighted", weights=weights
            )
        
        assert "Number of weights must match number of algorithms" in str(exc_info.value)
    
    def test_combine_predictions_unknown_method(self, ensemble_service):
        """Test prediction combination with unknown method."""
        predictions_list = [np.array([1, -1]), np.array([-1, 1])]
        
        with pytest.raises(ValueError) as exc_info:
            ensemble_service._combine_predictions(
                predictions_list, method="unknown_method"
            )
        
        assert "Unknown combination method: unknown_method" in str(exc_info.value)
    
    def test_combine_scores_majority_average(self, ensemble_service):
        """Test score combination for majority/average methods."""
        scores_list = [
            np.array([0.8, 0.3, 0.9]),
            np.array([0.6, 0.7, 0.2])
        ]
        
        combined = ensemble_service._combine_scores(
            scores_list, method="majority"
        )
        
        # Should return average
        expected = np.array([0.7, 0.5, 0.55])
        np.testing.assert_array_almost_equal(combined, expected)
        
        # Test average method gives same result
        combined_avg = ensemble_service._combine_scores(
            scores_list, method="average"
        )
        np.testing.assert_array_almost_equal(combined_avg, expected)
    
    def test_combine_scores_max(self, ensemble_service):
        """Test max score combination."""
        scores_list = [
            np.array([0.8, 0.3, 0.9]),
            np.array([0.6, 0.7, 0.2])
        ]
        
        combined = ensemble_service._combine_scores(
            scores_list, method="max"
        )
        
        expected = np.array([0.8, 0.7, 0.9])
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_combine_scores_weighted(self, ensemble_service):
        """Test weighted score combination."""
        scores_list = [
            np.array([0.8, 0.3]),
            np.array([0.6, 0.7])
        ]
        weights = [0.3, 0.7]
        
        combined = ensemble_service._combine_scores(
            scores_list, method="weighted", weights=weights
        )
        
        # 0.3*0.8 + 0.7*0.6 = 0.24 + 0.42 = 0.66
        # 0.3*0.3 + 0.7*0.7 = 0.09 + 0.49 = 0.58
        expected = np.array([0.66, 0.58])
        np.testing.assert_array_almost_equal(combined, expected)
    
    def test_detect_with_ensemble_with_scores(self, ensemble_service, sample_data, mock_detection_results):
        """Test ensemble detection that combines scores."""
        # Ensure both results have scores
        for result in mock_detection_results:
            result.scores = np.random.rand(100).astype(np.float64)
        
        ensemble_service.detection_service.detect_anomalies.side_effect = mock_detection_results
        
        result = ensemble_service.detect_with_ensemble(
            data=sample_data,
            combination_method="average"
        )
        
        assert result.scores is not None
        assert len(result.scores) == len(sample_data)
    
    def test_detect_with_ensemble_without_scores(self, ensemble_service, sample_data, mock_detection_results):
        """Test ensemble detection when some results don't have scores."""
        # Make one result have no scores
        mock_detection_results[0].scores = None
        mock_detection_results[1].scores = np.random.rand(100).astype(np.float64)
        
        ensemble_service.detection_service.detect_anomalies.side_effect = mock_detection_results
        
        result = ensemble_service.detect_with_ensemble(data=sample_data)
        
        # Should not combine scores if not all results have them
        assert result.scores is None
    
    def test_detect_with_ensemble_with_kwargs(self, ensemble_service, sample_data, mock_detection_results):
        """Test ensemble detection with additional kwargs."""
        ensemble_service.detection_service.detect_anomalies.side_effect = mock_detection_results
        
        result = ensemble_service.detect_with_ensemble(
            data=sample_data,
            contamination=0.15,
            n_estimators=200
        )
        
        # Verify kwargs were passed to individual detection calls
        calls = ensemble_service.detection_service.detect_anomalies.call_args_list
        for call in calls:
            assert 'contamination' in call.kwargs
            assert call.kwargs['contamination'] == 0.15
            assert 'n_estimators' in call.kwargs
            assert call.kwargs['n_estimators'] == 200
    
    @patch('anomaly_detection.domain.services.ensemble_service.f1_score')
    def test_optimize_ensemble_with_ground_truth(self, mock_f1_score, ensemble_service, sample_data):
        """Test ensemble optimization with ground truth labels."""
        # Mock f1_score to return different values for different methods
        mock_f1_score.side_effect = [0.85, 0.78, 0.82, 0.79, 0.81]  # Different scores for different attempts
        
        # Mock ensemble detection results
        mock_result = Mock(spec=DetectionResult)
        mock_result.predictions = np.array([1, -1] * 50)
        mock_result.anomaly_count = 50
        
        with patch.object(ensemble_service, 'detect_with_ensemble', return_value=mock_result):
            ground_truth = np.array([1, -1] * 50)
            
            best_params = ensemble_service.optimize_ensemble(
                data=sample_data,
                ground_truth=ground_truth
            )
        
        assert "algorithms" in best_params
        assert "combination_method" in best_params
        assert "score" in best_params
        assert best_params["score"] > 0
    
    def test_optimize_ensemble_without_ground_truth(self, ensemble_service, sample_data):
        """Test ensemble optimization without ground truth labels."""
        # Mock ensemble detection results with different anomaly rates
        def mock_detect_with_ensemble(*args, **kwargs):
            method = kwargs.get('combination_method', 'majority')
            mock_result = Mock(spec=DetectionResult)
            
            # Return different anomaly rates for different methods
            if method == "majority":
                mock_result.predictions = np.array([1] * 90 + [-1] * 10)  # 10% anomalies (good)
                mock_result.anomaly_count = 10
            elif method == "average":
                mock_result.predictions = np.array([1] * 50 + [-1] * 50)  # 50% anomalies (bad)
                mock_result.anomaly_count = 50
            else:  # weighted
                mock_result.predictions = np.array([1] * 85 + [-1] * 15)  # 15% anomalies (ok)
                mock_result.anomaly_count = 15
            
            return mock_result
        
        with patch.object(ensemble_service, 'detect_with_ensemble', side_effect=mock_detect_with_ensemble):
            best_params = ensemble_service.optimize_ensemble(data=sample_data)
        
        assert best_params["combination_method"] == "majority"  # Should prefer 10% anomaly rate
        assert best_params["score"] > 0
    
    def test_optimize_ensemble_all_methods_fail(self, ensemble_service, sample_data):
        """Test ensemble optimization when all methods fail."""
        with patch.object(ensemble_service, 'detect_with_ensemble', side_effect=Exception("Failed")):
            best_params = ensemble_service.optimize_ensemble(data=sample_data)
        
        # Should return empty dict when all methods fail
        assert best_params == {}
    
    def test_evaluate_result_with_ground_truth(self, ensemble_service):
        """Test result evaluation with ground truth."""
        mock_result = Mock(spec=DetectionResult)
        mock_result.predictions = np.array([1, -1, 1, -1])
        ground_truth = np.array([1, -1, 1, 1])
        
        with patch('anomaly_detection.domain.services.ensemble_service.f1_score', return_value=0.75):
            score = ensemble_service._evaluate_result(mock_result, ground_truth)
        
        assert score == 0.75
    
    def test_evaluate_result_without_ground_truth_good_rate(self, ensemble_service):
        """Test result evaluation without ground truth - good anomaly rate."""
        mock_result = Mock(spec=DetectionResult)
        mock_result.predictions = np.array([1] * 90 + [-1] * 10)  # 10% anomalies
        mock_result.anomaly_count = 10
        
        score = ensemble_service._evaluate_result(mock_result, None)
        
        # Should get high score for 10% anomaly rate (optimal)
        assert score == 1.0
    
    def test_evaluate_result_without_ground_truth_bad_rate(self, ensemble_service):
        """Test result evaluation without ground truth - bad anomaly rate."""
        mock_result = Mock(spec=DetectionResult)
        mock_result.predictions = np.array([1] * 99 + [-1])  # 1% anomalies (too low)
        mock_result.anomaly_count = 1
        
        score = ensemble_service._evaluate_result(mock_result, None)
        
        # Should get penalty for extreme anomaly rate
        assert score == 0.1
        
        # Test with too high rate
        mock_result.predictions = np.array([-1] * 60 + [1] * 40)  # 60% anomalies (too high)
        mock_result.anomaly_count = 60
        
        score = ensemble_service._evaluate_result(mock_result, None)
        assert score == 0.1
    
    def test_evaluate_result_without_ground_truth_moderate_rate(self, ensemble_service):
        """Test result evaluation without ground truth - moderate anomaly rate."""
        mock_result = Mock(spec=DetectionResult)
        mock_result.predictions = np.array([1] * 85 + [-1] * 15)  # 15% anomalies
        mock_result.anomaly_count = 15
        
        score = ensemble_service._evaluate_result(mock_result, None)
        
        # Should get good but not perfect score for 15% (close to optimal 10%)
        assert 0.1 < score < 1.0
        assert score > 0.5  # Should be reasonably good
    
    @pytest.mark.parametrize("method,weights", [
        ("majority", None),
        ("average", None),
        ("max", None),
        ("weighted", [0.6, 0.4]),
        ("weighted", None),  # Default equal weights
    ])
    def test_detect_with_ensemble_all_methods(self, ensemble_service, sample_data, 
                                            mock_detection_results, method, weights):
        """Test ensemble detection with all combination methods."""
        ensemble_service.detection_service.detect_anomalies.side_effect = mock_detection_results
        
        result = ensemble_service.detect_with_ensemble(
            data=sample_data,
            combination_method=method,
            weights=weights
        )
        
        assert isinstance(result, DetectionResult)
        assert result.metadata["combination_method"] == method
        assert result.metadata["weights"] == weights
        assert len(result.predictions) == len(sample_data)
    
    def test_optimize_ensemble_custom_algorithms(self, ensemble_service, sample_data):
        """Test ensemble optimization with custom algorithm list."""
        algorithms = ["iforest", "lof"]
        
        mock_result = Mock(spec=DetectionResult)
        mock_result.predictions = np.array([1] * 90 + [-1] * 10)
        mock_result.anomaly_count = 10
        
        with patch.object(ensemble_service, 'detect_with_ensemble', return_value=mock_result):
            best_params = ensemble_service.optimize_ensemble(
                data=sample_data,
                algorithms=algorithms
            )
        
        # Should include the specified algorithms in optimization
        assert "algorithms" in best_params
    
    def test_optimization_validation_split(self, ensemble_service, sample_data):
        """Test ensemble optimization with custom validation split."""
        mock_result = Mock(spec=DetectionResult)
        mock_result.predictions = np.array([1] * 16 + [-1] * 4)  # For 20 validation samples
        mock_result.anomaly_count = 4
        
        with patch.object(ensemble_service, 'detect_with_ensemble', return_value=mock_result):
            best_params = ensemble_service.optimize_ensemble(
                data=sample_data,
                validation_split=0.2  # Use 20% for validation
            )
        
        assert "score" in best_params
        assert best_params["score"] > 0