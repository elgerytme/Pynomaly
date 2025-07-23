"""Unit tests for DetectionResult entity."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List

from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.domain.entities.anomaly import Anomaly


class TestDetectionResultEntity:
    """Test suite for DetectionResult entity."""
    
    def test_detection_result_creation_basic(self):
        """Test basic detection result creation."""
        predictions = np.array([1, 1, -1, 1, -1])
        scores = np.array([0.1, 0.2, 0.8, 0.15, 0.9])
        
        result = DetectionResult(
            predictions=predictions,
            anomaly_scores=scores
        )
        
        assert np.array_equal(result.predictions, predictions)
        assert np.array_equal(result.anomaly_scores, scores)
        assert result.algorithm is None
        assert result.parameters == {}
        assert result.metadata == {}
    
    def test_detection_result_with_metadata(self):
        """Test detection result with full metadata."""
        predictions = np.array([1, -1, 1])
        scores = np.array([0.1, 0.9, 0.2])
        
        result = DetectionResult(
            predictions=predictions,
            anomaly_scores=scores,
            algorithm='isolation_forest',
            parameters={'n_estimators': 100, 'contamination': 0.1},
            metadata={'processing_time': 0.145, 'timestamp': '2024-01-23'}
        )
        
        assert result.algorithm == 'isolation_forest'
        assert result.parameters['n_estimators'] == 100
        assert result.metadata['processing_time'] == 0.145
    
    def test_anomaly_extraction(self):
        """Test automatic anomaly extraction from predictions."""
        predictions = np.array([1, 1, -1, 1, -1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.15, 0.9, 0.05])
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        assert len(result.anomalies) == 2  # Two anomalies (-1 predictions)
        assert result.anomalies[0].index == 2
        assert result.anomalies[0].score == 0.8
        assert result.anomalies[1].index == 4
        assert result.anomalies[1].score == 0.9
    
    def test_anomaly_statistics(self):
        """Test anomaly count and rate calculations."""
        predictions = np.array([1, 1, -1, 1, -1, 1, 1, -1, 1, 1])
        scores = np.random.random(10)
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        assert result.anomaly_count == 3
        assert result.anomaly_rate == 0.3
        assert len(result.anomaly_indices) == 3
        assert result.anomaly_indices == [2, 4, 7]
    
    def test_empty_detection_result(self):
        """Test handling of empty results."""
        predictions = np.array([])
        scores = np.array([])
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        assert len(result.anomalies) == 0
        assert result.anomaly_count == 0
        assert result.anomaly_rate == 0
        assert len(result.anomaly_indices) == 0
    
    def test_all_normal_result(self):
        """Test result with no anomalies."""
        predictions = np.ones(100)  # All normal
        scores = np.random.random(100) * 0.3  # Low scores
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        assert result.anomaly_count == 0
        assert result.anomaly_rate == 0.0
        assert len(result.anomalies) == 0
    
    def test_all_anomaly_result(self):
        """Test result with all anomalies."""
        predictions = -np.ones(50)  # All anomalies
        scores = np.random.random(50) * 0.5 + 0.5  # High scores
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        assert result.anomaly_count == 50
        assert result.anomaly_rate == 1.0
        assert len(result.anomalies) == 50
    
    def test_validation_predictions_scores_mismatch(self):
        """Test validation of mismatched predictions and scores."""
        predictions = np.array([1, -1, 1])
        scores = np.array([0.1, 0.9])  # Wrong length
        
        with pytest.raises(ValueError, match="Predictions and scores must have same length"):
            DetectionResult(predictions=predictions, anomaly_scores=scores)
    
    def test_validation_invalid_predictions(self):
        """Test validation of invalid prediction values."""
        # Valid predictions should be 1 or -1
        invalid_predictions = np.array([1, 0, -1, 2])
        scores = np.array([0.1, 0.5, 0.9, 0.2])
        
        with pytest.raises(ValueError, match="Predictions must be 1 .* or -1"):
            DetectionResult(predictions=invalid_predictions, anomaly_scores=scores)
    
    def test_validation_invalid_scores(self):
        """Test validation of invalid anomaly scores."""
        predictions = np.array([1, -1, 1])
        
        # Negative scores
        with pytest.raises(ValueError, match="Anomaly scores must be non-negative"):
            DetectionResult(predictions=predictions, anomaly_scores=np.array([-0.1, 0.5, 0.9]))
    
    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        predictions = np.array([1, -1, 1, -1, 1])
        scores = np.array([0.1, 0.8, 0.2, 0.9, 0.15])
        
        result = DetectionResult(
            predictions=predictions,
            anomaly_scores=scores,
            algorithm='lof'
        )
        
        df = result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'prediction' in df.columns
        assert 'anomaly_score' in df.columns
        assert 'is_anomaly' in df.columns
        
        # Check values
        assert df['prediction'].tolist() == [1, -1, 1, -1, 1]
        assert df['is_anomaly'].tolist() == [False, True, False, True, False]
        assert np.allclose(df['anomaly_score'].values, scores)
    
    def test_to_dataframe_with_features(self):
        """Test DataFrame conversion with feature data."""
        predictions = np.array([1, -1, 1])
        scores = np.array([0.1, 0.9, 0.2])
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        result = DetectionResult(
            predictions=predictions,
            anomaly_scores=scores,
            features=features,
            feature_names=['temp', 'pressure']
        )
        
        df = result.to_dataframe(include_features=True)
        
        assert 'temp' in df.columns
        assert 'pressure' in df.columns
        assert df['temp'].tolist() == [1.0, 3.0, 5.0]
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        predictions = np.array([1, -1, 1])
        scores = np.array([0.1, 0.9, 0.2])
        
        result = DetectionResult(
            predictions=predictions,
            anomaly_scores=scores,
            algorithm='iforest',
            parameters={'n_estimators': 100}
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert 'predictions' in data
        assert 'anomaly_scores' in data
        assert 'anomaly_count' in data
        assert 'anomaly_rate' in data
        assert 'algorithm' in data
        assert 'parameters' in data
        
        # Check values
        assert data['anomaly_count'] == 1
        assert data['anomaly_rate'] == 1/3
        assert data['algorithm'] == 'iforest'
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading results."""
        predictions = np.array([1, -1, 1, -1])
        scores = np.array([0.1, 0.8, 0.2, 0.9])
        
        result = DetectionResult(
            predictions=predictions,
            anomaly_scores=scores,
            algorithm='ocsvm',
            parameters={'nu': 0.1}
        )
        
        # Save to file
        filepath = tmp_path / "detection_result.pkl"
        result.save(str(filepath))
        
        assert filepath.exists()
        
        # Load from file
        loaded_result = DetectionResult.load(str(filepath))
        
        assert np.array_equal(loaded_result.predictions, predictions)
        assert np.array_equal(loaded_result.anomaly_scores, scores)
        assert loaded_result.algorithm == 'ocsvm'
        assert loaded_result.parameters['nu'] == 0.1
    
    def test_get_top_anomalies(self):
        """Test getting top anomalies by score."""
        predictions = np.array([1, -1, 1, -1, -1, 1, -1])
        scores = np.array([0.1, 0.6, 0.2, 0.9, 0.7, 0.3, 0.8])
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        # Get top 3 anomalies
        top_anomalies = result.get_top_anomalies(n=3)
        
        assert len(top_anomalies) == 3
        assert top_anomalies[0].index == 3  # Highest score 0.9
        assert top_anomalies[1].index == 6  # Score 0.8
        assert top_anomalies[2].index == 4  # Score 0.7
    
    def test_get_anomalies_above_threshold(self):
        """Test filtering anomalies by score threshold."""
        predictions = np.array([1, -1, 1, -1, -1, 1, -1])
        scores = np.array([0.1, 0.6, 0.2, 0.9, 0.7, 0.3, 0.4])
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        # Get anomalies with score >= 0.7
        high_score_anomalies = result.get_anomalies_above_threshold(0.7)
        
        assert len(high_score_anomalies) == 2
        assert all(a.score >= 0.7 for a in high_score_anomalies)
    
    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        predictions = np.array([1, -1, 1, -1, -1, 1, -1, 1, 1, 1])
        scores = np.array([0.1, 0.6, 0.2, 0.9, 0.7, 0.3, 0.8, 0.15, 0.25, 0.05])
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        stats = result.get_summary_statistics()
        
        assert 'total_samples' in stats
        assert 'anomaly_count' in stats
        assert 'anomaly_rate' in stats
        assert 'mean_score' in stats
        assert 'std_score' in stats
        assert 'min_score' in stats
        assert 'max_score' in stats
        assert 'mean_anomaly_score' in stats
        assert 'mean_normal_score' in stats
        
        assert stats['total_samples'] == 10
        assert stats['anomaly_count'] == 4
        assert stats['anomaly_rate'] == 0.4
        assert np.isclose(stats['mean_score'], np.mean(scores))
    
    def test_merge_results(self):
        """Test merging multiple detection results."""
        # First result
        result1 = DetectionResult(
            predictions=np.array([1, -1, 1]),
            anomaly_scores=np.array([0.1, 0.9, 0.2]),
            algorithm='iforest'
        )
        
        # Second result
        result2 = DetectionResult(
            predictions=np.array([-1, 1, -1]),
            anomaly_scores=np.array([0.8, 0.3, 0.7]),
            algorithm='iforest'
        )
        
        # Merge
        merged = DetectionResult.merge([result1, result2])
        
        assert len(merged.predictions) == 6
        assert merged.anomaly_count == 3
        assert np.array_equal(merged.predictions, [1, -1, 1, -1, 1, -1])
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation for anomaly scores."""
        predictions = np.ones(1000)
        predictions[900:] = -1  # Last 100 are anomalies
        scores = np.random.random(1000)
        scores[900:] = scores[900:] * 0.5 + 0.5  # Higher scores for anomalies
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        # Calculate confidence intervals
        ci_normal = result.get_score_confidence_interval(anomaly=False)
        ci_anomaly = result.get_score_confidence_interval(anomaly=True)
        
        assert ci_normal[0] < ci_normal[1]  # Lower < Upper
        assert ci_anomaly[0] < ci_anomaly[1]
        assert ci_anomaly[0] > ci_normal[1]  # Anomaly scores should be higher
    
    def test_result_filtering(self):
        """Test filtering results by indices."""
        predictions = np.array([1, -1, 1, -1, 1, -1, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4])
        
        result = DetectionResult(predictions=predictions, anomaly_scores=scores)
        
        # Filter to specific indices
        indices = [1, 3, 5]
        filtered = result.filter_by_indices(indices)
        
        assert len(filtered.predictions) == 3
        assert np.array_equal(filtered.predictions, [-1, -1, -1])
        assert np.array_equal(filtered.anomaly_scores, [0.9, 0.8, 0.7])
    
    def test_result_comparison(self):
        """Test comparing two detection results."""
        predictions1 = np.array([1, -1, 1, -1])
        predictions2 = np.array([1, 1, -1, -1])
        scores = np.array([0.1, 0.9, 0.8, 0.7])
        
        result1 = DetectionResult(predictions=predictions1, anomaly_scores=scores)
        result2 = DetectionResult(predictions=predictions2, anomaly_scores=scores)
        
        comparison = result1.compare_with(result2)
        
        assert 'agreement_rate' in comparison
        assert 'true_positives' in comparison
        assert 'false_positives' in comparison
        assert 'false_negatives' in comparison
        
        # Agreement on indices 0 and 3
        assert comparison['agreement_rate'] == 0.5
    
    def test_export_formats(self):
        """Test exporting results in different formats."""
        predictions = np.array([1, -1, 1])
        scores = np.array([0.1, 0.9, 0.2])
        
        result = DetectionResult(
            predictions=predictions,
            anomaly_scores=scores,
            algorithm='lof'
        )
        
        # Export as JSON-serializable dict
        json_data = result.to_json_dict()
        assert isinstance(json_data['predictions'], list)
        assert isinstance(json_data['anomaly_scores'], list)
        
        # Export anomalies only
        anomalies_data = result.export_anomalies()
        assert len(anomalies_data) == 1
        assert anomalies_data[0]['index'] == 1