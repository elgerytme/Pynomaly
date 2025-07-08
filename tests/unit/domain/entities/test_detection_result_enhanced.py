"""Tests for enhanced DetectionResult entity validation."""

import pytest
import numpy as np
from datetime import datetime, timezone
from uuid import uuid4

from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval


class TestDetectionResultBasicValidation:
    """Test basic validation rules for DetectionResult."""

    def test_valid_detection_result_creation(self):
        """Test creating a valid detection result."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create anomaly
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="test_detector"
        )
        
        # Create scores and labels
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert len(result.anomalies) == 1
        assert len(result.scores) == 2
        assert result.threshold == 0.5

    def test_invalid_mismatched_scores_labels(self):
        """Test validation of mismatched scores and labels."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1, 0])  # Mismatched length
        
        with pytest.raises(ValueError) as exc_info:
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[],
                scores=scores,
                labels=labels,
                threshold=0.5
            )
        
        assert "Number of scores" in str(exc_info.value)
        assert "doesn't match" in str(exc_info.value)

    def test_invalid_mismatched_anomalies_labels(self):
        """Test validation of mismatched anomalies and labels."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create two anomalies but only one anomaly label
        anomaly1 = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        anomaly2 = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 2.0},
            detector_name="test"
        )
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])  # Only 1 anomaly label but 2 anomalies
        
        with pytest.raises(ValueError) as exc_info:
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[anomaly1, anomaly2],
                scores=scores,
                labels=labels,
                threshold=0.5
            )
        
        assert "Number of anomalies" in str(exc_info.value)
        assert "doesn't match" in str(exc_info.value)

    def test_invalid_non_binary_labels(self):
        """Test validation of non-binary labels."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 2])  # Invalid label value
        
        with pytest.raises(ValueError) as exc_info:
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[],
                scores=scores,
                labels=labels,
                threshold=0.5
            )
        
        assert "Labels must be binary" in str(exc_info.value)

    def test_invalid_mismatched_confidence_intervals(self):
        """Test validation of mismatched confidence intervals."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])
        confidence_intervals = [ConfidenceInterval(0.1, 0.3, 0.95)]  # Only one interval
        
        with pytest.raises(ValueError) as exc_info:
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[],
                scores=scores,
                labels=labels,
                threshold=0.5,
                confidence_intervals=confidence_intervals
            )
        
        assert "Number of confidence intervals" in str(exc_info.value)
        assert "doesn't match" in str(exc_info.value)


class TestDetectionResultProperties:
    """Test properties of DetectionResult."""

    def test_basic_properties(self):
        """Test basic properties calculation."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create anomaly
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8), AnomalyScore(0.1)]
        labels = np.array([0, 1, 0])
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        assert result.n_samples == 3
        assert result.n_anomalies == 1
        assert result.n_normal == 2
        assert result.anomaly_rate == pytest.approx(1/3)

    def test_anomaly_indices(self):
        """Test anomaly indices calculation."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create anomalies
        anomaly1 = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        anomaly2 = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 2.0},
            detector_name="test"
        )
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8), AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = np.array([0, 1, 0, 1])
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly1, anomaly2],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        anomaly_indices = result.anomaly_indices
        normal_indices = result.normal_indices
        
        assert list(anomaly_indices) == [1, 3]
        assert list(normal_indices) == [0, 2]

    def test_score_statistics(self):
        """Test score statistics calculation."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.1), AnomalyScore(0.5), AnomalyScore(0.9)]
        labels = np.array([0, 0, 1])
        
        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        stats = result.score_statistics
        
        assert stats["min"] == 0.1
        assert stats["max"] == 0.9
        assert stats["mean"] == pytest.approx(0.5)
        assert stats["median"] == pytest.approx(0.5)
        assert stats["q25"] == pytest.approx(0.3)
        assert stats["q75"] == pytest.approx(0.7)

    def test_confidence_interval_properties(self):
        """Test confidence interval properties."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])
        confidence_intervals = [
            ConfidenceInterval(0.1, 0.3, 0.95),
            ConfidenceInterval(0.7, 0.9, 0.95)
        ]
        
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
            confidence_intervals=confidence_intervals
        )
        
        assert result.has_confidence_intervals
        
        # Test without confidence intervals
        result_no_ci = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        assert not result_no_ci.has_confidence_intervals


class TestDetectionResultMethods:
    """Test methods of DetectionResult."""

    def test_get_top_anomalies(self):
        """Test getting top anomalies by score."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create anomalies with different scores
        anomaly1 = Anomaly(
            score=AnomalyScore(0.6),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        anomaly2 = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 2.0},
            detector_name="test"
        )
        anomaly3 = Anomaly(
            score=AnomalyScore(0.7),
            data_point={"feature1": 3.0},
            detector_name="test"
        )
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.6), AnomalyScore(0.9), AnomalyScore(0.7)]
        labels = np.array([0, 1, 1, 1])
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly1, anomaly2, anomaly3],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        top_anomalies = result.get_top_anomalies(2)
        
        assert len(top_anomalies) == 2
        assert top_anomalies[0].score.value == 0.9
        assert top_anomalies[1].score.value == 0.7

    def test_get_scores_dataframe(self):
        """Test getting scores as DataFrame."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])
        
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        df = result.get_scores_dataframe()
        
        assert "score" in df.columns
        assert "label" in df.columns
        assert len(df) == 2
        assert df["score"].tolist() == [0.2, 0.8]
        assert df["label"].tolist() == [0, 1]

    def test_get_scores_dataframe_with_confidence_intervals(self):
        """Test getting scores DataFrame with confidence intervals."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])
        confidence_intervals = [
            ConfidenceInterval(0.1, 0.3, 0.95),
            ConfidenceInterval(0.7, 0.9, 0.95)
        ]
        
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
            confidence_intervals=confidence_intervals
        )
        
        df = result.get_scores_dataframe()
        
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns
        assert "ci_width" in df.columns
        assert df["ci_lower"].tolist() == [0.1, 0.7]
        assert df["ci_upper"].tolist() == [0.3, 0.9]
        assert df["ci_width"].tolist() == [0.2, 0.2]

    def test_filter_by_score(self):
        """Test filtering anomalies by score."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create anomalies with different scores
        anomaly1 = Anomaly(
            score=AnomalyScore(0.6),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        anomaly2 = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 2.0},
            detector_name="test"
        )
        anomaly3 = Anomaly(
            score=AnomalyScore(0.7),
            data_point={"feature1": 3.0},
            detector_name="test"
        )
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.6), AnomalyScore(0.9), AnomalyScore(0.7)]
        labels = np.array([0, 1, 1, 1])
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly1, anomaly2, anomaly3],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        filtered = result.filter_by_score(0.7)
        
        assert len(filtered) == 2
        assert all(a.score.value >= 0.7 for a in filtered)

    def test_filter_by_confidence(self):
        """Test filtering anomalies by confidence."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create anomalies with different confidence levels
        ci_high = ConfidenceInterval(0.85, 0.95, 0.99)  # High confidence
        ci_low = ConfidenceInterval(0.5, 0.9, 0.80)    # Low confidence
        
        anomaly1 = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 1.0},
            detector_name="test",
            confidence_interval=ci_high
        )
        anomaly2 = Anomaly(
            score=AnomalyScore(0.7),
            data_point={"feature1": 2.0},
            detector_name="test",
            confidence_interval=ci_low
        )
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.9), AnomalyScore(0.7)]
        labels = np.array([0, 1, 1])
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly1, anomaly2],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        high_confidence = result.filter_by_confidence(0.95)
        
        assert len(high_confidence) == 1
        assert high_confidence[0].confidence_interval.level == 0.99

    def test_add_metadata(self):
        """Test adding metadata to result."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])
        
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        result.add_metadata("algorithm", "isolation_forest")
        result.add_metadata("version", "1.0.0")
        
        assert result.metadata["algorithm"] == "isolation_forest"
        assert result.metadata["version"] == "1.0.0"

    def test_summary(self):
        """Test result summary generation."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])
        
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
            execution_time_ms=1500.0
        )
        
        result.add_metadata("algorithm", "isolation_forest")
        
        summary = result.summary()
        
        assert "id" in summary
        assert "detector_id" in summary
        assert "dataset_id" in summary
        assert "n_samples" in summary
        assert "n_anomalies" in summary
        assert "anomaly_rate" in summary
        assert "threshold" in summary
        assert "score_statistics" in summary
        assert "execution_time_ms" in summary
        assert "metadata" in summary
        
        assert summary["n_samples"] == 2
        assert summary["n_anomalies"] == 1
        assert summary["anomaly_rate"] == 0.5
        assert summary["threshold"] == 0.5
        assert summary["execution_time_ms"] == 1500.0
        assert summary["metadata"]["algorithm"] == "isolation_forest"


class TestDetectionResultValidation:
    """Test validation aspects of DetectionResult."""

    def test_labels_array_conversion(self):
        """Test automatic conversion of labels to numpy array."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = [0, 1]  # List instead of numpy array
        
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        assert isinstance(result.labels, np.ndarray)
        assert result.labels.tolist() == [0, 1]

    def test_empty_result(self):
        """Test creation of empty detection result."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = []
        labels = np.array([])
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        assert result.n_samples == 0
        assert result.n_anomalies == 0
        assert result.anomaly_rate == 0.0
        assert len(result.get_top_anomalies()) == 0


class TestDetectionResultStringRepresentation:
    """Test string representation of DetectionResult."""

    def test_repr(self):
        """Test __repr__ method."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        scores = [AnomalyScore(0.2), AnomalyScore(0.8)]
        labels = np.array([0, 1])
        
        anomaly = Anomaly(
            score=AnomalyScore(0.8),
            data_point={"feature1": 1.0},
            detector_name="test"
        )
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5
        )
        
        repr_str = repr(result)
        
        assert "DetectionResult" in repr_str
        assert "n_samples=2" in repr_str
        assert "n_anomalies=1" in repr_str
        assert "anomaly_rate=50.00%" in repr_str


class TestDetectionResultIntegration:
    """Integration tests for DetectionResult with realistic scenarios."""

    def test_realistic_detection_scenario(self):
        """Test realistic anomaly detection scenario."""
        detector_id = uuid4()
        dataset_id = uuid4()
        
        # Create multiple anomalies with different scores
        anomalies = []
        for i, score_val in enumerate([0.95, 0.85, 0.75]):
            anomaly = Anomaly(
                score=AnomalyScore(score_val),
                data_point={"feature1": i, "feature2": i * 2},
                detector_name="isolation_forest"
            )
            anomalies.append(anomaly)
        
        # Create scores for all samples (normal + anomalous)
        all_scores = [
            AnomalyScore(0.1), AnomalyScore(0.2), AnomalyScore(0.3),  # Normal
            AnomalyScore(0.95), AnomalyScore(0.85), AnomalyScore(0.75)  # Anomalous
        ]
        
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=all_scores,
            labels=labels,
            threshold=0.5,
            execution_time_ms=2500.0
        )
        
        # Test comprehensive analysis
        assert result.n_samples == 6
        assert result.n_anomalies == 3
        assert result.anomaly_rate == 0.5
        
        # Test top anomalies
        top_anomalies = result.get_top_anomalies(2)
        assert len(top_anomalies) == 2
        assert top_anomalies[0].score.value == 0.95
        assert top_anomalies[1].score.value == 0.85
        
        # Test score filtering
        high_score_anomalies = result.filter_by_score(0.8)
        assert len(high_score_anomalies) == 2
        
        # Test statistics
        stats = result.score_statistics
        assert stats["max"] == 0.95
        assert stats["min"] == 0.1
        
        # Test summary
        summary = result.summary()
        assert summary["n_samples"] == 6
        assert summary["n_anomalies"] == 3
        assert summary["execution_time_ms"] == 2500.0
