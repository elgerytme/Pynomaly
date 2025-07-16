"""Comprehensive tests for DetectionResult domain entity."""

from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from monorepo.domain.entities.anomaly import Anomaly
from monorepo.domain.entities.detection_result import DetectionResult
from monorepo.domain.value_objects import AnomalyScore, ConfidenceInterval


class TestDetectionResultInitialization:
    """Test detection result initialization and validation."""

    def test_detection_result_initialization_basic(self):
        """Test basic detection result initialization."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)]
        labels = np.array([0, 1, 0])

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert len(result.anomalies) == 1
        assert len(result.scores) == 3
        assert len(result.labels) == 3
        assert result.threshold == 0.5
        assert isinstance(result.id, type(uuid4()))
        assert isinstance(result.timestamp, datetime)
        assert result.metadata == {}
        assert result.confidence_intervals is None
        assert result.execution_time_ms is None

    def test_detection_result_initialization_with_all_fields(self):
        """Test detection result initialization with all fields."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomalies = [
            Anomaly(
                score=AnomalyScore(0.9),
                data_point={"feature1": 100},
                detector_name="Test Detector",
            ),
            Anomaly(
                score=AnomalyScore(0.8),
                data_point={"feature1": 200},
                detector_name="Test Detector",
            ),
        ]

        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.9),
            AnomalyScore(0.3),
            AnomalyScore(0.8),
        ]
        labels = np.array([0, 1, 0, 1])

        confidence_intervals = [
            ConfidenceInterval(lower=0.05, upper=0.15),
            ConfidenceInterval(lower=0.85, upper=0.95),
            ConfidenceInterval(lower=0.25, upper=0.35),
            ConfidenceInterval(lower=0.75, upper=0.85),
        ]

        metadata = {"algorithm": "IsolationForest", "version": "1.0"}

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.7,
            execution_time_ms=150.5,
            metadata=metadata,
            confidence_intervals=confidence_intervals,
        )

        assert len(result.anomalies) == 2
        assert len(result.scores) == 4
        assert len(result.labels) == 4
        assert result.threshold == 0.7
        assert result.execution_time_ms == 150.5
        assert result.metadata == metadata
        assert len(result.confidence_intervals) == 4

    def test_detection_result_labels_conversion(self):
        """Test that labels are converted to numpy array."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = [0, 1]  # List, should be converted to numpy array

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert isinstance(result.labels, np.ndarray)
        assert result.labels.tolist() == [0, 1]

    def test_detection_result_validation_mismatched_scores_labels(self):
        """Test validation with mismatched scores and labels."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]  # 2 scores
        labels = [0, 1, 0]  # 3 labels

        with pytest.raises(
            ValueError, match="Number of scores .* doesn't match number of labels"
        ):
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[],
                scores=scores,
                labels=labels,
                threshold=0.5,
            )

    def test_detection_result_validation_mismatched_confidence_intervals(self):
        """Test validation with mismatched confidence intervals."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = [0, 1]
        confidence_intervals = [
            ConfidenceInterval(lower=0.05, upper=0.15)
        ]  # Only 1 CI for 2 samples

        with pytest.raises(
            ValueError,
            match="Number of confidence intervals .* doesn't match number of samples",
        ):
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[],
                scores=scores,
                labels=labels,
                threshold=0.5,
                confidence_intervals=confidence_intervals,
            )

    def test_detection_result_validation_invalid_labels(self):
        """Test validation with invalid labels."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = [0, 2]  # Invalid label (not 0 or 1)

        with pytest.raises(
            ValueError, match="Labels must be binary .* got unique values"
        ):
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=[],
                scores=scores,
                labels=labels,
                threshold=0.5,
            )

    def test_detection_result_validation_mismatched_anomalies_labels(self):
        """Test validation with mismatched anomalies and labels."""
        detector_id = uuid4()
        dataset_id = uuid4()

        # 2 anomalies but labels show only 1 anomaly
        anomalies = [
            Anomaly(
                score=AnomalyScore(0.9), data_point={"f1": 1}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.8), data_point={"f1": 2}, detector_name="Test"
            ),
        ]
        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = [0, 1]  # Only 1 anomaly in labels, but 2 in anomalies list

        # This should work - 1 anomaly in labels, need to adjust anomalies list
        with pytest.raises(
            ValueError,
            match="Number of anomalies .* doesn't match number of anomaly labels",
        ):
            DetectionResult(
                detector_id=detector_id,
                dataset_id=dataset_id,
                anomalies=anomalies,  # 2 anomalies
                scores=scores,
                labels=labels,  # 1 anomaly in labels
                threshold=0.5,
            )


class TestDetectionResultProperties:
    """Test detection result properties."""

    def test_n_samples_property(self):
        """Test n_samples property."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)]
        labels = [0, 1, 0]

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.n_samples == 3

    def test_n_anomalies_property(self):
        """Test n_anomalies property."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomalies = [
            Anomaly(
                score=AnomalyScore(0.9), data_point={"f1": 1}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.8), data_point={"f1": 2}, detector_name="Test"
            ),
        ]
        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.9),
            AnomalyScore(0.3),
            AnomalyScore(0.8),
        ]
        labels = [0, 1, 0, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.n_anomalies == 2

    def test_n_normal_property(self):
        """Test n_normal property."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)]
        labels = [0, 1, 0]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.n_normal == 2

    def test_anomaly_rate_property(self):
        """Test anomaly_rate property."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.9),
            AnomalyScore(0.3),
            AnomalyScore(0.2),
        ]
        labels = [0, 1, 0, 0]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.anomaly_rate == 0.25  # 1/4

    def test_anomaly_rate_empty_result(self):
        """Test anomaly_rate with empty result."""
        detector_id = uuid4()
        dataset_id = uuid4()

        # This should not be possible due to validation, but test the property logic
        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[],
            scores=[],
            labels=np.array([]),
            threshold=0.5,
        )

        assert result.anomaly_rate == 0.0

    def test_anomaly_indices_property(self):
        """Test anomaly_indices property."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomalies = [
            Anomaly(
                score=AnomalyScore(0.9), data_point={"f1": 1}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.8), data_point={"f1": 2}, detector_name="Test"
            ),
        ]
        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.9),
            AnomalyScore(0.3),
            AnomalyScore(0.8),
        ]
        labels = [0, 1, 0, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        anomaly_indices = result.anomaly_indices
        assert anomaly_indices.tolist() == [1, 3]

    def test_normal_indices_property(self):
        """Test normal_indices property."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)]
        labels = [0, 1, 0]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        normal_indices = result.normal_indices
        assert normal_indices.tolist() == [0, 2]

    def test_score_statistics_property(self):
        """Test score_statistics property."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.9),
            AnomalyScore(0.3),
            AnomalyScore(0.5),
        ]
        labels = [0, 1, 0, 0]

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        stats = result.score_statistics
        assert stats["min"] == 0.1
        assert stats["max"] == 0.9
        assert stats["mean"] == 0.45  # (0.1 + 0.9 + 0.3 + 0.5) / 4
        assert stats["median"] == 0.4  # (0.3 + 0.5) / 2
        assert "std" in stats
        assert "q25" in stats
        assert "q75" in stats

    def test_has_confidence_intervals_property(self):
        """Test has_confidence_intervals property."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = [0, 1]

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        # Without confidence intervals
        result_without_ci = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result_without_ci.has_confidence_intervals is False

        # With confidence intervals
        confidence_intervals = [
            ConfidenceInterval(lower=0.05, upper=0.15),
            ConfidenceInterval(lower=0.85, upper=0.95),
        ]

        result_with_ci = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
            confidence_intervals=confidence_intervals,
        )

        assert result_with_ci.has_confidence_intervals is True


class TestDetectionResultMethods:
    """Test detection result methods."""

    def test_get_top_anomalies(self):
        """Test get_top_anomalies method."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomalies = [
            Anomaly(
                score=AnomalyScore(0.7), data_point={"f1": 1}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.9), data_point={"f1": 2}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.8), data_point={"f1": 3}, detector_name="Test"
            ),
        ]
        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.7),
            AnomalyScore(0.9),
            AnomalyScore(0.8),
        ]
        labels = [0, 1, 1, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        top_anomalies = result.get_top_anomalies(n=2)
        assert len(top_anomalies) == 2
        assert top_anomalies[0].score.value == 0.9
        assert top_anomalies[1].score.value == 0.8

    def test_get_top_anomalies_all(self):
        """Test get_top_anomalies with n greater than available anomalies."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomalies = [
            Anomaly(
                score=AnomalyScore(0.7), data_point={"f1": 1}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.9), data_point={"f1": 2}, detector_name="Test"
            ),
        ]
        scores = [AnomalyScore(0.1), AnomalyScore(0.7), AnomalyScore(0.9)]
        labels = [0, 1, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        top_anomalies = result.get_top_anomalies(n=10)
        assert len(top_anomalies) == 2

    def test_get_scores_dataframe_without_confidence_intervals(self):
        """Test get_scores_dataframe without confidence intervals."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)]
        labels = [0, 1, 0]

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        df = result.get_scores_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["score", "label"]
        assert df["score"].tolist() == [0.1, 0.9, 0.3]
        assert df["label"].tolist() == [0, 1, 0]

    def test_get_scores_dataframe_with_confidence_intervals(self):
        """Test get_scores_dataframe with confidence intervals."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = [0, 1]

        confidence_intervals = [
            ConfidenceInterval(lower=0.05, upper=0.15),
            ConfidenceInterval(lower=0.85, upper=0.95),
        ]

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
            confidence_intervals=confidence_intervals,
        )

        df = result.get_scores_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        expected_columns = ["score", "label", "ci_lower", "ci_upper", "ci_width"]
        assert list(df.columns) == expected_columns
        assert df["ci_lower"].tolist() == [0.05, 0.85]
        assert df["ci_upper"].tolist() == [0.15, 0.95]
        assert df["ci_width"].tolist() == [0.1, 0.1]

    def test_filter_by_score(self):
        """Test filter_by_score method."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomalies = [
            Anomaly(
                score=AnomalyScore(0.6), data_point={"f1": 1}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.9), data_point={"f1": 2}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.7), data_point={"f1": 3}, detector_name="Test"
            ),
        ]
        scores = [
            AnomalyScore(0.1),
            AnomalyScore(0.6),
            AnomalyScore(0.9),
            AnomalyScore(0.7),
        ]
        labels = [0, 1, 1, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        filtered = result.filter_by_score(min_score=0.75)
        assert len(filtered) == 2
        assert filtered[0].score.value == 0.9
        assert filtered[1].score.value == 0.7

    def test_filter_by_confidence_without_intervals(self):
        """Test filter_by_confidence without confidence intervals."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = [0, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        filtered = result.filter_by_confidence(min_level=0.95)
        assert len(filtered) == 0

    def test_add_metadata(self):
        """Test add_metadata method."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.1), AnomalyScore(0.9)]
        labels = [0, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        result.add_metadata("algorithm", "IsolationForest")
        result.add_metadata("version", "1.0")

        assert result.metadata["algorithm"] == "IsolationForest"
        assert result.metadata["version"] == "1.0"

    def test_summary_method(self):
        """Test summary method."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)]
        labels = [0, 1, 0]
        metadata = {"algorithm": "IsolationForest"}

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
            execution_time_ms=150.5,
            metadata=metadata,
        )

        summary = result.summary()

        assert summary["detector_id"] == str(detector_id)
        assert summary["dataset_id"] == str(dataset_id)
        assert summary["n_samples"] == 3
        assert summary["n_anomalies"] == 1
        assert summary["anomaly_rate"] == 1 / 3
        assert summary["threshold"] == 0.5
        assert summary["execution_time_ms"] == 150.5
        assert summary["metadata"] == metadata
        assert summary["has_confidence_intervals"] is False
        assert "score_statistics" in summary
        assert "id" in summary
        assert "timestamp" in summary

    def test_summary_method_minimal(self):
        """Test summary method with minimal fields."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.3)]
        labels = [0, 0]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        summary = result.summary()

        assert "execution_time_ms" not in summary
        assert "metadata" not in summary
        assert summary["n_anomalies"] == 0
        assert summary["anomaly_rate"] == 0.0

    def test_repr_method(self):
        """Test string representation."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.1), AnomalyScore(0.9), AnomalyScore(0.3)]
        labels = [0, 1, 0]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        repr_str = repr(result)
        assert "DetectionResult(" in repr_str
        assert "n_samples=3" in repr_str
        assert "n_anomalies=1" in repr_str
        assert "anomaly_rate=33.33%" in repr_str


class TestDetectionResultEdgeCases:
    """Test detection result edge cases and error conditions."""

    def test_detection_result_no_anomalies(self):
        """Test detection result with no anomalies."""
        detector_id = uuid4()
        dataset_id = uuid4()

        scores = [AnomalyScore(0.1), AnomalyScore(0.3), AnomalyScore(0.2)]
        labels = [0, 0, 0]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.n_anomalies == 0
        assert result.anomaly_rate == 0.0
        assert len(result.get_top_anomalies()) == 0
        assert result.anomaly_indices.tolist() == []
        assert result.normal_indices.tolist() == [0, 1, 2]

    def test_detection_result_all_anomalies(self):
        """Test detection result with all samples as anomalies."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomalies = [
            Anomaly(
                score=AnomalyScore(0.9), data_point={"f1": 1}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.8), data_point={"f1": 2}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.7), data_point={"f1": 3}, detector_name="Test"
            ),
        ]
        scores = [AnomalyScore(0.9), AnomalyScore(0.8), AnomalyScore(0.7)]
        labels = [1, 1, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.n_anomalies == 3
        assert result.n_normal == 0
        assert result.anomaly_rate == 1.0
        assert result.anomaly_indices.tolist() == [0, 1, 2]
        assert result.normal_indices.tolist() == []

    def test_detection_result_single_sample(self):
        """Test detection result with single sample."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.9),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.9)]
        labels = [1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.n_samples == 1
        assert result.n_anomalies == 1
        assert result.anomaly_rate == 1.0
        assert result.score_statistics["min"] == 0.9
        assert result.score_statistics["max"] == 0.9
        assert result.score_statistics["mean"] == 0.9

    def test_detection_result_extreme_scores(self):
        """Test detection result with extreme scores."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(1.0),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        scores = [AnomalyScore(0.0), AnomalyScore(1.0), AnomalyScore(0.5)]
        labels = [0, 1, 0]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        stats = result.score_statistics
        assert stats["min"] == 0.0
        assert stats["max"] == 1.0
        assert stats["mean"] == 0.5

    def test_detection_result_large_dataset(self):
        """Test detection result with large dataset."""
        detector_id = uuid4()
        dataset_id = uuid4()

        # Create large dataset with 1000 samples
        np.random.seed(42)
        n_samples = 1000
        n_anomalies = 50

        # Generate scores and labels
        scores = [AnomalyScore(np.random.rand()) for _ in range(n_samples)]
        labels = [0] * (n_samples - n_anomalies) + [1] * n_anomalies
        np.random.shuffle(labels)

        # Create anomalies for positive labels
        anomalies = []
        for i, label in enumerate(labels):
            if label == 1:
                anomalies.append(
                    Anomaly(
                        score=scores[i],
                        data_point={"feature1": i},
                        detector_name="Test Detector",
                    )
                )

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        assert result.n_samples == n_samples
        assert result.n_anomalies == n_anomalies
        assert result.anomaly_rate == n_anomalies / n_samples
        assert len(result.get_top_anomalies(10)) == 10

    def test_detection_result_with_identical_scores(self):
        """Test detection result with identical scores."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomaly = Anomaly(
            score=AnomalyScore(0.5),
            data_point={"feature1": 100},
            detector_name="Test Detector",
        )

        # All scores are identical
        scores = [AnomalyScore(0.5), AnomalyScore(0.5), AnomalyScore(0.5)]
        labels = [0, 1, 0]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=[anomaly],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        stats = result.score_statistics
        assert stats["min"] == 0.5
        assert stats["max"] == 0.5
        assert stats["mean"] == 0.5
        assert stats["std"] == 0.0

    def test_detection_result_filter_edge_cases(self):
        """Test filtering methods with edge cases."""
        detector_id = uuid4()
        dataset_id = uuid4()

        anomalies = [
            Anomaly(
                score=AnomalyScore(0.5), data_point={"f1": 1}, detector_name="Test"
            ),
            Anomaly(
                score=AnomalyScore(0.6), data_point={"f1": 2}, detector_name="Test"
            ),
        ]
        scores = [AnomalyScore(0.1), AnomalyScore(0.5), AnomalyScore(0.6)]
        labels = [0, 1, 1]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        # Filter with score higher than all anomalies
        filtered = result.filter_by_score(min_score=0.9)
        assert len(filtered) == 0

        # Filter with score equal to lowest anomaly
        filtered = result.filter_by_score(min_score=0.5)
        assert len(filtered) == 2

        # Filter with score lower than all anomalies
        filtered = result.filter_by_score(min_score=0.1)
        assert len(filtered) == 2
