"""Property-based tests for domain entities using Hypothesis."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import ValidationError
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate


# Custom strategies for domain objects
@st.composite
def valid_dataframes(draw):
    """Generate valid pandas DataFrames for testing."""
    n_rows = draw(st.integers(min_value=1, max_value=1000))
    n_cols = draw(st.integers(min_value=1, max_value=50))

    # Generate column names
    col_names = [f"feature_{i}" for i in range(n_cols)]

    # Generate data
    data = {}
    for col in col_names:
        if draw(st.booleans()):  # Numeric column
            data[col] = draw(
                st.lists(
                    st.floats(
                        min_value=-1000,
                        max_value=1000,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=n_rows,
                    max_size=n_rows,
                )
            )
        else:  # Categorical column
            categories = draw(
                st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10)
            )
            data[col] = draw(
                st.lists(st.sampled_from(categories), min_size=n_rows, max_size=n_rows)
            )

    return pd.DataFrame(data)


@st.composite
def valid_contamination_rates(draw):
    """Generate valid contamination rates."""
    return ContaminationRate(draw(st.floats(min_value=0.001, max_value=0.499)))


@st.composite
def valid_anomaly_scores(draw):
    """Generate valid anomaly scores."""
    score = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    method = draw(st.text(min_size=1, max_size=50))

    # Optional confidence interval
    if draw(st.booleans()):
        lower = draw(
            st.floats(
                min_value=0.0, max_value=score, allow_nan=False, allow_infinity=False
            )
        )
        upper = draw(
            st.floats(
                min_value=score, max_value=1.0, allow_nan=False, allow_infinity=False
            )
        )
        confidence_interval = (lower, upper)
    else:
        confidence_interval = None

    return AnomalyScore(
        value=score, method=method, confidence_interval=confidence_interval
    )


@st.composite
def valid_datasets(draw):
    """Generate valid Dataset objects."""
    df = draw(valid_dataframes())
    name = draw(st.text(min_size=1, max_size=100))

    # Optionally add a target column
    if draw(st.booleans()) and len(df.columns) > 1:
        target_column = draw(st.sampled_from(df.columns.tolist()))
    else:
        target_column = None

    return Dataset(name=name, data=df, target_column=target_column)


@st.composite
def valid_detectors(draw):
    """Generate valid Detector objects."""
    name = draw(st.text(min_size=1, max_size=100))
    algorithm_name = draw(st.text(min_size=1, max_size=50))
    contamination_rate = draw(valid_contamination_rates())

    # Optional parameters and metadata
    parameters = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.integers(), st.floats(), st.text(), st.booleans()),
            max_size=10,
        )
    )

    metadata = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=20), st.text(max_size=100), max_size=5
        )
    )

    return Detector(
        name=name,
        algorithm_name=algorithm_name,
        contamination_rate=contamination_rate,
        parameters=parameters,
        metadata=metadata,
    )


class TestDatasetProperties:
    """Property-based tests for Dataset entity."""

    @given(valid_datasets())
    def test_dataset_name_preserved(self, dataset):
        """Dataset name should be preserved after creation."""
        assert dataset.name is not None
        assert len(dataset.name) > 0

    @given(valid_datasets())
    def test_dataset_data_not_empty(self, dataset):
        """Dataset data should never be empty."""
        assert not dataset.data.empty
        assert dataset.n_samples > 0
        assert dataset.n_features > 0

    @given(valid_datasets())
    def test_dataset_sample_count_consistent(self, dataset):
        """Dataset sample count should be consistent with data."""
        assert dataset.n_samples == len(dataset.data)

    @given(valid_datasets())
    def test_dataset_feature_count_consistent(self, dataset):
        """Dataset feature count should be consistent with data."""
        assert dataset.n_features == len(dataset.data.columns)

    @given(valid_datasets())
    def test_dataset_numeric_features_subset(self, dataset):
        """Numeric features should be a subset of all features."""
        numeric_features = set(dataset.get_numeric_features())
        all_features = set(dataset.data.columns)
        assert numeric_features.issubset(all_features)

    @given(valid_datasets())
    def test_dataset_target_column_validation(self, dataset):
        """Target column should be valid if specified."""
        if dataset.target_column is not None:
            assert dataset.target_column in dataset.data.columns

    @given(st.text(min_size=1), valid_dataframes())
    def test_dataset_with_invalid_target_raises_error(self, name, df):
        """Dataset with invalid target column should raise ValidationError."""
        invalid_target = "nonexistent_column_12345"
        assume(invalid_target not in df.columns)

        with pytest.raises(ValidationError):
            Dataset(name=name, data=df, target_column=invalid_target)

    @given(st.text(min_size=1))
    def test_dataset_with_empty_data_raises_error(self, name):
        """Dataset with empty data should raise ValidationError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValidationError):
            Dataset(name=name, data=empty_df)


class TestContaminationRateProperties:
    """Property-based tests for ContaminationRate value object."""

    @given(st.floats(min_value=0.001, max_value=0.499))
    def test_contamination_rate_value_preserved(self, rate):
        """Contamination rate value should be preserved."""
        contamination = ContaminationRate(rate)
        assert contamination.value == rate
        assert not contamination.is_auto

    @given(st.floats(min_value=-1000, max_value=0.0))
    def test_contamination_rate_too_low_raises_error(self, rate):
        """Contamination rate below 0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ContaminationRate(rate)

    @given(st.floats(min_value=0.5, max_value=1000))
    def test_contamination_rate_too_high_raises_error(self, rate):
        """Contamination rate above 0.5 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ContaminationRate(rate)

    def test_auto_contamination_rate_properties(self):
        """Auto contamination rate should have expected properties."""
        auto_rate = ContaminationRate.auto()
        assert auto_rate.is_auto
        assert 0 < auto_rate.value <= 0.5


class TestAnomalyScoreProperties:
    """Property-based tests for AnomalyScore value object."""

    @given(valid_anomaly_scores())
    def test_anomaly_score_value_in_range(self, score):
        """Anomaly score value should always be in valid range."""
        assert 0.0 <= score.value <= 1.0

    @given(valid_anomaly_scores())
    def test_anomaly_score_method_preserved(self, score):
        """Anomaly score method should be preserved."""
        assert score.method is not None
        assert len(score.method) > 0

    @given(st.floats(min_value=-1000, max_value=-0.001), st.text(min_size=1))
    def test_anomaly_score_negative_raises_error(self, value, method):
        """Negative anomaly score should raise ValidationError."""
        with pytest.raises(ValidationError):
            AnomalyScore(value=value, method=method)

    @given(st.floats(min_value=1.001, max_value=1000), st.text(min_size=1))
    def test_anomaly_score_above_one_raises_error(self, value, method):
        """Anomaly score above 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            AnomalyScore(value=value, method=method)

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.text(min_size=1),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    def test_anomaly_score_confidence_interval_validation(
        self, value, method, ci_lower, ci_upper
    ):
        """Confidence interval should be validated correctly."""
        # Ensure confidence interval is valid
        if ci_lower <= ci_upper:
            score = AnomalyScore(
                value=value, method=method, confidence_interval=(ci_lower, ci_upper)
            )
            assert score.confidence_interval == (ci_lower, ci_upper)
        else:
            # Invalid confidence interval should raise error
            with pytest.raises(ValidationError):
                AnomalyScore(
                    value=value, method=method, confidence_interval=(ci_lower, ci_upper)
                )


class TestDetectorProperties:
    """Property-based tests for Detector entity."""

    @given(valid_detectors())
    def test_detector_initialization_properties(self, detector):
        """Detector should have valid properties after initialization."""
        assert detector.name is not None
        assert len(detector.name) > 0
        assert detector.algorithm_name is not None
        assert len(detector.algorithm_name) > 0
        assert isinstance(detector.contamination_rate, ContaminationRate)
        assert not detector.is_fitted
        assert detector.trained_at is None

    @given(valid_detectors())
    def test_detector_parameter_updates_preserve_existing(self, detector):
        """Parameter updates should preserve existing parameters."""
        original_params = detector.parameters.copy()
        new_params = {"new_param": "new_value"}

        detector.update_parameters(**new_params)

        # Check that original parameters are still there
        for key, value in original_params.items():
            assert detector.parameters[key] == value

        # Check that new parameter was added
        assert detector.parameters["new_param"] == "new_value"

    @given(valid_detectors(), st.text(min_size=1), st.text())
    def test_detector_metadata_updates(self, detector, key, value):
        """Metadata updates should work correctly."""
        original_metadata = detector.metadata.copy()

        detector.update_metadata(key, value)

        # Check that original metadata is preserved
        for orig_key, orig_value in original_metadata.items():
            if orig_key != key:  # Skip if we're updating an existing key
                assert detector.metadata[orig_key] == orig_value

        # Check that new/updated metadata is set
        assert detector.metadata[key] == value

    @given(valid_detectors())
    def test_detector_fitting_state_transition(self, detector):
        """Detector fitting state should transition correctly."""
        # Initially not fitted
        assert not detector.is_fitted
        assert detector.trained_at is None

        # Simulate fitting
        detector.is_fitted = True
        detector.trained_at = datetime.now()

        assert detector.is_fitted
        assert detector.trained_at is not None


class TestAnomalyProperties:
    """Property-based tests for Anomaly entity."""

    @given(
        valid_anomaly_scores(),
        st.dictionaries(st.text(min_size=1), st.floats()),
        st.text(min_size=1),
    )
    def test_anomaly_initialization(self, score, data_point, detector_name):
        """Anomaly should initialize with valid properties."""
        anomaly = Anomaly(
            score=score, data_point=data_point, detector_name=detector_name
        )

        assert anomaly.score == score
        assert anomaly.data_point == data_point
        assert anomaly.detector_name == detector_name
        assert isinstance(anomaly.detected_at, datetime)


class TestDetectionResultProperties:
    """Property-based tests for DetectionResult entity."""

    @given(
        st.text(min_size=1),  # detector_id
        st.text(min_size=1),  # dataset_id
        st.lists(
            st.builds(
                Anomaly,
                score=valid_anomaly_scores(),
                data_point=st.dictionaries(
                    st.text(min_size=1), st.floats(), min_size=1
                ),
                detector_name=st.text(min_size=1),
            ),
            max_size=10,
        ),  # anomalies
        st.lists(valid_anomaly_scores(), min_size=1, max_size=100),  # scores
        st.lists(
            st.integers(min_value=0, max_value=1), min_size=1, max_size=100
        ),  # labels
        st.floats(min_value=0.0, max_value=1.0),  # threshold
    )
    def test_detection_result_consistency(
        self, detector_id, dataset_id, anomalies, scores, labels, threshold
    ):
        """Detection result should maintain consistency between scores and labels."""
        # Ensure scores and labels have same length
        min_length = min(len(scores), len(labels))
        scores = scores[:min_length]
        labels = labels[:min_length]

        result = DetectionResult(
            detector_id=detector_id,
            dataset_id=dataset_id,
            anomalies=anomalies,
            scores=scores,
            labels=labels,
            threshold=threshold,
        )

        assert result.detector_id == detector_id
        assert result.dataset_id == dataset_id
        assert result.threshold == threshold
        assert result.n_samples == len(scores)
        assert result.n_anomalies == len(anomalies)
        assert len(result.scores) == len(result.labels)


class TestPropertyInvariants:
    """Test system-wide property invariants."""

    @given(valid_datasets())
    @settings(max_examples=50)
    def test_dataset_operations_preserve_invariants(self, dataset):
        """Dataset operations should preserve core invariants."""
        # Core invariants
        assert dataset.n_samples == len(dataset.data)
        assert dataset.n_features == len(dataset.data.columns)
        assert not dataset.data.empty

        # Get numeric features multiple times - should be consistent
        numeric_features_1 = dataset.get_numeric_features()
        numeric_features_2 = dataset.get_numeric_features()
        assert numeric_features_1 == numeric_features_2

        # All numeric features should actually be numeric
        for feature in numeric_features_1:
            assert feature in dataset.data.columns
            # Check if the column is actually numeric
            col_data = dataset.data[feature]
            assert pd.api.types.is_numeric_dtype(col_data) or col_data.dtype == bool

    @given(valid_detectors())
    @settings(max_examples=50)
    def test_detector_state_consistency(self, detector):
        """Detector state should remain consistent across operations."""
        # Initial state invariants
        assert not detector.is_fitted
        assert detector.trained_at is None

        # Parameter operations should not affect core properties
        original_name = detector.name
        original_algorithm = detector.algorithm_name

        detector.update_parameters(test_param="test_value")

        assert detector.name == original_name
        assert detector.algorithm_name == original_algorithm
        assert "test_param" in detector.parameters

        # Metadata operations should not affect core properties
        detector.update_metadata("test_key", "test_value")

        assert detector.name == original_name
        assert detector.algorithm_name == original_algorithm
        assert detector.metadata["test_key"] == "test_value"

    @given(
        st.lists(valid_anomaly_scores(), min_size=1, max_size=100),
        st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=100),
    )
    def test_detection_result_metrics_consistency(self, scores, labels):
        """Detection result metrics should be mathematically consistent."""
        # Ensure same length
        min_length = min(len(scores), len(labels))
        scores = scores[:min_length]
        labels = labels[:min_length]

        result = DetectionResult(
            detector_id="test_detector",
            dataset_id="test_dataset",
            anomalies=[],
            scores=scores,
            labels=labels,
            threshold=0.5,
        )

        # Basic consistency checks
        assert result.n_samples == len(scores)
        assert result.n_samples == len(labels)
        assert result.n_samples > 0

        # If we have anomalies, n_anomalies should be >= 0
        assert result.n_anomalies >= 0
