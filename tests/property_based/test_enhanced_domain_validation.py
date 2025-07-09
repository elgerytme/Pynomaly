"""Comprehensive property-based tests for enhanced domain validation using Hypothesis."""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import pandas as st_pandas
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate


# Custom strategies for enhanced testing
@st.composite
def valid_anomaly_scores(draw):
    """Generate valid anomaly scores with various configurations."""
    score_value = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    method = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Zs")))
    ))
    
    return AnomalyScore(value=score_value, method=method)


@st.composite
def valid_contamination_rates(draw):
    """Generate valid contamination rates."""
    rate_value = draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False))
    return ContaminationRate(value=rate_value)


@st.composite
def valid_data_points(draw):
    """Generate valid data points for anomalies."""
    n_fields = draw(st.integers(min_value=1, max_value=10))
    
    data_point = {}
    for i in range(n_fields):
        field_name = f"field_{i}"
        field_value = draw(st.one_of(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.integers(min_value=-1000000, max_value=1000000),
            st.text(min_size=1, max_size=100),
            st.booleans()
        ))
        data_point[field_name] = field_value
    
    return data_point


@st.composite
def valid_anomalies(draw):
    """Generate valid anomaly instances."""
    score = draw(valid_anomaly_scores())
    data_point = draw(valid_data_points())
    detector_name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd"))))
    
    # Optional explanation
    explanation = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=500)
    ))
    
    return Anomaly(
        score=score.value,
        data_point=data_point,
        detector_name=detector_name,
        explanation=explanation
    )


@st.composite
def valid_dataframes(draw):
    """Generate valid pandas DataFrames."""
    n_rows = draw(st.integers(min_value=2, max_value=100))
    n_cols = draw(st.integers(min_value=1, max_value=10))
    
    columns = [f"col_{i}" for i in range(n_cols)]
    
    data = {}
    for col in columns:
        data[col] = draw(st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows
        ))
    
    return pd.DataFrame(data)


@st.composite
def valid_datasets(draw):
    """Generate valid dataset instances."""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd"))))
    data = draw(valid_dataframes())
    
    # Optional description
    description = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=200)
    ))
    
    # Optional target column
    target_column = draw(st.one_of(
        st.none(),
        st.sampled_from(data.columns.tolist())
    ))
    
    return Dataset(
        name=name,
        data=data,
        description=description,
        target_column=target_column
    )


class TestAnomalyScoreProperties:
    """Property-based tests for AnomalyScore value object."""
    
    @given(valid_anomaly_scores())
    def test_anomaly_score_value_invariants(self, score):
        """Test that AnomalyScore maintains value invariants."""
        assert 0.0 <= score.value <= 1.0
        assert score.is_valid()
        assert score.normalized() == score.value
        
    @given(valid_anomaly_scores())
    def test_anomaly_score_serialization_roundtrip(self, score):
        """Test that AnomalyScore can be serialized and deserialized."""
        score_dict = score.to_dict()
        assert isinstance(score_dict, dict)
        assert "value" in score_dict
        assert score_dict["value"] == score.value
        
    @given(valid_anomaly_scores(), st.floats(min_value=0.0, max_value=1.0))
    def test_anomaly_score_threshold_comparison(self, score, threshold):
        """Test threshold comparison behavior."""
        result = score.exceeds_threshold(threshold)
        assert isinstance(result, bool)
        assert result == (score.value > threshold)
        
    @given(valid_anomaly_scores())
    def test_anomaly_score_severity_category(self, score):
        """Test severity category assignment."""
        category = score.severity_category
        assert category in ["info", "low", "medium", "high", "critical"]
        
        # Test category consistency
        if score.value >= 0.9:
            assert category == "critical"
        elif score.value >= 0.7:
            assert category == "high"
        elif score.value >= 0.5:
            assert category == "medium"
        elif score.value >= 0.3:
            assert category == "low"
        else:
            assert category == "info"
            
    @given(st.floats(min_value=-1000, max_value=-0.001))
    def test_anomaly_score_invalid_negative_values(self, invalid_value):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            AnomalyScore(value=invalid_value)
            
    @given(st.floats(min_value=1.001, max_value=1000))
    def test_anomaly_score_invalid_high_values(self, invalid_value):
        """Test that values above 1.0 are rejected."""
        with pytest.raises(ValidationError):
            AnomalyScore(value=invalid_value)
            
    @given(st.one_of(st.floats(allow_nan=True), st.floats(allow_infinity=True)))
    def test_anomaly_score_rejects_nan_infinity(self, invalid_value):
        """Test that NaN and infinity values are rejected."""
        assume(not (0.0 <= invalid_value <= 1.0))
        with pytest.raises(ValidationError):
            AnomalyScore(value=invalid_value)


class TestContaminationRateProperties:
    """Property-based tests for ContaminationRate value object."""
    
    @given(valid_contamination_rates())
    def test_contamination_rate_value_invariants(self, rate):
        """Test that ContaminationRate maintains value invariants."""
        assert 0.0 <= rate.value <= 0.5
        assert rate.is_valid()
        assert 0.0 <= rate.as_percentage() <= 50.0
        
    @given(valid_contamination_rates())
    def test_contamination_rate_percentage_conversion(self, rate):
        """Test percentage conversion accuracy."""
        percentage = rate.as_percentage()
        assert abs(percentage - (rate.value * 100.0)) < 1e-10
        
    @given(valid_contamination_rates())
    def test_contamination_rate_category_assignment(self, rate):
        """Test category assignment consistency."""
        category = rate.category
        assert category in ["low", "medium", "high", "very_high"]
        
        if rate.value <= 0.05:
            assert category == "low"
        elif rate.value <= 0.15:
            assert category == "medium"
        elif rate.value <= 0.3:
            assert category == "high"
        else:
            assert category == "very_high"
            
    @given(st.floats(min_value=0.0, max_value=50.0))
    def test_contamination_rate_from_percentage(self, percentage):
        """Test creation from percentage values."""
        rate = ContaminationRate.from_percentage(percentage)
        assert abs(rate.value - (percentage / 100.0)) < 1e-10
        assert abs(rate.as_percentage() - percentage) < 1e-10
        
    @given(st.floats(min_value=-1000, max_value=-0.001))
    def test_contamination_rate_invalid_negative_values(self, invalid_value):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            ContaminationRate(value=invalid_value)
            
    @given(st.floats(min_value=0.501, max_value=1000))
    def test_contamination_rate_invalid_high_values(self, invalid_value):
        """Test that values above 0.5 are rejected."""
        with pytest.raises(ValidationError):
            ContaminationRate(value=invalid_value)
            
    def test_contamination_rate_class_constants(self):
        """Test that class constants are properly initialized."""
        assert ContaminationRate.AUTO.value == 0.1
        assert ContaminationRate.LOW.value == 0.05
        assert ContaminationRate.MEDIUM.value == 0.1
        assert ContaminationRate.HIGH.value == 0.2
        
        # Test auto detection
        assert ContaminationRate.AUTO.is_auto
        assert not ContaminationRate.LOW.is_auto
        assert ContaminationRate.MEDIUM.is_auto


class TestAnomalyProperties:
    """Property-based tests for Anomaly entity."""
    
    @given(valid_anomalies())
    def test_anomaly_initialization_invariants(self, anomaly):
        """Test that Anomaly maintains initialization invariants."""
        assert 0.0 <= anomaly.score <= 1.0
        assert isinstance(anomaly.data_point, dict)
        assert len(anomaly.data_point) > 0
        assert anomaly.detector_name.strip() != ""
        assert isinstance(anomaly.timestamp, datetime)
        
    @given(valid_anomalies())
    def test_anomaly_severity_calculation(self, anomaly):
        """Test severity calculation consistency."""
        severity = anomaly.severity
        assert severity in ["info", "low", "medium", "high", "critical"]
        
        # Test consistency with score
        if anomaly.score >= 0.9:
            assert severity == "critical"
        elif anomaly.score >= 0.7:
            assert severity == "high"
        elif anomaly.score >= 0.5:
            assert severity == "medium"
        elif anomaly.score >= 0.3:
            assert severity == "low"
        else:
            assert severity == "info"
            
    @given(valid_anomalies())
    def test_anomaly_risk_score_calculation(self, anomaly):
        """Test risk score calculation."""
        risk_score = anomaly.risk_score
        assert 0.0 <= risk_score <= 1.0
        
        # Without confidence, risk should equal base score
        if anomaly.confidence_level == 0.0:
            assert risk_score == anomaly.score
            
    @given(valid_anomalies())
    def test_anomaly_serialization_roundtrip(self, anomaly):
        """Test anomaly serialization."""
        anomaly_dict = anomaly.to_dict()
        assert isinstance(anomaly_dict, dict)
        assert "score" in anomaly_dict
        assert "data_point" in anomaly_dict
        assert "detector_name" in anomaly_dict
        assert "severity" in anomaly_dict
        
    @given(valid_anomalies(), st.text(min_size=1, max_size=100))
    def test_anomaly_metadata_operations(self, anomaly, key):
        """Test metadata operations."""
        # Clean key to avoid reserved names
        clean_key = key.replace("_", "").replace("-", "")
        assume(clean_key not in ["id", "score", "detector", "timestamp"])
        
        test_value = "test_value"
        anomaly.add_metadata(clean_key, test_value)
        assert anomaly.metadata[clean_key] == test_value
        
    @given(st.floats(min_value=-1000, max_value=-0.001))
    def test_anomaly_invalid_negative_scores(self, invalid_score):
        """Test that negative scores are rejected."""
        with pytest.raises(ValidationError):
            Anomaly(
                score=invalid_score,
                data_point={"field": "value"},
                detector_name="test"
            )
            
    @given(st.floats(min_value=1.001, max_value=1000))
    def test_anomaly_invalid_high_scores(self, invalid_score):
        """Test that scores above 1.0 are rejected."""
        with pytest.raises(ValidationError):
            Anomaly(
                score=invalid_score,
                data_point={"field": "value"},
                detector_name="test"
            )


class TestDatasetProperties:
    """Property-based tests for Dataset entity."""
    
    @given(valid_datasets())
    def test_dataset_initialization_invariants(self, dataset):
        """Test that Dataset maintains initialization invariants."""
        assert dataset.name.strip() != ""
        assert not dataset.data.empty
        assert dataset.n_samples > 0
        assert dataset.n_features > 0
        assert dataset.n_samples == len(dataset.data)
        
    @given(valid_datasets())
    def test_dataset_feature_counting(self, dataset):
        """Test feature counting accuracy."""
        expected_features = len(dataset.data.columns)
        if dataset.target_column:
            expected_features -= 1
            
        assert dataset.n_features == expected_features
        
    @given(valid_datasets())
    def test_dataset_feature_names_consistency(self, dataset):
        """Test feature names consistency."""
        if dataset.feature_names:
            assert len(dataset.feature_names) == dataset.n_features
            assert all(isinstance(name, str) for name in dataset.feature_names)
            assert len(set(dataset.feature_names)) == len(dataset.feature_names)  # Unique names
            
    @given(valid_datasets())
    def test_dataset_target_column_validation(self, dataset):
        """Test target column validation."""
        if dataset.target_column:
            assert dataset.target_column in dataset.data.columns
            assert dataset.has_target
            assert dataset.target is not None
            assert len(dataset.target) == dataset.n_samples
        else:
            assert not dataset.has_target
            assert dataset.target is None
            
    @given(valid_datasets())
    def test_dataset_numeric_categorical_features(self, dataset):
        """Test numeric and categorical feature identification."""
        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()
        
        # Should be disjoint sets
        assert len(set(numeric_features) & set(categorical_features)) == 0
        
        # All should be valid feature names
        all_features = set(numeric_features + categorical_features)
        if dataset.target_column:
            all_features.discard(dataset.target_column)
            
        assert all_features.issubset(set(dataset.data.columns))
        
    @given(valid_datasets(), st.integers(min_value=1, max_value=50))
    def test_dataset_sampling(self, dataset, n):
        """Test dataset sampling functionality."""
        if n <= dataset.n_samples:
            sampled = dataset.sample(n)
            assert sampled.n_samples == n
            assert sampled.n_features == dataset.n_features
            assert sampled.target_column == dataset.target_column
            assert "parent_dataset_id" in sampled.metadata
        else:
            with pytest.raises(ValueError):
                dataset.sample(n)
                
    @given(valid_datasets(), st.floats(min_value=0.1, max_value=0.9))
    def test_dataset_splitting(self, dataset, test_size):
        """Test dataset splitting functionality."""
        train_dataset, test_dataset = dataset.split(test_size=test_size)
        
        # Check sizes
        expected_train_size = int(dataset.n_samples * (1 - test_size))
        expected_test_size = dataset.n_samples - expected_train_size
        
        assert abs(train_dataset.n_samples - expected_train_size) <= 1
        assert abs(test_dataset.n_samples - expected_test_size) <= 1
        
        # Check consistency
        assert train_dataset.n_features == dataset.n_features
        assert test_dataset.n_features == dataset.n_features
        assert train_dataset.target_column == dataset.target_column
        assert test_dataset.target_column == dataset.target_column
        
    @given(valid_datasets())
    def test_dataset_memory_usage(self, dataset):
        """Test memory usage calculation."""
        memory_usage = dataset.memory_usage
        assert isinstance(memory_usage, int)
        assert memory_usage > 0
        
    @given(valid_datasets())
    def test_dataset_summary_statistics(self, dataset):
        """Test summary statistics generation."""
        summary = dataset.summary()
        assert isinstance(summary, dict)
        
        required_keys = ["id", "name", "shape", "n_samples", "n_features", 
                        "memory_usage_mb", "has_target", "created_at"]
        assert all(key in summary for key in required_keys)
        
        assert summary["n_samples"] == dataset.n_samples
        assert summary["n_features"] == dataset.n_features
        assert summary["has_target"] == dataset.has_target


class TestBusinessRuleProperties:
    """Property-based tests for business rule validation."""
    
    @given(valid_anomalies())
    def test_high_confidence_anomaly_warnings(self, anomaly):
        """Test that high-confidence anomalies trigger appropriate warnings."""
        # High score anomalies should ideally have explanations
        if anomaly.score > 0.8:
            # This should trigger a warning if no explanation is provided
            assert anomaly.explanation is not None or anomaly.score <= 0.8
            
    @given(valid_anomalies())
    def test_anomaly_attention_requirements(self, anomaly):
        """Test immediate attention requirements."""
        requires_attention = anomaly.requires_immediate_attention
        assert isinstance(requires_attention, bool)
        
        # Logic should be consistent
        expected = (anomaly.severity in ["critical", "high"] and 
                   anomaly.confidence_level > 0.7)
        assert requires_attention == expected
        
    @given(valid_datasets())
    def test_dataset_minimum_requirements(self, dataset):
        """Test dataset minimum requirements."""
        # Datasets must have at least 2 rows and 1 column
        assert dataset.n_samples >= 2
        assert dataset.n_features >= 1
        assert not dataset.data.empty
        
    @given(valid_contamination_rates())
    def test_contamination_rate_business_constraints(self, rate):
        """Test business constraints for contamination rates."""
        # Contamination rates should be reasonable for anomaly detection
        assert 0.0 <= rate.value <= 0.5
        
        # Very high contamination rates should be flagged
        if rate.value > 0.3:
            assert rate.category == "very_high"
            
    @given(valid_anomaly_scores(), valid_anomaly_scores())
    def test_anomaly_score_comparison_transitivity(self, score1, score2):
        """Test transitivity of anomaly score comparisons."""
        # Test comparison operators
        if score1.value < score2.value:
            assert score1 < score2
            assert score2 > score1
            assert not (score1 > score2)
            assert not (score2 < score1)
        elif score1.value == score2.value:
            assert score1 <= score2
            assert score2 <= score1
            assert score1 >= score2
            assert score2 >= score1
        else:
            assert score1 > score2
            assert score2 < score1
            assert not (score1 < score2)
            assert not (score2 > score1)


class TestErrorHandlingProperties:
    """Property-based tests for error handling and edge cases."""
    
    @given(st.text(max_size=0))
    def test_empty_string_validation(self, empty_string):
        """Test empty string validation across entities."""
        with pytest.raises(ValidationError):
            Anomaly(score=0.5, data_point={"field": "value"}, detector_name=empty_string)
            
        with pytest.raises(ValidationError):
            Dataset(name=empty_string, data=pd.DataFrame({"col": [1, 2]}))
            
    @given(st.dictionaries(st.text(), st.text(), max_size=0))
    def test_empty_dict_validation(self, empty_dict):
        """Test empty dictionary validation."""
        with pytest.raises(ValidationError):
            Anomaly(score=0.5, data_point=empty_dict, detector_name="test")
            
    @given(st.text(min_size=1, max_size=10))
    def test_reserved_key_validation(self, field_name):
        """Test reserved key validation in data points."""
        reserved_keys = ["_id", "_score", "_detector", "_timestamp"]
        
        for reserved_key in reserved_keys:
            with pytest.raises(ValidationError):
                Anomaly(
                    score=0.5,
                    data_point={reserved_key: "value"},
                    detector_name="test"
                )
                
    @given(st.text(min_size=1, max_size=50))
    def test_special_character_validation(self, text_input):
        """Test special character handling in various fields."""
        # Only test inputs that would be invalid
        assume(not text_input.replace("_", "").replace("-", "").replace(".", "").isalnum())
        
        # Should fail for detector names with invalid characters
        with pytest.raises(ValidationError):
            Anomaly(
                score=0.5,
                data_point={"field": "value"},
                detector_name=text_input
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
