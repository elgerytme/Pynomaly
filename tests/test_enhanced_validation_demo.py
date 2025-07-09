"""Test demonstration of enhanced validation features implemented in Issue #6."""

import sys
import pandas as pd
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.domain.validation.error_strategies import (
    ValidationErrorHandler,
    StrictValidationStrategy,
    LenientValidationStrategy,
    ValidationResult,
    ValidationSeverity,
)


class TestEnhancedValidationFeatures:
    """Test suite demonstrating enhanced validation features."""
    
    def test_anomaly_basic_validation(self):
        """Test basic anomaly validation with current implementation."""
        # Valid anomaly
        anomaly = Anomaly(
            score=0.8,
            data_point={"temperature": 25.5, "pressure": 1013.25},
            detector_name="test-detector"
        )
        
        assert anomaly.score == 0.8
        assert anomaly.severity == "high"
        assert anomaly.detector_name == "test-detector"
        assert len(anomaly.data_point) == 2
        
    def test_anomaly_validation_errors(self):
        """Test anomaly validation error handling."""
        # Test invalid score
        with pytest.raises(TypeError):
            Anomaly(
                score="invalid",
                data_point={"field": "value"},
                detector_name="test"
            )
        
        # Test empty detector name
        with pytest.raises(ValueError):
            Anomaly(
                score=0.5,
                data_point={"field": "value"},
                detector_name=""
            )
        
        # Test invalid data point
        with pytest.raises(TypeError):
            Anomaly(
                score=0.5,
                data_point="not a dict",
                detector_name="test"
            )
    
    def test_dataset_basic_validation(self):
        """Test basic dataset validation."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "target": [0, 1, 0, 1]
        })
        
        dataset = Dataset(
            name="test-dataset",
            data=df,
            description="Test dataset for validation",
            target_column="target"
        )
        
        assert dataset.name == "test-dataset"
        assert dataset.n_samples == 4
        assert dataset.n_features == 2  # Excluding target
        assert dataset.has_target
        assert dataset.target_column == "target"
        
    def test_dataset_validation_errors(self):
        """Test dataset validation error handling."""
        # Test empty name
        with pytest.raises(ValueError):
            Dataset(
                name="",
                data=pd.DataFrame({"col": [1, 2, 3]})
            )
        
        # Test empty data
        with pytest.raises(ValueError):
            Dataset(
                name="test",
                data=pd.DataFrame()
            )
        
        # Test invalid target column
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(ValueError):
            Dataset(
                name="test",
                data=df,
                target_column="nonexistent"
            )
    
    def test_anomaly_score_validation(self):
        """Test anomaly score validation."""
        # Valid score
        score = AnomalyScore(value=0.75)
        assert score.value == 0.75
        assert score.is_valid()
        
        # Test with method
        score_with_method = AnomalyScore(value=0.6, method="isolation_forest")
        assert score_with_method.method == "isolation_forest"
        
        # Test comparison
        score1 = AnomalyScore(value=0.3)
        score2 = AnomalyScore(value=0.7)
        assert score1 < score2
        assert score2 > score1
        
    def test_contamination_rate_validation(self):
        """Test contamination rate validation."""
        # Valid rate
        rate = ContaminationRate(value=0.1)
        assert rate.value == 0.1
        assert rate.as_percentage() == 10.0
        assert rate.is_valid()
        
        # Test class methods
        auto_rate = ContaminationRate.auto()
        assert auto_rate.value == 0.1
        
        low_rate = ContaminationRate.low()
        assert low_rate.value == 0.05
        
        high_rate = ContaminationRate.high()
        assert high_rate.value == 0.2
        
    def test_validation_error_handlers(self):
        """Test validation error handling strategies."""
        # Test strict strategy
        strict_handler = ValidationErrorHandler(StrictValidationStrategy())
        
        # Test lenient strategy
        lenient_handler = ValidationErrorHandler(LenientValidationStrategy())
        
        # Test error handling
        result = ValidationResult(is_valid=False)
        result.add_error("Test error", field="test_field", code="test_error")
        
        assert not result.is_valid
        assert result.has_errors()
        assert len(result.errors) == 1
        assert result.errors[0]["message"] == "Test error"
        
        # Test warnings
        result.add_warning("Test warning", field="test_field", code="test_warning")
        assert result.has_warnings()
        assert len(result.warnings) == 1
        
    def test_anomaly_severity_calculation(self):
        """Test anomaly severity calculation."""
        # Test different severity levels
        critical_anomaly = Anomaly(
            score=0.95,
            data_point={"field": "value"},
            detector_name="test"
        )
        assert critical_anomaly.severity == "critical"
        
        high_anomaly = Anomaly(
            score=0.8,
            data_point={"field": "value"},
            detector_name="test"
        )
        assert high_anomaly.severity == "high"
        
        medium_anomaly = Anomaly(
            score=0.6,
            data_point={"field": "value"},
            detector_name="test"
        )
        assert medium_anomaly.severity == "medium"
        
        low_anomaly = Anomaly(
            score=0.4,
            data_point={"field": "value"},
            detector_name="test"
        )
        assert low_anomaly.severity == "low"
    
    def test_dataset_feature_operations(self):
        """Test dataset feature operations."""
        # Mixed data types
        df = pd.DataFrame({
            "numeric1": [1, 2, 3],
            "numeric2": [1.5, 2.5, 3.5],
            "categorical": ["A", "B", "C"],
            "boolean": [True, False, True]
        })
        
        dataset = Dataset(name="mixed-data", data=df)
        
        # Test feature identification
        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()
        
        # Should identify numeric and categorical features
        assert len(numeric_features) > 0
        assert len(categorical_features) > 0
        
        # Test sampling
        sample = dataset.sample(n=2)
        assert sample.n_samples == 2
        assert sample.n_features == dataset.n_features
        
        # Test splitting
        train, test = dataset.split(test_size=0.3)
        assert train.n_samples + test.n_samples == dataset.n_samples
        assert train.n_features == dataset.n_features
        assert test.n_features == dataset.n_features
    
    def test_validation_error_aggregation(self):
        """Test validation error aggregation."""
        handler = ValidationErrorHandler()
        
        # Create multiple validation results
        result1 = ValidationResult(is_valid=True)
        result1.add_warning("Warning 1", field="field1")
        
        result2 = ValidationResult(is_valid=False)
        result2.add_error("Error 1", field="field2")
        
        result3 = ValidationResult(is_valid=True)
        result3.add_warning("Warning 2", field="field3")
        
        # Aggregate results
        aggregated = handler.aggregate_results([result1, result2, result3])
        
        assert not aggregated.is_valid  # Should be invalid due to result2
        assert len(aggregated.errors) == 1
        assert len(aggregated.warnings) == 2
        
    def test_anomaly_metadata_operations(self):
        """Test anomaly metadata operations."""
        anomaly = Anomaly(
            score=0.7,
            data_point={"field": "value"},
            detector_name="test"
        )
        
        # Test metadata operations
        anomaly.add_metadata("confidence", 0.9)
        anomaly.add_metadata("algorithm", "isolation_forest")
        
        assert anomaly.metadata["confidence"] == 0.9
        assert anomaly.metadata["algorithm"] == "isolation_forest"
        
        # Test to_dict conversion
        anomaly_dict = anomaly.to_dict()
        assert "score" in anomaly_dict
        assert "severity" in anomaly_dict
        assert "metadata" in anomaly_dict
        assert anomaly_dict["metadata"]["confidence"] == 0.9
        
    def test_dataset_summary_statistics(self):
        """Test dataset summary statistics."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50]
        })
        
        dataset = Dataset(name="summary-test", data=df)
        
        summary = dataset.summary()
        
        assert summary["name"] == "summary-test"
        assert summary["n_samples"] == 5
        assert summary["n_features"] == 2
        assert summary["shape"] == (5, 2)
        assert not summary["has_target"]
        assert "memory_usage_mb" in summary
        assert "created_at" in summary
        
    def test_contamination_rate_constants(self):
        """Test contamination rate constants."""
        # Test class constants
        assert ContaminationRate.AUTO.value == 0.1
        assert ContaminationRate.LOW.value == 0.05
        assert ContaminationRate.MEDIUM.value == 0.1
        assert ContaminationRate.HIGH.value == 0.2
        
        # Test string representation
        rate = ContaminationRate(value=0.15)
        rate_str = str(rate)
        assert "%" in rate_str
        assert "15" in rate_str
        
    def test_validation_severity_levels(self):
        """Test validation severity levels."""
        # Test severity enumeration
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"
        
        # Test validation result with different severities
        result = ValidationResult(is_valid=False)
        result.add_error("Critical error", code="critical_issue")
        result.add_warning("Minor warning", code="minor_issue")
        
        assert result.has_errors()
        assert result.has_warnings()
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
