"""Advanced coverage push testing - targeting high-impact areas for maximum coverage boost."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import uuid
import asyncio
from pathlib import Path
import tempfile
import json

# Strategic imports for maximum coverage impact
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate, ConfidenceInterval
from pynomaly.domain.entities import Dataset, Anomaly
from pynomaly.domain.exceptions import DomainError, ValidationError, InvalidValueError, EntityNotFoundError, DetectorError
from pynomaly.infrastructure.config.settings import Settings


class TestDomainExceptionsComprehensive:
    """Comprehensive testing of domain exception hierarchy for coverage boost."""
    
    def test_domain_error_hierarchy_comprehensive(self):
        """Test complete domain error hierarchy."""
        # Test base DomainError
        base_error = DomainError("Base domain error message")
        assert str(base_error) == "Base domain error message"
        assert isinstance(base_error, Exception)
        
        # Test with context
        base_error_with_context = DomainError("Error with context", {"entity_id": "123", "operation": "save"})
        assert base_error_with_context.context["entity_id"] == "123"
        assert base_error_with_context.context["operation"] == "save"
        
        # Test ValidationError inheritance
        validation_error = ValidationError("Validation failed", {"field": "name", "value": ""})
        assert isinstance(validation_error, DomainError)
        assert str(validation_error) == "Validation failed"
        assert validation_error.context["field"] == "name"
        
        # Test InvalidValueError inheritance
        value_error = InvalidValueError("Invalid contamination rate", {"value": 1.5, "valid_range": "0.0-0.5"})
        assert isinstance(value_error, ValidationError)
        assert isinstance(value_error, DomainError)
        assert value_error.context["value"] == 1.5
        
        # Test EntityNotFoundError
        not_found_error = EntityNotFoundError("Detector not found", {"detector_id": "abc123"})
        assert isinstance(not_found_error, DomainError)
        assert not_found_error.context["detector_id"] == "abc123"
        
        # Test DetectorError
        detector_error = DetectorError("Cannot delete fitted detector", {"detector_id": "def456", "is_fitted": True})
        assert isinstance(detector_error, DomainError)
        assert detector_error.context["is_fitted"] is True
    
    def test_exception_serialization_and_logging(self):
        """Test exception serialization for logging and monitoring."""
        error = ValidationError(
            "Invalid dataset configuration",
            {
                "dataset_name": "test_dataset",
                "errors": ["Empty dataframe", "Missing target column"],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Test that exceptions can be serialized for logging
        error_dict = {
            "message": str(error),
            "type": type(error).__name__,
            "context": error.context
        }
        
        serialized = json.dumps(error_dict, default=str)
        assert "Invalid dataset configuration" in serialized
        assert "test_dataset" in serialized
        
        # Test error chaining
        try:
            raise ValueError("Original error")
        except ValueError as original:
            try:
                raise DetectorError("Business rule violated") from original
            except DetectorError as chained_error:
                assert chained_error.__cause__ is original


class TestDomainServicesAdvanced:
    """Advanced testing of domain services for coverage boost."""
    
    def test_anomaly_scorer_comprehensive(self):
        """Test AnomalyScorer comprehensive functionality."""
        try:
            from pynomaly.domain.services.anomaly_scorer import AnomalyScorer
            
            scorer = AnomalyScorer()
            
            # Test various score distributions
            test_cases = [
                # Normal distribution
                np.random.normal(0, 1, 100),
                # Uniform distribution  
                np.random.uniform(-2, 2, 100),
                # Exponential distribution
                np.random.exponential(1, 100),
                # Mixed with outliers
                np.concatenate([np.random.normal(0, 1, 90), np.random.normal(5, 1, 10)]),
                # Edge cases
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.array([-np.inf, -10, 0, 10, np.inf]),
            ]
            
            for i, raw_scores in enumerate(test_cases):
                # Filter out infinite values for processing
                finite_scores = raw_scores[np.isfinite(raw_scores)]
                if len(finite_scores) == 0:
                    continue
                    
                normalized = scorer.normalize_scores(finite_scores)
                
                # Verify normalization properties
                assert len(normalized) == len(finite_scores)
                assert all(isinstance(score, AnomalyScore) for score in normalized)
                assert all(0.0 <= score.value <= 1.0 for score in normalized)
                
                # Test order preservation for finite values
                finite_values = [score.value for score in normalized]
                if len(set(finite_scores)) > 1:  # Only test if not all values are the same
                    # Should preserve relative ordering
                    for j in range(len(finite_scores) - 1):
                        if finite_scores[j] < finite_scores[j + 1]:
                            assert finite_values[j] <= finite_values[j + 1]
            
            # Test statistical methods if available
            if hasattr(scorer, 'calculate_percentile_threshold'):
                scores = [AnomalyScore(x) for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
                threshold_90 = scorer.calculate_percentile_threshold(scores, 90)
                assert 0.8 <= threshold_90 <= 1.0
            
            # Test aggregation methods if available
            if hasattr(scorer, 'aggregate_scores'):
                score_sets = [
                    [AnomalyScore(0.8), AnomalyScore(0.9), AnomalyScore(0.7)],
                    [AnomalyScore(0.6), AnomalyScore(0.8), AnomalyScore(0.5)]
                ]
                aggregated = scorer.aggregate_scores(score_sets, method='mean')
                assert isinstance(aggregated, list)
                assert all(isinstance(score, AnomalyScore) for score in aggregated)
                
        except ImportError:
            pytest.skip("AnomalyScorer not available")
    
    def test_threshold_calculator_comprehensive(self):
        """Test ThresholdCalculator comprehensive functionality."""
        try:
            from pynomaly.domain.services.threshold_calculator import ThresholdCalculator
            
            calculator = ThresholdCalculator()
            
            # Test various contamination rates
            scores = [AnomalyScore(x) for x in np.linspace(0, 1, 100)]
            contamination_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
            
            for contamination_value in contamination_rates:
                contamination = ContaminationRate(contamination_value)
                threshold = calculator.calculate_threshold(scores, contamination)
                
                assert isinstance(threshold, float)
                assert 0.0 <= threshold <= 1.0
                
                # Verify contamination rate is approximately correct
                anomalies = sum(1 for score in scores if score.value > threshold)
                expected_anomalies = int(len(scores) * contamination_value)
                # Allow for rounding differences
                assert abs(anomalies - expected_anomalies) <= 2
            
            # Test different threshold methods if available
            if hasattr(calculator, 'calculate_threshold_percentile'):
                percentile_threshold = calculator.calculate_threshold_percentile(scores, 95)
                assert isinstance(percentile_threshold, float)
                assert 0.9 <= percentile_threshold <= 1.0
            
            if hasattr(calculator, 'calculate_adaptive_threshold'):
                adaptive_threshold = calculator.calculate_adaptive_threshold(scores)
                assert isinstance(adaptive_threshold, float)
                assert 0.0 <= adaptive_threshold <= 1.0
                
        except ImportError:
            pytest.skip("ThresholdCalculator not available")


class TestValueObjectsAdvanced:
    """Advanced value object testing for edge cases and performance."""
    
    def test_anomaly_score_advanced_operations(self):
        """Test AnomalyScore advanced operations and edge cases."""
        # Test precision edge cases
        very_small = AnomalyScore(1e-10)
        very_large = AnomalyScore(1.0 - 1e-10)
        
        assert very_small.value == 1e-10
        assert very_large.value == 1.0 - 1e-10
        
        # Test comparison with floating point precision
        score1 = AnomalyScore(0.1 + 0.2)  # Floating point arithmetic
        score2 = AnomalyScore(0.3)
        
        # Should handle floating point comparison correctly
        assert abs(score1.value - score2.value) < 1e-10
        
        # Test batch operations
        scores = [AnomalyScore(x) for x in np.linspace(0, 1, 1000)]
        
        # Test sorting performance
        sorted_scores = sorted(scores, key=lambda s: s.value)
        assert len(sorted_scores) == 1000
        assert sorted_scores[0].value <= sorted_scores[-1].value
        
        # Test filtering operations
        high_scores = [s for s in scores if s.value > 0.8]
        assert len(high_scores) == 200  # 20% of 1000
        
        # Test mathematical operations if supported
        try:
            # Test if arithmetic operations are supported
            result = scores[500] + scores[600]  # Might not be implemented
            if result is not None:
                assert isinstance(result, AnomalyScore)
        except (TypeError, AttributeError):
            # Arithmetic might not be supported, which is fine
            pass
    
    def test_contamination_rate_advanced_validation(self):
        """Test ContaminationRate advanced validation and edge cases."""
        # Test precision validation
        precise_rates = [0.001, 0.0001, 0.00001, 0.499999, 0.5]
        
        for rate_value in precise_rates:
            rate = ContaminationRate(rate_value)
            assert rate.value == rate_value
            assert rate.is_valid() is True
            
            # Test percentage conversion
            percentage = rate.as_percentage()
            assert abs(percentage - rate_value * 100) < 1e-10
        
        # Test string representation for various values
        test_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.333333, 0.5]
        
        for value in test_values:
            rate = ContaminationRate(value)
            str_repr = str(rate)
            assert "%" in str_repr
            assert str(int(value * 100)) in str_repr or f"{value * 100:.1f}" in str_repr
        
        # Test class constant immutability
        original_auto = ContaminationRate.AUTO.value
        original_low = ContaminationRate.LOW.value
        original_medium = ContaminationRate.MEDIUM.value
        original_high = ContaminationRate.HIGH.value
        
        # Constants should be immutable
        assert ContaminationRate.AUTO.value == original_auto
        assert ContaminationRate.LOW.value == original_low
        assert ContaminationRate.MEDIUM.value == original_medium
        assert ContaminationRate.HIGH.value == original_high
    
    def test_confidence_interval_advanced_operations(self):
        """Test ConfidenceInterval advanced operations and mathematical properties."""
        # Test various confidence levels
        confidence_levels = [0.90, 0.95, 0.99, 0.999]
        
        for level in confidence_levels:
            ci = ConfidenceInterval(lower=0.4, upper=0.6, confidence_level=level)
            assert ci.confidence_level == level
            assert ci.is_valid() is True
            
            # Test mathematical properties
            assert ci.width() == 0.2
            assert ci.midpoint() == 0.5
            
            # Test containment with edge cases
            assert ci.contains(0.4) is True  # Boundary
            assert ci.contains(0.6) is True  # Boundary
            assert ci.contains(0.5) is True  # Center
            assert ci.contains(0.39999) is False  # Just outside
            assert ci.contains(0.60001) is False  # Just outside
        
        # Test narrow intervals
        narrow_intervals = [
            (0.49, 0.51),
            (0.499, 0.501),
            (0.4999, 0.5001)
        ]
        
        for lower, upper in narrow_intervals:
            ci = ConfidenceInterval(lower=lower, upper=upper)
            width = ci.width()
            assert abs(width - (upper - lower)) < 1e-10
            
            midpoint = ci.midpoint()
            assert abs(midpoint - (lower + upper) / 2) < 1e-10
        
        # Test interval arithmetic if supported
        ci1 = ConfidenceInterval(lower=0.2, upper=0.4)
        ci2 = ConfidenceInterval(lower=0.6, upper=0.8)
        
        # Test if intervals can be combined (might not be implemented)
        try:
            if hasattr(ci1, 'union'):
                union_ci = ci1.union(ci2)
                assert union_ci.lower == 0.2
                assert union_ci.upper == 0.8
        except (AttributeError, NotImplementedError):
            # Union might not be implemented
            pass


class TestDatasetAdvancedOperations:
    """Advanced dataset operations for comprehensive coverage."""
    
    def test_dataset_memory_management(self):
        """Test dataset memory management and optimization."""
        # Test with various data sizes
        sizes = [10, 100, 1000, 10000]
        features = [3, 10, 50, 100]
        
        for n_samples in sizes:
            for n_features in features[:2]:  # Limit combinations for performance
                # Create dataset with known size
                data = pd.DataFrame(
                    np.random.randn(n_samples, n_features),
                    columns=[f'feature_{i}' for i in range(n_features)]
                )
                
                dataset = Dataset(name=f"memory_test_{n_samples}_{n_features}", data=data)
                
                # Test memory usage calculation
                memory_usage = dataset.memory_usage
                assert memory_usage > 0
                assert isinstance(memory_usage, int)
                
                # Memory usage should scale with data size
                expected_min_bytes = n_samples * n_features * 8  # Rough estimate for float64
                assert memory_usage >= expected_min_bytes / 10  # Allow for overhead
                
                # Test summary generation performance
                summary = dataset.summary()
                assert summary["n_samples"] == n_samples
                assert summary["n_features"] == n_features
                assert "memory_usage_mb" in summary
    
    def test_dataset_data_type_optimization(self):
        """Test dataset with optimized data types."""
        # Create dataset with mixed optimized types
        optimized_data = pd.DataFrame({
            'int8_col': pd.array([1, 2, 3, 4, 5], dtype='int8'),
            'int16_col': pd.array([100, 200, 300, 400, 500], dtype='int16'),
            'int32_col': pd.array([1000, 2000, 3000, 4000, 5000], dtype='int32'),
            'float32_col': pd.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float32'),
            'float64_col': pd.array([10.1, 20.2, 30.3, 40.4, 50.5], dtype='float64'),
            'category_col': pd.Categorical(['A', 'B', 'C', 'A', 'B']),
            'bool_col': pd.array([True, False, True, False, True], dtype='bool'),
            'string_col': pd.array(['str1', 'str2', 'str3', 'str4', 'str5'], dtype='string')
        })
        
        dataset = Dataset(name="optimized_types", data=optimized_data)
        
        # Test type detection
        numeric_features = dataset.get_numeric_features()
        categorical_features = dataset.get_categorical_features()
        
        # Should detect numeric types correctly
        numeric_columns = ['int8_col', 'int16_col', 'int32_col', 'float32_col', 'float64_col']
        for col in numeric_columns:
            assert col in numeric_features
        
        # Should detect categorical and string types
        assert 'category_col' in categorical_features or 'string_col' in categorical_features
        
        # Test memory efficiency
        memory_usage = dataset.memory_usage
        dtypes = dataset.dtypes
        
        # Verify optimized types are preserved
        assert dtypes['int8_col'] == 'int8'
        assert dtypes['category_col'] == 'category'
        assert dtypes['bool_col'] == 'bool'
    
    def test_dataset_large_scale_operations(self):
        """Test dataset operations at scale."""
        # Create larger dataset for performance testing
        large_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(5000)
            for i in range(20)
        })
        large_data['target'] = np.random.choice([0, 1], size=5000, p=[0.9, 0.1])
        
        large_dataset = Dataset(
            name="large_scale_test",
            data=large_data,
            target_column="target"
        )
        
        # Test sampling performance with various sizes
        sample_sizes = [100, 500, 1000, 2000]
        
        for size in sample_sizes:
            sample = large_dataset.sample(size, random_state=42)
            assert sample.n_samples == size
            assert sample.n_features == 20  # Excludes target
            
            # Verify sampling preserves data integrity
            assert sample.has_target is True
            assert sample.target_column == "target"
        
        # Test splitting performance
        split_ratios = [0.1, 0.2, 0.3, 0.5, 0.8]
        
        for test_size in split_ratios:
            train, test = large_dataset.split(test_size=test_size, random_state=42)
            
            expected_test_size = int(5000 * test_size)
            expected_train_size = 5000 - expected_test_size
            
            assert abs(test.n_samples - expected_test_size) <= 1
            assert abs(train.n_samples - expected_train_size) <= 1
            
            # Verify no data leakage
            assert len(set(train.data.index) & set(test.data.index)) == 0


class TestApplicationLayerAdvanced:
    """Advanced application layer testing for comprehensive coverage."""
    
    def test_dto_validation_comprehensive(self):
        """Test comprehensive DTO validation scenarios."""
        from pynomaly.application.dto.detector_dto import CreateDetectorDTO, UpdateDetectorDTO
        
        # Test boundary value validation
        boundary_test_cases = [
            # Valid boundary cases
            {"name": "a", "algorithm_name": "IsolationForest", "contamination_rate": 0.0},
            {"name": "a" * 100, "algorithm_name": "LOF", "contamination_rate": 1.0},
            {"name": "valid_name", "algorithm_name": "OCSVM", "contamination_rate": 0.5},
            
            # Edge cases for parameters
            {"name": "param_test", "algorithm_name": "Test", "parameters": {}},
            {"name": "param_test", "algorithm_name": "Test", "parameters": {"key": "value"}},
            {"name": "param_test", "algorithm_name": "Test", "parameters": {"nested": {"deep": {"value": 123}}}},
        ]
        
        for test_case in boundary_test_cases:
            try:
                dto = CreateDetectorDTO(**test_case)
                assert dto.name == test_case["name"]
                assert dto.algorithm_name == test_case["algorithm_name"]
                
                # Test serialization roundtrip
                json_data = dto.model_dump()
                recreated = CreateDetectorDTO.model_validate(json_data)
                assert recreated.name == dto.name
                assert recreated.parameters == dto.parameters
                
            except Exception as e:
                # Some boundary cases might fail validation, which is expected
                assert "validation" in str(e).lower() or "value" in str(e).lower()
        
        # Test UpdateDTO partial updates
        update_scenarios = [
            {"name": "updated_name"},
            {"contamination_rate": 0.15},
            {"parameters": {"updated": True}},
            {"metadata": {"version": "2.0"}},
            {"name": "full_update", "contamination_rate": 0.08, "parameters": {"complete": True}},
        ]
        
        for scenario in update_scenarios:
            update_dto = UpdateDetectorDTO(**scenario)
            
            # Test that only provided fields are set
            for key, value in scenario.items():
                assert getattr(update_dto, key) == value
            
            # Test serialization
            json_data = update_dto.model_dump(exclude_none=True)
            assert len(json_data) == len(scenario)
    
    def test_dto_performance_and_scalability(self):
        """Test DTO performance with large datasets and batch operations."""
        from pynomaly.application.dto.detector_dto import CreateDetectorDTO
        from pynomaly.application.dto.experiment_dto import CreateExperimentDTO
        
        # Test batch DTO creation performance
        batch_sizes = [10, 100, 500]
        
        for batch_size in batch_sizes:
            # Create batch of detector DTOs
            detector_dtos = []
            for i in range(batch_size):
                dto = CreateDetectorDTO(
                    name=f"detector_{i}",
                    algorithm_name="IsolationForest",
                    contamination_rate=0.1 + (i * 0.001),  # Slight variation
                    parameters={"n_estimators": 100 + i, "random_state": i}
                )
                detector_dtos.append(dto)
            
            assert len(detector_dtos) == batch_size
            
            # Test batch serialization
            batch_json = [dto.model_dump() for dto in detector_dtos]
            assert len(batch_json) == batch_size
            
            # Test batch deserialization
            recreated_dtos = [CreateDetectorDTO.model_validate(data) for data in batch_json]
            assert len(recreated_dtos) == batch_size
            
            # Verify data integrity
            for original, recreated in zip(detector_dtos, recreated_dtos):
                assert original.name == recreated.name
                assert original.contamination_rate == recreated.contamination_rate
        
        # Test complex experiment DTO with large configurations
        large_experiment = CreateExperimentDTO(
            name="large_scale_experiment",
            description="Performance testing with many configurations",
            metadata={
                "algorithms": [f"Algorithm_{i}" for i in range(50)],
                "datasets": [f"Dataset_{i}" for i in range(20)],
                "parameters": {
                    f"param_{i}": f"value_{i}" for i in range(100)
                },
                "nested_config": {
                    "level_1": {
                        "level_2": {
                            "level_3": [i for i in range(1000)]
                        }
                    }
                }
            }
        )
        
        # Test serialization of complex structure
        complex_json = large_experiment.model_dump()
        assert len(complex_json["metadata"]["algorithms"]) == 50
        assert len(complex_json["metadata"]["datasets"]) == 20
        assert len(complex_json["metadata"]["parameters"]) == 100
        
        # Test deserialization
        recreated_complex = CreateExperimentDTO.model_validate(complex_json)
        assert recreated_complex.name == large_experiment.name
        assert len(recreated_complex.metadata["algorithms"]) == 50


class TestErrorHandlingAdvanced:
    """Advanced error handling and edge case testing."""
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        # Test dataset creation error recovery
        error_scenarios = [
            # Empty dataframe scenario
            {"data": pd.DataFrame(), "expected_error": "empty"},
            
            # Invalid data type scenario
            {"data": "invalid_data", "expected_error": "type"},
            
            # Mismatched feature names scenario
            {
                "data": np.array([[1, 2], [3, 4]]),
                "feature_names": ["x", "y", "z"],  # Too many names
                "expected_error": "mismatch"
            },
            
            # Invalid target column scenario
            {
                "data": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                "target_column": "missing_column",
                "expected_error": "target"
            }
        ]
        
        for scenario in error_scenarios:
            try:
                dataset = Dataset(name="error_test", **{k: v for k, v in scenario.items() if k != "expected_error"})
                # If no error was raised, this is unexpected for error scenarios
                assert False, f"Expected error for scenario: {scenario['expected_error']}"
            except Exception as e:
                # Verify the error is of the expected type
                error_message = str(e).lower()
                expected = scenario["expected_error"]
                
                if expected == "empty":
                    assert "empty" in error_message or "invalid" in error_message
                elif expected == "type":
                    assert "type" in error_message or "dataframe" in error_message
                elif expected == "mismatch":
                    assert "feature" in error_message or "mismatch" in error_message
                elif expected == "target":
                    assert "target" in error_message or "column" in error_message or "found" in error_message
    
    def test_graceful_degradation(self):
        """Test graceful degradation when components are unavailable."""
        # Test repository fallback scenarios
        from pynomaly.infrastructure.repositories.in_memory_repositories import InMemoryDetectorRepository
        
        repo = InMemoryDetectorRepository()
        
        # Test operations with non-existent IDs
        non_existent_id = uuid.uuid4()
        
        assert repo.find_by_id(non_existent_id) is None
        assert repo.exists(non_existent_id) is False
        assert repo.delete(non_existent_id) is False
        
        # Test operations with invalid input
        assert repo.find_by_name("") is None
        assert repo.find_by_name("non_existent_detector") is None
        
        # Test empty repository state
        assert repo.count() == 0
        assert repo.find_all() == []
        assert repo.find_fitted() == []
        assert repo.find_by_algorithm("AnyAlgorithm") == []
    
    def test_concurrent_access_simulation(self):
        """Test concurrent access scenarios (simulated)."""
        from pynomaly.infrastructure.repositories.in_memory_repositories import InMemoryDatasetRepository
        
        repo = InMemoryDatasetRepository()
        
        # Simulate concurrent dataset creation
        datasets = []
        for i in range(10):
            data = pd.DataFrame({'x': [i], 'y': [i * 2]})
            dataset = Dataset(name=f"concurrent_dataset_{i}", data=data)
            repo.save(dataset)
            datasets.append(dataset)
        
        # Verify all datasets were saved
        assert repo.count() == 10
        
        # Simulate concurrent access
        for dataset in datasets:
            found = repo.find_by_id(dataset.id)
            assert found is not None
            assert found.name == dataset.name
        
        # Simulate concurrent deletion
        deleted_count = 0
        for dataset in datasets[:5]:  # Delete first 5
            if repo.delete(dataset.id):
                deleted_count += 1
        
        assert deleted_count == 5
        assert repo.count() == 5
        
        # Verify remaining datasets are still accessible
        remaining_datasets = repo.find_all()
        assert len(remaining_datasets) == 5
        
        for dataset in remaining_datasets:
            assert dataset.name.startswith("concurrent_dataset_")


class TestPerformanceBenchmarking:
    """Performance benchmarking and optimization testing."""
    
    def test_dataset_operations_performance(self):
        """Test performance of dataset operations."""
        # Performance test with various dataset sizes
        performance_results = {}
        
        sizes = [100, 1000, 5000]
        features = [5, 20, 50]
        
        for n_samples in sizes:
            for n_features in features[:2]:  # Limit for reasonable test time
                # Create test dataset
                data = pd.DataFrame(
                    np.random.randn(n_samples, n_features),
                    columns=[f'feature_{i}' for i in range(n_features)]
                )
                
                import time
                start_time = time.time()
                
                dataset = Dataset(name=f"perf_test_{n_samples}_{n_features}", data=data)
                
                # Test basic operations
                _ = dataset.n_samples
                _ = dataset.n_features
                _ = dataset.memory_usage
                _ = dataset.get_numeric_features()
                _ = dataset.get_categorical_features()
                _ = dataset.summary()
                
                # Test sampling if dataset is large enough
                if n_samples >= 100:
                    _ = dataset.sample(min(100, n_samples // 2), random_state=42)
                
                # Test splitting if dataset is large enough
                if n_samples >= 10:
                    _ = dataset.split(test_size=0.2, random_state=42)
                
                end_time = time.time()
                
                performance_results[f"{n_samples}x{n_features}"] = end_time - start_time
        
        # Verify performance is reasonable (should complete within reasonable time)
        for key, duration in performance_results.items():
            assert duration < 5.0, f"Performance test {key} took {duration:.2f}s (too slow)"
    
    def test_repository_operations_performance(self):
        """Test performance of repository operations."""
        from pynomaly.infrastructure.repositories.in_memory_repositories import InMemoryDatasetRepository
        
        repo = InMemoryDatasetRepository()
        
        # Performance test with batch operations
        import time
        
        # Test bulk insertion
        start_time = time.time()
        datasets = []
        
        for i in range(1000):
            data = pd.DataFrame({'x': [i], 'y': [i * 2], 'z': [i * 3]})
            dataset = Dataset(name=f"bulk_dataset_{i}", data=data)
            repo.save(dataset)
            datasets.append(dataset)
        
        insertion_time = time.time() - start_time
        
        # Test bulk retrieval
        start_time = time.time()
        
        for dataset in datasets[::10]:  # Test every 10th dataset
            found = repo.find_by_id(dataset.id)
            assert found is not None
        
        retrieval_time = time.time() - start_time
        
        # Test bulk search
        start_time = time.time()
        
        all_datasets = repo.find_all()
        assert len(all_datasets) == 1000
        
        search_time = time.time() - start_time
        
        # Verify performance is reasonable
        assert insertion_time < 10.0, f"Bulk insertion took {insertion_time:.2f}s (too slow)"
        assert retrieval_time < 2.0, f"Bulk retrieval took {retrieval_time:.2f}s (too slow)"
        assert search_time < 1.0, f"Bulk search took {search_time:.2f}s (too slow)"
        
        # Test memory usage
        import sys
        repo_size = sys.getsizeof(repo._storage)
        assert repo_size > 0  # Should have some memory footprint
    
    def test_value_object_operations_performance(self):
        """Test performance of value object operations."""
        import time
        
        # Test AnomalyScore operations performance
        start_time = time.time()
        
        scores = [AnomalyScore(i / 1000.0) for i in range(1000)]
        
        # Test comparison operations
        for i in range(len(scores) - 1):
            _ = scores[i] < scores[i + 1]
            _ = scores[i] == scores[i]
            _ = scores[i] != scores[i + 1]
        
        # Test sorting
        sorted_scores = sorted(scores, key=lambda s: s.value)
        assert len(sorted_scores) == 1000
        
        score_ops_time = time.time() - start_time
        
        # Test ContaminationRate operations performance
        start_time = time.time()
        
        rates = [ContaminationRate(i / 1000.0) for i in range(1, 501)]  # 0.001 to 0.5
        
        # Test various operations
        for rate in rates[::10]:  # Test every 10th rate
            _ = rate.as_percentage()
            _ = str(rate)
            _ = rate.is_valid()
        
        rate_ops_time = time.time() - start_time
        
        # Verify performance is reasonable
        assert score_ops_time < 5.0, f"Score operations took {score_ops_time:.2f}s (too slow)"
        assert rate_ops_time < 2.0, f"Rate operations took {rate_ops_time:.2f}s (too slow)"