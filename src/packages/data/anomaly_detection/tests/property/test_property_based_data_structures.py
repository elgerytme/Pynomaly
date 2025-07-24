"""Property-based tests for data structures and entities."""

import pytest
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.extra.numpy import arrays, array_shapes
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Optional
from unittest.mock import Mock

from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.domain.entities.model import Model as ModelEntity
from anomaly_detection.domain.value_objects.algorithm_config import AlgorithmConfig
from anomaly_detection.domain.value_objects.detection_metrics import DetectionMetrics


@pytest.mark.property
class TestDetectionResultProperties:
    """Property-based tests for DetectionResult entity."""
    
    @given(
        anomalies=st.lists(st.sampled_from([0, 1]), min_size=1, max_size=1000),
        scores=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=1000),
        algorithm=st.sampled_from(["isolation_forest", "one_class_svm", "local_outlier_factor"]),
        contamination=st.floats(min_value=0.01, max_value=0.5),
        execution_time=st.floats(min_value=0.001, max_value=300.0)
    )
    @settings(max_examples=100, deadline=5000)
    def test_detection_result_creation_properties(self, anomalies, scores, algorithm, contamination, execution_time):
        """Test properties of DetectionResult creation."""
        assume(len(anomalies) == len(scores))  # Must have same length
        
        result = DetectionResult(
            anomalies=anomalies,
            scores=scores,
            algorithm=algorithm,
            contamination=contamination,
            execution_time=execution_time
        )
        
        # Property 1: All fields should be preserved
        assert result.anomalies == anomalies
        assert result.scores == scores
        assert result.algorithm == algorithm
        assert result.contamination == contamination
        assert result.execution_time == execution_time
        
        # Property 2: Length consistency
        assert len(result.anomalies) == len(result.scores)
        
        # Property 3: Anomaly count should match binary sum
        expected_anomaly_count = sum(anomalies)
        assert result.get_anomaly_count() == expected_anomaly_count
        
        # Property 4: Anomaly rate should be correct
        expected_rate = expected_anomaly_count / len(anomalies)
        assert abs(result.get_anomaly_rate() - expected_rate) < 1e-10
    
    @given(
        anomalies=st.lists(st.sampled_from([0, 1]), min_size=5, max_size=100),
        scores=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=5, max_size=100)
    )
    @settings(max_examples=50, deadline=5000)
    def test_detection_result_invariants(self, anomalies, scores):
        """Test invariants of DetectionResult."""
        assume(len(anomalies) == len(scores))
        
        result = DetectionResult(
            anomalies=anomalies,
            scores=scores,
            algorithm="isolation_forest",
            contamination=0.1,
            execution_time=1.0
        )
        
        # Invariant 1: Anomaly count should never exceed total samples
        assert result.get_anomaly_count() <= len(anomalies)
        
        # Invariant 2: Anomaly rate should be between 0 and 1
        rate = result.get_anomaly_rate()
        assert 0.0 <= rate <= 1.0
        
        # Invariant 3: If no anomalies, rate should be 0
        if result.get_anomaly_count() == 0:
            assert rate == 0.0
        
        # Invariant 4: If all anomalies, rate should be 1
        if result.get_anomaly_count() == len(anomalies):
            assert rate == 1.0
    
    @given(
        data_size=st.integers(min_value=1, max_value=500),
        anomaly_probability=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50, deadline=5000)
    def test_detection_result_score_properties(self, data_size, anomaly_probability):
        """Test properties related to anomaly scores."""
        # Generate anomalies based on probability
        anomalies = [1 if np.random.random() < anomaly_probability else 0 for _ in range(data_size)]
        
        # Generate scores - anomalies should generally have higher scores
        scores = []
        for anomaly in anomalies:
            if anomaly == 1:
                score = np.random.uniform(0.5, 1.0)  # Higher scores for anomalies
            else:
                score = np.random.uniform(0.0, 0.6)  # Lower scores for normal points
            scores.append(score)
        
        result = DetectionResult(
            anomalies=anomalies,
            scores=scores,
            algorithm="isolation_forest",
            contamination=0.1,
            execution_time=1.0
        )
        
        # Property: Anomalous points should have higher average scores
        if result.get_anomaly_count() > 0 and result.get_anomaly_count() < len(anomalies):
            anomaly_scores = [scores[i] for i, a in enumerate(anomalies) if a == 1]
            normal_scores = [scores[i] for i, a in enumerate(anomalies) if a == 0]
            
            if len(anomaly_scores) > 0 and len(normal_scores) > 0:
                avg_anomaly_score = np.mean(anomaly_scores)
                avg_normal_score = np.mean(normal_scores)
                
                # With our generation method, this should generally hold
                assert avg_anomaly_score >= avg_normal_score - 0.1  # Allow some tolerance
    
    @given(
        anomalies1=st.lists(st.sampled_from([0, 1]), min_size=10, max_size=50),
        anomalies2=st.lists(st.sampled_from([0, 1]), min_size=10, max_size=50),
        scores1=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=10, max_size=50),
        scores2=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=10, max_size=50)
    )
    @settings(max_examples=30, deadline=8000)
    def test_detection_result_comparison_properties(self, anomalies1, anomalies2, scores1, scores2):
        """Test comparison properties between DetectionResults."""
        assume(len(anomalies1) == len(scores1))
        assume(len(anomalies2) == len(scores2))
        
        result1 = DetectionResult(
            anomalies=anomalies1, scores=scores1, algorithm="isolation_forest",
            contamination=0.1, execution_time=1.0
        )
        result2 = DetectionResult(
            anomalies=anomalies2, scores=scores2, algorithm="one_class_svm",
            contamination=0.1, execution_time=2.0
        )
        
        # Property 1: Results with same data should be equal
        result1_copy = DetectionResult(
            anomalies=anomalies1, scores=scores1, algorithm="isolation_forest",
            contamination=0.1, execution_time=1.0
        )
        assert result1 == result1_copy
        
        # Property 2: Results with different data should not be equal
        if anomalies1 != anomalies2 or scores1 != scores2:
            assert result1 != result2
        
        # Property 3: Hash consistency
        assert hash(result1) == hash(result1_copy)
        if result1 != result2:
            # Hash might be the same due to collisions, but usually different
            pass  # Can't guarantee hash inequality


@pytest.mark.property
class TestModelEntityProperties:
    """Property-based tests for ModelEntity."""
    
    @given(
        model_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        name=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        algorithm=st.sampled_from(["isolation_forest", "one_class_svm", "local_outlier_factor"]),
        version=st.integers(min_value=1, max_value=1000),
        training_data_size=st.integers(min_value=1, max_value=100000),
        feature_count=st.integers(min_value=1, max_value=1000),
        contamination=st.floats(min_value=0.01, max_value=0.5),
        performance_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100, deadline=5000)
    def test_model_entity_creation_properties(self, model_id, name, algorithm, version, 
                                            training_data_size, feature_count, 
                                            contamination, performance_score):
        """Test properties of ModelEntity creation."""
        created_at = datetime.now()
        
        model = ModelEntity(
            model_id=model_id,
            name=name,
            algorithm=algorithm,
            version=version,
            created_at=created_at,
            training_data_size=training_data_size,
            feature_count=feature_count,
            contamination=contamination,
            performance_score=performance_score
        )
        
        # Property 1: All fields should be preserved
        assert model.model_id == model_id
        assert model.name == name
        assert model.algorithm == algorithm
        assert model.version == version
        assert model.created_at == created_at
        assert model.training_data_size == training_data_size
        assert model.feature_count == feature_count
        assert model.contamination == contamination
        assert model.performance_score == performance_score
        
        # Property 2: Derived properties should be consistent
        assert model.is_trained() == True  # Model with training data is trained
        assert model.get_age_days() >= 0  # Age should be non-negative
        
        # Property 3: String representation should contain key information
        str_repr = str(model)
        assert model_id in str_repr
        assert algorithm in str_repr
    
    @given(
        performance_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=50, deadline=3000)
    def test_model_performance_threshold_properties(self, performance_score, threshold):
        """Test model performance threshold properties."""
        model = ModelEntity(
            model_id="test_model",
            name="Test Model",
            algorithm="isolation_forest",
            version=1,
            created_at=datetime.now(),
            training_data_size=1000,
            feature_count=10,
            contamination=0.1,
            performance_score=performance_score
        )
        
        # Property: Performance comparison should be consistent
        meets_threshold = model.meets_performance_threshold(threshold)
        assert meets_threshold == (performance_score >= threshold)
        
        # Property: Perfect performance should meet any reasonable threshold
        if performance_score == 1.0:
            assert model.meets_performance_threshold(0.9) == True
            assert model.meets_performance_threshold(1.0) == True
        
        # Property: Zero performance should not meet positive thresholds
        if performance_score == 0.0:
            assert model.meets_performance_threshold(0.1) == False
    
    @given(
        days_ago=st.integers(min_value=0, max_value=365)
    )
    @settings(max_examples=30, deadline=3000)
    def test_model_age_properties(self, days_ago):
        """Test model age calculation properties."""
        created_at = datetime.now() - timedelta(days=days_ago)
        
        model = ModelEntity(
            model_id="test_model",
            name="Test Model",
            algorithm="isolation_forest",
            version=1,
            created_at=created_at,
            training_data_size=1000,
            feature_count=10,
            contamination=0.1,
            performance_score=0.8
        )
        
        # Property: Age should be approximately correct
        calculated_age = model.get_age_days()
        assert abs(calculated_age - days_ago) <= 1  # Allow 1 day tolerance for timing
        
        # Property: Newer models should have smaller age
        if days_ago == 0:
            assert calculated_age < 1  # Should be less than 1 day old
    
    @given(
        version1=st.integers(min_value=1, max_value=100),
        version2=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50, deadline=3000)
    def test_model_version_comparison_properties(self, version1, version2):
        """Test model version comparison properties."""
        model1 = ModelEntity(
            model_id="test_model_1", name="Model 1", algorithm="isolation_forest",
            version=version1, created_at=datetime.now(), training_data_size=1000,
            feature_count=10, contamination=0.1, performance_score=0.8
        )
        
        model2 = ModelEntity(
            model_id="test_model_2", name="Model 2", algorithm="isolation_forest",
            version=version2, created_at=datetime.now(), training_data_size=1000,
            feature_count=10, contamination=0.1, performance_score=0.8
        )
        
        # Property: Version comparison should be consistent with integer comparison
        assert (model1.is_newer_version_than(model2)) == (version1 > version2)
        assert (model2.is_newer_version_than(model1)) == (version2 > version1)
        
        # Property: Model should not be newer than itself
        model1_copy = ModelEntity(
            model_id="test_model_1", name="Model 1", algorithm="isolation_forest",
            version=version1, created_at=datetime.now(), training_data_size=1000,
            feature_count=10, contamination=0.1, performance_score=0.8
        )
        assert not model1.is_newer_version_than(model1_copy)


@pytest.mark.property
class TestAlgorithmConfigProperties:
    """Property-based tests for AlgorithmConfig value object."""
    
    @given(
        algorithm=st.sampled_from(["isolation_forest", "one_class_svm", "local_outlier_factor"]),
        contamination=st.floats(min_value=0.01, max_value=0.5),
        random_state=st.one_of(st.none(), st.integers(min_value=0, max_value=2**31-1)),
        n_estimators=st.one_of(st.none(), st.integers(min_value=10, max_value=1000)),
        max_samples=st.one_of(st.none(), st.floats(min_value=0.1, max_value=1.0))
    )
    @settings(max_examples=100, deadline=5000)
    def test_algorithm_config_creation_properties(self, algorithm, contamination, 
                                                 random_state, n_estimators, max_samples):
        """Test properties of AlgorithmConfig creation."""
        config = AlgorithmConfig(
            algorithm=algorithm,
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples
        )
        
        # Property 1: All fields should be preserved
        assert config.algorithm == algorithm
        assert config.contamination == contamination
        assert config.random_state == random_state
        assert config.n_estimators == n_estimators
        assert config.max_samples == max_samples
        
        # Property 2: Configuration should be valid
        assert config.is_valid()
        
        # Property 3: Dictionary representation should contain all non-None values
        config_dict = config.to_dict()
        assert config_dict["algorithm"] == algorithm
        assert config_dict["contamination"] == contamination
        
        if random_state is not None:
            assert config_dict["random_state"] == random_state
        if n_estimators is not None:
            assert config_dict["n_estimators"] == n_estimators
        if max_samples is not None:
            assert config_dict["max_samples"] == max_samples
    
    @given(
        contamination=st.one_of(
            st.floats(min_value=-1.0, max_value=0.0),  # Invalid: negative/zero
            st.floats(min_value=1.0, max_value=2.0),   # Invalid: >= 1
        )
    )
    @settings(max_examples=30, deadline=3000)
    def test_invalid_algorithm_config_properties(self, contamination):
        """Test properties of invalid AlgorithmConfig."""
        config = AlgorithmConfig(
            algorithm="isolation_forest",
            contamination=contamination
        )
        
        # Property: Invalid configurations should be detected
        assert not config.is_valid()
        
        # Property: Validation should identify the specific issue
        validation_errors = config.validate()
        assert len(validation_errors) > 0
        assert any("contamination" in error.lower() for error in validation_errors)
    
    @given(
        config1_params=st.fixed_dictionaries({
            "algorithm": st.sampled_from(["isolation_forest", "one_class_svm"]),
            "contamination": st.floats(min_value=0.05, max_value=0.3),
            "random_state": st.integers(min_value=0, max_value=1000)
        }),
        config2_params=st.fixed_dictionaries({
            "algorithm": st.sampled_from(["isolation_forest", "one_class_svm"]),
            "contamination": st.floats(min_value=0.05, max_value=0.3),
            "random_state": st.integers(min_value=0, max_value=1000)
        })
    )
    @settings(max_examples=50, deadline=5000)
    def test_algorithm_config_equality_properties(self, config1_params, config2_params):
        """Test equality properties of AlgorithmConfig."""
        config1 = AlgorithmConfig(**config1_params)
        config2 = AlgorithmConfig(**config2_params)
        
        # Property 1: Config should equal itself
        assert config1 == config1
        
        # Property 2: Configs with same parameters should be equal
        config1_copy = AlgorithmConfig(**config1_params)
        assert config1 == config1_copy
        
        # Property 3: Hash consistency
        assert hash(config1) == hash(config1_copy)
        
        # Property 4: Configs with different parameters should not be equal
        if config1_params != config2_params:
            assert config1 != config2


@pytest.mark.property
class TestDetectionMetricsProperties:
    """Property-based tests for DetectionMetrics value object."""
    
    @given(
        true_positives=st.integers(min_value=0, max_value=1000),
        false_positives=st.integers(min_value=0, max_value=1000),
        true_negatives=st.integers(min_value=0, max_value=1000),
        false_negatives=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=100, deadline=5000)
    def test_detection_metrics_creation_properties(self, true_positives, false_positives, 
                                                  true_negatives, false_negatives):
        """Test properties of DetectionMetrics creation."""
        assume(true_positives + false_negatives > 0)  # Need some actual positives
        assume(true_negatives + false_positives > 0)  # Need some actual negatives
        
        metrics = DetectionMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives
        )
        
        # Property 1: All fields should be preserved
        assert metrics.true_positives == true_positives
        assert metrics.false_positives == false_positives
        assert metrics.true_negatives == true_negatives
        assert metrics.false_negatives == false_negatives
        
        # Property 2: Total counts should be sum of components
        total_predicted_positive = true_positives + false_positives
        total_predicted_negative = true_negatives + false_negatives
        total_actual_positive = true_positives + false_negatives
        total_actual_negative = true_negatives + false_positives
        
        assert metrics.get_total_samples() == (
            true_positives + false_positives + true_negatives + false_negatives
        )
        
        # Property 3: Derived metrics should be in valid ranges
        precision = metrics.get_precision()
        recall = metrics.get_recall()
        f1_score = metrics.get_f1_score()
        accuracy = metrics.get_accuracy()
        
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1_score <= 1.0
        assert 0.0 <= accuracy <= 1.0
    
    @given(
        tp=st.integers(min_value=1, max_value=100),
        fp=st.integers(min_value=0, max_value=100),
        tn=st.integers(min_value=1, max_value=100),
        fn=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=50, deadline=5000)
    def test_perfect_classifier_properties(self, tp, fp, tn, fn):
        """Test properties when creating perfect classifier metrics."""
        # Perfect classifier: no false positives or false negatives
        perfect_metrics = DetectionMetrics(
            true_positives=tp,
            false_positives=0,
            true_negatives=tn,
            false_negatives=0
        )
        
        # Property: Perfect classifier should have perfect scores
        assert perfect_metrics.get_precision() == 1.0
        assert perfect_metrics.get_recall() == 1.0
        assert perfect_metrics.get_f1_score() == 1.0
        assert perfect_metrics.get_accuracy() == 1.0
    
    @given(
        tp=st.integers(min_value=0, max_value=50),
        fp=st.integers(min_value=1, max_value=50),
        tn=st.integers(min_value=0, max_value=50),
        fn=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=30, deadline=5000)
    def test_worst_classifier_properties(self, tp, fp, tn, fn):
        """Test properties of worst-case classifier metrics."""
        # Worst classifier: only false positives and false negatives
        worst_metrics = DetectionMetrics(
            true_positives=0,
            false_positives=fp,
            true_negatives=0,
            false_negatives=fn
        )
        
        # Property: Worst classifier should have zero scores
        assert worst_metrics.get_precision() == 0.0
        assert worst_metrics.get_recall() == 0.0
        assert worst_metrics.get_f1_score() == 0.0
        assert worst_metrics.get_accuracy() == 0.0
    
    @given(
        tp1=st.integers(min_value=1, max_value=50),
        fp1=st.integers(min_value=0, max_value=20),
        tp2=st.integers(min_value=1, max_value=50),
        fp2=st.integers(min_value=0, max_value=20),
        tn=st.integers(min_value=10, max_value=100),
        fn=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=30, deadline=8000)
    def test_precision_comparison_properties(self, tp1, fp1, tp2, fp2, tn, fn):
        """Test precision comparison properties."""
        assume(tp1 + fp1 > 0)  # Need predictions
        assume(tp2 + fp2 > 0)  # Need predictions
        
        metrics1 = DetectionMetrics(tp1, fp1, tn, fn)
        metrics2 = DetectionMetrics(tp2, fp2, tn, fn)
        
        precision1 = metrics1.get_precision()
        precision2 = metrics2.get_precision()
        
        # Property: More true positives with same false positives should improve precision
        if fp1 == fp2 and tp1 > tp2:
            assert precision1 > precision2
        
        # Property: Fewer false positives with same true positives should improve precision
        if tp1 == tp2 and fp1 < fp2:
            assert precision1 > precision2


@pytest.mark.property
class TestDataStructureInvariants:
    """Property-based tests for data structure invariants."""
    
    @given(
        data=st.lists(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=20
            ),
            min_size=1, max_size=100
        )
    )
    @settings(max_examples=50, deadline=8000)
    def test_data_consistency_invariants(self, data):
        """Test data consistency invariants across structures."""
        assume(len(set(map(len, data))) == 1)  # All rows same length
        
        n_samples = len(data)
        n_features = len(data[0]) if data else 0
        
        # Create detection result with consistent dimensions
        anomalies = [0] * n_samples  # All normal
        scores = [0.1] * n_samples   # Low scores
        
        result = DetectionResult(
            anomalies=anomalies,
            scores=scores,
            algorithm="isolation_forest",
            contamination=0.1,
            execution_time=1.0
        )
        
        # Invariant 1: Result dimensions should match data dimensions
        assert len(result.anomalies) == n_samples
        assert len(result.scores) == n_samples
        
        # Invariant 2: Anomaly count should be consistent
        assert result.get_anomaly_count() == sum(anomalies)
        
        # Create model entity with consistent metadata
        model = ModelEntity(
            model_id="test_model",
            name="Test Model",
            algorithm="isolation_forest",
            version=1,
            created_at=datetime.now(),
            training_data_size=n_samples,
            feature_count=n_features,
            contamination=0.1,
            performance_score=0.8
        )
        
        # Invariant 3: Model metadata should be consistent with data
        assert model.training_data_size == n_samples
        assert model.feature_count == n_features


if __name__ == "__main__":
    # Run specific property-based tests
    pytest.main([
        __file__ + "::TestDetectionResultProperties::test_detection_result_creation_properties",
        "-v", "-s", "--tb=short"
    ])