"""Comprehensive tests for all domain value objects.

This module provides comprehensive test coverage for all domain value objects
including semantic versioning, model storage info, performance metrics,
and other value objects throughout the domain layer.
"""

from datetime import datetime

import pytest
from pynomaly.domain.exceptions import InvalidValueError
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
from pynomaly.domain.value_objects.contamination_rate import ContaminationRate
from pynomaly.domain.value_objects.model_storage_info import ModelStorageInfo
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics
from pynomaly.domain.value_objects.semantic_version import SemanticVersion
from pynomaly.domain.value_objects.threshold_config import ThresholdConfig


class TestSemanticVersion:
    """Comprehensive tests for SemanticVersion value object."""

    def test_semantic_version_creation(self):
        """Test creating semantic versions."""
        # From string
        version1 = SemanticVersion("1.2.3")
        assert version1.major == 1
        assert version1.minor == 2
        assert version1.patch == 3
        assert version1.prerelease is None
        assert version1.build is None

        # From components
        version2 = SemanticVersion(major=2, minor=0, patch=1)
        assert version2.major == 2
        assert version2.minor == 0
        assert version2.patch == 1

        # With prerelease and build
        version3 = SemanticVersion("1.0.0-alpha.1+build.123")
        assert version3.major == 1
        assert version3.minor == 0
        assert version3.patch == 0
        assert version3.prerelease == "alpha.1"
        assert version3.build == "build.123"

    def test_semantic_version_comparison(self):
        """Test semantic version comparison."""
        v1_0_0 = SemanticVersion("1.0.0")
        v1_0_1 = SemanticVersion("1.0.1")
        v1_1_0 = SemanticVersion("1.1.0")
        v2_0_0 = SemanticVersion("2.0.0")

        # Basic comparisons
        assert v1_0_0 < v1_0_1
        assert v1_0_1 < v1_1_0
        assert v1_1_0 < v2_0_0

        assert v2_0_0 > v1_1_0
        assert v1_1_0 > v1_0_1
        assert v1_0_1 > v1_0_0

        assert v1_0_0 == SemanticVersion("1.0.0")
        assert v1_0_0 != v1_0_1

        # Prerelease comparisons
        v1_0_0_alpha = SemanticVersion("1.0.0-alpha")
        v1_0_0_beta = SemanticVersion("1.0.0-beta")
        v1_0_0_rc = SemanticVersion("1.0.0-rc")

        assert v1_0_0_alpha < v1_0_0_beta
        assert v1_0_0_beta < v1_0_0_rc
        assert v1_0_0_rc < v1_0_0

    def test_semantic_version_operations(self):
        """Test semantic version operations."""
        version = SemanticVersion("1.2.3")

        # Increment operations
        major_bump = version.bump_major()
        assert major_bump == SemanticVersion("2.0.0")

        minor_bump = version.bump_minor()
        assert minor_bump == SemanticVersion("1.3.0")

        patch_bump = version.bump_patch()
        assert patch_bump == SemanticVersion("1.2.4")

        # Original version unchanged
        assert version == SemanticVersion("1.2.3")

    def test_semantic_version_validation(self):
        """Test semantic version validation."""
        # Valid versions
        valid_versions = [
            "0.0.0",
            "1.0.0",
            "1.2.3",
            "10.20.30",
            "1.0.0-alpha",
            "1.0.0-beta.1",
            "1.0.0-rc.1+build.1",
        ]

        for version_str in valid_versions:
            version = SemanticVersion(version_str)
            assert version.is_valid()

        # Invalid versions
        invalid_versions = [
            "1",
            "1.2",
            "1.2.3.4",
            "a.b.c",
            "-1.0.0",
            "1.-2.0",
            "1.0.-3",
        ]

        for version_str in invalid_versions:
            with pytest.raises(InvalidValueError):
                SemanticVersion(version_str)

    def test_semantic_version_compatibility(self):
        """Test semantic version compatibility checking."""
        v1_2_3 = SemanticVersion("1.2.3")

        # Compatible versions (same major)
        assert v1_2_3.is_compatible_with(SemanticVersion("1.3.0"))
        assert v1_2_3.is_compatible_with(SemanticVersion("1.2.4"))

        # Incompatible versions (different major)
        assert not v1_2_3.is_compatible_with(SemanticVersion("2.0.0"))
        assert not v1_2_3.is_compatible_with(SemanticVersion("0.9.0"))

        # Version ranges
        assert v1_2_3.satisfies_range(">=1.0.0")
        assert v1_2_3.satisfies_range("^1.2.0")  # Compatible with 1.2.x
        assert not v1_2_3.satisfies_range("~1.3.0")  # Not compatible with 1.3.x


class TestModelStorageInfo:
    """Comprehensive tests for ModelStorageInfo value object."""

    def test_model_storage_info_creation(self):
        """Test creating model storage info."""
        storage_info = ModelStorageInfo(
            location="s3://ml-models/anomaly-detector-v1.pkl",
            size_bytes=1024000,
            checksum="sha256:abc123def456",
            storage_type="s3",
            compression="gzip",
        )

        assert storage_info.location == "s3://ml-models/anomaly-detector-v1.pkl"
        assert storage_info.size_bytes == 1024000
        assert storage_info.checksum == "sha256:abc123def456"
        assert storage_info.storage_type == "s3"
        assert storage_info.compression == "gzip"

    def test_model_storage_info_validation(self):
        """Test model storage info validation."""
        # Valid storage info
        valid_info = ModelStorageInfo(
            location="/models/detector.pkl", size_bytes=1000, checksum="abc123"
        )
        assert valid_info.is_valid()

        # Invalid storage info
        with pytest.raises(InvalidValueError):
            ModelStorageInfo(
                location="",  # Empty location
                size_bytes=1000,
                checksum="abc123",
            )

        with pytest.raises(InvalidValueError):
            ModelStorageInfo(
                location="/models/detector.pkl",
                size_bytes=-1,  # Negative size
                checksum="abc123",
            )

        with pytest.raises(InvalidValueError):
            ModelStorageInfo(
                location="/models/detector.pkl",
                size_bytes=1000,
                checksum="",  # Empty checksum
            )

    def test_model_storage_info_size_formatting(self):
        """Test model storage info size formatting."""
        # Small file
        small_info = ModelStorageInfo(
            location="/models/small.pkl", size_bytes=1024, checksum="abc123"
        )
        assert small_info.get_formatted_size() == "1.0 KB"

        # Medium file
        medium_info = ModelStorageInfo(
            location="/models/medium.pkl",
            size_bytes=1048576,  # 1 MB
            checksum="def456",
        )
        assert medium_info.get_formatted_size() == "1.0 MB"

        # Large file
        large_info = ModelStorageInfo(
            location="/models/large.pkl",
            size_bytes=1073741824,  # 1 GB
            checksum="ghi789",
        )
        assert large_info.get_formatted_size() == "1.0 GB"

    def test_model_storage_info_checksum_verification(self):
        """Test checksum verification."""
        storage_info = ModelStorageInfo(
            location="/models/test.pkl", size_bytes=1000, checksum="sha256:a1b2c3d4e5f6"
        )

        # Mock file content for checksum verification
        mock_content = b"model_binary_data"

        # This would normally verify against actual file
        # For testing, we simulate verification
        is_valid = storage_info.verify_checksum(mock_content)
        # Implementation would calculate checksum and compare
        assert isinstance(is_valid, bool)

    def test_model_storage_info_metadata(self):
        """Test storage metadata management."""
        storage_info = ModelStorageInfo(
            location="/models/detector.pkl", size_bytes=1000, checksum="abc123"
        )

        # Add metadata
        storage_info.add_metadata("created_by", "ml_engineer_1")
        storage_info.add_metadata("model_type", "IsolationForest")
        storage_info.add_metadata("training_date", "2023-10-01")

        assert storage_info.get_metadata("created_by") == "ml_engineer_1"
        assert storage_info.get_metadata("model_type") == "IsolationForest"

        # Update metadata
        storage_info.update_metadata("created_by", "ml_engineer_2")
        assert storage_info.get_metadata("created_by") == "ml_engineer_2"

        # Get all metadata
        all_metadata = storage_info.get_all_metadata()
        assert len(all_metadata) == 3
        assert "model_type" in all_metadata


class TestPerformanceMetrics:
    """Comprehensive tests for PerformanceMetrics value object."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            roc_auc=0.94,
            pr_auc=0.89,
        )

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.92
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.90
        assert metrics.roc_auc == 0.94
        assert metrics.pr_auc == 0.89

    def test_performance_metrics_validation(self):
        """Test performance metrics validation."""
        # Valid metrics
        valid_metrics = PerformanceMetrics(
            accuracy=0.85, precision=0.80, recall=0.90, f1_score=0.85
        )
        assert valid_metrics.is_valid()

        # Invalid metrics - out of range
        with pytest.raises(InvalidValueError):
            PerformanceMetrics(
                accuracy=1.5,  # > 1.0
                precision=0.80,
                recall=0.90,
                f1_score=0.85,
            )

        with pytest.raises(InvalidValueError):
            PerformanceMetrics(
                accuracy=0.85,
                precision=-0.1,  # < 0.0
                recall=0.90,
                f1_score=0.85,
            )

    def test_performance_metrics_calculations(self):
        """Test performance metrics calculations."""
        # From confusion matrix
        true_positives = 85
        false_positives = 15
        true_negatives = 880
        false_negatives = 20

        metrics = PerformanceMetrics.from_confusion_matrix(
            tp=true_positives, fp=false_positives, tn=true_negatives, fn=false_negatives
        )

        expected_accuracy = (85 + 880) / (85 + 15 + 880 + 20)
        expected_precision = 85 / (85 + 15)
        expected_recall = 85 / (85 + 20)
        expected_f1 = (
            2
            * expected_precision
            * expected_recall
            / (expected_precision + expected_recall)
        )

        assert abs(metrics.accuracy - expected_accuracy) < 0.001
        assert abs(metrics.precision - expected_precision) < 0.001
        assert abs(metrics.recall - expected_recall) < 0.001
        assert abs(metrics.f1_score - expected_f1) < 0.001

    def test_performance_metrics_comparison(self):
        """Test performance metrics comparison."""
        metrics1 = PerformanceMetrics(
            accuracy=0.90, precision=0.85, recall=0.88, f1_score=0.86
        )

        metrics2 = PerformanceMetrics(
            accuracy=0.92, precision=0.87, recall=0.90, f1_score=0.88
        )

        # Compare overall performance
        comparison = metrics1.compare_with(metrics2)

        assert comparison["accuracy_diff"] < 0  # metrics1 < metrics2
        assert comparison["precision_diff"] < 0
        assert comparison["recall_diff"] < 0
        assert comparison["f1_score_diff"] < 0
        assert comparison["overall_better"] == "metrics2"

    def test_performance_metrics_aggregation(self):
        """Test aggregating multiple performance metrics."""
        metrics_list = [
            PerformanceMetrics(
                accuracy=0.90, precision=0.85, recall=0.88, f1_score=0.86
            ),
            PerformanceMetrics(
                accuracy=0.92, precision=0.87, recall=0.90, f1_score=0.88
            ),
            PerformanceMetrics(
                accuracy=0.88, precision=0.83, recall=0.86, f1_score=0.84
            ),
        ]

        # Average metrics
        avg_metrics = PerformanceMetrics.average(metrics_list)
        expected_accuracy = (0.90 + 0.92 + 0.88) / 3

        assert abs(avg_metrics.accuracy - expected_accuracy) < 0.001
        assert abs(avg_metrics.precision - (0.85 + 0.87 + 0.83) / 3) < 0.001

        # Weighted average
        weights = [0.5, 0.3, 0.2]
        weighted_avg = PerformanceMetrics.weighted_average(metrics_list, weights)
        expected_weighted_accuracy = 0.90 * 0.5 + 0.92 * 0.3 + 0.88 * 0.2

        assert abs(weighted_avg.accuracy - expected_weighted_accuracy) < 0.001

    def test_performance_metrics_confidence_intervals(self):
        """Test confidence intervals for performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.90, precision=0.85, recall=0.88, f1_score=0.86
        )

        # Calculate confidence intervals (bootstrap or analytical)
        confidence_intervals = metrics.calculate_confidence_intervals(
            n_samples=1000, confidence_level=0.95
        )

        assert "accuracy" in confidence_intervals
        assert "precision" in confidence_intervals
        assert "recall" in confidence_intervals
        assert "f1_score" in confidence_intervals

        # Verify interval structure
        acc_ci = confidence_intervals["accuracy"]
        assert "lower" in acc_ci
        assert "upper" in acc_ci
        assert acc_ci["lower"] <= 0.90 <= acc_ci["upper"]


class TestAnomalyScoreAdvanced:
    """Advanced comprehensive tests for AnomalyScore value object."""

    def test_anomaly_score_calibration(self):
        """Test anomaly score calibration."""
        raw_score = AnomalyScore(0.75)

        # Calibrate using Platt scaling
        calibrated_score = raw_score.calibrate(
            method="platt_scaling",
            calibration_data=[(0.8, 1), (0.6, 0), (0.9, 1), (0.3, 0)],
        )

        assert isinstance(calibrated_score, AnomalyScore)
        assert 0.0 <= calibrated_score.value <= 1.0

        # Calibrate using isotonic regression
        isotonic_score = raw_score.calibrate(
            method="isotonic_regression",
            calibration_data=[(0.8, 1), (0.6, 0), (0.9, 1), (0.3, 0)],
        )

        assert isinstance(isotonic_score, AnomalyScore)
        assert 0.0 <= isotonic_score.value <= 1.0

    def test_anomaly_score_uncertainty(self):
        """Test anomaly score with uncertainty estimation."""
        score_with_uncertainty = AnomalyScore(
            value=0.85, uncertainty=0.1, uncertainty_type="aleatoric"
        )

        assert score_with_uncertainty.value == 0.85
        assert score_with_uncertainty.uncertainty == 0.1
        assert score_with_uncertainty.uncertainty_type == "aleatoric"

        # Calculate confidence bounds
        confidence_bounds = score_with_uncertainty.get_confidence_bounds(
            confidence_level=0.95
        )

        assert "lower" in confidence_bounds
        assert "upper" in confidence_bounds
        assert confidence_bounds["lower"] <= 0.85 <= confidence_bounds["upper"]

    def test_anomaly_score_explanation(self):
        """Test anomaly score with explanation."""
        explained_score = AnomalyScore(
            value=0.92,
            feature_contributions={
                "temperature": 0.4,
                "pressure": 0.3,
                "humidity": 0.22,
            },
            explanation_method="shap",
        )

        assert explained_score.value == 0.92
        assert explained_score.feature_contributions["temperature"] == 0.4
        assert explained_score.explanation_method == "shap"

        # Get top contributing features
        top_features = explained_score.get_top_contributing_features(n=2)
        assert len(top_features) == 2
        assert top_features[0][0] == "temperature"  # Highest contribution

    def test_anomaly_score_temporal_context(self):
        """Test anomaly score with temporal context."""
        temporal_score = AnomalyScore(
            value=0.88,
            timestamp=datetime.now(),
            temporal_context={
                "trend": "increasing",
                "seasonal_adjustment": 0.05,
                "window_avg": 0.65,
            },
        )

        assert temporal_score.value == 0.88
        assert isinstance(temporal_score.timestamp, datetime)
        assert temporal_score.temporal_context["trend"] == "increasing"

        # Check if score is anomalous relative to temporal context
        is_temporal_anomaly = temporal_score.is_temporal_anomaly(
            baseline_score=0.65, threshold_multiplier=1.2
        )

        assert isinstance(is_temporal_anomaly, bool)


class TestContaminationRateAdvanced:
    """Advanced comprehensive tests for ContaminationRate value object."""

    def test_contamination_rate_adaptive(self):
        """Test adaptive contamination rate."""
        # Historical contamination rates
        historical_rates = [0.05, 0.06, 0.04, 0.07, 0.05]

        adaptive_rate = ContaminationRate.create_adaptive(
            historical_rates=historical_rates,
            adaptation_factor=0.1,
            trend_sensitivity=0.05,
        )

        assert isinstance(adaptive_rate, ContaminationRate)
        assert 0.0 <= adaptive_rate.value <= 0.5

        # Should be close to historical average
        historical_avg = sum(historical_rates) / len(historical_rates)
        assert abs(adaptive_rate.value - historical_avg) < 0.02

    def test_contamination_rate_confidence_interval(self):
        """Test contamination rate with confidence interval."""
        rate_with_ci = ContaminationRate(
            value=0.05,
            confidence_interval=ConfidenceInterval(
                lower=0.03, upper=0.07, confidence_level=0.95
            ),
        )

        assert rate_with_ci.value == 0.05
        assert rate_with_ci.confidence_interval.lower == 0.03
        assert rate_with_ci.confidence_interval.upper == 0.07

        # Verify rate is within confidence interval
        assert rate_with_ci.confidence_interval.contains(rate_with_ci.value)

    def test_contamination_rate_validation_advanced(self):
        """Test advanced contamination rate validation."""
        # Domain-specific validation
        financial_rate = ContaminationRate.create_for_domain(
            value=0.02,
            domain="financial_fraud",
            validation_rules={"max_rate": 0.1, "min_rate": 0.001},
        )

        assert financial_rate.value == 0.02
        assert financial_rate.domain == "financial_fraud"

        # Should reject rates outside domain limits
        with pytest.raises(InvalidValueError):
            ContaminationRate.create_for_domain(
                value=0.15,  # Too high for financial domain
                domain="financial_fraud",
                validation_rules={"max_rate": 0.1, "min_rate": 0.001},
            )

    def test_contamination_rate_optimization(self):
        """Test contamination rate optimization."""
        # Optimize based on performance metrics
        optimization_data = [
            {"rate": 0.03, "f1_score": 0.82},
            {"rate": 0.05, "f1_score": 0.87},
            {"rate": 0.07, "f1_score": 0.85},
            {"rate": 0.10, "f1_score": 0.79},
        ]

        optimal_rate = ContaminationRate.optimize(
            optimization_data=optimization_data,
            optimization_metric="f1_score",
            optimization_goal="maximize",
        )

        assert isinstance(optimal_rate, ContaminationRate)
        # Should select rate that maximizes F1 score (0.05)
        assert abs(optimal_rate.value - 0.05) < 0.001


class TestThresholdConfigAdvanced:
    """Advanced comprehensive tests for ThresholdConfig value object."""

    def test_threshold_config_multi_criteria(self):
        """Test threshold configuration with multiple criteria."""
        multi_criteria_config = ThresholdConfig(
            method="multi_criteria",
            criteria=[
                {"metric": "precision", "target": 0.9, "weight": 0.4},
                {"metric": "recall", "target": 0.85, "weight": 0.3},
                {"metric": "f1_score", "target": 0.87, "weight": 0.3},
            ],
            optimization_algorithm="genetic",
        )

        assert multi_criteria_config.method == "multi_criteria"
        assert len(multi_criteria_config.criteria) == 3
        assert multi_criteria_config.optimization_algorithm == "genetic"

        # Validate criteria weights sum to 1
        total_weight = sum(c["weight"] for c in multi_criteria_config.criteria)
        assert abs(total_weight - 1.0) < 0.001

    def test_threshold_config_adaptive(self):
        """Test adaptive threshold configuration."""
        adaptive_config = ThresholdConfig(
            method="adaptive",
            adaptation_strategy="online_learning",
            adaptation_rate=0.01,
            performance_window="24h",
            min_samples_for_adaptation=1000,
        )

        assert adaptive_config.method == "adaptive"
        assert adaptive_config.adaptation_strategy == "online_learning"
        assert adaptive_config.adaptation_rate == 0.01

        # Simulate threshold adaptation
        performance_feedback = [
            {"threshold": 0.8, "precision": 0.9, "recall": 0.75},
            {"threshold": 0.75, "precision": 0.85, "recall": 0.82},
            {"threshold": 0.7, "precision": 0.80, "recall": 0.88},
        ]

        adapted_threshold = adaptive_config.adapt_threshold(
            current_threshold=0.8,
            performance_feedback=performance_feedback,
            optimization_target="f1_score",
        )

        assert isinstance(adapted_threshold, float)
        assert 0.0 <= adapted_threshold <= 1.0

    def test_threshold_config_contextual(self):
        """Test contextual threshold configuration."""
        contextual_config = ThresholdConfig(
            method="contextual",
            context_features=["time_of_day", "day_of_week", "season"],
            context_thresholds={
                "business_hours": 0.8,
                "after_hours": 0.9,
                "weekend": 0.75,
                "holiday": 0.95,
            },
        )

        assert contextual_config.method == "contextual"
        assert "time_of_day" in contextual_config.context_features
        assert contextual_config.context_thresholds["business_hours"] == 0.8

        # Get threshold for specific context
        context = {
            "time_of_day": "business_hours",
            "day_of_week": "tuesday",
            "season": "winter",
        }

        contextual_threshold = contextual_config.get_threshold_for_context(context)
        assert contextual_threshold == 0.8  # business_hours threshold

    def test_threshold_config_validation_complex(self):
        """Test complex threshold configuration validation."""
        # Valid complex configuration
        complex_config = ThresholdConfig(
            method="ensemble",
            ensemble_methods=[
                {"method": "percentile", "percentile": 95, "weight": 0.4},
                {"method": "statistical", "method_type": "zscore", "weight": 0.3},
                {"method": "contamination", "rate": 0.05, "weight": 0.3},
            ],
            combination_strategy="weighted_average",
        )

        assert complex_config.method == "ensemble"
        assert len(complex_config.ensemble_methods) == 3

        # Validate ensemble weights
        total_weight = sum(m["weight"] for m in complex_config.ensemble_methods)
        assert abs(total_weight - 1.0) < 0.001

        # Invalid configuration - weights don't sum to 1
        with pytest.raises(InvalidValueError):
            ThresholdConfig(
                method="ensemble",
                ensemble_methods=[
                    {"method": "percentile", "percentile": 95, "weight": 0.6},
                    {
                        "method": "statistical",
                        "method_type": "zscore",
                        "weight": 0.6,
                    },  # Total > 1
                ],
            )

    def test_threshold_config_serialization(self):
        """Test threshold configuration serialization."""
        config = ThresholdConfig(
            method="percentile",
            value=95.0,
            auto_adjust=True,
            adjustment_frequency="daily",
        )

        # Serialize to dictionary
        config_dict = config.to_dict()
        assert config_dict["method"] == "percentile"
        assert config_dict["value"] == 95.0
        assert config_dict["auto_adjust"] is True

        # Deserialize from dictionary
        restored_config = ThresholdConfig.from_dict(config_dict)
        assert restored_config.method == config.method
        assert restored_config.value == config.value
        assert restored_config.auto_adjust == config.auto_adjust

        # Serialize to JSON
        config_json = config.to_json()
        assert isinstance(config_json, str)

        # Deserialize from JSON
        restored_from_json = ThresholdConfig.from_json(config_json)
        assert restored_from_json.method == config.method
