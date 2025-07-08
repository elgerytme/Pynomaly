"""Comprehensive unit and property-based tests for anomaly classification and service integration."""

import asyncio
import math
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from hypothesis import given, strategies as st

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.services.anomaly_classifiers import (
    DefaultSeverityClassifier,
    DefaultTypeClassifier,
    MLSeverityClassifier,
    BatchProcessingSeverityClassifier,
    DashboardTypeClassifier,
)
from pynomaly.infrastructure.config.container import create_container
from pynomaly.domain.exceptions import InvalidValueError

class TestEnumBackwardCompatibility(unittest.TestCase):
    def test_severity_levels_compatibility(self):
        """Test backward compatibility of enum severity levels."""
        severity_classifier = DefaultSeverityClassifier()
        # Ensure all existing levels work
        anomaly = Anomaly(score=AnomalyScore(0.85), data_point={}, detector_name="detector")
        self.assertIn(severity_classifier.classify_severity(anomaly), ["critical", "high", "medium", "low"])

@given(score_value=st.floats(min_value=0.0, max_value=1.0))
def test_property_based_classification(score_value):
    """Property-based test for severity classification thresholds."""
    severity_classifier = DefaultSeverityClassifier()
    anomaly = Anomaly(score=AnomalyScore(score_value), data_point={}, detector_name="detector")
    if score_value >= 0.9:
        assert severity_classifier.classify_severity(anomaly) == "critical"
    elif score_value >= 0.7:
        assert severity_classifier.classify_severity(anomaly) == "high"
    elif score_value >= 0.5:
        assert severity_classifier.classify_severity(anomaly) == "medium"
    else:
        assert severity_classifier.classify_severity(anomaly) == "low"

class TestServiceIntegrationFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.container = create_container(testing=True)
        self.classification_service = self.container.anomaly_classification_service

    def test_classification_service_integration(self):
        """Test the classification service integration flow."""
        # Test integrated classification service
        classification_service = self.classification_service()
        
        # Test with different score ranges
        anomaly = Anomaly(
            score=AnomalyScore(0.85),
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="test_detector"
        )
        
        classification_service.classify(anomaly)
        
        # Verify classification metadata was added
        self.assertIn("severity", anomaly.metadata)
        self.assertIn("type", anomaly.metadata)
        self.assertEqual(anomaly.metadata["severity"], "high")
        self.assertEqual(anomaly.metadata["type"], "point")
        
    def test_batch_processing_integration(self):
        """Test batch processing classifier integration."""
        classification_service = self.classification_service()
        classification_service.use_batch_processing_classifiers()
        
        # Process multiple anomalies
        anomalies = [
            Anomaly(
                score=AnomalyScore(0.9),
                data_point={"feature1": 1.0},
                detector_name="test_detector"
            ),
            Anomaly(
                score=AnomalyScore(0.6),
                data_point={"feature1": 2.0},
                detector_name="test_detector"
            ),
            Anomaly(
                score=AnomalyScore(0.3),
                data_point={"feature1": 3.0},
                detector_name="test_detector"
            )
        ]
        
        for anomaly in anomalies:
            classification_service.classify(anomaly)
        
        # Verify all classifications
        self.assertEqual(anomalies[0].metadata["severity"], "critical")
        self.assertEqual(anomalies[1].metadata["severity"], "medium")
        self.assertEqual(anomalies[2].metadata["severity"], "low")

class TestEdgeCases(unittest.TestCase):
    def test_nan_score_validation(self):
        """Test that NaN scores are properly validated by AnomalyScore."""
        # AnomalyScore should reject NaN values during construction
        with self.assertRaises(InvalidValueError):
            AnomalyScore(float('nan'))

    def test_extreme_scores_validation(self):
        """Test that extreme scores are properly validated by AnomalyScore."""
        # AnomalyScore should reject infinity values during construction
        with self.assertRaises(InvalidValueError):
            AnomalyScore(float('inf'))
        
        with self.assertRaises(InvalidValueError):
            AnomalyScore(float('-inf'))
        
        # Values outside [0, 1] should be rejected
        with self.assertRaises(InvalidValueError):
            AnomalyScore(1.5)
        
        with self.assertRaises(InvalidValueError):
            AnomalyScore(-0.5)

    def test_missing_metadata_defaults(self):
        """Test classification with missing metadata defaults correctly."""
        score = AnomalyScore(0.6)
        anomaly = Anomaly(score=score, data_point={}, detector_name="detector")
        # Missing metadata should default correctly
        self.assertEqual(DefaultSeverityClassifier().classify_severity(anomaly), "medium")
        
        # Test with empty metadata
        anomaly_empty = Anomaly(score=score, data_point={}, detector_name="detector")
        anomaly_empty.metadata = {}
        self.assertEqual(DefaultSeverityClassifier().classify_severity(anomaly_empty), "medium")

    def test_boundary_values(self):
        """Test classification with boundary values."""
        classifier = DefaultSeverityClassifier()
        
        # Test exact boundary values
        test_cases = [
            (0.0, "low"),
            (0.5, "medium"),
            (0.7, "high"),
            (0.9, "critical"),
            (1.0, "critical")
        ]
        
        for score_val, expected_severity in test_cases:
            anomaly = Anomaly(
                score=AnomalyScore(score_val),
                data_point={},
                detector_name="detector"
            )
            result = classifier.classify_severity(anomaly)
            self.assertEqual(result, expected_severity, 
                           f"Score {score_val} should be classified as {expected_severity}, got {result}")

    def test_edge_case_type_classification(self):
        """Test type classification with edge cases."""
        classifier = DefaultTypeClassifier()
        
        # Test with exactly 4 features (boundary for collective)
        anomaly = Anomaly(
            score=AnomalyScore(0.5),
            data_point={"f1": 1, "f2": 2, "f3": 3, "f4": 4},
            detector_name="detector"
        )
        result = classifier.classify_type(anomaly)
        self.assertEqual(result, "collective")
        
        # Test with 3 features (should be point)
        anomaly = Anomaly(
            score=AnomalyScore(0.5),
            data_point={"f1": 1, "f2": 2, "f3": 3},
            detector_name="detector"
        )
        result = classifier.classify_type(anomaly)
        self.assertEqual(result, "point")
        
        # Test with temporal context and few features (should be contextual)
        anomaly = Anomaly(
            score=AnomalyScore(0.5),
            data_point={"f1": 1, "f2": 2},
            detector_name="detector"
        )
        anomaly.add_metadata("temporal_context", True)
        result = classifier.classify_type(anomaly)
        self.assertEqual(result, "contextual")
        
        # Test with temporal context but many features (collective takes precedence)
        anomaly = Anomaly(
            score=AnomalyScore(0.5),
            data_point={"f1": 1, "f2": 2, "f3": 3, "f4": 4, "f5": 5},
            detector_name="detector"
        )
        anomaly.add_metadata("temporal_context", True)
        result = classifier.classify_type(anomaly)
        # According to the classifier logic, collective check (>3 features) comes first
        self.assertEqual(result, "collective")

