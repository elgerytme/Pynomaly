"""Tests for the enhanced detection service."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from packages.core.application.services.enhanced_detection_service import (
    EnhancedDetectionService,
)
from packages.core.domain.entities.anomaly import Anomaly


class TestEnhancedDetectionService:
    """Test suite for EnhancedDetectionService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = EnhancedDetectionService()

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        service = EnhancedDetectionService()
        assert service.enable_advanced_classification is True

    def test_init_with_disabled_classification(self):
        """Test initialization with disabled advanced classification."""
        service = EnhancedDetectionService(enable_advanced_classification=False)
        assert service.enable_advanced_classification is False

    def test_enhance_anomaly_basic_fallback(self):
        """Test anomaly enhancement with basic fallback."""
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="test_detector",
        )
        
        # Test with disabled advanced classification
        service = EnhancedDetectionService(enable_advanced_classification=False)
        enhanced = service.enhance_anomaly_with_classification(anomaly)
        
        assert enhanced.score == 0.8
        assert enhanced.detector_name == "test_detector"
        assert enhanced.get_anomaly_type() == "unknown"

    def test_enhance_anomaly_with_mock_classification(self):
        """Test anomaly enhancement with mocked classification service."""
        anomaly = Anomaly(
            score=0.9,
            data_point={"feature1": 1.5, "feature2": 2.5},
            detector_name="test_detector",
        )
        
        # Mock the classification service
        mock_classification_service = Mock()
        mock_classification_result = Mock()
        mock_classification_result.basic_classification = Mock()
        mock_classification_result.basic_classification.confidence_score = 0.85
        mock_classification_result.hierarchical_classification = Mock()
        mock_classification_result.hierarchical_classification.primary_category = "point"
        mock_classification_result.hierarchical_classification.get_full_path = Mock(
            return_value="point > outlier"
        )
        mock_classification_result.severity_classification = "high"
        
        mock_classification_service.classify_anomaly.return_value = mock_classification_result
        
        service = EnhancedDetectionService(
            classification_service=mock_classification_service
        )
        enhanced = service.enhance_anomaly_with_classification(anomaly)
        
        assert enhanced.score == 0.9
        assert enhanced.classification == mock_classification_result
        assert "classification_method" in enhanced.metadata
        assert enhanced.metadata["classification_method"] == "advanced"

    def test_detect_with_advanced_classification(self):
        """Test detection with advanced classification."""
        data_points = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 2.0, "feature2": 3.0},
        ]
        scores = [0.8, 0.9]
        
        anomalies = self.service.detect_with_advanced_classification(
            data_points=data_points,
            detector_name="test_detector",
            scores=scores,
        )
        
        assert len(anomalies) == 2
        assert all(isinstance(a, Anomaly) for a in anomalies)
        assert anomalies[0].score == 0.8
        assert anomalies[1].score == 0.9

    def test_batch_classify_anomalies(self):
        """Test batch classification of anomalies."""
        anomalies = [
            Anomaly(
                score=0.7,
                data_point={"feature1": 1.0},
                detector_name="detector1",
            ),
            Anomaly(
                score=0.9,
                data_point={"feature1": 2.0},
                detector_name="detector2",
            ),
        ]
        
        enhanced = self.service.batch_classify_anomalies(anomalies)
        
        assert len(enhanced) == 2
        assert all(isinstance(a, Anomaly) for a in enhanced)

    def test_get_classification_summary_empty(self):
        """Test classification summary for empty list."""
        summary = self.service.get_classification_summary([])
        assert summary["total"] == 0
        assert summary["summary"] == {}

    def test_get_classification_summary_with_anomalies(self):
        """Test classification summary with anomalies."""
        anomalies = [
            Anomaly(score=0.6, data_point={}, detector_name="test"),
            Anomaly(score=0.8, data_point={}, detector_name="test"),
            Anomaly(score=0.95, data_point={}, detector_name="test"),
        ]
        
        summary = self.service.get_classification_summary(anomalies)
        
        assert summary["total"] == 3
        assert "severity_distribution" in summary
        assert "type_distribution" in summary
        assert "average_confidence" in summary
        assert isinstance(summary["average_confidence"], float)

    def test_filter_by_confidence(self):
        """Test filtering anomalies by confidence."""
        anomalies = [
            Anomaly(score=0.3, data_point={}, detector_name="test"),  # Low confidence
            Anomaly(score=0.7, data_point={}, detector_name="test"),  # High confidence
            Anomaly(score=0.9, data_point={}, detector_name="test"),  # High confidence
        ]
        
        filtered = self.service.filter_by_confidence(anomalies, min_confidence=0.6)
        
        assert len(filtered) == 2
        assert all(a.get_confidence_score() >= 0.6 for a in filtered)

    def test_filter_by_severity(self):
        """Test filtering anomalies by severity."""
        anomalies = [
            Anomaly(score=0.3, data_point={}, detector_name="test"),  # Low severity
            Anomaly(score=0.7, data_point={}, detector_name="test"),  # High severity
            Anomaly(score=0.95, data_point={}, detector_name="test"),  # Critical severity
        ]
        
        filtered = self.service.filter_by_severity(anomalies, min_severity="high")
        
        assert len(filtered) == 2
        assert all(a.severity in ["high", "critical"] for a in filtered)

    def test_filter_by_severity_invalid_level(self):
        """Test filtering with invalid severity level."""
        anomalies = [
            Anomaly(score=0.7, data_point={}, detector_name="test"),
        ]
        
        filtered = self.service.filter_by_severity(anomalies, min_severity="invalid")
        
        # Should default to medium level
        assert len(filtered) == 1

    def test_enhanced_anomaly_methods(self):
        """Test enhanced anomaly methods."""
        anomaly = Anomaly(
            score=0.85,
            data_point={"feature1": 1.0},
            detector_name="test_detector",
        )
        
        # Test basic methods
        assert anomaly.get_anomaly_type() == "unknown"
        assert anomaly.get_confidence_score() == 0.85
        assert anomaly.severity == "high"
        assert not anomaly.has_advanced_classification()
        assert anomaly.is_highly_confident()
        assert not anomaly.is_critical_severity()

    def test_enhanced_anomaly_to_dict(self):
        """Test enhanced anomaly dictionary conversion."""
        anomaly = Anomaly(
            score=0.75,
            data_point={"feature1": 1.0},
            detector_name="test_detector",
        )
        
        result = anomaly.to_dict()
        
        assert "anomaly_type" in result
        assert "confidence_score" in result
        assert "has_advanced_classification" in result
        assert result["anomaly_type"] == "unknown"
        assert result["confidence_score"] == 0.75
        assert result["has_advanced_classification"] is False

    def test_classification_error_handling(self):
        """Test error handling in classification."""
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0},
            detector_name="test_detector",
        )
        
        # Mock classification service that raises an error
        mock_classification_service = Mock()
        mock_classification_service.classify_anomaly.side_effect = Exception("Test error")
        
        service = EnhancedDetectionService(
            classification_service=mock_classification_service
        )
        
        enhanced = service.enhance_anomaly_with_classification(anomaly)
        
        assert "classification_error" in enhanced.metadata
        assert enhanced.metadata["classification_method"] == "fallback"
        assert enhanced.metadata["classification_error"] == "Test error"

    def test_anomaly_with_context(self):
        """Test anomaly enhancement with context."""
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0},
            detector_name="test_detector",
        )
        
        context = {"dataset": "test_data", "timestamp": "2025-01-01"}
        
        enhanced = self.service.enhance_anomaly_with_classification(
            anomaly, data_context=context
        )
        
        # The context should be passed to the classification service
        assert enhanced.score == 0.8