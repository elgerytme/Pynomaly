"""Tests for detection pipeline integration."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.anomaly_event import EventType, EventSeverity
from pynomaly.domain.services.advanced_classification_service import AdvancedClassificationService
from pynomaly.domain.services.detection_pipeline_integration import DetectionPipelineIntegration
from pynomaly.domain.services.threshold_severity_classifier import ThresholdSeverityClassifier
from pynomaly.domain.value_objects import ContaminationRate


class TestDetectionPipelineIntegration:
    """Test suite for DetectionPipelineIntegration."""

    @pytest.fixture
    def mock_detector(self):
        """Create a mock detector for testing."""
        return Detector(
            name="test_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate.from_value(0.1),
            parameters={"n_estimators": 100},
            metadata={"model_version": "1.0.0"},
        )

    @pytest.fixture
    def classification_service(self):
        """Create a classification service for testing."""
        severity_classifier = ThresholdSeverityClassifier()
        return AdvancedClassificationService(
            severity_classifier=severity_classifier,
            enable_hierarchical=True,
            enable_multiclass=True,
        )

    @pytest.fixture
    def pipeline_integration(self, classification_service):
        """Create a pipeline integration for testing."""
        return DetectionPipelineIntegration(classification_service)

    def test_process_detection_result_anomaly(self, pipeline_integration, mock_detector):
        """Test processing a detection result for an anomaly."""
        raw_data = {"sensor_value": 150.0, "threshold": 100.0}
        feature_data = {"deviation": 50.0, "zscore": 2.5}
        context_data = {
            "timestamp": datetime.utcnow(),
            "location": {"lat": 40.7128, "lon": -74.0060},
            "business_context": {"priority": "high"},
        }
        
        classification, event = pipeline_integration.process_detection_result(
            anomaly_score=0.85,
            detector=mock_detector,
            raw_data=raw_data,
            feature_data=feature_data,
            context_data=context_data,
        )
        
        # Check classification
        assert classification.get_primary_class() == "anomaly"
        assert classification.get_confidence_score() == 0.85
        assert classification.severity_classification == "high"
        
        # Check event
        assert event.event_type == EventType.ANOMALY_DETECTED
        assert event.severity == EventSeverity.HIGH
        assert event.detector_id == mock_detector.id
        assert event.anomaly_data is not None
        assert event.anomaly_data.anomaly_score == 0.85
        assert "isolation_forest" in event.title
        assert event.raw_data == raw_data

    def test_process_detection_result_normal(self, pipeline_integration, mock_detector):
        """Test processing a detection result for normal data."""
        raw_data = {"sensor_value": 75.0, "threshold": 100.0}
        
        classification, event = pipeline_integration.process_detection_result(
            anomaly_score=0.2,
            detector=mock_detector,
            raw_data=raw_data,
        )
        
        # Check classification
        assert classification.get_primary_class() == "normal"
        assert classification.get_confidence_score() == 0.2
        assert classification.severity_classification == "low"
        
        # Check event
        assert event.event_type == EventType.CUSTOM
        assert event.severity == EventSeverity.LOW

    def test_process_detection_result_critical(self, pipeline_integration, mock_detector):
        """Test processing a critical anomaly detection result."""
        raw_data = {"sensor_value": 200.0, "threshold": 100.0}
        
        classification, event = pipeline_integration.process_detection_result(
            anomaly_score=0.95,
            detector=mock_detector,
            raw_data=raw_data,
        )
        
        # Check classification
        assert classification.severity_classification == "critical"
        
        # Check event
        assert event.event_type == EventType.ANOMALY_ESCALATED
        assert event.severity == EventSeverity.CRITICAL
        assert "CRITICAL" in event.title

    def test_process_batch_detection_results(self, pipeline_integration, mock_detector):
        """Test processing a batch of detection results."""
        anomaly_scores = [0.8, 0.3, 0.9, 0.1]
        raw_data_batch = [
            {"value": 150.0},
            {"value": 75.0},
            {"value": 180.0},
            {"value": 50.0},
        ]
        feature_data_batch = [
            {"feature1": 2.0},
            {"feature1": 0.5},
            {"feature1": 3.0},
            {"feature1": 0.1},
        ]
        
        classifications, events = pipeline_integration.process_batch_detection_results(
            anomaly_scores=anomaly_scores,
            detector=mock_detector,
            raw_data_batch=raw_data_batch,
            feature_data_batch=feature_data_batch,
        )
        
        # Check classifications
        assert len(classifications) == 4
        assert classifications[0].get_primary_class() == "anomaly"
        assert classifications[1].get_primary_class() == "normal"
        assert classifications[2].get_primary_class() == "anomaly"
        assert classifications[3].get_primary_class() == "normal"
        
        # Check events
        assert len(events) == 4
        assert events[0].severity == EventSeverity.HIGH
        assert events[1].severity == EventSeverity.LOW
        assert events[2].severity == EventSeverity.HIGH
        assert events[3].severity == EventSeverity.LOW

    def test_process_empty_batch(self, pipeline_integration, mock_detector):
        """Test processing an empty batch."""
        classifications, events = pipeline_integration.process_batch_detection_results(
            anomaly_scores=[],
            detector=mock_detector,
            raw_data_batch=[],
        )
        
        assert classifications == []
        assert events == []

    def test_event_type_determination(self, pipeline_integration):
        """Test event type determination logic."""
        # Mock classification for normal data
        mock_classification_normal = Mock()
        mock_classification_normal.get_primary_class.return_value = "normal"
        mock_classification_normal.severity_classification = "low"
        
        event_type = pipeline_integration._determine_event_type(mock_classification_normal)
        assert event_type == EventType.CUSTOM
        
        # Mock classification for critical anomaly
        mock_classification_critical = Mock()
        mock_classification_critical.get_primary_class.return_value = "anomaly"
        mock_classification_critical.severity_classification = "critical"
        
        event_type = pipeline_integration._determine_event_type(mock_classification_critical)
        assert event_type == EventType.ANOMALY_ESCALATED
        
        # Mock classification for regular anomaly
        mock_classification_anomaly = Mock()
        mock_classification_anomaly.get_primary_class.return_value = "anomaly"
        mock_classification_anomaly.severity_classification = "high"
        
        event_type = pipeline_integration._determine_event_type(mock_classification_anomaly)
        assert event_type == EventType.ANOMALY_DETECTED

    def test_severity_mapping(self, pipeline_integration):
        """Test severity mapping from classification to event."""
        test_cases = [
            ("low", EventSeverity.LOW),
            ("medium", EventSeverity.MEDIUM),
            ("high", EventSeverity.HIGH),
            ("critical", EventSeverity.CRITICAL),
            ("unknown", EventSeverity.MEDIUM),  # Default case
        ]
        
        for classification_severity, expected_event_severity in test_cases:
            event_severity = pipeline_integration._map_to_event_severity(classification_severity)
            assert event_severity == expected_event_severity

    def test_explanation_generation(self, pipeline_integration):
        """Test explanation generation for different classification types."""
        # Mock basic classification
        mock_basic = Mock()
        mock_basic.confidence_level.value = "high"
        mock_basic.feature_contributions = {"feature1": 0.6, "feature2": 0.4}
        
        # Mock hierarchical classification
        mock_hierarchical = Mock()
        mock_hierarchical.get_full_path.return_value = "ensemble > high_confidence > outlier"
        
        # Mock multi-class classification
        mock_multiclass = Mock()
        mock_multiclass.has_ambiguous_classification.return_value = True
        
        # Mock advanced classification
        mock_classification = Mock()
        mock_classification.get_primary_class.return_value = "anomaly"
        mock_classification.basic_classification = mock_basic
        mock_classification.severity_classification = "high"
        mock_classification.is_hierarchical.return_value = True
        mock_classification.hierarchical_classification = mock_hierarchical
        mock_classification.is_multi_class.return_value = True
        mock_classification.multi_class_classification = mock_multiclass
        
        explanation = pipeline_integration._generate_explanation(mock_classification)
        
        assert "anomaly with high confidence" in explanation
        assert "Severity level: high" in explanation
        assert "ensemble > high_confidence > outlier" in explanation
        assert "ambiguous" in explanation
        assert "feature1, feature2" in explanation

    def test_event_title_generation(self, pipeline_integration, mock_detector):
        """Test event title generation."""
        # Mock hierarchical classification
        mock_hierarchical = Mock()
        mock_hierarchical.primary_category = "ensemble"
        
        # Mock classification with hierarchy
        mock_classification = Mock()
        mock_classification.severity_classification = "high"
        mock_classification.is_hierarchical.return_value = True
        mock_classification.hierarchical_classification = mock_hierarchical
        
        title = pipeline_integration._generate_event_title(mock_classification, mock_detector)
        assert "HIGH ensemble anomaly detected by test_detector" == title
        
        # Mock classification without hierarchy
        mock_classification.is_hierarchical.return_value = False
        
        title = pipeline_integration._generate_event_title(mock_classification, mock_detector)
        assert "HIGH anomaly detected by test_detector" == title

    def test_event_description_generation(self, pipeline_integration, mock_detector):
        """Test event description generation."""
        # Mock hierarchical classification
        mock_hierarchical = Mock()
        mock_hierarchical.get_full_path.return_value = "ensemble > outlier"
        
        # Mock classification
        mock_classification = Mock()
        mock_classification.get_confidence_score.return_value = 0.85
        mock_classification.is_hierarchical.return_value = True
        mock_classification.hierarchical_classification = mock_hierarchical
        mock_classification.has_temporal_context.return_value = True
        mock_classification.has_spatial_context.return_value = False
        mock_classification.requires_escalation.return_value = True
        
        description = pipeline_integration._generate_event_description(
            mock_classification, mock_detector
        )
        
        assert "isolation_forest algorithm" in description
        assert "0.850" in description
        assert "ensemble > outlier" in description
        assert "Temporal context analysis" in description
        assert "immediate attention" in description

    def test_technical_context_creation(self, pipeline_integration, mock_detector):
        """Test technical context creation."""
        # Mock basic classification
        mock_basic = Mock()
        mock_basic.classification_method.value = "ensemble"
        mock_basic.confidence_level.value = "high"
        
        # Mock hierarchical classification
        mock_hierarchical = Mock()
        mock_hierarchical.get_hierarchy_depth.return_value = 3
        
        # Mock multi-class classification
        mock_multiclass = Mock()
        mock_multiclass.alternative_results = [Mock(), Mock()]
        mock_multiclass.has_ambiguous_classification.return_value = True
        
        # Mock advanced classification
        mock_classification = Mock()
        mock_classification.basic_classification = mock_basic
        mock_classification.requires_escalation.return_value = True
        mock_classification.is_hierarchical.return_value = True
        mock_classification.hierarchical_classification = mock_hierarchical
        mock_classification.is_multi_class.return_value = True
        mock_classification.multi_class_classification = mock_multiclass
        
        context = pipeline_integration._create_technical_context(
            mock_classification, mock_detector
        )
        
        assert "detector_info" in context
        assert context["classification_method"] == "ensemble"
        assert context["confidence_level"] == "high"
        assert context["requires_escalation"] is True
        assert context["hierarchy_depth"] == 3
        assert context["alternative_classifications"] == 2
        assert context["has_ambiguous_classification"] is True

    def test_classification_tags_addition(self, pipeline_integration):
        """Test addition of classification-specific tags to events."""
        # Mock event
        mock_event = Mock()
        mock_event.add_tag = Mock()
        
        # Mock basic classification
        mock_basic = Mock()
        mock_basic.confidence_level.value = "high"
        mock_basic.classification_method.value = "ensemble"
        
        # Mock hierarchical classification
        mock_hierarchical = Mock()
        mock_hierarchical.primary_category = "distance"
        mock_hierarchical.sub_type.value = "outlier"
        
        # Mock multi-class classification
        mock_multiclass = Mock()
        mock_multiclass.has_ambiguous_classification.return_value = True
        
        # Mock advanced classification
        mock_classification = Mock()
        mock_classification.severity_classification = "high"
        mock_classification.basic_classification = mock_basic
        mock_classification.is_hierarchical.return_value = True
        mock_classification.hierarchical_classification = mock_hierarchical
        mock_classification.is_multi_class.return_value = True
        mock_classification.multi_class_classification = mock_multiclass
        mock_classification.has_temporal_context.return_value = True
        mock_classification.has_spatial_context.return_value = False
        mock_classification.requires_escalation.return_value = True
        
        pipeline_integration._add_classification_tags(mock_event, mock_classification)
        
        # Verify tags were added
        expected_tags = [
            "severity:high",
            "confidence:high",
            "method:ensemble",
            "hierarchical_classification",
            "primary_category:distance",
            "subtype:outlier",
            "multiclass_classification",
            "ambiguous_classification",
            "temporal_context",
            "requires_escalation",
        ]
        
        for tag in expected_tags:
            mock_event.add_tag.assert_any_call(tag)

    def test_classification_summary_event_creation(self, pipeline_integration, mock_detector):
        """Test creation of classification summary event."""
        # Mock classifications
        mock_classifications = [Mock() for _ in range(3)]
        
        # Mock classification service summary
        mock_summary = {
            "total_classifications": 3,
            "anomaly_count": 2,
            "severity_distribution": {"high": 2, "low": 1},
        }
        
        with patch.object(
            pipeline_integration.classification_service,
            'get_classification_summary',
            return_value=mock_summary
        ):
            event = pipeline_integration.create_classification_summary_event(
                mock_classifications, mock_detector
            )
        
        assert event.event_type == EventType.BATCH_COMPLETED
        assert event.severity == EventSeverity.INFO
        assert "3 classifications" in event.description
        assert "2 anomalies detected" in event.description
        assert event.raw_data["classification_summary"] == mock_summary

    def test_process_detection_with_missing_context_data(self, pipeline_integration, mock_detector):
        """Test processing detection result with missing context data."""
        raw_data = {"value": 100.0}
        
        # Test with None context_data
        classification, event = pipeline_integration.process_detection_result(
            anomaly_score=0.7,
            detector=mock_detector,
            raw_data=raw_data,
            context_data=None,
        )
        
        assert classification is not None
        assert event is not None
        assert event.business_context == {}
        
        # Test with empty context_data
        classification, event = pipeline_integration.process_detection_result(
            anomaly_score=0.7,
            detector=mock_detector,
            raw_data=raw_data,
            context_data={},
        )
        
        assert classification is not None
        assert event is not None