"""Tests for advanced classification service."""

import pytest
from unittest.mock import MagicMock, patch

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.services.advanced_classification_service import AdvancedClassificationService
from pynomaly.domain.services.threshold_severity_classifier import ThresholdSeverityClassifier
from pynomaly.domain.value_objects.anomaly_classification import (
    AnomalySubType,
    ClassificationMethod,
    ConfidenceLevel,
)
from pynomaly.domain.value_objects import ContaminationRate


class TestAdvancedClassificationService:
    """Test suite for AdvancedClassificationService."""

    @pytest.fixture
    def detector(self):
        """Create a test detector."""
        return Detector(
            name="test_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate(0.1),
        )

    @pytest.fixture
    def severity_classifier(self):
        """Create a test severity classifier."""
        return ThresholdSeverityClassifier()

    @pytest.fixture
    def service(self, severity_classifier):
        """Create an advanced classification service."""
        return AdvancedClassificationService(
            severity_classifier=severity_classifier,
            enable_hierarchical=True,
            enable_multiclass=True,
            confidence_threshold=0.5,
        )

    def test_init(self, service):
        """Test service initialization."""
        assert service.enable_hierarchical is True
        assert service.enable_multiclass is True
        assert service.confidence_threshold == 0.5
        assert service.severity_classifier is not None

    def test_classify_anomaly_high_score(self, service, detector):
        """Test classification with high anomaly score."""
        anomaly_score = 0.9
        feature_data = {"feature1": 1.5, "feature2": 2.0}
        context_data = {"timestamp": "2023-01-01T00:00:00Z"}

        classification = service.classify_anomaly(
            anomaly_score=anomaly_score,
            detector=detector,
            feature_data=feature_data,
            context_data=context_data,
        )

        assert classification.get_primary_class() == "anomaly"
        assert classification.get_confidence_score() == anomaly_score
        assert classification.severity_classification == "critical"
        assert classification.is_hierarchical() is True
        assert classification.is_multi_class() is True

    def test_classify_anomaly_low_score(self, service, detector):
        """Test classification with low anomaly score."""
        anomaly_score = 0.2
        
        classification = service.classify_anomaly(
            anomaly_score=anomaly_score,
            detector=detector,
        )

        assert classification.get_primary_class() == "normal"
        assert classification.get_confidence_score() == anomaly_score
        assert classification.severity_classification == "low"

    def test_create_basic_classification(self, service, detector):
        """Test basic classification creation."""
        anomaly_score = 0.8
        feature_data = {"feature1": 1.0}

        basic_classification = service._create_basic_classification(
            anomaly_score, detector, feature_data
        )

        assert basic_classification.predicted_class == "anomaly"
        assert basic_classification.confidence_score == anomaly_score
        assert basic_classification.confidence_level == ConfidenceLevel.HIGH
        assert "detector_name" in basic_classification.metadata
        assert basic_classification.probability_distribution["anomaly"] == anomaly_score

    def test_create_hierarchical_classification(self, service, detector):
        """Test hierarchical classification creation."""
        basic_classification = service._create_basic_classification(
            0.8, detector, {"feature1": 1.0}
        )
        
        hierarchical = service._create_hierarchical_classification(
            basic_classification, detector, {"feature1": 1.0}
        )

        assert hierarchical is not None
        assert hierarchical.primary_category == "ensemble"
        assert hierarchical.secondary_category is not None
        assert hierarchical.get_hierarchy_depth() >= 2

    def test_create_multiclass_classification(self, service, detector):
        """Test multi-class classification creation."""
        basic_classification = service._create_basic_classification(
            0.8, detector, {"feature1": 1.0}
        )
        
        multiclass = service._create_multiclass_classification(
            basic_classification, detector, {"feature1": 1.0}
        )

        assert multiclass is not None
        assert multiclass.primary_result == basic_classification
        assert len(multiclass.alternative_results) > 0
        assert multiclass.classification_threshold == 0.5

    def test_calculate_feature_contributions(self, service):
        """Test feature contribution calculation."""
        feature_data = {"feature1": 2.0, "feature2": 1.0, "feature3": 3.0}
        
        contributions = service._calculate_feature_contributions(feature_data)
        
        assert len(contributions) == 3
        assert all(0 <= contrib <= 1 for contrib in contributions.values())
        assert abs(sum(contributions.values()) - 1.0) < 1e-6

    def test_calculate_feature_contributions_empty(self, service):
        """Test feature contribution calculation with empty data."""
        contributions = service._calculate_feature_contributions({})
        assert contributions == {}

    def test_determine_classification_method(self, service, detector):
        """Test classification method determination."""
        # Test ensemble detection
        detector.algorithm_name = "ensemble_isolation_forest"
        method = service._determine_classification_method(detector)
        assert method == ClassificationMethod.ENSEMBLE

        # Test supervised detection
        detector.algorithm_name = "svm_one_class"
        method = service._determine_classification_method(detector)
        assert method == ClassificationMethod.SUPERVISED

        # Test unsupervised detection
        detector.algorithm_name = "isolation_forest"
        method = service._determine_classification_method(detector)
        assert method == ClassificationMethod.UNSUPERVISED

    def test_get_primary_category_from_detector(self, service, detector):
        """Test primary category determination from detector."""
        # Test isolation forest
        detector.algorithm_name = "isolation_forest"
        category = service._get_primary_category_from_detector(detector)
        assert category == "ensemble"

        # Test clustering
        detector.algorithm_name = "kmeans_anomaly_detector"
        category = service._get_primary_category_from_detector(detector)
        assert category == "clustering"

        # Test statistical
        detector.algorithm_name = "gaussian_anomaly_detector"
        category = service._get_primary_category_from_detector(detector)
        assert category == "statistical"

    def test_determine_secondary_category(self, service):
        """Test secondary category determination."""
        basic_classification = MagicMock()
        basic_classification.confidence_score = 0.9
        
        secondary = service._determine_secondary_category(
            basic_classification, {"feature1": 1.0}
        )
        
        assert secondary == "high_confidence"

    def test_determine_tertiary_category(self, service):
        """Test tertiary category determination."""
        basic_classification = MagicMock()
        
        # Test univariate
        tertiary = service._determine_tertiary_category(
            basic_classification, {"feature1": 1.0}
        )
        assert tertiary == "univariate"

        # Test multivariate
        tertiary = service._determine_tertiary_category(
            basic_classification, {f"feature{i}": 1.0 for i in range(10)}
        )
        assert tertiary == "medium_dimensional"

    def test_determine_anomaly_subtype(self, service):
        """Test anomaly subtype determination."""
        basic_classification = MagicMock()
        
        # Test extreme value
        basic_classification.confidence_score = 0.95
        subtype = service._determine_anomaly_subtype(basic_classification, {})
        assert subtype == AnomalySubType.EXTREME_VALUE

        # Test outlier
        basic_classification.confidence_score = 0.8
        subtype = service._determine_anomaly_subtype(basic_classification, {})
        assert subtype == AnomalySubType.OUTLIER

        # Test novelty
        basic_classification.confidence_score = 0.6
        subtype = service._determine_anomaly_subtype(basic_classification, {})
        assert subtype == AnomalySubType.NOVELTY

    def test_extract_temporal_context(self, service):
        """Test temporal context extraction."""
        context_data = {
            "timestamp": "2023-01-01T00:00:00Z",
            "time_series": [1, 2, 3, 4, 5],
            "seasonality": True,
        }
        
        temporal_context = service._extract_temporal_context(context_data)
        
        assert temporal_context["has_timestamp"] is True
        assert temporal_context["is_time_series"] is True
        assert temporal_context["series_length"] == 5
        assert temporal_context["seasonality_detected"] is True

    def test_extract_spatial_context(self, service):
        """Test spatial context extraction."""
        context_data = {
            "location": "New York",
            "coordinates": [40.7128, -74.0060],
            "region": "Northeast",
        }
        
        spatial_context = service._extract_spatial_context(context_data)
        
        assert spatial_context["has_location"] is True
        assert spatial_context["has_coordinates"] is True
        assert spatial_context["region"] == "Northeast"

    def test_extract_general_context(self, service):
        """Test general context extraction."""
        context_data = {
            "business_unit": "Finance",
            "data_source": "transactions",
            "priority": "high",
            "model_version": "v1.2.3",
            "environment": "production",
        }
        
        general_context = service._extract_general_context(context_data)
        
        assert general_context["business_unit"] == "Finance"
        assert general_context["data_source"] == "transactions"
        assert general_context["priority"] == "high"
        assert general_context["model_version"] == "v1.2.3"
        assert general_context["environment"] == "production"

    def test_classify_batch(self, service, detector):
        """Test batch classification."""
        anomaly_scores = [0.9, 0.2, 0.7, 0.1]
        feature_data_batch = [
            {"feature1": 1.0},
            {"feature1": 0.5},
            {"feature1": 1.5},
            {"feature1": 0.2},
        ]
        
        classifications = service.classify_batch(
            anomaly_scores, detector, feature_data_batch
        )
        
        assert len(classifications) == 4
        assert classifications[0].get_primary_class() == "anomaly"
        assert classifications[1].get_primary_class() == "normal"
        assert classifications[2].get_primary_class() == "anomaly"
        assert classifications[3].get_primary_class() == "normal"

    def test_classify_batch_empty(self, service, detector):
        """Test batch classification with empty input."""
        classifications = service.classify_batch([], detector)
        assert classifications == []

    def test_get_classification_summary(self, service, detector):
        """Test classification summary generation."""
        anomaly_scores = [0.9, 0.2, 0.7, 0.1]
        classifications = service.classify_batch(anomaly_scores, detector)
        
        summary = service.get_classification_summary(classifications)
        
        assert summary["total_classifications"] == 4
        assert summary["anomaly_count"] == 2
        assert "severity_distribution" in summary
        assert "confidence_distribution" in summary
        assert "hierarchical_depth_distribution" in summary

    def test_get_classification_summary_empty(self, service):
        """Test classification summary with empty input."""
        summary = service.get_classification_summary([])
        assert summary == {}

    def test_disabled_hierarchical_classification(self, severity_classifier):
        """Test service with disabled hierarchical classification."""
        service = AdvancedClassificationService(
            severity_classifier=severity_classifier,
            enable_hierarchical=False,
            enable_multiclass=True,
        )
        
        detector = Detector(
            name="test_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate(0.1),
        )
        
        classification = service.classify_anomaly(0.8, detector)
        
        assert classification.is_hierarchical() is False
        assert classification.is_multi_class() is True

    def test_disabled_multiclass_classification(self, severity_classifier):
        """Test service with disabled multi-class classification."""
        service = AdvancedClassificationService(
            severity_classifier=severity_classifier,
            enable_hierarchical=True,
            enable_multiclass=False,
        )
        
        detector = Detector(
            name="test_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate(0.1),
        )
        
        classification = service.classify_anomaly(0.8, detector)
        
        assert classification.is_hierarchical() is True
        assert classification.is_multi_class() is False

    def test_custom_confidence_threshold(self, severity_classifier):
        """Test service with custom confidence threshold."""
        service = AdvancedClassificationService(
            severity_classifier=severity_classifier,
            confidence_threshold=0.7,
        )
        
        detector = Detector(
            name="test_detector",
            algorithm_name="isolation_forest",
            contamination_rate=ContaminationRate(0.1),
        )
        
        # Score below threshold should be normal
        classification = service.classify_anomaly(0.6, detector)
        assert classification.get_primary_class() == "normal"
        
        # Score above threshold should be anomaly
        classification = service.classify_anomaly(0.8, detector)
        assert classification.get_primary_class() == "anomaly"

    @patch('numpy.random.uniform')
    def test_generate_alternative_classifications(self, mock_random, service, detector):
        """Test alternative classification generation."""
        mock_random.return_value = 0.8
        
        basic_classification = service._create_basic_classification(
            0.8, detector, {"feature1": 1.0}
        )
        
        alternatives = service._generate_alternative_classifications(
            basic_classification, detector, {"feature1": 1.0}
        )
        
        assert len(alternatives) <= 3
        assert all(alt.metadata.get("alternative_classification") is True for alt in alternatives)

    def test_classification_with_missing_data(self, service, detector):
        """Test classification with missing feature or context data."""
        classification = service.classify_anomaly(
            anomaly_score=0.7,
            detector=detector,
            feature_data=None,
            context_data=None,
        )
        
        assert classification is not None
        assert classification.get_primary_class() == "anomaly"
        assert classification.context_classification == {}
        assert classification.temporal_classification == {}
        assert classification.spatial_classification == {}

    def test_classification_requires_escalation(self, service, detector):
        """Test classification that requires escalation."""
        # High severity should require escalation
        classification = service.classify_anomaly(0.95, detector)
        assert classification.requires_escalation() is True
        
        # Low confidence should require escalation
        classification = service.classify_anomaly(0.2, detector)
        assert classification.requires_escalation() is True or classification.basic_classification.requires_review()

    def test_classification_full_summary(self, service, detector):
        """Test full classification summary generation."""
        classification = service.classify_anomaly(
            anomaly_score=0.8,
            detector=detector,
            feature_data={"feature1": 1.0},
            context_data={"timestamp": "2023-01-01T00:00:00Z"},
        )
        
        summary = classification.get_full_classification_summary()
        
        assert "primary_class" in summary
        assert "confidence_score" in summary
        assert "confidence_level" in summary
        assert "severity" in summary
        assert "classification_method" in summary
        assert "hierarchical_path" in summary
        assert "hierarchy_depth" in summary
        assert "alternative_classes" in summary