"""Tests for anomaly classification value objects."""

import pytest

from monorepo.domain.value_objects.anomaly_classification import (
    AdvancedAnomalyClassification,
    AnomalySubType,
    ClassificationMethod,
    ClassificationResult,
    ConfidenceLevel,
    HierarchicalClassification,
    MultiClassClassification,
)


class TestClassificationResult:
    """Test suite for ClassificationResult."""

    def test_valid_classification_result(self):
        """Test creating a valid classification result."""
        result = ClassificationResult(
            predicted_class="anomaly",
            confidence_score=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            probability_distribution={"anomaly": 0.8, "normal": 0.2},
            feature_contributions={"feature1": 0.6, "feature2": 0.4},
            classification_method=ClassificationMethod.ENSEMBLE,
            metadata={"model_version": "1.0"},
        )
        
        assert result.predicted_class == "anomaly"
        assert result.confidence_score == 0.8
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.is_confident()
        assert not result.requires_review()

    def test_invalid_confidence_score(self):
        """Test validation of confidence score."""
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            ClassificationResult(
                predicted_class="anomaly",
                confidence_score=1.5,
                confidence_level=ConfidenceLevel.HIGH,
            )

    def test_invalid_probability_distribution(self):
        """Test validation of probability distribution."""
        with pytest.raises(ValueError, match="Probability distribution must sum to 1.0"):
            ClassificationResult(
                predicted_class="anomaly",
                confidence_score=0.8,
                confidence_level=ConfidenceLevel.HIGH,
                probability_distribution={"anomaly": 0.8, "normal": 0.3},
            )

    def test_invalid_feature_contributions(self):
        """Test validation of feature contributions."""
        with pytest.raises(ValueError, match="Feature contribution for feature1 must be numeric"):
            ClassificationResult(
                predicted_class="anomaly",
                confidence_score=0.8,
                confidence_level=ConfidenceLevel.HIGH,
                feature_contributions={"feature1": "invalid"},
            )

    def test_from_confidence_score_factory(self):
        """Test factory method for creating result from confidence score."""
        test_cases = [
            (0.95, ConfidenceLevel.VERY_HIGH),
            (0.8, ConfidenceLevel.HIGH),
            (0.6, ConfidenceLevel.MEDIUM),
            (0.4, ConfidenceLevel.LOW),
            (0.2, ConfidenceLevel.VERY_LOW),
        ]
        
        for score, expected_level in test_cases:
            result = ClassificationResult.from_confidence_score("anomaly", score)
            assert result.confidence_score == score
            assert result.confidence_level == expected_level

    def test_confidence_methods(self):
        """Test confidence-related methods."""
        # High confidence
        high_result = ClassificationResult.from_confidence_score("anomaly", 0.9)
        assert high_result.is_confident()
        assert not high_result.requires_review()
        
        # Low confidence
        low_result = ClassificationResult.from_confidence_score("anomaly", 0.2)
        assert not low_result.is_confident()
        assert low_result.requires_review()


class TestHierarchicalClassification:
    """Test suite for HierarchicalClassification."""

    def test_valid_hierarchical_classification(self):
        """Test creating a valid hierarchical classification."""
        classification = HierarchicalClassification(
            primary_category="ensemble",
            secondary_category="high_confidence",
            tertiary_category="outlier",
            sub_type=AnomalySubType.EXTREME_VALUE,
            confidence_scores={"primary": 0.9, "secondary": 0.8},
        )
        
        assert classification.primary_category == "ensemble"
        assert classification.get_hierarchy_depth() == 4
        assert classification.get_full_path() == "ensemble > high_confidence > outlier > extreme_value"

    def test_minimal_hierarchical_classification(self):
        """Test creating hierarchical classification with only primary category."""
        classification = HierarchicalClassification(primary_category="clustering")
        
        assert classification.primary_category == "clustering"
        assert classification.secondary_category is None
        assert classification.get_hierarchy_depth() == 1
        assert classification.get_full_path() == "clustering"

    def test_empty_primary_category(self):
        """Test validation of empty primary category."""
        with pytest.raises(ValueError, match="Primary category cannot be empty"):
            HierarchicalClassification(primary_category="")

    def test_is_child_of_method(self):
        """Test is_child_of method."""
        classification = HierarchicalClassification(
            primary_category="ensemble",
            secondary_category="high_confidence",
            tertiary_category="outlier",
        )
        
        assert classification.is_child_of("ensemble")
        assert classification.is_child_of("ensemble > high_confidence")
        assert not classification.is_child_of("clustering")
        assert not classification.is_child_of("ensemble > low_confidence")


class TestMultiClassClassification:
    """Test suite for MultiClassClassification."""

    def test_valid_multiclass_classification(self):
        """Test creating a valid multi-class classification."""
        primary_result = ClassificationResult.from_confidence_score("severe_anomaly", 0.8)
        alternative_results = [
            ClassificationResult.from_confidence_score("moderate_anomaly", 0.6),
            ClassificationResult.from_confidence_score("weak_anomaly", 0.4),
        ]
        
        multiclass = MultiClassClassification(
            primary_result=primary_result,
            alternative_results=alternative_results,
            classification_threshold=0.5,
            multi_class_strategy="one_vs_rest",
        )
        
        assert multiclass.primary_result == primary_result
        assert len(multiclass.alternative_results) == 2
        assert len(multiclass.get_all_results()) == 3

    def test_invalid_threshold(self):
        """Test validation of classification threshold."""
        primary_result = ClassificationResult.from_confidence_score("anomaly", 0.8)
        
        with pytest.raises(ValueError, match="Classification threshold must be between 0.0 and 1.0"):
            MultiClassClassification(
                primary_result=primary_result,
                classification_threshold=1.5,
            )

    def test_invalid_strategy(self):
        """Test validation of multi-class strategy."""
        primary_result = ClassificationResult.from_confidence_score("anomaly", 0.8)
        
        with pytest.raises(ValueError, match="Strategy must be one of"):
            MultiClassClassification(
                primary_result=primary_result,
                multi_class_strategy="invalid_strategy",
            )

    def test_confident_results(self):
        """Test getting confident results."""
        primary_result = ClassificationResult.from_confidence_score("severe_anomaly", 0.9)
        alternative_results = [
            ClassificationResult.from_confidence_score("moderate_anomaly", 0.8),
            ClassificationResult.from_confidence_score("weak_anomaly", 0.3),
        ]
        
        multiclass = MultiClassClassification(
            primary_result=primary_result,
            alternative_results=alternative_results,
        )
        
        confident_results = multiclass.get_confident_results()
        assert len(confident_results) == 2  # Primary and first alternative

    def test_ambiguous_classification(self):
        """Test detection of ambiguous classification."""
        primary_result = ClassificationResult.from_confidence_score("severe_anomaly", 0.9)
        alternative_results = [
            ClassificationResult.from_confidence_score("moderate_anomaly", 0.85),  # Also high confidence
            ClassificationResult.from_confidence_score("weak_anomaly", 0.3),
        ]
        
        multiclass = MultiClassClassification(
            primary_result=primary_result,
            alternative_results=alternative_results,
        )
        
        assert multiclass.has_ambiguous_classification()

    def test_top_n_results(self):
        """Test getting top N results by confidence."""
        primary_result = ClassificationResult.from_confidence_score("severe_anomaly", 0.8)
        alternative_results = [
            ClassificationResult.from_confidence_score("moderate_anomaly", 0.9),  # Higher than primary
            ClassificationResult.from_confidence_score("weak_anomaly", 0.3),
        ]
        
        multiclass = MultiClassClassification(
            primary_result=primary_result,
            alternative_results=alternative_results,
        )
        
        top_2 = multiclass.get_top_n_results(2)
        assert len(top_2) == 2
        assert top_2[0].predicted_class == "moderate_anomaly"  # Highest confidence
        assert top_2[1].predicted_class == "severe_anomaly"    # Second highest


class TestAdvancedAnomalyClassification:
    """Test suite for AdvancedAnomalyClassification."""

    def test_valid_advanced_classification(self):
        """Test creating a valid advanced anomaly classification."""
        basic_result = ClassificationResult.from_confidence_score("anomaly", 0.8)
        hierarchical = HierarchicalClassification(primary_category="ensemble")
        multiclass = MultiClassClassification(primary_result=basic_result)
        
        advanced = AdvancedAnomalyClassification(
            basic_classification=basic_result,
            hierarchical_classification=hierarchical,
            multi_class_classification=multiclass,
            severity_classification="high",
            context_classification={"source": "sensor"},
            temporal_classification={"has_timestamp": True},
            spatial_classification={"location": "warehouse"},
        )
        
        assert advanced.get_primary_class() == "anomaly"
        assert advanced.get_confidence_score() == 0.8
        assert advanced.is_hierarchical()
        assert advanced.is_multi_class()
        assert advanced.has_spatial_context()
        assert advanced.has_temporal_context()

    def test_minimal_advanced_classification(self):
        """Test creating minimal advanced classification."""
        basic_result = ClassificationResult.from_confidence_score("normal", 0.3)
        
        advanced = AdvancedAnomalyClassification(basic_classification=basic_result)
        
        assert advanced.get_primary_class() == "normal"
        assert advanced.severity_classification == "medium"  # Default
        assert not advanced.is_hierarchical()
        assert not advanced.is_multi_class()
        assert not advanced.has_spatial_context()
        assert not advanced.has_temporal_context()

    def test_invalid_severity(self):
        """Test validation of severity classification."""
        basic_result = ClassificationResult.from_confidence_score("anomaly", 0.8)
        
        with pytest.raises(ValueError, match="Severity must be one of"):
            AdvancedAnomalyClassification(
                basic_classification=basic_result,
                severity_classification="invalid",
            )

    def test_classification_summary(self):
        """Test comprehensive classification summary."""
        basic_result = ClassificationResult.from_confidence_score("anomaly", 0.85)
        hierarchical = HierarchicalClassification(
            primary_category="ensemble",
            secondary_category="high_confidence",
            sub_type=AnomalySubType.OUTLIER,
        )
        alternative_results = [
            ClassificationResult.from_confidence_score("moderate_anomaly", 0.7),
        ]
        multiclass = MultiClassClassification(
            primary_result=basic_result,
            alternative_results=alternative_results,
        )
        
        advanced = AdvancedAnomalyClassification(
            basic_classification=basic_result,
            hierarchical_classification=hierarchical,
            multi_class_classification=multiclass,
            severity_classification="high",
            temporal_classification={"seasonality": True},
            spatial_classification={"region": "north"},
        )
        
        summary = advanced.get_full_classification_summary()
        
        assert summary["primary_class"] == "anomaly"
        assert summary["confidence_score"] == 0.85
        assert summary["confidence_level"] == "high"
        assert summary["severity"] == "high"
        assert summary["hierarchical_path"] == "ensemble > high_confidence > outlier"
        assert summary["hierarchy_depth"] == 3
        assert summary["alternative_classes"] == ["moderate_anomaly"]
        assert not summary["has_ambiguous_classification"]
        assert summary["temporal_context"] == {"seasonality": True}
        assert summary["spatial_context"] == {"region": "north"}

    def test_requires_escalation(self):
        """Test escalation requirement logic."""
        # High severity should require escalation
        basic_result = ClassificationResult.from_confidence_score("anomaly", 0.8)
        advanced_high = AdvancedAnomalyClassification(
            basic_classification=basic_result,
            severity_classification="critical",
        )
        assert advanced_high.requires_escalation()
        
        # Low confidence should require escalation
        low_confidence_result = ClassificationResult.from_confidence_score("anomaly", 0.2)
        advanced_low_conf = AdvancedAnomalyClassification(
            basic_classification=low_confidence_result,
            severity_classification="medium",
        )
        assert advanced_low_conf.requires_escalation()
        
        # Ambiguous classification should require escalation
        multiclass_ambiguous = MultiClassClassification(
            primary_result=ClassificationResult.from_confidence_score("severe_anomaly", 0.9),
            alternative_results=[
                ClassificationResult.from_confidence_score("moderate_anomaly", 0.85),
            ],
        )
        advanced_ambiguous = AdvancedAnomalyClassification(
            basic_classification=basic_result,
            multi_class_classification=multiclass_ambiguous,
            severity_classification="medium",
        )
        assert advanced_ambiguous.requires_escalation()
        
        # Normal case should not require escalation
        normal_advanced = AdvancedAnomalyClassification(
            basic_classification=ClassificationResult.from_confidence_score("anomaly", 0.7),
            severity_classification="medium",
        )
        assert not normal_advanced.requires_escalation()


class TestAnomalySubType:
    """Test suite for AnomalySubType enum."""

    def test_enum_values(self):
        """Test all enum values are accessible."""
        assert AnomalySubType.OUTLIER == "outlier"
        assert AnomalySubType.EXTREME_VALUE == "extreme_value"
        assert AnomalySubType.NOVELTY == "novelty"
        assert AnomalySubType.CONDITIONAL == "conditional"
        assert AnomalySubType.TEMPORAL == "temporal"
        assert AnomalySubType.SPATIAL == "spatial"
        assert AnomalySubType.SEQUENCE == "sequence"
        assert AnomalySubType.PATTERN == "pattern"
        assert AnomalySubType.CLUSTER == "cluster"
        assert AnomalySubType.SYSTEM_WIDE == "system_wide"
        assert AnomalySubType.TREND == "trend"
        assert AnomalySubType.SEASONAL == "seasonal"
        assert AnomalySubType.REGIONAL == "regional"
        assert AnomalySubType.NEIGHBORHOOD == "neighborhood"
        assert AnomalySubType.LOCALIZED == "localized"


class TestClassificationMethod:
    """Test suite for ClassificationMethod enum."""

    def test_enum_values(self):
        """Test all enum values are accessible."""
        assert ClassificationMethod.SUPERVISED == "supervised"
        assert ClassificationMethod.UNSUPERVISED == "unsupervised"
        assert ClassificationMethod.SEMI_SUPERVISED == "semi_supervised"
        assert ClassificationMethod.ENSEMBLE == "ensemble"
        assert ClassificationMethod.HYBRID == "hybrid"


class TestConfidenceLevel:
    """Test suite for ConfidenceLevel enum."""

    def test_enum_values(self):
        """Test all enum values are accessible."""
        assert ConfidenceLevel.VERY_LOW == "very_low"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.VERY_HIGH == "very_high"