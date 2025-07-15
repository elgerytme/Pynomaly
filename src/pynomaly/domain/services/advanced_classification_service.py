"""Advanced anomaly classification service."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.services.threshold_severity_classifier import ThresholdSeverityClassifier
from pynomaly.domain.value_objects.anomaly_classification import (
    AdvancedAnomalyClassification,
    AnomalySubType,
    ClassificationMethod,
    ClassificationResult,
    ConfidenceLevel,
    HierarchicalClassification,
    MultiClassClassification,
)
from pynomaly.domain.value_objects.anomaly_category import AnomalyCategory

logger = logging.getLogger(__name__)


class AdvancedClassificationService:
    """Service for performing advanced anomaly classification.
    
    This service orchestrates various classification strategies to provide
    comprehensive multi-dimensional anomaly classification including:
    - Basic classification with confidence levels
    - Hierarchical classification with parent-child relationships
    - Multi-class classification with alternative results
    - Severity-based classification
    - Context-aware classification (temporal, spatial)
    """

    def __init__(
        self,
        severity_classifier: ThresholdSeverityClassifier | None = None,
        enable_hierarchical: bool = True,
        enable_multiclass: bool = True,
        confidence_threshold: float = 0.5,
    ):
        """Initialize the advanced classification service.
        
        Args:
            severity_classifier: Threshold-based severity classifier
            enable_hierarchical: Enable hierarchical classification
            enable_multiclass: Enable multi-class classification
            confidence_threshold: Threshold for confident classifications
        """
        self.severity_classifier = severity_classifier or ThresholdSeverityClassifier()
        self.enable_hierarchical = enable_hierarchical
        self.enable_multiclass = enable_multiclass
        self.confidence_threshold = confidence_threshold

    def classify_anomaly(
        self,
        anomaly_score: float,
        detector: Detector,
        feature_data: dict[str, Any] | None = None,
        context_data: dict[str, Any] | None = None,
    ) -> AdvancedAnomalyClassification:
        """Perform comprehensive anomaly classification.
        
        Args:
            anomaly_score: Primary anomaly score
            detector: Detector that produced the score
            feature_data: Feature-level data for analysis
            context_data: Additional context information
            
        Returns:
            Advanced anomaly classification result
        """
        feature_data = feature_data or {}
        context_data = context_data or {}
        
        # Basic classification
        basic_classification = self._create_basic_classification(
            anomaly_score, detector, feature_data
        )
        
        # Severity classification
        severity = self.severity_classifier.classify_single(anomaly_score)
        
        # Hierarchical classification
        hierarchical_classification = None
        if self.enable_hierarchical:
            hierarchical_classification = self._create_hierarchical_classification(
                basic_classification, detector, feature_data
            )
        
        # Multi-class classification
        multi_class_classification = None
        if self.enable_multiclass:
            multi_class_classification = self._create_multiclass_classification(
                basic_classification, detector, feature_data
            )
        
        # Context classifications
        temporal_context = self._extract_temporal_context(context_data)
        spatial_context = self._extract_spatial_context(context_data)
        general_context = self._extract_general_context(context_data)
        
        return AdvancedAnomalyClassification(
            basic_classification=basic_classification,
            hierarchical_classification=hierarchical_classification,
            multi_class_classification=multi_class_classification,
            severity_classification=severity,
            context_classification=general_context,
            temporal_classification=temporal_context,
            spatial_classification=spatial_context,
        )

    def _create_basic_classification(
        self,
        anomaly_score: float,
        detector: Detector,
        feature_data: dict[str, Any],
    ) -> ClassificationResult:
        """Create basic classification result."""
        # Determine predicted class based on score and threshold
        predicted_class = "anomaly" if anomaly_score > self.confidence_threshold else "normal"
        
        # Calculate feature contributions if available
        feature_contributions = self._calculate_feature_contributions(feature_data)
        
        # Determine classification method from detector metadata
        classification_method = self._determine_classification_method(detector)
        
        # Create probability distribution
        prob_distribution = {
            "anomaly": anomaly_score,
            "normal": 1.0 - anomaly_score,
        }
        
        # Additional metadata
        metadata = {
            "detector_name": detector.name,
            "algorithm": detector.algorithm_name,
            "contamination_rate": detector.contamination_rate.value,
            "threshold_used": self.confidence_threshold,
        }
        
        return ClassificationResult.from_confidence_score(predicted_class, anomaly_score).replace(
            probability_distribution=prob_distribution,
            feature_contributions=feature_contributions,
            classification_method=classification_method,
            metadata=metadata,
        )

    def _create_hierarchical_classification(
        self,
        basic_classification: ClassificationResult,
        detector: Detector,
        feature_data: dict[str, Any],
    ) -> HierarchicalClassification | None:
        """Create hierarchical classification based on algorithm and features."""
        if basic_classification.predicted_class == "normal":
            return None
        
        # Primary category based on detector algorithm
        primary_category = self._get_primary_category_from_detector(detector)
        
        # Secondary category based on anomaly characteristics
        secondary_category = self._determine_secondary_category(
            basic_classification, feature_data
        )
        
        # Tertiary category for more specific classification
        tertiary_category = self._determine_tertiary_category(
            basic_classification, feature_data
        )
        
        # Sub-type based on detailed analysis
        sub_type = self._determine_anomaly_subtype(basic_classification, feature_data)
        
        # Confidence scores for each level
        confidence_scores = {
            "primary": basic_classification.confidence_score,
            "secondary": basic_classification.confidence_score * 0.8,
            "tertiary": basic_classification.confidence_score * 0.6,
            "subtype": basic_classification.confidence_score * 0.4,
        }
        
        return HierarchicalClassification(
            primary_category=primary_category,
            secondary_category=secondary_category,
            tertiary_category=tertiary_category,
            sub_type=sub_type,
            confidence_scores=confidence_scores,
        )

    def _create_multiclass_classification(
        self,
        basic_classification: ClassificationResult,
        detector: Detector,
        feature_data: dict[str, Any],
    ) -> MultiClassClassification | None:
        """Create multi-class classification with alternative results."""
        if basic_classification.predicted_class == "normal":
            return None
        
        # Generate alternative classifications
        alternative_results = self._generate_alternative_classifications(
            basic_classification, detector, feature_data
        )
        
        return MultiClassClassification(
            primary_result=basic_classification,
            alternative_results=alternative_results,
            classification_threshold=self.confidence_threshold,
            multi_class_strategy="one_vs_rest",
        )

    def _generate_alternative_classifications(
        self,
        basic_classification: ClassificationResult,
        detector: Detector,
        feature_data: dict[str, Any],
    ) -> list[ClassificationResult]:
        """Generate alternative classification results."""
        alternatives = []
        
        # Different anomaly types based on confidence ranges
        confidence_score = basic_classification.confidence_score
        
        if confidence_score > 0.7:
            # High confidence anomaly - consider specific types
            alt_classes = ["severe_anomaly", "outlier", "extreme_value"]
        elif confidence_score > 0.5:
            # Medium confidence - consider broader types
            alt_classes = ["moderate_anomaly", "potential_outlier", "suspicious"]
        else:
            # Lower confidence - consider edge cases
            alt_classes = ["weak_anomaly", "borderline", "uncertain"]
        
        for alt_class in alt_classes:
            # Adjust confidence for alternative classifications
            alt_confidence = confidence_score * np.random.uniform(0.7, 0.9)
            
            alt_result = ClassificationResult.from_confidence_score(
                alt_class, alt_confidence
            ).replace(
                classification_method=basic_classification.classification_method,
                metadata={
                    **basic_classification.metadata,
                    "alternative_classification": True,
                    "base_confidence": confidence_score,
                }
            )
            alternatives.append(alt_result)
        
        return alternatives[:3]  # Limit to top 3 alternatives

    def _calculate_feature_contributions(
        self, feature_data: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate feature contributions to anomaly score."""
        contributions = {}
        
        if not feature_data:
            return contributions
        
        # Simple heuristic: features with higher variance contribute more
        total_variance = 0.0
        feature_variances = {}
        
        for feature_name, feature_value in feature_data.items():
            if isinstance(feature_value, (int, float)):
                # Simple variance estimation (in real implementation, use historical data)
                variance = abs(feature_value) * 0.1
                feature_variances[feature_name] = variance
                total_variance += variance
        
        # Normalize contributions
        if total_variance > 0:
            for feature_name, variance in feature_variances.items():
                contributions[feature_name] = variance / total_variance
        
        return contributions

    def _determine_classification_method(self, detector: Detector) -> ClassificationMethod:
        """Determine classification method from detector type."""
        algorithm = detector.algorithm_name.lower()
        
        if "ensemble" in algorithm:
            return ClassificationMethod.ENSEMBLE
        elif "supervised" in algorithm or "svm" in algorithm:
            return ClassificationMethod.SUPERVISED
        elif "semi" in algorithm:
            return ClassificationMethod.SEMI_SUPERVISED
        elif "hybrid" in algorithm:
            return ClassificationMethod.HYBRID
        else:
            return ClassificationMethod.UNSUPERVISED

    def _get_primary_category_from_detector(self, detector: Detector) -> str:
        """Get primary category based on detector algorithm."""
        algorithm = detector.algorithm_name.lower()
        
        if any(term in algorithm for term in ["isolation", "tree"]):
            return "ensemble"
        elif any(term in algorithm for term in ["cluster", "kmeans", "dbscan"]):
            return "clustering"
        elif any(term in algorithm for term in ["distance", "knn", "nearest"]):
            return "distance"
        elif any(term in algorithm for term in ["density", "lof"]):
            return "density"
        elif any(term in algorithm for term in ["neural", "autoencoder", "deep"]):
            return "neural"
        elif any(term in algorithm for term in ["statistical", "gaussian", "zscore"]):
            return "statistical"
        else:
            return "threshold"

    def _determine_secondary_category(
        self, basic_classification: ClassificationResult, feature_data: dict[str, Any]
    ) -> str | None:
        """Determine secondary category based on anomaly characteristics."""
        if basic_classification.confidence_score > 0.8:
            return "high_confidence"
        elif basic_classification.confidence_score > 0.6:
            return "medium_confidence"
        elif basic_classification.confidence_score > 0.4:
            return "low_confidence"
        else:
            return "very_low_confidence"

    def _determine_tertiary_category(
        self, basic_classification: ClassificationResult, feature_data: dict[str, Any]
    ) -> str | None:
        """Determine tertiary category for specific classification."""
        if not feature_data:
            return None
        
        num_features = len(feature_data)
        if num_features == 1:
            return "univariate"
        elif num_features <= 5:
            return "low_dimensional"
        elif num_features <= 20:
            return "medium_dimensional"
        else:
            return "high_dimensional"

    def _determine_anomaly_subtype(
        self, basic_classification: ClassificationResult, feature_data: dict[str, Any]
    ) -> AnomalySubType | None:
        """Determine specific anomaly subtype."""
        confidence = basic_classification.confidence_score
        
        if confidence > 0.9:
            return AnomalySubType.EXTREME_VALUE
        elif confidence > 0.7:
            return AnomalySubType.OUTLIER
        elif confidence > 0.5:
            return AnomalySubType.NOVELTY
        else:
            return AnomalySubType.CONDITIONAL

    def _extract_temporal_context(self, context_data: dict[str, Any]) -> dict[str, Any]:
        """Extract temporal classification context."""
        temporal_context = {}
        
        if "timestamp" in context_data:
            temporal_context["has_timestamp"] = True
            temporal_context["timestamp"] = context_data["timestamp"]
        
        if "time_series" in context_data:
            temporal_context["is_time_series"] = True
            temporal_context["series_length"] = len(context_data["time_series"])
        
        if "seasonality" in context_data:
            temporal_context["seasonality_detected"] = context_data["seasonality"]
        
        return temporal_context

    def _extract_spatial_context(self, context_data: dict[str, Any]) -> dict[str, Any]:
        """Extract spatial classification context."""
        spatial_context = {}
        
        if "location" in context_data:
            spatial_context["has_location"] = True
            spatial_context["location"] = context_data["location"]
        
        if "coordinates" in context_data:
            spatial_context["has_coordinates"] = True
            spatial_context["coordinates"] = context_data["coordinates"]
        
        if "region" in context_data:
            spatial_context["region"] = context_data["region"]
        
        return spatial_context

    def _extract_general_context(self, context_data: dict[str, Any]) -> dict[str, Any]:
        """Extract general classification context."""
        general_context = {}
        
        # Business context
        if "business_unit" in context_data:
            general_context["business_unit"] = context_data["business_unit"]
        
        if "data_source" in context_data:
            general_context["data_source"] = context_data["data_source"]
        
        if "priority" in context_data:
            general_context["priority"] = context_data["priority"]
        
        # Technical context
        if "model_version" in context_data:
            general_context["model_version"] = context_data["model_version"]
        
        if "environment" in context_data:
            general_context["environment"] = context_data["environment"]
        
        return general_context

    def classify_batch(
        self,
        anomaly_scores: list[float],
        detector: Detector,
        feature_data_batch: list[dict[str, Any]] | None = None,
        context_data_batch: list[dict[str, Any]] | None = None,
    ) -> list[AdvancedAnomalyClassification]:
        """Classify a batch of anomaly scores."""
        if not anomaly_scores:
            return []
        
        feature_data_batch = feature_data_batch or [{}] * len(anomaly_scores)
        context_data_batch = context_data_batch or [{}] * len(anomaly_scores)
        
        results = []
        for i, score in enumerate(anomaly_scores):
            feature_data = feature_data_batch[i] if i < len(feature_data_batch) else {}
            context_data = context_data_batch[i] if i < len(context_data_batch) else {}
            
            classification = self.classify_anomaly(
                score, detector, feature_data, context_data
            )
            results.append(classification)
        
        return results

    def get_classification_summary(
        self, classifications: list[AdvancedAnomalyClassification]
    ) -> dict[str, Any]:
        """Get summary statistics for a batch of classifications."""
        if not classifications:
            return {}
        
        summary = {
            "total_classifications": len(classifications),
            "anomaly_count": sum(
                1 for c in classifications 
                if c.get_primary_class() == "anomaly"
            ),
            "severity_distribution": {},
            "confidence_distribution": {},
            "hierarchical_depth_distribution": {},
            "multiclass_ambiguous_count": 0,
        }
        
        # Severity distribution
        for classification in classifications:
            severity = classification.severity_classification
            summary["severity_distribution"][severity] = (
                summary["severity_distribution"].get(severity, 0) + 1
            )
        
        # Confidence distribution
        for classification in classifications:
            confidence_level = classification.basic_classification.confidence_level.value
            summary["confidence_distribution"][confidence_level] = (
                summary["confidence_distribution"].get(confidence_level, 0) + 1
            )
        
        # Hierarchical depth distribution
        for classification in classifications:
            if classification.is_hierarchical():
                depth = classification.hierarchical_classification.get_hierarchy_depth()
                summary["hierarchical_depth_distribution"][depth] = (
                    summary["hierarchical_depth_distribution"].get(depth, 0) + 1
                )
        
        # Ambiguous multi-class count
        summary["multiclass_ambiguous_count"] = sum(
            1 for c in classifications
            if c.is_multi_class() and c.multi_class_classification.has_ambiguous_classification()
        )
        
        return summary