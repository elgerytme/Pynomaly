"""Enhanced detection service with advanced anomaly classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from packages.core.domain.entities.anomaly import Anomaly
from monorepo.domain.value_objects import AnomalyScore

# Import advanced classification components
try:
    from monorepo.domain.services.advanced_classification_service import (
        AdvancedClassificationService,
    )
    from monorepo.domain.services.threshold_severity_classifier import (
        ThresholdSeverityClassifier,
    )
    from monorepo.domain.value_objects.anomaly_classification import (
        AdvancedAnomalyClassification,
        ClassificationResult,
        ConfidenceLevel,
    )
    from monorepo.domain.value_objects.anomaly_type import AnomalyType
    from monorepo.domain.value_objects.severity_score import SeverityLevel
except ImportError:
    # Fallback for development
    AdvancedClassificationService = None
    ThresholdSeverityClassifier = None
    AdvancedAnomalyClassification = None
    ClassificationResult = None
    ConfidenceLevel = None
    AnomalyType = None
    SeverityLevel = None


class EnhancedDetectionService:
    """Enhanced detection service with advanced anomaly classification."""

    def __init__(
        self,
        classification_service: Optional[AdvancedClassificationService] = None,
        severity_classifier: Optional[ThresholdSeverityClassifier] = None,
        enable_advanced_classification: bool = True,
    ):
        """Initialize the enhanced detection service.
        
        Args:
            classification_service: Advanced classification service instance
            severity_classifier: Threshold-based severity classifier
            enable_advanced_classification: Whether to use advanced classification
        """
        self.classification_service = classification_service or (
            AdvancedClassificationService() if AdvancedClassificationService else None
        )
        self.severity_classifier = severity_classifier or (
            ThresholdSeverityClassifier() if ThresholdSeverityClassifier else None
        )
        self.enable_advanced_classification = enable_advanced_classification

    def enhance_anomaly_with_classification(
        self, anomaly: Anomaly, data_context: Optional[Dict[str, Any]] = None
    ) -> Anomaly:
        """Enhance an anomaly with advanced classification.
        
        Args:
            anomaly: The anomaly to enhance
            data_context: Additional context for classification
            
        Returns:
            Enhanced anomaly with advanced classification
        """
        if not self.enable_advanced_classification or not self.classification_service:
            return anomaly

        try:
            # Get the score value
            score_value = (
                anomaly.score.value if isinstance(anomaly.score, AnomalyScore) 
                else anomaly.score
            )

            # Perform advanced classification
            classification_result = self.classification_service.classify_anomaly(
                data_point=anomaly.data_point,
                score=score_value,
                detector_name=anomaly.detector_name,
                context=data_context or {},
            )

            # Set the advanced classification
            anomaly.classification = classification_result

            # Extract and set individual classification components
            if classification_result.basic_classification:
                # Map confidence score to confidence level
                confidence_score = classification_result.basic_classification.confidence_score
                if ConfidenceLevel:
                    if confidence_score >= 0.9:
                        anomaly.confidence_level = ConfidenceLevel.VERY_HIGH
                    elif confidence_score >= 0.7:
                        anomaly.confidence_level = ConfidenceLevel.HIGH
                    elif confidence_score >= 0.5:
                        anomaly.confidence_level = ConfidenceLevel.MEDIUM
                    elif confidence_score >= 0.3:
                        anomaly.confidence_level = ConfidenceLevel.LOW
                    else:
                        anomaly.confidence_level = ConfidenceLevel.VERY_LOW

            # Set severity level using severity classifier
            if self.severity_classifier and SeverityLevel:
                severity_score = self.severity_classifier.classify_severity(score_value)
                anomaly.severity_level = severity_score.level

            # Set anomaly type from hierarchical classification
            if (
                classification_result.hierarchical_classification
                and AnomalyType
            ):
                primary_category = classification_result.hierarchical_classification.primary_category
                # Map primary category to anomaly type
                anomaly_type_mapping = {
                    "point": AnomalyType.POINT,
                    "contextual": AnomalyType.CONTEXTUAL,
                    "collective": AnomalyType.COLLECTIVE,
                    "global": AnomalyType.GLOBAL,
                    "local": AnomalyType.LOCAL,
                }
                anomaly.anomaly_type = anomaly_type_mapping.get(
                    primary_category.lower(), AnomalyType.POINT
                )

            # Add classification metadata
            anomaly.add_metadata("classification_method", "advanced")
            anomaly.add_metadata("classification_timestamp", anomaly.timestamp.isoformat())
            
            if classification_result.hierarchical_classification:
                anomaly.add_metadata(
                    "hierarchy_path", 
                    classification_result.hierarchical_classification.get_full_path()
                )

        except Exception as e:
            # Log the error but don't fail the detection
            anomaly.add_metadata("classification_error", str(e))
            anomaly.add_metadata("classification_method", "fallback")

        return anomaly

    def detect_with_advanced_classification(
        self,
        data_points: List[Dict[str, Any]],
        detector_name: str,
        scores: List[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Anomaly]:
        """Detect anomalies with advanced classification.
        
        Args:
            data_points: List of data points to analyze
            detector_name: Name of the detector
            scores: Anomaly scores for each data point
            context: Additional context for classification
            
        Returns:
            List of enhanced anomalies with advanced classification
        """
        anomalies = []
        
        for i, (data_point, score) in enumerate(zip(data_points, scores)):
            # Create basic anomaly
            anomaly = Anomaly(
                score=score,
                data_point=data_point,
                detector_name=detector_name,
            )
            
            # Enhance with advanced classification
            enhanced_anomaly = self.enhance_anomaly_with_classification(
                anomaly, context
            )
            
            anomalies.append(enhanced_anomaly)
        
        return anomalies

    def batch_classify_anomalies(
        self, anomalies: List[Anomaly], context: Optional[Dict[str, Any]] = None
    ) -> List[Anomaly]:
        """Classify a batch of anomalies with advanced classification.
        
        Args:
            anomalies: List of anomalies to classify
            context: Additional context for classification
            
        Returns:
            List of enhanced anomalies
        """
        enhanced_anomalies = []
        
        for anomaly in anomalies:
            enhanced_anomaly = self.enhance_anomaly_with_classification(
                anomaly, context
            )
            enhanced_anomalies.append(enhanced_anomaly)
        
        return enhanced_anomalies

    def get_classification_summary(self, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """Get a summary of classifications for a list of anomalies.
        
        Args:
            anomalies: List of anomalies to summarize
            
        Returns:
            Classification summary statistics
        """
        if not anomalies:
            return {"total": 0, "summary": {}}

        # Count by severity
        severity_counts = {}
        # Count by anomaly type
        type_counts = {}
        # Count by confidence level
        confidence_counts = {}
        # Count advanced vs basic classification
        classification_type_counts = {"advanced": 0, "basic": 0}

        for anomaly in anomalies:
            # Count severity
            severity = anomaly.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count anomaly type
            anomaly_type = anomaly.get_anomaly_type()
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
            
            # Count confidence level
            if anomaly.confidence_level:
                confidence_level = anomaly.confidence_level.value.lower()
                confidence_counts[confidence_level] = confidence_counts.get(confidence_level, 0) + 1
            
            # Count classification type
            if anomaly.has_advanced_classification():
                classification_type_counts["advanced"] += 1
            else:
                classification_type_counts["basic"] += 1

        # Calculate confidence statistics
        confidence_scores = [anomaly.get_confidence_score() for anomaly in anomalies]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            "total": len(anomalies),
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "confidence_distribution": confidence_counts,
            "classification_type_distribution": classification_type_counts,
            "average_confidence": avg_confidence,
            "high_confidence_count": sum(1 for a in anomalies if a.is_highly_confident()),
            "critical_severity_count": sum(1 for a in anomalies if a.is_critical_severity()),
        }

    def filter_by_confidence(
        self, anomalies: List[Anomaly], min_confidence: float = 0.5
    ) -> List[Anomaly]:
        """Filter anomalies by minimum confidence score.
        
        Args:
            anomalies: List of anomalies to filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of anomalies
        """
        return [
            anomaly for anomaly in anomalies
            if anomaly.get_confidence_score() >= min_confidence
        ]

    def filter_by_severity(
        self, anomalies: List[Anomaly], min_severity: str = "medium"
    ) -> List[Anomaly]:
        """Filter anomalies by minimum severity level.
        
        Args:
            anomalies: List of anomalies to filter
            min_severity: Minimum severity level ("low", "medium", "high", "critical")
            
        Returns:
            Filtered list of anomalies
        """
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_severity_value = severity_order.get(min_severity, 1)
        
        return [
            anomaly for anomaly in anomalies
            if severity_order.get(anomaly.severity, 0) >= min_severity_value
        ]