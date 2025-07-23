"""Advanced explanation analyzers for anomaly detection results."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .....domain.entities.detection_result import DetectionResult
from .....infrastructure.logging import get_logger

logger = get_logger(__name__)


class ExplanationMethod(Enum):
    """Available explanation methods."""
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP_VALUES = "shap_values"
    LIME_EXPLANATION = "lime_explanation"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    DISTANCE_BASED = "distance_based"
    OUTLIER_SCORE_BREAKDOWN = "outlier_score_breakdown"


@dataclass
class FeatureExplanation:
    """Explanation for a single feature."""
    feature_name: str
    feature_index: int
    importance_score: float
    contribution: float  # Positive = anomalous, Negative = normal
    confidence: float
    percentile: Optional[float] = None
    statistical_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationResult:
    """Result of explanation analysis."""
    method: ExplanationMethod
    timestamp: datetime
    sample_explanations: List[List[FeatureExplanation]]  # One list per sample
    global_feature_importance: List[FeatureExplanation]
    summary_statistics: Dict[str, Any]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_features(self, n: int = 5, sample_index: int = 0) -> List[FeatureExplanation]:
        """Get top N most important features for a specific sample."""
        if sample_index >= len(self.sample_explanations):
            return []
        
        explanations = self.sample_explanations[sample_index]
        return sorted(explanations, key=lambda x: abs(x.importance_score), reverse=True)[:n]
    
    def get_global_top_features(self, n: int = 5) -> List[FeatureExplanation]:
        """Get top N most important features globally."""
        return sorted(self.global_feature_importance, 
                     key=lambda x: abs(x.importance_score), reverse=True)[:n]


class ExplanationAnalyzers:
    """Advanced explanation analysis service for anomaly detection results."""
    
    def __init__(
        self,
        default_method: ExplanationMethod = ExplanationMethod.FEATURE_IMPORTANCE,
        confidence_threshold: float = 0.7,
        max_features_to_explain: int = 20
    ):
        """Initialize explanation analyzers.
        
        Args:
            default_method: Default explanation method to use
            confidence_threshold: Minimum confidence for explanations
            max_features_to_explain: Maximum number of features to analyze
        """
        self.default_method = default_method
        self.confidence_threshold = confidence_threshold
        self.max_features_to_explain = max_features_to_explain
        
        logger.info("ExplanationAnalyzers initialized",
                   default_method=default_method.value,
                   confidence_threshold=confidence_threshold)
    
    def explain_predictions(
        self,
        detection_result: DetectionResult,
        input_data: npt.NDArray[np.floating],
        method: Optional[ExplanationMethod] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> ExplanationResult:
        """Generate explanations for anomaly detection predictions.
        
        Args:
            detection_result: Results from anomaly detection
            input_data: Original input data
            method: Explanation method to use
            feature_names: Names of features (optional)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Explanation results
        """
        method = method or self.default_method
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(input_data.shape[1])]
        
        logger.info("Generating explanations",
                   method=method.value,
                   samples=input_data.shape[0],
                   features=input_data.shape[1],
                   anomalies=detection_result.anomaly_count)
        
        try:
            if method == ExplanationMethod.FEATURE_IMPORTANCE:
                return self._feature_importance_explanation(
                    detection_result, input_data, feature_names, **kwargs
                )
            elif method == ExplanationMethod.STATISTICAL_ANALYSIS:
                return self._statistical_analysis_explanation(
                    detection_result, input_data, feature_names, **kwargs
                )
            elif method == ExplanationMethod.DISTANCE_BASED:
                return self._distance_based_explanation(
                    detection_result, input_data, feature_names, **kwargs
                )
            elif method == ExplanationMethod.OUTLIER_SCORE_BREAKDOWN:
                return self._outlier_score_breakdown(
                    detection_result, input_data, feature_names, **kwargs
                )
            elif method == ExplanationMethod.SHAP_VALUES:
                return self._shap_explanation(
                    detection_result, input_data, feature_names, **kwargs
                )
            elif method == ExplanationMethod.LIME_EXPLANATION:
                return self._lime_explanation(
                    detection_result, input_data, feature_names, **kwargs
                )
            elif method == ExplanationMethod.PERMUTATION_IMPORTANCE:
                return self._permutation_importance_explanation(
                    detection_result, input_data, feature_names, **kwargs
                )
            else:
                raise ValueError(f"Unknown explanation method: {method}")
                
        except Exception as e:
            logger.error("Explanation generation failed",
                        method=method.value,
                        error=str(e))
            # Return fallback explanation
            return self._fallback_explanation(detection_result, input_data, feature_names)
    
    def _feature_importance_explanation(
        self,
        detection_result: DetectionResult,
        input_data: npt.NDArray[np.floating],
        feature_names: List[str],
        **kwargs
    ) -> ExplanationResult:
        """Generate feature importance based explanations."""
        
        # Calculate feature statistics for anomalies vs normal samples
        anomaly_mask = detection_result.predictions == -1
        normal_mask = detection_result.predictions == 1
        
        if not np.any(anomaly_mask):
            logger.warning("No anomalies found for explanation")
            return self._fallback_explanation(detection_result, input_data, feature_names)
        
        anomaly_data = input_data[anomaly_mask]
        normal_data = input_data[normal_mask] if np.any(normal_mask) else input_data
        
        # Calculate importance scores based on statistical differences
        sample_explanations = []
        feature_importance_scores = []
        
        for i in range(len(feature_names)):
            anomaly_values = anomaly_data[:, i]
            normal_values = normal_data[:, i]
            
            # Calculate statistical difference
            anomaly_mean = np.mean(anomaly_values)
            normal_mean = np.mean(normal_values)
            
            # Use standard deviation to normalize importance
            combined_std = np.std(input_data[:, i])
            if combined_std > 0:
                importance = abs(anomaly_mean - normal_mean) / combined_std
            else:
                importance = 0.0
            
            feature_importance_scores.append(importance)
        
        # Generate per-sample explanations
        for sample_idx in range(input_data.shape[0]):
            sample_features = []
            sample_data = input_data[sample_idx]
            
            for i, feature_name in enumerate(feature_names):
                # Calculate contribution for this sample
                feature_value = sample_data[i]
                normal_mean = np.mean(normal_data[:, i])
                
                # Contribution is how much this feature deviates from normal
                contribution = (feature_value - normal_mean)
                if np.std(normal_data[:, i]) > 0:
                    contribution /= np.std(normal_data[:, i])
                
                # Calculate percentile
                percentile = (np.sum(normal_data[:, i] <= feature_value) / len(normal_data)) * 100
                
                explanation = FeatureExplanation(
                    feature_name=feature_name,
                    feature_index=i,
                    importance_score=feature_importance_scores[i],
                    contribution=contribution,
                    confidence=min(1.0, feature_importance_scores[i]),
                    percentile=percentile,
                    statistical_info={
                        "feature_value": feature_value,
                        "normal_mean": normal_mean,
                        "normal_std": np.std(normal_data[:, i]),
                        "anomaly_mean": np.mean(anomaly_data[:, i]) if len(anomaly_data) > 0 else normal_mean
                    }
                )
                sample_features.append(explanation)
            
            sample_explanations.append(sample_features)
        
        # Generate global feature importance
        global_importance = []
        for i, feature_name in enumerate(feature_names):
            global_importance.append(FeatureExplanation(
                feature_name=feature_name,
                feature_index=i,
                importance_score=feature_importance_scores[i],
                contribution=0.0,  # Global doesn't have per-sample contribution
                confidence=min(1.0, feature_importance_scores[i]),
                statistical_info={
                    "importance_rank": sorted(range(len(feature_importance_scores)), 
                                            key=lambda x: feature_importance_scores[x], reverse=True).index(i) + 1
                }
            ))
        
        return ExplanationResult(
            method=ExplanationMethod.FEATURE_IMPORTANCE,
            timestamp=datetime.utcnow(),
            sample_explanations=sample_explanations,
            global_feature_importance=global_importance,
            summary_statistics={
                "total_samples": len(input_data),
                "anomaly_samples": np.sum(anomaly_mask),
                "normal_samples": np.sum(normal_mask),
                "top_feature": feature_names[np.argmax(feature_importance_scores)],
                "max_importance": np.max(feature_importance_scores),
                "mean_importance": np.mean(feature_importance_scores)
            },
            confidence_score=np.mean([min(1.0, score) for score in feature_importance_scores]),
            metadata={
                "algorithm": detection_result.algorithm,
                "method_params": kwargs
            }
        )
    
    def _statistical_analysis_explanation(
        self,
        detection_result: DetectionResult, 
        input_data: npt.NDArray[np.floating],
        feature_names: List[str],
        **kwargs
    ) -> ExplanationResult:
        """Generate statistical analysis based explanations."""
        
        # Perform comprehensive statistical analysis
        sample_explanations = []
        global_stats = []
        
        # Calculate statistical measures for each feature
        for i, feature_name in enumerate(feature_names):
            feature_data = input_data[:, i]
            
            # Basic statistics
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)
            median_val = np.median(feature_data)
            q25 = np.percentile(feature_data, 25)
            q75 = np.percentile(feature_data, 75)
            iqr = q75 - q25
            
            # Calculate outlier boundaries
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Importance based on how often this feature creates outliers
            outlier_count = np.sum((feature_data < lower_bound) | (feature_data > upper_bound))
            importance = outlier_count / len(feature_data) if len(feature_data) > 0 else 0
            
            global_stats.append(FeatureExplanation(
                feature_name=feature_name,
                feature_index=i,
                importance_score=importance,
                contribution=0.0,
                confidence=0.8,
                statistical_info={
                    "mean": mean_val,
                    "std": std_val,
                    "median": median_val,
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_rate": importance
                }
            ))
        
        # Generate per-sample explanations
        for sample_idx in range(input_data.shape[0]):
            sample_features = []
            sample_data = input_data[sample_idx]
            
            for i, feature_name in enumerate(feature_names):
                feature_value = sample_data[i]
                stats = global_stats[i].statistical_info
                
                # Calculate how anomalous this value is
                z_score = (feature_value - stats["mean"]) / (stats["std"] + 1e-8)
                is_outlier = (feature_value < stats["lower_bound"]) or (feature_value > stats["upper_bound"])
                
                # Calculate percentile
                percentile = (np.sum(input_data[:, i] <= feature_value) / len(input_data)) * 100
                
                explanation = FeatureExplanation(
                    feature_name=feature_name,
                    feature_index=i,
                    importance_score=abs(z_score),
                    contribution=z_score,
                    confidence=0.8,
                    percentile=percentile,
                    statistical_info={
                        "feature_value": feature_value,
                        "z_score": z_score,
                        "is_outlier": is_outlier,
                        "distance_from_median": abs(feature_value - stats["median"])
                    }
                )
                sample_features.append(explanation)
            
            sample_explanations.append(sample_features)
        
        return ExplanationResult(
            method=ExplanationMethod.STATISTICAL_ANALYSIS,
            timestamp=datetime.utcnow(),
            sample_explanations=sample_explanations,
            global_feature_importance=global_stats,
            summary_statistics={
                "total_samples": len(input_data),
                "features_analyzed": len(feature_names),
                "mean_outlier_rate": np.mean([s.importance_score for s in global_stats])
            },
            confidence_score=0.8,
            metadata={"algorithm": detection_result.algorithm}
        )
    
    def _distance_based_explanation(
        self,
        detection_result: DetectionResult,
        input_data: npt.NDArray[np.floating], 
        feature_names: List[str],
        **kwargs
    ) -> ExplanationResult:
        """Generate distance-based explanations."""
        
        # Calculate centroid of normal samples
        normal_mask = detection_result.predictions == 1
        if np.any(normal_mask):
            normal_centroid = np.mean(input_data[normal_mask], axis=0)
        else:
            normal_centroid = np.mean(input_data, axis=0)
        
        sample_explanations = []
        feature_distances = []
        
        # Calculate feature-wise distances for global importance
        for i in range(len(feature_names)):
            feature_distances.append(np.std(input_data[:, i]))
        
        # Generate per-sample explanations
        for sample_idx in range(input_data.shape[0]):
            sample_features = []
            sample_data = input_data[sample_idx]
            
            # Calculate distance from normal centroid for each feature
            for i, feature_name in enumerate(feature_names):
                distance = abs(sample_data[i] - normal_centroid[i])
                normalized_distance = distance / (feature_distances[i] + 1e-8)
                
                explanation = FeatureExplanation(
                    feature_name=feature_name,
                    feature_index=i,
                    importance_score=normalized_distance,
                    contribution=sample_data[i] - normal_centroid[i],
                    confidence=0.7,
                    statistical_info={
                        "feature_value": sample_data[i],
                        "normal_centroid": normal_centroid[i],
                        "distance": distance,
                        "normalized_distance": normalized_distance
                    }
                )
                sample_features.append(explanation)
            
            sample_explanations.append(sample_features)
        
        # Global feature importance based on average distances
        global_importance = []
        for i, feature_name in enumerate(feature_names):
            avg_distance = np.mean([abs(sample[i] - normal_centroid[i]) for sample in input_data])
            global_importance.append(FeatureExplanation(
                feature_name=feature_name,
                feature_index=i,
                importance_score=avg_distance / (feature_distances[i] + 1e-8),
                contribution=0.0,
                confidence=0.7
            ))
        
        return ExplanationResult(
            method=ExplanationMethod.DISTANCE_BASED,
            timestamp=datetime.utcnow(),
            sample_explanations=sample_explanations,
            global_feature_importance=global_importance,
            summary_statistics={
                "normal_samples": np.sum(normal_mask),
                "centroid": normal_centroid.tolist()
            },
            confidence_score=0.7,
            metadata={"algorithm": detection_result.algorithm}
        )
    
    def _outlier_score_breakdown(
        self,
        detection_result: DetectionResult,
        input_data: npt.NDArray[np.floating],
        feature_names: List[str],
        **kwargs
    ) -> ExplanationResult:
        """Break down outlier scores by feature contribution."""
        
        # Use confidence scores if available, otherwise use simple outlier analysis
        if detection_result.confidence_scores is not None:
            scores = detection_result.confidence_scores
        else:
            # Calculate simple outlier scores based on distance from median
            scores = np.zeros(len(input_data))
            for i in range(input_data.shape[1]):
                feature_median = np.median(input_data[:, i])
                feature_mad = np.median(np.abs(input_data[:, i] - feature_median))
                if feature_mad > 0:
                    scores += np.abs(input_data[:, i] - feature_median) / feature_mad
            scores = scores / input_data.shape[1]  # Normalize by number of features
        
        sample_explanations = []
        global_importance = []
        
        # Calculate feature contributions to outlier scores
        for i, feature_name in enumerate(feature_names):
            feature_data = input_data[:, i]
            feature_median = np.median(feature_data)
            feature_mad = np.median(np.abs(feature_data - feature_median))
            
            if feature_mad > 0:
                feature_outlier_scores = np.abs(feature_data - feature_median) / feature_mad
                importance = np.mean(feature_outlier_scores)
            else:
                importance = 0.0
            
            global_importance.append(FeatureExplanation(
                feature_name=feature_name,
                feature_index=i,
                importance_score=importance,
                contribution=0.0,
                confidence=0.75
            ))
        
        # Generate per-sample explanations
        for sample_idx in range(input_data.shape[0]):
            sample_features = []
            sample_score = scores[sample_idx]
            
            for i, feature_name in enumerate(feature_names):
                feature_value = input_data[sample_idx, i]
                feature_median = np.median(input_data[:, i])
                feature_mad = np.median(np.abs(input_data[:, i] - feature_median))
                
                if feature_mad > 0:
                    feature_contribution = abs(feature_value - feature_median) / feature_mad
                    # Normalize by total score to get relative contribution
                    relative_contribution = feature_contribution / (sample_score + 1e-8)
                else:
                    feature_contribution = 0.0
                    relative_contribution = 0.0
                
                explanation = FeatureExplanation(
                    feature_name=feature_name,
                    feature_index=i,
                    importance_score=feature_contribution,
                    contribution=relative_contribution,
                    confidence=0.75,
                    statistical_info={
                        "feature_value": feature_value,
                        "median": feature_median,
                        "mad": feature_mad,
                        "total_sample_score": sample_score
                    }
                )
                sample_features.append(explanation)
            
            sample_explanations.append(sample_features)
        
        return ExplanationResult(
            method=ExplanationMethod.OUTLIER_SCORE_BREAKDOWN,
            timestamp=datetime.utcnow(),
            sample_explanations=sample_explanations,
            global_feature_importance=global_importance,
            summary_statistics={
                "mean_outlier_score": np.mean(scores),
                "max_outlier_score": np.max(scores),
                "has_confidence_scores": detection_result.confidence_scores is not None
            },
            confidence_score=0.75,
            metadata={"algorithm": detection_result.algorithm}
        )
    
    def _shap_explanation(
        self,
        detection_result: DetectionResult,
        input_data: npt.NDArray[np.floating],
        feature_names: List[str],
        **kwargs
    ) -> ExplanationResult:
        """Generate SHAP-based explanations (requires shap library)."""
        try:
            import shap
            logger.info("Using SHAP for explanations")
            # SHAP implementation would go here
            # For now, fall back to feature importance
            logger.warning("SHAP explanation not fully implemented, falling back to feature importance")
            return self._feature_importance_explanation(detection_result, input_data, feature_names, **kwargs)
        except ImportError:
            logger.warning("SHAP library not available, falling back to feature importance")
            return self._feature_importance_explanation(detection_result, input_data, feature_names, **kwargs)
    
    def _lime_explanation(
        self,
        detection_result: DetectionResult,
        input_data: npt.NDArray[np.floating],
        feature_names: List[str],
        **kwargs
    ) -> ExplanationResult:
        """Generate LIME-based explanations (requires lime library)."""
        try:
            import lime
            logger.info("Using LIME for explanations")
            # LIME implementation would go here
            # For now, fall back to statistical analysis
            logger.warning("LIME explanation not fully implemented, falling back to statistical analysis")
            return self._statistical_analysis_explanation(detection_result, input_data, feature_names, **kwargs)
        except ImportError:
            logger.warning("LIME library not available, falling back to statistical analysis")
            return self._statistical_analysis_explanation(detection_result, input_data, feature_names, **kwargs)
    
    def _permutation_importance_explanation(
        self,
        detection_result: DetectionResult,
        input_data: npt.NDArray[np.floating],
        feature_names: List[str],
        **kwargs
    ) -> ExplanationResult:
        """Generate permutation importance based explanations."""
        # This would require re-running the model with permuted features
        # For now, fall back to distance-based explanation
        logger.warning("Permutation importance requires model re-evaluation, falling back to distance-based")
        return self._distance_based_explanation(detection_result, input_data, feature_names, **kwargs)
    
    def _fallback_explanation(
        self,
        detection_result: DetectionResult,
        input_data: npt.NDArray[np.floating],
        feature_names: List[str]
    ) -> ExplanationResult:
        """Generate fallback explanation when other methods fail."""
        
        logger.warning("Using fallback explanation method")
        
        # Simple fallback: equal importance for all features
        sample_explanations = []
        for sample_idx in range(input_data.shape[0]):
            sample_features = []
            for i, feature_name in enumerate(feature_names):
                explanation = FeatureExplanation(
                    feature_name=feature_name,
                    feature_index=i,
                    importance_score=1.0 / len(feature_names),
                    contribution=0.0,
                    confidence=0.5,
                    statistical_info={"feature_value": input_data[sample_idx, i]}
                )
                sample_features.append(explanation)
            sample_explanations.append(sample_features)
        
        global_importance = [
            FeatureExplanation(
                feature_name=name,
                feature_index=i,
                importance_score=1.0 / len(feature_names),
                contribution=0.0,
                confidence=0.5
            ) for i, name in enumerate(feature_names)
        ]
        
        return ExplanationResult(
            method=ExplanationMethod.FEATURE_IMPORTANCE,
            timestamp=datetime.utcnow(),
            sample_explanations=sample_explanations,
            global_feature_importance=global_importance,
            summary_statistics={
                "total_samples": len(input_data),
                "method": "fallback"
            },
            confidence_score=0.5,
            metadata={"algorithm": detection_result.algorithm, "fallback": True}
        )
    
    def get_available_methods(self) -> List[ExplanationMethod]:
        """Get list of available explanation methods."""
        return list(ExplanationMethod)
    
    def set_default_method(self, method: ExplanationMethod) -> None:
        """Set the default explanation method."""
        self.default_method = method
        logger.info("Default explanation method updated", method=method.value)
    
    def validate_explanation_result(self, result: ExplanationResult) -> bool:
        """Validate an explanation result."""
        try:
            if not result.sample_explanations:
                return False
            
            if not result.global_feature_importance:
                return False
            
            if result.confidence_score < 0 or result.confidence_score > 1:
                return False
            
            return True
        except Exception as e:
            logger.error("Explanation result validation failed", error=str(e))
            return False


# Global instance management
_explanation_analyzers: Optional[ExplanationAnalyzers] = None


def get_explanation_analyzers() -> ExplanationAnalyzers:
    """Get the global explanation analyzers instance."""
    global _explanation_analyzers
    
    if _explanation_analyzers is None:
        _explanation_analyzers = ExplanationAnalyzers()
    
    return _explanation_analyzers


def initialize_explanation_analyzers(
    default_method: ExplanationMethod = ExplanationMethod.FEATURE_IMPORTANCE,
    confidence_threshold: float = 0.7
) -> ExplanationAnalyzers:
    """Initialize the global explanation analyzers instance."""
    global _explanation_analyzers
    
    _explanation_analyzers = ExplanationAnalyzers(
        default_method=default_method,
        confidence_threshold=confidence_threshold
    )
    
    return _explanation_analyzers