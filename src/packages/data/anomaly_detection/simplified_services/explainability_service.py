"""Simplified explainability service for anomaly detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from .core_detection_service import DetectionResult


@dataclass
class FeatureImportance:
    """Feature importance for anomaly explanation."""
    feature_index: int
    feature_name: str
    importance_score: float
    contribution_type: str  # "positive", "negative", "neutral"


@dataclass
class AnomalyExplanation:
    """Explanation for detected anomalies."""
    sample_index: int
    anomaly_score: float
    is_anomaly: bool
    feature_importances: List[FeatureImportance]
    explanation_text: str
    confidence: float


class ExplainabilityService:
    """Simplified explainability service for anomaly detection.
    
    This service provides explanations for why certain data points are
    classified as anomalies, helping users understand the detection results.
    """

    def __init__(self):
        """Initialize explainability service."""
        pass

    def explain_anomalies(
        self,
        data: npt.NDArray[np.floating],
        result: DetectionResult,
        feature_names: Optional[List[str]] = None,
        n_features: int = 5
    ) -> List[AnomalyExplanation]:
        """Explain why specific data points were classified as anomalies.
        
        Args:
            data: Original input data
            result: Detection result from anomaly detection
            feature_names: Optional names for features
            n_features: Number of top features to include in explanation
            
        Returns:
            List of explanations for anomalous samples
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        explanations = []
        anomaly_indices = np.where(result.predictions == 1)[0]
        
        for idx in anomaly_indices:
            explanation = self._explain_single_anomaly(
                data[idx], idx, result, feature_names, n_features
            )
            explanations.append(explanation)
        
        return explanations

    def _explain_single_anomaly(
        self,
        sample: npt.NDArray[np.floating],
        sample_index: int,
        result: DetectionResult,
        feature_names: List[str],
        n_features: int
    ) -> AnomalyExplanation:
        """Explain a single anomalous sample."""
        
        # Calculate basic feature statistics for explanation
        sample_stats = self._calculate_feature_stats(sample, feature_names)
        
        # Select top contributing features
        top_features = sorted(sample_stats, key=lambda x: x.importance_score, reverse=True)[:n_features]
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(top_features, sample_index)
        
        # Calculate confidence based on anomaly score if available
        confidence = 0.8  # Default confidence
        anomaly_score = 0.5  # Default score
        
        if result.scores is not None and len(result.scores) > sample_index:
            anomaly_score = float(result.scores[sample_index])
            confidence = min(0.95, max(0.5, anomaly_score))
        
        return AnomalyExplanation(
            sample_index=sample_index,
            anomaly_score=anomaly_score,
            is_anomaly=True,
            feature_importances=top_features,
            explanation_text=explanation_text,
            confidence=confidence
        )

    def _calculate_feature_stats(
        self,
        sample: npt.NDArray[np.floating],
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Calculate feature importance statistics for a sample."""
        importances = []
        
        for i, (value, name) in enumerate(zip(sample, feature_names)):
            # Simple importance based on absolute deviation from zero
            # In a real implementation, this would use proper explainability methods
            importance_score = abs(float(value))
            
            if value > 2.0:
                contribution_type = "positive"
            elif value < -2.0:
                contribution_type = "negative"
            else:
                contribution_type = "neutral"
            
            importances.append(FeatureImportance(
                feature_index=i,
                feature_name=name,
                importance_score=importance_score,
                contribution_type=contribution_type
            ))
        
        return importances

    def _generate_explanation_text(
        self,
        top_features: List[FeatureImportance],
        sample_index: int
    ) -> str:
        """Generate human-readable explanation text."""
        if not top_features:
            return f"Sample {sample_index} is anomalous but no clear contributing features identified."
        
        # Find the most important feature
        most_important = top_features[0]
        
        explanation_parts = [
            f"Sample {sample_index} is anomalous primarily due to {most_important.feature_name}"
        ]
        
        if most_important.contribution_type == "positive":
            explanation_parts.append(f"which has an unusually high value ({most_important.importance_score:.2f})")
        elif most_important.contribution_type == "negative":
            explanation_parts.append(f"which has an unusually low value ({most_important.importance_score:.2f})")
        else:
            explanation_parts.append(f"which shows abnormal behavior")
        
        # Add other contributing features
        if len(top_features) > 1:
            other_features = [f.feature_name for f in top_features[1:3]]  # Top 2 additional
            if other_features:
                explanation_parts.append(f"Other contributing features: {', '.join(other_features)}")
        
        return ". ".join(explanation_parts) + "."

    def explain_feature_contributions(
        self,
        data: npt.NDArray[np.floating],
        result: DetectionResult,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Explain overall feature contributions to anomaly detection.
        
        Args:
            data: Original input data
            result: Detection result
            feature_names: Optional feature names
            
        Returns:
            Dictionary with overall feature contribution analysis
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        anomaly_indices = np.where(result.predictions == 1)[0]
        normal_indices = np.where(result.predictions == 0)[0]
        
        if len(anomaly_indices) == 0:
            return {"error": "No anomalies found to explain"}
        
        # Calculate statistics for anomalous vs normal samples
        anomaly_data = data[anomaly_indices]
        normal_data = data[normal_indices] if len(normal_indices) > 0 else data
        
        feature_contributions = {}
        
        for i, feature_name in enumerate(feature_names):
            anomaly_values = anomaly_data[:, i]
            normal_values = normal_data[:, i]
            
            # Calculate simple statistics
            anomaly_mean = np.mean(anomaly_values)
            normal_mean = np.mean(normal_values)
            difference = abs(anomaly_mean - normal_mean)
            
            # Calculate contribution score
            anomaly_std = np.std(anomaly_values)
            normal_std = np.std(normal_values) if len(normal_indices) > 0 else np.std(data[:, i])
            
            contribution_score = difference / max(normal_std, 0.1)  # Avoid division by zero
            
            feature_contributions[feature_name] = {
                "contribution_score": float(contribution_score),
                "anomaly_mean": float(anomaly_mean),
                "normal_mean": float(normal_mean),
                "difference": float(difference),
                "anomaly_std": float(anomaly_std),
                "normal_std": float(normal_std)
            }
        
        # Sort by contribution score
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1]["contribution_score"],
            reverse=True
        )
        
        return {
            "total_anomalies": len(anomaly_indices),
            "total_normal": len(normal_indices),
            "feature_contributions": dict(sorted_features),
            "top_contributing_features": [f[0] for f in sorted_features[:5]],
            "summary": self._generate_feature_summary(sorted_features[:3])
        }

    def _generate_feature_summary(self, top_features: List) -> str:
        """Generate summary of top contributing features."""
        if not top_features:
            return "No clear feature patterns identified."
        
        summary_parts = ["Top contributing features to anomaly detection:"]
        
        for i, (feature_name, stats) in enumerate(top_features, 1):
            diff = stats["anomaly_mean"] - stats["normal_mean"]
            direction = "higher" if diff > 0 else "lower"
            
            summary_parts.append(
                f"{i}. {feature_name}: anomalies have {direction} values "
                f"(avg {stats['anomaly_mean']:.2f} vs {stats['normal_mean']:.2f})"
            )
        
        return " ".join(summary_parts)

    def generate_detection_report(
        self,
        data: npt.NDArray[np.floating],
        result: DetectionResult,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive detection report with explanations.
        
        Args:
            data: Original input data
            result: Detection result
            feature_names: Optional feature names
            
        Returns:
            Comprehensive report dictionary
        """
        # Get anomaly explanations
        anomaly_explanations = self.explain_anomalies(data, result, feature_names)
        
        # Get feature contributions
        feature_analysis = self.explain_feature_contributions(data, result, feature_names)
        
        # Generate summary statistics
        anomaly_rate = result.n_anomalies / result.n_samples
        
        report = {
            "detection_summary": {
                "algorithm": result.algorithm,
                "total_samples": result.n_samples,
                "anomalies_detected": result.n_anomalies,
                "anomaly_rate": f"{anomaly_rate:.1%}",
                "contamination_setting": result.contamination,
                "execution_time": f"{result.execution_time:.3f}s"
            },
            "anomaly_explanations": [
                {
                    "sample_index": exp.sample_index,
                    "anomaly_score": exp.anomaly_score,
                    "confidence": exp.confidence,
                    "explanation": exp.explanation_text,
                    "top_features": [
                        {
                            "name": fi.feature_name,
                            "importance": fi.importance_score,
                            "type": fi.contribution_type
                        }
                        for fi in exp.feature_importances[:3]
                    ]
                }
                for exp in anomaly_explanations[:10]  # Limit to first 10
            ],
            "feature_analysis": feature_analysis,
            "recommendations": self._generate_recommendations(result, feature_analysis),
            "report_timestamp": result.timestamp.isoformat()
        }
        
        return report

    def _generate_recommendations(
        self,
        result: DetectionResult,
        feature_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on detection results."""
        recommendations = []
        
        anomaly_rate = result.n_anomalies / result.n_samples
        
        if anomaly_rate > 0.3:
            recommendations.append(
                "High anomaly rate detected. Consider adjusting contamination parameter or reviewing data quality."
            )
        
        if anomaly_rate < 0.01:
            recommendations.append(
                "Very low anomaly rate. Consider increasing sensitivity or checking for data preprocessing issues."
            )
        
        if "top_contributing_features" in feature_analysis:
            top_features = feature_analysis["top_contributing_features"][:2]
            if top_features:
                recommendations.append(
                    f"Focus investigation on features: {', '.join(top_features)} which contribute most to anomaly detection."
                )
        
        if result.execution_time > 10.0:
            recommendations.append(
                "Detection took significant time. Consider using faster algorithms for real-time applications."
            )
        
        recommendations.append(
            "Validate detected anomalies with domain experts to confirm relevance and accuracy."
        )
        
        return recommendations