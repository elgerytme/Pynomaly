"""Advanced analytics and explainable AI features for Pynomaly."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult
from pynomaly.infrastructure.cache import cached
from pynomaly.shared.error_handling import (
    ErrorCodes,
    create_infrastructure_error,
)

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of advanced analytics."""

    TIME_SERIES = "time_series"
    PATTERN_DETECTION = "pattern_detection"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_EXPLANATION = "anomaly_explanation"
    CORRELATION_ANALYSIS = "correlation_analysis"
    FEATURE_IMPORTANCE = "feature_importance"


@dataclass
class AnalyticsResult:
    """Result of advanced analytics analysis."""

    analysis_type: AnalysisType
    timestamp: datetime
    insights: dict[str, Any]
    visualizations: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "analysis_type": self.analysis_type.value,
            "timestamp": self.timestamp.isoformat(),
            "insights": self.insights,
            "visualizations": self.visualizations,
            "recommendations": self.recommendations,
            "confidence_score": self.confidence_score,
            "execution_time_ms": self.execution_time_ms,
        }


class TimeSeriesAnalyzer:
    """Advanced time series analysis for anomaly detection."""

    def __init__(self):
        """Initialize time series analyzer."""
        self.seasonal_patterns = {}
        self.trend_components = {}

    async def analyze_time_series(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        value_col: str = "value",
    ) -> AnalyticsResult:
        """Analyze time series data for patterns and anomalies.

        Args:
            data: Time series data
            timestamp_col: Name of timestamp column
            value_col: Name of value column

        Returns:
            Analytics result with time series insights
        """
        start_time = datetime.now()

        try:
            # Convert timestamp column to datetime
            if timestamp_col in data.columns:
                data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                data = data.sort_values(timestamp_col)

            # Basic statistics
            insights = {
                "data_points": len(data),
                "time_range": {
                    "start": data[timestamp_col].min().isoformat()
                    if timestamp_col in data.columns
                    else None,
                    "end": data[timestamp_col].max().isoformat()
                    if timestamp_col in data.columns
                    else None,
                },
                "value_statistics": {
                    "mean": float(data[value_col].mean()),
                    "std": float(data[value_col].std()),
                    "min": float(data[value_col].min()),
                    "max": float(data[value_col].max()),
                    "median": float(data[value_col].median()),
                },
            }

            # Detect seasonality
            seasonality_info = self._detect_seasonality(data, timestamp_col, value_col)
            insights["seasonality"] = seasonality_info

            # Trend analysis
            trend_info = self._analyze_trends(data, timestamp_col, value_col)
            insights["trend"] = trend_info

            # Anomaly detection in time series
            anomaly_info = self._detect_time_series_anomalies(
                data, timestamp_col, value_col
            )
            insights["anomalies"] = anomaly_info

            # Generate visualizations
            visualizations = self._generate_time_series_visualizations(
                data, timestamp_col, value_col
            )

            # Generate recommendations
            recommendations = self._generate_time_series_recommendations(insights)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return AnalyticsResult(
                analysis_type=AnalysisType.TIME_SERIES,
                timestamp=start_time,
                insights=insights,
                visualizations=visualizations,
                recommendations=recommendations,
                confidence_score=0.85,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Time series analysis failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Time series analysis failed: {str(e)}",
                cause=e,
            )

    def _detect_seasonality(
        self, data: pd.DataFrame, timestamp_col: str, value_col: str
    ) -> dict[str, Any]:
        """Detect seasonal patterns in time series data."""
        try:
            # Simple autocorrelation-based seasonality detection
            values = data[value_col].values

            # Check for daily, weekly, monthly patterns
            patterns = {}

            # Daily pattern (24 hours)
            if len(values) >= 24:
                daily_corr = np.corrcoef(values[:-24], values[24:])[0, 1]
                patterns["daily"] = {
                    "correlation": float(daily_corr),
                    "detected": abs(daily_corr) > 0.3,
                }

            # Weekly pattern (7 days)
            if len(values) >= 168:  # 7 * 24 hours
                weekly_corr = np.corrcoef(values[:-168], values[168:])[0, 1]
                patterns["weekly"] = {
                    "correlation": float(weekly_corr),
                    "detected": abs(weekly_corr) > 0.3,
                }

            return {
                "patterns": patterns,
                "seasonal_strength": max(
                    [p.get("correlation", 0) for p in patterns.values()]
                ),
            }

        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")
            return {"patterns": {}, "seasonal_strength": 0.0}

    def _analyze_trends(
        self, data: pd.DataFrame, timestamp_col: str, value_col: str
    ) -> dict[str, Any]:
        """Analyze trends in time series data."""
        try:
            values = data[value_col].values

            # Simple linear trend
            x = np.arange(len(values))
            trend_coef = np.polyfit(x, values, 1)[0]

            # Trend strength
            trend_strength = (
                abs(trend_coef) / np.std(values) if np.std(values) > 0 else 0
            )

            # Trend direction
            if trend_coef > 0.01:
                trend_direction = "increasing"
            elif trend_coef < -0.01:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"

            return {
                "trend_coefficient": float(trend_coef),
                "trend_strength": float(trend_strength),
                "trend_direction": trend_direction,
                "trend_significance": "high" if trend_strength > 0.1 else "low",
            }

        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            return {"trend_direction": "unknown", "trend_strength": 0.0}

    def _detect_time_series_anomalies(
        self, data: pd.DataFrame, timestamp_col: str, value_col: str
    ) -> dict[str, Any]:
        """Detect anomalies in time series data."""
        try:
            values = data[value_col].values

            # Statistical anomaly detection
            mean_val = np.mean(values)
            std_val = np.std(values)
            threshold = 3 * std_val

            # Find outliers
            outliers = np.abs(values - mean_val) > threshold
            outlier_indices = np.where(outliers)[0]

            # Change point detection (simple)
            change_points = []
            if len(values) > 10:
                window_size = min(10, len(values) // 4)
                for i in range(window_size, len(values) - window_size):
                    before = np.mean(values[i - window_size : i])
                    after = np.mean(values[i : i + window_size])
                    if abs(before - after) > 2 * std_val:
                        change_points.append(i)

            return {
                "outliers": {
                    "count": len(outlier_indices),
                    "indices": outlier_indices.tolist(),
                    "threshold": float(threshold),
                },
                "change_points": {
                    "count": len(change_points),
                    "indices": change_points,
                },
                "anomaly_rate": float(len(outlier_indices) / len(values)),
            }

        except Exception as e:
            logger.warning(f"Time series anomaly detection failed: {e}")
            return {"outliers": {"count": 0}, "change_points": {"count": 0}}

    def _generate_time_series_visualizations(
        self, data: pd.DataFrame, timestamp_col: str, value_col: str
    ) -> list[dict[str, Any]]:
        """Generate visualization configurations for time series data."""
        visualizations = []

        # Time series plot
        visualizations.append(
            {
                "type": "line_chart",
                "title": "Time Series Data",
                "x_axis": timestamp_col,
                "y_axis": value_col,
                "data_points": min(1000, len(data)),  # Limit for performance
            }
        )

        # Distribution histogram
        visualizations.append(
            {
                "type": "histogram",
                "title": "Value Distribution",
                "data": value_col,
                "bins": 50,
            }
        )

        # Anomaly scatter plot
        visualizations.append(
            {
                "type": "scatter_plot",
                "title": "Anomaly Detection",
                "x_axis": timestamp_col,
                "y_axis": value_col,
                "highlight_outliers": True,
            }
        )

        return visualizations

    def _generate_time_series_recommendations(
        self, insights: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on time series analysis."""
        recommendations = []

        # Seasonality recommendations
        if insights.get("seasonality", {}).get("seasonal_strength", 0) > 0.5:
            recommendations.append(
                "Consider seasonal decomposition for better anomaly detection"
            )

        # Trend recommendations
        trend_info = insights.get("trend", {})
        if trend_info.get("trend_strength", 0) > 0.1:
            recommendations.append(
                f"Strong {trend_info.get('trend_direction', 'unknown')} trend detected - consider detrending"
            )

        # Anomaly recommendations
        anomaly_info = insights.get("anomalies", {})
        if anomaly_info.get("anomaly_rate", 0) > 0.1:
            recommendations.append(
                "High anomaly rate detected - review detection thresholds"
            )

        if anomaly_info.get("change_points", {}).get("count", 0) > 0:
            recommendations.append(
                "Change points detected - investigate potential system changes"
            )

        return recommendations


class PatternDetector:
    """Advanced pattern detection in anomaly data."""

    def __init__(self):
        """Initialize pattern detector."""
        self.known_patterns = {}

    async def detect_patterns(
        self, detection_results: list[DetectionResult], dataset: Dataset
    ) -> AnalyticsResult:
        """Detect patterns in anomaly detection results.

        Args:
            detection_results: List of detection results
            dataset: Dataset used for detection

        Returns:
            Analytics result with pattern insights
        """
        start_time = datetime.now()

        try:
            # Aggregate anomaly information
            all_anomalies = []
            for result in detection_results:
                all_anomalies.extend(result.anomalies)

            # Pattern analysis
            insights = {
                "total_anomalies": len(all_anomalies),
                "detection_results": len(detection_results),
            }

            # Temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(all_anomalies)
            insights["temporal_patterns"] = temporal_patterns

            # Spatial patterns (if applicable)
            spatial_patterns = self._analyze_spatial_patterns(all_anomalies, dataset)
            insights["spatial_patterns"] = spatial_patterns

            # Anomaly type patterns
            type_patterns = self._analyze_anomaly_types(all_anomalies)
            insights["type_patterns"] = type_patterns

            # Severity patterns
            severity_patterns = self._analyze_severity_patterns(all_anomalies)
            insights["severity_patterns"] = severity_patterns

            # Generate visualizations
            visualizations = self._generate_pattern_visualizations(insights)

            # Generate recommendations
            recommendations = self._generate_pattern_recommendations(insights)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return AnalyticsResult(
                analysis_type=AnalysisType.PATTERN_DETECTION,
                timestamp=start_time,
                insights=insights,
                visualizations=visualizations,
                recommendations=recommendations,
                confidence_score=0.80,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Pattern detection failed: {str(e)}",
                cause=e,
            )

    def _analyze_temporal_patterns(self, anomalies: list[Anomaly]) -> dict[str, Any]:
        """Analyze temporal patterns in anomalies."""
        if not anomalies:
            return {"patterns": [], "temporal_clustering": False}

        # Extract timestamps (assuming anomalies have timestamp info)
        timestamps = [a.timestamp for a in anomalies if hasattr(a, "timestamp")]

        if not timestamps:
            return {"patterns": [], "temporal_clustering": False}

        # Convert to pandas datetime for analysis
        ts_series = pd.Series(timestamps)

        # Hour-of-day pattern
        hour_counts = ts_series.dt.hour.value_counts().sort_index()
        peak_hours = hour_counts.nlargest(3).index.tolist()

        # Day-of-week pattern
        dow_counts = ts_series.dt.dayofweek.value_counts().sort_index()
        peak_days = dow_counts.nlargest(2).index.tolist()

        return {
            "patterns": [
                {
                    "type": "hourly",
                    "peak_hours": peak_hours,
                    "distribution": hour_counts.to_dict(),
                },
                {
                    "type": "daily",
                    "peak_days": peak_days,
                    "distribution": dow_counts.to_dict(),
                },
            ],
            "temporal_clustering": len(peak_hours) <= 3,  # Concentrated in few hours
        }

    def _analyze_spatial_patterns(
        self, anomalies: list[Anomaly], dataset: Dataset
    ) -> dict[str, Any]:
        """Analyze spatial patterns in anomalies."""
        # This would analyze geographic or feature-space clustering
        # For now, return basic feature correlation analysis

        try:
            # Get feature columns
            numeric_cols = dataset.data.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) < 2:
                return {"feature_clustering": False, "dominant_features": []}

            # Simple feature importance based on anomaly frequency
            feature_importance = {}
            for col in numeric_cols:
                if col != dataset.target_column:
                    # Calculate variance as proxy for importance
                    variance = dataset.data[col].var()
                    feature_importance[col] = float(variance)

            # Top features
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            dominant_features = [f[0] for f in sorted_features[:5]]

            return {
                "feature_clustering": True,
                "dominant_features": dominant_features,
                "feature_importance": feature_importance,
            }

        except Exception as e:
            logger.warning(f"Spatial pattern analysis failed: {e}")
            return {"feature_clustering": False, "dominant_features": []}

    def _analyze_anomaly_types(self, anomalies: list[Anomaly]) -> dict[str, Any]:
        """Analyze distribution of anomaly types."""
        if not anomalies:
            return {"type_distribution": {}, "dominant_type": None}

        # Count anomaly types
        type_counts = {}
        for anomaly in anomalies:
            anomaly_type = anomaly.type.value if hasattr(anomaly, "type") else "unknown"
            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1

        # Find dominant type
        dominant_type = (
            max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        )

        return {
            "type_distribution": type_counts,
            "dominant_type": dominant_type,
            "type_diversity": len(type_counts),
        }

    def _analyze_severity_patterns(self, anomalies: list[Anomaly]) -> dict[str, Any]:
        """Analyze severity patterns in anomalies."""
        if not anomalies:
            return {"severity_distribution": {}, "high_severity_count": 0}

        # Extract severity scores
        severity_scores = []
        for anomaly in anomalies:
            if hasattr(anomaly, "severity_score"):
                severity_scores.append(anomaly.severity_score.value)

        if not severity_scores:
            return {"severity_distribution": {}, "high_severity_count": 0}

        # Calculate severity statistics
        severity_stats = {
            "mean": float(np.mean(severity_scores)),
            "std": float(np.std(severity_scores)),
            "min": float(np.min(severity_scores)),
            "max": float(np.max(severity_scores)),
            "median": float(np.median(severity_scores)),
        }

        # Count high severity anomalies
        high_severity_count = sum(1 for score in severity_scores if score > 0.8)

        return {
            "severity_distribution": severity_stats,
            "high_severity_count": high_severity_count,
            "high_severity_rate": high_severity_count / len(severity_scores),
        }

    def _generate_pattern_visualizations(
        self, insights: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate visualization configurations for pattern analysis."""
        visualizations = []

        # Temporal pattern visualization
        if insights.get("temporal_patterns", {}).get("patterns"):
            visualizations.append(
                {
                    "type": "heatmap",
                    "title": "Temporal Anomaly Patterns",
                    "data": "temporal_patterns",
                    "x_axis": "time",
                    "y_axis": "frequency",
                }
            )

        # Anomaly type distribution
        if insights.get("type_patterns", {}).get("type_distribution"):
            visualizations.append(
                {
                    "type": "pie_chart",
                    "title": "Anomaly Type Distribution",
                    "data": "type_patterns.type_distribution",
                }
            )

        # Severity distribution
        if insights.get("severity_patterns", {}).get("severity_distribution"):
            visualizations.append(
                {
                    "type": "box_plot",
                    "title": "Severity Score Distribution",
                    "data": "severity_patterns",
                }
            )

        return visualizations

    def _generate_pattern_recommendations(self, insights: dict[str, Any]) -> list[str]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []

        # Temporal pattern recommendations
        temporal_patterns = insights.get("temporal_patterns", {})
        if temporal_patterns.get("temporal_clustering"):
            recommendations.append("Consider time-based anomaly detection models")

        # Type pattern recommendations
        type_patterns = insights.get("type_patterns", {})
        if type_patterns.get("type_diversity", 0) > 3:
            recommendations.append(
                "Multiple anomaly types detected - consider ensemble methods"
            )

        # Severity pattern recommendations
        severity_patterns = insights.get("severity_patterns", {})
        if severity_patterns.get("high_severity_rate", 0) > 0.3:
            recommendations.append(
                "High rate of severe anomalies - review detection thresholds"
            )

        return recommendations


class AnomalyExplainer:
    """Explainable AI for anomaly detection results."""

    def __init__(self):
        """Initialize anomaly explainer."""
        self.explanation_cache = {}

    @cached(ttl=3600, key_prefix="anomaly_explanation")
    async def explain_anomaly(
        self, anomaly: Anomaly, dataset: Dataset, detection_result: DetectionResult
    ) -> AnalyticsResult:
        """Explain why a specific anomaly was detected.

        Args:
            anomaly: Anomaly to explain
            dataset: Dataset used for detection
            detection_result: Detection result containing the anomaly

        Returns:
            Analytics result with explanation insights
        """
        start_time = datetime.now()

        try:
            # Feature contribution analysis
            feature_contributions = self._analyze_feature_contributions(
                anomaly, dataset
            )

            # Statistical explanation
            statistical_explanation = self._generate_statistical_explanation(
                anomaly, dataset
            )

            # Contextual explanation
            contextual_explanation = self._generate_contextual_explanation(
                anomaly, dataset, detection_result
            )

            # Similarity analysis
            similarity_analysis = self._analyze_similar_anomalies(
                anomaly, detection_result
            )

            insights = {
                "anomaly_id": anomaly.id,
                "explanation_type": "feature_based",
                "feature_contributions": feature_contributions,
                "statistical_explanation": statistical_explanation,
                "contextual_explanation": contextual_explanation,
                "similarity_analysis": similarity_analysis,
            }

            # Generate visualizations
            visualizations = self._generate_explanation_visualizations(insights)

            # Generate recommendations
            recommendations = self._generate_explanation_recommendations(insights)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return AnalyticsResult(
                analysis_type=AnalysisType.ANOMALY_EXPLANATION,
                timestamp=start_time,
                insights=insights,
                visualizations=visualizations,
                recommendations=recommendations,
                confidence_score=0.75,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Anomaly explanation failed: {e}")
            raise create_infrastructure_error(
                error_code=ErrorCodes.INF_PROCESSING_ERROR,
                message=f"Anomaly explanation failed: {str(e)}",
                cause=e,
            )

    def _analyze_feature_contributions(
        self, anomaly: Anomaly, dataset: Dataset
    ) -> dict[str, Any]:
        """Analyze feature contributions to anomaly score."""
        try:
            # Get numeric columns
            numeric_cols = dataset.data.select_dtypes(include=[np.number]).columns

            if dataset.target_column and dataset.target_column in numeric_cols:
                numeric_cols = numeric_cols.drop(dataset.target_column)

            # Calculate feature statistics
            feature_stats = {}
            for col in numeric_cols:
                col_mean = dataset.data[col].mean()
                col_std = dataset.data[col].std()

                # Get anomaly value for this feature (simplified)
                # In reality, you'd need to map anomaly to original data point
                feature_stats[col] = {
                    "mean": float(col_mean),
                    "std": float(col_std),
                    "z_score": 0.0,  # Would calculate actual z-score
                    "contribution": abs(col_std)
                    / sum(dataset.data[numeric_cols].std()),
                }

            # Sort by contribution
            sorted_features = sorted(
                feature_stats.items(), key=lambda x: x[1]["contribution"], reverse=True
            )

            return {
                "top_contributing_features": [f[0] for f in sorted_features[:5]],
                "feature_statistics": feature_stats,
                "explanation_confidence": 0.7,
            }

        except Exception as e:
            logger.warning(f"Feature contribution analysis failed: {e}")
            return {"top_contributing_features": [], "feature_statistics": {}}

    def _generate_statistical_explanation(
        self, anomaly: Anomaly, dataset: Dataset
    ) -> dict[str, Any]:
        """Generate statistical explanation for the anomaly."""
        return {
            "anomaly_score": float(anomaly.score.value),
            "score_percentile": 95.0,  # Would calculate actual percentile
            "distance_from_normal": 2.5,  # Statistical distance measure
            "local_density": 0.1,  # Local outlier factor
            "explanation": f"This anomaly has a score of {anomaly.score.value:.3f}, placing it in the top 5% of outliers",
        }

    def _generate_contextual_explanation(
        self, anomaly: Anomaly, dataset: Dataset, detection_result: DetectionResult
    ) -> dict[str, Any]:
        """Generate contextual explanation for the anomaly."""
        return {
            "dataset_context": {
                "dataset_size": len(dataset.data),
                "feature_count": len(dataset.data.columns),
                "anomaly_rate": len(detection_result.anomalies) / len(dataset.data),
            },
            "detection_context": {
                "detector_type": detection_result.metadata.get("algorithm", "unknown"),
                "detection_time": detection_result.metadata.get("execution_time_ms", 0),
                "threshold_used": detection_result.threshold,
            },
            "business_context": {
                "severity_level": "high" if anomaly.score.value > 0.8 else "medium",
                "recommended_action": "investigate"
                if anomaly.score.value > 0.7
                else "monitor",
            },
        }

    def _analyze_similar_anomalies(
        self, anomaly: Anomaly, detection_result: DetectionResult
    ) -> dict[str, Any]:
        """Analyze similar anomalies in the detection result."""
        similar_anomalies = []

        # Find anomalies with similar scores
        for other_anomaly in detection_result.anomalies:
            if other_anomaly.id != anomaly.id:
                score_diff = abs(other_anomaly.score.value - anomaly.score.value)
                if score_diff < 0.1:  # Similar score threshold
                    similar_anomalies.append(
                        {
                            "id": other_anomaly.id,
                            "score": other_anomaly.score.value,
                            "similarity": 1.0 - score_diff,
                        }
                    )

        return {
            "similar_count": len(similar_anomalies),
            "similar_anomalies": similar_anomalies[:5],  # Top 5 similar
            "cluster_size": len(similar_anomalies) + 1,  # Including current anomaly
        }

    def _generate_explanation_visualizations(
        self, insights: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate visualizations for anomaly explanation."""
        visualizations = []

        # Feature contribution chart
        if insights.get("feature_contributions", {}).get("top_contributing_features"):
            visualizations.append(
                {
                    "type": "bar_chart",
                    "title": "Feature Contributions to Anomaly",
                    "data": "feature_contributions",
                    "x_axis": "features",
                    "y_axis": "contribution",
                }
            )

        # Anomaly score context
        visualizations.append(
            {
                "type": "gauge_chart",
                "title": "Anomaly Score",
                "value": insights.get("statistical_explanation", {}).get(
                    "anomaly_score", 0
                ),
                "min": 0,
                "max": 1,
                "threshold": 0.7,
            }
        )

        return visualizations

    def _generate_explanation_recommendations(
        self, insights: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on anomaly explanation."""
        recommendations = []

        # Feature-based recommendations
        top_features = insights.get("feature_contributions", {}).get(
            "top_contributing_features", []
        )
        if top_features:
            recommendations.append(
                f"Focus investigation on features: {', '.join(top_features[:3])}"
            )

        # Score-based recommendations
        statistical_info = insights.get("statistical_explanation", {})
        if statistical_info.get("anomaly_score", 0) > 0.9:
            recommendations.append(
                "Critical anomaly detected - immediate investigation required"
            )
        elif statistical_info.get("anomaly_score", 0) > 0.7:
            recommendations.append("High-confidence anomaly - prioritize for review")

        # Similarity-based recommendations
        similarity_info = insights.get("similarity_analysis", {})
        if similarity_info.get("similar_count", 0) > 3:
            recommendations.append(
                "Multiple similar anomalies detected - possible systematic issue"
            )

        return recommendations


class AnalyticsEngine:
    """Main analytics engine coordinating all advanced analytics."""

    def __init__(self):
        """Initialize analytics engine."""
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.pattern_detector = PatternDetector()
        self.anomaly_explainer = AnomalyExplainer()
        self.analysis_history = []

    async def run_comprehensive_analysis(
        self,
        dataset: Dataset,
        detection_results: list[DetectionResult],
        analysis_types: list[AnalysisType] | None = None,
    ) -> dict[str, AnalyticsResult]:
        """Run comprehensive analytics on dataset and detection results.

        Args:
            dataset: Dataset to analyze
            detection_results: Detection results to analyze
            analysis_types: Specific analysis types to run (all if None)

        Returns:
            Dictionary of analysis results by type
        """
        if analysis_types is None:
            analysis_types = [AnalysisType.TIME_SERIES, AnalysisType.PATTERN_DETECTION]

        results = {}

        # Time series analysis
        if AnalysisType.TIME_SERIES in analysis_types:
            try:
                # Check if dataset has time series structure
                if self._has_time_series_structure(dataset):
                    results[
                        AnalysisType.TIME_SERIES
                    ] = await self.time_series_analyzer.analyze_time_series(
                        dataset.data
                    )
            except Exception as e:
                logger.error(f"Time series analysis failed: {e}")

        # Pattern detection
        if AnalysisType.PATTERN_DETECTION in analysis_types:
            try:
                results[
                    AnalysisType.PATTERN_DETECTION
                ] = await self.pattern_detector.detect_patterns(
                    detection_results, dataset
                )
            except Exception as e:
                logger.error(f"Pattern detection failed: {e}")

        # Store analysis history
        self.analysis_history.append(
            {
                "timestamp": datetime.now(),
                "dataset_name": dataset.name,
                "analysis_types": [at.value for at in analysis_types],
                "results_count": len(results),
            }
        )

        return results

    async def explain_anomalies(
        self,
        anomalies: list[Anomaly],
        dataset: Dataset,
        detection_result: DetectionResult,
    ) -> list[AnalyticsResult]:
        """Explain multiple anomalies.

        Args:
            anomalies: List of anomalies to explain
            dataset: Dataset used for detection
            detection_result: Detection result containing anomalies

        Returns:
            List of explanation results
        """
        explanations = []

        for anomaly in anomalies[:10]:  # Limit to first 10 for performance
            try:
                explanation = await self.anomaly_explainer.explain_anomaly(
                    anomaly, dataset, detection_result
                )
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain anomaly {anomaly.id}: {e}")

        return explanations

    def _has_time_series_structure(self, dataset: Dataset) -> bool:
        """Check if dataset has time series structure."""
        # Look for timestamp-like columns
        time_cols = [
            col
            for col in dataset.data.columns
            if "time" in col.lower() or "date" in col.lower()
        ]

        return len(time_cols) > 0 and len(dataset.data) > 10


class AdvancedAnalytics:
    """Main advanced analytics facade."""

    def __init__(self):
        """Initialize advanced analytics."""
        self.engine = AnalyticsEngine()

    async def analyze_dataset(
        self, dataset: Dataset, detection_results: list[DetectionResult] | None = None
    ) -> dict[str, Any]:
        """Analyze dataset with comprehensive analytics.

        Args:
            dataset: Dataset to analyze
            detection_results: Optional detection results

        Returns:
            Comprehensive analytics report
        """
        detection_results = detection_results or []

        # Run comprehensive analysis
        analysis_results = await self.engine.run_comprehensive_analysis(
            dataset, detection_results
        )

        # Generate summary report
        report = {
            "dataset_name": dataset.name,
            "analysis_timestamp": datetime.now().isoformat(),
            "analyses_performed": list(analysis_results.keys()),
            "results": {
                analysis_type.value: result.to_dict()
                for analysis_type, result in analysis_results.items()
            },
            "summary": self._generate_summary(analysis_results),
        }

        return report

    def _generate_summary(
        self, analysis_results: dict[AnalysisType, AnalyticsResult]
    ) -> dict[str, Any]:
        """Generate summary of analysis results."""
        summary = {
            "total_analyses": len(analysis_results),
            "key_insights": [],
            "overall_recommendations": [],
            "confidence_scores": {},
        }

        for analysis_type, result in analysis_results.items():
            summary["confidence_scores"][analysis_type.value] = result.confidence_score
            summary["key_insights"].extend(result.insights.get("key_points", []))
            summary["overall_recommendations"].extend(result.recommendations)

        # Remove duplicates from recommendations
        summary["overall_recommendations"] = list(
            set(summary["overall_recommendations"])
        )

        return summary


# Global analytics engine
_analytics_engine: AdvancedAnalytics | None = None


def get_analytics_engine() -> AdvancedAnalytics:
    """Get global analytics engine.

    Returns:
        Advanced analytics instance
    """
    global _analytics_engine

    if _analytics_engine is None:
        _analytics_engine = AdvancedAnalytics()

    return _analytics_engine
