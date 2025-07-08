"""Interactive anomaly investigation dashboard with drill-down analysis and SHAP explanations."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from pynomaly.domain.entities import Dataset, Detector

logger = logging.getLogger(__name__)


class InvestigationType(str, Enum):
    """Types of anomaly investigations."""

    SINGLE_ANOMALY = "single_anomaly"
    ANOMALY_CLUSTER = "anomaly_cluster"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    FEATURE_ANALYSIS = "feature_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class ExplanationMethod(str, Enum):
    """Methods for generating explanations."""

    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"
    ANCHOR = "anchor"
    GRADIENT_BASED = "gradient_based"


class VisualizationType(str, Enum):
    """Types of visualizations."""

    SCATTER_PLOT = "scatter_plot"
    TIME_SERIES = "time_series"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    CORRELATION_MATRIX = "correlation_matrix"
    FEATURE_IMPORTANCE_BAR = "feature_importance_bar"
    DECISION_BOUNDARY = "decision_boundary"
    ANOMALY_SCORE_DISTRIBUTION = "anomaly_score_distribution"


@dataclass
class AnomalyRecord:
    """Record of an anomaly with metadata."""

    anomaly_id: str
    timestamp: datetime
    anomaly_score: float
    feature_values: dict[str, Any]
    detector_name: str
    confidence: float
    severity: str = "medium"
    status: str = "new"
    investigation_notes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related_anomalies: list[str] = field(default_factory=list)
    business_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvestigationSession:
    """Investigation session tracking."""

    session_id: str
    user_id: str
    start_time: datetime
    investigation_type: InvestigationType
    anomaly_ids: list[str]
    current_focus: str | None = None
    analysis_steps: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    session_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationResult:
    """Result of an explanation analysis."""

    method: ExplanationMethod
    anomaly_id: str
    feature_contributions: dict[str, float]
    explanation_text: str
    confidence: float
    visualizations: list[dict[str, Any]] = field(default_factory=list)
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    similar_cases: list[str] = field(default_factory=list)


class SHAPExplainer:
    """SHAP-based explanation service."""

    def __init__(self):
        self.explainers = {}
        self.background_data = {}

    async def explain_anomaly(
        self,
        detector: Detector,
        anomaly_data: np.ndarray,
        background_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> ExplanationResult:
        """Generate SHAP explanation for an anomaly."""
        try:
            # This is a simplified SHAP explanation
            # In practice, you'd use the actual SHAP library

            if background_data is not None:
                self.background_data[detector.__class__.__name__] = background_data

            # Simulate SHAP values calculation
            feature_contributions = await self._calculate_shap_values(
                detector, anomaly_data, background_data, feature_names
            )

            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                feature_contributions, feature_names
            )

            # Create visualizations
            visualizations = await self._create_shap_visualizations(
                feature_contributions, feature_names
            )

            return ExplanationResult(
                method=ExplanationMethod.SHAP,
                anomaly_id="",  # Will be set by caller
                feature_contributions=feature_contributions,
                explanation_text=explanation_text,
                confidence=0.85,
                visualizations=visualizations,
            )

        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            return ExplanationResult(
                method=ExplanationMethod.SHAP,
                anomaly_id="",
                feature_contributions={},
                explanation_text=f"Error generating explanation: {str(e)}",
                confidence=0.0,
            )

    async def _calculate_shap_values(
        self,
        detector: Detector,
        anomaly_data: np.ndarray,
        background_data: np.ndarray | None,
        feature_names: list[str] | None,
    ) -> dict[str, float]:
        """Calculate SHAP values for the anomaly."""
        # Simplified SHAP calculation
        # In practice, you'd use actual SHAP explainer

        if len(anomaly_data.shape) == 1:
            anomaly_data = anomaly_data.reshape(1, -1)

        feature_contributions = {}

        # Get baseline prediction
        if background_data is not None and len(background_data) > 0:
            baseline_score = np.mean(
                detector.predict(
                    Dataset(
                        name="baseline",
                        data=background_data,
                        features=feature_names or [],
                    )
                )
            )
        else:
            baseline_score = 0.5  # Default baseline

        # Get anomaly prediction
        anomaly_score = detector.predict(
            Dataset(name="anomaly", data=anomaly_data, features=feature_names or [])
        )[0]

        # Calculate feature importance (simplified)
        num_features = anomaly_data.shape[1]
        feature_names = feature_names or [f"feature_{i}" for i in range(num_features)]

        for i, feature_name in enumerate(feature_names):
            if i < num_features:
                # Perturb feature and measure impact
                perturbed_data = anomaly_data.copy()
                if background_data is not None and len(background_data) > 0:
                    # Replace with background mean
                    perturbed_data[0, i] = np.mean(background_data[:, i])
                else:
                    # Replace with zero
                    perturbed_data[0, i] = 0.0

                perturbed_score = detector.predict(
                    Dataset(
                        name="perturbed", data=perturbed_data, features=feature_names
                    )
                )[0]

                # SHAP value = original - perturbed
                shap_value = anomaly_score - perturbed_score
                feature_contributions[feature_name] = float(shap_value)

        return feature_contributions

    def _generate_explanation_text(
        self,
        feature_contributions: dict[str, float],
        feature_names: list[str] | None,
    ) -> str:
        """Generate human-readable explanation text."""
        if not feature_contributions:
            return "Unable to generate explanation due to insufficient data."

        # Sort features by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        explanation_parts = []

        # Top contributing features
        top_features = sorted_features[:3]
        if top_features:
            explanation_parts.append("The anomaly is primarily driven by:")

            for feature_name, contribution in top_features:
                direction = "increasing" if contribution > 0 else "decreasing"
                magnitude = "strongly" if abs(contribution) > 0.1 else "moderately"
                explanation_parts.append(
                    f"• {feature_name}: {magnitude} {direction} the anomaly score ({contribution:.3f})"
                )

        # Overall explanation
        total_positive = sum(v for v in feature_contributions.values() if v > 0)
        total_negative = sum(v for v in feature_contributions.values() if v < 0)

        if abs(total_positive) > abs(total_negative):
            explanation_parts.append(
                f"\nOverall, the features contribute positively to the anomaly detection "
                f"(net contribution: {total_positive + total_negative:.3f})."
            )
        else:
            explanation_parts.append(
                f"\nThe anomaly score is driven by a mix of positive and negative feature contributions "
                f"(net contribution: {total_positive + total_negative:.3f})."
            )

        return "\n".join(explanation_parts)

    async def _create_shap_visualizations(
        self,
        feature_contributions: dict[str, float],
        feature_names: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Create SHAP visualizations."""
        visualizations = []

        if not feature_contributions:
            return visualizations

        # Feature importance bar chart
        sorted_features = sorted(
            feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        bar_chart = {
            "type": "bar_chart",
            "title": "Feature Contributions (SHAP Values)",
            "data": {
                "features": [item[0] for item in sorted_features],
                "contributions": [item[1] for item in sorted_features],
                "colors": ["red" if val < 0 else "blue" for _, val in sorted_features],
            },
            "config": {
                "x_label": "SHAP Value",
                "y_label": "Features",
                "horizontal": True,
            },
        }
        visualizations.append(bar_chart)

        # Waterfall chart data
        waterfall_chart = {
            "type": "waterfall",
            "title": "SHAP Waterfall Plot",
            "data": {
                "features": [item[0] for item in sorted_features],
                "contributions": [item[1] for item in sorted_features],
                "cumulative": self._calculate_cumulative_contributions(sorted_features),
            },
        }
        visualizations.append(waterfall_chart)

        return visualizations

    def _calculate_cumulative_contributions(
        self, sorted_features: list[tuple[str, float]]
    ) -> list[float]:
        """Calculate cumulative contributions for waterfall chart."""
        cumulative = []
        running_sum = 0.0

        for _, contribution in sorted_features:
            running_sum += contribution
            cumulative.append(running_sum)

        return cumulative


class InteractiveInvestigationDashboard:
    """Interactive dashboard for anomaly investigation with drill-down capabilities."""

    def __init__(
        self,
        explanation_methods: list[ExplanationMethod] = None,
        max_sessions: int = 100,
        session_timeout_hours: int = 24,
    ):
        """Initialize interactive investigation dashboard.

        Args:
            explanation_methods: Available explanation methods
            max_sessions: Maximum concurrent investigation sessions
            session_timeout_hours: Session timeout in hours
        """
        self.explanation_methods = explanation_methods or [
            ExplanationMethod.SHAP,
            ExplanationMethod.FEATURE_IMPORTANCE,
        ]
        self.max_sessions = max_sessions
        self.session_timeout = timedelta(hours=session_timeout_hours)

        # Data storage
        self.anomaly_records: dict[str, AnomalyRecord] = {}
        self.investigation_sessions: dict[str, InvestigationSession] = {}
        self.explanation_cache: dict[str, ExplanationResult] = {}

        # Explainers
        self.shap_explainer = SHAPExplainer()

        # Analytics
        self.investigation_analytics = defaultdict(list)
        self.user_patterns = defaultdict(dict)

        # Background tasks
        self.background_tasks = set()

        logger.info("Initialized interactive investigation dashboard")

    async def create_investigation_session(
        self,
        user_id: str,
        investigation_type: InvestigationType,
        anomaly_ids: list[str],
        session_config: dict[str, Any] | None = None,
    ) -> str:
        """Create a new investigation session.

        Args:
            user_id: User identifier
            investigation_type: Type of investigation
            anomaly_ids: List of anomaly IDs to investigate
            session_config: Optional session configuration

        Returns:
            Session ID
        """
        # Clean up expired sessions
        await self._cleanup_expired_sessions()

        # Generate session ID
        session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"

        # Create session
        session = InvestigationSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            investigation_type=investigation_type,
            anomaly_ids=anomaly_ids,
            session_data=session_config or {},
        )

        self.investigation_sessions[session_id] = session

        # Initialize session with anomaly data
        await self._initialize_session_data(session)

        # Track analytics
        self.investigation_analytics["sessions_created"].append(
            {
                "timestamp": datetime.now(),
                "user_id": user_id,
                "investigation_type": investigation_type.value,
                "anomaly_count": len(anomaly_ids),
            }
        )

        logger.info(f"Created investigation session {session_id} for user {user_id}")
        return session_id

    async def _initialize_session_data(self, session: InvestigationSession) -> None:
        """Initialize session with relevant data and context."""
        # Load anomaly records
        session_anomalies = {}
        for anomaly_id in session.anomaly_ids:
            if anomaly_id in self.anomaly_records:
                session_anomalies[anomaly_id] = self.anomaly_records[anomaly_id]

        session.session_data["anomalies"] = session_anomalies

        # Generate initial summary
        if session_anomalies:
            summary = await self._generate_session_summary(session_anomalies)
            session.session_data["summary"] = summary

        # Set initial focus to highest severity anomaly
        if session_anomalies:
            severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            highest_severity = max(
                session_anomalies.values(),
                key=lambda a: severity_order.get(a.severity, 0),
            )
            session.current_focus = highest_severity.anomaly_id

    async def _generate_session_summary(
        self, anomalies: dict[str, AnomalyRecord]
    ) -> dict[str, Any]:
        """Generate summary statistics for the session."""
        if not anomalies:
            return {}

        anomaly_list = list(anomalies.values())

        # Time range
        timestamps = [a.timestamp for a in anomaly_list]
        time_range = {
            "start": min(timestamps),
            "end": max(timestamps),
            "duration_hours": (max(timestamps) - min(timestamps)).total_seconds()
            / 3600,
        }

        # Severity distribution
        severity_counts = defaultdict(int)
        for anomaly in anomaly_list:
            severity_counts[anomaly.severity] += 1

        # Score statistics
        scores = [a.anomaly_score for a in anomaly_list]
        score_stats = {
            "min": min(scores),
            "max": max(scores),
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores),
        }

        # Detector distribution
        detector_counts = defaultdict(int)
        for anomaly in anomaly_list:
            detector_counts[anomaly.detector_name] += 1

        return {
            "total_anomalies": len(anomaly_list),
            "time_range": time_range,
            "severity_distribution": dict(severity_counts),
            "score_statistics": score_stats,
            "detector_distribution": dict(detector_counts),
            "unique_features": len(
                set().union(*[a.feature_values.keys() for a in anomaly_list])
            ),
        }

    async def get_session_overview(self, session_id: str) -> dict[str, Any]:
        """Get overview of an investigation session."""
        if session_id not in self.investigation_sessions:
            return {"error": "Session not found"}

        session = self.investigation_sessions[session_id]

        # Basic session info
        overview = {
            "session_id": session_id,
            "user_id": session.user_id,
            "investigation_type": session.investigation_type.value,
            "start_time": session.start_time.isoformat(),
            "duration_minutes": (datetime.now() - session.start_time).total_seconds()
            / 60,
            "current_focus": session.current_focus,
            "anomaly_count": len(session.anomaly_ids),
            "analysis_steps_count": len(session.analysis_steps),
            "findings_count": len(session.findings),
            "hypotheses_count": len(session.hypotheses),
        }

        # Session summary
        if "summary" in session.session_data:
            overview["summary"] = session.session_data["summary"]

        # Recent activity
        overview["recent_analysis_steps"] = session.analysis_steps[-5:]
        overview["recent_findings"] = session.findings[-3:]

        # Available actions
        overview["available_actions"] = await self._get_available_actions(session)

        return overview

    async def _get_available_actions(
        self, session: InvestigationSession
    ) -> list[dict[str, Any]]:
        """Get available actions for the current session state."""
        actions = []

        # Always available actions
        actions.extend(
            [
                {
                    "action": "explain_anomaly",
                    "description": "Generate explanation for focused anomaly",
                    "requires_focus": True,
                },
                {
                    "action": "compare_anomalies",
                    "description": "Compare multiple anomalies",
                    "requires_focus": False,
                },
                {
                    "action": "temporal_analysis",
                    "description": "Analyze temporal patterns",
                    "requires_focus": False,
                },
                {
                    "action": "feature_analysis",
                    "description": "Analyze feature distributions",
                    "requires_focus": False,
                },
            ]
        )

        # Context-specific actions
        if session.current_focus:
            actions.append(
                {
                    "action": "drill_down",
                    "description": "Drill down into specific anomaly",
                    "requires_focus": True,
                }
            )

        if len(session.anomaly_ids) > 1:
            actions.append(
                {
                    "action": "cluster_analysis",
                    "description": "Find anomaly clusters",
                    "requires_focus": False,
                }
            )

        return actions

    async def explain_anomaly(
        self,
        session_id: str,
        anomaly_id: str,
        explanation_method: ExplanationMethod = ExplanationMethod.SHAP,
        detector: Detector | None = None,
        background_data: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Generate explanation for a specific anomaly.

        Args:
            session_id: Investigation session ID
            anomaly_id: Anomaly to explain
            explanation_method: Method to use for explanation
            detector: Detector that identified the anomaly
            background_data: Background data for comparison

        Returns:
            Explanation results
        """
        if session_id not in self.investigation_sessions:
            return {"error": "Session not found"}

        session = self.investigation_sessions[session_id]

        if anomaly_id not in self.anomaly_records:
            return {"error": "Anomaly not found"}

        anomaly = self.anomaly_records[anomaly_id]

        # Check cache
        cache_key = f"{anomaly_id}_{explanation_method.value}"
        if cache_key in self.explanation_cache:
            cached_result = self.explanation_cache[cache_key]
            # Update session
            session.current_focus = anomaly_id
            session.analysis_steps.append(
                {
                    "timestamp": datetime.now(),
                    "action": "explain_anomaly",
                    "method": explanation_method.value,
                    "anomaly_id": anomaly_id,
                    "cached": True,
                }
            )
            return {"explanation": cached_result, "cached": True}

        try:
            # Generate explanation
            if explanation_method == ExplanationMethod.SHAP and detector:
                # Convert anomaly feature values to numpy array
                feature_values = self._anomaly_to_array(anomaly)

                explanation = await self.shap_explainer.explain_anomaly(
                    detector=detector,
                    anomaly_data=feature_values,
                    background_data=background_data,
                    feature_names=list(anomaly.feature_values.keys()),
                )
                explanation.anomaly_id = anomaly_id

            elif explanation_method == ExplanationMethod.FEATURE_IMPORTANCE:
                explanation = await self._feature_importance_explanation(anomaly)

            else:
                explanation = ExplanationResult(
                    method=explanation_method,
                    anomaly_id=anomaly_id,
                    feature_contributions={},
                    explanation_text=f"Explanation method {explanation_method.value} not implemented",
                    confidence=0.0,
                )

            # Cache result
            self.explanation_cache[cache_key] = explanation

            # Update session
            session.current_focus = anomaly_id
            session.analysis_steps.append(
                {
                    "timestamp": datetime.now(),
                    "action": "explain_anomaly",
                    "method": explanation_method.value,
                    "anomaly_id": anomaly_id,
                    "confidence": explanation.confidence,
                }
            )

            # Add to findings if high confidence
            if explanation.confidence > 0.7:
                session.findings.append(
                    f"High-confidence explanation found for anomaly {anomaly_id}: "
                    f"{explanation.explanation_text[:100]}..."
                )

            return {"explanation": explanation, "cached": False}

        except Exception as e:
            logger.error(f"Error explaining anomaly {anomaly_id}: {e}")
            return {"error": str(e)}

    def _anomaly_to_array(self, anomaly: AnomalyRecord) -> np.ndarray:
        """Convert anomaly feature values to numpy array."""
        # Sort features by name for consistent ordering
        sorted_features = sorted(anomaly.feature_values.items())
        values = [float(value) for _, value in sorted_features]
        return np.array(values).reshape(1, -1)

    async def _feature_importance_explanation(
        self, anomaly: AnomalyRecord
    ) -> ExplanationResult:
        """Generate feature importance explanation."""
        # Simplified feature importance based on deviation from normal
        feature_contributions = {}

        for feature_name, value in anomaly.feature_values.items():
            # Simplified: use absolute value as importance
            # In practice, you'd compare against normal distribution
            normalized_value = (
                abs(float(value)) if isinstance(value, (int, float)) else 0.0
            )
            feature_contributions[feature_name] = normalized_value

        # Normalize contributions
        max_contrib = (
            max(feature_contributions.values()) if feature_contributions else 1.0
        )
        if max_contrib > 0:
            feature_contributions = {
                k: v / max_contrib for k, v in feature_contributions.items()
            }

        # Generate explanation text
        top_features = sorted(
            feature_contributions.items(), key=lambda x: x[1], reverse=True
        )[:3]

        explanation_text = "Feature importance analysis:\n"
        for feature, importance in top_features:
            explanation_text += f"• {feature}: {importance:.3f} importance\n"

        return ExplanationResult(
            method=ExplanationMethod.FEATURE_IMPORTANCE,
            anomaly_id=anomaly.anomaly_id,
            feature_contributions=feature_contributions,
            explanation_text=explanation_text,
            confidence=0.6,
        )

    async def compare_anomalies(
        self,
        session_id: str,
        anomaly_ids: list[str],
        comparison_metrics: list[str] = None,
    ) -> dict[str, Any]:
        """Compare multiple anomalies.

        Args:
            session_id: Investigation session ID
            anomaly_ids: List of anomalies to compare
            comparison_metrics: Metrics to use for comparison

        Returns:
            Comparison results
        """
        if session_id not in self.investigation_sessions:
            return {"error": "Session not found"}

        session = self.investigation_sessions[session_id]

        # Get anomaly records
        anomalies = []
        for anomaly_id in anomaly_ids:
            if anomaly_id in self.anomaly_records:
                anomalies.append(self.anomaly_records[anomaly_id])

        if len(anomalies) < 2:
            return {"error": "Need at least 2 anomalies for comparison"}

        comparison_metrics = comparison_metrics or [
            "anomaly_score",
            "timestamp",
            "severity",
            "detector_name",
        ]

        # Perform comparison
        comparison_result = {
            "anomaly_count": len(anomalies),
            "comparison_metrics": comparison_metrics,
            "similarities": {},
            "differences": {},
            "patterns": {},
            "visualizations": [],
        }

        # Calculate similarities and differences
        await self._calculate_anomaly_similarities(anomalies, comparison_result)
        await self._identify_anomaly_patterns(anomalies, comparison_result)

        # Create comparison visualizations
        comparison_visualizations = await self._create_comparison_visualizations(
            anomalies
        )
        comparison_result["visualizations"] = comparison_visualizations

        # Update session
        session.analysis_steps.append(
            {
                "timestamp": datetime.now(),
                "action": "compare_anomalies",
                "anomaly_ids": anomaly_ids,
                "comparison_metrics": comparison_metrics,
            }
        )

        return comparison_result

    async def _calculate_anomaly_similarities(
        self, anomalies: list[AnomalyRecord], result: dict[str, Any]
    ) -> None:
        """Calculate similarities between anomalies."""
        similarities = {}

        # Score similarity
        scores = [a.anomaly_score for a in anomalies]
        similarities["score_variance"] = float(np.var(scores))
        similarities["score_range"] = float(max(scores) - min(scores))

        # Temporal similarity
        timestamps = [a.timestamp for a in anomalies]
        time_diffs = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        if time_diffs:
            similarities["avg_time_diff_minutes"] = np.mean(time_diffs) / 60

        # Feature similarity
        common_features = set(anomalies[0].feature_values.keys())
        for anomaly in anomalies[1:]:
            common_features &= set(anomaly.feature_values.keys())

        similarities["common_features"] = list(common_features)
        similarities["feature_overlap_ratio"] = len(common_features) / len(
            set().union(*[a.feature_values.keys() for a in anomalies])
        )

        # Detector similarity
        detectors = [a.detector_name for a in anomalies]
        similarities["same_detector"] = len(set(detectors)) == 1
        similarities["detector_diversity"] = len(set(detectors)) / len(detectors)

        result["similarities"] = similarities

    async def _identify_anomaly_patterns(
        self, anomalies: list[AnomalyRecord], result: dict[str, Any]
    ) -> None:
        """Identify patterns in anomalies."""
        patterns = {}

        # Temporal patterns
        timestamps = [a.timestamp for a in anomalies]
        if len(timestamps) > 1:
            # Check for regular intervals
            intervals = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            if intervals:
                interval_std = np.std(intervals)
                patterns["regular_intervals"] = interval_std < np.mean(intervals) * 0.1
                patterns["avg_interval_minutes"] = np.mean(intervals) / 60

        # Severity patterns
        severities = [a.severity for a in anomalies]
        severity_counts = defaultdict(int)
        for severity in severities:
            severity_counts[severity] += 1
        patterns["severity_distribution"] = dict(severity_counts)

        # Score patterns
        scores = [a.anomaly_score for a in anomalies]
        if len(scores) > 2:
            # Check for increasing/decreasing trend
            score_diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
            avg_diff = np.mean(score_diffs)
            if abs(avg_diff) > 0.01:
                patterns["score_trend"] = "increasing" if avg_diff > 0 else "decreasing"
                patterns["trend_strength"] = abs(avg_diff)

        result["patterns"] = patterns

    async def _create_comparison_visualizations(
        self, anomalies: list[AnomalyRecord]
    ) -> list[dict[str, Any]]:
        """Create visualizations for anomaly comparison."""
        visualizations = []

        # Score comparison
        score_chart = {
            "type": "bar_chart",
            "title": "Anomaly Score Comparison",
            "data": {
                "anomaly_ids": [a.anomaly_id for a in anomalies],
                "scores": [a.anomaly_score for a in anomalies],
                "severities": [a.severity for a in anomalies],
            },
        }
        visualizations.append(score_chart)

        # Timeline visualization
        timeline_chart = {
            "type": "timeline",
            "title": "Anomaly Timeline",
            "data": {
                "timestamps": [a.timestamp.isoformat() for a in anomalies],
                "anomaly_ids": [a.anomaly_id for a in anomalies],
                "scores": [a.anomaly_score for a in anomalies],
            },
        }
        visualizations.append(timeline_chart)

        # Feature heatmap (if common features exist)
        common_features = set(anomalies[0].feature_values.keys())
        for anomaly in anomalies[1:]:
            common_features &= set(anomaly.feature_values.keys())

        if common_features:
            heatmap_data = []
            for anomaly in anomalies:
                feature_row = []
                for feature in sorted(common_features):
                    value = anomaly.feature_values.get(feature, 0)
                    feature_row.append(
                        float(value) if isinstance(value, (int, float)) else 0.0
                    )
                heatmap_data.append(feature_row)

            heatmap_chart = {
                "type": "heatmap",
                "title": "Feature Value Heatmap",
                "data": {
                    "anomaly_ids": [a.anomaly_id for a in anomalies],
                    "features": sorted(common_features),
                    "values": heatmap_data,
                },
            }
            visualizations.append(heatmap_chart)

        return visualizations

    async def temporal_analysis(
        self,
        session_id: str,
        time_window_hours: int = 24,
        analysis_type: str = "pattern_detection",
    ) -> dict[str, Any]:
        """Perform temporal analysis of anomalies.

        Args:
            session_id: Investigation session ID
            time_window_hours: Time window for analysis
            analysis_type: Type of temporal analysis

        Returns:
            Temporal analysis results
        """
        if session_id not in self.investigation_sessions:
            return {"error": "Session not found"}

        session = self.investigation_sessions[session_id]

        # Get anomalies in time window
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)

        windowed_anomalies = []
        for anomaly_id in session.anomaly_ids:
            if anomaly_id in self.anomaly_records:
                anomaly = self.anomaly_records[anomaly_id]
                if start_time <= anomaly.timestamp <= end_time:
                    windowed_anomalies.append(anomaly)

        if not windowed_anomalies:
            return {"error": "No anomalies found in specified time window"}

        # Sort by timestamp
        windowed_anomalies.sort(key=lambda a: a.timestamp)

        # Perform temporal analysis
        analysis_result = {
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": time_window_hours,
            },
            "anomaly_count": len(windowed_anomalies),
            "temporal_patterns": {},
            "frequency_analysis": {},
            "correlations": {},
            "visualizations": [],
        }

        # Analyze temporal patterns
        await self._analyze_temporal_patterns(windowed_anomalies, analysis_result)

        # Analyze frequency
        await self._analyze_anomaly_frequency(windowed_anomalies, analysis_result)

        # Create temporal visualizations
        temporal_visualizations = await self._create_temporal_visualizations(
            windowed_anomalies
        )
        analysis_result["visualizations"] = temporal_visualizations

        # Update session
        session.analysis_steps.append(
            {
                "timestamp": datetime.now(),
                "action": "temporal_analysis",
                "time_window_hours": time_window_hours,
                "anomalies_analyzed": len(windowed_anomalies),
            }
        )

        return analysis_result

    async def _analyze_temporal_patterns(
        self, anomalies: list[AnomalyRecord], result: dict[str, Any]
    ) -> None:
        """Analyze temporal patterns in anomalies."""
        if len(anomalies) < 2:
            return

        patterns = {}

        # Calculate time intervals
        intervals = []
        for i in range(len(anomalies) - 1):
            interval = (
                anomalies[i + 1].timestamp - anomalies[i].timestamp
            ).total_seconds()
            intervals.append(interval)

        if intervals:
            patterns["avg_interval_minutes"] = np.mean(intervals) / 60
            patterns["interval_std_minutes"] = np.std(intervals) / 60
            patterns["min_interval_minutes"] = min(intervals) / 60
            patterns["max_interval_minutes"] = max(intervals) / 60

            # Check for burst patterns (multiple anomalies in short time)
            short_intervals = [i for i in intervals if i < 300]  # 5 minutes
            patterns["burst_count"] = len(short_intervals)
            patterns["burst_ratio"] = len(short_intervals) / len(intervals)

        # Analyze hourly distribution
        hours = [a.timestamp.hour for a in anomalies]
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1

        patterns["hourly_distribution"] = dict(hour_counts)
        patterns["peak_hour"] = (
            max(hour_counts, key=hour_counts.get) if hour_counts else None
        )

        # Analyze day-of-week distribution
        weekdays = [a.timestamp.weekday() for a in anomalies]
        weekday_counts = defaultdict(int)
        for weekday in weekdays:
            weekday_counts[weekday] += 1

        patterns["weekday_distribution"] = dict(weekday_counts)

        result["temporal_patterns"] = patterns

    async def _analyze_anomaly_frequency(
        self, anomalies: list[AnomalyRecord], result: dict[str, Any]
    ) -> None:
        """Analyze frequency of anomalies."""
        if not anomalies:
            return

        # Group anomalies by time buckets
        bucket_size_minutes = 60  # 1-hour buckets
        buckets = defaultdict(int)

        for anomaly in anomalies:
            # Round timestamp to nearest bucket
            bucket_time = anomaly.timestamp.replace(minute=0, second=0, microsecond=0)
            bucket_key = bucket_time.isoformat()
            buckets[bucket_key] += 1

        frequency_data = {
            "bucket_size_minutes": bucket_size_minutes,
            "buckets": dict(buckets),
            "max_frequency": max(buckets.values()) if buckets else 0,
            "avg_frequency": np.mean(list(buckets.values())) if buckets else 0,
        }

        result["frequency_analysis"] = frequency_data

    async def _create_temporal_visualizations(
        self, anomalies: list[AnomalyRecord]
    ) -> list[dict[str, Any]]:
        """Create temporal visualizations."""
        visualizations = []

        # Timeline chart
        timeline = {
            "type": "timeline",
            "title": "Anomaly Timeline",
            "data": {
                "timestamps": [a.timestamp.isoformat() for a in anomalies],
                "scores": [a.anomaly_score for a in anomalies],
                "severities": [a.severity for a in anomalies],
                "anomaly_ids": [a.anomaly_id for a in anomalies],
            },
        }
        visualizations.append(timeline)

        # Hourly distribution
        hours = [a.timestamp.hour for a in anomalies]
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1

        hourly_dist = {
            "type": "bar_chart",
            "title": "Hourly Distribution of Anomalies",
            "data": {
                "hours": list(range(24)),
                "counts": [hour_counts[h] for h in range(24)],
            },
        }
        visualizations.append(hourly_dist)

        # Score over time
        score_timeline = {
            "type": "line_chart",
            "title": "Anomaly Scores Over Time",
            "data": {
                "timestamps": [a.timestamp.isoformat() for a in anomalies],
                "scores": [a.anomaly_score for a in anomalies],
            },
        }
        visualizations.append(score_timeline)

        return visualizations

    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired investigation sessions."""
        current_time = datetime.now()
        expired_sessions = []

        for session_id, session in self.investigation_sessions.items():
            if current_time - session.start_time > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.investigation_sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")

    async def get_dashboard_analytics(self) -> dict[str, Any]:
        """Get analytics about dashboard usage."""
        current_time = datetime.now()

        # Session statistics
        active_sessions = len(
            [
                s
                for s in self.investigation_sessions.values()
                if current_time - s.start_time < self.session_timeout
            ]
        )

        # Investigation patterns
        investigation_types = defaultdict(int)
        for session in self.investigation_sessions.values():
            investigation_types[session.investigation_type.value] += 1

        # User activity
        user_sessions = defaultdict(int)
        for session in self.investigation_sessions.values():
            user_sessions[session.user_id] += 1

        # Explanation method usage
        explanation_usage = defaultdict(int)
        for step in self.investigation_analytics.get("explanation_requests", []):
            explanation_usage[step.get("method", "unknown")] += 1

        return {
            "session_statistics": {
                "total_sessions": len(self.investigation_sessions),
                "active_sessions": active_sessions,
                "average_session_duration": self._calculate_avg_session_duration(),
                "sessions_by_type": dict(investigation_types),
            },
            "user_activity": {
                "unique_users": len(user_sessions),
                "sessions_per_user": dict(user_sessions),
                "most_active_user": (
                    max(user_sessions, key=user_sessions.get) if user_sessions else None
                ),
            },
            "explanation_analytics": {
                "method_usage": dict(explanation_usage),
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "avg_explanation_confidence": self._calculate_avg_explanation_confidence(),
            },
            "anomaly_analytics": {
                "total_anomalies": len(self.anomaly_records),
                "anomalies_investigated": len(
                    set().union(
                        *[
                            session.anomaly_ids
                            for session in self.investigation_sessions.values()
                        ]
                    )
                ),
                "avg_anomalies_per_session": (
                    np.mean(
                        [
                            len(session.anomaly_ids)
                            for session in self.investigation_sessions.values()
                        ]
                    )
                    if self.investigation_sessions
                    else 0
                ),
            },
        }

    def _calculate_avg_session_duration(self) -> float:
        """Calculate average session duration in minutes."""
        if not self.investigation_sessions:
            return 0.0

        current_time = datetime.now()
        durations = []

        for session in self.investigation_sessions.values():
            duration = (current_time - session.start_time).total_seconds() / 60
            durations.append(duration)

        return np.mean(durations)

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate explanation cache hit rate."""
        total_requests = len(
            self.investigation_analytics.get("explanation_requests", [])
        )
        cached_requests = len(
            [
                req
                for req in self.investigation_analytics.get("explanation_requests", [])
                if req.get("cached", False)
            ]
        )

        return cached_requests / total_requests if total_requests > 0 else 0.0

    def _calculate_avg_explanation_confidence(self) -> float:
        """Calculate average explanation confidence."""
        confidences = [
            explanation.confidence for explanation in self.explanation_cache.values()
        ]

        return np.mean(confidences) if confidences else 0.0
