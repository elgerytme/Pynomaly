"""Performance history service for tracking and analyzing degradation trends."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from statistics import mean, stdev
from collections import defaultdict

from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask,
)
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    DegradationMetricType,
    DegradationSeverity,
)
from packages.data_science.domain.entities.model_performance_degradation import (
    ModelPerformanceDegradation,
    DegradationStatus,
    RecoveryAction,
)
from packages.data_science.infrastructure.repositories.performance_degradation_history_repository import (
    PerformanceDegradationHistoryRepository,
)


class PerformanceHistoryService:
    """Service for tracking and analyzing model performance degradation history.
    
    This service provides comprehensive functionality for storing, retrieving,
    and analyzing historical performance degradation data to identify patterns,
    trends, and insights for model monitoring and improvement.
    """
    
    def __init__(
        self,
        history_repository: PerformanceDegradationHistoryRepository,
        default_retention_days: int = 90,
    ):
        """Initialize the history service.
        
        Args:
            history_repository: Repository for degradation history data
            default_retention_days: Default retention period for history data
        """
        self.history_repository = history_repository
        self.default_retention_days = default_retention_days
    
    async def record_degradation_event(
        self,
        degradation: ModelPerformanceDegradation,
        evaluation_result: Dict[str, Any],
        metrics: ModelPerformanceMetrics,
    ) -> None:
        """Record a degradation evaluation event.
        
        Args:
            degradation: The degradation monitoring entity
            evaluation_result: Results from performance evaluation
            metrics: Current performance metrics
        """
        degradation_data = {
            "model_id": degradation.model_id,
            "model_name": degradation.model_name,
            "model_version": degradation.model_version,
            "task_type": degradation.task_type.value,
            "status": evaluation_result.get("status"),
            "previous_status": evaluation_result.get("previous_status"),
            "degradations": evaluation_result.get("degradations", []),
            "overall_severity": evaluation_result.get("overall_severity"),
            "consecutive_healthy_evaluations": evaluation_result.get("consecutive_healthy_evaluations", 0),
            "should_alert": evaluation_result.get("should_alert", False),
            "recovery_actions_recommended": evaluation_result.get("recovery_actions_recommended", []),
            "evaluation_interval_minutes": degradation.evaluation_interval_minutes,
            "auto_recovery_enabled": degradation.auto_recovery_enabled,
        }
        
        await self.history_repository.save_degradation_event(
            degradation.model_id,
            degradation_data,
            metrics,
            datetime.utcnow()
        )
    
    async def record_recovery_action(
        self,
        model_id: str,
        action: RecoveryAction,
        initiated_by: str,
        success: bool,
        context: Dict[str, Any] = None,
    ) -> None:
        """Record a recovery action taken.
        
        Args:
            model_id: ID of the model
            action: Recovery action taken
            initiated_by: Who initiated the action
            success: Whether the action was successful
            context: Additional context about the action
        """
        recovery_data = {
            "model_id": model_id,
            "action": action.value,
            "initiated_by": initiated_by,
            "success": success,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Store as a special degradation event with recovery action type
        await self.history_repository.save_degradation_event(
            model_id,
            {"event_type": "recovery_action", **recovery_data},
            None,  # No metrics for recovery actions
            datetime.utcnow()
        )
    
    async def get_degradation_timeline(
        self,
        model_id: str,
        days_back: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get degradation timeline for a model.
        
        Args:
            model_id: ID of the model
            days_back: Number of days to look back
            
        Returns:
            Timeline of degradation events
        """
        start_date = datetime.utcnow() - timedelta(days=days_back)
        history = await self.history_repository.get_degradation_history(
            model_id, start_date=start_date
        )
        
        # Process timeline to include status changes and key events
        timeline = []
        previous_status = None
        
        for event in history:
            current_status = event.get("status")
            
            # Add status change events
            if current_status != previous_status and previous_status is not None:
                timeline.append({
                    "timestamp": event.get("timestamp"),
                    "event_type": "status_change",
                    "from_status": previous_status,
                    "to_status": current_status,
                    "degradations": event.get("degradations", []),
                })
            
            # Add degradation detection events
            if event.get("degradations"):
                timeline.append({
                    "timestamp": event.get("timestamp"),
                    "event_type": "degradation_detected",
                    "status": current_status,
                    "degradations": event.get("degradations"),
                    "severity": event.get("overall_severity"),
                    "should_alert": event.get("should_alert", False),
                })
            
            # Add recovery events
            if event.get("event_type") == "recovery_action":
                timeline.append({
                    "timestamp": event.get("timestamp"),
                    "event_type": "recovery_action",
                    "action": event.get("action"),
                    "initiated_by": event.get("initiated_by"),
                    "success": event.get("success"),
                    "context": event.get("context", {}),
                })
            
            previous_status = current_status
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x.get("timestamp", ""))
        
        return timeline
    
    async def analyze_degradation_patterns(
        self,
        model_id: str,
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """Analyze degradation patterns for a model.
        
        Args:
            model_id: ID of the model
            days_back: Number of days to analyze
            
        Returns:
            Analysis of degradation patterns
        """
        summary = await self.history_repository.get_degradation_summary(
            model_id, days_back
        )
        
        # Get trends for each metric type
        metric_trends = {}
        for metric_type in DegradationMetricType:
            trends = await self.history_repository.get_degradation_trends(
                model_id, metric_type, days_back
            )
            if trends:
                metric_trends[metric_type.value] = self._analyze_metric_trends(trends)
        
        # Get recovery effectiveness
        recovery_history = await self.history_repository.get_recovery_history(
            model_id, days_back
        )
        recovery_analysis = self._analyze_recovery_effectiveness(recovery_history)
        
        # Calculate pattern metrics
        patterns = self._identify_degradation_patterns(summary, metric_trends)
        
        return {
            "model_id": model_id,
            "analysis_period_days": days_back,
            "summary": summary,
            "metric_trends": metric_trends,
            "recovery_analysis": recovery_analysis,
            "patterns": patterns,
            "recommendations": self._generate_recommendations(patterns, recovery_analysis),
        }
    
    async def get_performance_stability_score(
        self,
        model_id: str,
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """Calculate performance stability score for a model.
        
        Args:
            model_id: ID of the model
            days_back: Number of days to analyze
            
        Returns:
            Stability score and metrics
        """
        history = await self.history_repository.get_degradation_history(
            model_id, 
            start_date=datetime.utcnow() - timedelta(days=days_back)
        )
        
        if not history:
            return {
                "stability_score": 1.0,
                "stability_grade": "A",
                "metrics": {},
                "message": "No degradation history found"
            }
        
        # Calculate stability metrics
        total_evaluations = len(history)
        degraded_evaluations = len([h for h in history if h.get("degradations")])
        critical_evaluations = len([
            h for h in history 
            if h.get("overall_severity") == DegradationSeverity.CRITICAL.value
        ])
        
        # Calculate scores
        degradation_rate = degraded_evaluations / total_evaluations if total_evaluations > 0 else 0
        critical_rate = critical_evaluations / total_evaluations if total_evaluations > 0 else 0
        
        # Calculate consecutive healthy periods
        healthy_streaks = self._calculate_healthy_streaks(history)
        avg_healthy_streak = mean(healthy_streaks) if healthy_streaks else 0
        
        # Overall stability score (0-1, higher is better)
        stability_score = max(0, min(1, 
            1.0 - (degradation_rate * 0.6) - (critical_rate * 0.3) - 
            (1.0 / (avg_healthy_streak + 1)) * 0.1
        ))
        
        # Assign grade
        if stability_score >= 0.9:
            grade = "A"
        elif stability_score >= 0.8:
            grade = "B"
        elif stability_score >= 0.7:
            grade = "C"
        elif stability_score >= 0.6:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "stability_score": round(stability_score, 3),
            "stability_grade": grade,
            "metrics": {
                "total_evaluations": total_evaluations,
                "degraded_evaluations": degraded_evaluations,
                "critical_evaluations": critical_evaluations,
                "degradation_rate": round(degradation_rate, 3),
                "critical_rate": round(critical_rate, 3),
                "average_healthy_streak": round(avg_healthy_streak, 1),
                "max_healthy_streak": max(healthy_streaks) if healthy_streaks else 0,
            },
            "analysis_period_days": days_back,
        }
    
    async def compare_model_performance_history(
        self,
        model_ids: List[str],
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """Compare performance history across multiple models.
        
        Args:
            model_ids: List of model IDs to compare
            days_back: Number of days to analyze
            
        Returns:
            Comparative analysis of model performance
        """
        model_analyses = {}
        
        for model_id in model_ids:
            try:
                stability = await self.get_performance_stability_score(model_id, days_back)
                patterns = await self.analyze_degradation_patterns(model_id, days_back)
                
                model_analyses[model_id] = {
                    "stability": stability,
                    "patterns": patterns,
                }
            except Exception as e:
                model_analyses[model_id] = {
                    "error": str(e),
                    "stability": None,
                    "patterns": None,
                }
        
        # Rank models by stability
        valid_models = {
            k: v for k, v in model_analyses.items() 
            if v.get("stability") and "error" not in v
        }
        
        ranked_models = sorted(
            valid_models.items(),
            key=lambda x: x[1]["stability"]["stability_score"],
            reverse=True
        )
        
        # Calculate summary statistics
        stability_scores = [
            model["stability"]["stability_score"] 
            for model in valid_models.values()
        ]
        
        summary_stats = {
            "total_models_analyzed": len(model_ids),
            "valid_models": len(valid_models),
            "avg_stability_score": round(mean(stability_scores), 3) if stability_scores else 0,
            "stability_score_std": round(stdev(stability_scores), 3) if len(stability_scores) > 1 else 0,
            "best_performing_model": ranked_models[0][0] if ranked_models else None,
            "worst_performing_model": ranked_models[-1][0] if ranked_models else None,
        }
        
        return {
            "analysis_period_days": days_back,
            "summary_stats": summary_stats,
            "model_analyses": model_analyses,
            "ranked_models": [model_id for model_id, _ in ranked_models],
            "comparison_insights": self._generate_comparison_insights(model_analyses),
        }
    
    async def cleanup_old_history(self) -> int:
        """Clean up old history records."""
        return await self.history_repository.cleanup_old_history(self.default_retention_days)
    
    def _analyze_metric_trends(self, trends: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends for a specific metric."""
        if not trends:
            return {"trend": "no_data", "trend_strength": 0}
        
        values = [t.get("degradation_percentage", 0) for t in trends]
        timestamps = [t.get("timestamp") for t in trends]
        
        if len(values) < 2:
            return {"trend": "insufficient_data", "trend_strength": 0}
        
        # Simple linear trend analysis
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_xx = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
        
        # Determine trend direction and strength
        if abs(slope) < 0.1:
            trend = "stable"
        elif slope > 0:
            trend = "worsening"
        else:
            trend = "improving"
        
        trend_strength = min(1.0, abs(slope) / 10.0)  # Normalize to 0-1
        
        return {
            "trend": trend,
            "trend_strength": round(trend_strength, 3),
            "slope": round(slope, 3),
            "recent_values": values[-5:],  # Last 5 values
            "avg_degradation": round(mean(values), 2),
            "max_degradation": round(max(values), 2),
        }
    
    def _analyze_recovery_effectiveness(self, recovery_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effectiveness of recovery actions."""
        if not recovery_history:
            return {"total_actions": 0, "success_rate": 0, "effectiveness": {}}
        
        total_actions = len(recovery_history)
        successful_actions = len([r for r in recovery_history if r.get("success", False)])
        success_rate = successful_actions / total_actions if total_actions > 0 else 0
        
        # Analyze by action type
        action_effectiveness = defaultdict(lambda: {"total": 0, "successful": 0})
        
        for recovery in recovery_history:
            action = recovery.get("action")
            success = recovery.get("success", False)
            
            action_effectiveness[action]["total"] += 1
            if success:
                action_effectiveness[action]["successful"] += 1
        
        # Calculate success rates for each action
        for action_data in action_effectiveness.values():
            action_data["success_rate"] = (
                action_data["successful"] / action_data["total"] 
                if action_data["total"] > 0 else 0
            )
        
        return {
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": round(success_rate, 3),
            "action_effectiveness": dict(action_effectiveness),
            "most_effective_action": max(
                action_effectiveness.items(),
                key=lambda x: x[1]["success_rate"],
                default=(None, None)
            )[0] if action_effectiveness else None,
        }
    
    def _identify_degradation_patterns(
        self, 
        summary: Dict[str, Any], 
        metric_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify degradation patterns from summary and trends."""
        patterns = {
            "frequent_degradations": summary.get("degradation_events", 0) > 10,
            "chronic_issues": any(
                trend.get("trend") == "worsening" 
                for trend in metric_trends.values()
            ),
            "recovery_cycles": summary.get("recovery_events", 0) > 3,
            "critical_incidents": summary.get("critical_events", 0) > 0,
            "unstable_metrics": [
                metric for metric, trend in metric_trends.items()
                if trend.get("trend_strength", 0) > 0.5
            ],
        }
        
        # Identify primary concern
        if patterns["critical_incidents"]:
            primary_concern = "critical_incidents"
        elif patterns["chronic_issues"]:
            primary_concern = "chronic_degradation"
        elif patterns["frequent_degradations"]:
            primary_concern = "frequent_degradations"
        else:
            primary_concern = "stable"
        
        patterns["primary_concern"] = primary_concern
        
        return patterns
    
    def _generate_recommendations(
        self, 
        patterns: Dict[str, Any], 
        recovery_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on patterns and recovery analysis."""
        recommendations = []
        
        primary_concern = patterns.get("primary_concern")
        
        if primary_concern == "critical_incidents":
            recommendations.extend([
                "Investigate root causes of critical performance incidents",
                "Implement stricter monitoring thresholds",
                "Consider model rollback procedures for critical degradation",
            ])
        
        if primary_concern == "chronic_degradation":
            recommendations.extend([
                "Schedule model retraining with recent data",
                "Review feature engineering and data quality processes",
                "Evaluate model architecture for current use case",
            ])
        
        if patterns.get("frequent_degradations"):
            recommendations.extend([
                "Adjust monitoring sensitivity to reduce false positives",
                "Implement automated recovery actions for minor degradations",
                "Review baseline calculation methodology",
            ])
        
        if recovery_analysis.get("success_rate", 0) < 0.5:
            recommendations.extend([
                "Review and improve recovery action procedures",
                "Provide additional training on degradation response",
                "Implement more granular recovery action types",
            ])
        
        if patterns.get("unstable_metrics"):
            recommendations.append(
                f"Focus on stabilizing metrics: {', '.join(patterns['unstable_metrics'])}"
            )
        
        if not recommendations:
            recommendations.append("Model performance appears stable - continue current monitoring")
        
        return recommendations
    
    def _generate_comparison_insights(self, model_analyses: Dict[str, Any]) -> List[str]:
        """Generate insights from model comparison."""
        insights = []
        
        valid_models = {
            k: v for k, v in model_analyses.items() 
            if v.get("stability") and "error" not in v
        }
        
        if len(valid_models) < 2:
            return ["Insufficient valid models for comparison"]
        
        # Compare stability scores
        stability_scores = {
            model_id: analysis["stability"]["stability_score"]
            for model_id, analysis in valid_models.items()
        }
        
        best_model = max(stability_scores, key=stability_scores.get)
        worst_model = min(stability_scores, key=stability_scores.get)
        
        insights.append(f"Most stable model: {best_model} (score: {stability_scores[best_model]:.3f})")
        insights.append(f"Least stable model: {worst_model} (score: {stability_scores[worst_model]:.3f})")
        
        # Identify common issues
        common_patterns = defaultdict(int)
        for analysis in valid_models.values():
            for pattern, value in analysis["patterns"]["patterns"].items():
                if value:
                    common_patterns[pattern] += 1
        
        if common_patterns:
            most_common = max(common_patterns, key=common_patterns.get)
            insights.append(f"Most common issue: {most_common} (affects {common_patterns[most_common]} models)")
        
        return insights
    
    def _calculate_healthy_streaks(self, history: List[Dict[str, Any]]) -> List[int]:
        """Calculate consecutive healthy evaluation streaks."""
        streaks = []
        current_streak = 0
        
        for event in history:
            if not event.get("degradations"):
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        # Add final streak if healthy
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks if streaks else [0]