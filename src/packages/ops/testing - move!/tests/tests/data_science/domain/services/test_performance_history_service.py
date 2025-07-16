"""Tests for PerformanceHistoryService."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from packages.data_science.domain.services.performance_history_service import (
    PerformanceHistoryService,
)
from packages.data_science.domain.entities.model_performance_degradation import (
    ModelPerformanceDegradation,
    DegradationStatus,
    RecoveryAction,
)
from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask,
)
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationMetricType,
    DegradationSeverity,
)


class TestPerformanceHistoryService:
    """Test suite for PerformanceHistoryService."""
    
    @pytest.fixture
    def mock_history_repository(self):
        """Create mock history repository."""
        return AsyncMock()
    
    @pytest.fixture
    def history_service(self, mock_history_repository):
        """Create history service instance."""
        return PerformanceHistoryService(
            history_repository=mock_history_repository,
            default_retention_days=90
        )
    
    @pytest.fixture
    def sample_degradation_entity(self):
        """Create sample degradation entity."""
        baseline_metrics = ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.90,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            roc_auc=0.95,
            prediction_time_seconds=0.05,
        )
        
        degradation_metrics = [
            PerformanceDegradationMetrics(
                metric_type=DegradationMetricType.ACCURACY_DROP,
                threshold_value=0.85,
                baseline_value=0.90,
            ),
        ]
        
        return ModelPerformanceDegradation(
            model_id="test-model-123",
            model_name="Test Model",
            model_version="1.0.0",
            task_type=ModelTask.BINARY_CLASSIFICATION,
            baseline_metrics=baseline_metrics,
            degradation_metrics=degradation_metrics,
        )
    
    @pytest.fixture
    def sample_evaluation_result(self):
        """Create sample evaluation result."""
        return {
            "status": "degraded",
            "previous_status": "healthy",
            "degradations": [
                {
                    "metric_type": "accuracy_drop",
                    "degradation_percentage": 15.0,
                    "severity": "moderate",
                    "baseline_value": 0.90,
                    "current_value": 0.80,
                    "threshold_value": 0.85,
                }
            ],
            "overall_severity": "moderate",
            "consecutive_healthy_evaluations": 0,
            "should_alert": True,
            "recovery_actions_recommended": ["retrain_model"],
        }
    
    @pytest.fixture
    def sample_current_metrics(self):
        """Create sample current metrics."""
        return ModelPerformanceMetrics(
            task_type=ModelTask.BINARY_CLASSIFICATION,
            sample_size=1000,
            accuracy=0.80,
            precision=0.78,
            recall=0.82,
            f1_score=0.80,
            roc_auc=0.85,
            prediction_time_seconds=0.06,
        )
    
    @pytest.mark.asyncio
    async def test_record_degradation_event(
        self, 
        history_service, 
        mock_history_repository, 
        sample_degradation_entity,
        sample_evaluation_result,
        sample_current_metrics
    ):
        """Test recording degradation event."""
        await history_service.record_degradation_event(
            sample_degradation_entity,
            sample_evaluation_result,
            sample_current_metrics
        )
        
        mock_history_repository.save_degradation_event.assert_called_once()
        
        # Verify call arguments
        call_args = mock_history_repository.save_degradation_event.call_args
        assert call_args[0][0] == "test-model-123"  # model_id
        assert call_args[0][1]["model_name"] == "Test Model"
        assert call_args[0][1]["status"] == "degraded"
        assert call_args[0][1]["degradations"] == sample_evaluation_result["degradations"]
        assert call_args[0][2] == sample_current_metrics  # metrics
        assert isinstance(call_args[0][3], datetime)  # timestamp
    
    @pytest.mark.asyncio
    async def test_record_recovery_action(
        self, 
        history_service, 
        mock_history_repository
    ):
        """Test recording recovery action."""
        await history_service.record_recovery_action(
            "test-model-123",
            RecoveryAction.RETRAIN_MODEL,
            "test_user",
            True,
            {"reason": "performance degradation"}
        )
        
        mock_history_repository.save_degradation_event.assert_called_once()
        
        # Verify call arguments
        call_args = mock_history_repository.save_degradation_event.call_args
        assert call_args[0][0] == "test-model-123"  # model_id
        assert call_args[0][1]["event_type"] == "recovery_action"
        assert call_args[0][1]["action"] == "retrain_model"
        assert call_args[0][1]["initiated_by"] == "test_user"
        assert call_args[0][1]["success"] is True
        assert call_args[0][1]["context"]["reason"] == "performance degradation"
        assert call_args[0][2] is None  # No metrics for recovery actions
    
    @pytest.mark.asyncio
    async def test_get_degradation_timeline(
        self, 
        history_service, 
        mock_history_repository
    ):
        """Test getting degradation timeline."""
        # Mock history data
        mock_history_data = [
            {
                "timestamp": "2023-01-01T00:00:00",
                "status": "healthy",
                "degradations": [],
            },
            {
                "timestamp": "2023-01-02T00:00:00",
                "status": "degraded",
                "degradations": [{"metric_type": "accuracy_drop"}],
                "overall_severity": "moderate",
                "should_alert": True,
            },
            {
                "timestamp": "2023-01-03T00:00:00",
                "event_type": "recovery_action",
                "action": "retrain_model",
                "initiated_by": "test_user",
                "success": True,
                "context": {},
            },
        ]
        
        mock_history_repository.get_degradation_history.return_value = mock_history_data
        
        timeline = await history_service.get_degradation_timeline("test-model-123", 30)
        
        assert len(timeline) >= 2  # At least status change and degradation detection
        
        # Check for status change event
        status_change_events = [e for e in timeline if e.get("event_type") == "status_change"]
        assert len(status_change_events) >= 1
        
        # Check for degradation detection event
        degradation_events = [e for e in timeline if e.get("event_type") == "degradation_detected"]
        assert len(degradation_events) >= 1
        
        # Check for recovery action event
        recovery_events = [e for e in timeline if e.get("event_type") == "recovery_action"]
        assert len(recovery_events) >= 1
    
    @pytest.mark.asyncio
    async def test_analyze_degradation_patterns(
        self, 
        history_service, 
        mock_history_repository
    ):
        """Test analyzing degradation patterns."""
        # Mock repository responses
        mock_history_repository.get_degradation_summary.return_value = {
            "degradation_events": 5,
            "recovery_events": 2,
            "critical_events": 1,
        }
        
        mock_history_repository.get_degradation_trends.return_value = [
            {"timestamp": "2023-01-01T00:00:00", "degradation_percentage": 10.0},
            {"timestamp": "2023-01-02T00:00:00", "degradation_percentage": 15.0},
            {"timestamp": "2023-01-03T00:00:00", "degradation_percentage": 20.0},
        ]
        
        mock_history_repository.get_recovery_history.return_value = [
            {"action": "retrain_model", "success": True},
            {"action": "adjust_threshold", "success": False},
        ]
        
        patterns = await history_service.analyze_degradation_patterns("test-model-123", 30)
        
        assert patterns["model_id"] == "test-model-123"
        assert patterns["analysis_period_days"] == 30
        assert "summary" in patterns
        assert "metric_trends" in patterns
        assert "recovery_analysis" in patterns
        assert "patterns" in patterns
        assert "recommendations" in patterns
        
        # Verify repository calls
        mock_history_repository.get_degradation_summary.assert_called_once()
        mock_history_repository.get_recovery_history.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_performance_stability_score(
        self, 
        history_service, 
        mock_history_repository
    ):
        """Test calculating performance stability score."""
        # Mock history data with mixed healthy/degraded events
        mock_history_data = [
            {"degradations": []},  # healthy
            {"degradations": [{"severity": "minor"}]},  # degraded
            {"degradations": []},  # healthy
            {"degradations": []},  # healthy
            {"degradations": [{"severity": "critical"}], "overall_severity": "critical"},  # critical
        ]
        
        mock_history_repository.get_degradation_history.return_value = mock_history_data
        
        stability = await history_service.get_performance_stability_score("test-model-123", 30)
        
        assert "stability_score" in stability
        assert "stability_grade" in stability
        assert "metrics" in stability
        assert "analysis_period_days" in stability
        
        assert 0 <= stability["stability_score"] <= 1
        assert stability["stability_grade"] in ["A", "B", "C", "D", "F"]
        assert stability["metrics"]["total_evaluations"] == 5
        assert stability["metrics"]["degraded_evaluations"] == 2
        assert stability["metrics"]["critical_evaluations"] == 1
    
    @pytest.mark.asyncio
    async def test_get_performance_stability_score_no_history(
        self, 
        history_service, 
        mock_history_repository
    ):
        """Test stability score with no history."""
        mock_history_repository.get_degradation_history.return_value = []
        
        stability = await history_service.get_performance_stability_score("test-model-123", 30)
        
        assert stability["stability_score"] == 1.0
        assert stability["stability_grade"] == "A"
        assert stability["message"] == "No degradation history found"
    
    @pytest.mark.asyncio
    async def test_compare_model_performance_history(
        self, 
        history_service, 
        mock_history_repository
    ):
        """Test comparing model performance history."""
        # Mock individual model data
        mock_stability_data = {
            "stability_score": 0.85,
            "stability_grade": "B",
            "metrics": {"total_evaluations": 10, "degraded_evaluations": 2},
        }
        
        mock_patterns_data = {
            "patterns": {
                "frequent_degradations": False,
                "chronic_issues": False,
                "critical_incidents": True,
            }
        }
        
        # Mock the individual service calls
        history_service.get_performance_stability_score = AsyncMock(return_value=mock_stability_data)
        history_service.analyze_degradation_patterns = AsyncMock(return_value=mock_patterns_data)
        
        comparison = await history_service.compare_model_performance_history(
            ["model-1", "model-2"], 30
        )
        
        assert comparison["analysis_period_days"] == 30
        assert "summary_stats" in comparison
        assert "model_analyses" in comparison
        assert "ranked_models" in comparison
        assert "comparison_insights" in comparison
        
        assert comparison["summary_stats"]["total_models_analyzed"] == 2
        assert len(comparison["model_analyses"]) == 2
    
    @pytest.mark.asyncio
    async def test_cleanup_old_history(
        self, 
        history_service, 
        mock_history_repository
    ):
        """Test cleaning up old history."""
        mock_history_repository.cleanup_old_history.return_value = 50
        
        deleted_count = await history_service.cleanup_old_history()
        
        assert deleted_count == 50
        mock_history_repository.cleanup_old_history.assert_called_once_with(90)
    
    def test_analyze_metric_trends_improving(self, history_service):
        """Test metric trend analysis for improving trend."""
        trends = [
            {"timestamp": "2023-01-01", "degradation_percentage": 20.0},
            {"timestamp": "2023-01-02", "degradation_percentage": 15.0},
            {"timestamp": "2023-01-03", "degradation_percentage": 10.0},
            {"timestamp": "2023-01-04", "degradation_percentage": 5.0},
        ]
        
        analysis = history_service._analyze_metric_trends(trends)
        
        assert analysis["trend"] == "improving"
        assert analysis["slope"] < 0
        assert analysis["avg_degradation"] == 12.5
        assert analysis["max_degradation"] == 20.0
        assert len(analysis["recent_values"]) == 4
    
    def test_analyze_metric_trends_worsening(self, history_service):
        """Test metric trend analysis for worsening trend."""
        trends = [
            {"timestamp": "2023-01-01", "degradation_percentage": 5.0},
            {"timestamp": "2023-01-02", "degradation_percentage": 10.0},
            {"timestamp": "2023-01-03", "degradation_percentage": 15.0},
            {"timestamp": "2023-01-04", "degradation_percentage": 20.0},
        ]
        
        analysis = history_service._analyze_metric_trends(trends)
        
        assert analysis["trend"] == "worsening"
        assert analysis["slope"] > 0
        assert analysis["avg_degradation"] == 12.5
        assert analysis["max_degradation"] == 20.0
    
    def test_analyze_metric_trends_stable(self, history_service):
        """Test metric trend analysis for stable trend."""
        trends = [
            {"timestamp": "2023-01-01", "degradation_percentage": 10.0},
            {"timestamp": "2023-01-02", "degradation_percentage": 10.1},
            {"timestamp": "2023-01-03", "degradation_percentage": 9.9},
            {"timestamp": "2023-01-04", "degradation_percentage": 10.0},
        ]
        
        analysis = history_service._analyze_metric_trends(trends)
        
        assert analysis["trend"] == "stable"
        assert abs(analysis["slope"]) < 0.1
    
    def test_analyze_metric_trends_no_data(self, history_service):
        """Test metric trend analysis with no data."""
        analysis = history_service._analyze_metric_trends([])
        
        assert analysis["trend"] == "no_data"
        assert analysis["trend_strength"] == 0
    
    def test_analyze_metric_trends_insufficient_data(self, history_service):
        """Test metric trend analysis with insufficient data."""
        trends = [{"timestamp": "2023-01-01", "degradation_percentage": 10.0}]
        
        analysis = history_service._analyze_metric_trends(trends)
        
        assert analysis["trend"] == "insufficient_data"
        assert analysis["trend_strength"] == 0
    
    def test_analyze_recovery_effectiveness(self, history_service):
        """Test recovery effectiveness analysis."""
        recovery_history = [
            {"action": "retrain_model", "success": True},
            {"action": "retrain_model", "success": False},
            {"action": "adjust_threshold", "success": True},
            {"action": "adjust_threshold", "success": True},
        ]
        
        analysis = history_service._analyze_recovery_effectiveness(recovery_history)
        
        assert analysis["total_actions"] == 4
        assert analysis["successful_actions"] == 3
        assert analysis["success_rate"] == 0.75
        assert "action_effectiveness" in analysis
        assert analysis["action_effectiveness"]["retrain_model"]["success_rate"] == 0.5
        assert analysis["action_effectiveness"]["adjust_threshold"]["success_rate"] == 1.0
        assert analysis["most_effective_action"] == "adjust_threshold"
    
    def test_analyze_recovery_effectiveness_no_data(self, history_service):
        """Test recovery effectiveness analysis with no data."""
        analysis = history_service._analyze_recovery_effectiveness([])
        
        assert analysis["total_actions"] == 0
        assert analysis["success_rate"] == 0
        assert analysis["effectiveness"] == {}
    
    def test_identify_degradation_patterns(self, history_service):
        """Test degradation pattern identification."""
        summary = {
            "degradation_events": 15,
            "recovery_events": 5,
            "critical_events": 2,
        }
        
        metric_trends = {
            "accuracy_drop": {"trend": "worsening", "trend_strength": 0.8},
            "precision_drop": {"trend": "stable", "trend_strength": 0.2},
        }
        
        patterns = history_service._identify_degradation_patterns(summary, metric_trends)
        
        assert patterns["frequent_degradations"] is True
        assert patterns["chronic_issues"] is True
        assert patterns["recovery_cycles"] is True
        assert patterns["critical_incidents"] is True
        assert "accuracy_drop" in patterns["unstable_metrics"]
        assert patterns["primary_concern"] == "critical_incidents"
    
    def test_generate_recommendations_critical_incidents(self, history_service):
        """Test recommendation generation for critical incidents."""
        patterns = {
            "primary_concern": "critical_incidents",
            "frequent_degradations": False,
            "unstable_metrics": [],
        }
        
        recovery_analysis = {"success_rate": 0.8}
        
        recommendations = history_service._generate_recommendations(patterns, recovery_analysis)
        
        assert any("critical" in rec.lower() for rec in recommendations)
        assert any("investigate" in rec.lower() for rec in recommendations)
        assert any("rollback" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_chronic_degradation(self, history_service):
        """Test recommendation generation for chronic degradation."""
        patterns = {
            "primary_concern": "chronic_degradation",
            "frequent_degradations": False,
            "unstable_metrics": [],
        }
        
        recovery_analysis = {"success_rate": 0.8}
        
        recommendations = history_service._generate_recommendations(patterns, recovery_analysis)
        
        assert any("retraining" in rec.lower() for rec in recommendations)
        assert any("data quality" in rec.lower() for rec in recommendations)
        assert any("architecture" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_frequent_degradations(self, history_service):
        """Test recommendation generation for frequent degradations."""
        patterns = {
            "primary_concern": "stable",
            "frequent_degradations": True,
            "unstable_metrics": [],
        }
        
        recovery_analysis = {"success_rate": 0.8}
        
        recommendations = history_service._generate_recommendations(patterns, recovery_analysis)
        
        assert any("sensitivity" in rec.lower() for rec in recommendations)
        assert any("automated" in rec.lower() for rec in recommendations)
        assert any("baseline" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_low_recovery_success(self, history_service):
        """Test recommendation generation for low recovery success rate."""
        patterns = {
            "primary_concern": "stable",
            "frequent_degradations": False,
            "unstable_metrics": [],
        }
        
        recovery_analysis = {"success_rate": 0.3}
        
        recommendations = history_service._generate_recommendations(patterns, recovery_analysis)
        
        assert any("recovery action" in rec.lower() for rec in recommendations)
        assert any("training" in rec.lower() for rec in recommendations)
        assert any("granular" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_stable(self, history_service):
        """Test recommendation generation for stable model."""
        patterns = {
            "primary_concern": "stable",
            "frequent_degradations": False,
            "unstable_metrics": [],
        }
        
        recovery_analysis = {"success_rate": 0.8}
        
        recommendations = history_service._generate_recommendations(patterns, recovery_analysis)
        
        assert any("stable" in rec.lower() for rec in recommendations)
        assert any("continue" in rec.lower() for rec in recommendations)
    
    def test_calculate_healthy_streaks(self, history_service):
        """Test calculating healthy streaks."""
        history = [
            {"degradations": []},  # healthy
            {"degradations": []},  # healthy
            {"degradations": [{"type": "accuracy"}]},  # degraded
            {"degradations": []},  # healthy
            {"degradations": []},  # healthy
            {"degradations": []},  # healthy
        ]
        
        streaks = history_service._calculate_healthy_streaks(history)
        
        assert streaks == [2, 3]  # Two streaks: 2 healthy, then 3 healthy
    
    def test_calculate_healthy_streaks_all_healthy(self, history_service):
        """Test calculating healthy streaks for all healthy history."""
        history = [
            {"degradations": []},  # healthy
            {"degradations": []},  # healthy
            {"degradations": []},  # healthy
        ]
        
        streaks = history_service._calculate_healthy_streaks(history)
        
        assert streaks == [3]
    
    def test_calculate_healthy_streaks_no_healthy(self, history_service):
        """Test calculating healthy streaks with no healthy periods."""
        history = [
            {"degradations": [{"type": "accuracy"}]},  # degraded
            {"degradations": [{"type": "precision"}]},  # degraded
        ]
        
        streaks = history_service._calculate_healthy_streaks(history)
        
        assert streaks == [0]
    
    def test_generate_comparison_insights(self, history_service):
        """Test generating comparison insights."""
        model_analyses = {
            "model-1": {
                "stability": {"stability_score": 0.9},
                "patterns": {"patterns": {"frequent_degradations": True}},
            },
            "model-2": {
                "stability": {"stability_score": 0.7},
                "patterns": {"patterns": {"frequent_degradations": True}},
            },
            "model-3": {
                "error": "Failed to analyze",
            },
        }
        
        insights = history_service._generate_comparison_insights(model_analyses)
        
        assert len(insights) >= 3
        assert any("model-1" in insight for insight in insights)  # Best model
        assert any("model-2" in insight for insight in insights)  # Worst model
        assert any("frequent_degradations" in insight for insight in insights)  # Common issue
    
    def test_generate_comparison_insights_insufficient_models(self, history_service):
        """Test generating comparison insights with insufficient models."""
        model_analyses = {
            "model-1": {
                "stability": {"stability_score": 0.9},
                "patterns": {"patterns": {}},
            },
        }
        
        insights = history_service._generate_comparison_insights(model_analyses)
        
        assert insights == ["Insufficient valid models for comparison"]