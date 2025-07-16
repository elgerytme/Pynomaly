"""
Integration tests for self-healing data pipeline.
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import uuid

from data_quality.application.services.self_healing_data_pipeline import (
    SelfHealingDataPipeline,
    SelfHealingConfig,
    create_self_healing_pipeline
)
from data_quality.domain.entities.quality_anomaly import QualityAnomaly
from data_quality.application.services.pipeline_integration_framework import PipelineStage, PipelineContext


class TestSelfHealingDataPipeline:
    """Test suite for self-healing data pipeline integration."""
    
    @pytest.fixture
    async def pipeline_config(self):
        """Create test pipeline configuration."""
        return {
            "monitoring": {
                "monitoring_interval": 10,
                "prediction_horizon": 300,
                "anomaly_threshold": 0.95,
                "learning_enabled": True
            },
            "remediation": {
                "max_concurrent_remediations": 5,
                "auto_approval_threshold": 0.9,
                "rollback_on_failure": True,
                "learning_enabled": True
            },
            "adaptive_controls": {
                "adaptation_interval": 60,
                "learning_rate": 0.1,
                "adaptation_strategy": "context_aware"
            },
            "pipeline": {
                "max_execution_time": 300,
                "quality_threshold": 0.7,
                "circuit_breakers": {
                    "data_quality_check": {
                        "failure_threshold": 3,
                        "recovery_timeout": 60
                    }
                }
            },
            "orchestration": {
                "max_concurrent_workflows": 10,
                "workflow_timeout": 1800,
                "resource_optimization_interval": 300
            },
            "self_monitoring": {
                "monitoring_interval": 30,
                "optimization_interval": 300,
                "alert_retention_days": 7,
                "auto_optimization_enabled": True
            },
            "auto_healing_enabled": True,
            "learning_enabled": True,
            "human_oversight_required": False,
            "performance_optimization_enabled": True
        }
    
    @pytest.fixture
    async def self_healing_pipeline(self, pipeline_config):
        """Create test self-healing pipeline."""
        config = SelfHealingConfig(**pipeline_config)
        pipeline = SelfHealingDataPipeline(config)
        await pipeline.initialize()
        yield pipeline
        await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization."""
        config = SelfHealingConfig(**pipeline_config)
        pipeline = SelfHealingDataPipeline(config)
        
        assert not pipeline.initialized
        assert not pipeline.running
        assert pipeline.health_status == "initializing"
        
        await pipeline.initialize()
        
        assert pipeline.initialized
        assert pipeline.running
        assert pipeline.health_status == "healthy"
        
        await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_dataset_registration(self, self_healing_pipeline):
        """Test dataset registration for monitoring."""
        dataset_id = "test_dataset_001"
        initial_metrics = {
            "completeness": 0.95,
            "validity": 0.88,
            "consistency": 0.92,
            "uniqueness": 0.97
        }
        
        await self_healing_pipeline.register_dataset(dataset_id, initial_metrics)
        
        # Verify dataset is registered
        quality_state = await self_healing_pipeline.monitoring_service.get_quality_state(dataset_id)
        assert quality_state is not None
        assert quality_state.dataset_id == dataset_id
        assert quality_state.overall_score > 0.0
    
    @pytest.mark.asyncio
    async def test_metrics_update(self, self_healing_pipeline):
        """Test metrics update for monitored dataset."""
        dataset_id = "test_dataset_002"
        initial_metrics = {
            "completeness": 0.95,
            "validity": 0.88,
            "consistency": 0.92,
            "uniqueness": 0.97
        }
        
        await self_healing_pipeline.register_dataset(dataset_id, initial_metrics)
        
        # Update metrics
        updated_metrics = {
            "completeness": 0.85,  # Decreased
            "validity": 0.90,      # Increased
            "consistency": 0.88,   # Decreased
            "uniqueness": 0.95     # Decreased
        }
        
        await self_healing_pipeline.update_dataset_metrics(dataset_id, updated_metrics)
        
        # Verify metrics were updated
        quality_state = await self_healing_pipeline.monitoring_service.get_quality_state(dataset_id)
        assert quality_state is not None
        assert quality_state.metric_scores["completeness"] == 0.85
        assert quality_state.metric_scores["validity"] == 0.90
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_and_healing(self, self_healing_pipeline):
        """Test anomaly detection and automatic healing."""
        dataset_id = "test_dataset_003"
        initial_metrics = {
            "completeness": 0.95,
            "validity": 0.88,
            "consistency": 0.92,
            "uniqueness": 0.97
        }
        
        await self_healing_pipeline.register_dataset(dataset_id, initial_metrics)
        
        # Simulate poor quality metrics to trigger anomaly detection
        poor_metrics = {
            "completeness": 0.45,  # Very low - should trigger anomaly
            "validity": 0.40,      # Very low - should trigger anomaly
            "consistency": 0.50,   # Low - should trigger anomaly
            "uniqueness": 0.60     # Low - should trigger anomaly
        }
        
        await self_healing_pipeline.update_dataset_metrics(dataset_id, poor_metrics)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify anomalies were detected
        quality_state = await self_healing_pipeline.monitoring_service.get_quality_state(dataset_id)
        assert quality_state is not None
        
        # Check if healing was triggered (in practice, this would be more complex)
        assert self_healing_pipeline.total_issues_detected > 0
    
    @pytest.mark.asyncio
    async def test_manual_healing_trigger(self, self_healing_pipeline):
        """Test manual healing trigger."""
        dataset_id = "test_dataset_004"
        initial_metrics = {
            "completeness": 0.95,
            "validity": 0.88,
            "consistency": 0.92,
            "uniqueness": 0.97
        }
        
        await self_healing_pipeline.register_dataset(dataset_id, initial_metrics)
        
        # Trigger manual healing
        workflow_id = await self_healing_pipeline.trigger_manual_healing(
            dataset_id, "Data quality degradation detected manually"
        )
        
        assert workflow_id is not None
        assert self_healing_pipeline.total_manual_interventions > 0
        
        # Check workflow status
        workflow_status = await self_healing_pipeline.orchestration_service.get_workflow_status(workflow_id)
        assert workflow_status is not None
        assert workflow_status["name"] == "Data Quality Incident Response"
    
    @pytest.mark.asyncio
    async def test_pipeline_integration(self, self_healing_pipeline):
        """Test pipeline integration functionality."""
        # Register a test pipeline
        pipeline_config = {
            "dags": {
                "test_dag": {
                    "schedule_interval": "0 */6 * * *",
                    "catchup": False
                }
            }
        }
        
        success = await self_healing_pipeline.pipeline_framework.register_pipeline(
            "airflow", pipeline_config
        )
        assert success
        
        # Create pipeline context
        context = PipelineContext(
            pipeline_id="test_dag",
            stage=PipelineStage.QUALITY_CHECK,
            data_batch_id="batch_001",
            timestamp=datetime.utcnow(),
            metadata={"test": True}
        )
        
        # Execute quality check
        result = await self_healing_pipeline.pipeline_framework.execute_quality_check(
            "airflow", PipelineStage.QUALITY_CHECK, context
        )
        
        assert result is not None
        assert result.stage == PipelineStage.QUALITY_CHECK
        assert result.execution_time.total_seconds() > 0
    
    @pytest.mark.asyncio
    async def test_auto_healing_configuration(self, self_healing_pipeline):
        """Test auto-healing configuration."""
        # Initially enabled
        assert self_healing_pipeline.config.auto_healing_enabled
        
        # Disable auto-healing
        await self_healing_pipeline.configure_auto_healing(False, False)
        assert not self_healing_pipeline.config.auto_healing_enabled
        assert not self_healing_pipeline.config.learning_enabled
        
        # Re-enable auto-healing
        await self_healing_pipeline.configure_auto_healing(True, True)
        assert self_healing_pipeline.config.auto_healing_enabled
        assert self_healing_pipeline.config.learning_enabled
    
    @pytest.mark.asyncio
    async def test_dashboard_generation(self, self_healing_pipeline):
        """Test dashboard generation."""
        dashboard = await self_healing_pipeline.get_self_healing_dashboard()
        
        assert "system_status" in dashboard
        assert "performance_metrics" in dashboard
        assert "quality_monitoring" in dashboard
        assert "pipeline_health" in dashboard
        assert "orchestration" in dashboard
        assert "system_health" in dashboard
        assert "timestamp" in dashboard
        
        # Check system status
        system_status = dashboard["system_status"]
        assert system_status["health"] == "healthy"
        assert system_status["running"]
        assert system_status["initialized"]
        
        # Check performance metrics
        performance_metrics = dashboard["performance_metrics"]
        assert "automation_rate" in performance_metrics
        assert "success_rate" in performance_metrics
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, self_healing_pipeline):
        """Test optimization recommendations."""
        recommendations = await self_healing_pipeline.get_optimization_recommendations()
        
        # Should be a list (may be empty initially)
        assert isinstance(recommendations, list)
        
        # Wait for optimization system to generate recommendations
        await asyncio.sleep(2)
        
        recommendations = await self_healing_pipeline.get_optimization_recommendations()
        
        # Check structure if recommendations exist
        if recommendations:
            rec = recommendations[0]
            assert "recommendation_id" in rec
            assert "optimization_type" in rec
            assert "description" in rec
            assert "expected_improvement" in rec
            assert "cost_benefit_ratio" in rec
    
    @pytest.mark.asyncio
    async def test_healing_history(self, self_healing_pipeline):
        """Test healing history retrieval."""
        # Register dataset and create some history
        dataset_id = "test_dataset_005"
        initial_metrics = {
            "completeness": 0.95,
            "validity": 0.88,
            "consistency": 0.92,
            "uniqueness": 0.97
        }
        
        await self_healing_pipeline.register_dataset(dataset_id, initial_metrics)
        
        # Get healing history
        history = await self_healing_pipeline.get_healing_history(dataset_id)
        
        assert "remediation_history" in history
        assert "total_healing_actions" in history
        assert "automation_effectiveness" in history
        assert "timestamp" in history
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, self_healing_pipeline):
        """Test circuit breaker functionality."""
        # Get initial pipeline health
        pipeline_health = await self_healing_pipeline.pipeline_framework.get_pipeline_health()
        
        assert "circuit_breakers" in pipeline_health
        assert "overall_health" in pipeline_health
        
        # Circuit breakers should be initially closed
        for breaker_name, breaker_status in pipeline_health["circuit_breakers"].items():
            assert breaker_status["state"] == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, self_healing_pipeline):
        """Test performance optimization."""
        # Get initial recommendations
        recommendations = await self_healing_pipeline.get_optimization_recommendations()
        
        # If there are recommendations, try to apply one
        if recommendations:
            rec = recommendations[0]
            if rec.get("priority") == "low" and rec.get("cost_benefit_ratio", 0) > 2.0:
                success = await self_healing_pipeline.apply_optimization(rec["recommendation_id"])
                assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, self_healing_pipeline):
        """Test system health monitoring."""
        # Wait for health monitoring to collect data
        await asyncio.sleep(2)
        
        # Get system health report
        dashboard = await self_healing_pipeline.get_self_healing_dashboard()
        system_health = dashboard["system_health"]
        
        assert "overall_health" in system_health
        assert "components" in system_health
        assert "system_metrics" in system_health
        assert "alerts" in system_health
        
        # Check component health
        components = system_health["components"]
        assert len(components) > 0
        
        for component_name, component_health in components.items():
            assert "health_status" in component_health
            assert "performance_score" in component_health
            assert "last_check" in component_health
    
    @pytest.mark.asyncio
    async def test_cost_effectiveness_analysis(self, self_healing_pipeline):
        """Test cost-effectiveness analysis."""
        dashboard = await self_healing_pipeline.get_self_healing_dashboard()
        cost_effectiveness = dashboard["cost_effectiveness"]
        
        if cost_effectiveness:  # May be empty initially
            assert "cost_metrics" in cost_effectiveness
            assert "effectiveness_metrics" in cost_effectiveness
            assert "cost_effectiveness_ratio" in cost_effectiveness
    
    @pytest.mark.asyncio
    async def test_error_handling(self, self_healing_pipeline):
        """Test error handling in pipeline."""
        # Test invalid dataset registration
        with pytest.raises(Exception):
            await self_healing_pipeline.register_dataset("", {})
        
        # Test invalid metrics update
        with pytest.raises(Exception):
            await self_healing_pipeline.update_dataset_metrics("nonexistent_dataset", {"invalid": "data"})
    
    @pytest.mark.asyncio
    async def test_pipeline_factory_function(self, pipeline_config):
        """Test pipeline factory function."""
        pipeline = await create_self_healing_pipeline(pipeline_config)
        
        assert pipeline.initialized
        assert pipeline.running
        assert pipeline.health_status == "healthy"
        
        await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_pipeline_shutdown(self, self_healing_pipeline):
        """Test pipeline shutdown."""
        assert self_healing_pipeline.running
        assert self_healing_pipeline.health_status == "healthy"
        
        await self_healing_pipeline.shutdown()
        
        assert not self_healing_pipeline.running
        assert not self_healing_pipeline.initialized
        assert self_healing_pipeline.health_status == "stopped"


class TestSelfHealingPipelineIntegration:
    """Test suite for end-to-end pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_issue_resolution(self):
        """Test end-to-end quality issue detection and resolution."""
        # Create pipeline with minimal config
        config = {
            "monitoring": {"monitoring_interval": 1},
            "remediation": {"max_concurrent_remediations": 2},
            "auto_healing_enabled": True,
            "learning_enabled": True
        }
        
        pipeline = await create_self_healing_pipeline(config)
        
        try:
            # Register dataset
            dataset_id = "e2e_test_dataset"
            initial_metrics = {
                "completeness": 0.95,
                "validity": 0.90,
                "consistency": 0.85,
                "uniqueness": 0.98
            }
            
            await pipeline.register_dataset(dataset_id, initial_metrics)
            
            # Simulate quality degradation
            degraded_metrics = {
                "completeness": 0.45,  # Severely degraded
                "validity": 0.30,      # Severely degraded
                "consistency": 0.25,   # Severely degraded
                "uniqueness": 0.60     # Degraded
            }
            
            await pipeline.update_dataset_metrics(dataset_id, degraded_metrics)
            
            # Wait for processing
            await asyncio.sleep(5)
            
            # Check that healing was triggered
            dashboard = await pipeline.get_self_healing_dashboard()
            performance_metrics = dashboard["performance_metrics"]
            
            assert performance_metrics["total_issues_detected"] > 0
            
            # Check healing history
            history = await pipeline.get_healing_history(dataset_id)
            assert "remediation_history" in history
            
        finally:
            await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_multi_dataset_monitoring(self):
        """Test monitoring multiple datasets simultaneously."""
        config = {
            "monitoring": {"monitoring_interval": 1},
            "auto_healing_enabled": True
        }
        
        pipeline = await create_self_healing_pipeline(config)
        
        try:
            # Register multiple datasets
            datasets = [
                ("dataset_001", {"completeness": 0.95, "validity": 0.90}),
                ("dataset_002", {"completeness": 0.88, "validity": 0.85}),
                ("dataset_003", {"completeness": 0.92, "validity": 0.88})
            ]
            
            for dataset_id, metrics in datasets:
                await pipeline.register_dataset(dataset_id, metrics)
            
            # Update metrics for all datasets
            for dataset_id, _ in datasets:
                await pipeline.update_dataset_metrics(dataset_id, {
                    "completeness": 0.80,
                    "validity": 0.75
                })
            
            # Wait for processing
            await asyncio.sleep(3)
            
            # Check dashboard
            dashboard = await pipeline.get_self_healing_dashboard()
            quality_monitoring = dashboard["quality_monitoring"]
            
            assert quality_monitoring["total_datasets"] == 3
            assert len(quality_monitoring["datasets"]) == 3
            
        finally:
            await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_pipeline_integration_workflow(self):
        """Test integration with different pipeline types."""
        config = {
            "pipeline": {
                "airflow_dags": {"test_dag": {"schedule": "daily"}},
                "streaming_config": {"test_stream": {"batch_size": 1000}}
            }
        }
        
        pipeline = await create_self_healing_pipeline(config)
        
        try:
            # Test Airflow integration
            context = PipelineContext(
                pipeline_id="test_dag",
                stage=PipelineStage.INGESTION,
                data_batch_id="batch_001",
                timestamp=datetime.utcnow()
            )
            
            result = await pipeline.pipeline_framework.execute_quality_check(
                "airflow", PipelineStage.INGESTION, context
            )
            
            assert result is not None
            assert result.stage == PipelineStage.INGESTION
            
            # Test streaming integration
            streaming_context = PipelineContext(
                pipeline_id="test_stream",
                stage=PipelineStage.PROCESSING,
                data_batch_id="stream_batch_001",
                timestamp=datetime.utcnow()
            )
            
            streaming_result = await pipeline.pipeline_framework.execute_quality_check(
                "streaming", PipelineStage.PROCESSING, streaming_context
            )
            
            assert streaming_result is not None
            assert streaming_result.stage == PipelineStage.PROCESSING
            
        finally:
            await pipeline.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])