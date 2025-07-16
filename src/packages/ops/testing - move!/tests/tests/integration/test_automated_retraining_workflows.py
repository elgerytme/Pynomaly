"""
Integration tests for Automated Retraining Workflows (Issue #9)

These tests verify the complete workflow orchestration for automated model retraining.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.monorepo.application.services.automated_retraining_orchestrator import (
    AutomatedRetrainingOrchestrator,
    WorkflowConfig,
    WorkflowType,
    WorkflowStatus,
    WorkflowExecution
)
from src.monorepo.application.services.retraining_workflow_templates import (
    RetrainingWorkflowTemplates,
    WorkflowConfigurationHelper
)
from src.monorepo.application.services.automated_retraining_service import (
    AutomatedRetrainingService,
    RetrainingTrigger,
    RetrainingResult
)


class TestAutomatedRetrainingOrchestrator:
    """Test suite for the automated retraining orchestrator"""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing"""
        return {
            'retraining_service': AsyncMock(spec=AutomatedRetrainingService),
            'performance_degradation_service': AsyncMock(),
            'intelligent_retraining_service': AsyncMock(),
            'monitoring_service': AsyncMock(),
            'pipeline_orchestration_service': AsyncMock()
        }
    
    @pytest.fixture
    def orchestrator(self, mock_services):
        """Create orchestrator instance with mock services"""
        return AutomatedRetrainingOrchestrator(
            retraining_service=mock_services['retraining_service'],
            performance_degradation_service=mock_services['performance_degradation_service'],
            intelligent_retraining_service=mock_services['intelligent_retraining_service'],
            monitoring_service=mock_services['monitoring_service'],
            pipeline_orchestration_service=mock_services['pipeline_orchestration_service']
        )
    
    @pytest.fixture
    def sample_workflow_config(self):
        """Create sample workflow configuration"""
        return WorkflowConfig(
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=["model_1", "model_2"],
            monitoring_interval_minutes=15,
            accuracy_threshold=0.05,
            max_concurrent_retraining=2,
            validation_dataset_size=1000,
            auto_rollback_enabled=True
        )
    
    @pytest.mark.asyncio
    async def test_register_workflow_success(self, orchestrator, sample_workflow_config):
        """Test successful workflow registration"""
        workflow_id = "test_workflow_1"
        
        success = await orchestrator.register_workflow(workflow_id, sample_workflow_config)
        
        assert success is True
        assert workflow_id in orchestrator.active_workflows
        assert orchestrator.active_workflows[workflow_id] == sample_workflow_config
    
    @pytest.mark.asyncio
    async def test_register_duplicate_workflow(self, orchestrator, sample_workflow_config):
        """Test registering duplicate workflow"""
        workflow_id = "test_workflow_1"
        
        # Register first time
        success1 = await orchestrator.register_workflow(workflow_id, sample_workflow_config)
        assert success1 is True
        
        # Register duplicate
        success2 = await orchestrator.register_workflow(workflow_id, sample_workflow_config)
        assert success2 is False
    
    @pytest.mark.asyncio
    async def test_performance_degradation_workflow(self, orchestrator, mock_services, sample_workflow_config):
        """Test performance degradation triggered workflow"""
        workflow_id = "perf_degradation_workflow"
        
        # Mock performance degradation detection
        mock_services['performance_degradation_service'].check_degradation.return_value = MagicMock(
            severity=MagicMock(MODERATE="moderate")
        )
        
        # Mock intelligent retraining decision
        mock_services['intelligent_retraining_service'].make_retraining_decision.return_value = MagicMock(
            should_retrain=True,
            confidence=0.8,
            reasoning="Performance degraded beyond threshold"
        )
        
        # Mock retraining execution
        mock_retraining_result = MagicMock(
            model_id="model_1",
            status="completed",
            performance_metrics={
                "baseline_accuracy": 0.85,
                "new_accuracy": 0.90
            },
            model_version="v2.0"
        )
        mock_services['retraining_service'].execute_retraining.return_value = mock_retraining_result
        
        # Mock deployment
        mock_services['pipeline_orchestration_service'].deploy_model.return_value = MagicMock(
            success=True
        )
        
        # Register and trigger workflow
        await orchestrator.register_workflow(workflow_id, sample_workflow_config)
        
        # Trigger workflow manually for testing
        await orchestrator._trigger_workflow(workflow_id, "model_1", sample_workflow_config)
        
        # Wait for workflow execution
        await asyncio.sleep(0.1)
        
        # Verify workflow execution
        execution = await orchestrator.get_workflow_status(workflow_id)
        assert execution is not None
        assert execution.workflow_type == WorkflowType.PERFORMANCE_DEGRADATION
        assert execution.trigger == RetrainingTrigger.PERFORMANCE_DEGRADATION
    
    @pytest.mark.asyncio
    async def test_workflow_validation_failure(self, orchestrator, mock_services, sample_workflow_config):
        """Test workflow handling validation failure"""
        workflow_id = "validation_failure_workflow"
        
        # Mock retraining result with poor performance
        mock_retraining_result = MagicMock(
            model_id="model_1",
            status="completed",
            performance_metrics={
                "baseline_accuracy": 0.85,
                "new_accuracy": 0.84  # Worse performance
            }
        )
        mock_services['retraining_service'].execute_retraining.return_value = mock_retraining_result
        
        # Mock intelligent decision
        mock_services['intelligent_retraining_service'].make_retraining_decision.return_value = MagicMock(
            should_retrain=True,
            confidence=0.7
        )
        
        # Create execution and test validation
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=["model_1"],
            trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
            status=WorkflowStatus.EXECUTING,
            started_at=datetime.utcnow(),
            retraining_results=[mock_retraining_result]
        )
        
        # Test validation failure
        validation_result = await orchestrator._validate_retraining_results(execution, sample_workflow_config)
        assert validation_result is False
    
    @pytest.mark.asyncio
    async def test_workflow_rollback(self, orchestrator, mock_services, sample_workflow_config):
        """Test workflow rollback functionality"""
        workflow_id = "rollback_workflow"
        
        # Mock rollback scenario
        mock_retraining_result = MagicMock(
            model_id="model_1",
            rollback_model_id="model_1_backup"
        )
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=["model_1"],
            trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
            status=WorkflowStatus.FAILED,
            started_at=datetime.utcnow(),
            retraining_results=[mock_retraining_result]
        )
        
        # Test rollback
        await orchestrator._rollback_workflow(execution, sample_workflow_config)
        
        # Verify rollback was called
        mock_services['pipeline_orchestration_service'].rollback_model.assert_called_once_with(
            "model_1", "model_1_backup"
        )
        assert execution.status == WorkflowStatus.ROLLED_BACK
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, orchestrator, mock_services):
        """Test concurrent workflow execution limits"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=["model_1", "model_2", "model_3"],
            max_concurrent_retraining=2
        )
        
        # Mock services
        mock_services['intelligent_retraining_service'].make_retraining_decision.return_value = MagicMock(
            should_retrain=True
        )
        mock_services['retraining_service'].execute_retraining.return_value = MagicMock(
            status="completed"
        )
        
        # Create multiple executions
        executions = []
        for i in range(3):
            execution = WorkflowExecution(
                workflow_id=f"workflow_{i}",
                workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
                model_ids=[f"model_{i}"],
                trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                status=WorkflowStatus.EXECUTING,
                started_at=datetime.utcnow()
            )
            executions.append(execution)
        
        # Test concurrent execution
        tasks = [orchestrator._execute_workflow(exec, config) for exec in executions]
        await asyncio.gather(*tasks)
        
        # Verify semaphore was respected (max 3 concurrent based on orchestrator default)
        assert len(mock_services['retraining_service'].execute_retraining.call_args_list) == 3
    
    @pytest.mark.asyncio
    async def test_workflow_context_manager(self, orchestrator, sample_workflow_config):
        """Test workflow context manager"""
        workflow_id = "context_workflow"
        
        async with orchestrator.workflow_context(workflow_id, sample_workflow_config):
            # Verify workflow is registered
            assert workflow_id in orchestrator.active_workflows
        
        # Verify workflow is stopped after context
        assert workflow_id not in orchestrator.active_workflows
    
    @pytest.mark.asyncio
    async def test_list_active_workflows(self, orchestrator, sample_workflow_config):
        """Test listing active workflows"""
        workflow_ids = ["workflow_1", "workflow_2", "workflow_3"]
        
        # Register multiple workflows
        for workflow_id in workflow_ids:
            await orchestrator.register_workflow(workflow_id, sample_workflow_config)
        
        # Get active workflows
        active_workflows = await orchestrator.list_active_workflows()
        
        assert len(active_workflows) == 3
        assert all(wf_id in active_workflows for wf_id in workflow_ids)
    
    @pytest.mark.asyncio
    async def test_stop_workflow(self, orchestrator, sample_workflow_config):
        """Test stopping a workflow"""
        workflow_id = "stop_workflow"
        
        # Register workflow
        await orchestrator.register_workflow(workflow_id, sample_workflow_config)
        assert workflow_id in orchestrator.active_workflows
        
        # Stop workflow
        success = await orchestrator.stop_workflow(workflow_id)
        
        assert success is True
        assert workflow_id not in orchestrator.active_workflows


class TestWorkflowTemplates:
    """Test suite for workflow templates"""
    
    def test_get_all_templates(self):
        """Test getting all workflow templates"""
        templates = RetrainingWorkflowTemplates.get_all_templates()
        
        assert len(templates) == 7
        assert all(template.name for template in templates)
        assert all(template.description for template in templates)
        assert all(template.default_config for template in templates)
    
    def test_get_template_by_name(self):
        """Test getting template by name"""
        template = RetrainingWorkflowTemplates.get_template_by_name("Performance Degradation Response")
        
        assert template is not None
        assert template.workflow_type == WorkflowType.PERFORMANCE_DEGRADATION
        assert len(template.use_cases) > 0
    
    def test_get_templates_by_use_case(self):
        """Test getting templates by use case"""
        templates = RetrainingWorkflowTemplates.get_templates_by_use_case("fraud detection")
        
        assert len(templates) > 0
        assert any("fraud" in " ".join(template.use_cases).lower() for template in templates)
    
    def test_customize_template(self):
        """Test customizing a template"""
        template = RetrainingWorkflowTemplates.performance_degradation_template()
        model_ids = ["model_1", "model_2"]
        custom_config = {
            "monitoring_interval_minutes": 30,
            "accuracy_threshold": 0.03,
            "business_rules": {
                "max_training_time": 8
            }
        }
        
        config = RetrainingWorkflowTemplates.customize_template(
            template, model_ids, custom_config
        )
        
        assert config.model_ids == model_ids
        assert config.monitoring_interval_minutes == 30
        assert config.accuracy_threshold == 0.03
        assert config.business_rules["max_training_time"] == 8
    
    def test_performance_degradation_template(self):
        """Test performance degradation template"""
        template = RetrainingWorkflowTemplates.performance_degradation_template()
        
        assert template.workflow_type == WorkflowType.PERFORMANCE_DEGRADATION
        assert template.default_config.monitoring_interval_minutes == 15
        assert template.default_config.accuracy_threshold == 0.05
        assert template.default_config.auto_rollback_enabled is True
        assert len(template.use_cases) > 0
        assert len(template.prerequisites) > 0
    
    def test_scheduled_maintenance_template(self):
        """Test scheduled maintenance template"""
        template = RetrainingWorkflowTemplates.scheduled_maintenance_template()
        
        assert template.workflow_type == WorkflowType.SCHEDULED_MAINTENANCE
        assert template.default_config.schedule_cron == "0 2 * * 0"
        assert template.default_config.champion_challenger_duration_hours == 72
        assert "weekend" in template.default_config.business_rules.get("maintenance_window", "")
    
    def test_data_drift_response_template(self):
        """Test data drift response template"""
        template = RetrainingWorkflowTemplates.data_drift_response_template()
        
        assert template.workflow_type == WorkflowType.DATA_DRIFT_RESPONSE
        assert template.default_config.drift_threshold == 0.1
        assert "ks_test" in template.default_config.business_rules.get("statistical_tests", [])
        assert "psi" in template.default_config.business_rules.get("statistical_tests", [])


class TestWorkflowConfigurationHelper:
    """Test suite for workflow configuration helper"""
    
    def test_validate_valid_configuration(self):
        """Test validation of valid configuration"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=["model_1"],
            monitoring_interval_minutes=15,
            accuracy_threshold=0.05,
            max_concurrent_retraining=2,
            validation_dataset_size=1000
        )
        
        issues = WorkflowConfigurationHelper.validate_configuration(config)
        assert len(issues) == 0
    
    def test_validate_invalid_configuration(self):
        """Test validation of invalid configuration"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=[],  # Empty model IDs
            monitoring_interval_minutes=0,  # Invalid interval
            accuracy_threshold=1.5,  # Invalid threshold
            max_concurrent_retraining=0,  # Invalid concurrency
            validation_dataset_size=50  # Too small
        )
        
        issues = WorkflowConfigurationHelper.validate_configuration(config)
        assert len(issues) >= 5
        assert any("Model IDs cannot be empty" in issue for issue in issues)
        assert any("Monitoring interval" in issue for issue in issues)
        assert any("Accuracy threshold" in issue for issue in issues)
        assert any("Max concurrent retraining" in issue for issue in issues)
        assert any("Validation dataset size" in issue for issue in issues)
    
    def test_estimate_resource_requirements(self):
        """Test resource requirements estimation"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=["model_1", "model_2", "model_3"],
            max_concurrent_retraining=2,
            business_rules={"max_training_time": 6}
        )
        
        resources = WorkflowConfigurationHelper.estimate_resource_requirements(config)
        
        assert resources["estimated_cpu_hours"] == 18  # 3 models * 6 hours
        assert resources["estimated_memory_gb"] == 16  # 2 concurrent * 8GB
        assert resources["estimated_storage_gb"] == 30  # 3 models * 10GB
        assert resources["concurrent_training_slots"] == 2
        assert resources["estimated_duration_hours"] == 9  # 18 hours / 2 concurrent
    
    def test_generate_configuration_summary(self):
        """Test configuration summary generation"""
        config = WorkflowConfig(
            workflow_type=WorkflowType.PERFORMANCE_DEGRADATION,
            model_ids=["model_1", "model_2"],
            monitoring_interval_minutes=15,
            max_concurrent_retraining=2,
            validation_dataset_size=1000,
            champion_challenger_duration_hours=24,
            auto_rollback_enabled=True,
            rollback_threshold=0.02,
            schedule_cron="0 2 * * 0"
        )
        
        summary = WorkflowConfigurationHelper.generate_configuration_summary(config)
        
        assert "performance_degradation" in summary
        assert "2 model(s)" in summary
        assert "Every 15 minutes" in summary
        assert "2 model(s)" in summary
        assert "1000 samples" in summary
        assert "24 hours" in summary
        assert "Enabled" in summary
        assert "0.02" in summary
        assert "0 2 * * 0" in summary


class TestWorkflowIntegration:
    """Integration tests for complete workflow scenarios"""
    
    @pytest.fixture
    def integration_orchestrator(self):
        """Create orchestrator with realistic mock services"""
        mock_services = {
            'retraining_service': AsyncMock(),
            'performance_degradation_service': AsyncMock(),
            'intelligent_retraining_service': AsyncMock(),
            'monitoring_service': AsyncMock(),
            'pipeline_orchestration_service': AsyncMock()
        }
        
        return AutomatedRetrainingOrchestrator(**mock_services)
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance_degradation_workflow(self, integration_orchestrator):
        """Test complete end-to-end performance degradation workflow"""
        # Setup template
        template = RetrainingWorkflowTemplates.performance_degradation_template()
        config = RetrainingWorkflowTemplates.customize_template(
            template,
            model_ids=["production_model_1"],
            custom_config={"monitoring_interval_minutes": 5}
        )
        
        # Register workflow
        workflow_id = "e2e_perf_degradation"
        success = await integration_orchestrator.register_workflow(workflow_id, config)
        assert success is True
        
        # Verify workflow is active
        active_workflows = await integration_orchestrator.list_active_workflows()
        assert workflow_id in active_workflows
        
        # Stop workflow
        stop_success = await integration_orchestrator.stop_workflow(workflow_id)
        assert stop_success is True
    
    @pytest.mark.asyncio
    async def test_multi_workflow_coordination(self, integration_orchestrator):
        """Test coordination between multiple workflows"""
        # Register multiple workflows
        workflows = [
            ("perf_workflow", RetrainingWorkflowTemplates.performance_degradation_template()),
            ("drift_workflow", RetrainingWorkflowTemplates.data_drift_response_template()),
            ("scheduled_workflow", RetrainingWorkflowTemplates.scheduled_maintenance_template())
        ]
        
        for workflow_id, template in workflows:
            config = RetrainingWorkflowTemplates.customize_template(
                template,
                model_ids=[f"model_{workflow_id}"]
            )
            await integration_orchestrator.register_workflow(workflow_id, config)
        
        # Verify all workflows are active
        active_workflows = await integration_orchestrator.list_active_workflows()
        assert len(active_workflows) == 3
        
        # Stop all workflows
        for workflow_id, _ in workflows:
            await integration_orchestrator.stop_workflow(workflow_id)
        
        # Verify all workflows are stopped
        active_workflows = await integration_orchestrator.list_active_workflows()
        assert len(active_workflows) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])