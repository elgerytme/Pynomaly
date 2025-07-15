import pytest
import pandas as pd
from uuid import UUID
from datetime import datetime

from data_transformation.application.dto.pipeline_result import (
    PipelineResult, StepResult, ValidationResult, ExecutionMetrics
)


class TestStepResult:
    def test_create_step_result(self):
        step_result = StepResult(
            step_id="test_step",
            step_type="cleaning",
            status="completed",
            execution_time=1.5,
            metadata={"rows_processed": 1000}
        )
        
        assert step_result.step_id == "test_step"
        assert step_result.step_type == "cleaning"
        assert step_result.status == "completed"
        assert step_result.execution_time == 1.5
        assert step_result.metadata["rows_processed"] == 1000
        assert step_result.error_message is None

    def test_step_result_with_error(self):
        step_result = StepResult(
            step_id="test_step",
            step_type="cleaning",
            status="failed",
            execution_time=0.5,
            error_message="Test error"
        )
        
        assert step_result.status == "failed"
        assert step_result.error_message == "Test error"

    def test_step_result_to_dict(self):
        step_result = StepResult(
            step_id="test_step",
            step_type="cleaning",
            status="completed",
            execution_time=1.5,
            metadata={"rows_processed": 1000}
        )
        
        result_dict = step_result.to_dict()
        assert result_dict["step_id"] == "test_step"
        assert result_dict["step_type"] == "cleaning"
        assert result_dict["status"] == "completed"
        assert result_dict["execution_time"] == 1.5
        assert result_dict["metadata"]["rows_processed"] == 1000


class TestValidationResult:
    def test_create_validation_result_valid(self):
        validation_result = ValidationResult(
            is_valid=True,
            score=0.95,
            checks_passed=8,
            total_checks=10,
            issues=[]
        )
        
        assert validation_result.is_valid is True
        assert validation_result.score == 0.95
        assert validation_result.checks_passed == 8
        assert validation_result.total_checks == 10
        assert len(validation_result.issues) == 0

    def test_create_validation_result_invalid(self):
        issues = [
            {"type": "missing_values", "severity": "warning", "message": "Found missing values"},
            {"type": "outliers", "severity": "error", "message": "Detected outliers"}
        ]
        
        validation_result = ValidationResult(
            is_valid=False,
            score=0.6,
            checks_passed=6,
            total_checks=10,
            issues=issues
        )
        
        assert validation_result.is_valid is False
        assert validation_result.score == 0.6
        assert len(validation_result.issues) == 2

    def test_validation_result_to_dict(self):
        validation_result = ValidationResult(
            is_valid=True,
            score=0.95,
            checks_passed=8,
            total_checks=10,
            issues=[]
        )
        
        result_dict = validation_result.to_dict()
        assert result_dict["is_valid"] is True
        assert result_dict["score"] == 0.95
        assert result_dict["checks_passed"] == 8
        assert result_dict["total_checks"] == 10


class TestExecutionMetrics:
    def test_create_execution_metrics(self):
        metrics = ExecutionMetrics(
            total_execution_time=10.5,
            memory_usage_mb=256.7,
            cpu_usage_percent=45.2,
            rows_processed=10000,
            columns_processed=25,
            data_size_mb=128.3
        )
        
        assert metrics.total_execution_time == 10.5
        assert metrics.memory_usage_mb == 256.7
        assert metrics.cpu_usage_percent == 45.2
        assert metrics.rows_processed == 10000
        assert metrics.columns_processed == 25
        assert metrics.data_size_mb == 128.3

    def test_execution_metrics_optional_fields(self):
        metrics = ExecutionMetrics(
            total_execution_time=10.5,
            rows_processed=10000
        )
        
        assert metrics.total_execution_time == 10.5
        assert metrics.rows_processed == 10000
        assert metrics.memory_usage_mb is None
        assert metrics.cpu_usage_percent is None

    def test_execution_metrics_to_dict(self):
        metrics = ExecutionMetrics(
            total_execution_time=10.5,
            memory_usage_mb=256.7,
            rows_processed=10000,
            columns_processed=25
        )
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict["total_execution_time"] == 10.5
        assert metrics_dict["memory_usage_mb"] == 256.7
        assert metrics_dict["rows_processed"] == 10000
        assert metrics_dict["columns_processed"] == 25


class TestPipelineResult:
    def test_create_successful_pipeline_result(self, sample_dataframe):
        pipeline_result = PipelineResult(
            success=True,
            data=sample_dataframe,
            pipeline_id="test-pipeline-123",
            execution_time=5.2,
            steps_executed=[
                StepResult("step1", "loading", "completed", 1.0),
                StepResult("step2", "cleaning", "completed", 2.0)
            ]
        )
        
        assert pipeline_result.success is True
        assert pipeline_result.data is not None
        assert pipeline_result.pipeline_id == "test-pipeline-123"
        assert pipeline_result.execution_time == 5.2
        assert len(pipeline_result.steps_executed) == 2
        assert pipeline_result.error_message is None

    def test_create_failed_pipeline_result(self):
        pipeline_result = PipelineResult(
            success=False,
            data=None,
            pipeline_id="test-pipeline-123",
            execution_time=1.0,
            error_message="Pipeline failed during cleaning step",
            steps_executed=[
                StepResult("step1", "loading", "completed", 1.0)
            ]
        )
        
        assert pipeline_result.success is False
        assert pipeline_result.data is None
        assert pipeline_result.error_message == "Pipeline failed during cleaning step"
        assert len(pipeline_result.steps_executed) == 1

    def test_pipeline_result_with_validation(self, sample_dataframe):
        validation_result = ValidationResult(
            is_valid=True,
            score=0.95,
            checks_passed=8,
            total_checks=10,
            issues=[]
        )
        
        pipeline_result = PipelineResult(
            success=True,
            data=sample_dataframe,
            pipeline_id="test-pipeline-123",
            execution_time=5.2,
            steps_executed=[],
            validation_results=validation_result
        )
        
        assert pipeline_result.validation_results is not None
        assert pipeline_result.validation_results.is_valid is True

    def test_pipeline_result_with_metrics(self, sample_dataframe):
        metrics = ExecutionMetrics(
            total_execution_time=10.5,
            memory_usage_mb=256.7,
            rows_processed=10000,
            columns_processed=25
        )
        
        pipeline_result = PipelineResult(
            success=True,
            data=sample_dataframe,
            pipeline_id="test-pipeline-123",
            execution_time=5.2,
            steps_executed=[],
            metrics=metrics
        )
        
        assert pipeline_result.metrics is not None
        assert pipeline_result.metrics.total_execution_time == 10.5

    def test_pipeline_result_to_dict(self, sample_dataframe):
        step_results = [
            StepResult("step1", "loading", "completed", 1.0),
            StepResult("step2", "cleaning", "completed", 2.0)
        ]
        
        pipeline_result = PipelineResult(
            success=True,
            data=sample_dataframe,
            pipeline_id="test-pipeline-123",
            execution_time=5.2,
            steps_executed=step_results
        )
        
        result_dict = pipeline_result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["pipeline_id"] == "test-pipeline-123"
        assert result_dict["execution_time"] == 5.2
        assert len(result_dict["steps_executed"]) == 2
        assert "data_shape" in result_dict
        assert result_dict["data_shape"] == list(sample_dataframe.shape)

    def test_pipeline_result_to_dict_no_data(self):
        pipeline_result = PipelineResult(
            success=False,
            data=None,
            pipeline_id="test-pipeline-123",
            execution_time=1.0,
            error_message="Failed",
            steps_executed=[]
        )
        
        result_dict = pipeline_result.to_dict()
        assert result_dict["success"] is False
        assert result_dict["data_shape"] is None
        assert result_dict["error_message"] == "Failed"

    def test_pipeline_result_summary(self, sample_dataframe):
        step_results = [
            StepResult("step1", "loading", "completed", 1.0),
            StepResult("step2", "cleaning", "completed", 2.0),
            StepResult("step3", "scaling", "failed", 0.5, error_message="Error")
        ]
        
        pipeline_result = PipelineResult(
            success=False,
            data=sample_dataframe,
            pipeline_id="test-pipeline-123",
            execution_time=5.2,
            steps_executed=step_results,
            error_message="Pipeline failed"
        )
        
        summary = pipeline_result.get_summary()
        assert summary["success"] is False
        assert summary["total_steps"] == 3
        assert summary["completed_steps"] == 2
        assert summary["failed_steps"] == 1
        assert summary["success_rate"] == 2/3