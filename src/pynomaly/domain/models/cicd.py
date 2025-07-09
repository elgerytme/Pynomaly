"""CI/CD domain models for automated testing and deployment pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import numpy as np


class PipelineStatus(Enum):
    """Status of CI/CD pipeline execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class DeploymentEnvironment(Enum):
    """Deployment environments."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    PREVIEW = "preview"


class TriggerType(Enum):
    """Pipeline trigger types."""
    
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    SCHEDULE = "schedule"
    MANUAL = "manual"
    WEBHOOK = "webhook"
    TAG = "tag"
    RELEASE = "release"


class TestType(Enum):
    """Types of tests in CI/CD pipeline."""
    
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SMOKE = "smoke"
    REGRESSION = "regression"
    END_TO_END = "end_to_end"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


@dataclass
class TestResult:
    """Individual test result."""
    
    test_id: UUID
    test_name: str
    test_type: TestType
    status: PipelineStatus
    
    # Test execution details
    duration_seconds: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Test output
    output: str = ""
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Test metrics
    assertions_passed: int = 0
    assertions_failed: int = 0
    coverage_percentage: Optional[float] = None
    
    # Test artifacts
    artifacts: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    
    # Test metadata
    test_file: Optional[str] = None
    test_class: Optional[str] = None
    test_method: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.duration_seconds < 0:
            raise ValueError("Duration must be non-negative")
        if self.coverage_percentage is not None and not 0 <= self.coverage_percentage <= 100:
            raise ValueError("Coverage percentage must be between 0 and 100")
    
    def mark_completed(self, status: PipelineStatus, error_message: Optional[str] = None) -> None:
        """Mark test as completed."""
        self.status = status
        self.end_time = datetime.utcnow()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        if error_message:
            self.error_message = error_message
    
    def is_successful(self) -> bool:
        """Check if test was successful."""
        return self.status == PipelineStatus.SUCCESS


@dataclass
class TestSuite:
    """Collection of tests for a specific test type."""
    
    suite_id: UUID
    name: str
    test_type: TestType
    
    # Test configuration
    test_files: List[str] = field(default_factory=list)
    test_patterns: List[str] = field(default_factory=list)
    excluded_patterns: List[str] = field(default_factory=list)
    
    # Test execution
    tests: List[TestResult] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PENDING
    
    # Test metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    
    # Test timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Test environment
    test_environment: Dict[str, str] = field(default_factory=dict)
    test_command: str = ""
    working_directory: str = ""
    
    # Quality metrics
    overall_coverage: Optional[float] = None
    quality_gate_passed: bool = True
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Test suite name cannot be empty")
    
    def add_test_result(self, test_result: TestResult) -> None:
        """Add test result to suite."""
        self.tests.append(test_result)
        self.total_tests += 1
        
        if test_result.is_successful():
            self.passed_tests += 1
        elif test_result.status == PipelineStatus.FAILED:
            self.failed_tests += 1
        elif test_result.status == PipelineStatus.SKIPPED:
            self.skipped_tests += 1
    
    def calculate_metrics(self) -> None:
        """Calculate test suite metrics."""
        if self.tests:
            # Calculate overall coverage
            coverage_values = [t.coverage_percentage for t in self.tests if t.coverage_percentage is not None]
            if coverage_values:
                self.overall_coverage = sum(coverage_values) / len(coverage_values)
            
            # Update status based on test results
            if self.failed_tests > 0:
                self.status = PipelineStatus.FAILED
            elif self.passed_tests > 0 and self.failed_tests == 0:
                self.status = PipelineStatus.SUCCESS
            else:
                self.status = PipelineStatus.SKIPPED
    
    def get_success_rate(self) -> float:
        """Get test success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary."""
        return {
            "suite_id": str(self.suite_id),
            "name": self.name,
            "test_type": self.test_type.value,
            "status": self.status.value,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "success_rate": self.get_success_rate(),
            "duration_seconds": self.duration_seconds,
            "coverage": self.overall_coverage,
            "quality_gate_passed": self.quality_gate_passed,
        }


@dataclass
class PipelineStage:
    """Individual stage in CI/CD pipeline."""
    
    stage_id: UUID
    name: str
    stage_type: str  # "build", "test", "deploy", "quality_check", "security_scan"
    
    # Stage configuration
    depends_on: List[str] = field(default_factory=list)  # Stage names this depends on
    commands: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: str = ""
    timeout_minutes: int = 30
    
    # Execution status
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Stage output
    output: str = ""
    error_message: Optional[str] = None
    exit_code: Optional[int] = None
    
    # Test suites (for test stages)
    test_suites: List[TestSuite] = field(default_factory=list)
    
    # Artifacts
    artifacts: List[str] = field(default_factory=list)
    cache_key: Optional[str] = None
    
    # Retry configuration
    max_retries: int = 0
    retry_count: int = 0
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Stage name cannot be empty")
        if self.timeout_minutes <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
    
    def start_execution(self) -> None:
        """Mark stage as started."""
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.utcnow()
    
    def complete_execution(self, status: PipelineStatus, exit_code: Optional[int] = None, error_message: Optional[str] = None) -> None:
        """Mark stage as completed."""
        self.status = status
        self.end_time = datetime.utcnow()
        self.exit_code = exit_code
        self.error_message = error_message
        
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def should_retry(self) -> bool:
        """Check if stage should be retried."""
        return (
            self.status == PipelineStatus.FAILED and 
            self.retry_count < self.max_retries
        )
    
    def is_successful(self) -> bool:
        """Check if stage was successful."""
        return self.status == PipelineStatus.SUCCESS
    
    def add_test_suite(self, test_suite: TestSuite) -> None:
        """Add test suite to stage."""
        self.test_suites.append(test_suite)
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test suites in stage."""
        if not self.test_suites:
            return {}
        
        total_tests = sum(suite.total_tests for suite in self.test_suites)
        passed_tests = sum(suite.passed_tests for suite in self.test_suites)
        failed_tests = sum(suite.failed_tests for suite in self.test_suites)
        
        return {
            "test_suites": len(self.test_suites),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "suites": [suite.get_summary() for suite in self.test_suites],
        }


@dataclass
class Deployment:
    """Deployment record for tracking deployments."""
    
    deployment_id: UUID
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    
    # Deployment details
    version: str
    commit_sha: str
    branch: str
    
    # Deployment configuration
    replicas: int = 1
    cpu_request: str = "100m"
    memory_request: str = "128Mi"
    health_check_path: str = "/health"
    
    # Deployment status
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Rollback information
    previous_version: Optional[str] = None
    rollback_available: bool = True
    
    # Deployment metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    response_time_p95: float = 0.0
    
    # URLs and endpoints
    deployment_url: Optional[str] = None
    health_check_url: Optional[str] = None
    
    # Metadata
    deployed_by: UUID = field(default_factory=uuid4)
    deployment_notes: str = ""
    
    def __post_init__(self):
        if not self.version:
            raise ValueError("Version cannot be empty")
        if not self.commit_sha:
            raise ValueError("Commit SHA cannot be empty")
        if self.replicas <= 0:
            raise ValueError("Replicas must be positive")
    
    def start_deployment(self) -> None:
        """Mark deployment as started."""
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.utcnow()
    
    def complete_deployment(self, status: PipelineStatus, error_message: Optional[str] = None) -> None:
        """Mark deployment as completed."""
        self.status = status
        self.end_time = datetime.utcnow()
        
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def is_healthy(self) -> bool:
        """Check if deployment is healthy."""
        return (
            self.status == PipelineStatus.SUCCESS and
            self.error_rate < 5.0 and  # Less than 5% error rate
            self.response_time_p95 < 1000  # Less than 1 second P95
        )


@dataclass
class Pipeline:
    """Complete CI/CD pipeline definition and execution."""
    
    pipeline_id: UUID
    name: str
    description: str
    
    # Pipeline configuration
    repository_url: str
    branch: str = "main"
    trigger_type: TriggerType = TriggerType.PUSH
    
    # Pipeline structure
    stages: List[PipelineStage] = field(default_factory=list)
    parallel_stages: Dict[str, List[str]] = field(default_factory=dict)  # stage_name -> [parallel_stage_names]
    
    # Execution context
    commit_sha: str = ""
    commit_message: str = ""
    author: str = ""
    
    # Pipeline status
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Environment configuration
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)  # Would be encrypted in practice
    
    # Deployment tracking
    deployments: List[Deployment] = field(default_factory=list)
    
    # Pipeline metadata
    pipeline_number: int = 0
    triggered_by: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Quality gates
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    quality_gate_passed: bool = True
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")
        if not self.repository_url:
            raise ValueError("Repository URL cannot be empty")
    
    def add_stage(self, stage: PipelineStage) -> None:
        """Add stage to pipeline."""
        self.stages.append(stage)
    
    def start_pipeline(self) -> None:
        """Start pipeline execution."""
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.utcnow()
    
    def complete_pipeline(self, status: PipelineStatus) -> None:
        """Complete pipeline execution."""
        self.status = status
        self.end_time = datetime.utcnow()
        
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def get_stage_by_name(self, stage_name: str) -> Optional[PipelineStage]:
        """Get stage by name."""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def get_failed_stages(self) -> List[PipelineStage]:
        """Get list of failed stages."""
        return [stage for stage in self.stages if stage.status == PipelineStatus.FAILED]
    
    def get_successful_stages(self) -> List[PipelineStage]:
        """Get list of successful stages."""
        return [stage for stage in self.stages if stage.status == PipelineStatus.SUCCESS]
    
    def get_overall_test_summary(self) -> Dict[str, Any]:
        """Get overall test summary across all stages."""
        all_suites = []
        for stage in self.stages:
            all_suites.extend(stage.test_suites)
        
        if not all_suites:
            return {}
        
        total_tests = sum(suite.total_tests for suite in all_suites)
        passed_tests = sum(suite.passed_tests for suite in all_suites)
        failed_tests = sum(suite.failed_tests for suite in all_suites)
        
        # Calculate overall coverage
        coverage_values = [suite.overall_coverage for suite in all_suites if suite.overall_coverage is not None]
        overall_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else None
        
        return {
            "total_suites": len(all_suites),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "overall_coverage": overall_coverage,
            "quality_gate_passed": self.quality_gate_passed,
        }
    
    def add_deployment(self, deployment: Deployment) -> None:
        """Add deployment to pipeline."""
        self.deployments.append(deployment)
    
    def get_latest_deployment(self, environment: DeploymentEnvironment) -> Optional[Deployment]:
        """Get latest deployment for environment."""
        env_deployments = [d for d in self.deployments if d.environment == environment]
        if not env_deployments:
            return None
        
        return max(env_deployments, key=lambda d: d.start_time or datetime.min)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        return {
            "pipeline_id": str(self.pipeline_id),
            "name": self.name,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "commit_sha": self.commit_sha,
            "commit_message": self.commit_message,
            "author": self.author,
            "branch": self.branch,
            "trigger_type": self.trigger_type.value,
            "pipeline_number": self.pipeline_number,
            "created_at": self.created_at.isoformat(),
            "started_at": self.start_time.isoformat() if self.start_time else None,
            "completed_at": self.end_time.isoformat() if self.end_time else None,
            "stages": {
                "total": len(self.stages),
                "successful": len(self.get_successful_stages()),
                "failed": len(self.get_failed_stages()),
                "pending": len([s for s in self.stages if s.status == PipelineStatus.PENDING]),
                "running": len([s for s in self.stages if s.status == PipelineStatus.RUNNING]),
            },
            "tests": self.get_overall_test_summary(),
            "deployments": {
                "total": len(self.deployments),
                "by_environment": {
                    env.value: len([d for d in self.deployments if d.environment == env])
                    for env in DeploymentEnvironment
                },
            },
            "quality_gate_passed": self.quality_gate_passed,
        }


@dataclass
class PipelineTemplate:
    """Reusable pipeline template."""
    
    template_id: UUID
    name: str
    description: str
    template_type: str  # "web_app", "api", "ml_model", "microservice"
    
    # Template configuration
    stages_config: List[Dict[str, Any]] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    
    # Template metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: UUID = field(default_factory=uuid4)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Usage tracking
    usage_count: int = 0
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Template name cannot be empty")
        if not self.template_type:
            raise ValueError("Template type cannot be empty")
    
    def create_pipeline(
        self,
        pipeline_name: str,
        repository_url: str,
        branch: str = "main",
        commit_sha: str = "",
    ) -> Pipeline:
        """Create pipeline from template."""
        
        pipeline = Pipeline(
            pipeline_id=uuid4(),
            name=pipeline_name,
            description=f"Pipeline created from template: {self.name}",
            repository_url=repository_url,
            branch=branch,
            commit_sha=commit_sha,
            environment_variables=self.environment_variables.copy(),
            quality_gates=self.quality_gates.copy(),
        )
        
        # Create stages from template
        for stage_config in self.stages_config:
            stage = PipelineStage(
                stage_id=uuid4(),
                name=stage_config["name"],
                stage_type=stage_config["stage_type"],
                commands=stage_config.get("commands", []),
                environment=stage_config.get("environment", {}),
                working_directory=stage_config.get("working_directory", ""),
                timeout_minutes=stage_config.get("timeout_minutes", 30),
                depends_on=stage_config.get("depends_on", []),
                max_retries=stage_config.get("max_retries", 0),
            )
            pipeline.add_stage(stage)
        
        self.usage_count += 1
        self.updated_at = datetime.utcnow()
        
        return pipeline


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics and analytics."""
    
    metrics_id: UUID
    pipeline_id: UUID
    
    # Performance metrics
    average_duration_seconds: float = 0.0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    
    # Test metrics
    average_test_count: int = 0
    average_test_success_rate: float = 0.0
    average_coverage: Optional[float] = None
    
    # Deployment metrics
    deployment_frequency: float = 0.0  # deployments per week
    deployment_success_rate: float = 0.0
    mean_time_to_recovery: float = 0.0  # hours
    
    # Quality metrics
    quality_gate_pass_rate: float = 0.0
    security_scan_pass_rate: float = 0.0
    
    # Efficiency metrics
    stage_efficiency: Dict[str, float] = field(default_factory=dict)  # stage_name -> efficiency score
    resource_utilization: float = 0.0
    
    # Trend data
    execution_trend: List[Dict[str, Any]] = field(default_factory=list)  # Last 30 executions
    
    # Time period
    period_start: datetime = field(default_factory=lambda: datetime.utcnow() - timedelta(days=30))
    period_end: datetime = field(default_factory=datetime.utcnow)
    
    # Calculation metadata
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    sample_size: int = 0
    
    def calculate_metrics(self, pipeline_executions: List[Pipeline]) -> None:
        """Calculate metrics from pipeline executions."""
        
        if not pipeline_executions:
            return
        
        self.sample_size = len(pipeline_executions)
        
        # Performance metrics
        successful_pipelines = [p for p in pipeline_executions if p.status == PipelineStatus.SUCCESS]
        failed_pipelines = [p for p in pipeline_executions if p.status == PipelineStatus.FAILED]
        
        self.success_rate = (len(successful_pipelines) / len(pipeline_executions)) * 100
        self.failure_rate = (len(failed_pipelines) / len(pipeline_executions)) * 100
        
        # Average duration
        completed_pipelines = [p for p in pipeline_executions if p.duration_seconds > 0]
        if completed_pipelines:
            self.average_duration_seconds = sum(p.duration_seconds for p in completed_pipelines) / len(completed_pipelines)
        
        # Test metrics
        all_test_summaries = [p.get_overall_test_summary() for p in pipeline_executions]
        test_summaries_with_data = [ts for ts in all_test_summaries if ts]
        
        if test_summaries_with_data:
            self.average_test_count = sum(ts["total_tests"] for ts in test_summaries_with_data) / len(test_summaries_with_data)
            self.average_test_success_rate = sum(ts["success_rate"] for ts in test_summaries_with_data) / len(test_summaries_with_data)
            
            coverage_values = [ts["overall_coverage"] for ts in test_summaries_with_data if ts["overall_coverage"] is not None]
            if coverage_values:
                self.average_coverage = sum(coverage_values) / len(coverage_values)
        
        # Deployment metrics
        all_deployments = [d for p in pipeline_executions for d in p.deployments]
        if all_deployments:
            # Calculate deployment frequency (deployments per week)
            time_span_weeks = (self.period_end - self.period_start).days / 7
            self.deployment_frequency = len(all_deployments) / max(time_span_weeks, 1)
            
            successful_deployments = [d for d in all_deployments if d.status == PipelineStatus.SUCCESS]
            self.deployment_success_rate = (len(successful_deployments) / len(all_deployments)) * 100
        
        # Quality metrics
        quality_passed_pipelines = [p for p in pipeline_executions if p.quality_gate_passed]
        self.quality_gate_pass_rate = (len(quality_passed_pipelines) / len(pipeline_executions)) * 100
        
        # Build execution trend
        self.execution_trend = []
        for pipeline in sorted(pipeline_executions[-30:], key=lambda p: p.created_at):
            self.execution_trend.append({
                "pipeline_id": str(pipeline.pipeline_id),
                "status": pipeline.status.value,
                "duration": pipeline.duration_seconds,
                "test_success_rate": pipeline.get_overall_test_summary().get("success_rate", 0),
                "timestamp": pipeline.created_at.isoformat(),
            })
        
        self.calculated_at = datetime.utcnow()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "metrics_id": str(self.metrics_id),
            "pipeline_id": str(self.pipeline_id),
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
                "sample_size": self.sample_size,
            },
            "performance": {
                "average_duration_seconds": self.average_duration_seconds,
                "success_rate": self.success_rate,
                "failure_rate": self.failure_rate,
            },
            "testing": {
                "average_test_count": self.average_test_count,
                "average_test_success_rate": self.average_test_success_rate,
                "average_coverage": self.average_coverage,
            },
            "deployment": {
                "frequency_per_week": self.deployment_frequency,
                "success_rate": self.deployment_success_rate,
                "mean_time_to_recovery_hours": self.mean_time_to_recovery,
            },
            "quality": {
                "quality_gate_pass_rate": self.quality_gate_pass_rate,
                "security_scan_pass_rate": self.security_scan_pass_rate,
            },
            "efficiency": {
                "resource_utilization": self.resource_utilization,
                "stage_efficiency": self.stage_efficiency,
            },
            "calculated_at": self.calculated_at.isoformat(),
        }