"""Integration tests for CI/CD pipeline system."""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pynomaly.domain.models.cicd import (
    DeploymentEnvironment,
    DeploymentStrategy,
    PipelineStatus,
    TestType,
    TriggerType,
)
from pynomaly.infrastructure.cicd.deployment_manager import DeploymentManager
from pynomaly.infrastructure.cicd.pipeline_service import PipelineService
from pynomaly.infrastructure.cicd.test_runner import TestRunner


@pytest.fixture
def pipeline_service():
    """Create pipeline service for testing."""
    return PipelineService(workspace_path="/tmp/test-integration-ci")


@pytest.fixture
def test_runner():
    """Create test runner for testing."""
    return TestRunner()


@pytest.fixture
def deployment_manager():
    """Create deployment manager for testing."""
    return DeploymentManager()


@pytest.fixture
def test_workspace(tmp_path):
    """Create test workspace with sample project structure."""
    # Create project structure
    (tmp_path / "src" / "pynomaly").mkdir(parents=True)
    (tmp_path / "src" / "pynomaly" / "__init__.py").touch()
    (tmp_path / "src" / "pynomaly" / "main.py").write_text("""
def hello_world():
    return "Hello, World!"
""")
    
    # Create test structure
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_main.py").write_text("""
import pytest
from pynomaly.main import hello_world

def test_hello_world():
    assert hello_world() == "Hello, World!"

def test_another():
    assert 1 + 1 == 2
""")
    
    (tmp_path / "tests" / "integration").mkdir(parents=True)
    (tmp_path / "tests" / "integration" / "test_api.py").write_text("""
def test_api_integration():
    # Mock integration test
    assert True
""")
    
    # Create configuration files
    (tmp_path / "pyproject.toml").write_text("""
[tool.poetry]
name = "test-project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.11"

[tool.poetry.group.dev.dependencies]
pytest = "^6.0"
pytest-cov = "^4.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
""")
    
    (tmp_path / "requirements.txt").write_text("""
pytest>=6.0
pytest-cov>=4.0
""")
    
    (tmp_path / "README.md").write_text("# Test Project")
    
    return tmp_path


class TestCICDIntegration:
    """Integration tests for complete CI/CD workflow."""
    
    @patch("pynomaly.infrastructure.cicd.pipeline_service.asyncio.create_subprocess_exec")
    @patch("pynomaly.infrastructure.cicd.pipeline_service.asyncio.create_subprocess_shell")
    async def test_complete_cicd_workflow(
        self,
        mock_subprocess_shell,
        mock_subprocess_exec,
        pipeline_service,
        test_runner,
        deployment_manager,
        test_workspace,
    ):
        """Test complete CI/CD workflow from pipeline creation to deployment."""
        
        # Mock subprocess calls for git and test execution
        mock_git_process = AsyncMock()
        mock_git_process.returncode = 0
        mock_git_process.communicate.return_value = (b"Cloning...", b"")
        mock_subprocess_exec.return_value = mock_git_process
        
        mock_test_process = AsyncMock()
        mock_test_process.returncode = 0
        mock_test_process.stdout.readline.side_effect = [
            b"test_main.py::test_hello_world PASSED",
            b"test_main.py::test_another PASSED",
            b"====== 2 passed in 1.0s ======",
            b"",
        ]
        mock_test_process.wait.return_value = None
        mock_subprocess_shell.return_value = mock_test_process
        
        # Start services
        await pipeline_service.start_pipeline_executor()
        await deployment_manager.start_monitoring()
        
        try:
            # 1. Create pipeline from template
            python_template_id = None
            for template_id, template in pipeline_service.pipeline_templates.items():
                if template.template_type == "python_package":
                    python_template_id = template_id
                    break
            
            assert python_template_id is not None
            
            pipeline = await pipeline_service.create_pipeline_from_template(
                template_id=python_template_id,
                pipeline_name="Test Integration Pipeline",
                repository_url="https://github.com/test/repo.git",
                branch="main",
                commit_sha="abc123",
                triggered_by=uuid4(),
                trigger_type=TriggerType.PUSH,
            )
            
            assert pipeline.name == "Test Integration Pipeline"
            assert pipeline.status == PipelineStatus.PENDING
            assert len(pipeline.stages) > 0
            
            # 2. Trigger pipeline execution
            success = await pipeline_service.trigger_pipeline(pipeline.pipeline_id, priority="high")
            assert success
            
            # 3. Execute pipeline (mocked)
            success = await pipeline_service.execute_pipeline(pipeline.pipeline_id)
            assert success
            assert pipeline.status == PipelineStatus.SUCCESS
            
            # 4. Verify stages completed
            successful_stages = pipeline.get_successful_stages()
            assert len(successful_stages) > 0
            
            # 5. Get pipeline status
            status = await pipeline_service.get_pipeline_status(pipeline.pipeline_id)
            assert status["pipeline"]["status"] == "success"
            
            # 6. Deploy to development environment
            deployment = await deployment_manager.deploy(
                environment=DeploymentEnvironment.DEVELOPMENT,
                version="v1.0.0",
                commit_sha=pipeline.commit_sha,
                branch=pipeline.branch,
                strategy=DeploymentStrategy.ROLLING,
                deployed_by=pipeline.triggered_by,
                deployment_notes="Automated deployment from CI/CD pipeline",
            )
            
            assert deployment.environment == DeploymentEnvironment.DEVELOPMENT
            assert deployment.version == "v1.0.0"
            assert deployment.status == PipelineStatus.SUCCESS
            
            # 7. Get deployment status
            deployment_status = await deployment_manager.get_deployment_status(
                environment=DeploymentEnvironment.DEVELOPMENT
            )
            assert deployment_status["current_deployment"]["version"] == "v1.0.0"
            assert deployment_status["current_deployment"]["status"] == "success"
            
            # 8. Test rollback functionality
            rollback_deployment = await deployment_manager.rollback(
                environment=DeploymentEnvironment.DEVELOPMENT,
                rolled_back_by=pipeline.triggered_by,
            )
            
            # Should fail because no previous deployment
            assert rollback_deployment is None
            
        finally:
            # Cleanup
            await pipeline_service.stop_pipeline_executor()
            await deployment_manager.stop_monitoring()
    
    async def test_test_runner_integration(self, test_runner, test_workspace):
        """Test test runner integration with real test files."""
        
        # 1. Validate test environment
        unit_validation = await test_runner.validate_test_environment(
            TestType.UNIT,
            test_workspace,
        )
        assert unit_validation["valid"] is True
        assert "pyproject.toml found" in unit_validation["requirements_met"]
        
        integration_validation = await test_runner.validate_test_environment(
            TestType.INTEGRATION,
            test_workspace,
        )
        assert integration_validation["valid"] is True
        
        # 2. Discover tests
        unit_tests = await test_runner.discover_tests(
            TestType.UNIT,
            test_workspace,
        )
        assert len(unit_tests) >= 1
        assert any("test_main.py" in test for test in unit_tests)
        
        integration_tests = await test_runner.discover_tests(
            TestType.INTEGRATION,
            test_workspace,
        )
        assert len(integration_tests) >= 1
        assert any("test_api.py" in test for test in integration_tests)
        
        # 3. Execute test suites (mocked)
        with patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b"test_main.py::test_hello_world PASSED\\ntest_main.py::test_another PASSED\\n====== 2 passed in 1.0s ======",
                b""
            )
            mock_subprocess.return_value = mock_process
            
            from pynomaly.domain.models.cicd import TestSuite
            
            unit_suite = TestSuite(
                suite_id=uuid4(),
                name="Unit Tests",
                test_type=TestType.UNIT,
                test_files=unit_tests,
            )
            
            success = await test_runner.execute_test_suite(
                unit_suite,
                test_workspace,
                {"PYTHONPATH": str(test_workspace)}
            )
            
            assert success
            assert unit_suite.status == PipelineStatus.SUCCESS
            assert unit_suite.passed_tests == 2
            assert unit_suite.failed_tests == 0
    
    async def test_deployment_strategies(self, deployment_manager):
        """Test different deployment strategies."""
        
        await deployment_manager.start_monitoring()
        
        try:
            # Test Rolling Deployment
            rolling_deployment = await deployment_manager.deploy(
                environment=DeploymentEnvironment.STAGING,
                version="v1.0.0",
                commit_sha="abc123",
                branch="main",
                strategy=DeploymentStrategy.ROLLING,
                deployed_by=uuid4(),
            )
            
            assert rolling_deployment.strategy == DeploymentStrategy.ROLLING
            assert rolling_deployment.status == PipelineStatus.SUCCESS
            
            # Test Blue-Green Deployment
            blue_green_deployment = await deployment_manager.deploy(
                environment=DeploymentEnvironment.PRODUCTION,
                version="v1.1.0",
                commit_sha="def456",
                branch="main",
                strategy=DeploymentStrategy.BLUE_GREEN,
                deployed_by=uuid4(),
            )
            
            assert blue_green_deployment.strategy == DeploymentStrategy.BLUE_GREEN
            assert blue_green_deployment.status == PipelineStatus.SUCCESS
            
            # Test Canary Deployment
            canary_deployment = await deployment_manager.deploy(
                environment=DeploymentEnvironment.CANARY,
                version="v1.2.0",
                commit_sha="ghi789",
                branch="main",
                strategy=DeploymentStrategy.CANARY,
                deployed_by=uuid4(),
            )
            
            assert canary_deployment.strategy == DeploymentStrategy.CANARY
            assert canary_deployment.status == PipelineStatus.SUCCESS
            
            # Test Rollback
            rollback = await deployment_manager.rollback(
                environment=DeploymentEnvironment.PRODUCTION,
                target_version="v1.0.0",
            )
            
            assert rollback is not None
            assert rollback.version == "v1.0.0"
            assert rollback.status == PipelineStatus.SUCCESS
            
        finally:
            await deployment_manager.stop_monitoring()
    
    async def test_pipeline_failure_handling(self, pipeline_service, test_workspace):
        """Test pipeline failure handling and recovery."""
        
        await pipeline_service.start_pipeline_executor()
        
        try:
            # Get template
            python_template_id = None
            for template_id, template in pipeline_service.pipeline_templates.items():
                if template.template_type == "python_package":
                    python_template_id = template_id
                    break
            
            # Create pipeline
            pipeline = await pipeline_service.create_pipeline_from_template(
                template_id=python_template_id,
                pipeline_name="Failure Test Pipeline",
                repository_url="https://github.com/test/repo.git",
            )
            
            # Mock failed execution
            with patch("pynomaly.infrastructure.cicd.pipeline_service.asyncio.create_subprocess_exec") as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.returncode = 1  # Failure
                mock_process.communicate.return_value = (b"", b"Git clone failed")
                mock_subprocess.return_value = mock_process
                
                # Execute pipeline (should fail)
                success = await pipeline_service.execute_pipeline(pipeline.pipeline_id)
                
                assert not success
                assert pipeline.status == PipelineStatus.FAILED
            
            # Test pipeline cancellation
            pipeline2 = await pipeline_service.create_pipeline_from_template(
                template_id=python_template_id,
                pipeline_name="Cancellation Test Pipeline",
                repository_url="https://github.com/test/repo.git",
            )
            
            # Mark as running and cancel
            pipeline_service.running_pipelines.add(pipeline2.pipeline_id)
            success = await pipeline_service.cancel_pipeline(pipeline2.pipeline_id)
            
            assert success
            assert pipeline2.status == PipelineStatus.CANCELLED
            
        finally:
            await pipeline_service.stop_pipeline_executor()
    
    async def test_metrics_and_monitoring(self, pipeline_service, deployment_manager):
        """Test metrics collection and monitoring."""
        
        await pipeline_service.start_pipeline_executor()
        await deployment_manager.start_monitoring()
        
        try:
            # Create and execute pipeline
            python_template_id = None
            for template_id, template in pipeline_service.pipeline_templates.items():
                if template.template_type == "python_package":
                    python_template_id = template_id
                    break
            
            pipeline = await pipeline_service.create_pipeline_from_template(
                template_id=python_template_id,
                pipeline_name="Metrics Test Pipeline",
                repository_url="https://github.com/test/repo.git",
            )
            
            # Complete pipeline
            pipeline.complete_pipeline(PipelineStatus.SUCCESS)
            
            # Calculate metrics
            await pipeline_service._calculate_pipeline_metrics(pipeline.pipeline_id)
            
            # Get metrics
            metrics = await pipeline_service.get_pipeline_metrics(pipeline.pipeline_id)
            assert metrics is not None
            assert metrics.sample_size >= 1
            
            # Create deployment
            deployment = await deployment_manager.deploy(
                environment=DeploymentEnvironment.DEVELOPMENT,
                version="v1.0.0",
                commit_sha="abc123",
                branch="main",
                strategy=DeploymentStrategy.ROLLING,
                deployed_by=uuid4(),
            )
            
            # Get deployment status
            status = await deployment_manager.get_deployment_status()
            assert status["total_deployments"] >= 1
            assert status["active_deployments"] == 0  # Should be completed
            
        finally:
            await pipeline_service.stop_pipeline_executor()
            await deployment_manager.stop_monitoring()
    
    async def test_concurrent_pipeline_execution(self, pipeline_service):
        """Test concurrent pipeline execution."""
        
        await pipeline_service.start_pipeline_executor()
        
        try:
            # Get template
            python_template_id = None
            for template_id, template in pipeline_service.pipeline_templates.items():
                if template.template_type == "python_package":
                    python_template_id = template_id
                    break
            
            # Create multiple pipelines
            pipelines = []
            for i in range(3):
                pipeline = await pipeline_service.create_pipeline_from_template(
                    template_id=python_template_id,
                    pipeline_name=f"Concurrent Pipeline {i}",
                    repository_url="https://github.com/test/repo.git",
                )
                pipelines.append(pipeline)
            
            # Trigger all pipelines
            for pipeline in pipelines:
                success = await pipeline_service.trigger_pipeline(pipeline.pipeline_id)
                assert success
            
            # Check queue has all pipelines
            total_queued = sum(len(queue) for queue in pipeline_service.pipeline_queues.values())
            assert total_queued == 3
            
            # Check pipeline numbers are unique
            pipeline_numbers = [p.pipeline_number for p in pipelines]
            assert len(set(pipeline_numbers)) == 3
            
        finally:
            await pipeline_service.stop_pipeline_executor()
    
    async def test_environment_progression(self, deployment_manager):
        """Test deployment progression through environments."""
        
        await deployment_manager.start_monitoring()
        
        try:
            environments = [
                DeploymentEnvironment.DEVELOPMENT,
                DeploymentEnvironment.TESTING,
                DeploymentEnvironment.STAGING,
                DeploymentEnvironment.PRODUCTION,
            ]
            
            version = "v1.0.0"
            commit_sha = "abc123"
            branch = "main"
            deployed_by = uuid4()
            
            # Deploy progressively through environments
            deployments = []
            for env in environments:
                deployment = await deployment_manager.deploy(
                    environment=env,
                    version=version,
                    commit_sha=commit_sha,
                    branch=branch,
                    strategy=DeploymentStrategy.ROLLING,
                    deployed_by=deployed_by,
                    deployment_notes=f"Deployment to {env.value}",
                )
                
                assert deployment.environment == env
                assert deployment.version == version
                assert deployment.status == PipelineStatus.SUCCESS
                deployments.append(deployment)
            
            # Verify all environments have the deployment
            for env in environments:
                status = await deployment_manager.get_deployment_status(environment=env)
                assert status["current_deployment"]["version"] == version
                assert status["current_deployment"]["environment"] == env.value
            
            # Test production rollback
            rollback = await deployment_manager.rollback(
                environment=DeploymentEnvironment.PRODUCTION,
            )
            
            # Should fail because no previous deployment
            assert rollback is None
            
        finally:
            await deployment_manager.stop_monitoring()


@pytest.mark.asyncio
class TestCICDErrorScenarios:
    """Test error scenarios and edge cases in CI/CD system."""
    
    async def test_invalid_repository_handling(self, pipeline_service):
        """Test handling of invalid repository URLs."""
        
        python_template_id = None
        for template_id, template in pipeline_service.pipeline_templates.items():
            if template.template_type == "python_package":
                python_template_id = template_id
                break
        
        # Create pipeline with invalid repo
        pipeline = await pipeline_service.create_pipeline_from_template(
            template_id=python_template_id,
            pipeline_name="Invalid Repo Pipeline",
            repository_url="invalid://repo.url",
        )
        
        # Mock git clone failure
        with patch("pynomaly.infrastructure.cicd.pipeline_service.asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 128  # Git error
            mock_process.communicate.return_value = (b"", b"fatal: repository not found")
            mock_subprocess.return_value = mock_process
            
            success = await pipeline_service.execute_pipeline(pipeline.pipeline_id)
            
            assert not success
            assert pipeline.status == PipelineStatus.FAILED
    
    async def test_deployment_health_check_failure(self, deployment_manager):
        """Test deployment with failing health checks."""
        
        await deployment_manager.start_monitoring()
        
        try:
            # Mock health check failure
            with patch.object(deployment_manager, '_perform_health_checks', return_value=False):
                deployment = await deployment_manager.deploy(
                    environment=DeploymentEnvironment.STAGING,
                    version="v1.0.0",
                    commit_sha="abc123",
                    branch="main",
                    strategy=DeploymentStrategy.ROLLING,
                    deployed_by=uuid4(),
                )
                
                assert deployment.status == PipelineStatus.FAILED
                # Should trigger rollback
        
        finally:
            await deployment_manager.stop_monitoring()
    
    async def test_test_execution_timeout(self, test_runner, test_workspace):
        """Test test execution timeout handling."""
        
        from pynomaly.domain.models.cicd import TestSuite
        
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Timeout Test Suite",
            test_type=TestType.UNIT,
        )
        
        # Mock long-running test
        with patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate.side_effect = asyncio.TimeoutError("Test timeout")
            mock_subprocess.return_value = mock_process
            
            success = await test_runner.execute_test_suite(
                test_suite,
                test_workspace,
            )
            
            assert not success
            assert test_suite.status == PipelineStatus.FAILED
    
    async def test_resource_cleanup(self, pipeline_service, deployment_manager):
        """Test proper resource cleanup."""
        
        await pipeline_service.start_pipeline_executor()
        await deployment_manager.start_monitoring()
        
        try:
            # Create some resources
            python_template_id = None
            for template_id, template in pipeline_service.pipeline_templates.items():
                if template.template_type == "python_package":
                    python_template_id = template_id
                    break
            
            pipeline = await pipeline_service.create_pipeline_from_template(
                template_id=python_template_id,
                pipeline_name="Cleanup Test Pipeline",
                repository_url="https://github.com/test/repo.git",
            )
            
            deployment = await deployment_manager.deploy(
                environment=DeploymentEnvironment.DEVELOPMENT,
                version="v1.0.0",
                commit_sha="abc123",
                branch="main",
                strategy=DeploymentStrategy.ROLLING,
                deployed_by=uuid4(),
            )
            
            # Verify resources exist
            assert pipeline.pipeline_id in pipeline_service.pipelines
            assert deployment.deployment_id in deployment_manager.deployments
            
            # Test cleanup
            await pipeline_service._cleanup_old_pipelines()
            await deployment_manager._cleanup_old_deployments()
            
            # Resources should still exist (not old enough)
            assert pipeline.pipeline_id in pipeline_service.pipelines
            assert deployment.deployment_id in deployment_manager.deployments
            
        finally:
            await pipeline_service.stop_pipeline_executor()
            await deployment_manager.stop_monitoring()