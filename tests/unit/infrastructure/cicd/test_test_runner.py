"""Unit tests for test runner service."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
from uuid import uuid4

from pynomaly.domain.models.cicd import (
    PipelineStatus,
    TestResult,
    TestSuite,
    TestType,
)
from pynomaly.infrastructure.cicd.test_runner import TestRunner


@pytest.fixture
def test_runner():
    """Create test runner for testing."""
    return TestRunner()


@pytest.fixture
def sample_test_suite():
    """Create sample test suite."""
    return TestSuite(
        suite_id=uuid4(),
        name="Unit Tests",
        test_type=TestType.UNIT,
        test_files=["tests/unit/test_example.py"],
        test_command="pytest tests/unit/ -v",
    )


@pytest.fixture
def workspace_path(tmp_path):
    """Create temporary workspace path."""
    return tmp_path


class TestTestRunner:
    """Test cases for TestRunner."""

    def test_initialization(self, test_runner):
        """Test test runner initialization."""
        assert isinstance(test_runner.running_tests, set)
        assert len(test_runner.running_tests) == 0
        assert TestType.UNIT in test_runner.test_configurations
        assert TestType.INTEGRATION in test_runner.test_configurations
        assert TestType.PERFORMANCE in test_runner.test_configurations
        assert TestType.SECURITY in test_runner.test_configurations

    def test_test_configurations(self, test_runner):
        """Test test configurations are properly set."""
        unit_config = test_runner.test_configurations[TestType.UNIT]
        assert unit_config["timeout"] == 300
        assert unit_config["parallel"] is True
        assert unit_config["coverage_required"] is True
        assert "pytest" in unit_config["default_command"]

        integration_config = test_runner.test_configurations[TestType.INTEGRATION]
        assert integration_config["timeout"] == 900
        assert integration_config["parallel"] is False
        assert integration_config["setup_required"] is True

        performance_config = test_runner.test_configurations[TestType.PERFORMANCE]
        assert performance_config["timeout"] == 3600
        assert performance_config["resource_intensive"] is True

        security_config = test_runner.test_configurations[TestType.SECURITY]
        assert security_config["security_context"] is True
        assert "bandit" in security_config["default_command"]

    @patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell")
    async def test_execute_test_suite_success(self, mock_subprocess, test_runner, sample_test_suite, workspace_path):
        """Test successful test suite execution."""
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b"====== 5 passed in 2.5s ======",
            b""
        )
        mock_subprocess.return_value = mock_process

        # Execute test suite
        success = await test_runner.execute_test_suite(
            sample_test_suite,
            workspace_path,
            {"TEST_ENV": "unit"}
        )

        assert success
        assert sample_test_suite.status == PipelineStatus.SUCCESS
        assert sample_test_suite.suite_id not in test_runner.running_tests
        assert sample_test_suite.passed_tests == 5
        assert sample_test_suite.failed_tests == 0

    @patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell")
    async def test_execute_test_suite_failure(self, mock_subprocess, test_runner, sample_test_suite, workspace_path):
        """Test failed test suite execution."""
        # Mock subprocess failure
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (
            b"====== 3 passed, 2 failed in 2.5s ======",
            b"errors occurred"
        )
        mock_subprocess.return_value = mock_process

        # Execute test suite
        success = await test_runner.execute_test_suite(
            sample_test_suite,
            workspace_path,
        )

        assert not success
        assert sample_test_suite.status == PipelineStatus.FAILED
        assert sample_test_suite.suite_id not in test_runner.running_tests
        assert sample_test_suite.passed_tests == 3
        assert sample_test_suite.failed_tests == 2

    async def test_execute_already_running_test_suite(self, test_runner, sample_test_suite, workspace_path):
        """Test executing already running test suite returns False."""
        # Mark as running
        test_runner.running_tests.add(sample_test_suite.suite_id)

        # Try to execute
        success = await test_runner.execute_test_suite(
            sample_test_suite,
            workspace_path,
        )

        assert not success

    async def test_discover_tests_unit(self, test_runner, workspace_path):
        """Test discovering unit tests."""
        # Create test files
        unit_dir = workspace_path / "tests" / "unit"
        unit_dir.mkdir(parents=True)
        (unit_dir / "test_example.py").touch()
        (unit_dir / "test_another.py").touch()

        # Discover tests
        discovered = await test_runner.discover_tests(
            TestType.UNIT,
            workspace_path,
        )

        assert len(discovered) >= 2
        assert any("test_example.py" in test for test in discovered)
        assert any("test_another.py" in test for test in discovered)

    async def test_discover_tests_integration(self, test_runner, workspace_path):
        """Test discovering integration tests."""
        # Create test files
        integration_dir = workspace_path / "tests" / "integration"
        integration_dir.mkdir(parents=True)
        (integration_dir / "test_integration.py").touch()

        # Discover tests
        discovered = await test_runner.discover_tests(
            TestType.INTEGRATION,
            workspace_path,
        )

        assert len(discovered) >= 1
        assert any("test_integration.py" in test for test in discovered)

    async def test_discover_tests_custom_patterns(self, test_runner, workspace_path):
        """Test discovering tests with custom patterns."""
        # Create test files
        custom_dir = workspace_path / "custom_tests"
        custom_dir.mkdir(parents=True)
        (custom_dir / "custom_test.py").touch()

        # Discover with custom pattern
        discovered = await test_runner.discover_tests(
            TestType.UNIT,
            workspace_path,
            ["custom_tests/*.py"]
        )

        assert len(discovered) >= 1
        assert any("custom_test.py" in test for test in discovered)

    async def test_validate_test_environment_valid(self, test_runner, workspace_path):
        """Test validating valid test environment."""
        # Create required directories and files
        (workspace_path / "tests" / "unit").mkdir(parents=True)
        (workspace_path / "pytest.ini").touch()
        (workspace_path / "requirements.txt").write_text("pytest>=6.0")

        # Validate environment
        result = await test_runner.validate_test_environment(
            TestType.UNIT,
            workspace_path,
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert "Unit test directory exists" in result["requirements_met"]
        assert "Pytest configuration found" in result["requirements_met"]

    async def test_validate_test_environment_missing_directory(self, test_runner, workspace_path):
        """Test validating environment with missing test directory."""
        # Create minimal setup (missing test directory)
        (workspace_path / "requirements.txt").touch()

        # Validate environment
        result = await test_runner.validate_test_environment(
            TestType.UNIT,
            workspace_path,
        )

        assert result["valid"] is True  # Still valid, just warnings
        assert "Unit test directory not found" in result["warnings"]

    async def test_validate_test_environment_missing_dependencies(self, test_runner, workspace_path):
        """Test validating environment with missing dependencies."""
        # Create directory but no dependency files
        (workspace_path / "tests" / "unit").mkdir(parents=True)

        # Validate environment
        result = await test_runner.validate_test_environment(
            TestType.UNIT,
            workspace_path,
        )

        assert result["valid"] is False
        assert "No dependency file found" in result["errors"]

    async def test_validate_test_environment_nonexistent_workspace(self, test_runner):
        """Test validating non-existent workspace."""
        nonexistent_path = Path("/nonexistent/path")

        result = await test_runner.validate_test_environment(
            TestType.UNIT,
            nonexistent_path,
        )

        assert result["valid"] is False
        assert "Workspace path does not exist" in result["errors"]

    @patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell")
    async def test_execute_unit_tests(self, mock_subprocess, test_runner, workspace_path):
        """Test executing unit tests with coverage."""
        # Create test suite
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Unit Tests",
            test_type=TestType.UNIT,
        )

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b"test_example.py::test_function PASSED\n====== 1 passed in 1.0s ======",
            b""
        )
        mock_subprocess.return_value = mock_process

        # Execute unit tests
        success = await test_runner._execute_unit_tests(
            test_suite,
            workspace_path,
            "pytest -v",
            {"PYTHONPATH": str(workspace_path)}
        )

        assert success
        assert len(test_suite.tests) >= 1
        # Verify coverage options were added
        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        assert "--cov" in args[0]

    @patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell")
    async def test_execute_integration_tests(self, mock_subprocess, test_runner, workspace_path):
        """Test executing integration tests with setup/teardown."""
        # Create test suite
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Integration Tests",
            test_type=TestType.INTEGRATION,
        )

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b"test_integration.py::test_api PASSED\n====== 1 passed in 5.0s ======",
            b""
        )
        mock_subprocess.return_value = mock_process

        # Execute integration tests
        success = await test_runner._execute_integration_tests(
            test_suite,
            workspace_path,
            "pytest tests/integration/ -v",
            {"PYTHONPATH": str(workspace_path)}
        )

        assert success
        assert len(test_suite.tests) >= 1

    @patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell")
    async def test_execute_performance_tests(self, mock_subprocess, test_runner, workspace_path):
        """Test executing performance tests with benchmarking."""
        # Create test suite
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Performance Tests",
            test_type=TestType.PERFORMANCE,
        )

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b"test_performance.py::test_benchmark PASSED\n====== 1 passed in 10.0s ======",
            b""
        )
        mock_subprocess.return_value = mock_process

        # Execute performance tests
        success = await test_runner._execute_performance_tests(
            test_suite,
            workspace_path,
            "pytest tests/performance/ --benchmark-only",
            {"PYTHONPATH": str(workspace_path)}
        )

        assert success
        # Verify benchmark options were added
        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        if "pytest-benchmark" in args[0]:
            assert "--benchmark-json" in args[0]

    @patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell")
    async def test_execute_security_tests(self, mock_subprocess, test_runner, workspace_path):
        """Test executing security tests with multiple tools."""
        # Create test suite
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Security Tests",
            test_type=TestType.SECURITY,
        )

        # Mock subprocess for multiple security tools
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"No issues found", b"")
        mock_subprocess.return_value = mock_process

        # Execute security tests with multiple commands
        success = await test_runner._execute_security_tests(
            test_suite,
            workspace_path,
            "bandit -r src/ && safety check",
            {"PYTHONPATH": str(workspace_path)}
        )

        assert success
        assert len(test_suite.tests) == 2  # One for each security tool
        assert any("bandit" in test.test_name for test in test_suite.tests)
        assert any("safety" in test.test_name for test in test_suite.tests)

    async def test_parse_pytest_output_success(self, test_runner):
        """Test parsing successful pytest output."""
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Test Suite",
            test_type=TestType.UNIT,
        )

        stdout = """
        test_example.py::test_function PASSED
        test_example.py::test_another PASSED
        ====== 2 passed in 1.5s ======
        """

        success = await test_runner._parse_pytest_output(
            test_suite, stdout, "", 0
        )

        assert success
        assert test_suite.passed_tests == 2
        assert test_suite.failed_tests == 0
        assert len(test_suite.tests) >= 2

    async def test_parse_pytest_output_with_failures(self, test_runner):
        """Test parsing pytest output with failures."""
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Test Suite",
            test_type=TestType.UNIT,
        )

        stdout = """
        test_example.py::test_function PASSED
        test_example.py::test_failing FAILED
        ====== 1 passed, 1 failed in 2.0s ======
        """

        success = await test_runner._parse_pytest_output(
            test_suite, stdout, "", 1
        )

        assert not success
        assert test_suite.passed_tests == 1
        assert test_suite.failed_tests == 1
        assert len(test_suite.tests) >= 2

    async def test_parse_pytest_output_no_specific_results(self, test_runner):
        """Test parsing pytest output with no specific test results."""
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Test Suite",
            test_type=TestType.UNIT,
        )

        stdout = "Tests completed successfully"

        success = await test_runner._parse_pytest_output(
            test_suite, stdout, "", 0
        )

        assert success
        assert len(test_suite.tests) == 1  # Summary result created
        assert test_suite.tests[0].test_name == "unit_summary"

    @patch("builtins.open", new_callable=mock_open, read_data='{"totals": {"percent_covered": 85.5}}')
    async def test_parse_coverage_results(self, mock_file, test_runner, workspace_path):
        """Test parsing coverage results."""
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Test Suite",
            test_type=TestType.UNIT,
        )

        # Create coverage file
        coverage_file = workspace_path / "coverage.json"
        coverage_file.touch()

        await test_runner._parse_coverage_results(test_suite, workspace_path)

        assert test_suite.overall_coverage == 85.5

    @patch("builtins.open", new_callable=mock_open, read_data='{"benchmarks": [{"name": "test_benchmark", "stats": {"mean": 0.001}}]}')
    async def test_parse_benchmark_results(self, mock_file, test_runner, workspace_path):
        """Test parsing benchmark results."""
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Test Suite",
            test_type=TestType.PERFORMANCE,
        )

        # Create benchmark file
        benchmark_file = workspace_path / "benchmark_results.json"
        benchmark_file.touch()

        await test_runner._parse_benchmark_results(test_suite, workspace_path)

        assert len(test_suite.tests) == 1
        assert test_suite.tests[0].test_name == "test_benchmark"
        assert test_suite.tests[0].test_type == TestType.PERFORMANCE

    async def test_validate_unit_test_environment(self, test_runner, workspace_path):
        """Test validating unit test environment specifically."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        # Create unit test directory and pytest config
        (workspace_path / "tests" / "unit").mkdir(parents=True)
        (workspace_path / "pytest.ini").touch()

        await test_runner._validate_unit_test_environment(workspace_path, validation_result)

        assert "Unit test directory exists" in validation_result["requirements_met"]
        assert "Pytest configuration found" in validation_result["requirements_met"]

    async def test_validate_integration_test_environment(self, test_runner, workspace_path):
        """Test validating integration test environment specifically."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        # Create integration test directory and docker compose
        (workspace_path / "tests" / "integration").mkdir(parents=True)
        (workspace_path / "docker-compose.test.yml").touch()

        await test_runner._validate_integration_test_environment(workspace_path, validation_result)

        assert "Integration test directory exists" in validation_result["requirements_met"]
        assert "Docker test environment configuration found" in validation_result["requirements_met"]

    async def test_validate_performance_test_environment(self, test_runner, workspace_path):
        """Test validating performance test environment specifically."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        # Create performance test directory and requirements with benchmark tool
        (workspace_path / "tests" / "performance").mkdir(parents=True)
        (workspace_path / "requirements.txt").write_text("pytest-benchmark>=3.0")

        await test_runner._validate_performance_test_environment(workspace_path, validation_result)

        assert "Performance test directory exists" in validation_result["requirements_met"]
        assert "pytest-benchmark dependency found" in validation_result["requirements_met"]

    async def test_validate_security_test_environment(self, test_runner, workspace_path):
        """Test validating security test environment specifically."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        # Create requirements with security tools
        (workspace_path / "requirements.txt").write_text("bandit>=1.7\\nsafety>=2.0")

        await test_runner._validate_security_test_environment(workspace_path, validation_result)

        assert any("bandit" in req for req in validation_result["requirements_met"])

    async def test_validate_e2e_test_environment(self, test_runner, workspace_path):
        """Test validating end-to-end test environment specifically."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        # Create E2E test directory
        (workspace_path / "tests" / "e2e").mkdir(parents=True)

        await test_runner._validate_e2e_test_environment(workspace_path, validation_result)

        assert "End-to-end test directory exists" in validation_result["requirements_met"]

    async def test_check_python_dependencies_pyproject(self, test_runner, workspace_path):
        """Test checking Python dependencies with pyproject.toml."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        # Create pyproject.toml
        (workspace_path / "pyproject.toml").touch()

        await test_runner._check_python_dependencies(workspace_path, validation_result)

        assert validation_result["valid"] is True
        assert "pyproject.toml found" in validation_result["requirements_met"]

    async def test_check_python_dependencies_requirements(self, test_runner, workspace_path):
        """Test checking Python dependencies with requirements.txt."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        # Create requirements.txt
        (workspace_path / "requirements.txt").touch()

        await test_runner._check_python_dependencies(workspace_path, validation_result)

        assert validation_result["valid"] is True
        assert "requirements.txt found" in validation_result["requirements_met"]

    async def test_check_python_dependencies_missing(self, test_runner, workspace_path):
        """Test checking Python dependencies with no dependency files."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        # No dependency files created

        await test_runner._check_python_dependencies(workspace_path, validation_result)

        assert validation_result["valid"] is False
        assert "No dependency file found" in validation_result["errors"]

    @patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_exec")
    async def test_setup_integration_environment(self, mock_subprocess, test_runner, workspace_path):
        """Test setting up integration test environment with docker-compose."""
        # Create docker-compose test file
        (workspace_path / "docker-compose.test.yml").touch()

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess.return_value = mock_process

        await test_runner._setup_integration_environment(workspace_path, {})

        # Verify docker-compose up was called
        mock_subprocess.assert_called_once_with(
            "docker-compose", "-f", str(workspace_path / "docker-compose.test.yml"), "up", "-d",
            cwd=workspace_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_exec")
    async def test_cleanup_integration_environment(self, mock_subprocess, test_runner, workspace_path):
        """Test cleaning up integration test environment."""
        # Create docker-compose test file
        (workspace_path / "docker-compose.test.yml").touch()

        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess.return_value = mock_process

        await test_runner._cleanup_integration_environment(workspace_path)

        # Verify docker-compose down was called
        mock_subprocess.assert_called_once_with(
            "docker-compose", "-f", str(workspace_path / "docker-compose.test.yml"), "down",
            cwd=workspace_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def test_add_test_error(self, test_runner):
        """Test adding test error to suite."""
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Test Suite",
            test_type=TestType.UNIT,
        )

        await test_runner._add_test_error(
            test_suite,
            "Execution Error",
            "Test execution failed due to missing dependency"
        )

        assert test_suite.status == PipelineStatus.FAILED
        assert len(test_suite.tests) == 1
        assert test_suite.tests[0].test_name == "error_execution_error"
        assert test_suite.tests[0].status == PipelineStatus.FAILED
        assert test_suite.tests[0].error_message == "Test execution failed due to missing dependency"


@pytest.mark.asyncio
class TestTestRunnerIntegration:
    """Integration tests for TestRunner."""

    async def test_full_test_execution_workflow(self, test_runner, workspace_path):
        """Test complete test execution workflow."""
        # Create test environment
        unit_dir = workspace_path / "tests" / "unit"
        unit_dir.mkdir(parents=True)
        (unit_dir / "test_example.py").write_text("""
def test_example():
    assert True

def test_another():
    assert 1 + 1 == 2
""")
        (workspace_path / "pytest.ini").write_text("[tool:pytest]\\ntestpaths = tests")
        (workspace_path / "requirements.txt").write_text("pytest>=6.0")

        # Create test suite
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Full Integration Test",
            test_type=TestType.UNIT,
        )

        # Validate environment
        validation = await test_runner.validate_test_environment(
            TestType.UNIT,
            workspace_path,
        )
        assert validation["valid"] is True

        # Discover tests
        discovered = await test_runner.discover_tests(
            TestType.UNIT,
            workspace_path,
        )
        assert len(discovered) >= 1
        assert any("test_example.py" in test for test in discovered)

        # Note: Actual execution would require real pytest, so we mock it
        with patch("pynomaly.infrastructure.cicd.test_runner.asyncio.create_subprocess_shell") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate.return_value = (
                b"test_example.py::test_example PASSED\\ntest_example.py::test_another PASSED\\n====== 2 passed in 0.1s ======",
                b""
            )
            mock_subprocess.return_value = mock_process

            # Execute test suite
            success = await test_runner.execute_test_suite(
                test_suite,
                workspace_path,
                {"PYTHONPATH": str(workspace_path)}
            )

            assert success
            assert test_suite.status == PipelineStatus.SUCCESS
            assert test_suite.passed_tests == 2
            assert test_suite.failed_tests == 0

    async def test_multiple_test_types_workflow(self, test_runner, workspace_path):
        """Test workflow with multiple test types."""
        # Create test structure
        for test_type_dir in ["unit", "integration", "performance"]:
            test_dir = workspace_path / "tests" / test_type_dir
            test_dir.mkdir(parents=True)
            (test_dir / f"test_{test_type_dir}.py").touch()

        (workspace_path / "requirements.txt").write_text("pytest>=6.0\\npytest-benchmark>=3.0")

        # Test discovery for different types
        unit_tests = await test_runner.discover_tests(TestType.UNIT, workspace_path)
        integration_tests = await test_runner.discover_tests(TestType.INTEGRATION, workspace_path)
        performance_tests = await test_runner.discover_tests(TestType.PERFORMANCE, workspace_path)

        assert len(unit_tests) >= 1
        assert len(integration_tests) >= 1
        assert len(performance_tests) >= 1

        # Validate environments
        for test_type in [TestType.UNIT, TestType.INTEGRATION, TestType.PERFORMANCE]:
            validation = await test_runner.validate_test_environment(test_type, workspace_path)
            assert validation["valid"] is True

    async def test_error_handling_workflow(self, test_runner, workspace_path):
        """Test error handling in test execution workflow."""
        # Create test suite without proper environment
        test_suite = TestSuite(
            suite_id=uuid4(),
            name="Error Test",
            test_type=TestType.UNIT,
            test_command="nonexistent_command",
        )

        # Execute with missing command (should handle gracefully)
        success = await test_runner.execute_test_suite(
            test_suite,
            workspace_path,
        )

        assert not success
        assert test_suite.status == PipelineStatus.FAILED
        assert len(test_suite.tests) >= 1
        assert any(test.status == PipelineStatus.FAILED for test in test_suite.tests)
