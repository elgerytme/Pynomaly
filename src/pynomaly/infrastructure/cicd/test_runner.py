"""Test runner service for executing different types of tests in CI/CD pipelines."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from pynomaly.domain.models.cicd import PipelineStatus, TestResult, TestSuite, TestType


class TestRunner:
    """Service for running various types of tests in CI/CD pipelines."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Test execution tracking
        self.running_tests: set[UUID] = set()

        # Test configuration
        self.test_configurations: dict[TestType, dict[str, Any]] = {
            TestType.UNIT: {
                "timeout": 300,  # 5 minutes
                "parallel": True,
                "coverage_required": True,
                "default_command": "pytest tests/unit/ -v --cov --cov-report=json",
            },
            TestType.INTEGRATION: {
                "timeout": 900,  # 15 minutes
                "parallel": False,
                "setup_required": True,
                "default_command": "pytest tests/integration/ -v --maxfail=1",
            },
            TestType.FUNCTIONAL: {
                "timeout": 1800,  # 30 minutes
                "parallel": False,
                "environment_setup": True,
                "default_command": "pytest tests/functional/ -v",
            },
            TestType.PERFORMANCE: {
                "timeout": 3600,  # 60 minutes
                "parallel": False,
                "resource_intensive": True,
                "default_command": "pytest tests/performance/ -v --benchmark-only",
            },
            TestType.SECURITY: {
                "timeout": 1200,  # 20 minutes
                "parallel": True,
                "security_context": True,
                "default_command": "bandit -r src/ && safety check",
            },
            TestType.SMOKE: {
                "timeout": 300,  # 5 minutes
                "parallel": True,
                "quick_execution": True,
                "default_command": "pytest tests/smoke/ -v --maxfail=5",
            },
            TestType.REGRESSION: {
                "timeout": 7200,  # 2 hours
                "parallel": False,
                "comprehensive": True,
                "default_command": "pytest tests/regression/ -v",
            },
            TestType.END_TO_END: {
                "timeout": 3600,  # 60 minutes
                "parallel": False,
                "full_environment": True,
                "default_command": "pytest tests/e2e/ -v --dist=no",
            },
        }

        self.logger.info("Test runner initialized")

    async def execute_test_suite(
        self,
        test_suite: TestSuite,
        workspace_path: Path,
        environment: Optional[dict[str, str]] = None,
    ) -> bool:
        """Execute a test suite and update results."""

        if test_suite.suite_id in self.running_tests:
            self.logger.warning(f"Test suite {test_suite.name} is already running")
            return False

        self.running_tests.add(test_suite.suite_id)

        try:
            # Start test suite execution
            test_suite.status = PipelineStatus.RUNNING
            test_suite.start_time = await self._get_current_time()

            # Get test configuration
            config = self.test_configurations.get(test_suite.test_type, {})

            # Execute tests based on type
            success = await self._execute_tests(
                test_suite, workspace_path, environment, config
            )

            # Complete test suite
            test_suite.end_time = await self._get_current_time()
            if test_suite.start_time:
                test_suite.duration_seconds = (
                    test_suite.end_time - test_suite.start_time
                ).total_seconds()

            test_suite.calculate_metrics()

            self.logger.info(
                f"Test suite {test_suite.name} completed: {test_suite.status.value}"
            )
            return success

        except Exception as e:
            test_suite.status = PipelineStatus.FAILED
            test_suite.end_time = await self._get_current_time()
            self.logger.error(f"Test suite {test_suite.name} failed: {e}")
            return False

        finally:
            self.running_tests.discard(test_suite.suite_id)

    async def discover_tests(
        self,
        test_type: TestType,
        workspace_path: Path,
        test_patterns: Optional[list[str]] = None,
    ) -> list[str]:
        """Discover test files based on type and patterns."""

        discovered_tests = []

        # Default test discovery patterns
        type_patterns = {
            TestType.UNIT: ["tests/unit/**/*.py", "tests/**/test_*.py"],
            TestType.INTEGRATION: ["tests/integration/**/*.py"],
            TestType.FUNCTIONAL: ["tests/functional/**/*.py"],
            TestType.PERFORMANCE: ["tests/performance/**/*.py", "tests/**/bench_*.py"],
            TestType.SECURITY: ["tests/security/**/*.py"],
            TestType.SMOKE: ["tests/smoke/**/*.py"],
            TestType.REGRESSION: ["tests/regression/**/*.py"],
            TestType.END_TO_END: ["tests/e2e/**/*.py", "tests/end_to_end/**/*.py"],
        }

        patterns = test_patterns or type_patterns.get(test_type, ["tests/**/*.py"])

        for pattern in patterns:
            try:
                # Use pathlib for pattern matching
                test_files = list(workspace_path.glob(pattern))
                discovered_tests.extend(
                    [str(f.relative_to(workspace_path)) for f in test_files]
                )
            except Exception as e:
                self.logger.warning(
                    f"Error discovering tests with pattern {pattern}: {e}"
                )

        # Remove duplicates and sort
        discovered_tests = sorted(list(set(discovered_tests)))

        self.logger.debug(
            f"Discovered {len(discovered_tests)} test files for {test_type.value}"
        )
        return discovered_tests

    async def validate_test_environment(
        self,
        test_type: TestType,
        workspace_path: Path,
        environment: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Validate test environment requirements."""

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "requirements_met": [],
        }

        try:
            # Check workspace exists
            if not workspace_path.exists():
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Workspace path does not exist: {workspace_path}"
                )
                return validation_result

            # Check test-specific requirements
            if test_type == TestType.UNIT:
                await self._validate_unit_test_environment(
                    workspace_path, validation_result
                )
            elif test_type == TestType.INTEGRATION:
                await self._validate_integration_test_environment(
                    workspace_path, validation_result
                )
            elif test_type == TestType.PERFORMANCE:
                await self._validate_performance_test_environment(
                    workspace_path, validation_result
                )
            elif test_type == TestType.SECURITY:
                await self._validate_security_test_environment(
                    workspace_path, validation_result
                )
            elif test_type == TestType.END_TO_END:
                await self._validate_e2e_test_environment(
                    workspace_path, validation_result
                )

            # Check Python environment
            await self._check_python_dependencies(workspace_path, validation_result)

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Environment validation failed: {e}")

        return validation_result

    async def _execute_tests(
        self,
        test_suite: TestSuite,
        workspace_path: Path,
        environment: Optional[dict[str, str]],
        config: dict[str, Any],
    ) -> bool:
        """Execute tests for a test suite."""

        # Prepare environment
        env = dict(environment or {})
        env.update(
            {
                "PYTHONPATH": str(workspace_path),
                "TEST_TYPE": test_suite.test_type.value,
                "TEST_SUITE": test_suite.name,
            }
        )

        # Determine test command
        if test_suite.test_command:
            command = test_suite.test_command
        else:
            command = config.get("default_command", "pytest -v")

        # Add test patterns if specified
        if test_suite.test_patterns:
            command += " " + " ".join(test_suite.test_patterns)

        # Execute based on test type
        if test_suite.test_type == TestType.UNIT:
            return await self._execute_unit_tests(
                test_suite, workspace_path, command, env
            )
        elif test_suite.test_type == TestType.INTEGRATION:
            return await self._execute_integration_tests(
                test_suite, workspace_path, command, env
            )
        elif test_suite.test_type == TestType.PERFORMANCE:
            return await self._execute_performance_tests(
                test_suite, workspace_path, command, env
            )
        elif test_suite.test_type == TestType.SECURITY:
            return await self._execute_security_tests(
                test_suite, workspace_path, command, env
            )
        else:
            # Generic test execution
            return await self._execute_generic_tests(
                test_suite, workspace_path, command, env
            )

    async def _execute_unit_tests(
        self,
        test_suite: TestSuite,
        workspace_path: Path,
        command: str,
        environment: dict[str, str],
    ) -> bool:
        """Execute unit tests with coverage analysis."""

        try:
            # Add coverage options
            if "--cov" not in command:
                command += " --cov=src --cov-report=json --cov-report=term"

            # Execute tests
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=workspace_path,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Parse test results
            success = await self._parse_pytest_output(
                test_suite, stdout.decode(), stderr.decode(), process.returncode
            )

            # Parse coverage results
            await self._parse_coverage_results(test_suite, workspace_path)

            return success

        except Exception as e:
            await self._add_test_error(test_suite, "Unit test execution failed", str(e))
            return False

    async def _execute_integration_tests(
        self,
        test_suite: TestSuite,
        workspace_path: Path,
        command: str,
        environment: dict[str, str],
    ) -> bool:
        """Execute integration tests with setup/teardown."""

        try:
            # Setup integration test environment
            await self._setup_integration_environment(workspace_path, environment)

            # Execute tests
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=workspace_path,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Parse results
            success = await self._parse_pytest_output(
                test_suite, stdout.decode(), stderr.decode(), process.returncode
            )

            return success

        except Exception as e:
            await self._add_test_error(
                test_suite, "Integration test execution failed", str(e)
            )
            return False

        finally:
            # Cleanup integration environment
            await self._cleanup_integration_environment(workspace_path)

    async def _execute_performance_tests(
        self,
        test_suite: TestSuite,
        workspace_path: Path,
        command: str,
        environment: dict[str, str],
    ) -> bool:
        """Execute performance tests with benchmarking."""

        try:
            # Add benchmark options
            if "pytest-benchmark" in command and "--benchmark-json" not in command:
                command += " --benchmark-json=benchmark_results.json"

            # Execute tests
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=workspace_path,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Parse results
            success = await self._parse_pytest_output(
                test_suite, stdout.decode(), stderr.decode(), process.returncode
            )

            # Parse benchmark results
            await self._parse_benchmark_results(test_suite, workspace_path)

            return success

        except Exception as e:
            await self._add_test_error(
                test_suite, "Performance test execution failed", str(e)
            )
            return False

    async def _execute_security_tests(
        self,
        test_suite: TestSuite,
        workspace_path: Path,
        command: str,
        environment: dict[str, str],
    ) -> bool:
        """Execute security tests and scans."""

        try:
            # Security tools often have different command patterns
            security_commands = command.split(" && ")

            overall_success = True

            for cmd in security_commands:
                cmd = cmd.strip()
                if not cmd:
                    continue

                process = await asyncio.create_subprocess_shell(
                    cmd,
                    cwd=workspace_path,
                    env=environment,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await process.communicate()

                # Create test result for each security tool
                test_result = TestResult(
                    test_id=uuid4(),
                    test_name=f"security_{cmd.split()[0]}",
                    test_type=TestType.SECURITY,
                    status=PipelineStatus.SUCCESS
                    if process.returncode == 0
                    else PipelineStatus.FAILED,
                    output=stdout.decode(),
                    error_message=stderr.decode() if process.returncode != 0 else None,
                )

                test_suite.add_test_result(test_result)

                if process.returncode != 0:
                    overall_success = False

            return overall_success

        except Exception as e:
            await self._add_test_error(
                test_suite, "Security test execution failed", str(e)
            )
            return False

    async def _execute_generic_tests(
        self,
        test_suite: TestSuite,
        workspace_path: Path,
        command: str,
        environment: dict[str, str],
    ) -> bool:
        """Execute generic tests."""

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=workspace_path,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            return await self._parse_pytest_output(
                test_suite, stdout.decode(), stderr.decode(), process.returncode
            )

        except Exception as e:
            await self._add_test_error(test_suite, "Test execution failed", str(e))
            return False

    async def _parse_pytest_output(
        self,
        test_suite: TestSuite,
        stdout: str,
        stderr: str,
        return_code: int,
    ) -> bool:
        """Parse pytest output and create test results."""

        try:
            # Simple parsing - in production would use pytest JSON report
            lines = stdout.split("\n")

            # Look for test results summary
            for line in lines:
                line = line.strip()

                # Example: "====== 5 passed, 2 failed in 10.5s ======"
                if "passed" in line and "failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed,":
                            passed_count = int(parts[i - 1])
                            test_suite.passed_tests = passed_count
                        elif part == "failed":
                            failed_count = int(parts[i - 1])
                            test_suite.failed_tests = failed_count

                # Individual test results (simplified parsing)
                if "::" in line and (" PASSED" in line or " FAILED" in line):
                    test_name = line.split("::")[1].split()[0]
                    status = (
                        PipelineStatus.SUCCESS
                        if "PASSED" in line
                        else PipelineStatus.FAILED
                    )

                    test_result = TestResult(
                        test_id=uuid4(),
                        test_name=test_name,
                        test_type=test_suite.test_type,
                        status=status,
                        output=line,
                    )

                    test_suite.add_test_result(test_result)

            # If no specific results found, create summary result
            if not test_suite.tests:
                test_result = TestResult(
                    test_id=uuid4(),
                    test_name=f"{test_suite.test_type.value}_summary",
                    test_type=test_suite.test_type,
                    status=PipelineStatus.SUCCESS
                    if return_code == 0
                    else PipelineStatus.FAILED,
                    output=stdout,
                    error_message=stderr if return_code != 0 else None,
                )
                test_suite.add_test_result(test_result)

            return return_code == 0

        except Exception as e:
            self.logger.error(f"Error parsing pytest output: {e}")
            return False

    async def _parse_coverage_results(
        self, test_suite: TestSuite, workspace_path: Path
    ) -> None:
        """Parse coverage results from coverage.json."""

        try:
            coverage_file = workspace_path / "coverage.json"
            if coverage_file.exists():
                import json

                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                # Extract overall coverage percentage
                totals = coverage_data.get("totals", {})
                percent_covered = totals.get("percent_covered")

                if percent_covered is not None:
                    test_suite.overall_coverage = float(percent_covered)

        except Exception as e:
            self.logger.warning(f"Could not parse coverage results: {e}")

    async def _parse_benchmark_results(
        self, test_suite: TestSuite, workspace_path: Path
    ) -> None:
        """Parse benchmark results from benchmark JSON file."""

        try:
            benchmark_file = workspace_path / "benchmark_results.json"
            if benchmark_file.exists():
                import json

                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)

                # Process benchmark data and add to test results
                benchmarks = benchmark_data.get("benchmarks", [])
                for benchmark in benchmarks:
                    test_result = TestResult(
                        test_id=uuid4(),
                        test_name=benchmark.get("name", "benchmark"),
                        test_type=TestType.PERFORMANCE,
                        status=PipelineStatus.SUCCESS,
                        output=json.dumps(benchmark.get("stats", {}), indent=2),
                    )
                    test_suite.add_test_result(test_result)

        except Exception as e:
            self.logger.warning(f"Could not parse benchmark results: {e}")

    async def _validate_unit_test_environment(
        self,
        workspace_path: Path,
        validation_result: dict[str, Any],
    ) -> None:
        """Validate unit test environment."""

        # Check for test directory
        test_dir = workspace_path / "tests" / "unit"
        if not test_dir.exists():
            validation_result["warnings"].append(
                "Unit test directory not found: tests/unit/"
            )
        else:
            validation_result["requirements_met"].append("Unit test directory exists")

        # Check for pytest configuration
        pytest_configs = ["pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"]
        config_found = any(
            (workspace_path / config).exists() for config in pytest_configs
        )

        if config_found:
            validation_result["requirements_met"].append("Pytest configuration found")
        else:
            validation_result["warnings"].append("No pytest configuration found")

    async def _validate_integration_test_environment(
        self,
        workspace_path: Path,
        validation_result: dict[str, Any],
    ) -> None:
        """Validate integration test environment."""

        # Check for integration test directory
        test_dir = workspace_path / "tests" / "integration"
        if not test_dir.exists():
            validation_result["warnings"].append(
                "Integration test directory not found: tests/integration/"
            )
        else:
            validation_result["requirements_met"].append(
                "Integration test directory exists"
            )

        # Check for docker-compose or test services
        if (workspace_path / "docker-compose.test.yml").exists():
            validation_result["requirements_met"].append(
                "Docker test environment configuration found"
            )
        elif (workspace_path / "tests" / "conftest.py").exists():
            validation_result["requirements_met"].append(
                "Pytest configuration for fixtures found"
            )

    async def _validate_performance_test_environment(
        self,
        workspace_path: Path,
        validation_result: dict[str, Any],
    ) -> None:
        """Validate performance test environment."""

        # Check for performance test directory
        test_dir = workspace_path / "tests" / "performance"
        if not test_dir.exists():
            validation_result["warnings"].append(
                "Performance test directory not found: tests/performance/"
            )
        else:
            validation_result["requirements_met"].append(
                "Performance test directory exists"
            )

        # Check for benchmark tools
        requirements_file = workspace_path / "requirements.txt"
        if requirements_file.exists():
            content = requirements_file.read_text()
            if "pytest-benchmark" in content:
                validation_result["requirements_met"].append(
                    "pytest-benchmark dependency found"
                )

    async def _validate_security_test_environment(
        self,
        workspace_path: Path,
        validation_result: dict[str, Any],
    ) -> None:
        """Validate security test environment."""

        # Check for security tools in requirements
        requirements_file = workspace_path / "requirements.txt"
        if requirements_file.exists():
            content = requirements_file.read_text()

            security_tools = ["bandit", "safety", "semgrep"]
            found_tools = [tool for tool in security_tools if tool in content]

            if found_tools:
                validation_result["requirements_met"].append(
                    f"Security tools found: {', '.join(found_tools)}"
                )
            else:
                validation_result["warnings"].append(
                    "No security scanning tools found in requirements"
                )

    async def _validate_e2e_test_environment(
        self,
        workspace_path: Path,
        validation_result: dict[str, Any],
    ) -> None:
        """Validate end-to-end test environment."""

        # Check for E2E test directory
        e2e_dirs = ["tests/e2e", "tests/end_to_end"]
        e2e_found = any((workspace_path / e2e_dir).exists() for e2e_dir in e2e_dirs)

        if e2e_found:
            validation_result["requirements_met"].append(
                "End-to-end test directory exists"
            )
        else:
            validation_result["warnings"].append("No end-to-end test directory found")

    async def _check_python_dependencies(
        self,
        workspace_path: Path,
        validation_result: dict[str, Any],
    ) -> None:
        """Check Python dependencies."""

        # Check for requirements or pyproject.toml
        if (workspace_path / "pyproject.toml").exists():
            validation_result["requirements_met"].append("pyproject.toml found")
        elif (workspace_path / "requirements.txt").exists():
            validation_result["requirements_met"].append("requirements.txt found")
        else:
            validation_result["errors"].append(
                "No dependency file found (requirements.txt or pyproject.toml)"
            )
            validation_result["valid"] = False

    async def _setup_integration_environment(
        self,
        workspace_path: Path,
        environment: dict[str, str],
    ) -> None:
        """Setup integration test environment."""

        # Check for docker-compose test file
        docker_compose_test = workspace_path / "docker-compose.test.yml"
        if docker_compose_test.exists():
            # Start test services
            process = await asyncio.create_subprocess_exec(
                "docker-compose",
                "-f",
                str(docker_compose_test),
                "up",
                "-d",
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

    async def _cleanup_integration_environment(self, workspace_path: Path) -> None:
        """Cleanup integration test environment."""

        docker_compose_test = workspace_path / "docker-compose.test.yml"
        if docker_compose_test.exists():
            # Stop test services
            process = await asyncio.create_subprocess_exec(
                "docker-compose",
                "-f",
                str(docker_compose_test),
                "down",
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

    async def _add_test_error(
        self,
        test_suite: TestSuite,
        error_type: str,
        error_message: str,
    ) -> None:
        """Add test error to suite."""

        test_result = TestResult(
            test_id=uuid4(),
            test_name=f"error_{error_type.lower().replace(' ', '_')}",
            test_type=test_suite.test_type,
            status=PipelineStatus.FAILED,
            error_message=error_message,
        )

        test_suite.add_test_result(test_result)
        test_suite.status = PipelineStatus.FAILED

    async def _get_current_time(self):
        """Get current time (async method for consistency)."""
        from datetime import datetime

        return datetime.utcnow()
