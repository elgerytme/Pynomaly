"""Integration testing framework for end-to-end testing scenarios."""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import pytest


class TestResult(Enum):
    """Test execution results."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class IntegrationTestStep:
    """Individual step in an integration test."""
    
    name: str
    description: str
    action: Callable
    expected_result: Any = None
    timeout_seconds: int = 30
    retry_count: int = 0
    retry_delay: float = 1.0
    cleanup_action: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    
    # Execution results
    result: Optional[TestResult] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    actual_result: Any = None
    executed_at: Optional[datetime] = None


@dataclass
class IntegrationTestSuite:
    """Collection of integration test steps."""
    
    name: str
    description: str
    steps: List[IntegrationTestStep] = field(default_factory=list)
    setup_action: Optional[Callable] = None
    teardown_action: Optional[Callable] = None
    
    # Execution results
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    error_steps: int = 0
    total_execution_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class IntegrationTestRunner:
    """Runner for executing integration test suites."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the integration test runner.
        
        Args:
            logger: Logger instance for test execution
        """
        self.logger = logger or logging.getLogger(__name__)
        self.test_results: Dict[str, IntegrationTestSuite] = {}
    
    async def run_suite(
        self,
        suite: IntegrationTestSuite,
        fail_fast: bool = False,
        parallel: bool = False
    ) -> IntegrationTestSuite:
        """Run an integration test suite.
        
        Args:
            suite: Test suite to execute
            fail_fast: Stop execution on first failure
            parallel: Run independent steps in parallel
            
        Returns:
            Updated test suite with execution results
        """
        self.logger.info(f"Starting integration test suite: {suite.name}")
        suite.started_at = datetime.utcnow()
        
        try:
            # Run setup if provided
            if suite.setup_action:
                self.logger.info("Running suite setup...")
                await self._execute_action(suite.setup_action, "Setup")
            
            # Execute test steps
            if parallel:
                await self._run_steps_parallel(suite, fail_fast)
            else:
                await self._run_steps_sequential(suite, fail_fast)
            
        except Exception as e:
            self.logger.error(f"Suite execution failed: {e}")
            suite.error_steps = len([s for s in suite.steps if s.result is None])
        
        finally:
            # Run teardown if provided
            if suite.teardown_action:
                self.logger.info("Running suite teardown...")
                try:
                    await self._execute_action(suite.teardown_action, "Teardown")
                except Exception as e:
                    self.logger.error(f"Teardown failed: {e}")
            
            # Calculate final results
            suite.completed_at = datetime.utcnow()
            suite.total_steps = len(suite.steps)
            suite.passed_steps = len([s for s in suite.steps if s.result == TestResult.PASSED])
            suite.failed_steps = len([s for s in suite.steps if s.result == TestResult.FAILED])
            suite.skipped_steps = len([s for s in suite.steps if s.result == TestResult.SKIPPED])
            suite.error_steps = len([s for s in suite.steps if s.result == TestResult.ERROR])
            suite.total_execution_time = sum(step.execution_time for step in suite.steps)
            
            # Store results
            self.test_results[suite.name] = suite
            
            self.logger.info(
                f"Suite completed: {suite.passed_steps} passed, "
                f"{suite.failed_steps} failed, {suite.skipped_steps} skipped, "
                f"{suite.error_steps} errors"
            )
        
        return suite
    
    async def _run_steps_sequential(
        self,
        suite: IntegrationTestSuite,
        fail_fast: bool
    ):
        """Run test steps sequentially."""
        for step in suite.steps:
            # Check dependencies
            if not self._check_dependencies(step, suite.steps):
                step.result = TestResult.SKIPPED
                step.error_message = "Dependencies not met"
                self.logger.warning(f"Skipping step {step.name}: dependencies not met")
                continue
            
            # Execute step
            await self._execute_step(step)
            
            # Handle fail-fast
            if fail_fast and step.result in [TestResult.FAILED, TestResult.ERROR]:
                self.logger.error(f"Failing fast due to step failure: {step.name}")
                # Mark remaining steps as skipped
                remaining_steps = suite.steps[suite.steps.index(step) + 1:]
                for remaining_step in remaining_steps:
                    remaining_step.result = TestResult.SKIPPED
                    remaining_step.error_message = "Skipped due to fail-fast"
                break
    
    async def _run_steps_parallel(
        self,
        suite: IntegrationTestSuite,
        fail_fast: bool
    ):
        """Run independent test steps in parallel."""
        # Group steps by dependencies
        independent_steps = []
        dependent_steps = []
        
        for step in suite.steps:
            if not step.dependencies:
                independent_steps.append(step)
            else:
                dependent_steps.append(step)
        
        # Run independent steps in parallel
        if independent_steps:
            tasks = [self._execute_step(step) for step in independent_steps]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run dependent steps sequentially (could be optimized further)
        for step in dependent_steps:
            if not self._check_dependencies(step, suite.steps):
                step.result = TestResult.SKIPPED
                step.error_message = "Dependencies not met"
                continue
            
            await self._execute_step(step)
            
            if fail_fast and step.result in [TestResult.FAILED, TestResult.ERROR]:
                break
    
    async def _execute_step(self, step: IntegrationTestStep):
        """Execute a single test step."""
        self.logger.info(f"Executing step: {step.name}")
        step.executed_at = datetime.utcnow()
        start_time = time.time()
        
        try:
            # Execute with retries
            for attempt in range(step.retry_count + 1):
                try:
                    step.actual_result = await self._execute_action(
                        step.action,
                        step.name,
                        step.timeout_seconds
                    )
                    
                    # Check expected result if provided
                    if step.expected_result is not None:
                        if step.actual_result == step.expected_result:
                            step.result = TestResult.PASSED
                        else:
                            step.result = TestResult.FAILED
                            step.error_message = (
                                f"Expected {step.expected_result}, "
                                f"got {step.actual_result}"
                            )
                    else:
                        step.result = TestResult.PASSED
                    
                    break  # Success, no need to retry
                    
                except Exception as e:
                    if attempt < step.retry_count:
                        self.logger.warning(
                            f"Step {step.name} failed (attempt {attempt + 1}), "
                            f"retrying in {step.retry_delay}s: {e}"
                        )
                        await asyncio.sleep(step.retry_delay)
                    else:
                        raise  # Final attempt failed
        
        except asyncio.TimeoutError:
            step.result = TestResult.ERROR
            step.error_message = f"Step timed out after {step.timeout_seconds}s"
            self.logger.error(f"Step {step.name} timed out")
        
        except Exception as e:
            step.result = TestResult.ERROR
            step.error_message = str(e)
            self.logger.error(f"Step {step.name} failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        finally:
            step.execution_time = time.time() - start_time
            
            # Run cleanup if provided
            if step.cleanup_action:
                try:
                    await self._execute_action(step.cleanup_action, f"{step.name}_cleanup")
                except Exception as e:
                    self.logger.warning(f"Cleanup failed for step {step.name}: {e}")
    
    async def _execute_action(
        self,
        action: Callable,
        action_name: str,
        timeout_seconds: int = 30
    ) -> Any:
        """Execute an action with timeout."""
        if asyncio.iscoroutinefunction(action):
            return await asyncio.wait_for(action(), timeout=timeout_seconds)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, action),
                timeout=timeout_seconds
            )
    
    def _check_dependencies(
        self,
        step: IntegrationTestStep,
        all_steps: List[IntegrationTestStep]
    ) -> bool:
        """Check if step dependencies are satisfied."""
        if not step.dependencies:
            return True
        
        step_by_name = {s.name: s for s in all_steps}
        
        for dependency in step.dependencies:
            if dependency not in step_by_name:
                return False  # Dependency doesn't exist
            
            dep_step = step_by_name[dependency]
            if dep_step.result != TestResult.PASSED:
                return False  # Dependency didn't pass
        
        return True
    
    def generate_report(self, output_file: str = "integration_test_report.html") -> str:
        """Generate HTML report of integration test results."""
        if not self.test_results:
            raise ValueError("No test results available")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Integration Test Report - Pynomaly</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .suite { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
                .suite-header { background: #f8f9fa; padding: 15px; font-weight: bold; }
                .suite-summary { padding: 10px; background: #e9ecef; }
                .step { padding: 10px; border-bottom: 1px solid #eee; }
                .step:last-child { border-bottom: none; }
                .passed { color: #28a745; }
                .failed { color: #dc3545; }
                .skipped { color: #6c757d; }
                .error { color: #fd7e14; }
                .metrics { display: flex; gap: 20px; margin: 20px 0; }
                .metric { background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; flex: 1; }
                pre { background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Integration Test Report</h1>
                <p>Generated on {timestamp}</p>
            </div>
        """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Overall metrics
        total_suites = len(self.test_results)
        total_steps = sum(suite.total_steps for suite in self.test_results.values())
        total_passed = sum(suite.passed_steps for suite in self.test_results.values())
        total_failed = sum(suite.failed_steps for suite in self.test_results.values())
        total_skipped = sum(suite.skipped_steps for suite in self.test_results.values())
        total_errors = sum(suite.error_steps for suite in self.test_results.values())
        
        html_content += f"""
            <div class="metrics">
                <div class="metric">
                    <h3>Test Suites</h3>
                    <div style="font-size: 24px; font-weight: bold;">{total_suites}</div>
                </div>
                <div class="metric">
                    <h3>Total Steps</h3>
                    <div style="font-size: 24px; font-weight: bold;">{total_steps}</div>
                </div>
                <div class="metric">
                    <h3>Passed</h3>
                    <div style="font-size: 24px; font-weight: bold; color: #28a745;">{total_passed}</div>
                </div>
                <div class="metric">
                    <h3>Failed</h3>
                    <div style="font-size: 24px; font-weight: bold; color: #dc3545;">{total_failed}</div>
                </div>
            </div>
        """
        
        # Suite details
        for suite_name, suite in self.test_results.items():
            success_rate = (suite.passed_steps / suite.total_steps * 100) if suite.total_steps > 0 else 0
            
            html_content += f"""
            <div class="suite">
                <div class="suite-header">{suite.name}</div>
                <div class="suite-summary">
                    <p>{suite.description}</p>
                    <p><strong>Success Rate:</strong> {success_rate:.1f}%</p>
                    <p><strong>Execution Time:</strong> {suite.total_execution_time:.2f}s</p>
                    <p><strong>Results:</strong> 
                        <span class="passed">{suite.passed_steps} passed</span>, 
                        <span class="failed">{suite.failed_steps} failed</span>, 
                        <span class="skipped">{suite.skipped_steps} skipped</span>, 
                        <span class="error">{suite.error_steps} errors</span>
                    </p>
                </div>
            """
            
            # Step details
            for step in suite.steps:
                status_class = step.result.value if step.result else "unknown"
                
                html_content += f"""
                <div class="step">
                    <div><strong class="{status_class}">{step.name}</strong> - {step.description}</div>
                    <div>Status: <span class="{status_class}">{step.result.value.upper() if step.result else "NOT_RUN"}</span></div>
                    <div>Execution Time: {step.execution_time:.3f}s</div>
                """
                
                if step.error_message:
                    html_content += f'<div>Error: <pre>{step.error_message}</pre></div>'
                
                if step.actual_result is not None:
                    html_content += f'<div>Result: <pre>{step.actual_result}</pre></div>'
                
                html_content += "</div>"
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return str(output_path)


class IntegrationTestBuilder:
    """Builder for creating integration test suites."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize builder.
        
        Args:
            name: Name of the test suite
            description: Description of the test suite
        """
        self.suite = IntegrationTestSuite(name=name, description=description)
    
    def with_setup(self, action: Callable) -> 'IntegrationTestBuilder':
        """Add setup action to the suite."""
        self.suite.setup_action = action
        return self
    
    def with_teardown(self, action: Callable) -> 'IntegrationTestBuilder':
        """Add teardown action to the suite."""
        self.suite.teardown_action = action
        return self
    
    def add_step(
        self,
        name: str,
        description: str,
        action: Callable,
        expected_result: Any = None,
        timeout_seconds: int = 30,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        cleanup_action: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None
    ) -> 'IntegrationTestBuilder':
        """Add a step to the test suite."""
        step = IntegrationTestStep(
            name=name,
            description=description,
            action=action,
            expected_result=expected_result,
            timeout_seconds=timeout_seconds,
            retry_count=retry_count,
            retry_delay=retry_delay,
            cleanup_action=cleanup_action,
            dependencies=dependencies or []
        )
        self.suite.steps.append(step)
        return self
    
    def build(self) -> IntegrationTestSuite:
        """Build the test suite."""
        return self.suite


# Helper context managers
@contextmanager
def mock_environment(**mocks):
    """Context manager for mocking environment variables or services."""
    patches = []
    try:
        for name, value in mocks.items():
            if isinstance(value, MagicMock):
                # Mock object
                patcher = patch(name, value)
            else:
                # Mock value
                patcher = patch(name, return_value=value)
            patches.append(patcher)
            patcher.start()
        
        yield
        
    finally:
        for patcher in patches:
            patcher.stop()


@asynccontextmanager
async def temporary_resources(**resources):
    """Context manager for managing temporary resources."""
    created_resources = {}
    try:
        # Create resources
        for name, factory in resources.items():
            if asyncio.iscoroutinefunction(factory):
                created_resources[name] = await factory()
            else:
                created_resources[name] = factory()
        
        yield created_resources
        
    finally:
        # Cleanup resources
        for name, resource in created_resources.items():
            try:
                if hasattr(resource, 'cleanup'):
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        await resource.cleanup()
                    else:
                        resource.cleanup()
                elif hasattr(resource, 'close'):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
            except Exception as e:
                logging.warning(f"Failed to cleanup resource {name}: {e}")


# Pytest integration
def pytest_integration_test(suite_factory: Callable[[], IntegrationTestSuite]):
    """Decorator to create pytest integration tests from test suites."""
    def decorator(func):
        @pytest.mark.integration
        @pytest.mark.asyncio
        async def wrapper(*args, **kwargs):
            runner = IntegrationTestRunner()
            suite = suite_factory()
            
            # Run the suite
            result_suite = await runner.run_suite(suite)
            
            # Assert overall success
            if result_suite.failed_steps > 0 or result_suite.error_steps > 0:
                # Generate report for debugging
                report_path = runner.generate_report(f"failed_integration_test_{suite.name}.html")
                pytest.fail(
                    f"Integration test suite '{suite.name}' failed: "
                    f"{result_suite.failed_steps} failures, {result_suite.error_steps} errors. "
                    f"See report: {report_path}"
                )
            
            return result_suite
        
        return wrapper
    return decorator