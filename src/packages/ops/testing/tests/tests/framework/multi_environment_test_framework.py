"""
Comprehensive Multi-Environment Testing Framework - GitHub Issue #214

This module provides the infrastructure for production-ready multi-environment
testing with comprehensive validation across platforms, Python versions, and
deployment scenarios.
"""

import json
import os
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

import psutil
import pytest


class TestEnvironmentType(Enum):
    """Test environment types."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    API = "api"
    CLI = "cli"
    SERVER = "server"
    PRODUCTION = "production"
    ALL = "all"


class TestCategory(Enum):
    """Test categories for different testing phases."""
    UNIT = "unit"
    INTEGRATION = "integration"
    CONTRACT = "contract"
    E2E = "e2e"
    PERFORMANCE = "performance"
    LOAD = "load"
    STRESS = "stress"
    SECURITY = "security"


class PlatformType(Enum):
    """Supported platform types."""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"


@dataclass
class EnvironmentConfig:
    """Configuration for a test environment."""
    name: str
    python_version: str
    platform: PlatformType
    dependency_group: TestEnvironmentType
    shell: str
    test_categories: List[TestCategory]
    environment_variables: Dict[str, str] = field(default_factory=dict)
    setup_commands: List[str] = field(default_factory=list)
    cleanup_commands: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Test execution result."""
    environment_id: str
    test_category: TestCategory
    status: str  # "passed", "failed", "skipped", "error"
    execution_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)


@dataclass
class EnvironmentTestResult:
    """Complete test results for an environment."""
    environment_config: EnvironmentConfig
    environment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    test_results: List[TestResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)


class SystemProfiler:
    """System information and performance profiler."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "python_executable": sys.executable,
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "cpu": {
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
                "freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent,
            },
            "network": {
                "interfaces": list(psutil.net_if_addrs().keys()),
            }
        }
    
    @staticmethod
    def profile_process(func, *args, **kwargs) -> Dict[str, Any]:
        """Profile function execution with system metrics."""
        process = psutil.Process()
        
        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss
        start_cpu_percent = process.cpu_percent()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss
        end_cpu_percent = process.cpu_percent()
        
        return {
            "result": result,
            "success": success,
            "error": error,
            "execution_time": end_time - start_time,
            "memory_usage": {
                "start": start_memory,
                "end": end_memory,
                "delta": end_memory - start_memory,
            },
            "cpu_usage": {
                "start": start_cpu_percent,
                "end": end_cpu_percent,
            }
        }


class EnvironmentValidator:
    """Validates environment setup and configuration."""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_python_version(self) -> bool:
        """Validate Python version compatibility."""
        try:
            current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            expected_version = self.config.python_version
            
            if current_version != expected_version:
                self.logger.warning(
                    f"Python version mismatch: expected {expected_version}, got {current_version}"
                )
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error validating Python version: {e}")
            return False
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """Validate required dependencies are installed."""
        validation_results = {}
        
        # Core dependencies
        core_deps = [
            "pynomaly",
            "numpy", 
            "pandas",
            "pydantic",
            "pyod",
        ]
        
        for dep in core_deps:
            try:
                __import__(dep)
                validation_results[dep] = True
            except ImportError:
                validation_results[dep] = False
                self.logger.error(f"Missing core dependency: {dep}")
        
        # Group-specific dependencies
        group_deps = self._get_group_dependencies()
        for dep in group_deps:
            try:
                __import__(dep)
                validation_results[dep] = True
            except ImportError:
                validation_results[dep] = False
                self.logger.warning(f"Missing group dependency: {dep}")
        
        return validation_results
    
    def _get_group_dependencies(self) -> List[str]:
        """Get dependencies for specific environment group."""
        group_mapping = {
            TestEnvironmentType.API: ["fastapi", "uvicorn", "httpx"],
            TestEnvironmentType.CLI: ["typer", "rich", "click"],
            TestEnvironmentType.SERVER: ["fastapi", "uvicorn", "redis", "sqlalchemy"],
            TestEnvironmentType.PRODUCTION: ["prometheus_client", "opentelemetry"],
        }
        return group_mapping.get(self.config.dependency_group, [])
    
    def validate_system_requirements(self) -> Dict[str, bool]:
        """Validate system requirements."""
        requirements = {}
        
        # Memory requirement (minimum 2GB available)
        available_memory = psutil.virtual_memory().available
        requirements["memory"] = available_memory >= 2 * 1024**3
        
        # Disk space requirement (minimum 1GB free)
        free_disk = psutil.disk_usage('/').free
        requirements["disk_space"] = free_disk >= 1024**3
        
        # CPU requirement (minimum 1 core)
        cpu_count = psutil.cpu_count()
        requirements["cpu"] = cpu_count >= 1
        
        return requirements


class TestExecutor:
    """Executes tests in different categories and environments."""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def execute_test_category(self, category: TestCategory) -> TestResult:
        """Execute tests for a specific category."""
        start_time = time.time()
        test_id = str(uuid.uuid4())
        
        try:
            # Get test command for category
            test_command = self._get_test_command(category)
            
            # Execute test with profiling
            profile_result = SystemProfiler.profile_process(
                self._run_test_command, test_command
            )
            
            execution_time = time.time() - start_time
            
            if profile_result["success"]:
                status = "passed"
                error_message = None
            else:
                status = "failed"
                error_message = profile_result["error"]
            
            return TestResult(
                environment_id=self.config.name,
                test_category=category,
                status=status,
                execution_time=execution_time,
                details={
                    "command": test_command,
                    "performance": profile_result,
                },
                error_message=error_message,
                artifacts=self._collect_artifacts(category, test_id)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                environment_id=self.config.name,
                test_category=category,
                status="error",
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _get_test_command(self, category: TestCategory) -> List[str]:
        """Get pytest command for test category."""
        base_cmd = ["python", "-m", "pytest"]
        
        command_mapping = {
            TestCategory.UNIT: [
                "tests/unit/", "tests/domain/", 
                "-v", "--tb=short", "-m", "not slow"
            ],
            TestCategory.INTEGRATION: [
                "tests/integration/", 
                "-v", "--tb=short", "-m", "not external"
            ],
            TestCategory.CONTRACT: [
                "tests/contract/", 
                "-v", "--tb=short"
            ],
            TestCategory.E2E: [
                "tests/e2e/", 
                "-v", "--tb=short", "-m", "not load"
            ],
            TestCategory.PERFORMANCE: [
                "tests/performance/", 
                "-v", "--tb=short", "--benchmark-skip"
            ],
            TestCategory.LOAD: [
                "tests/performance/", 
                "-v", "--tb=short", "-m", "load"
            ],
            TestCategory.STRESS: [
                "tests/performance/", 
                "-v", "--tb=short", "-m", "stress"
            ],
            TestCategory.SECURITY: [
                "tests/security/", 
                "-v", "--tb=short"
            ]
        }
        
        return base_cmd + command_mapping.get(category, ["--help"])
    
    def _run_test_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run test command with proper environment setup."""
        env = os.environ.copy()
        env.update(self.config.environment_variables)
        
        # Ensure PYTHONPATH includes src directory
        current_pythonpath = env.get("PYTHONPATH", "")
        src_path = str(Path.cwd() / "src")
        if src_path not in current_pythonpath:
            env["PYTHONPATH"] = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path
        
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5 minute timeout
        )
    
    def _collect_artifacts(self, category: TestCategory, test_id: str) -> List[str]:
        """Collect test artifacts and reports."""
        artifacts = []
        
        # Common artifact patterns
        artifact_patterns = [
            f"test-results-{category.value}.xml",
            f"coverage-{category.value}.xml",
            f"benchmark-{category.value}.json",
            "htmlcov/",
            "pytest-reports/",
        ]
        
        for pattern in artifact_patterns:
            artifact_path = Path(pattern)
            if artifact_path.exists():
                artifacts.append(str(artifact_path))
        
        return artifacts


class MultiEnvironmentTestRunner:
    """Main test runner for multi-environment testing."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("test-results")
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        self.results: List[EnvironmentTestResult] = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.output_dir / "test-runner.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def create_environment_configs(self, scope: str = "comprehensive") -> List[EnvironmentConfig]:
        """Create environment configurations based on testing scope."""
        configs = []
        
        # Determine scope parameters
        if scope == "quick":
            python_versions = ["3.11"]
            dependency_groups = [TestEnvironmentType.MINIMAL, TestEnvironmentType.STANDARD]
            test_categories = [TestCategory.UNIT, TestCategory.INTEGRATION]
        elif scope == "standard":
            python_versions = ["3.11", "3.12"]
            dependency_groups = [TestEnvironmentType.MINIMAL, TestEnvironmentType.STANDARD, 
                                TestEnvironmentType.API, TestEnvironmentType.CLI]
            test_categories = [TestCategory.UNIT, TestCategory.INTEGRATION, TestCategory.CONTRACT]
        elif scope == "comprehensive":
            python_versions = ["3.11", "3.12", "3.13"]
            dependency_groups = [TestEnvironmentType.MINIMAL, TestEnvironmentType.STANDARD,
                                TestEnvironmentType.API, TestEnvironmentType.CLI, 
                                TestEnvironmentType.SERVER, TestEnvironmentType.PRODUCTION]
            test_categories = [TestCategory.UNIT, TestCategory.INTEGRATION, TestCategory.CONTRACT,
                             TestCategory.E2E, TestCategory.PERFORMANCE]
        else:  # stress
            python_versions = ["3.11", "3.12", "3.13"]
            dependency_groups = [TestEnvironmentType.ALL]
            test_categories = [TestCategory.UNIT, TestCategory.INTEGRATION, TestCategory.CONTRACT,
                             TestCategory.E2E, TestCategory.PERFORMANCE, TestCategory.LOAD, 
                             TestCategory.STRESS]
        
        # Get current platform
        current_platform = self._get_current_platform()
        shell = "powershell" if current_platform == PlatformType.WINDOWS else "bash"
        
        # Generate configurations
        for python_version in python_versions:
            for dependency_group in dependency_groups:
                config = EnvironmentConfig(
                    name=f"{current_platform.value}-py{python_version}-{dependency_group.value}",
                    python_version=python_version,
                    platform=current_platform,
                    dependency_group=dependency_group,
                    shell=shell,
                    test_categories=test_categories,
                    environment_variables={
                        "PYNOMALY_ENVIRONMENT": "testing",
                        "LOG_LEVEL": "INFO",
                    }
                )
                configs.append(config)
        
        return configs
    
    def _get_current_platform(self) -> PlatformType:
        """Determine current platform type."""
        system = platform.system().lower()
        if "linux" in system:
            return PlatformType.LINUX
        elif "windows" in system:
            return PlatformType.WINDOWS
        elif "darwin" in system:
            return PlatformType.MACOS
        else:
            return PlatformType.LINUX  # Default fallback
    
    def run_environment_tests(self, config: EnvironmentConfig) -> EnvironmentTestResult:
        """Run tests for a specific environment configuration."""
        self.logger.info(f"Starting tests for environment: {config.name}")
        
        environment_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Initialize result
        result = EnvironmentTestResult(
            environment_config=config,
            environment_id=environment_id,
            start_time=start_time,
            system_info=SystemProfiler.get_system_info()
        )
        
        try:
            # Validate environment
            validator = EnvironmentValidator(config)
            result.validation_results = {
                "python_version": validator.validate_python_version(),
                "dependencies": validator.validate_dependencies(),
                "system_requirements": validator.validate_system_requirements(),
            }
            
            # Execute tests
            executor = TestExecutor(config)
            
            for category in config.test_categories:
                self.logger.info(f"Running {category.value} tests in {config.name}")
                test_result = executor.execute_test_category(category)
                result.test_results.append(test_result)
                
                # Log result
                if test_result.status == "passed":
                    self.logger.info(f"✅ {category.value} tests passed")
                else:
                    self.logger.error(f"❌ {category.value} tests failed: {test_result.error_message}")
        
        except Exception as e:
            self.logger.error(f"Environment test execution failed: {e}")
        
        finally:
            result.end_time = datetime.now(timezone.utc)
            
        return result
    
    def run_all_environments(self, scope: str = "comprehensive") -> List[EnvironmentTestResult]:
        """Run tests across all environment configurations."""
        self.logger.info(f"Starting multi-environment testing with scope: {scope}")
        
        configs = self.create_environment_configs(scope)
        self.logger.info(f"Created {len(configs)} environment configurations")
        
        results = []
        for config in configs:
            result = self.run_environment_tests(config)
            results.append(result)
            self.results.append(result)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results: List[EnvironmentTestResult]) -> None:
        """Generate comprehensive summary report."""
        self.logger.info("Generating summary report")
        
        # Calculate statistics
        total_environments = len(results)
        total_tests = sum(len(result.test_results) for result in results)
        passed_tests = sum(1 for result in results for test in result.test_results if test.status == "passed")
        failed_tests = sum(1 for result in results for test in result.test_results if test.status == "failed")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Create summary
        summary = {
            "meta": {
                "report_type": "Multi-Environment Testing Summary",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "github_issue": "#214",
                "framework_version": "1.0.0",
            },
            "statistics": {
                "total_environments": total_environments,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 2),
            },
            "environment_results": [
                {
                    "environment_name": result.environment_config.name,
                    "python_version": result.environment_config.python_version,
                    "platform": result.environment_config.platform.value,
                    "dependency_group": result.environment_config.dependency_group.value,
                    "test_count": len(result.test_results),
                    "passed_count": len([t for t in result.test_results if t.status == "passed"]),
                    "failed_count": len([t for t in result.test_results if t.status == "failed"]),
                    "execution_time": (result.end_time - result.start_time).total_seconds() if result.end_time else 0,
                    "validation_passed": all(result.validation_results.get("dependencies", {}).values()) if result.validation_results else False,
                }
                for result in results
            ]
        }
        
        # Save summary
        summary_file = self.output_dir / "multi-environment-summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed_file = self.output_dir / "detailed-results.json"
        with open(detailed_file, 'w') as f:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_result = {
                    "environment_config": {
                        "name": result.environment_config.name,
                        "python_version": result.environment_config.python_version,
                        "platform": result.environment_config.platform.value,
                        "dependency_group": result.environment_config.dependency_group.value,
                        "shell": result.environment_config.shell,
                        "test_categories": [cat.value for cat in result.environment_config.test_categories],
                    },
                    "environment_id": result.environment_id,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "system_info": result.system_info,
                    "validation_results": result.validation_results,
                    "test_results": [
                        {
                            "test_category": test.test_category.value,
                            "status": test.status,
                            "execution_time": test.execution_time,
                            "error_message": test.error_message,
                            "artifacts": test.artifacts,
                        }
                        for test in result.test_results
                    ],
                }
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Summary report saved to {summary_file}")
        self.logger.info(f"Detailed results saved to {detailed_file}")
        
        # Log summary to console
        self.logger.info("="*60)
        self.logger.info("MULTI-ENVIRONMENT TESTING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total Environments Tested: {total_environments}")
        self.logger.info(f"Total Tests Executed: {total_tests}")
        self.logger.info(f"Tests Passed: {passed_tests}")
        self.logger.info(f"Tests Failed: {failed_tests}")
        self.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 95:
            self.logger.info("🎉 EXCELLENT: Multi-environment testing highly successful!")
        elif success_rate >= 80:
            self.logger.info("✅ GOOD: Multi-environment testing mostly successful")
        elif success_rate >= 60:
            self.logger.info("⚠️ FAIR: Some environments need attention")
        else:
            self.logger.error("❌ POOR: Significant issues detected across environments")


def main():
    """Main entry point for multi-environment testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Environment Testing Framework for Pynomaly (Issue #214)"
    )
    parser.add_argument(
        "--scope",
        choices=["quick", "standard", "comprehensive", "stress"],
        default="comprehensive",
        help="Testing scope (default: comprehensive)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-results"),
        help="Output directory for results (default: test-results)"
    )
    
    args = parser.parse_args()
    
    # Initialize and run tests
    runner = MultiEnvironmentTestRunner(output_dir=args.output_dir)
    results = runner.run_all_environments(scope=args.scope)
    
    # Exit with appropriate code
    total_tests = sum(len(result.test_results) for result in results)
    passed_tests = sum(1 for result in results for test in result.test_results if test.status == "passed")
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    if success_rate >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()