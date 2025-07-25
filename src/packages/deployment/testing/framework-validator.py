#!/usr/bin/env python3
"""
Framework Validation and Testing Suite
Validates the deployment automation framework components
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TestSeverity(Enum):
    """Test importance level"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestResult:
    """Individual test result"""
    name: str
    description: str
    category: str
    severity: TestSeverity
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    output: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FrameworkValidator:
    """Main framework validation system"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.deployment_dir = self.project_root / "src/packages/deployment"
        self.test_results: Dict[str, TestResult] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'/tmp/framework-validation-{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self._initialize_tests()
    
    def _initialize_tests(self):
        """Initialize all validation tests"""
        
        # File structure tests
        self.test_results.update({
            "deployment_scripts_exist": TestResult(
                "deployment_scripts_exist",
                "Verify all deployment scripts exist",
                "structure",
                TestSeverity.CRITICAL
            ),
            "scripts_executable": TestResult(
                "scripts_executable", 
                "Verify scripts have execute permissions",
                "structure",
                TestSeverity.HIGH
            ),
            "documentation_complete": TestResult(
                "documentation_complete",
                "Verify documentation is complete",
                "structure", 
                TestSeverity.MEDIUM
            )
        })
        
        # Script functionality tests
        self.test_results.update({
            "deployment_script_syntax": TestResult(
                "deployment_script_syntax",
                "Validate deployment script syntax",
                "functionality",
                TestSeverity.CRITICAL
            ),
            "monitoring_script_syntax": TestResult(
                "monitoring_script_syntax", 
                "Validate monitoring script syntax",
                "functionality",
                TestSeverity.CRITICAL
            ),
            "validator_script_syntax": TestResult(
                "validator_script_syntax",
                "Validate validator script syntax", 
                "functionality",
                TestSeverity.CRITICAL
            ),
            "disaster_recovery_syntax": TestResult(
                "disaster_recovery_syntax",
                "Validate disaster recovery script syntax",
                "functionality",
                TestSeverity.CRITICAL
            )
        })
        
        # Configuration tests
        self.test_results.update({
            "makefile_targets": TestResult(
                "makefile_targets",
                "Verify Makefile targets are valid",
                "configuration",
                TestSeverity.HIGH
            ),
            "docker_compose_valid": TestResult(
                "docker_compose_valid",
                "Validate Docker Compose configurations",
                "configuration", 
                TestSeverity.HIGH
            ),
            "kubernetes_manifests": TestResult(
                "kubernetes_manifests",
                "Validate Kubernetes manifests",
                "configuration",
                TestSeverity.HIGH
            )
        })
        
        # Integration tests
        self.test_results.update({
            "deployment_dry_run": TestResult(
                "deployment_dry_run",
                "Test deployment script dry run",
                "integration",
                TestSeverity.HIGH
            ),
            "monitoring_startup": TestResult(
                "monitoring_startup",
                "Test monitoring system startup", 
                "integration",
                TestSeverity.MEDIUM
            ),
            "validation_execution": TestResult(
                "validation_execution",
                "Test validation framework execution",
                "integration",
                TestSeverity.MEDIUM
            )
        })
    
    async def run_all_tests(self) -> Dict[str, TestResult]:
        """Run all validation tests"""
        self.logger.info("Starting framework validation tests")
        
        # Group tests by category for organized execution
        test_categories = {
            "structure": [],
            "functionality": [],
            "configuration": [],
            "integration": []
        }
        
        for test_name, test_result in self.test_results.items():
            test_categories[test_result.category].append(test_name)
        
        # Run tests by category (structure -> functionality -> configuration -> integration)
        for category in ["structure", "functionality", "configuration", "integration"]:
            self.logger.info(f"Running {category} tests...")
            
            tasks = []
            for test_name in test_categories[category]:
                task = self._run_test(test_name)
                tasks.append(task)
            
            # Run tests in parallel within each category
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return self.test_results
    
    async def _run_test(self, test_name: str) -> TestResult:
        """Run individual test"""
        test_result = self.test_results[test_name]
        test_result.status = TestStatus.RUNNING
        test_result.start_time = datetime.now()
        
        self.logger.info(f"Running test: {test_name}")
        
        try:
            # Dispatch to specific test method
            test_method = getattr(self, f"_test_{test_name}", None)
            if test_method:
                await test_method(test_result)
            else:
                test_result.status = TestStatus.SKIPPED
                test_result.error_message = "Test method not implemented"
        
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = str(e)
            self.logger.error(f"Test {test_name} failed: {e}")
        
        test_result.end_time = datetime.now()
        if test_result.start_time:
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
        
        status_symbol = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.SKIPPED: "⏭️",
            TestStatus.RUNNING: "🔄"
        }.get(test_result.status, "❓")
        
        self.logger.info(f"Test {test_name}: {status_symbol} {test_result.status.value}")
        
        return test_result
    
    # Structure tests
    async def _test_deployment_scripts_exist(self, test_result: TestResult):
        """Test that all required deployment scripts exist"""
        required_scripts = [
            "scripts/automated-deployment.sh",
            "scripts/disaster-recovery.sh",
            "scripts/build-images.sh",
            "scripts/deploy-kubernetes.sh",
            "scripts/deploy-docker-compose.sh",
            "monitoring/production-monitoring.py",
            "validation/production-validator.py",
            "Makefile"
        ]
        
        missing_scripts = []
        for script in required_scripts:
            script_path = self.deployment_dir / script
            if not script_path.exists():
                missing_scripts.append(str(script))
        
        if missing_scripts:
            test_result.status = TestStatus.FAILED
            test_result.error_message = f"Missing scripts: {', '.join(missing_scripts)}"
        else:
            test_result.status = TestStatus.PASSED
            test_result.metadata["scripts_found"] = len(required_scripts)
    
    async def _test_scripts_executable(self, test_result: TestResult):
        """Test that scripts have proper execute permissions"""
        executable_scripts = [
            "scripts/automated-deployment.sh",
            "scripts/disaster-recovery.sh", 
            "scripts/build-images.sh",
            "scripts/deploy-kubernetes.sh",
            "scripts/deploy-docker-compose.sh",
            "monitoring/production-monitoring.py",
            "validation/production-validator.py"
        ]
        
        non_executable = []
        for script in executable_scripts:
            script_path = self.deployment_dir / script
            if script_path.exists() and not os.access(script_path, os.X_OK):
                non_executable.append(str(script))
        
        if non_executable:
            test_result.status = TestStatus.FAILED
            test_result.error_message = f"Non-executable scripts: {', '.join(non_executable)}"
        else:
            test_result.status = TestStatus.PASSED
            test_result.metadata["executable_scripts"] = len(executable_scripts)
    
    async def _test_documentation_complete(self, test_result: TestResult):
        """Test that documentation is complete"""
        required_docs = [
            "PRODUCTION_OPERATIONS_GUIDE.md",
            "DEVELOPER_ONBOARDING.md",
            "ADVANCED_PATTERNS_GUIDE.md",
            "FRAMEWORK_COMPLETION_SUMMARY.md"
        ]
        
        missing_docs = []
        for doc in required_docs:
            doc_path = self.project_root / "src/packages" / doc
            if not doc_path.exists():
                missing_docs.append(doc)
        
        if missing_docs:
            test_result.status = TestStatus.FAILED
            test_result.error_message = f"Missing documentation: {', '.join(missing_docs)}"
        else:
            test_result.status = TestStatus.PASSED
            test_result.metadata["docs_found"] = len(required_docs)
    
    # Functionality tests
    async def _test_deployment_script_syntax(self, test_result: TestResult):
        """Test deployment script syntax"""
        script_path = self.deployment_dir / "scripts/automated-deployment.sh"
        
        if not script_path.exists():
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Deployment script not found"
            return
        
        try:
            # Test bash syntax
            result = subprocess.run([
                "bash", "-n", str(script_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                test_result.status = TestStatus.PASSED
                test_result.metadata["syntax_check"] = "valid"
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Syntax error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Syntax check timed out"
    
    async def _test_monitoring_script_syntax(self, test_result: TestResult):
        """Test monitoring script Python syntax"""
        script_path = self.deployment_dir / "monitoring/production-monitoring.py"
        
        if not script_path.exists():
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Monitoring script not found"
            return
        
        try:
            # Test Python syntax
            result = subprocess.run([
                "python3", "-m", "py_compile", str(script_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                test_result.status = TestStatus.PASSED
                test_result.metadata["syntax_check"] = "valid"
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Syntax error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Syntax check timed out"
    
    async def _test_validator_script_syntax(self, test_result: TestResult):
        """Test validator script Python syntax"""
        script_path = self.deployment_dir / "validation/production-validator.py"
        
        if not script_path.exists():
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Validator script not found"
            return
        
        try:
            # Test Python syntax
            result = subprocess.run([
                "python3", "-m", "py_compile", str(script_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                test_result.status = TestStatus.PASSED
                test_result.metadata["syntax_check"] = "valid"
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Syntax error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Syntax check timed out"
    
    async def _test_disaster_recovery_syntax(self, test_result: TestResult):
        """Test disaster recovery script syntax"""
        script_path = self.deployment_dir / "scripts/disaster-recovery.sh"
        
        if not script_path.exists():
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Disaster recovery script not found"
            return
        
        try:
            # Test bash syntax
            result = subprocess.run([
                "bash", "-n", str(script_path)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                test_result.status = TestStatus.PASSED
                test_result.metadata["syntax_check"] = "valid"
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Syntax error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Syntax check timed out"
    
    # Configuration tests
    async def _test_makefile_targets(self, test_result: TestResult):
        """Test Makefile targets are valid"""
        makefile_path = self.deployment_dir / "Makefile"
        
        if not makefile_path.exists():
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Makefile not found"
            return
        
        try:
            # Test make syntax by listing targets
            result = subprocess.run([
                "make", "-f", str(makefile_path), "-n", "help"
            ], capture_output=True, text=True, timeout=10, cwd=str(self.deployment_dir))
            
            if result.returncode == 0:
                test_result.status = TestStatus.PASSED
                test_result.metadata["makefile_valid"] = True
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Makefile error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Makefile check timed out"
    
    async def _test_docker_compose_valid(self, test_result: TestResult):
        """Test Docker Compose configurations"""
        compose_dir = self.deployment_dir / "compose"
        
        if not compose_dir.exists():
            test_result.status = TestStatus.SKIPPED
            test_result.error_message = "Docker Compose directory not found"
            return
        
        compose_files = list(compose_dir.glob("*.yml")) + list(compose_dir.glob("*.yaml"))
        
        if not compose_files:
            test_result.status = TestStatus.SKIPPED
            test_result.error_message = "No Docker Compose files found"
            return
        
        try:
            valid_files = 0
            for compose_file in compose_files:
                result = subprocess.run([
                    "docker-compose", "-f", str(compose_file), "config"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    valid_files += 1
            
            if valid_files == len(compose_files):
                test_result.status = TestStatus.PASSED
                test_result.metadata["valid_compose_files"] = valid_files
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Invalid compose files: {len(compose_files) - valid_files}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Docker Compose validation timed out"
        except FileNotFoundError:
            test_result.status = TestStatus.SKIPPED
            test_result.error_message = "docker-compose command not found"
    
    async def _test_kubernetes_manifests(self, test_result: TestResult):
        """Test Kubernetes manifest validity"""
        k8s_dir = self.deployment_dir / "kubernetes"
        
        if not k8s_dir.exists():
            test_result.status = TestStatus.SKIPPED
            test_result.error_message = "Kubernetes directory not found"
            return
        
        yaml_files = list(k8s_dir.rglob("*.yml")) + list(k8s_dir.rglob("*.yaml"))
        
        if not yaml_files:
            test_result.status = TestStatus.SKIPPED
            test_result.error_message = "No Kubernetes YAML files found"
            return
        
        try:
            valid_files = 0
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, 'r') as f:
                        yaml.safe_load_all(f)
                    valid_files += 1
                except yaml.YAMLError:
                    pass
            
            if valid_files == len(yaml_files):
                test_result.status = TestStatus.PASSED
                test_result.metadata["valid_yaml_files"] = valid_files
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Invalid YAML files: {len(yaml_files) - valid_files}"
                
        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.error_message = f"YAML validation error: {e}"
    
    # Integration tests
    async def _test_deployment_dry_run(self, test_result: TestResult):
        """Test deployment script dry run functionality"""
        script_path = self.deployment_dir / "scripts/automated-deployment.sh"
        
        if not script_path.exists():
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Deployment script not found"
            return
        
        try:
            # Test help functionality
            result = subprocess.run([
                str(script_path), "--help"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and "Usage:" in result.stdout:
                test_result.status = TestStatus.PASSED
                test_result.metadata["help_functionality"] = "working"
                test_result.output = result.stdout[:500]  # First 500 chars
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Help functionality failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Deployment script help timed out"
    
    async def _test_monitoring_startup(self, test_result: TestResult):
        """Test monitoring system startup"""
        script_path = self.deployment_dir / "monitoring/production-monitoring.py"
        
        if not script_path.exists():
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Monitoring script not found"
            return
        
        try:
            # Test help functionality
            result = subprocess.run([
                "python3", str(script_path), "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "Production Monitoring System" in result.stdout:
                test_result.status = TestStatus.PASSED
                test_result.metadata["help_functionality"] = "working"
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Help functionality failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Monitoring script help timed out"
    
    async def _test_validation_execution(self, test_result: TestResult):
        """Test validation framework execution"""
        script_path = self.deployment_dir / "validation/production-validator.py"
        
        if not script_path.exists():
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Validation script not found"
            return
        
        try:
            # Test help functionality
            result = subprocess.run([
                "python3", str(script_path), "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "Production Deployment Validator" in result.stdout:
                test_result.status = TestStatus.PASSED
                test_result.metadata["help_functionality"] = "working"
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Help functionality failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            test_result.status = TestStatus.FAILED
            test_result.error_message = "Validation script help timed out"
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FRAMEWORK VALIDATION TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results.values() if t.status == TestStatus.PASSED)
        failed_tests = sum(1 for t in self.test_results.values() if t.status == TestStatus.FAILED)
        skipped_tests = sum(1 for t in self.test_results.values() if t.status == TestStatus.SKIPPED)
        
        # Critical failures
        critical_failures = [
            t.name for t in self.test_results.values()
            if t.severity == TestSeverity.CRITICAL and t.status == TestStatus.FAILED
        ]
        
        # Overall status
        if critical_failures:
            overall_status = "❌ FAILED (Critical test failures)"
        elif failed_tests > 0:
            overall_status = "⚠️ PASSED WITH WARNINGS (Non-critical failures)"
        else:
            overall_status = "✅ PASSED"
        
        report_lines.append(f"Overall Status: {overall_status}")
        report_lines.append("")
        report_lines.append("Summary:")
        report_lines.append(f"  Total Tests: {total_tests}")
        report_lines.append(f"  Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        report_lines.append(f"  Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        report_lines.append(f"  Skipped: {skipped_tests} ({skipped_tests/total_tests*100:.1f}%)")
        
        if critical_failures:
            report_lines.append("")
            report_lines.append("🚨 CRITICAL FAILURES:")
            for failure in critical_failures:
                report_lines.append(f"  - {failure}")
        
        report_lines.append("")
        report_lines.append("-" * 80)
        
        # Detailed results by category
        categories = ["structure", "functionality", "configuration", "integration"]
        for category in categories:
            category_tests = [t for t in self.test_results.values() if t.category == category]
            if not category_tests:
                continue
                
            report_lines.append(f"CATEGORY: {category.upper()}")
            report_lines.append("-" * 40)
            
            for test in category_tests:
                status_symbol = {
                    TestStatus.PASSED: "✅",
                    TestStatus.FAILED: "❌",
                    TestStatus.SKIPPED: "⏭️",
                    TestStatus.PENDING: "⏸️",
                    TestStatus.RUNNING: "🔄"
                }.get(test.status, "❓")
                
                severity_symbol = {
                    TestSeverity.CRITICAL: "🔴",
                    TestSeverity.HIGH: "🟠", 
                    TestSeverity.MEDIUM: "🟡",
                    TestSeverity.LOW: "🟢"
                }.get(test.severity, "⚪")
                
                duration = f"{test.duration:.2f}s" if test.duration else "N/A"
                
                report_lines.append(f"  {status_symbol} {severity_symbol} {test.name:<30} {duration:>8}")
                
                if test.description:
                    report_lines.append(f"      Description: {test.description}")
                
                if test.error_message:
                    report_lines.append(f"      Error: {test.error_message}")
                
                if test.output:
                    report_lines.append(f"      Output: {test.output[:100]}...")
                
                if test.metadata:
                    for key, value in test.metadata.items():
                        report_lines.append(f"      {key}: {value}")
                
                report_lines.append("")
            
            report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("Legend:")
        report_lines.append("  ✅ PASSED  ❌ FAILED  ⏭️ SKIPPED  ⏸️ PENDING  🔄 RUNNING")
        report_lines.append("  🔴 CRITICAL  🟠 HIGH  🟡 MEDIUM  🟢 LOW")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Framework Validation Testing")
    parser.add_argument("--project-root", help="Project root directory")
    parser.add_argument("--report", help="Output file for test report")
    parser.add_argument("--category", help="Run tests for specific category only")
    args = parser.parse_args()
    
    validator = FrameworkValidator(args.project_root)
    
    # Run validation tests
    results = await validator.run_all_tests()
    
    # Generate report
    report = validator.generate_test_report()
    
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"Test report saved to: {args.report}")
    else:
        print(report)
    
    # Return appropriate exit code
    critical_failures = any(
        test.severity == TestSeverity.CRITICAL and test.status == TestStatus.FAILED
        for test in results.values()
    )
    
    return 1 if critical_failures else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)