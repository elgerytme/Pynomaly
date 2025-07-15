#!/usr/bin/env python3
"""
CI/CD Quality Gates Implementation

Automated quality gates system that validates code quality, security, and performance
before allowing deployment to production.
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import yaml


@dataclass
class QualityGate:
    """Quality gate definition."""

    name: str
    type: str  # test_coverage, security_scan, performance, code_quality
    threshold: int | float
    operator: str  # >=, <=, ==, !=, >, <
    description: str
    blocking: bool = True  # Whether failure blocks deployment
    timeout: int = 300  # Timeout in seconds


@dataclass
class QualityResult:
    """Quality gate execution result."""

    gate: QualityGate
    passed: bool
    actual_value: int | float | str
    message: str
    execution_time: float
    details: dict = None


class QualityGatesOrchestrator:
    """Quality gates execution orchestrator."""

    def __init__(self, config_path: str = "config/quality_gates.yml"):
        self.logger = self._setup_logging()
        self.config_path = config_path
        self.gates = self._load_quality_gates()
        self.results: list[QualityResult] = []

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def _load_quality_gates(self) -> list[QualityGate]:
        """Load quality gates from configuration."""
        if not os.path.exists(self.config_path):
            return self._get_default_quality_gates()

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            gates = []
            for gate_config in config.get("quality_gates", []):
                gate = QualityGate(**gate_config)
                gates.append(gate)

            return gates
        except Exception as e:
            self.logger.warning(f"Failed to load quality gates config: {e}")
            return self._get_default_quality_gates()

    def _get_default_quality_gates(self) -> list[QualityGate]:
        """Get default quality gates configuration."""
        return [
            QualityGate(
                name="test_coverage",
                type="test_coverage",
                threshold=90.0,
                operator=">=",
                description="Minimum test coverage percentage",
                blocking=True,
            ),
            QualityGate(
                name="security_vulnerabilities",
                type="security_scan",
                threshold=0,
                operator="==",
                description="No high/critical security vulnerabilities",
                blocking=True,
            ),
            QualityGate(
                name="code_quality_score",
                type="code_quality",
                threshold=8.0,
                operator=">=",
                description="Minimum code quality score (1-10)",
                blocking=True,
            ),
            QualityGate(
                name="performance_regression",
                type="performance",
                threshold=10.0,
                operator="<=",
                description="Maximum performance regression percentage",
                blocking=True,
            ),
            QualityGate(
                name="build_success",
                type="build",
                threshold=1,
                operator="==",
                description="Build must succeed",
                blocking=True,
            ),
        ]

    async def execute_quality_gates(self) -> bool:
        """Execute all quality gates."""
        self.logger.info("Starting quality gates execution...")

        start_time = time.time()
        overall_success = True

        for gate in self.gates:
            self.logger.info(f"Executing quality gate: {gate.name}")

            gate_start = time.time()
            try:
                result = await self._execute_gate(gate)
                result.execution_time = time.time() - gate_start

                self.results.append(result)

                if not result.passed:
                    self.logger.error(
                        f"Quality gate '{gate.name}' failed: {result.message}"
                    )
                    if gate.blocking:
                        overall_success = False
                else:
                    self.logger.info(
                        f"Quality gate '{gate.name}' passed: {result.message}"
                    )

            except Exception as e:
                self.logger.error(f"Quality gate '{gate.name}' execution failed: {e}")
                result = QualityResult(
                    gate=gate,
                    passed=False,
                    actual_value="ERROR",
                    message=f"Execution failed: {e}",
                    execution_time=time.time() - gate_start,
                )
                self.results.append(result)

                if gate.blocking:
                    overall_success = False

        total_time = time.time() - start_time
        self.logger.info(f"Quality gates execution completed in {total_time:.2f}s")

        # Generate report
        await self._generate_report()

        return overall_success

    async def _execute_gate(self, gate: QualityGate) -> QualityResult:
        """Execute a single quality gate."""
        if gate.type == "test_coverage":
            return await self._check_test_coverage(gate)
        elif gate.type == "security_scan":
            return await self._check_security_vulnerabilities(gate)
        elif gate.type == "code_quality":
            return await self._check_code_quality(gate)
        elif gate.type == "performance":
            return await self._check_performance_regression(gate)
        elif gate.type == "build":
            return await self._check_build_success(gate)
        else:
            raise ValueError(f"Unknown quality gate type: {gate.type}")

    async def _check_test_coverage(self, gate: QualityGate) -> QualityResult:
        """Check test coverage quality gate."""
        try:
            # Run coverage report
            result = subprocess.run(
                ["coverage", "report", "--format=json"],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            if result.returncode != 0:
                return QualityResult(
                    gate=gate,
                    passed=False,
                    actual_value=0,
                    message="Failed to generate coverage report",
                )

            coverage_data = json.loads(result.stdout)
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)

            passed = self._evaluate_condition(
                total_coverage, gate.threshold, gate.operator
            )

            return QualityResult(
                gate=gate,
                passed=passed,
                actual_value=total_coverage,
                message=f"Test coverage: {total_coverage:.1f}% (threshold: {gate.operator} {gate.threshold}%)",
                details=coverage_data,
            )

        except Exception as e:
            return QualityResult(
                gate=gate,
                passed=False,
                actual_value=0,
                message=f"Coverage check failed: {e}",
            )

    async def _check_security_vulnerabilities(self, gate: QualityGate) -> QualityResult:
        """Check security vulnerabilities quality gate."""
        try:
            high_severity_count = 0
            security_details = {}

            # Run safety check
            safety_result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            if safety_result.stdout:
                safety_data = json.loads(safety_result.stdout)
                high_severity_count += len(
                    [
                        vuln
                        for vuln in safety_data
                        if vuln.get("severity", "").lower() in ["high", "critical"]
                    ]
                )
                security_details["safety"] = safety_data

            # Run bandit security scan
            bandit_result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            if bandit_result.stdout:
                bandit_data = json.loads(bandit_result.stdout)
                high_severity_count += len(
                    [
                        issue
                        for issue in bandit_data.get("results", [])
                        if issue.get("issue_severity") in ["HIGH", "CRITICAL"]
                    ]
                )
                security_details["bandit"] = bandit_data

            passed = self._evaluate_condition(
                high_severity_count, gate.threshold, gate.operator
            )

            return QualityResult(
                gate=gate,
                passed=passed,
                actual_value=high_severity_count,
                message=f"High/Critical vulnerabilities: {high_severity_count} (threshold: {gate.operator} {gate.threshold})",
                details=security_details,
            )

        except Exception as e:
            return QualityResult(
                gate=gate,
                passed=False,
                actual_value=-1,
                message=f"Security scan failed: {e}",
            )

    async def _check_code_quality(self, gate: QualityGate) -> QualityResult:
        """Check code quality quality gate."""
        try:
            quality_score = 10.0
            quality_details = {}

            # Run flake8
            flake8_result = subprocess.run(
                ["flake8", "src/", "--count", "--statistics"],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            if flake8_result.returncode != 0:
                quality_score -= 2.0
            quality_details["flake8"] = {
                "returncode": flake8_result.returncode,
                "output": flake8_result.stdout,
            }

            # Run mypy
            mypy_result = subprocess.run(
                ["mypy", "src/", "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            if mypy_result.returncode != 0:
                quality_score -= 1.5
            quality_details["mypy"] = {
                "returncode": mypy_result.returncode,
                "output": mypy_result.stdout,
            }

            # Check code formatting
            black_result = subprocess.run(
                ["black", "--check", "src/"],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            if black_result.returncode != 0:
                quality_score -= 1.0
            quality_details["black"] = {
                "returncode": black_result.returncode,
                "output": black_result.stdout,
            }

            # Check import sorting
            isort_result = subprocess.run(
                ["isort", "--check-only", "src/"],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            if isort_result.returncode != 0:
                quality_score -= 0.5
            quality_details["isort"] = {
                "returncode": isort_result.returncode,
                "output": isort_result.stdout,
            }

            passed = self._evaluate_condition(
                quality_score, gate.threshold, gate.operator
            )

            return QualityResult(
                gate=gate,
                passed=passed,
                actual_value=quality_score,
                message=f"Code quality score: {quality_score:.1f}/10 (threshold: {gate.operator} {gate.threshold})",
                details=quality_details,
            )

        except Exception as e:
            return QualityResult(
                gate=gate,
                passed=False,
                actual_value=0,
                message=f"Code quality check failed: {e}",
            )

    async def _check_performance_regression(self, gate: QualityGate) -> QualityResult:
        """Check performance regression quality gate."""
        try:
            # Run performance tests
            perf_result = subprocess.run(
                ["python", "-m", "pytest", "tests/performance/", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            # For now, assume no regression if tests pass
            # In production, you'd compare against baseline metrics
            regression_percentage = 0.0 if perf_result.returncode == 0 else 100.0

            passed = self._evaluate_condition(
                regression_percentage, gate.threshold, gate.operator
            )

            return QualityResult(
                gate=gate,
                passed=passed,
                actual_value=regression_percentage,
                message=f"Performance regression: {regression_percentage:.1f}% (threshold: {gate.operator} {gate.threshold}%)",
                details={"test_output": perf_result.stdout},
            )

        except Exception as e:
            return QualityResult(
                gate=gate,
                passed=False,
                actual_value=100.0,
                message=f"Performance check failed: {e}",
            )

    async def _check_build_success(self, gate: QualityGate) -> QualityResult:
        """Check build success quality gate."""
        try:
            # Check if build artifacts exist and are valid
            build_success = 1

            # Check Docker images can be built
            docker_result = subprocess.run(
                [
                    "docker",
                    "build",
                    "-f",
                    "deploy/production/Dockerfile.api",
                    "-t",
                    "test-build",
                    ".",
                ],
                capture_output=True,
                text=True,
                timeout=gate.timeout,
            )

            if docker_result.returncode != 0:
                build_success = 0

            passed = self._evaluate_condition(
                build_success, gate.threshold, gate.operator
            )

            return QualityResult(
                gate=gate,
                passed=passed,
                actual_value=build_success,
                message=f"Build success: {'Yes' if build_success else 'No'}",
                details={"docker_output": docker_result.stdout},
            )

        except Exception as e:
            return QualityResult(
                gate=gate,
                passed=False,
                actual_value=0,
                message=f"Build check failed: {e}",
            )

    def _evaluate_condition(
        self, actual: int | float, threshold: int | float, operator: str
    ) -> bool:
        """Evaluate quality gate condition."""
        if operator == ">=":
            return actual >= threshold
        elif operator == "<=":
            return actual <= threshold
        elif operator == ">":
            return actual > threshold
        elif operator == "<":
            return actual < threshold
        elif operator == "==":
            return actual == threshold
        elif operator == "!=":
            return actual != threshold
        else:
            raise ValueError(f"Unknown operator: {operator}")

    async def _generate_report(self):
        """Generate quality gates execution report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_success": all(
                r.passed or not r.gate.blocking for r in self.results
            ),
            "total_gates": len(self.gates),
            "passed_gates": len([r for r in self.results if r.passed]),
            "failed_gates": len([r for r in self.results if not r.passed]),
            "blocking_failures": len(
                [r for r in self.results if not r.passed and r.gate.blocking]
            ),
            "results": [
                {
                    "gate_name": r.gate.name,
                    "gate_type": r.gate.type,
                    "passed": r.passed,
                    "actual_value": r.actual_value,
                    "threshold": r.gate.threshold,
                    "operator": r.gate.operator,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "blocking": r.gate.blocking,
                }
                for r in self.results
            ],
        }

        # Save report
        os.makedirs("reports", exist_ok=True)
        report_file = f"reports/quality_gates_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Quality gates report saved: {report_file}")

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print quality gates execution summary."""
        print("\n" + "=" * 60)
        print("QUALITY GATES EXECUTION SUMMARY")
        print("=" * 60)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            blocking = "(BLOCKING)" if result.gate.blocking else "(NON-BLOCKING)"

            print(f"{status} {result.gate.name} {blocking}")
            print(f"    {result.message}")
            print(f"    Execution time: {result.execution_time:.2f}s")
            print()

        passed = len([r for r in self.results if r.passed])
        failed = len([r for r in self.results if not r.passed])
        blocking_failures = len(
            [r for r in self.results if not r.passed and r.gate.blocking]
        )

        print(f"Total gates: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Blocking failures: {blocking_failures}")

        overall_success = blocking_failures == 0
        print(f"\nOVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
        print("=" * 60)


async def main():
    """Main quality gates execution."""
    orchestrator = QualityGatesOrchestrator()

    success = await orchestrator.execute_quality_gates()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
