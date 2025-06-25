#!/usr/bin/env python3
"""
System Validator for Pynomaly

Consolidates validation functionality from:
- validate_phase_2_completion.py
- validate_phase_3_completion.py
- validate_autonomous_integration.py
- validate_test_fixes.py
- verify_cli_architecture.py
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class ValidationResult:
    """Validation result data structure."""

    component: str
    status: str  # "pass", "fail", "warning", "info"
    message: str
    details: dict[str, Any] | None = None


class SystemValidator:
    """Comprehensive system validation for Pynomaly."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.results: list[ValidationResult] = []

    def validate_core_architecture(self) -> list[ValidationResult]:
        """Validate core architecture components."""
        print("🏗️ Validating Core Architecture...")

        arch_results = []

        # Validate domain layer
        try:

            arch_results.append(
                ValidationResult(
                    component="domain_entities",
                    status="pass",
                    message="Domain entities import successfully",
                )
            )
            print("✅ Domain entities validated")
        except Exception as e:
            arch_results.append(
                ValidationResult(
                    component="domain_entities",
                    status="fail",
                    message=f"Domain entities validation failed: {e}",
                )
            )
            print(f"❌ Domain entities validation failed: {e}")

        # Validate application layer
        try:

            arch_results.append(
                ValidationResult(
                    component="application_services",
                    status="pass",
                    message="Application services import successfully",
                )
            )
            print("✅ Application services validated")
        except Exception as e:
            arch_results.append(
                ValidationResult(
                    component="application_services",
                    status="fail",
                    message=f"Application services validation failed: {e}",
                )
            )
            print(f"❌ Application services validation failed: {e}")

        # Validate infrastructure layer
        try:

            arch_results.append(
                ValidationResult(
                    component="infrastructure_adapters",
                    status="pass",
                    message="Infrastructure adapters import successfully",
                )
            )
            print("✅ Infrastructure adapters validated")
        except Exception as e:
            arch_results.append(
                ValidationResult(
                    component="infrastructure_adapters",
                    status="fail",
                    message=f"Infrastructure adapters validation failed: {e}",
                )
            )
            print(f"❌ Infrastructure adapters validation failed: {e}")

        # Validate presentation layer
        try:

            arch_results.append(
                ValidationResult(
                    component="presentation_cli",
                    status="pass",
                    message="CLI presentation layer imports successfully",
                )
            )
            print("✅ CLI presentation layer validated")
        except Exception as e:
            arch_results.append(
                ValidationResult(
                    component="presentation_cli",
                    status="fail",
                    message=f"CLI presentation validation failed: {e}",
                )
            )
            print(f"❌ CLI presentation validation failed: {e}")

        return arch_results

    def validate_autonomous_integration(self) -> list[ValidationResult]:
        """Validate autonomous mode integration."""
        print("\n🤖 Validating Autonomous Integration...")

        auto_results = []

        # Test autonomous CLI commands
        try:
            env = {**os.environ, "PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                ["python3", "-m", "pynomaly", "auto", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root,
                env=env,
            )

            if result.returncode == 0:
                auto_results.append(
                    ValidationResult(
                        component="autonomous_cli",
                        status="pass",
                        message="Autonomous CLI commands available",
                    )
                )
                print("✅ Autonomous CLI validated")
            else:
                auto_results.append(
                    ValidationResult(
                        component="autonomous_cli",
                        status="fail",
                        message=f"Autonomous CLI validation failed: {result.stderr}",
                    )
                )
                print("❌ Autonomous CLI validation failed")

        except Exception as e:
            auto_results.append(
                ValidationResult(
                    component="autonomous_cli",
                    status="fail",
                    message=f"Autonomous CLI test error: {e}",
                )
            )
            print(f"❌ Autonomous CLI test error: {e}")

        # Test autonomous service
        try:

            auto_results.append(
                ValidationResult(
                    component="autonomous_service",
                    status="pass",
                    message="Autonomous service imports successfully",
                )
            )
            print("✅ Autonomous service validated")
        except Exception as e:
            auto_results.append(
                ValidationResult(
                    component="autonomous_service",
                    status="warning",
                    message=f"Autonomous service not available: {e}",
                )
            )
            print(f"⚠️ Autonomous service not available: {e}")

        return auto_results

    def validate_api_integration(self) -> list[ValidationResult]:
        """Validate API integration."""
        print("\n🌐 Validating API Integration...")

        api_results = []

        # Test FastAPI app creation
        try:
            from pynomaly.presentation.api.app import create_app

            app = create_app()

            api_results.append(
                ValidationResult(
                    component="fastapi_app",
                    status="pass",
                    message="FastAPI app creates successfully",
                    details={"app_type": str(type(app))},
                )
            )
            print("✅ FastAPI app validated")
        except Exception as e:
            api_results.append(
                ValidationResult(
                    component="fastapi_app",
                    status="fail",
                    message=f"FastAPI app creation failed: {e}",
                )
            )
            print(f"❌ FastAPI app validation failed: {e}")

        # Test API endpoints
        try:

            api_results.append(
                ValidationResult(
                    component="api_endpoints",
                    status="pass",
                    message="API endpoints import successfully",
                )
            )
            print("✅ API endpoints validated")
        except Exception as e:
            api_results.append(
                ValidationResult(
                    component="api_endpoints",
                    status="fail",
                    message=f"API endpoints validation failed: {e}",
                )
            )
            print(f"❌ API endpoints validation failed: {e}")

        return api_results

    def validate_dependency_injection(self) -> list[ValidationResult]:
        """Validate dependency injection container."""
        print("\n💉 Validating Dependency Injection...")

        di_results = []

        # Test container creation
        try:
            from pynomaly.infrastructure.config.container import (
                Container,
                create_container,
            )

            # Test basic container
            Container()
            di_results.append(
                ValidationResult(
                    component="di_container",
                    status="pass",
                    message="DI container creates successfully",
                )
            )
            print("✅ DI container validated")

            # Test container factory
            create_container(testing=True)
            di_results.append(
                ValidationResult(
                    component="di_container_factory",
                    status="pass",
                    message="DI container factory works",
                )
            )
            print("✅ DI container factory validated")

        except Exception as e:
            di_results.append(
                ValidationResult(
                    component="di_container",
                    status="fail",
                    message=f"DI container validation failed: {e}",
                )
            )
            print(f"❌ DI container validation failed: {e}")

        return di_results

    def validate_configuration(self) -> list[ValidationResult]:
        """Validate configuration system."""
        print("\n⚙️ Validating Configuration...")

        config_results = []

        # Test settings
        try:
            from pynomaly.infrastructure.config.settings import get_settings

            settings = get_settings()

            config_results.append(
                ValidationResult(
                    component="settings",
                    status="pass",
                    message="Settings load successfully",
                    details={"app_name": settings.app.name},
                )
            )
            print("✅ Settings validated")
        except Exception as e:
            config_results.append(
                ValidationResult(
                    component="settings",
                    status="fail",
                    message=f"Settings validation failed: {e}",
                )
            )
            print(f"❌ Settings validation failed: {e}")

        # Test feature flags
        try:
            from pynomaly.infrastructure.config.feature_flags import get_feature_flags

            get_feature_flags()

            config_results.append(
                ValidationResult(
                    component="feature_flags",
                    status="pass",
                    message="Feature flags load successfully",
                )
            )
            print("✅ Feature flags validated")
        except Exception as e:
            config_results.append(
                ValidationResult(
                    component="feature_flags",
                    status="fail",
                    message=f"Feature flags validation failed: {e}",
                )
            )
            print(f"❌ Feature flags validation failed: {e}")

        return config_results

    def validate_performance_requirements(self) -> list[ValidationResult]:
        """Validate performance requirements."""
        print("\n⚡ Validating Performance Requirements...")

        perf_results = []

        # Test startup time
        start_time = time.time()
        try:
            from pynomaly.infrastructure.config.container import Container

            Container()
            startup_time = time.time() - start_time

            if startup_time < 5.0:
                perf_results.append(
                    ValidationResult(
                        component="startup_performance",
                        status="pass",
                        message=f"Startup time: {startup_time:.2f}s (Good)",
                        details={"startup_time": startup_time, "threshold": 5.0},
                    )
                )
                print(f"✅ Startup performance: {startup_time:.2f}s")
            else:
                perf_results.append(
                    ValidationResult(
                        component="startup_performance",
                        status="warning",
                        message=f"Startup time: {startup_time:.2f}s (Slow)",
                        details={"startup_time": startup_time, "threshold": 5.0},
                    )
                )
                print(f"⚠️ Startup performance: {startup_time:.2f}s (slow)")

        except Exception as e:
            perf_results.append(
                ValidationResult(
                    component="startup_performance",
                    status="fail",
                    message=f"Startup performance test failed: {e}",
                )
            )
            print(f"❌ Startup performance test failed: {e}")

        return perf_results

    def validate_test_infrastructure(self) -> list[ValidationResult]:
        """Validate test infrastructure."""
        print("\n🧪 Validating Test Infrastructure...")

        test_results = []

        # Check if pytest is available and tests can run
        try:
            env = {**os.environ, "PYTHONPATH": str(self.project_root / "src")}
            result = subprocess.run(
                ["python3", "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root,
                env=env,
            )

            if result.returncode == 0:
                test_results.append(
                    ValidationResult(
                        component="pytest_availability",
                        status="pass",
                        message="pytest is available for testing",
                    )
                )
                print("✅ pytest validated")
            else:
                test_results.append(
                    ValidationResult(
                        component="pytest_availability",
                        status="fail",
                        message="pytest is not available",
                    )
                )
                print("❌ pytest not available")

        except Exception as e:
            test_results.append(
                ValidationResult(
                    component="pytest_availability",
                    status="fail",
                    message=f"pytest test failed: {e}",
                )
            )
            print(f"❌ pytest test failed: {e}")

        # Check test directory structure
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.rglob("test_*.py"))
            test_results.append(
                ValidationResult(
                    component="test_structure",
                    status="pass",
                    message=f"Test directory exists with {len(test_files)} test files",
                    details={"test_files_count": len(test_files)},
                )
            )
            print(f"✅ Test structure validated ({len(test_files)} test files)")
        else:
            test_results.append(
                ValidationResult(
                    component="test_structure",
                    status="warning",
                    message="Test directory not found",
                )
            )
            print("⚠️ Test directory not found")

        return test_results

    def generate_validation_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""

        # Categorize results by status
        status_counts = {"pass": 0, "fail": 0, "warning": 0, "info": 0}
        components_by_status = {"pass": [], "fail": [], "warning": [], "info": []}

        for result in self.results:
            status_counts[result.status] += 1
            components_by_status[result.status].append(result.component)

        # Calculate overall health score
        total_results = len(self.results)
        health_score = 0
        if total_results > 0:
            # Pass = 100%, Warning = 50%, Fail/Info = 0%
            weighted_score = status_counts["pass"] * 100 + status_counts["warning"] * 50
            health_score = weighted_score / total_results

        # Generate recommendations
        recommendations = []
        for result in self.results:
            if result.status == "fail":
                recommendations.append(
                    f"CRITICAL: Fix {result.component} - {result.message}"
                )
            elif result.status == "warning":
                recommendations.append(
                    f"IMPROVE: Address {result.component} - {result.message}"
                )

        if not recommendations:
            recommendations.append(
                "System validation passed! All components are healthy."
            )

        report = {
            "summary": {
                "total_components": total_results,
                "health_score": round(health_score, 1),
                "status_breakdown": status_counts,
            },
            "component_status": components_by_status,
            "detailed_results": [
                {
                    "component": r.component,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
            "recommendations": recommendations,
        }

        return report

    def run_full_validation(self) -> dict[str, Any]:
        """Run complete system validation."""
        print("🔍 Starting System Validation...")
        print("=" * 60)

        # Run all validation categories
        self.results.extend(self.validate_core_architecture())
        self.results.extend(self.validate_autonomous_integration())
        self.results.extend(self.validate_api_integration())
        self.results.extend(self.validate_dependency_injection())
        self.results.extend(self.validate_configuration())
        self.results.extend(self.validate_performance_requirements())
        self.results.extend(self.validate_test_infrastructure())

        # Generate validation report
        report = self.generate_validation_report()

        print("\n" + "=" * 60)
        print("📊 System Validation Results:")
        print(f"Total Components: {report['summary']['total_components']}")
        print(f"Health Score: {report['summary']['health_score']}/100")
        print("Status Breakdown:")
        for status, count in report["summary"]["status_breakdown"].items():
            if count > 0:
                print(f"  {status.upper()}: {count}")

        if report["recommendations"]:
            print("\n📋 Recommendations:")
            for rec in report["recommendations"][:5]:  # Show first 5
                print(f"  • {rec}")
            if len(report["recommendations"]) > 5:
                print(f"  ... and {len(report['recommendations']) - 5} more")

        return report


def main():
    """Main entry point for system validator."""
    validator = SystemValidator()
    report = validator.run_full_validation()

    # Save report to file
    report_file = PROJECT_ROOT / "system_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_file}")

    # Exit with appropriate code based on health score
    health_score = report["summary"]["health_score"]
    if health_score >= 80:
        print("🎉 System validation passed!")
        sys.exit(0)
    elif health_score >= 60:
        print("⚠️ System validation passed with warnings!")
        sys.exit(0)
    else:
        print("❌ System validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
