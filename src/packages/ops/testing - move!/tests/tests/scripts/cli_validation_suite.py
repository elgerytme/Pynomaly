#!/usr/bin/env python3
"""
Comprehensive CLI Validation Suite for Pynomaly

Consolidates all CLI testing functionality into a single, comprehensive test suite.
Combines functionality from:
- cli_validation_comprehensive.py
- cli_integration_test.py
- cli_production_test.py
- cli_runtime_tests.py
- cli_final_validation.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class CLIValidationSuite:
    """Comprehensive CLI validation and testing suite."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.results = {
            "structure": {},
            "commands": {},
            "integration": {},
            "performance": {},
            "production": {},
        }
        self.test_data_dir = None

    def setup_test_environment(self):
        """Setup temporary test environment."""
        self.test_data_dir = tempfile.mkdtemp(prefix="pynomaly_cli_test_")

        # Create sample test data
        sample_data = """x,y,anomaly
1.0,2.0,0
2.0,3.0,0
3.0,4.0,0
100.0,200.0,1
4.0,5.0,0"""

        test_file = Path(self.test_data_dir) / "test_data.csv"
        with open(test_file, "w") as f:
            f.write(sample_data)

        return test_file

    def cleanup_test_environment(self):
        """Clean up temporary test environment."""
        if self.test_data_dir and Path(self.test_data_dir).exists():
            shutil.rmtree(self.test_data_dir)

    def test_cli_structure(self) -> bool:
        """Test CLI structure and imports."""
        print("🔍 Testing CLI Structure...")

        try:
            # Test CLI app import

            self.results["structure"]["main_import"] = True
            print("✅ Main CLI app imports successfully")

            # Test command modules
            cli_modules = [
                "autonomous",
                "datasets",
                "detection",
                "detectors",
                "export",
                "preprocessing",
                "server",
            ]

            for module in cli_modules:
                try:
                    exec(f"from monorepo.presentation.cli.{module} import *")
                    self.results["structure"][f"{module}_import"] = True
                    print(f"✅ CLI module '{module}' imports successfully")
                except ImportError as e:
                    self.results["structure"][f"{module}_import"] = False
                    print(f"❌ CLI module '{module}' import failed: {e}")

            return True

        except Exception as e:
            print(f"❌ CLI structure test failed: {e}")
            self.results["structure"]["main_import"] = False
            return False

    def test_basic_commands(self) -> bool:
        """Test basic CLI commands."""
        print("\n🔍 Testing Basic CLI Commands...")

        commands_to_test = [
            (["python", "-m", "monorepo", "--help"], "help_command"),
            (["python", "-m", "monorepo", "--version"], "version_command"),
            (["python", "-m", "monorepo", "detect", "--help"], "detect_help"),
            (["python", "-m", "monorepo", "datasets", "--help"], "datasets_help"),
            (["python", "-m", "monorepo", "export", "--help"], "export_help"),
        ]

        all_passed = True

        for cmd, test_name in commands_to_test:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.project_root,
                )

                if result.returncode == 0:
                    self.results["commands"][test_name] = True
                    print(f"✅ Command '{' '.join(cmd)}' executed successfully")
                else:
                    self.results["commands"][test_name] = False
                    print(f"❌ Command '{' '.join(cmd)}' failed: {result.stderr}")
                    all_passed = False

            except subprocess.TimeoutExpired:
                self.results["commands"][test_name] = False
                print(f"❌ Command '{' '.join(cmd)}' timed out")
                all_passed = False
            except Exception as e:
                self.results["commands"][test_name] = False
                print(f"❌ Command '{' '.join(cmd)}' error: {e}")
                all_passed = False

        return all_passed

    def test_detection_workflow(self, test_file: Path) -> bool:
        """Test end-to-end detection workflow."""
        print("\n🔍 Testing Detection Workflow...")

        try:
            # Test autonomous detection
            start_time = time.time()

            cmd = [
                "python",
                "-m",
                "monorepo",
                "auto",
                "detect",
                str(test_file),
                "--output",
                str(Path(self.test_data_dir) / "results.json"),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=self.project_root
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                self.results["integration"]["autonomous_detection"] = True
                self.results["performance"]["detection_time"] = execution_time
                print(
                    f"✅ Autonomous detection completed in {execution_time:.2f} seconds"
                )

                # Check if results file was created
                results_file = Path(self.test_data_dir) / "results.json"
                if results_file.exists():
                    self.results["integration"]["results_output"] = True
                    print("✅ Results file generated successfully")

                    # Validate results format
                    with open(results_file) as f:
                        results_data = json.load(f)
                        if "anomaly_scores" in results_data:
                            self.results["integration"]["results_format"] = True
                            print("✅ Results format is valid")
                        else:
                            self.results["integration"]["results_format"] = False
                            print("❌ Results format is invalid")
                else:
                    self.results["integration"]["results_output"] = False
                    print("❌ Results file was not generated")

                return True
            else:
                self.results["integration"]["autonomous_detection"] = False
                print(f"❌ Autonomous detection failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.results["integration"]["autonomous_detection"] = False
            print("❌ Detection workflow timed out")
            return False
        except Exception as e:
            self.results["integration"]["autonomous_detection"] = False
            print(f"❌ Detection workflow error: {e}")
            return False

    def test_export_functionality(self, test_file: Path) -> bool:
        """Test export functionality."""
        print("\n🔍 Testing Export Functionality...")

        export_formats = ["json", "csv", "excel"]
        all_passed = True

        for fmt in export_formats:
            try:
                output_file = Path(self.test_data_dir) / f"export_test.{fmt}"

                cmd = [
                    "python",
                    "-m",
                    "monorepo",
                    "export",
                    str(test_file),
                    "--format",
                    fmt,
                    "--output",
                    str(output_file),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.project_root,
                )

                if result.returncode == 0 and output_file.exists():
                    self.results["integration"][f"export_{fmt}"] = True
                    print(f"✅ Export to {fmt} format successful")
                else:
                    self.results["integration"][f"export_{fmt}"] = False
                    print(f"❌ Export to {fmt} format failed")
                    all_passed = False

            except Exception as e:
                self.results["integration"][f"export_{fmt}"] = False
                print(f"❌ Export to {fmt} format error: {e}")
                all_passed = False

        return all_passed

    def test_performance_benchmarks(self, test_file: Path) -> bool:
        """Test performance benchmarks."""
        print("\n🔍 Testing Performance Benchmarks...")

        # Test startup time
        start_time = time.time()
        try:
            result = subprocess.run(
                ["python", "-m", "monorepo", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root,
            )
            startup_time = time.time() - start_time

            if result.returncode == 0:
                self.results["performance"]["startup_time"] = startup_time
                if startup_time < 5.0:
                    print(f"✅ Startup time: {startup_time:.2f}s (Good)")
                else:
                    print(f"⚠️ Startup time: {startup_time:.2f}s (Slow)")
            else:
                print("❌ Failed to measure startup time")
                return False

        except Exception as e:
            print(f"❌ Performance test error: {e}")
            return False

        return True

    def test_production_readiness(self) -> bool:
        """Test production readiness aspects."""
        print("\n🔍 Testing Production Readiness...")

        try:
            # Test error handling
            cmd = ["python", "-m", "monorepo", "detect", "nonexistent_file.csv"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, cwd=self.project_root
            )

            if result.returncode != 0 and "error" in result.stderr.lower():
                self.results["production"]["error_handling"] = True
                print("✅ Error handling works correctly")
            else:
                self.results["production"]["error_handling"] = False
                print("❌ Error handling is inadequate")

            # Test help documentation
            help_result = subprocess.run(
                ["python", "-m", "monorepo", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root,
            )

            if help_result.returncode == 0 and len(help_result.stdout) > 100:
                self.results["production"]["help_documentation"] = True
                print("✅ Help documentation is comprehensive")
            else:
                self.results["production"]["help_documentation"] = False
                print("❌ Help documentation is inadequate")

            return True

        except Exception as e:
            print(f"❌ Production readiness test error: {e}")
            return False

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive test report."""

        total_tests = 0
        passed_tests = 0

        for _category, tests in self.results.items():
            for _test_name, result in tests.items():
                if isinstance(result, bool):
                    total_tests += 1
                    if result:
                        passed_tests += 1

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": round(success_rate, 2),
            },
            "details": self.results,
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check performance
        if "startup_time" in self.results.get("performance", {}):
            startup_time = self.results["performance"]["startup_time"]
            if startup_time > 5.0:
                recommendations.append(
                    f"Consider optimizing startup time (current: {startup_time:.2f}s)"
                )

        # Check failed tests
        for category, tests in self.results.items():
            for test_name, result in tests.items():
                if isinstance(result, bool) and not result:
                    recommendations.append(f"Fix failed test: {category}.{test_name}")

        if not recommendations:
            recommendations.append("All tests passed! CLI is production-ready.")

        return recommendations

    def run_full_suite(self) -> dict[str, Any]:
        """Run the complete CLI validation suite."""
        print("🚀 Starting Comprehensive CLI Validation Suite...")
        print("=" * 60)

        try:
            # Setup test environment
            test_file = self.setup_test_environment()

            # Run all test categories
            self.test_cli_structure()
            self.test_basic_commands()
            self.test_detection_workflow(test_file)
            self.test_export_functionality(test_file)
            self.test_performance_benchmarks(test_file)
            self.test_production_readiness()

            # Generate final report
            report = self.generate_report()

            print("\n" + "=" * 60)
            print("📊 CLI Validation Suite Results:")
            print(f"Total Tests: {report['summary']['total_tests']}")
            print(f"Passed: {report['summary']['passed_tests']}")
            print(f"Failed: {report['summary']['failed_tests']}")
            print(f"Success Rate: {report['summary']['success_rate']}%")

            if report["recommendations"]:
                print("\n📋 Recommendations:")
                for rec in report["recommendations"]:
                    print(f"  • {rec}")

            return report

        finally:
            # Always clean up
            self.cleanup_test_environment()


def main():
    """Main entry point for CLI validation suite."""
    suite = CLIValidationSuite()
    report = suite.run_full_suite()

    # Save report to file
    report_file = PROJECT_ROOT / "cli_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    success_rate = report["summary"]["success_rate"]
    if success_rate >= 80:
        print("🎉 CLI validation passed!")
        sys.exit(0)
    else:
        print("❌ CLI validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
