#!/usr/bin/env python3
"""
Buck2 Integration Validator
Validates Buck2 integration with the existing Pynomaly codebase and provides fallback strategies.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Buck2IntegrationValidator:
    """Validates Buck2 integration and provides fallback strategies."""

    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path.cwd()
        self.validation_results = {}

    def check_buck2_installation(self) -> Dict:
        """Check if Buck2 is installed and working."""
        logger.info("Checking Buck2 installation...")

        result = {"component": "buck2_installation", "status": "pending", "details": {}}

        try:
            # Check if buck2 command exists
            version_result = subprocess.run(
                ["buck2", "--version"], capture_output=True, text=True, timeout=10
            )

            if version_result.returncode == 0:
                result["status"] = "available"
                result["details"]["version"] = version_result.stdout.strip()
                result["details"]["installation"] = "system"
                logger.info(f"✓ Buck2 available: {result['details']['version']}")
            else:
                result["status"] = "error"
                result["details"]["error"] = version_result.stderr

        except FileNotFoundError:
            result["status"] = "not_installed"
            result["details"]["error"] = "Buck2 not found in PATH"
            logger.warning("✗ Buck2 not installed")

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["details"]["error"] = "Buck2 command timed out"

        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)

        return result

    def validate_buck_configuration(self) -> Dict:
        """Validate Buck2 configuration files."""
        logger.info("Validating Buck2 configuration...")

        result = {"component": "buck_configuration", "status": "pending", "details": {}}

        try:
            # Check BUCK file
            buck_file = self.repo_root / "BUCK"
            if buck_file.exists():
                result["details"]["buck_file"] = "exists"

                # Basic syntax check
                with open(buck_file, "r") as f:
                    content = f.read()

                # Check for required targets
                required_targets = [
                    "domain",
                    "application",
                    "infrastructure",
                    "presentation",
                    "test-domain",
                    "test-application",
                    "test-infrastructure",
                    "test-presentation",
                ]

                found_targets = []
                for target in required_targets:
                    if f'name = "{target}"' in content:
                        found_targets.append(target)

                result["details"]["required_targets"] = {
                    "found": found_targets,
                    "missing": list(set(required_targets) - set(found_targets)),
                }

            else:
                result["details"]["buck_file"] = "missing"

            # Check .buckconfig
            buckconfig_file = self.repo_root / ".buckconfig"
            if buckconfig_file.exists():
                result["details"]["buckconfig_file"] = "exists"
            else:
                result["details"]["buckconfig_file"] = "missing"

            # Overall status
            if (
                result["details"].get("buck_file") == "exists"
                and result["details"].get("buckconfig_file") == "exists"
            ):
                result["status"] = "valid"
                logger.info("✓ Buck2 configuration files found")
            else:
                result["status"] = "incomplete"
                logger.warning("✗ Buck2 configuration incomplete")

        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)

        return result

    def test_buck2_targets(self) -> Dict:
        """Test Buck2 targets if Buck2 is available."""
        logger.info("Testing Buck2 targets...")

        result = {"component": "buck2_targets", "status": "pending", "details": {}}

        try:
            # List targets
            targets_result = subprocess.run(
                ["buck2", "targets", "//..."],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.repo_root,
            )

            if targets_result.returncode == 0:
                targets = [
                    t.strip() for t in targets_result.stdout.split("\n") if t.strip()
                ]
                result["details"]["targets"] = targets
                result["details"]["target_count"] = len(targets)
                result["status"] = "success"
                logger.info(f"✓ Found {len(targets)} Buck2 targets")
            else:
                result["status"] = "failed"
                result["details"]["error"] = targets_result.stderr
                logger.error(f"✗ Failed to list Buck2 targets: {targets_result.stderr}")

        except FileNotFoundError:
            result["status"] = "buck2_not_available"
            result["details"]["error"] = "Buck2 not installed"

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["details"]["error"] = "Buck2 targets command timed out"

        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)

        return result

    def test_basic_build(self) -> Dict:
        """Test a basic Buck2 build if possible."""
        logger.info("Testing basic Buck2 build...")

        result = {"component": "basic_build", "status": "pending", "details": {}}

        try:
            # Try to build a simple target
            build_result = subprocess.run(
                ["buck2", "build", ":domain"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.repo_root,
            )

            if build_result.returncode == 0:
                result["status"] = "success"
                result["details"]["build_output"] = build_result.stdout
                logger.info("✓ Basic Buck2 build successful")
            else:
                result["status"] = "failed"
                result["details"]["error"] = build_result.stderr
                result["details"]["stdout"] = build_result.stdout
                logger.error(f"✗ Buck2 build failed: {build_result.stderr}")

        except FileNotFoundError:
            result["status"] = "buck2_not_available"
            result["details"]["error"] = "Buck2 not installed"

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["details"]["error"] = "Buck2 build timed out"

        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)

        return result

    def validate_python_environment(self) -> Dict:
        """Validate Python environment compatibility."""
        logger.info("Validating Python environment...")

        result = {"component": "python_environment", "status": "pending", "details": {}}

        try:
            # Check Python version
            result["details"]["python_version"] = sys.version
            result["details"]["python_executable"] = sys.executable

            # Check if we can import our scripts
            import_tests = [
                "buck2_change_detector",
                "buck2_incremental_test",
                "buck2_git_integration",
                "buck2_impact_analyzer",
                "buck2_workflow",
            ]

            sys.path.insert(0, str(self.repo_root / "scripts"))

            successful_imports = []
            failed_imports = []

            for module_name in import_tests:
                try:
                    __import__(module_name)
                    successful_imports.append(module_name)
                except Exception as e:
                    failed_imports.append({"module": module_name, "error": str(e)})

            result["details"]["successful_imports"] = successful_imports
            result["details"]["failed_imports"] = failed_imports

            if not failed_imports:
                result["status"] = "compatible"
                logger.info("✓ Python environment compatible")
            else:
                result["status"] = "issues"
                logger.warning(f"✗ {len(failed_imports)} import failures")

        except Exception as e:
            result["status"] = "error"
            result["details"]["error"] = str(e)

        return result

    def create_fallback_strategy(self, validation_results: Dict) -> Dict:
        """Create fallback strategy based on validation results."""
        logger.info("Creating fallback strategy...")

        buck2_available = (
            validation_results.get("buck2_installation", {}).get("status")
            == "available"
        )
        config_valid = (
            validation_results.get("buck_configuration", {}).get("status") == "valid"
        )
        python_compatible = (
            validation_results.get("python_environment", {}).get("status")
            == "compatible"
        )

        strategy = {
            "fallback_strategy": "determine",
            "recommendations": [],
            "alternatives": {},
        }

        if buck2_available and config_valid and python_compatible:
            strategy["fallback_strategy"] = "full_buck2"
            strategy["recommendations"] = [
                "Use Buck2 incremental testing system as designed",
                "Run full test suite with Buck2",
                "Enable Buck2 CI/CD integration",
            ]

        elif not buck2_available and config_valid and python_compatible:
            strategy["fallback_strategy"] = "pytest_based"
            strategy["recommendations"] = [
                "Use pytest with our incremental logic",
                "Map Buck2 targets to pytest commands",
                "Install Buck2 for full functionality",
            ]
            strategy["alternatives"][
                "pytest_commands"
            ] = self._generate_pytest_commands()

        elif python_compatible:
            strategy["fallback_strategy"] = "basic_incremental"
            strategy["recommendations"] = [
                "Use basic incremental testing",
                "Run change detection without Buck2",
                "Use pytest for actual test execution",
            ]
            strategy["alternatives"]["basic_commands"] = self._generate_basic_commands()

        else:
            strategy["fallback_strategy"] = "fix_environment"
            strategy["recommendations"] = [
                "Fix Python environment issues first",
                "Ensure all dependencies are installed",
                "Check Python path and module imports",
            ]

        return strategy

    def _generate_pytest_commands(self) -> Dict:
        """Generate pytest commands as Buck2 alternatives."""
        return {
            "domain_tests": "pytest tests/domain/ -v",
            "application_tests": "pytest tests/application/ -v",
            "infrastructure_tests": "pytest tests/infrastructure/ -v",
            "presentation_tests": "pytest tests/presentation/ -v",
            "integration_tests": "pytest tests/integration/ tests/e2e/ -v",
            "all_tests": "pytest -v",
            "coverage": "pytest --cov=src/pynomaly --cov-report=html",
        }

    def _generate_basic_commands(self) -> Dict:
        """Generate basic test commands."""
        return {
            "change_analysis": "python3 scripts/buck2_change_detector.py",
            "impact_analysis": "python3 scripts/buck2_impact_analyzer.py",
            "run_tests": "pytest -v",
            "workflow": "python3 scripts/buck2_workflow.py --help",
        }

    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive validation and return results."""
        logger.info("Starting comprehensive Buck2 integration validation...")
        start_time = time.time()

        results = {
            "validation_type": "buck2_integration",
            "start_time": start_time,
            "components": {},
            "summary": {},
            "fallback_strategy": {},
        }

        # Run all validation components
        validation_functions = [
            self.check_buck2_installation,
            self.validate_buck_configuration,
            self.validate_python_environment,
        ]

        # Only test Buck2 functionality if Buck2 is available
        for func in validation_functions:
            try:
                component_result = func()
                results["components"][component_result["component"]] = component_result
                logger.info(
                    f"✓ {component_result['component']}: {component_result['status']}"
                )
            except Exception as e:
                error_result = {
                    "component": func.__name__,
                    "status": "error",
                    "error": str(e),
                }
                results["components"][func.__name__] = error_result
                logger.error(f"✗ {func.__name__}: {e}")

        # Test Buck2 targets and build if Buck2 is available
        buck2_status = results["components"].get("buck2_installation", {}).get("status")
        if buck2_status == "available":
            try:
                targets_result = self.test_buck2_targets()
                results["components"]["buck2_targets"] = targets_result

                if targets_result["status"] == "success":
                    build_result = self.test_basic_build()
                    results["components"]["basic_build"] = build_result

            except Exception as e:
                logger.error(f"Buck2 testing failed: {e}")

        # Create fallback strategy
        results["fallback_strategy"] = self.create_fallback_strategy(
            results["components"]
        )

        # Generate summary
        total_components = len(results["components"])
        successful_components = sum(
            1
            for c in results["components"].values()
            if c["status"] in ["available", "valid", "compatible", "success"]
        )

        results["summary"] = {
            "total_components": total_components,
            "successful": successful_components,
            "success_rate": successful_components / total_components
            if total_components > 0
            else 0,
            "duration": time.time() - start_time,
            "overall_status": "ready"
            if successful_components >= total_components * 0.8
            else "needs_work",
        }

        return results

    def print_validation_summary(self, results: Dict):
        """Print human-readable validation summary."""
        print(f"\n=== Buck2 Integration Validation Results ===")
        print(f"Overall Status: {results['summary']['overall_status'].upper()}")
        print(f"Duration: {results['summary']['duration']:.2f}s")
        print(f"Success Rate: {results['summary']['success_rate']:.1%}")

        print(f"\nComponent Results:")
        for component, result in results["components"].items():
            status_symbol = {
                "available": "✓",
                "valid": "✓",
                "compatible": "✓",
                "success": "✓",
                "not_installed": "!",
                "incomplete": "!",
                "issues": "!",
                "failed": "✗",
                "error": "✗",
                "timeout": "!",
            }.get(result["status"], "?")

            print(f"  {status_symbol} {component}: {result['status']}")

            if result["status"] in ["error", "failed"] and "error" in result.get(
                "details", {}
            ):
                print(f"    Error: {result['details']['error']}")

        # Print fallback strategy
        strategy = results["fallback_strategy"]
        print(
            f"\n=== Recommended Strategy: {strategy['fallback_strategy'].upper()} ==="
        )

        if strategy["recommendations"]:
            print("Recommendations:")
            for rec in strategy["recommendations"]:
                print(f"  • {rec}")

        if strategy.get("alternatives"):
            print(f"\nAlternative Commands:")
            for category, commands in strategy["alternatives"].items():
                print(f"  {category}:")
                if isinstance(commands, dict):
                    for name, cmd in commands.items():
                        print(f"    {name}: {cmd}")
                else:
                    for cmd in commands:
                        print(f"    {cmd}")

    def save_validation_results(self, results: Dict, output_file: Path = None) -> Path:
        """Save validation results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = (
                self.repo_root / f"buck2_integration_validation_{timestamp}.json"
            )

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Validation results saved to {output_file}")
        return output_file


def main():
    """Main entry point for Buck2 integration validation."""
    parser = argparse.ArgumentParser(description="Buck2 Integration Validator")
    parser.add_argument(
        "--output", type=Path, help="Output file for validation results"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize validator
        validator = Buck2IntegrationValidator()

        # Run comprehensive validation
        results = validator.run_comprehensive_validation()

        # Print summary
        validator.print_validation_summary(results)

        # Save results
        if args.output:
            validator.save_validation_results(results, args.output)

        # Exit with appropriate code
        overall_status = results["summary"]["overall_status"]
        sys.exit(0 if overall_status == "ready" else 1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
