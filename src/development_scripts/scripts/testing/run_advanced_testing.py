#!/usr/bin/env python3
"""
Advanced Testing Runner for anomaly_detection
Convenient CLI interface for running mutation testing, property-based testing, and comprehensive analysis.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ["hypothesis", "pytest", "pytest-cov"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False

    return True


def run_mutation_testing(args):
    """Run mutation testing only."""
    logger.info("Running mutation testing...")

    cmd = [
        sys.executable,
        "tests/advanced/mutation_testing_framework.py",
        "--source-dir",
        args.source_dir,
        "--test-dir",
        args.test_dir,
        "--max-mutations",
        str(args.max_mutations),
    ]

    if args.target_files:
        cmd.extend(["--target-files"] + args.target_files)

    if args.output_dir:
        output_file = Path(args.output_dir) / "mutation_results.json"
        cmd.extend(["--output", str(output_file)])

    if args.verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Mutation testing failed: {e}")
        return False


def run_property_testing(args):
    """Run property-based testing only."""
    logger.info("Running property-based testing...")

    cmd = [
        sys.executable,
        "tests/advanced/property_testing_framework.py",
        "--max-examples",
        str(args.max_examples),
        "--timeout",
        str(args.timeout),
    ]

    if args.target_files:
        cmd.extend(["--target-files"] + args.target_files)

    if args.output_dir:
        output_file = Path(args.output_dir) / "property_results.json"
        cmd.extend(["--output", str(output_file)])

    if args.verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Property testing failed: {e}")
        return False


def run_comprehensive_testing(args):
    """Run comprehensive advanced testing."""
    logger.info("Running comprehensive advanced testing...")

    cmd = [
        sys.executable,
        "tests/advanced/advanced_testing_orchestrator.py",
        "--source-dir",
        args.source_dir,
        "--test-dir",
        args.test_dir,
        "--output-dir",
        args.output_dir,
        "--max-mutations",
        str(args.max_mutations),
        "--max-examples",
        str(args.max_examples),
    ]

    if args.target_files:
        cmd.extend(["--target-files"] + args.target_files)

    if args.config:
        cmd.extend(["--config", str(args.config)])

    if args.parallel:
        cmd.append("--parallel")

    if args.verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Comprehensive testing failed: {e}")
        return False


def generate_config_template(output_file: Path):
    """Generate a configuration template file."""
    config = {
        "source_dir": "src/anomaly_detection",
        "test_dir": "tests",
        "output_dir": "reports",
        "mutation_config": {"max_mutations": 100, "timeout": 60, "parallel": True},
        "property_config": {"max_examples": 100, "timeout": 60},
        "traditional_tests": {
            "enable_coverage": True,
            "coverage_threshold": 0.8,
            "enable_doctests": True,
        },
        "quality_thresholds": {
            "mutation_score_target": 80.0,
            "property_coverage_target": 70.0,
            "code_coverage_target": 85.0,
            "overall_effectiveness_target": 80.0,
        },
        "parallel_execution": True,
    }

    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration template saved to {output_file}")


def setup_advanced_testing():
    """Set up advanced testing infrastructure."""
    logger.info("Setting up advanced testing infrastructure...")

    # Create necessary directories
    directories = ["tests/advanced", "reports", "storage"]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    # Check if test files exist
    test_files = [
        "tests/advanced/mutation_testing_framework.py",
        "tests/advanced/property_testing_framework.py",
        "tests/advanced/advanced_testing_orchestrator.py",
    ]

    for test_file in test_files:
        if not Path(test_file).exists():
            logger.warning(f"Missing test framework file: {test_file}")

    # Generate configuration template
    config_file = Path("advanced_testing_config.json")
    if not config_file.exists():
        generate_config_template(config_file)

    logger.info("Advanced testing setup completed!")


def validate_environment():
    """Validate that the environment is set up correctly."""
    logger.info("Validating environment...")

    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ is required")

    # Check required directories
    required_dirs = ["src/anomaly_detection", "tests"]
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            issues.append(f"Required directory missing: {dir_path}")

    # Check dependencies
    if not check_dependencies():
        issues.append("Missing required Python packages")

    # Check for test files
    if not any(Path("tests").rglob("test_*.py")):
        issues.append("No test files found in tests/ directory")

    if issues:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("Environment validation passed!")
        return True


def clean_reports(output_dir: str):
    """Clean old test reports."""
    reports_dir = Path(output_dir)
    if reports_dir.exists():
        logger.info(f"Cleaning reports directory: {output_dir}")
        shutil.rmtree(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Reports directory cleaned")
    else:
        logger.info("Reports directory doesn't exist, creating...")
        reports_dir.mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point for advanced testing runner."""
    parser = argparse.ArgumentParser(description="Advanced Testing Runner for anomaly_detection")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Set up advanced testing infrastructure"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate environment setup"
    )

    # Generate config command
    config_parser = subparsers.add_parser(
        "generate-config", help="Generate configuration template"
    )
    config_parser.add_argument(
        "--output", default="advanced_testing_config.json", help="Output file"
    )

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean old test reports")
    clean_parser.add_argument(
        "--output-dir", default="reports", help="Reports directory to clean"
    )

    # Mutation testing command
    mutation_parser = subparsers.add_parser(
        "mutation", help="Run mutation testing only"
    )
    mutation_parser.add_argument(
        "--source-dir", default="src/anomaly_detection", help="Source directory"
    )
    mutation_parser.add_argument("--test-dir", default="tests", help="Test directory")
    mutation_parser.add_argument(
        "--target-files", nargs="+", help="Specific files to test"
    )
    mutation_parser.add_argument(
        "--max-mutations", type=int, default=100, help="Maximum mutations to test"
    )
    mutation_parser.add_argument(
        "--output-dir", default="reports", help="Output directory"
    )
    mutation_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Property testing command
    property_parser = subparsers.add_parser(
        "property", help="Run property-based testing only"
    )
    property_parser.add_argument(
        "--target-files", nargs="+", help="Specific files to test"
    )
    property_parser.add_argument(
        "--max-examples", type=int, default=100, help="Maximum examples per property"
    )
    property_parser.add_argument(
        "--timeout", type=int, default=60, help="Timeout per property test"
    )
    property_parser.add_argument(
        "--output-dir", default="reports", help="Output directory"
    )
    property_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Comprehensive testing command
    comprehensive_parser = subparsers.add_parser(
        "comprehensive", help="Run comprehensive advanced testing"
    )
    comprehensive_parser.add_argument(
        "--source-dir", default="src/anomaly_detection", help="Source directory"
    )
    comprehensive_parser.add_argument(
        "--test-dir", default="tests", help="Test directory"
    )
    comprehensive_parser.add_argument(
        "--target-files", nargs="+", help="Specific files to test"
    )
    comprehensive_parser.add_argument("--config", type=Path, help="Configuration file")
    comprehensive_parser.add_argument(
        "--max-mutations", type=int, default=50, help="Maximum mutations to test"
    )
    comprehensive_parser.add_argument(
        "--max-examples", type=int, default=50, help="Maximum examples per property"
    )
    comprehensive_parser.add_argument(
        "--output-dir", default="reports", help="Output directory"
    )
    comprehensive_parser.add_argument(
        "--parallel", action="store_true", default=True, help="Run tests in parallel"
    )
    comprehensive_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # All command (alias for comprehensive)
    all_parser = subparsers.add_parser(
        "all", help="Run all advanced testing (alias for comprehensive)"
    )
    all_parser.add_argument(
        "--source-dir", default="src/anomaly_detection", help="Source directory"
    )
    all_parser.add_argument("--test-dir", default="tests", help="Test directory")
    all_parser.add_argument("--target-files", nargs="+", help="Specific files to test")
    all_parser.add_argument("--config", type=Path, help="Configuration file")
    all_parser.add_argument(
        "--max-mutations", type=int, default=50, help="Maximum mutations to test"
    )
    all_parser.add_argument(
        "--max-examples", type=int, default=50, help="Maximum examples per property"
    )
    all_parser.add_argument("--output-dir", default="reports", help="Output directory")
    all_parser.add_argument(
        "--parallel", action="store_true", default=True, help="Run tests in parallel"
    )
    all_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose if hasattr(args, "verbose") else False:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle commands
    if args.command == "setup":
        setup_advanced_testing()
        return 0

    elif args.command == "validate":
        return 0 if validate_environment() else 1

    elif args.command == "generate-config":
        generate_config_template(Path(args.output))
        return 0

    elif args.command == "clean":
        clean_reports(args.output_dir)
        return 0

    elif args.command == "mutation":
        if not check_dependencies():
            return 1
        return 0 if run_mutation_testing(args) else 1

    elif args.command == "property":
        if not check_dependencies():
            return 1
        return 0 if run_property_testing(args) else 1

    elif args.command in ["comprehensive", "all"]:
        if not check_dependencies():
            return 1
        return 0 if run_comprehensive_testing(args) else 1

    else:
        parser.print_help()
        print("\nQuick start:")
        print(
            "  python scripts/run_advanced_testing.py setup          # Set up infrastructure"
        )
        print(
            "  python scripts/run_advanced_testing.py validate       # Validate environment"
        )
        print("  python scripts/run_advanced_testing.py comprehensive  # Run all tests")
        print(
            "  python scripts/run_advanced_testing.py mutation       # Run mutation testing only"
        )
        print(
            "  python scripts/run_advanced_testing.py property       # Run property testing only"
        )
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)
