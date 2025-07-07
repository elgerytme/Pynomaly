#!/usr/bin/env python3
"""
Advanced Testing Orchestrator for Pynomaly
Integrates mutation testing, property-based testing, and comprehensive analysis.
"""

import argparse
import concurrent.futures
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our testing frameworks
try:
    from .mutation_testing_framework import MutationTester, MutationTestSummary
    from .property_testing_framework import PropertyTester, PropertyTestSummary
except ImportError:
    # Handle direct execution
    import sys

    sys.path.append(str(Path(__file__).parent))
    from mutation_testing_framework import MutationTester, MutationTestSummary
    from property_testing_framework import PropertyTester, PropertyTestSummary

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestQualityMetrics:
    """Comprehensive test quality metrics."""

    mutation_score: float
    property_coverage: float
    code_coverage: float
    test_effectiveness: float
    risk_assessment: str
    recommendations: List[str]


@dataclass
class AdvancedTestResult:
    """Comprehensive testing results."""

    success: bool
    duration: float
    mutation_results: Optional[MutationTestSummary]
    property_results: Optional[PropertyTestSummary]
    traditional_test_results: Dict
    quality_metrics: TestQualityMetrics
    detailed_analysis: Dict


class AdvancedTestingOrchestrator:
    """Orchestrates advanced testing including mutation and property-based testing."""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.source_dir = Path(self.config.get("source_dir", "src/pynomaly"))
        self.test_dir = Path(self.config.get("test_dir", "tests"))
        self.output_dir = Path(self.config.get("output_dir", "reports"))
        self.output_dir.mkdir(exist_ok=True)

        # Initialize testing frameworks
        self.mutation_tester = None
        self.property_tester = None

    def _default_config(self) -> Dict:
        """Default configuration for advanced testing."""
        return {
            "source_dir": "src/pynomaly",
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
        }

    def run_comprehensive_testing(
        self, target_files: List[str] = None
    ) -> AdvancedTestResult:
        """Run complete advanced testing suite."""
        logger.info("Starting comprehensive advanced testing...")
        start_time = time.time()

        # Initialize results
        mutation_results = None
        property_results = None
        traditional_results = {}

        try:
            # Run traditional tests first
            logger.info("Running traditional test suite...")
            traditional_results = self._run_traditional_tests()

            if not traditional_results.get("passed", False):
                logger.error("Traditional tests failed - stopping advanced testing")
                return AdvancedTestResult(
                    success=False,
                    duration=time.time() - start_time,
                    mutation_results=None,
                    property_results=None,
                    traditional_test_results=traditional_results,
                    quality_metrics=self._calculate_default_metrics(),
                    detailed_analysis={"error": "Traditional tests failed"},
                )

            # Run advanced tests in parallel if configured
            if self.config.get("parallel_execution", True):
                mutation_results, property_results = self._run_parallel_advanced_tests(
                    target_files
                )
            else:
                mutation_results = self._run_mutation_testing(target_files)
                property_results = self._run_property_testing(target_files)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                mutation_results, property_results, traditional_results
            )

            # Perform detailed analysis
            detailed_analysis = self._perform_detailed_analysis(
                mutation_results, property_results, traditional_results
            )

            duration = time.time() - start_time
            success = self._evaluate_overall_success(quality_metrics)

            logger.info(f"Comprehensive testing completed in {duration:.2f}s")

            return AdvancedTestResult(
                success=success,
                duration=duration,
                mutation_results=mutation_results,
                property_results=property_results,
                traditional_test_results=traditional_results,
                quality_metrics=quality_metrics,
                detailed_analysis=detailed_analysis,
            )

        except Exception as e:
            logger.error(f"Advanced testing failed: {e}")
            return AdvancedTestResult(
                success=False,
                duration=time.time() - start_time,
                mutation_results=mutation_results,
                property_results=property_results,
                traditional_test_results=traditional_results,
                quality_metrics=self._calculate_default_metrics(),
                detailed_analysis={"error": str(e)},
            )

    def _run_traditional_tests(self) -> Dict:
        """Run traditional pytest-based tests."""
        logger.info("Executing traditional test suite...")
        results = {
            "passed": False,
            "coverage": 0.0,
            "test_count": 0,
            "failures": 0,
            "errors": 0,
            "duration": 0.0,
        }

        try:
            start_time = time.time()

            # Run tests with coverage
            cmd = [
                "python",
                "-m",
                "pytest",
                str(self.test_dir),
                "--cov=" + str(self.source_dir),
                "--cov-report=json",
                "--cov-report=term-missing",
                "-v",
                "--tb=short",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, cwd=Path.cwd()
            )

            results["duration"] = time.time() - start_time
            results["passed"] = result.returncode == 0

            # Parse pytest output for test counts
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "failed" in line or "passed" in line:
                    # Extract test counts from pytest summary
                    words = line.split()
                    for i, word in enumerate(words):
                        if word == "passed" and i > 0:
                            try:
                                results["test_count"] += int(words[i - 1])
                            except ValueError:
                                pass
                        elif word == "failed" and i > 0:
                            try:
                                results["failures"] = int(words[i - 1])
                            except ValueError:
                                pass
                        elif word == "error" and i > 0:
                            try:
                                results["errors"] = int(words[i - 1])
                            except ValueError:
                                pass

            # Read coverage report
            coverage_file = Path.cwd() / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, "r") as f:
                    coverage_data = json.load(f)
                    results["coverage"] = (
                        coverage_data.get("totals", {}).get("percent_covered", 0.0)
                        / 100.0
                    )

            logger.info(
                f"Traditional tests: {results['test_count']} tests, "
                f"{results['failures']} failures, {results['errors']} errors, "
                f"{results['coverage']:.1%} coverage"
            )

        except subprocess.TimeoutExpired:
            logger.error("Traditional tests timed out")
            results["errors"] = 1
        except Exception as e:
            logger.error(f"Traditional tests failed: {e}")
            results["errors"] = 1

        return results

    def _run_parallel_advanced_tests(self, target_files: List[str] = None) -> tuple:
        """Run mutation and property testing in parallel."""
        logger.info("Running mutation and property testing in parallel...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            mutation_future = executor.submit(self._run_mutation_testing, target_files)
            property_future = executor.submit(self._run_property_testing, target_files)

            # Wait for completion
            concurrent.futures.wait([mutation_future, property_future])

            # Get results
            try:
                mutation_results = mutation_future.result()
            except Exception as e:
                logger.error(f"Mutation testing failed: {e}")
                mutation_results = None

            try:
                property_results = property_future.result()
            except Exception as e:
                logger.error(f"Property testing failed: {e}")
                property_results = None

        return mutation_results, property_results

    def _run_mutation_testing(
        self, target_files: List[str] = None
    ) -> Optional[MutationTestSummary]:
        """Run mutation testing."""
        try:
            logger.info("Starting mutation testing...")

            if not self.mutation_tester:
                self.mutation_tester = MutationTester(
                    source_dir=self.source_dir, test_dir=self.test_dir
                )

            max_mutations = self.config["mutation_config"].get("max_mutations", 100)

            results = self.mutation_tester.run_mutation_testing(
                target_files=target_files, max_mutations=max_mutations
            )

            # Save results
            results_file = self.output_dir / "mutation_test_results.json"
            self.mutation_tester.save_results(results, results_file)

            logger.info(
                f"Mutation testing completed: {results.mutation_score:.1f}% score"
            )
            return results

        except Exception as e:
            logger.error(f"Mutation testing failed: {e}")
            return None

    def _run_property_testing(
        self, target_files: List[str] = None
    ) -> Optional[PropertyTestSummary]:
        """Run property-based testing."""
        try:
            logger.info("Starting property-based testing...")

            if not self.property_tester:
                max_examples = self.config["property_config"].get("max_examples", 100)
                timeout = self.config["property_config"].get("timeout", 60)
                self.property_tester = PropertyTester(
                    max_examples=max_examples, timeout=timeout
                )

            results = self.property_tester.run_property_testing(
                target_files=target_files
            )

            # Save results
            results_file = self.output_dir / "property_test_results.json"
            self.property_tester.save_results(results, results_file)

            coverage_pct = (
                (results.passed_properties / results.total_properties * 100)
                if results.total_properties > 0
                else 0
            )
            logger.info(
                f"Property testing completed: {coverage_pct:.1f}% properties passed"
            )
            return results

        except Exception as e:
            logger.error(f"Property testing failed: {e}")
            return None

    def _calculate_quality_metrics(
        self,
        mutation_results: Optional[MutationTestSummary],
        property_results: Optional[PropertyTestSummary],
        traditional_results: Dict,
    ) -> TestQualityMetrics:
        """Calculate comprehensive test quality metrics."""

        # Extract individual metrics
        mutation_score = mutation_results.mutation_score if mutation_results else 0.0

        property_coverage = 0.0
        if property_results and property_results.total_properties > 0:
            property_coverage = (
                property_results.passed_properties / property_results.total_properties
            ) * 100

        code_coverage = traditional_results.get("coverage", 0.0) * 100

        # Calculate overall test effectiveness
        weights = {"mutation": 0.4, "property": 0.3, "coverage": 0.3}
        test_effectiveness = (
            mutation_score * weights["mutation"]
            + property_coverage * weights["property"]
            + code_coverage * weights["coverage"]
        )

        # Risk assessment
        risk_assessment = self._assess_risk(
            mutation_score, property_coverage, code_coverage, test_effectiveness
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            mutation_score, property_coverage, code_coverage, test_effectiveness
        )

        return TestQualityMetrics(
            mutation_score=mutation_score,
            property_coverage=property_coverage,
            code_coverage=code_coverage,
            test_effectiveness=test_effectiveness,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
        )

    def _assess_risk(
        self,
        mutation_score: float,
        property_coverage: float,
        code_coverage: float,
        test_effectiveness: float,
    ) -> str:
        """Assess overall testing risk level."""
        thresholds = self.config["quality_thresholds"]

        scores = [mutation_score, property_coverage, code_coverage, test_effectiveness]
        targets = [
            thresholds["mutation_score_target"],
            thresholds["property_coverage_target"],
            thresholds["code_coverage_target"],
            thresholds["overall_effectiveness_target"],
        ]

        # Calculate how many metrics meet targets
        meets_target = sum(
            1 for score, target in zip(scores, targets) if score >= target
        )

        if meets_target >= 3:
            return "LOW"
        elif meets_target >= 2:
            return "MEDIUM"
        elif meets_target >= 1:
            return "HIGH"
        else:
            return "CRITICAL"

    def _generate_recommendations(
        self,
        mutation_score: float,
        property_coverage: float,
        code_coverage: float,
        test_effectiveness: float,
    ) -> List[str]:
        """Generate specific recommendations for improving test quality."""
        recommendations = []
        thresholds = self.config["quality_thresholds"]

        if mutation_score < thresholds["mutation_score_target"]:
            recommendations.append(
                f"Improve mutation score (current: {mutation_score:.1f}%, target: {thresholds['mutation_score_target']:.1f}%). "
                "Add tests for edge cases and error conditions."
            )

        if property_coverage < thresholds["property_coverage_target"]:
            recommendations.append(
                f"Increase property test coverage (current: {property_coverage:.1f}%, target: {thresholds['property_coverage_target']:.1f}%). "
                "Add property-based tests for mathematical invariants and business rules."
            )

        if code_coverage < thresholds["code_coverage_target"]:
            recommendations.append(
                f"Improve code coverage (current: {code_coverage:.1f}%, target: {thresholds['code_coverage_target']:.1f}%). "
                "Add unit tests for uncovered code paths."
            )

        if test_effectiveness < thresholds["overall_effectiveness_target"]:
            recommendations.append(
                f"Overall test effectiveness is low (current: {test_effectiveness:.1f}%, target: {thresholds['overall_effectiveness_target']:.1f}%). "
                "Focus on improving test quality across all dimensions."
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Test quality is excellent! Consider maintaining this level and reviewing periodically."
            )

        return recommendations

    def _perform_detailed_analysis(
        self,
        mutation_results: Optional[MutationTestSummary],
        property_results: Optional[PropertyTestSummary],
        traditional_results: Dict,
    ) -> Dict:
        """Perform detailed analysis of test results."""
        analysis = {
            "test_gaps": [],
            "high_risk_areas": [],
            "performance_metrics": {},
            "trends": {},
            "actionable_insights": [],
        }

        # Analyze mutation testing results
        if mutation_results:
            survived_mutations = [
                r for r in mutation_results.results if r.test_passed and r.error is None
            ]
            analysis["test_gaps"] = [
                {
                    "type": "mutation_survived",
                    "location": r.location,
                    "operator": r.operator,
                    "description": f"Mutation survived: {r.operator} at {r.location}",
                }
                for r in survived_mutations[:10]  # Top 10 gaps
            ]

            # Identify high-risk areas (files with many survived mutations)
            location_counts = {}
            for result in survived_mutations:
                file_location = (
                    result.location.split(":")[0]
                    if ":" in result.location
                    else result.location
                )
                location_counts[file_location] = (
                    location_counts.get(file_location, 0) + 1
                )

            analysis["high_risk_areas"] = [
                {"file": file, "survived_mutations": count}
                for file, count in sorted(
                    location_counts.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ]

        # Analyze property testing results
        if property_results:
            failed_properties = [
                r for r in property_results.results if not r.test_passed or r.error
            ]
            analysis["test_gaps"].extend(
                [
                    {
                        "type": "property_failed",
                        "function": r.function_name,
                        "property": r.property_name,
                        "description": f"Property test failed: {r.property_name} for {r.function_name}",
                    }
                    for r in failed_properties[:5]  # Top 5 failures
                ]
            )

        # Performance metrics
        analysis["performance_metrics"] = {
            "mutation_testing_time": (
                mutation_results.execution_time if mutation_results else 0
            ),
            "property_testing_time": (
                property_results.execution_time if property_results else 0
            ),
            "traditional_testing_time": traditional_results.get("duration", 0),
            "total_examples_generated": (
                (mutation_results.total_mutations if mutation_results else 0)
                + (property_results.total_examples if property_results else 0)
            ),
        }

        # Generate actionable insights
        analysis["actionable_insights"] = self._generate_actionable_insights(
            mutation_results, property_results, traditional_results
        )

        return analysis

    def _generate_actionable_insights(
        self, mutation_results, property_results, traditional_results
    ) -> List[str]:
        """Generate specific actionable insights from test results."""
        insights = []

        if mutation_results:
            if mutation_results.survived_mutations > 0:
                insights.append(
                    f"Focus on testing edge cases: {mutation_results.survived_mutations} mutations survived, "
                    "indicating potential gaps in test scenarios."
                )

            if (
                mutation_results.failed_mutations
                > mutation_results.total_mutations * 0.1
            ):
                insights.append(
                    "High mutation failure rate suggests code complexity issues. "
                    "Consider refactoring complex functions."
                )

        if property_results:
            if property_results.error_properties > 0:
                insights.append(
                    f"{property_results.error_properties} property tests had errors. "
                    "Review function signatures and type annotations."
                )

        if traditional_results.get("coverage", 0) < 0.8:
            insights.append(
                "Add unit tests for uncovered code paths to improve overall coverage."
            )

        return insights

    def _calculate_default_metrics(self) -> TestQualityMetrics:
        """Calculate default metrics when testing fails."""
        return TestQualityMetrics(
            mutation_score=0.0,
            property_coverage=0.0,
            code_coverage=0.0,
            test_effectiveness=0.0,
            risk_assessment="CRITICAL",
            recommendations=[
                "Fix test execution issues before proceeding with quality assessment."
            ],
        )

    def _evaluate_overall_success(self, metrics: TestQualityMetrics) -> bool:
        """Evaluate if overall testing meets success criteria."""
        thresholds = self.config["quality_thresholds"]

        return (
            metrics.mutation_score >= thresholds["mutation_score_target"]
            and metrics.property_coverage >= thresholds["property_coverage_target"]
            and metrics.code_coverage >= thresholds["code_coverage_target"]
            and metrics.test_effectiveness >= thresholds["overall_effectiveness_target"]
        )

    def save_comprehensive_report(self, result: AdvancedTestResult, output_file: Path):
        """Save comprehensive testing report."""
        report = {
            "summary": asdict(result),
            "timestamp": time.time(),
            "config": self.config,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Comprehensive report saved to {output_file}")

    def print_comprehensive_summary(self, result: AdvancedTestResult):
        """Print comprehensive testing summary."""
        print(f"\n=== Comprehensive Advanced Testing Summary ===")
        print(f"Overall Success: {'PASS' if result.success else 'FAIL'}")
        print(f"Total Duration: {result.duration:.2f}s")
        print(f"Risk Assessment: {result.quality_metrics.risk_assessment}")

        print(f"\n=== Quality Metrics ===")
        print(f"Mutation Score: {result.quality_metrics.mutation_score:.1f}%")
        print(f"Property Coverage: {result.quality_metrics.property_coverage:.1f}%")
        print(f"Code Coverage: {result.quality_metrics.code_coverage:.1f}%")
        print(f"Test Effectiveness: {result.quality_metrics.test_effectiveness:.1f}%")

        print(f"\n=== Traditional Tests ===")
        trad = result.traditional_test_results
        print(
            f"Tests: {trad.get('test_count', 0)}, "
            f"Failures: {trad.get('failures', 0)}, "
            f"Errors: {trad.get('errors', 0)}"
        )

        if result.mutation_results:
            print(f"\n=== Mutation Testing ===")
            mut = result.mutation_results
            print(f"Total Mutations: {mut.total_mutations}")
            print(f"Killed: {mut.killed_mutations}, Survived: {mut.survived_mutations}")
            print(f"Score: {mut.mutation_score:.1f}%")

        if result.property_results:
            print(f"\n=== Property Testing ===")
            prop = result.property_results
            print(f"Properties Tested: {prop.total_properties}")
            print(f"Passed: {prop.passed_properties}, Failed: {prop.failed_properties}")
            print(f"Examples Generated: {prop.total_examples}")

        print(f"\n=== Recommendations ===")
        for i, rec in enumerate(result.quality_metrics.recommendations, 1):
            print(f"{i}. {rec}")

        if result.detailed_analysis.get("high_risk_areas"):
            print(f"\n=== High Risk Areas ===")
            for area in result.detailed_analysis["high_risk_areas"][:3]:
                print(
                    f"  • {area['file']}: {area['survived_mutations']} survived mutations"
                )


def main():
    """Main entry point for comprehensive advanced testing."""
    parser = argparse.ArgumentParser(description="Advanced Testing Orchestrator")
    parser.add_argument("--config", type=Path, help="Configuration file (JSON)")
    parser.add_argument("--target-files", nargs="+", help="Specific files to test")
    parser.add_argument("--source-dir", default="src/pynomaly", help="Source directory")
    parser.add_argument("--test-dir", default="tests", help="Test directory")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument(
        "--max-mutations", type=int, default=100, help="Max mutations to test"
    )
    parser.add_argument(
        "--max-examples", type=int, default=100, help="Max examples per property"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=True, help="Run tests in parallel"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = {}
    if args.config and args.config.exists():
        with open(args.config, "r") as f:
            config = json.load(f)

    # Override with command line arguments
    config.update(
        {
            "source_dir": args.source_dir,
            "test_dir": args.test_dir,
            "output_dir": args.output_dir,
            "parallel_execution": args.parallel,
            "mutation_config": {"max_mutations": args.max_mutations},
            "property_config": {"max_examples": args.max_examples},
        }
    )

    # Initialize orchestrator
    orchestrator = AdvancedTestingOrchestrator(config)

    try:
        # Run comprehensive testing
        result = orchestrator.run_comprehensive_testing(target_files=args.target_files)

        # Print summary
        orchestrator.print_comprehensive_summary(result)

        # Save comprehensive report
        report_file = Path(args.output_dir) / "comprehensive_testing_report.json"
        orchestrator.save_comprehensive_report(result, report_file)

        # Exit with appropriate code
        sys.exit(0 if result.success else 1)

    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Advanced testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
