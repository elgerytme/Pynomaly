#!/usr/bin/env python3
"""
Buck2 Impact Analysis Tool for Pynomaly
Analyzes the impact of changes on the codebase and determines optimal test strategy.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
import argparse
import logging
from collections import defaultdict, Counter

# import networkx as nx  # Not currently used

# Import our existing tools
from buck2_change_detector import Buck2ChangeDetector, ChangeAnalysis
from buck2_incremental_test import Buck2IncrementalTestRunner, TestRunSummary
from buck2_git_integration import Buck2GitIntegration, CommitInfo, BranchInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ImpactRisk:
    """Risk assessment for a change."""

    level: str  # "low", "medium", "high", "critical"
    score: float  # 0-1
    reasons: List[str]
    affected_components: Set[str]


@dataclass
class TestStrategy:
    """Recommended testing strategy based on impact analysis."""

    priority: str  # "minimal", "standard", "comprehensive", "full"
    test_targets: Set[str]
    build_targets: Set[str]
    estimated_duration: float
    risk_factors: List[str]
    recommendations: List[str]


@dataclass
class ComponentMetrics:
    """Metrics for a component/module."""

    lines_of_code: int
    complexity: float
    test_coverage: float
    change_frequency: int
    bug_frequency: int
    dependencies: Set[str]
    dependents: Set[str]


@dataclass
class ImpactAnalysisResult:
    """Complete impact analysis result."""

    change_analysis: ChangeAnalysis
    risk_assessment: ImpactRisk
    test_strategy: TestStrategy
    component_metrics: Dict[str, ComponentMetrics]
    analysis_metadata: Dict


class Buck2ImpactAnalyzer:
    """Analyzes the impact of changes and recommends testing strategies."""

    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path.cwd()
        self.change_detector = Buck2ChangeDetector(repo_root)
        self.test_runner = Buck2IncrementalTestRunner(repo_root)
        self.git_integration = Buck2GitIntegration(repo_root)

        # Component categorization
        self.component_categories = {
            "critical": {
                "src/pynomaly/domain/entities/",
                "src/pynomaly/domain/services/",
                "src/pynomaly/application/use_cases/",
                "src/pynomaly/infrastructure/adapters/",
            },
            "important": {
                "src/pynomaly/application/services/",
                "src/pynomaly/infrastructure/persistence/",
                "src/pynomaly/presentation/api/",
                "src/pynomaly/presentation/cli/",
            },
            "standard": {
                "src/pynomaly/shared/",
                "src/pynomaly/infrastructure/config/",
                "src/pynomaly/presentation/web/",
            },
            "low_risk": {
                "docs/",
                "examples/",
                "scripts/",
                "tests/",
            },
        }

        # Risk multipliers for different file types
        self.risk_multipliers = {
            ".py": 1.0,
            ".yaml": 0.3,
            ".yml": 0.3,
            ".json": 0.2,
            ".md": 0.1,
            ".txt": 0.1,
            ".toml": 0.4,
            ".cfg": 0.3,
        }

    def calculate_component_metrics(self, file_path: str) -> ComponentMetrics:
        """Calculate metrics for a component based on file path."""
        try:
            # Get lines of code
            loc = 0
            complexity = 0.0

            if Path(file_path).exists():
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    loc = len(
                        [
                            line
                            for line in lines
                            if line.strip() and not line.strip().startswith("#")
                        ]
                    )

                    # Simple complexity estimation
                    complexity_keywords = [
                        "if",
                        "elif",
                        "else",
                        "for",
                        "while",
                        "try",
                        "except",
                        "with",
                        "def",
                        "class",
                    ]
                    for line in lines:
                        for keyword in complexity_keywords:
                            if keyword in line:
                                complexity += 0.5

            # Get change frequency from git
            change_frequency = self._get_change_frequency(file_path)

            # Estimate test coverage (simplified)
            test_coverage = self._estimate_test_coverage(file_path)

            # Get dependencies
            dependencies, dependents = self._analyze_dependencies(file_path)

            return ComponentMetrics(
                lines_of_code=loc,
                complexity=complexity,
                test_coverage=test_coverage,
                change_frequency=change_frequency,
                bug_frequency=0,  # Would need bug tracking integration
                dependencies=dependencies,
                dependents=dependents,
            )
        except Exception as e:
            logger.warning(f"Failed to calculate metrics for {file_path}: {e}")
            return ComponentMetrics(
                lines_of_code=0,
                complexity=0.0,
                test_coverage=0.0,
                change_frequency=0,
                bug_frequency=0,
                dependencies=set(),
                dependents=set(),
            )

    def _get_change_frequency(self, file_path: str) -> int:
        """Get the frequency of changes to a file from git history."""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--", file_path],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )
            return (
                len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            )
        except subprocess.CalledProcessError:
            return 0

    def _estimate_test_coverage(self, file_path: str) -> float:
        """Estimate test coverage for a file (simplified approach)."""
        # Look for corresponding test files
        test_patterns = [
            file_path.replace("src/", "tests/").replace(".py", "_test.py"),
            file_path.replace("src/", "tests/test_").replace(".py", ".py"),
            file_path.replace("src/pynomaly/", "tests/").replace(".py", "_test.py"),
        ]

        for test_pattern in test_patterns:
            if Path(test_pattern).exists():
                # Simple heuristic: if test file exists, assume 70% coverage
                return 0.7

        return 0.0  # No test file found

    def _analyze_dependencies(self, file_path: str) -> Tuple[Set[str], Set[str]]:
        """Analyze dependencies and dependents for a file."""
        dependencies = set()
        dependents = set()

        if not Path(file_path).exists() or not file_path.endswith(".py"):
            return dependencies, dependents

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

                # Find imports
                import_lines = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip().startswith(("import ", "from "))
                ]

                for line in import_lines:
                    if "pynomaly" in line:
                        # Extract pynomaly modules
                        if line.startswith("from pynomaly"):
                            module = line.split()[1].split(".")[0]
                            dependencies.add(f"pynomaly.{module}")
                        elif line.startswith("import pynomaly"):
                            module = (
                                line.split()[1].split(".")[1] if "." in line else "core"
                            )
                            dependencies.add(f"pynomaly.{module}")
        except Exception as e:
            logger.debug(f"Failed to analyze dependencies for {file_path}: {e}")

        return dependencies, dependents

    def assess_risk(self, change_analysis: ChangeAnalysis) -> ImpactRisk:
        """Assess the risk level of changes."""
        risk_score = 0.0
        risk_reasons = []
        affected_components = set()

        for file_path in change_analysis.changed_files:
            # Determine component category
            component_risk = 0.0
            for category, patterns in self.component_categories.items():
                if any(file_path.startswith(pattern) for pattern in patterns):
                    if category == "critical":
                        component_risk = 0.8
                        risk_reasons.append(f"Critical component modified: {file_path}")
                    elif category == "important":
                        component_risk = 0.6
                        risk_reasons.append(
                            f"Important component modified: {file_path}"
                        )
                    elif category == "standard":
                        component_risk = 0.4
                    else:  # low_risk
                        component_risk = 0.1
                    break
            else:
                # Unknown component - assume medium risk
                component_risk = 0.5
                risk_reasons.append(f"Unknown component modified: {file_path}")

            # Apply file type multiplier
            file_ext = Path(file_path).suffix
            multiplier = self.risk_multipliers.get(file_ext, 0.5)
            component_risk *= multiplier

            # Get component metrics
            metrics = self.calculate_component_metrics(file_path)

            # Adjust risk based on metrics
            if metrics.complexity > 10:
                component_risk *= 1.3
                risk_reasons.append(f"High complexity file: {file_path}")

            if metrics.test_coverage < 0.5:
                component_risk *= 1.2
                risk_reasons.append(f"Low test coverage: {file_path}")

            if metrics.change_frequency > 10:
                component_risk *= 1.1
                risk_reasons.append(f"Frequently changed file: {file_path}")

            risk_score += component_risk
            affected_components.add(file_path)

        # Normalize risk score
        if change_analysis.changed_files:
            risk_score /= len(change_analysis.changed_files)

        # Additional risk factors
        if len(change_analysis.affected_targets) > 10:
            risk_score *= 1.2
            risk_reasons.append("Many targets affected")

        if len(change_analysis.changed_files) > 20:
            risk_score *= 1.1
            risk_reasons.append("Many files changed")

        # Determine risk level
        if risk_score >= 0.8:
            risk_level = "critical"
        elif risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        return ImpactRisk(
            level=risk_level,
            score=min(risk_score, 1.0),
            reasons=risk_reasons,
            affected_components=affected_components,
        )

    def recommend_test_strategy(
        self, change_analysis: ChangeAnalysis, risk: ImpactRisk
    ) -> TestStrategy:
        """Recommend testing strategy based on change analysis and risk."""
        test_targets = set()
        build_targets = set()
        recommendations = []
        risk_factors = []

        # Base strategy on risk level
        if risk.level == "critical":
            priority = "full"
            test_targets = {
                "test-all",
                "benchmarks",
                "property-tests",
                "security-tests",
                "mutation-tests",
            }
            build_targets = {"build-all"}
            estimated_duration = 1800  # 30 minutes
            recommendations.extend(
                [
                    "Run full test suite due to critical changes",
                    "Consider manual testing of critical paths",
                    "Review changes with team before deployment",
                    "Run performance benchmarks",
                ]
            )

        elif risk.level == "high":
            priority = "comprehensive"
            test_targets = change_analysis.test_targets | {
                "test-integration",
                "security-tests",
            }
            build_targets = change_analysis.build_targets
            estimated_duration = 900  # 15 minutes
            recommendations.extend(
                [
                    "Run comprehensive tests including integration tests",
                    "Review security implications",
                    "Consider running benchmarks",
                ]
            )

        elif risk.level == "medium":
            priority = "standard"
            test_targets = change_analysis.test_targets
            build_targets = change_analysis.build_targets
            estimated_duration = 300  # 5 minutes
            recommendations.extend(
                [
                    "Run standard incremental tests",
                    "Monitor for any unexpected failures",
                ]
            )

        else:  # low risk
            priority = "minimal"
            # Only run tests for directly affected components
            test_targets = {
                t
                for t in change_analysis.test_targets
                if any(
                    f.startswith(("docs/", "examples/", "scripts/"))
                    for f in change_analysis.changed_files
                )
            }
            build_targets = set()
            estimated_duration = 120  # 2 minutes
            recommendations.extend(
                [
                    "Minimal testing sufficient for low-risk changes",
                    "Consider running quick smoke tests",
                ]
            )

        # Additional recommendations based on specific changes
        if any(f.endswith(".py") for f in change_analysis.changed_files):
            recommendations.append(
                "Python code changes detected - ensure lint/type checks pass"
            )

        if any("api" in f for f in change_analysis.changed_files):
            recommendations.append("API changes detected - run API contract tests")
            test_targets.add("test-presentation")

        if any(
            "database" in f.lower() or "persistence" in f
            for f in change_analysis.changed_files
        ):
            recommendations.append("Database changes detected - run integration tests")
            test_targets.add("test-infrastructure")

        # Risk factors
        if risk.score > 0.7:
            risk_factors.append("High risk score - thorough testing recommended")

        if len(change_analysis.changed_files) > 10:
            risk_factors.append("Many files changed - increased complexity")

        return TestStrategy(
            priority=priority,
            test_targets=test_targets,
            build_targets=build_targets,
            estimated_duration=estimated_duration,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )

    def analyze_impact(
        self, base_commit: str = "HEAD~1", target_commit: str = "HEAD"
    ) -> ImpactAnalysisResult:
        """Perform comprehensive impact analysis."""
        logger.info(
            f"Analyzing impact of changes from {base_commit} to {target_commit}"
        )

        # Get change analysis
        change_analysis = self.change_detector.analyze_changes(
            base_commit, target_commit
        )

        # Assess risk
        risk_assessment = self.assess_risk(change_analysis)

        # Calculate component metrics
        component_metrics = {}
        for file_path in change_analysis.changed_files:
            component_metrics[file_path] = self.calculate_component_metrics(file_path)

        # Recommend test strategy
        test_strategy = self.recommend_test_strategy(change_analysis, risk_assessment)

        # Generate metadata
        analysis_metadata = {
            "analysis_timestamp": subprocess.run(
                ["date", "-Iseconds"], capture_output=True, text=True
            ).stdout.strip(),
            "total_files_analyzed": len(change_analysis.changed_files),
            "total_targets_affected": len(change_analysis.affected_targets),
            "risk_level": risk_assessment.level,
            "risk_score": risk_assessment.score,
            "recommended_priority": test_strategy.priority,
            "estimated_test_duration": test_strategy.estimated_duration,
        }

        return ImpactAnalysisResult(
            change_analysis=change_analysis,
            risk_assessment=risk_assessment,
            test_strategy=test_strategy,
            component_metrics=component_metrics,
            analysis_metadata=analysis_metadata,
        )

    def save_analysis(
        self, result: ImpactAnalysisResult, output_file: Path = None
    ) -> Path:
        """Save impact analysis to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = self.repo_root / f"buck2_impact_analysis_{timestamp}.json"

        # Prepare data for JSON serialization
        result_dict = asdict(result)

        # Convert sets to lists
        def convert_sets(obj):
            if isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_sets(item) for item in obj]
            else:
                return obj

        result_dict = convert_sets(result_dict)

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info(f"Impact analysis saved to {output_file}")
        return output_file

    def print_analysis(self, result: ImpactAnalysisResult):
        """Print human-readable impact analysis."""
        print(f"\n=== Buck2 Impact Analysis ===")
        print(f"Commit range: {result.change_analysis.commit_range}")
        print(f"Files changed: {len(result.change_analysis.changed_files)}")
        print(f"Targets affected: {len(result.change_analysis.affected_targets)}")

        print(f"\n=== Risk Assessment ===")
        print(f"Risk level: {result.risk_assessment.level.upper()}")
        print(f"Risk score: {result.risk_assessment.score:.2f}")

        if result.risk_assessment.reasons:
            print("Risk factors:")
            for reason in result.risk_assessment.reasons[:5]:
                print(f"  - {reason}")

        print(f"\n=== Test Strategy ===")
        print(f"Priority: {result.test_strategy.priority}")
        print(
            f"Estimated duration: {result.test_strategy.estimated_duration / 60:.1f} minutes"
        )
        print(f"Test targets: {len(result.test_strategy.test_targets)}")
        print(f"Build targets: {len(result.test_strategy.build_targets)}")

        if result.test_strategy.test_targets:
            print("Tests to run:")
            for target in sorted(result.test_strategy.test_targets):
                print(f"  - {target}")

        if result.test_strategy.recommendations:
            print("Recommendations:")
            for rec in result.test_strategy.recommendations:
                print(f"  - {rec}")


def main():
    """Main entry point for the impact analyzer."""
    parser = argparse.ArgumentParser(description="Buck2 Impact Analysis Tool")
    parser.add_argument("--base", default="HEAD~1", help="Base commit for comparison")
    parser.add_argument("--target", default="HEAD", help="Target commit for comparison")
    parser.add_argument("--output", type=Path, help="Output file for analysis results")
    parser.add_argument(
        "--format", choices=["json", "summary"], default="summary", help="Output format"
    )
    parser.add_argument(
        "--run-tests", action="store_true", help="Run recommended tests after analysis"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize analyzer
    analyzer = Buck2ImpactAnalyzer()

    try:
        # Analyze impact
        result = analyzer.analyze_impact(args.base, args.target)

        # Output results
        if args.format == "json":
            output_file = args.output or Path("buck2_impact_analysis.json")
            analyzer.save_analysis(result, output_file)
            print(f"Analysis saved to: {output_file}")
        else:
            analyzer.print_analysis(result)

        # Run tests if requested
        if args.run_tests:
            print(f"\n=== Running Recommended Tests ===")
            summary = analyzer.test_runner.run_incremental_tests(
                args.base,
                args.target,
                fail_fast=result.risk_assessment.level in ["high", "critical"],
            )
            analyzer.test_runner.print_summary(summary)

            if summary.failed_targets > 0:
                sys.exit(1)

    except Exception as e:
        logger.error(f"Impact analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
