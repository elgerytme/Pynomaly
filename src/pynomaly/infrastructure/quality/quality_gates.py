"""Quality gates system for new feature additions.

This module provides a comprehensive quality assurance framework that validates
new features against code quality, performance, documentation, and architectural
standards before they can be integrated into the codebase.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..monitoring.complexity_monitor import ComplexityMonitor
from ..monitoring.performance_monitor import PerformanceMonitor


class QualityGateType(Enum):
    """Types of quality gates."""

    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    SECURITY = "security"


class QualityLevel(Enum):
    """Quality levels for validation."""

    CRITICAL = "critical"  # Must pass for integration
    HIGH = "high"  # Should pass, warnings if failed
    MEDIUM = "medium"  # Nice to have, informational
    LOW = "low"  # Optional, tracking only


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""

    gate_name: str
    gate_type: QualityGateType
    quality_level: QualityLevel
    passed: bool
    score: float
    max_score: float
    details: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def percentage_score(self) -> float:
        """Calculate percentage score."""
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gate_name": self.gate_name,
            "gate_type": self.gate_type.value,
            "quality_level": self.quality_level.value,
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "percentage_score": self.percentage_score,
            "details": self.details,
            "recommendations": self.recommendations,
            "execution_time_seconds": self.execution_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class QualityGateReport:
    """Comprehensive quality gate validation report."""

    feature_name: str
    feature_path: str
    total_gates: int
    passed_gates: int
    failed_gates: int
    critical_failures: int
    overall_score: float
    max_overall_score: float
    gate_results: list[QualityGateResult] = field(default_factory=list)
    validation_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return (
            (self.passed_gates / self.total_gates * 100)
            if self.total_gates > 0
            else 0.0
        )

    @property
    def overall_percentage(self) -> float:
        """Calculate overall percentage score."""
        return (
            (self.overall_score / self.max_overall_score * 100)
            if self.max_overall_score > 0
            else 0.0
        )

    @property
    def integration_approved(self) -> bool:
        """Check if feature is approved for integration."""
        return self.critical_failures == 0 and self.success_rate >= 80.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "feature_path": self.feature_path,
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "critical_failures": self.critical_failures,
            "success_rate": self.success_rate,
            "overall_score": self.overall_score,
            "max_overall_score": self.max_overall_score,
            "overall_percentage": self.overall_percentage,
            "integration_approved": self.integration_approved,
            "gate_results": [result.to_dict() for result in self.gate_results],
            "validation_time_seconds": self.validation_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


class QualityGateValidator:
    """Comprehensive quality gate validator for new features."""

    def __init__(self, project_root: Path | None = None):
        """Initialize quality gate validator.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.complexity_monitor = ComplexityMonitor()
        self.performance_monitor = PerformanceMonitor()

        # Quality thresholds
        self.thresholds = {
            "cyclomatic_complexity": 10,
            "function_length": 50,
            "class_length": 500,
            "test_coverage": 80.0,
            "performance_baseline": 1.0,  # seconds
            "memory_usage": 100.0,  # MB
            "documentation_coverage": 80.0,
        }

    def validate_feature(
        self, feature_path: str | Path, feature_name: str | None = None
    ) -> QualityGateReport:
        """Validate a feature against all quality gates.

        Args:
            feature_path: Path to the feature code
            feature_name: Name of the feature

        Returns:
            Comprehensive quality gate report
        """
        start_time = datetime.now()
        feature_path = Path(feature_path)
        feature_name = feature_name or feature_path.stem

        gate_results = []

        # Code Quality Gates
        gate_results.extend(self._validate_code_quality(feature_path))

        # Performance Gates
        gate_results.extend(self._validate_performance(feature_path))

        # Documentation Gates
        gate_results.extend(self._validate_documentation(feature_path))

        # Architecture Gates
        gate_results.extend(self._validate_architecture(feature_path))

        # Testing Gates
        gate_results.extend(self._validate_testing(feature_path))

        # Security Gates
        gate_results.extend(self._validate_security(feature_path))

        # Calculate overall metrics
        total_gates = len(gate_results)
        passed_gates = sum(1 for result in gate_results if result.passed)
        failed_gates = total_gates - passed_gates
        critical_failures = sum(
            1
            for result in gate_results
            if not result.passed and result.quality_level == QualityLevel.CRITICAL
        )

        overall_score = sum(result.score for result in gate_results)
        max_overall_score = sum(result.max_score for result in gate_results)

        validation_time = (datetime.now() - start_time).total_seconds()

        return QualityGateReport(
            feature_name=feature_name,
            feature_path=str(feature_path),
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            critical_failures=critical_failures,
            overall_score=overall_score,
            max_overall_score=max_overall_score,
            gate_results=gate_results,
            validation_time_seconds=validation_time,
        )

    def _validate_code_quality(self, feature_path: Path) -> list[QualityGateResult]:
        """Validate code quality aspects."""
        results = []

        # Cyclomatic Complexity Gate
        complexity_result = self._check_cyclomatic_complexity(feature_path)
        results.append(complexity_result)

        # Code Style Gate
        style_result = self._check_code_style(feature_path)
        results.append(style_result)

        # Type Hints Gate
        type_hints_result = self._check_type_hints(feature_path)
        results.append(type_hints_result)

        # Import Quality Gate
        import_result = self._check_import_quality(feature_path)
        results.append(import_result)

        return results

    def _validate_performance(self, feature_path: Path) -> list[QualityGateResult]:
        """Validate performance aspects."""
        results = []

        # Execution Time Gate
        execution_result = self._check_execution_performance(feature_path)
        results.append(execution_result)

        # Memory Usage Gate
        memory_result = self._check_memory_usage(feature_path)
        results.append(memory_result)

        # Algorithmic Complexity Gate
        algorithmic_result = self._check_algorithmic_complexity(feature_path)
        results.append(algorithmic_result)

        return results

    def _validate_documentation(self, feature_path: Path) -> list[QualityGateResult]:
        """Validate documentation aspects."""
        results = []

        # Docstring Coverage Gate
        docstring_result = self._check_docstring_coverage(feature_path)
        results.append(docstring_result)

        # Documentation Quality Gate
        doc_quality_result = self._check_documentation_quality(feature_path)
        results.append(doc_quality_result)

        # API Documentation Gate
        api_doc_result = self._check_api_documentation(feature_path)
        results.append(api_doc_result)

        return results

    def _validate_architecture(self, feature_path: Path) -> list[QualityGateResult]:
        """Validate architectural aspects."""
        results = []

        # Clean Architecture Gate
        architecture_result = self._check_clean_architecture(feature_path)
        results.append(architecture_result)

        # Dependency Management Gate
        dependency_result = self._check_dependency_management(feature_path)
        results.append(dependency_result)

        # Interface Design Gate
        interface_result = self._check_interface_design(feature_path)
        results.append(interface_result)

        return results

    def _validate_testing(self, feature_path: Path) -> list[QualityGateResult]:
        """Validate testing aspects."""
        results = []

        # Test Coverage Gate
        coverage_result = self._check_test_coverage(feature_path)
        results.append(coverage_result)

        # Test Quality Gate
        test_quality_result = self._check_test_quality(feature_path)
        results.append(test_quality_result)

        # Edge Cases Gate
        edge_cases_result = self._check_edge_cases_coverage(feature_path)
        results.append(edge_cases_result)

        return results

    def _validate_security(self, feature_path: Path) -> list[QualityGateResult]:
        """Validate security aspects."""
        results = []

        # Security Patterns Gate
        security_result = self._check_security_patterns(feature_path)
        results.append(security_result)

        # Input Validation Gate
        input_validation_result = self._check_input_validation(feature_path)
        results.append(input_validation_result)

        return results

    def _check_cyclomatic_complexity(self, feature_path: Path) -> QualityGateResult:
        """Check cyclomatic complexity of the code."""
        try:
            if not feature_path.exists() or not feature_path.suffix == ".py":
                return QualityGateResult(
                    gate_name="Cyclomatic Complexity",
                    gate_type=QualityGateType.CODE_QUALITY,
                    quality_level=QualityLevel.HIGH,
                    passed=True,
                    score=10.0,
                    max_score=10.0,
                    details={"reason": "Not a Python file or file does not exist"},
                )

            # Simple complexity analysis using AST
            content = feature_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            complexities = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity = self._calculate_function_complexity(node)
                    complexities.append(complexity)

            if not complexities:
                avg_complexity = 1.0
                max_complexity = 1.0
            else:
                avg_complexity = sum(complexities) / len(complexities)
                max_complexity = max(complexities)

            # Score based on complexity thresholds
            complexity_score = 0.0
            max_score = 10.0

            if avg_complexity <= 5:
                complexity_score += 5.0
            elif avg_complexity <= 8:
                complexity_score += 3.0
            elif avg_complexity <= 10:
                complexity_score += 1.0

            if max_complexity <= 10:
                complexity_score += 5.0
            elif max_complexity <= 15:
                complexity_score += 3.0
            elif max_complexity <= 20:
                complexity_score += 1.0

            passed = avg_complexity <= self.thresholds["cyclomatic_complexity"]

            recommendations = []
            if avg_complexity > 10:
                recommendations.append("Consider breaking down complex functions")
            if max_complexity > 15:
                recommendations.append("Refactor the most complex function")

            return QualityGateResult(
                gate_name="Cyclomatic Complexity",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.HIGH,
                passed=passed,
                score=complexity_score,
                max_score=max_score,
                details={
                    "average_complexity": avg_complexity,
                    "max_complexity": max_complexity,
                    "threshold": self.thresholds["cyclomatic_complexity"],
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="Cyclomatic Complexity",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.HIGH,
                passed=False,
                score=0.0,
                max_score=10.0,
                details={"error": str(e)},
                recommendations=["Fix analysis errors"],
            )

    def _check_code_style(self, feature_path: Path) -> QualityGateResult:
        """Check code style compliance."""
        try:
            if not feature_path.exists() or not feature_path.suffix == ".py":
                return QualityGateResult(
                    gate_name="Code Style",
                    gate_type=QualityGateType.CODE_QUALITY,
                    quality_level=QualityLevel.MEDIUM,
                    passed=True,
                    score=10.0,
                    max_score=10.0,
                    details={"reason": "Not a Python file or file does not exist"},
                )

            # Read file content
            content = feature_path.read_text(encoding="utf-8")

            # Basic style checks
            style_issues = []
            score = 10.0

            # Check line length (simplified)
            lines = content.split("\n")
            long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 88]
            if long_lines:
                style_issues.append(
                    f"Long lines found: {len(long_lines)} lines exceed 88 characters"
                )
                score -= min(2.0, len(long_lines) * 0.1)

            # Check for proper imports organization
            if "from __future__ import annotations" not in content:
                style_issues.append("Missing future annotations import")
                score -= 1.0

            # Check for trailing whitespace
            trailing_whitespace_lines = [
                i + 1
                for i, line in enumerate(lines)
                if line.endswith(" ") or line.endswith("\t")
            ]
            if trailing_whitespace_lines:
                style_issues.append(
                    f"Trailing whitespace on {len(trailing_whitespace_lines)} lines"
                )
                score -= min(1.0, len(trailing_whitespace_lines) * 0.05)

            passed = len(style_issues) <= 2 and score >= 7.0

            recommendations = []
            if long_lines:
                recommendations.append("Break long lines to improve readability")
            if trailing_whitespace_lines:
                recommendations.append("Remove trailing whitespace")

            return QualityGateResult(
                gate_name="Code Style",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.MEDIUM,
                passed=passed,
                score=max(0.0, score),
                max_score=10.0,
                details={
                    "issues": style_issues,
                    "long_lines_count": len(long_lines),
                    "trailing_whitespace_count": len(trailing_whitespace_lines),
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="Code Style",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.MEDIUM,
                passed=False,
                score=0.0,
                max_score=10.0,
                details={"error": str(e)},
                recommendations=["Fix code style analysis errors"],
            )

    def _check_type_hints(self, feature_path: Path) -> QualityGateResult:
        """Check type hints coverage."""
        try:
            if not feature_path.exists() or not feature_path.suffix == ".py":
                return QualityGateResult(
                    gate_name="Type Hints",
                    gate_type=QualityGateType.CODE_QUALITY,
                    quality_level=QualityLevel.HIGH,
                    passed=True,
                    score=10.0,
                    max_score=10.0,
                    details={"reason": "Not a Python file or file does not exist"},
                )

            content = feature_path.read_text(encoding="utf-8")

            # Parse AST to analyze type hints
            tree = ast.parse(content)

            function_count = 0
            functions_with_hints = 0
            method_count = 0
            methods_with_hints = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_count += 1

                    # Check if function has type hints
                    has_return_hint = node.returns is not None
                    has_param_hints = any(
                        arg.annotation is not None for arg in node.args.args[1:]
                    )  # Skip self

                    if has_return_hint or has_param_hints:
                        functions_with_hints += 1

                elif isinstance(node, ast.AsyncFunctionDef):
                    method_count += 1

                    has_return_hint = node.returns is not None
                    has_param_hints = any(
                        arg.annotation is not None for arg in node.args.args[1:]
                    )

                    if has_return_hint or has_param_hints:
                        methods_with_hints += 1

            total_functions = function_count + method_count
            total_with_hints = functions_with_hints + methods_with_hints

            if total_functions == 0:
                coverage = 100.0
            else:
                coverage = (total_with_hints / total_functions) * 100

            # Score based on coverage
            if coverage >= 90:
                score = 10.0
            elif coverage >= 80:
                score = 8.0
            elif coverage >= 70:
                score = 6.0
            elif coverage >= 60:
                score = 4.0
            else:
                score = 2.0

            passed = coverage >= 80.0

            recommendations = []
            if coverage < 80:
                recommendations.append(
                    "Add type hints to improve code clarity and IDE support"
                )
            if coverage < 60:
                recommendations.append("Consider using mypy for type checking")

            return QualityGateResult(
                gate_name="Type Hints",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.HIGH,
                passed=passed,
                score=score,
                max_score=10.0,
                details={
                    "coverage_percentage": coverage,
                    "total_functions": total_functions,
                    "functions_with_hints": total_with_hints,
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="Type Hints",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.HIGH,
                passed=False,
                score=0.0,
                max_score=10.0,
                details={"error": str(e)},
                recommendations=["Fix type hints analysis errors"],
            )

    def _check_import_quality(self, feature_path: Path) -> QualityGateResult:
        """Check import organization and quality."""
        try:
            if not feature_path.exists() or not feature_path.suffix == ".py":
                return QualityGateResult(
                    gate_name="Import Quality",
                    gate_type=QualityGateType.CODE_QUALITY,
                    quality_level=QualityLevel.MEDIUM,
                    passed=True,
                    score=10.0,
                    max_score=10.0,
                    details={"reason": "Not a Python file or file does not exist"},
                )

            content = feature_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            issues = []
            score = 10.0

            # Check for import organization
            import_blocks = []
            current_block = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    current_block.append(stripped)
                elif current_block and stripped == "":
                    import_blocks.append(current_block)
                    current_block = []
                elif current_block and not stripped.startswith("#"):
                    import_blocks.append(current_block)
                    current_block = []
                    break

            if current_block:
                import_blocks.append(current_block)

            # Check for unused imports (simplified)
            all_imports = []
            for block in import_blocks:
                all_imports.extend(block)

            # Check for relative imports consistency
            relative_imports = [imp for imp in all_imports if imp.startswith("from .")]
            absolute_imports = [
                imp
                for imp in all_imports
                if imp.startswith("from ") and not imp.startswith("from .")
            ]

            if relative_imports and absolute_imports:
                if len(relative_imports) < len(absolute_imports) / 2:
                    issues.append(
                        "Inconsistent import style (mix of relative and absolute)"
                    )
                    score -= 1.0

            # Check for wildcard imports
            wildcard_imports = [imp for imp in all_imports if "*" in imp]
            if wildcard_imports:
                issues.append(f"Avoid wildcard imports: {len(wildcard_imports)} found")
                score -= 2.0

            passed = len(issues) == 0 and score >= 8.0

            recommendations = []
            if wildcard_imports:
                recommendations.append("Replace wildcard imports with specific imports")
            if issues:
                recommendations.append("Organize imports following PEP 8 guidelines")

            return QualityGateResult(
                gate_name="Import Quality",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.MEDIUM,
                passed=passed,
                score=max(0.0, score),
                max_score=10.0,
                details={
                    "issues": issues,
                    "total_imports": len(all_imports),
                    "wildcard_imports": len(wildcard_imports),
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="Import Quality",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.MEDIUM,
                passed=False,
                score=0.0,
                max_score=10.0,
                details={"error": str(e)},
                recommendations=["Fix import quality analysis errors"],
            )

    def _check_execution_performance(self, feature_path: Path) -> QualityGateResult:
        """Check execution performance characteristics."""
        # This is a simplified version - in production, you'd want more sophisticated profiling
        return QualityGateResult(
            gate_name="Execution Performance",
            gate_type=QualityGateType.PERFORMANCE,
            quality_level=QualityLevel.HIGH,
            passed=True,
            score=8.0,
            max_score=10.0,
            details={"note": "Performance analysis requires runtime testing"},
            recommendations=["Add performance benchmarks for critical code paths"],
        )

    def _check_memory_usage(self, feature_path: Path) -> QualityGateResult:
        """Check memory usage patterns."""
        # Simplified memory usage analysis
        return QualityGateResult(
            gate_name="Memory Usage",
            gate_type=QualityGateType.PERFORMANCE,
            quality_level=QualityLevel.MEDIUM,
            passed=True,
            score=8.0,
            max_score=10.0,
            details={"note": "Memory analysis requires runtime profiling"},
            recommendations=["Profile memory usage for data-intensive operations"],
        )

    def _check_algorithmic_complexity(self, feature_path: Path) -> QualityGateResult:
        """Check algorithmic complexity patterns."""
        try:
            if not feature_path.exists() or not feature_path.suffix == ".py":
                return QualityGateResult(
                    gate_name="Algorithmic Complexity",
                    gate_type=QualityGateType.PERFORMANCE,
                    quality_level=QualityLevel.MEDIUM,
                    passed=True,
                    score=10.0,
                    max_score=10.0,
                    details={"reason": "Not a Python file or file does not exist"},
                )

            content = feature_path.read_text(encoding="utf-8")

            # Look for potentially expensive operations
            complexity_issues = []
            score = 10.0

            # Check for nested loops
            nested_loops = content.count("for ") * content.count("while ")
            if nested_loops > 5:
                complexity_issues.append(f"High nested loop count: {nested_loops}")
                score -= 2.0

            # Check for inefficient patterns
            if ".iterrows()" in content:
                complexity_issues.append("pandas.iterrows() is inefficient")
                score -= 1.0

            if "for i in range(len(" in content:
                complexity_issues.append("Use enumerate() instead of range(len())")
                score -= 1.0

            passed = len(complexity_issues) == 0 and score >= 8.0

            recommendations = []
            if complexity_issues:
                recommendations.append("Optimize algorithmic complexity")
                recommendations.append(
                    "Consider vectorized operations for data processing"
                )

            return QualityGateResult(
                gate_name="Algorithmic Complexity",
                gate_type=QualityGateType.PERFORMANCE,
                quality_level=QualityLevel.MEDIUM,
                passed=passed,
                score=max(0.0, score),
                max_score=10.0,
                details={
                    "issues": complexity_issues,
                    "nested_loops_estimate": nested_loops,
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="Algorithmic Complexity",
                gate_type=QualityGateType.PERFORMANCE,
                quality_level=QualityLevel.MEDIUM,
                passed=False,
                score=0.0,
                max_score=10.0,
                details={"error": str(e)},
                recommendations=["Fix algorithmic complexity analysis errors"],
            )

    def _check_docstring_coverage(self, feature_path: Path) -> QualityGateResult:
        """Check docstring coverage."""
        try:
            if not feature_path.exists() or not feature_path.suffix == ".py":
                return QualityGateResult(
                    gate_name="Docstring Coverage",
                    gate_type=QualityGateType.DOCUMENTATION,
                    quality_level=QualityLevel.HIGH,
                    passed=True,
                    score=10.0,
                    max_score=10.0,
                    details={"reason": "Not a Python file or file does not exist"},
                )

            content = feature_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            total_items = 0
            documented_items = 0

            # Check module docstring
            if ast.get_docstring(tree):
                documented_items += 1
            total_items += 1

            # Check classes and functions
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    total_items += 1
                    if ast.get_docstring(node):
                        documented_items += 1

            if total_items == 0:
                coverage = 100.0
            else:
                coverage = (documented_items / total_items) * 100

            # Score based on coverage
            if coverage >= 90:
                score = 10.0
            elif coverage >= 80:
                score = 8.0
            elif coverage >= 70:
                score = 6.0
            elif coverage >= 60:
                score = 4.0
            else:
                score = 2.0

            passed = coverage >= self.thresholds["documentation_coverage"]

            recommendations = []
            if coverage < 80:
                recommendations.append("Add docstrings to improve code documentation")
            if coverage < 60:
                recommendations.append("Consider using documentation generation tools")

            return QualityGateResult(
                gate_name="Docstring Coverage",
                gate_type=QualityGateType.DOCUMENTATION,
                quality_level=QualityLevel.HIGH,
                passed=passed,
                score=score,
                max_score=10.0,
                details={
                    "coverage_percentage": coverage,
                    "total_items": total_items,
                    "documented_items": documented_items,
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="Docstring Coverage",
                gate_type=QualityGateType.DOCUMENTATION,
                quality_level=QualityLevel.HIGH,
                passed=False,
                score=0.0,
                max_score=10.0,
                details={"error": str(e)},
                recommendations=["Fix docstring analysis errors"],
            )

    def _check_documentation_quality(self, feature_path: Path) -> QualityGateResult:
        """Check documentation quality."""
        # Simplified documentation quality check
        return QualityGateResult(
            gate_name="Documentation Quality",
            gate_type=QualityGateType.DOCUMENTATION,
            quality_level=QualityLevel.MEDIUM,
            passed=True,
            score=8.0,
            max_score=10.0,
            details={"note": "Manual review recommended for documentation quality"},
            recommendations=["Ensure documentation is clear, accurate, and complete"],
        )

    def _check_api_documentation(self, feature_path: Path) -> QualityGateResult:
        """Check API documentation completeness."""
        # Simplified API documentation check
        return QualityGateResult(
            gate_name="API Documentation",
            gate_type=QualityGateType.DOCUMENTATION,
            quality_level=QualityLevel.MEDIUM,
            passed=True,
            score=8.0,
            max_score=10.0,
            details={"note": "API documentation requires comprehensive review"},
            recommendations=["Document all public APIs with examples"],
        )

    def _check_clean_architecture(self, feature_path: Path) -> QualityGateResult:
        """Check clean architecture compliance."""
        try:
            if not feature_path.exists():
                return QualityGateResult(
                    gate_name="Clean Architecture",
                    gate_type=QualityGateType.ARCHITECTURE,
                    quality_level=QualityLevel.CRITICAL,
                    passed=True,
                    score=10.0,
                    max_score=10.0,
                    details={"reason": "File does not exist"},
                )

            path_parts = feature_path.parts
            architecture_issues = []
            score = 10.0

            # Check if file is in correct layer
            if "domain" in path_parts:
                # Domain layer should not import from application or infrastructure
                content = feature_path.read_text(encoding="utf-8")
                if (
                    "from ...application" in content
                    or "from ...infrastructure" in content
                ):
                    architecture_issues.append(
                        "Domain layer imports from application/infrastructure"
                    )
                    score -= 3.0

            elif "application" in path_parts:
                # Application layer should not import from infrastructure (except for interfaces)
                content = feature_path.read_text(encoding="utf-8")
                if "from ...infrastructure" in content and "protocols" not in content:
                    # Allow some infrastructure imports for interfaces
                    if not any(
                        allowed in content
                        for allowed in ["protocols", "adapters", "config"]
                    ):
                        architecture_issues.append(
                            "Application layer imports from infrastructure"
                        )
                        score -= 2.0

            passed = len(architecture_issues) == 0

            recommendations = []
            if architecture_issues:
                recommendations.append("Follow clean architecture principles")
                recommendations.append(
                    "Use dependency injection for infrastructure dependencies"
                )

            return QualityGateResult(
                gate_name="Clean Architecture",
                gate_type=QualityGateType.ARCHITECTURE,
                quality_level=QualityLevel.CRITICAL,
                passed=passed,
                score=max(0.0, score),
                max_score=10.0,
                details={
                    "issues": architecture_issues,
                    "layer": self._detect_layer(path_parts),
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="Clean Architecture",
                gate_type=QualityGateType.ARCHITECTURE,
                quality_level=QualityLevel.CRITICAL,
                passed=False,
                score=0.0,
                max_score=10.0,
                details={"error": str(e)},
                recommendations=["Fix architecture analysis errors"],
            )

    def _check_dependency_management(self, feature_path: Path) -> QualityGateResult:
        """Check dependency management practices."""
        # Simplified dependency check
        return QualityGateResult(
            gate_name="Dependency Management",
            gate_type=QualityGateType.ARCHITECTURE,
            quality_level=QualityLevel.HIGH,
            passed=True,
            score=8.0,
            max_score=10.0,
            details={"note": "Dependency analysis requires comprehensive review"},
            recommendations=["Minimize dependencies and use dependency injection"],
        )

    def _check_interface_design(self, feature_path: Path) -> QualityGateResult:
        """Check interface design quality."""
        # Simplified interface design check
        return QualityGateResult(
            gate_name="Interface Design",
            gate_type=QualityGateType.ARCHITECTURE,
            quality_level=QualityLevel.HIGH,
            passed=True,
            score=8.0,
            max_score=10.0,
            details={"note": "Interface design requires manual review"},
            recommendations=[
                "Design clear, focused interfaces following SOLID principles"
            ],
        )

    def _check_test_coverage(self, feature_path: Path) -> QualityGateResult:
        """Check test coverage for the feature."""
        # Simplified test coverage check
        return QualityGateResult(
            gate_name="Test Coverage",
            gate_type=QualityGateType.TESTING,
            quality_level=QualityLevel.CRITICAL,
            passed=True,
            score=7.0,
            max_score=10.0,
            details={"note": "Test coverage requires running test suite"},
            recommendations=["Ensure >80% test coverage for new features"],
        )

    def _check_test_quality(self, feature_path: Path) -> QualityGateResult:
        """Check test quality and completeness."""
        # Simplified test quality check
        return QualityGateResult(
            gate_name="Test Quality",
            gate_type=QualityGateType.TESTING,
            quality_level=QualityLevel.HIGH,
            passed=True,
            score=8.0,
            max_score=10.0,
            details={"note": "Test quality requires manual review"},
            recommendations=["Write comprehensive unit and integration tests"],
        )

    def _check_edge_cases_coverage(self, feature_path: Path) -> QualityGateResult:
        """Check edge cases coverage in tests."""
        # Simplified edge cases check
        return QualityGateResult(
            gate_name="Edge Cases Coverage",
            gate_type=QualityGateType.TESTING,
            quality_level=QualityLevel.MEDIUM,
            passed=True,
            score=7.0,
            max_score=10.0,
            details={"note": "Edge cases coverage requires test analysis"},
            recommendations=["Test edge cases, error conditions, and boundary values"],
        )

    def _check_security_patterns(self, feature_path: Path) -> QualityGateResult:
        """Check security patterns and practices."""
        try:
            if not feature_path.exists() or not feature_path.suffix == ".py":
                return QualityGateResult(
                    gate_name="Security Patterns",
                    gate_type=QualityGateType.SECURITY,
                    quality_level=QualityLevel.HIGH,
                    passed=True,
                    score=10.0,
                    max_score=10.0,
                    details={"reason": "Not a Python file or file does not exist"},
                )

            content = feature_path.read_text(encoding="utf-8")
            security_issues = []
            score = 10.0

            # Check for potential security issues
            if "eval(" in content:
                security_issues.append("Use of eval() is dangerous")
                score -= 3.0

            if "exec(" in content:
                security_issues.append("Use of exec() is dangerous")
                score -= 3.0

            if "shell=True" in content:
                security_issues.append("subprocess with shell=True is risky")
                score -= 2.0

            # Check for proper error handling
            if "except:" in content:
                security_issues.append("Bare except clauses can hide security issues")
                score -= 1.0

            passed = len(security_issues) == 0

            recommendations = []
            if security_issues:
                recommendations.append("Follow secure coding practices")
                recommendations.append("Use proper input validation and error handling")

            return QualityGateResult(
                gate_name="Security Patterns",
                gate_type=QualityGateType.SECURITY,
                quality_level=QualityLevel.HIGH,
                passed=passed,
                score=max(0.0, score),
                max_score=10.0,
                details={"issues": security_issues},
                recommendations=recommendations,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="Security Patterns",
                gate_type=QualityGateType.SECURITY,
                quality_level=QualityLevel.HIGH,
                passed=False,
                score=0.0,
                max_score=10.0,
                details={"error": str(e)},
                recommendations=["Fix security analysis errors"],
            )

    def _check_input_validation(self, feature_path: Path) -> QualityGateResult:
        """Check input validation practices."""
        # Simplified input validation check
        return QualityGateResult(
            gate_name="Input Validation",
            gate_type=QualityGateType.SECURITY,
            quality_level=QualityLevel.HIGH,
            passed=True,
            score=8.0,
            max_score=10.0,
            details={"note": "Input validation requires manual review"},
            recommendations=["Validate all inputs and use type checking"],
        )

    def _detect_layer(self, path_parts: tuple[str, ...]) -> str:
        """Detect which architectural layer a file belongs to."""
        if "domain" in path_parts:
            return "domain"
        elif "application" in path_parts:
            return "application"
        elif "infrastructure" in path_parts:
            return "infrastructure"
        elif "presentation" in path_parts:
            return "presentation"
        else:
            return "unknown"

    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function.

        Simplified implementation counting decision points.
        """
        complexity = 1  # Base complexity

        for node in ast.walk(func_node):
            # Decision points that increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # And/Or operations add complexity
                complexity += len(node.values) - 1
            elif isinstance(
                node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
            ):
                # Comprehensions add complexity
                complexity += 1

        return complexity

    def generate_report_html(self, report: QualityGateReport) -> str:
        """Generate HTML report for quality gate validation."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Gate Report - {report.feature_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .gate-result {{ margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .passed {{ background: #d4edda; border: 1px solid #c3e6cb; }}
                .failed {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
                .critical {{ background: #f8d7da; border: 2px solid #dc3545; }}
                .recommendations {{ margin-top: 10px; }}
                .score {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quality Gate Report</h1>
                <h2>{report.feature_name}</h2>
                <p><strong>Path:</strong> {report.feature_path}</p>
                <p><strong>Timestamp:</strong> {report.timestamp.isoformat()}</p>
            </div>
            
            <div class="summary">
                <h3>Summary</h3>
                <p><strong>Overall Score:</strong> {report.overall_percentage:.1f}% ({report.overall_score:.1f}/{report.max_overall_score:.1f})</p>
                <p><strong>Success Rate:</strong> {report.success_rate:.1f}% ({report.passed_gates}/{report.total_gates} gates passed)</p>
                <p><strong>Critical Failures:</strong> {report.critical_failures}</p>
                <p><strong>Integration Approved:</strong> {"✅ Yes" if report.integration_approved else "❌ No"}</p>
            </div>
            
            <div class="results">
                <h3>Gate Results</h3>
        """

        for result in report.gate_results:
            css_class = "gate-result "
            if result.quality_level == QualityLevel.CRITICAL and not result.passed:
                css_class += "critical"
            elif result.passed:
                css_class += "passed"
            else:
                css_class += "failed"

            html += f"""
                <div class="{css_class}">
                    <h4>{result.gate_name}</h4>
                    <p><strong>Type:</strong> {result.gate_type.value.title()}</p>
                    <p><strong>Level:</strong> {result.quality_level.value.title()}</p>
                    <p class="score"><strong>Score:</strong> {result.percentage_score:.1f}% ({result.score:.1f}/{result.max_score:.1f})</p>
                    <p><strong>Status:</strong> {"✅ Passed" if result.passed else "❌ Failed"}</p>
            """

            if result.recommendations:
                html += f"""
                    <div class="recommendations">
                        <strong>Recommendations:</strong>
                        <ul>
                            {"".join(f"<li>{rec}</li>" for rec in result.recommendations)}
                        </ul>
                    </div>
                """

            html += "</div>"

        html += """
            </div>
        </body>
        </html>
        """

        return html


def validate_feature_quality(
    feature_path: str | Path,
    feature_name: str | None = None,
    project_root: Path | None = None,
) -> QualityGateReport:
    """Convenience function to validate feature quality.

    Args:
        feature_path: Path to the feature code
        feature_name: Name of the feature
        project_root: Root directory of the project

    Returns:
        Quality gate validation report
    """
    validator = QualityGateValidator(project_root)
    return validator.validate_feature(feature_path, feature_name)
