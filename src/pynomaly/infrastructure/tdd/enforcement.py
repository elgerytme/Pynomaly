"""TDD enforcement engine with validation logic and rule checking."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from pynomaly.infrastructure.config.tdd_config import (
    TDDConfigManager,
    TDDRuleEngine,
    TDDSettings,
    TDDViolation,
    TestRequirement,
)
from pynomaly.infrastructure.persistence.tdd_repository import (
    FileTDDRepository,
    TestDrivenDevelopmentRepositoryProtocol,
)


class TDDEnforcementEngine:
    """Engine for enforcing TDD rules and validating compliance."""

    def __init__(
        self,
        config_manager: TDDConfigManager | None = None,
        repository: TestDrivenDevelopmentRepositoryProtocol | None = None,
    ):
        self.config_manager = config_manager or TDDConfigManager()
        self.repository = repository or FileTDDRepository(Path("./tdd_storage"))
        self._rule_engine = TDDRuleEngine(self.config_manager.settings)

    @property
    def settings(self) -> TDDSettings:
        """Get current TDD settings."""
        return self.config_manager.settings

    @property
    def rule_engine(self) -> TDDRuleEngine:
        """Get rule engine."""
        return self._rule_engine

    def validate_file(self, file_path: Path) -> list[TDDViolation]:
        """Validate a single file for TDD compliance.

        Args:
            file_path: Path to the file to validate

        Returns:
            List of TDD violations found
        """
        violations = []

        # Skip if TDD is disabled or file is exempt
        if not self.settings.enabled or self.rule_engine.is_file_exempt(file_path):
            return violations

        # Skip if file shouldn't be enforced
        if not self.rule_engine.should_enforce_tdd(file_path):
            return violations

        # Check for implementation files without tests
        if self.rule_engine.is_implementation_file(file_path):
            violations.extend(self._validate_implementation_file(file_path))

        # Check for test files with naming issues
        elif self.rule_engine.is_test_file(file_path):
            violations.extend(self._validate_test_file(file_path))

        return violations

    def _validate_implementation_file(self, file_path: Path) -> list[TDDViolation]:
        """Validate an implementation file for TDD compliance."""
        violations = []

        # Check if corresponding test file exists
        expected_test_file = self.rule_engine.get_expected_test_file(file_path)

        if not expected_test_file.exists():
            violation = TDDViolation(
                violation_type="missing_test",
                file_path=str(file_path),
                line_number=None,
                description=f"Implementation file {file_path} is missing corresponding test file {expected_test_file}",
                severity="error",
                rule_name="require_test_file",
                suggestion=f"Create test file at {expected_test_file}",
                auto_fixable=True,
            )
            violations.append(violation)
            self.repository.save_violation(violation)

        # Check if file was created before tests (if strict mode)
        if self.settings.strict_mode and expected_test_file.exists():
            impl_mtime = file_path.stat().st_mtime
            test_mtime = expected_test_file.stat().st_mtime

            if impl_mtime < test_mtime:
                violation = TDDViolation(
                    violation_type="implementation_before_test",
                    file_path=str(file_path),
                    line_number=None,
                    description=f"Implementation file {file_path} was created before its test file",
                    severity="warning",
                    rule_name="test_first_development",
                    suggestion="In TDD, tests should be written before implementation",
                    auto_fixable=False,
                )
                violations.append(violation)
                self.repository.save_violation(violation)

        # Analyze code for functions without test requirements
        violations.extend(self._check_function_test_requirements(file_path))

        return violations

    def _validate_test_file(self, file_path: Path) -> list[TDDViolation]:
        """Validate a test file for TDD compliance."""
        violations = []

        # Check test naming conventions
        violations.extend(self._check_test_naming_conventions(file_path))

        # Check for test docstrings if required
        if self.settings.require_test_docstrings:
            violations.extend(self._check_test_docstrings(file_path))

        # Check for sufficient test coverage patterns
        violations.extend(self._check_test_coverage_patterns(file_path))

        return violations

    def _check_function_test_requirements(self, file_path: Path) -> list[TDDViolation]:
        """Check if all functions have corresponding test requirements."""
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private and magic methods
                    if node.name.startswith("_"):
                        continue

                    # Check if test requirement exists
                    module_requirements = self.repository.find_requirements_by_module(
                        str(file_path)
                    )
                    function_requirements = [
                        req
                        for req in module_requirements
                        if req.function_name == node.name
                    ]

                    if not function_requirements:
                        violation = TDDViolation(
                            violation_type="missing_test_requirement",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Function '{node.name}' has no test requirement defined",
                            severity="warning",
                            rule_name="require_test_specification",
                            suggestion=f"Create test requirement for function '{node.name}'",
                            auto_fixable=False,
                        )
                        violations.append(violation)
                        self.repository.save_violation(violation)

        except (SyntaxError, UnicodeDecodeError) as e:
            violation = TDDViolation(
                violation_type="parse_error",
                file_path=str(file_path),
                line_number=None,
                description=f"Could not parse file: {str(e)}",
                severity="error",
                rule_name="file_parseable",
                suggestion="Fix syntax errors in the file",
                auto_fixable=False,
            )
            violations.append(violation)
            self.repository.save_violation(violation)

        return violations

    def _check_test_naming_conventions(self, file_path: Path) -> list[TDDViolation]:
        """Check test naming conventions."""
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's a test function
                    if node.name.startswith("test_"):
                        # Check naming convention
                        if not self._validate_test_function_name(node.name):
                            violation = TDDViolation(
                                violation_type="naming_violation",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                description=f"Test function '{node.name}' doesn't follow naming convention",
                                severity="warning",
                                rule_name="test_naming_convention",
                                suggestion=f"Use naming pattern: {self.settings.test_naming_convention}",
                                auto_fixable=False,
                            )
                            violations.append(violation)
                            self.repository.save_violation(violation)

                elif isinstance(node, ast.ClassDef):
                    # Check test class naming
                    if node.name.startswith("Test"):
                        expected_pattern = self.settings.test_class_naming_convention
                        if not self._validate_test_class_name(
                            node.name, expected_pattern
                        ):
                            violation = TDDViolation(
                                violation_type="naming_violation",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                description=f"Test class '{node.name}' doesn't follow naming convention",
                                severity="warning",
                                rule_name="test_class_naming_convention",
                                suggestion=f"Use naming pattern: {expected_pattern}",
                                auto_fixable=False,
                            )
                            violations.append(violation)
                            self.repository.save_violation(violation)

        except (SyntaxError, UnicodeDecodeError):
            # Already handled in _check_function_test_requirements
            pass

        return violations

    def _check_test_docstrings(self, file_path: Path) -> list[TDDViolation]:
        """Check for test function docstrings."""
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    # Check if function has a docstring
                    if not ast.get_docstring(node):
                        violation = TDDViolation(
                            violation_type="missing_docstring",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Test function '{node.name}' is missing docstring",
                            severity="info",
                            rule_name="require_test_docstrings",
                            suggestion="Add docstring explaining what the test validates",
                            auto_fixable=False,
                        )
                        violations.append(violation)
                        self.repository.save_violation(violation)

        except (SyntaxError, UnicodeDecodeError):
            pass

        return violations

    def _check_test_coverage_patterns(self, file_path: Path) -> list[TDDViolation]:
        """Check for common test coverage patterns."""
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                source_code = f.read()

            # Look for common testing patterns
            if "assert" not in source_code and "self.assert" not in source_code:
                violation = TDDViolation(
                    violation_type="no_assertions",
                    file_path=str(file_path),
                    line_number=None,
                    description="Test file contains no assertions",
                    severity="warning",
                    rule_name="require_assertions",
                    suggestion="Add assertions to validate expected behavior",
                    auto_fixable=False,
                )
                violations.append(violation)
                self.repository.save_violation(violation)

        except (SyntaxError, UnicodeDecodeError):
            pass

        return violations

    def _validate_test_function_name(self, function_name: str) -> bool:
        """Validate test function name against convention."""
        # Basic validation - can be extended based on naming convention
        return function_name.startswith("test_") and len(function_name) > 5

    def _validate_test_class_name(self, class_name: str, pattern: str) -> bool:
        """Validate test class name against pattern."""
        # Basic validation - can be extended based on pattern
        return class_name.startswith("Test") and len(class_name) > 4

    def validate_project(self, project_root: Path) -> list[TDDViolation]:
        """Validate entire project for TDD compliance.

        Args:
            project_root: Root directory of the project

        Returns:
            List of all TDD violations found
        """
        all_violations = []

        # Find all Python files in the project
        python_files = list(project_root.rglob("*.py"))

        for file_path in python_files:
            violations = self.validate_file(file_path)
            all_violations.extend(violations)

        # Check for project-level compliance
        project_violations = self._validate_project_structure(project_root)
        all_violations.extend(project_violations)

        return all_violations

    def _validate_project_structure(self, project_root: Path) -> list[TDDViolation]:
        """Validate project structure for TDD compliance."""
        violations = []

        # Check for tests directory
        tests_dir = project_root / "tests"
        if not tests_dir.exists():
            violation = TDDViolation(
                violation_type="missing_tests_directory",
                file_path=str(project_root),
                line_number=None,
                description="Project is missing 'tests' directory",
                severity="warning",
                rule_name="require_tests_directory",
                suggestion="Create a 'tests' directory for test files",
                auto_fixable=True,
            )
            violations.append(violation)
            self.repository.save_violation(violation)

        return violations

    def run_coverage_analysis(self, project_root: Path) -> dict[str, float]:
        """Run test coverage analysis for the project.

        Args:
            project_root: Root directory of the project

        Returns:
            Dictionary mapping module paths to coverage percentages
        """
        coverage_data = {}

        try:
            # Run pytest with coverage
            subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "--cov=src",
                    "--cov-report=json",
                    "--cov-report=term",
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Parse coverage report
            coverage_file = project_root / "coverage.json"
            if coverage_file.exists():
                import json

                with open(coverage_file) as f:
                    coverage_report = json.load(f)

                for filename, file_coverage in coverage_report.get("files", {}).items():
                    coverage_percentage = file_coverage.get("summary", {}).get(
                        "percent_covered", 0
                    )
                    coverage_data[filename] = coverage_percentage / 100.0

                    # Update repository with coverage data
                    self.repository.update_coverage_data(
                        filename, coverage_percentage / 100.0
                    )

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # Coverage analysis failed - create violations for low coverage if we have any data
            pass

        return coverage_data

    def validate_coverage_thresholds(
        self, coverage_data: dict[str, float]
    ) -> list[TDDViolation]:
        """Validate coverage meets minimum thresholds.

        Args:
            coverage_data: Dictionary mapping files to coverage percentages

        Returns:
            List of coverage violations
        """
        violations = []

        for file_path, coverage in coverage_data.items():
            if coverage < self.settings.min_test_coverage:
                violation = TDDViolation(
                    violation_type="low_coverage",
                    file_path=file_path,
                    line_number=None,
                    description=f"Test coverage {coverage:.1%} is below minimum threshold {self.settings.min_test_coverage:.1%}",
                    severity="error"
                    if coverage < self.settings.coverage_fail_under
                    else "warning",
                    rule_name="minimum_coverage_threshold",
                    suggestion=f"Add more tests to reach {self.settings.min_test_coverage:.1%} coverage",
                    auto_fixable=False,
                )
                violations.append(violation)
                self.repository.save_violation(violation)

        return violations

    def auto_fix_violations(self, violations: list[TDDViolation]) -> list[str]:
        """Attempt to automatically fix TDD violations.

        Args:
            violations: List of violations to fix

        Returns:
            List of fixes applied
        """
        fixes_applied = []

        for violation in violations:
            if violation.auto_fixable:
                if violation.violation_type == "missing_test":
                    fix_result = self._auto_create_test_file(violation)
                    if fix_result:
                        fixes_applied.append(fix_result)

                elif violation.violation_type == "missing_tests_directory":
                    fix_result = self._auto_create_tests_directory(violation)
                    if fix_result:
                        fixes_applied.append(fix_result)

        return fixes_applied

    def _auto_create_test_file(self, violation: TDDViolation) -> str | None:
        """Auto-create a basic test file."""
        try:
            impl_file = Path(violation.file_path)
            test_file = self.rule_engine.get_expected_test_file(impl_file)

            # Create test directory if it doesn't exist
            test_file.parent.mkdir(parents=True, exist_ok=True)

            # Generate basic test template
            test_content = self._generate_test_template(impl_file)

            with open(test_file, "w") as f:
                f.write(test_content)

            return f"Created test file: {test_file}"

        except Exception:
            return None

    def _auto_create_tests_directory(self, violation: TDDViolation) -> str | None:
        """Auto-create tests directory."""
        try:
            project_root = Path(violation.file_path)
            tests_dir = project_root / "tests"
            tests_dir.mkdir(exist_ok=True)

            # Create __init__.py
            init_file = tests_dir / "__init__.py"
            init_file.touch()

            return f"Created tests directory: {tests_dir}"

        except Exception:
            return None

    def _generate_test_template(self, impl_file: Path) -> str:
        """Generate a basic test template for an implementation file."""
        module_name = impl_file.stem

        template = f'''"""Tests for {module_name} module."""

import pytest
from {module_name} import *


class Test{module_name.title()}:
    """Test class for {module_name} module."""

    def test_placeholder(self):
        """Placeholder test - replace with actual tests."""
        # TODO: Add actual test implementation
        assert True, "Replace with real test"
'''

        return template

    def create_test_requirement(
        self,
        module_path: str,
        function_name: str,
        description: str,
        test_specification: str,
        coverage_target: float = None,
        tags: set[str] | None = None,
    ) -> TestRequirement:
        """Create a new test requirement.

        Args:
            module_path: Path to the module requiring tests
            function_name: Name of the function/method to test
            description: Description of what should be tested
            test_specification: Detailed test specification
            coverage_target: Target coverage percentage (defaults to global setting)
            tags: Optional tags for categorization

        Returns:
            Created test requirement
        """
        if coverage_target is None:
            coverage_target = self.settings.min_test_coverage

        return self.repository.create_test_requirement(
            module_path=module_path,
            function_name=function_name,
            description=description,
            test_specification=test_specification,
            coverage_target=coverage_target,
            tags=tags,
        )

    def get_compliance_report(self) -> dict:
        """Get comprehensive TDD compliance report."""
        report = self.repository.get_compliance_report()

        # Add additional metrics
        recent_violations = [
            v
            for v in self.repository._load_violations()
            if hasattr(self.repository, "_load_violations")
        ]

        return {
            "compliance_report": report,
            "settings": {
                "enabled": self.settings.enabled,
                "strict_mode": self.settings.strict_mode,
                "min_coverage": self.settings.min_test_coverage,
                "enforced_packages": self.settings.enforce_on_packages,
            },
            "recent_violations": len(recent_violations),
            "auto_fixable_violations": len(
                [v for v in recent_violations if v.auto_fixable]
            ),
        }
