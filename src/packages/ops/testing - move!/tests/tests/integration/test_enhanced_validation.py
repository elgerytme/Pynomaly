"""Test enhanced validation system with rich output and GitHub integration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from pynomaly.presentation.cli.validation import (
    EnhancedValidator,
    GitHubCommentGenerator,
    RichOutputFormatter,
    ValidationResult,
    ValidationViolation,
    ViolationSeverity,
)


class TestEnhancedValidator:
    """Test the enhanced validator functionality."""

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = EnhancedValidator()
        assert validator.root_path == Path.cwd()
        assert isinstance(validator.result, ValidationResult)

    def test_validator_with_custom_path(self):
        """Test validator with custom path."""
        custom_path = Path("/tmp/test")
        validator = EnhancedValidator(custom_path)
        assert validator.root_path == custom_path

    def test_structure_validation(self):
        """Test structure validation detects issues."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create forbidden directory
            forbidden_dir = tmp_path / "build"
            forbidden_dir.mkdir()

            validator = EnhancedValidator(tmp_path)
            validator._validate_structure()

            # Should have one violation for forbidden directory
            assert len(validator.result.violations) == 1
            assert validator.result.violations[0].severity == ViolationSeverity.HIGH
            assert "build" in validator.result.violations[0].message

    def test_code_quality_validation(self):
        """Test code quality validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create Python file with issues
            py_file = tmp_path / "test.py"
            py_file.write_text("""
# TODO: Fix this later
def test_function():
    print("Debug message")
    return True
""")

            validator = EnhancedValidator(tmp_path)
            validator._validate_code_quality()

            # Should have violations for TODO and print statement
            assert len(validator.result.violations) >= 2

            # Check for TODO violation
            todo_violations = [
                v for v in validator.result.violations if "TODO" in v.message
            ]
            assert len(todo_violations) == 1
            assert todo_violations[0].severity == ViolationSeverity.LOW

            # Check for print statement violation
            print_violations = [
                v for v in validator.result.violations if "print" in v.message.lower()
            ]
            assert len(print_violations) == 1
            assert print_violations[0].severity == ViolationSeverity.MEDIUM

    def test_security_validation(self):
        """Test security validation detects dangerous patterns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create Python file with security issues
            py_file = tmp_path / "insecure.py"
            py_file.write_text("""
def dangerous_function(user_input):
    result = eval(user_input)
    return result
""")

            validator = EnhancedValidator(tmp_path)
            validator._validate_security()

            # Should have violation for eval usage
            assert len(validator.result.violations) >= 1
            eval_violations = [
                v for v in validator.result.violations if "eval" in v.message
            ]
            assert len(eval_violations) == 1
            assert eval_violations[0].severity == ViolationSeverity.CRITICAL

    def test_skip_file_patterns(self):
        """Test file skipping patterns."""
        validator = EnhancedValidator()

        # Test skip patterns
        skip_files = [
            Path(".venv/test.py"),
            Path("node_modules/test.py"),
            Path("__pycache__/test.py"),
            Path(".git/test.py"),
        ]

        for file_path in skip_files:
            assert validator._should_skip_file(file_path) is True

        # Test non-skip files
        normal_files = [
            Path("src/test.py"),
            Path("tests/test.py"),
            Path("main.py"),
        ]

        for file_path in normal_files:
            assert validator._should_skip_file(file_path) is False


class TestValidationResult:
    """Test validation result functionality."""

    def test_validation_result_initialization(self):
        """Test validation result initializes correctly."""
        result = ValidationResult()
        assert result.violations == []
        assert result.passed is True
        assert result.file_count == 0
        assert result.duration_seconds == 0.0
        assert result.metrics == {}

    def test_add_violation(self):
        """Test adding violations."""
        result = ValidationResult()

        # Add low severity violation - should still pass
        low_violation = ValidationViolation(
            "Low severity issue", ViolationSeverity.LOW, "test.py"
        )
        result.add_violation(low_violation)

        assert len(result.violations) == 1
        assert result.passed is True  # Still passes with low severity

        # Add high severity violation - should fail
        high_violation = ValidationViolation(
            "High severity issue", ViolationSeverity.HIGH, "test.py"
        )
        result.add_violation(high_violation)

        assert len(result.violations) == 2
        assert result.passed is False  # Fails with high severity

    def test_group_by_severity(self):
        """Test grouping violations by severity."""
        result = ValidationResult()

        # Add violations of different severities
        violations = [
            ValidationViolation("Critical", ViolationSeverity.CRITICAL, "test.py"),
            ValidationViolation("High", ViolationSeverity.HIGH, "test.py"),
            ValidationViolation("Medium", ViolationSeverity.MEDIUM, "test.py"),
            ValidationViolation("Low", ViolationSeverity.LOW, "test.py"),
            ValidationViolation(
                "Another Critical", ViolationSeverity.CRITICAL, "test.py"
            ),
        ]

        for violation in violations:
            result.add_violation(violation)

        grouped = result.group_by_severity()

        assert len(grouped[ViolationSeverity.CRITICAL]) == 2
        assert len(grouped[ViolationSeverity.HIGH]) == 1
        assert len(grouped[ViolationSeverity.MEDIUM]) == 1
        assert len(grouped[ViolationSeverity.LOW]) == 1

    def test_errors_and_warnings_properties(self):
        """Test errors and warnings properties."""
        result = ValidationResult()

        # Add violations
        violations = [
            ValidationViolation("Critical", ViolationSeverity.CRITICAL, "test.py"),
            ValidationViolation("High", ViolationSeverity.HIGH, "test.py"),
            ValidationViolation("Medium", ViolationSeverity.MEDIUM, "test.py"),
            ValidationViolation("Low", ViolationSeverity.LOW, "test.py"),
        ]

        for violation in violations:
            result.add_violation(violation)

        # Errors should include critical and high
        assert len(result.errors) == 2
        assert "Critical" in result.errors
        assert "High" in result.errors

        # Warnings should include medium and low
        assert len(result.warnings) == 2
        assert "Medium" in result.warnings
        assert "Low" in result.warnings


class TestRichOutputFormatter:
    """Test rich output formatting."""

    def test_formatter_initialization(self):
        """Test formatter initializes correctly."""
        console = Console()
        formatter = RichOutputFormatter(console)
        assert formatter.console == console

    def test_display_success_results(self):
        """Test displaying successful validation results."""
        console = Console()
        formatter = RichOutputFormatter(console)

        result = ValidationResult()
        result.passed = True

        # This should not raise an exception
        formatter.display_results(result)

    def test_display_failure_results(self):
        """Test displaying failed validation results."""
        console = Console()
        formatter = RichOutputFormatter(console)

        result = ValidationResult()
        violation = ValidationViolation(
            "Test violation",
            ViolationSeverity.HIGH,
            "test.py",
            line_number=10,
            rule_id="TEST_001",
            fix_suggestion="Fix this issue",
        )
        result.add_violation(violation)

        # This should not raise an exception
        formatter.display_results(result)


class TestGitHubCommentGenerator:
    """Test GitHub comment generation."""

    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        generator = GitHubCommentGenerator()
        assert generator.max_violations == 10

    def test_generate_success_comment(self):
        """Test generating success comment."""
        generator = GitHubCommentGenerator()
        result = ValidationResult()
        result.passed = True

        comment = generator.generate_comment(result)

        assert "✅ Validation Passed" in comment
        assert "All validation checks have passed successfully!" in comment
        assert "Ready for merge!" in comment

    def test_generate_failure_comment(self):
        """Test generating failure comment."""
        generator = GitHubCommentGenerator()
        result = ValidationResult()

        # Add violations
        violations = [
            ValidationViolation(
                "Critical issue",
                ViolationSeverity.CRITICAL,
                "test.py",
                line_number=10,
                rule_id="CRIT_001",
                fix_suggestion="Fix critical issue",
            ),
            ValidationViolation(
                "High issue",
                ViolationSeverity.HIGH,
                "test.py",
                line_number=20,
                rule_id="HIGH_001",
                fix_suggestion="Fix high issue",
            ),
        ]

        for violation in violations:
            result.add_violation(violation)

        comment = generator.generate_comment(result)

        assert "❌ Validation Failed" in comment
        assert "Critical issue" in comment
        assert "High issue" in comment
        assert "Fix critical issue" in comment
        assert "Fix high issue" in comment
        assert "pre-commit install" in comment

    def test_generate_comment_with_many_violations(self):
        """Test generating comment with many violations (should truncate)."""
        generator = GitHubCommentGenerator()
        result = ValidationResult()

        # Add more than max violations
        for i in range(15):
            violation = ValidationViolation(
                f"Issue {i}",
                ViolationSeverity.HIGH,
                f"test{i}.py",
                fix_suggestion=f"Fix issue {i}",
            )
            result.add_violation(violation)

        comment = generator.generate_comment(result)

        # Should only show first 10 violations
        assert "Top 10 Violations" in comment
        assert "Issue 0" in comment
        assert "Issue 9" in comment
        assert "Issue 10" not in comment

    @patch("requests.post")
    def test_post_to_github_success(self, mock_post):
        """Test successful GitHub comment posting."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        generator = GitHubCommentGenerator()

        # Set environment variables
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "test-token",
                "GITHUB_REPOSITORY": "test/repo",
                "GITHUB_PR_NUMBER": "123",
            },
        ):
            success = generator.post_to_github("Test comment")
            assert success is True

            # Verify the request was made correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["headers"]["Authorization"] == "token test-token"
            assert call_args[1]["json"]["body"] == "Test comment"

    @patch("requests.post")
    def test_post_to_github_failure(self, mock_post):
        """Test failed GitHub comment posting."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        generator = GitHubCommentGenerator()

        # Set environment variables
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "test-token",
                "GITHUB_REPOSITORY": "test/repo",
                "GITHUB_PR_NUMBER": "123",
            },
        ):
            success = generator.post_to_github("Test comment")
            assert success is False

    def test_post_to_github_missing_env(self):
        """Test GitHub comment posting with missing environment variables."""
        generator = GitHubCommentGenerator()

        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            success = generator.post_to_github("Test comment")
            assert success is False


class TestValidationViolation:
    """Test validation violation class."""

    def test_violation_creation(self):
        """Test creating a validation violation."""
        violation = ValidationViolation(
            "Test message",
            ViolationSeverity.HIGH,
            "test.py",
            line_number=10,
            rule_id="TEST_001",
            fix_suggestion="Fix this",
        )

        assert violation.message == "Test message"
        assert violation.severity == ViolationSeverity.HIGH
        assert violation.file_path == "test.py"
        assert violation.line_number == 10
        assert violation.rule_id == "TEST_001"
        assert violation.fix_suggestion == "Fix this"

    def test_violation_repr(self):
        """Test violation string representation."""
        violation = ValidationViolation(
            "Test message", ViolationSeverity.HIGH, "test.py"
        )

        repr_str = repr(violation)
        assert "ValidationViolation" in repr_str
        assert "Test message" in repr_str
        assert "HIGH" in repr_str
        assert "test.py" in repr_str


@pytest.mark.integration
class TestFullValidationWorkflow:
    """Test the complete validation workflow."""

    def test_full_validation_workflow(self):
        """Test running the complete validation workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Set up test directory structure
            src_dir = tmp_path / "src"
            src_dir.mkdir()

            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()

            tests_dir = tmp_path / "tests"
            tests_dir.mkdir()

            # Create README
            readme = tmp_path / "README.md"
            readme.write_text("# Test Project")

            # Create clean Python file
            py_file = src_dir / "clean.py"
            py_file.write_text("""
\"\"\"Clean Python module.\"\"\"

def clean_function():
    \"\"\"A clean function.\"\"\"
    return "clean"
""")

            # Run validation
            validator = EnhancedValidator(tmp_path)
            result = validator.validate_project()

            # Should pass with clean structure
            assert isinstance(result, ValidationResult)
            # May have some violations but should not be critical
            critical_violations = [
                v for v in result.violations if v.severity == ViolationSeverity.CRITICAL
            ]
            assert len(critical_violations) == 0

    def test_validation_with_issues(self):
        """Test validation with various issues."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create forbidden directory
            build_dir = tmp_path / "build"
            build_dir.mkdir()

            # Create Python file with issues
            py_file = tmp_path / "bad.py"
            py_file.write_text("""
# TODO: Fix this
def bad_function(user_input):
    print("Debug message")
    return eval(user_input)
""")

            # Run validation
            validator = EnhancedValidator(tmp_path)
            result = validator.validate_project()

            # Should have multiple violations
            assert len(result.violations) > 0
            assert result.passed is False

            # Check for different severity levels
            grouped = result.group_by_severity()
            assert ViolationSeverity.CRITICAL in grouped  # eval usage
            assert ViolationSeverity.HIGH in grouped  # forbidden directory
            assert ViolationSeverity.MEDIUM in grouped  # print statement
            assert ViolationSeverity.LOW in grouped  # TODO comment
