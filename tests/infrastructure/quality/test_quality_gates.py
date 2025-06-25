"""Tests for quality gates system."""

import tempfile
from pathlib import Path
from textwrap import dedent

from pynomaly.infrastructure.quality.quality_gates import (
    QualityGateReport,
    QualityGateResult,
    QualityGateType,
    QualityGateValidator,
    QualityLevel,
    validate_feature_quality,
)


class TestQualityGateValidator:
    """Test quality gate validator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = QualityGateValidator()

    def test_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator is not None
        assert hasattr(self.validator, "thresholds")
        assert "cyclomatic_complexity" in self.validator.thresholds
        assert "test_coverage" in self.validator.thresholds

    def test_quality_gate_result_creation(self):
        """Test quality gate result data class."""
        result = QualityGateResult(
            gate_name="Test Gate",
            gate_type=QualityGateType.CODE_QUALITY,
            quality_level=QualityLevel.HIGH,
            passed=True,
            score=8.5,
            max_score=10.0,
        )

        assert result.gate_name == "Test Gate"
        assert result.gate_type == QualityGateType.CODE_QUALITY
        assert result.quality_level == QualityLevel.HIGH
        assert result.passed is True
        assert result.percentage_score == 85.0

        # Test serialization
        result_dict = result.to_dict()
        assert result_dict["gate_name"] == "Test Gate"
        assert result_dict["percentage_score"] == 85.0

    def test_quality_gate_report_creation(self):
        """Test quality gate report data class."""
        results = [
            QualityGateResult(
                gate_name="Gate 1",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.HIGH,
                passed=True,
                score=9.0,
                max_score=10.0,
            ),
            QualityGateResult(
                gate_name="Gate 2",
                gate_type=QualityGateType.TESTING,
                quality_level=QualityLevel.CRITICAL,
                passed=False,
                score=6.0,
                max_score=10.0,
            ),
        ]

        report = QualityGateReport(
            feature_name="test_feature",
            feature_path="/test/path",
            total_gates=2,
            passed_gates=1,
            failed_gates=1,
            critical_failures=1,
            overall_score=15.0,
            max_overall_score=20.0,
            gate_results=results,
        )

        assert report.feature_name == "test_feature"
        assert report.success_rate == 50.0  # 1/2 * 100
        assert report.overall_percentage == 75.0  # 15/20 * 100
        assert report.integration_approved is False  # Has critical failures

        # Test serialization
        report_dict = report.to_dict()
        assert report_dict["feature_name"] == "test_feature"
        assert report_dict["success_rate"] == 50.0
        assert len(report_dict["gate_results"]) == 2

    def test_validate_non_existent_file(self):
        """Test validation of non-existent file."""
        non_existent_path = Path("/non/existent/file.py")
        report = self.validator.validate_feature(non_existent_path)

        assert report is not None
        assert report.feature_name == "file"
        assert report.total_gates > 0
        # Most gates should pass for non-existent files (skipped)
        assert report.success_rate >= 80.0

    def test_validate_good_quality_code(self):
        """Test validation of high-quality code."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write high-quality code
            code = dedent('''
                """High-quality module with good practices."""

                from __future__ import annotations

                from typing import Dict, List, Optional


                class HighQualityClass:
                    """A well-documented class following best practices."""

                    def __init__(self, name: str) -> None:
                        """Initialize with name.

                        Args:
                            name: The name to use
                        """
                        self.name = name

                    def process_data(self, data: List[int]) -> Dict[str, int]:
                        """Process data and return statistics.

                        Args:
                            data: List of integers to process

                        Returns:
                            Dictionary with statistics
                        """
                        if not data:
                            return {}

                        return {
                            'count': len(data),
                            'sum': sum(data),
                            'max': max(data),
                            'min': min(data)
                        }


                def helper_function(value: int) -> bool:
                    """Helper function with type hints.

                    Args:
                        value: Integer value to check

                    Returns:
                        True if positive, False otherwise
                    """
                    return value > 0
            ''')
            f.write(code)
            f.flush()

            # Validate the high-quality code
            report = self.validator.validate_feature(
                Path(f.name), "high_quality_feature"
            )

            assert report is not None
            assert report.feature_name == "high_quality_feature"
            assert report.total_gates > 0

            # Should have high success rate for good code
            assert report.success_rate >= 70.0
            assert report.overall_percentage >= 70.0

            # Should have no critical failures
            assert report.critical_failures == 0

    def test_validate_poor_quality_code(self):
        """Test validation of poor-quality code."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write poor-quality code
            code = dedent("""
                # No module docstring
                # No future imports
                # No type hints

                import *
                from os import *

                class BadClass:
                    def bad_method(self):
                        # Very complex method with many branches
                        for i in range(len(data)):
                            for j in range(len(data)):
                                for k in range(len(data)):
                                    if i > 0:
                                        if j > 0:
                                            if k > 0:
                                                if i + j + k > 10:
                                                    if i * j * k > 100:
                                                        eval("print('bad')")
                                                        exec("import sys")
                                                        return True
                        return False

                def bad_function():
                    try:
                        risky_operation()
                    except:
                        pass
            """)
            f.write(code)
            f.flush()

            # Validate the poor-quality code
            report = self.validator.validate_feature(
                Path(f.name), "poor_quality_feature"
            )

            assert report is not None
            assert report.feature_name == "poor_quality_feature"
            assert report.total_gates > 0

            # Should have lower success rate for poor code
            assert report.success_rate < 100.0

            # Should have some failures
            assert report.failed_gates > 0

            # Check specific gate failures
            gate_names = [result.gate_name for result in report.gate_results]
            assert "Security Patterns" in gate_names
            assert "Import Quality" in gate_names
            assert "Type Hints" in gate_names

    def test_cyclomatic_complexity_gate(self):
        """Test cyclomatic complexity gate specifically."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write code with high complexity
            code = dedent('''
                """Module with complex function."""

                def complex_function(x):
                    """Very complex function."""
                    if x > 0:
                        if x > 10:
                            if x > 20:
                                if x > 30:
                                    if x > 40:
                                        if x > 50:
                                            return "very high"
                                        return "high"
                                    return "medium-high"
                                return "medium"
                            return "low-medium"
                        return "low"
                    return "zero or negative"
            ''')
            f.write(code)
            f.flush()

            # Test complexity gate
            result = self.validator._check_cyclomatic_complexity(Path(f.name))

            assert result.gate_name == "Cyclomatic Complexity"
            assert result.gate_type == QualityGateType.CODE_QUALITY
            assert "average_complexity" in result.details
            assert "max_complexity" in result.details

    def test_docstring_coverage_gate(self):
        """Test docstring coverage gate specifically."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write code with partial docstring coverage
            code = dedent('''
                """Module docstring."""

                class DocumentedClass:
                    """This class has a docstring."""

                    def documented_method(self):
                        """This method has a docstring."""
                        pass

                    def undocumented_method(self):
                        pass

                def documented_function():
                    """This function has a docstring."""
                    pass

                def undocumented_function():
                    pass
            ''')
            f.write(code)
            f.flush()

            # Test docstring coverage gate
            result = self.validator._check_docstring_coverage(Path(f.name))

            assert result.gate_name == "Docstring Coverage"
            assert result.gate_type == QualityGateType.DOCUMENTATION
            assert "coverage_percentage" in result.details
            assert "total_items" in result.details
            assert "documented_items" in result.details

            # Should have partial coverage (module + 1 class + 2/4 methods)
            coverage = result.details["coverage_percentage"]
            assert 0 < coverage < 100

    def test_type_hints_coverage_gate(self):
        """Test type hints coverage gate specifically."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write code with partial type hints
            code = dedent('''
                """Module with mixed type hints."""

                from typing import List, Optional

                def typed_function(data: List[int]) -> int:
                    """Function with type hints."""
                    return sum(data)

                def untyped_function(data):
                    """Function without type hints."""
                    return len(data)

                def partially_typed_function(data: List[int]):
                    """Function with partial type hints."""
                    return data[0] if data else None
            ''')
            f.write(code)
            f.flush()

            # Test type hints coverage gate
            result = self.validator._check_type_hints(Path(f.name))

            assert result.gate_name == "Type Hints"
            assert result.gate_type == QualityGateType.CODE_QUALITY
            assert "coverage_percentage" in result.details
            assert "total_functions" in result.details
            assert "functions_with_hints" in result.details

    def test_security_patterns_gate(self):
        """Test security patterns gate specifically."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write code with security issues
            code = dedent('''
                """Module with security issues."""

                import subprocess

                def insecure_function(user_input):
                    """Function with security issues."""
                    try:
                        # Dangerous eval usage
                        result = eval(user_input)

                        # Dangerous subprocess usage
                        subprocess.run(user_input, shell=True)

                        # Dangerous exec usage
                        exec(user_input)

                        return result
                    except:
                        # Bare except clause
                        return None
            ''')
            f.write(code)
            f.flush()

            # Test security patterns gate
            result = self.validator._check_security_patterns(Path(f.name))

            assert result.gate_name == "Security Patterns"
            assert result.gate_type == QualityGateType.SECURITY
            assert result.passed is False  # Should fail due to security issues
            assert "issues" in result.details
            assert len(result.details["issues"]) > 0

    def test_clean_architecture_gate(self):
        """Test clean architecture gate specifically."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.name = "/test/domain/entities/test_entity.py"  # Simulate domain layer

            # Write code that violates clean architecture
            code = dedent('''
                """Domain entity that incorrectly imports from infrastructure."""

                from ...infrastructure.database import DatabaseConnection
                from ...application.services import SomeService

                class TestEntity:
                    """Entity that violates clean architecture."""

                    def __init__(self):
                        self.db = DatabaseConnection()  # Domain shouldn't depend on infrastructure
                        self.service = SomeService()    # Domain shouldn't depend on application
            ''')
            f.write(code)
            f.flush()

            # Create a path that looks like domain layer
            domain_path = Path(f.name).parent / "domain" / "entities" / "test_entity.py"

            # Test clean architecture gate (this will pass since it's a mock path)
            result = self.validator._check_clean_architecture(domain_path)

            assert result.gate_name == "Clean Architecture"
            assert result.gate_type == QualityGateType.ARCHITECTURE
            assert result.quality_level == QualityLevel.CRITICAL

    def test_generate_html_report(self):
        """Test HTML report generation."""
        # Create a sample report
        results = [
            QualityGateResult(
                gate_name="Test Gate 1",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.HIGH,
                passed=True,
                score=9.0,
                max_score=10.0,
                recommendations=["Keep up the good work"],
            ),
            QualityGateResult(
                gate_name="Test Gate 2",
                gate_type=QualityGateType.TESTING,
                quality_level=QualityLevel.CRITICAL,
                passed=False,
                score=5.0,
                max_score=10.0,
                recommendations=["Add more tests", "Improve coverage"],
            ),
        ]

        report = QualityGateReport(
            feature_name="test_feature",
            feature_path="/test/path/feature.py",
            total_gates=2,
            passed_gates=1,
            failed_gates=1,
            critical_failures=1,
            overall_score=14.0,
            max_overall_score=20.0,
            gate_results=results,
        )

        # Generate HTML report
        html = self.validator.generate_report_html(report)

        assert html is not None
        assert isinstance(html, str)
        assert "Quality Gate Report" in html
        assert "test_feature" in html
        assert "Test Gate 1" in html
        assert "Test Gate 2" in html
        assert "Keep up the good work" in html
        assert "Add more tests" in html

    def test_convenience_function(self):
        """Test convenience function for feature validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            code = dedent('''
                """Simple test module."""

                def simple_function():
                    """A simple function."""
                    return "hello world"
            ''')
            f.write(code)
            f.flush()

            # Test convenience function
            report = validate_feature_quality(Path(f.name), "test_feature")

            assert report is not None
            assert report.feature_name == "test_feature"
            assert report.total_gates > 0


class TestQualityGateEnums:
    """Test quality gate enumerations."""

    def test_quality_gate_type_enum(self):
        """Test quality gate type enumeration."""
        assert QualityGateType.CODE_QUALITY.value == "code_quality"
        assert QualityGateType.PERFORMANCE.value == "performance"
        assert QualityGateType.DOCUMENTATION.value == "documentation"
        assert QualityGateType.ARCHITECTURE.value == "architecture"
        assert QualityGateType.TESTING.value == "testing"
        assert QualityGateType.SECURITY.value == "security"

    def test_quality_level_enum(self):
        """Test quality level enumeration."""
        assert QualityLevel.CRITICAL.value == "critical"
        assert QualityLevel.HIGH.value == "high"
        assert QualityLevel.MEDIUM.value == "medium"
        assert QualityLevel.LOW.value == "low"


class TestQualityGateIntegration:
    """Test quality gate system integration."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Write a realistic feature file
            code = dedent('''
                """User authentication service.

                This module provides secure user authentication capabilities
                with proper input validation and error handling.
                """

                from __future__ import annotations

                import hashlib
                import secrets
                from typing import Dict, Optional, Tuple


                class AuthenticationError(Exception):
                    """Raised when authentication fails."""
                    pass


                class UserAuthenticator:
                    """Secure user authentication service."""

                    def __init__(self) -> None:
                        """Initialize authenticator."""
                        self._users: Dict[str, str] = {}
                        self._salt_length = 32

                    def register_user(self, username: str, password: str) -> bool:
                        """Register a new user with secure password hashing.

                        Args:
                            username: Unique username
                            password: Plain text password to hash

                        Returns:
                            True if registration successful, False if user exists

                        Raises:
                            ValueError: If username or password is invalid
                        """
                        if not username or not password:
                            raise ValueError("Username and password cannot be empty")

                        if username in self._users:
                            return False

                        # Generate secure password hash
                        salt = secrets.token_hex(self._salt_length)
                        password_hash = self._hash_password(password, salt)
                        self._users[username] = f"{salt}:{password_hash}"

                        return True

                    def authenticate_user(self, username: str, password: str) -> bool:
                        """Authenticate user with username and password.

                        Args:
                            username: Username to authenticate
                            password: Password to verify

                        Returns:
                            True if authentication successful

                        Raises:
                            AuthenticationError: If authentication fails
                        """
                        if username not in self._users:
                            raise AuthenticationError("Invalid username")

                        stored_data = self._users[username]
                        salt, stored_hash = stored_data.split(':', 1)

                        computed_hash = self._hash_password(password, salt)

                        if computed_hash != stored_hash:
                            raise AuthenticationError("Invalid password")

                        return True

                    def _hash_password(self, password: str, salt: str) -> str:
                        """Hash password with salt using SHA-256.

                        Args:
                            password: Plain text password
                            salt: Salt for hashing

                        Returns:
                            Hexadecimal hash string
                        """
                        combined = f"{password}{salt}".encode('utf-8')
                        return hashlib.sha256(combined).hexdigest()
            ''')
            f.write(code)
            f.flush()

            # Validate the realistic feature
            validator = QualityGateValidator()
            report = validator.validate_feature(Path(f.name), "user_authentication")

            # Verify report structure
            assert report is not None
            assert report.feature_name == "user_authentication"
            assert report.total_gates > 10  # Should have many gates

            # Should have good scores for well-written code
            assert report.overall_percentage > 60.0
            assert report.success_rate > 60.0

            # Should have no critical failures for good code
            assert report.critical_failures == 0

            # Should be approved for integration
            assert report.integration_approved is True

            # Verify specific gates are present
            gate_names = [result.gate_name for result in report.gate_results]
            expected_gates = [
                "Cyclomatic Complexity",
                "Code Style",
                "Type Hints",
                "Import Quality",
                "Docstring Coverage",
                "Clean Architecture",
                "Security Patterns",
            ]

            for expected_gate in expected_gates:
                assert expected_gate in gate_names
