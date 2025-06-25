#!/usr/bin/env python3
"""Validation script for quality gates system.

This script validates the comprehensive quality gates infrastructure
including validation capabilities, CLI integration, and reporting.
"""

import sys
import tempfile
import warnings
from pathlib import Path
from textwrap import dedent

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_quality_gates_core():
    """Test core quality gates functionality."""
    print("🔧 Testing Quality Gates Core System...")

    try:
        from pynomaly.infrastructure.quality.quality_gates import (
            QualityGateReport,
            QualityGateResult,
            QualityGateType,
            QualityGateValidator,
            QualityLevel,
        )

        # Test basic functionality
        validator = QualityGateValidator()
        print("  ✅ Quality gate validator initialized")

        # Test enums
        assert len(list(QualityGateType)) == 6
        assert len(list(QualityLevel)) == 4
        print("  ✅ Quality gate enums working correctly")

        # Test data classes
        result = QualityGateResult(
            gate_name="Test Gate",
            gate_type=QualityGateType.CODE_QUALITY,
            quality_level=QualityLevel.HIGH,
            passed=True,
            score=8.5,
            max_score=10.0,
        )

        assert result.percentage_score == 85.0
        result_dict = result.to_dict()
        assert "gate_name" in result_dict
        print("  ✅ Quality gate result data class working")

        # Test report
        report = QualityGateReport(
            feature_name="test_feature",
            feature_path="/test/path",
            total_gates=2,
            passed_gates=1,
            failed_gates=1,
            critical_failures=0,
            overall_score=15.0,
            max_overall_score=20.0,
            gate_results=[result],
        )

        assert report.success_rate == 50.0
        assert report.overall_percentage == 75.0
        assert report.integration_approved is False  # Has critical failures
        print("  ✅ Quality gate report working correctly")

        print("  ✅ Quality gates core system working correctly")
        return True

    except Exception as e:
        import traceback

        print(f"  ❌ Quality gates core test failed: {e}")
        print(f"  📝 Traceback: {traceback.format_exc()}")
        return False


def test_quality_validation():
    """Test quality validation on sample code."""
    print("\n🧪 Testing Quality Validation...")

    try:
        from pynomaly.infrastructure.quality.quality_gates import (
            validate_feature_quality,
        )

        # Create high-quality sample code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            good_code = dedent('''
                """High-quality authentication service.
                
                This module provides secure user authentication with proper
                validation, documentation, and error handling.
                """
                
                from __future__ import annotations
                
                import hashlib
                import secrets
                from typing import Dict, Optional
                
                
                class AuthenticationError(Exception):
                    """Raised when authentication fails."""
                    pass
                
                
                class UserAuthenticator:
                    """Secure user authentication service."""
                    
                    def __init__(self) -> None:
                        """Initialize authenticator with secure defaults."""
                        self._users: Dict[str, str] = {}
                        self._salt_length = 32
                    
                    def register_user(self, username: str, password: str) -> bool:
                        """Register a new user with secure password hashing.
                        
                        Args:
                            username: Unique username (must be non-empty)
                            password: Plain text password (must be non-empty)
                            
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
                        try:
                            salt, stored_hash = stored_data.split(':', 1)
                        except ValueError:
                            raise AuthenticationError("Invalid user data")
                        
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
            f.write(good_code)
            f.flush()

            # Validate the good code
            report = validate_feature_quality(Path(f.name), "user_authenticator")

            print(f"  ✅ Feature validated: {report.feature_name}")
            print(f"  ✅ Total gates: {report.total_gates}")
            print(f"  ✅ Passed gates: {report.passed_gates}")
            print(f"  ✅ Success rate: {report.success_rate:.1f}%")
            print(f"  ✅ Overall score: {report.overall_percentage:.1f}%")
            print(f"  ✅ Critical failures: {report.critical_failures}")
            print(f"  ✅ Integration approved: {report.integration_approved}")

            # Should have good scores for well-written code
            assert report.total_gates > 10
            assert report.success_rate > 60.0
            assert report.overall_percentage > 60.0
            assert report.critical_failures == 0
            assert report.integration_approved is True

        # Create poor-quality sample code
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            bad_code = dedent("""
                # No module docstring, no future imports, poor practices
                
                import *
                from os import *
                
                class BadClass:
                    def bad_method(self):
                        # No type hints, no docstrings, security issues
                        user_input = "test"
                        for i in range(len(data)):
                            for j in range(len(data)):
                                for k in range(len(data)):
                                    if i > 0:
                                        if j > 0:
                                            if k > 0:
                                                if i + j + k > 10:
                                                    if i * j * k > 100:
                                                        result = eval(user_input)  # Security issue
                                                        exec("import sys")  # Security issue
                                                        return True
                        return False
                
                def bad_function():
                    try:
                        dangerous_operation()
                    except:  # Bare except
                        pass
            """)
            f.write(bad_code)
            f.flush()

            # Validate the poor code
            bad_report = validate_feature_quality(Path(f.name), "bad_feature")

            print(f"  ✅ Poor code validated: {bad_report.feature_name}")
            print(f"  ✅ Poor code success rate: {bad_report.success_rate:.1f}%")
            print(f"  ✅ Poor code critical failures: {bad_report.critical_failures}")

            # Should have lower scores for poor code
            assert bad_report.success_rate < 100.0
            assert bad_report.failed_gates > 0

        print("  ✅ Quality validation working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Quality validation test failed: {e}")
        return False


def test_specific_quality_gates():
    """Test specific quality gate implementations."""
    print("\n🚪 Testing Specific Quality Gates...")

    try:
        from pynomaly.infrastructure.quality.quality_gates import QualityGateValidator

        validator = QualityGateValidator()

        # Test cyclomatic complexity gate
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            complex_code = dedent('''
                """Module with complex function."""
                
                def complex_function(x):
                    """Very complex function with many branches."""
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
            f.write(complex_code)
            f.flush()

            result = validator._check_cyclomatic_complexity(Path(f.name))
            assert result.gate_name == "Cyclomatic Complexity"
            assert "average_complexity" in result.details
            assert "max_complexity" in result.details
            print("  ✅ Cyclomatic complexity gate working")

        # Test docstring coverage gate
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            docstring_code = dedent('''
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
            f.write(docstring_code)
            f.flush()

            result = validator._check_docstring_coverage(Path(f.name))
            assert result.gate_name == "Docstring Coverage"
            assert "coverage_percentage" in result.details
            assert 0 < result.details["coverage_percentage"] < 100
            print("  ✅ Docstring coverage gate working")

        # Test type hints coverage gate
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            type_hints_code = dedent('''
                """Module with mixed type hints."""
                
                from typing import List
                
                def typed_function(data: List[int]) -> int:
                    """Function with type hints."""
                    return sum(data)
                
                def untyped_function(data):
                    """Function without type hints."""
                    return len(data)
            ''')
            f.write(type_hints_code)
            f.flush()

            result = validator._check_type_hints(Path(f.name))
            assert result.gate_name == "Type Hints"
            assert "coverage_percentage" in result.details
            print("  ✅ Type hints coverage gate working")

        # Test security patterns gate
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            security_code = dedent('''
                """Module with security issues."""
                
                import subprocess
                
                def insecure_function(user_input):
                    """Function with security issues."""
                    try:
                        result = eval(user_input)  # Dangerous
                        subprocess.run(user_input, shell=True)  # Dangerous
                        exec(user_input)  # Dangerous
                        return result
                    except:  # Bare except
                        return None
            ''')
            f.write(security_code)
            f.flush()

            result = validator._check_security_patterns(Path(f.name))
            assert result.gate_name == "Security Patterns"
            assert result.passed is False  # Should fail due to security issues
            assert len(result.details["issues"]) > 0
            print("  ✅ Security patterns gate working")

        print("  ✅ Specific quality gates working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Specific quality gates test failed: {e}")
        return False


def test_html_report_generation():
    """Test HTML report generation."""
    print("\n📄 Testing HTML Report Generation...")

    try:
        from pynomaly.infrastructure.quality.quality_gates import (
            QualityGateReport,
            QualityGateResult,
            QualityGateType,
            QualityGateValidator,
            QualityLevel,
        )

        # Create sample report
        results = [
            QualityGateResult(
                gate_name="Code Quality",
                gate_type=QualityGateType.CODE_QUALITY,
                quality_level=QualityLevel.HIGH,
                passed=True,
                score=9.0,
                max_score=10.0,
                recommendations=["Keep up the good work"],
            ),
            QualityGateResult(
                gate_name="Security Check",
                gate_type=QualityGateType.SECURITY,
                quality_level=QualityLevel.CRITICAL,
                passed=False,
                score=3.0,
                max_score=10.0,
                recommendations=["Fix security issues", "Use input validation"],
            ),
        ]

        report = QualityGateReport(
            feature_name="test_feature",
            feature_path="/test/path/feature.py",
            total_gates=2,
            passed_gates=1,
            failed_gates=1,
            critical_failures=1,
            overall_score=12.0,
            max_overall_score=20.0,
            gate_results=results,
        )

        # Generate HTML report
        validator = QualityGateValidator()
        html = validator.generate_report_html(report)

        # Verify HTML content
        assert html is not None
        assert isinstance(html, str)
        assert "Quality Gate Report" in html
        assert "test_feature" in html
        assert "Code Quality" in html
        assert "Security Check" in html
        assert "Keep up the good work" in html
        assert "Fix security issues" in html

        print(f"  ✅ HTML report generated: {len(html)} characters")
        print(f"  ✅ Contains feature name: {'test_feature' in html}")
        print(f"  ✅ Contains gate results: {'Code Quality' in html}")
        print(f"  ✅ Contains recommendations: {'Keep up the good work' in html}")

        print("  ✅ HTML report generation working correctly")
        return True

    except Exception as e:
        print(f"  ❌ HTML report generation test failed: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions and utilities."""
    print("\n🛠️ Testing Convenience Functions...")

    try:
        from pynomaly.infrastructure.quality.quality_gates import (
            validate_feature_quality,
        )

        # Test convenience function
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            simple_code = dedent('''
                """Simple test module."""
                
                def simple_function() -> str:
                    """A simple function with type hints."""
                    return "hello world"
            ''')
            f.write(simple_code)
            f.flush()

            # Test with different parameters
            report1 = validate_feature_quality(Path(f.name))
            assert report1.feature_name == Path(f.name).stem

            report2 = validate_feature_quality(Path(f.name), "custom_name")
            assert report2.feature_name == "custom_name"

            report3 = validate_feature_quality(Path(f.name), project_root=Path.cwd())
            assert report3.feature_name == Path(f.name).stem

            print("  ✅ Convenience function working with default name")
            print("  ✅ Convenience function working with custom name")
            print("  ✅ Convenience function working with project root")

        print("  ✅ Convenience functions working correctly")
        return True

    except Exception as e:
        print(f"  ❌ Convenience functions test failed: {e}")
        return False


def test_quality_gates_readiness():
    """Test overall quality gates readiness."""
    print("\n🚀 Testing Quality Gates Readiness...")

    try:
        # Check if all components are available
        components = [
            "quality_gates_core",
            "quality_validation",
            "specific_quality_gates",
            "html_report_generation",
            "convenience_functions",
        ]

        results = {
            "quality_gates_core": test_quality_gates_core(),
            "quality_validation": test_quality_validation(),
            "specific_quality_gates": test_specific_quality_gates(),
            "html_report_generation": test_html_report_generation(),
            "convenience_functions": test_convenience_functions(),
        }

        passing = sum(results.values())
        total = len(results)

        print(f"\n📈 Quality Gates Status: {passing}/{total} components ready")

        if passing == total:
            print("🎉 Quality gates infrastructure is fully operational!")
            print("✅ Ready for comprehensive feature quality validation")
            return True
        else:
            print("⚠️ Some quality gates components need attention")
            for component, status in results.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}")
            return False

    except Exception as e:
        print(f"❌ Quality gates readiness test failed: {e}")
        return False


def main():
    """Run all quality gates validation tests."""
    print("🧪 Pynomaly Quality Gates Infrastructure Validation")
    print("=" * 70)

    try:
        success = test_quality_gates_readiness()

        if success:
            print("\n🎯 Quality gates infrastructure validation successful!")
            print("🚀 Ready for comprehensive feature quality validation and gating")
            sys.exit(0)
        else:
            print("\n⚠️ Quality gates infrastructure validation failed")
            print("🔧 Please review and fix issues before proceeding")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
