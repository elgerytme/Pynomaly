"""Comprehensive tests for infrastructure security validation."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from pynomaly.infrastructure.security.validation import (
    InputSanitizer,
    InputValidator,
    SecureBaseModel,
    SecurityConfig,
    ThreatDetector,
    ValidationError,
    validate_input,
)


class TestInputSanitizer:
    """Test input sanitization functionality."""

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        sanitizer = InputSanitizer()

        # Normal string
        result = sanitizer.sanitize_string("Hello World")
        assert result == "Hello World"

        # String with HTML
        result = sanitizer.sanitize_string("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result
        assert "&lt;/script&gt;" in result

        # String with null bytes
        result = sanitizer.sanitize_string("Hello\x00World")
        assert result == "Hello World"

        # String with extra whitespace
        result = sanitizer.sanitize_string("  Hello   World  ")
        assert result == "Hello World"

    def test_sanitize_string_length_validation(self):
        """Test string length validation."""
        sanitizer = InputSanitizer()

        # Test with custom max length
        with pytest.raises(ValidationError, match="String too long"):
            sanitizer.sanitize_string("a" * 10, max_length=5)

        # Test with default max length
        long_string = "a" * (SecurityConfig.MAX_STRING_LENGTH + 1)
        with pytest.raises(ValidationError, match="String too long"):
            sanitizer.sanitize_string(long_string)

    def test_sanitize_string_invalid_input(self):
        """Test sanitization with invalid input types."""
        sanitizer = InputSanitizer()

        with pytest.raises(ValidationError, match="Input must be a string"):
            sanitizer.sanitize_string(123)

        with pytest.raises(ValidationError, match="Input must be a string"):
            sanitizer.sanitize_string(None)

    def test_sanitize_html_with_bleach(self):
        """Test HTML sanitization with bleach library."""
        sanitizer = InputSanitizer()

        # Test with allowed tags
        html_input = "<p>Hello <b>World</b></p><script>alert('xss')</script>"
        result = sanitizer.sanitize_html(html_input)

        # Should keep allowed tags and remove dangerous ones
        assert "<p>" in result or "&lt;p&gt;" in result
        assert "<b>" in result or "&lt;b&gt;" in result
        assert "<script>" not in result

    def test_sanitize_html_without_bleach(self):
        """Test HTML sanitization fallback without bleach."""
        sanitizer = InputSanitizer()

        with patch(
            "pynomaly.infrastructure.security.validation.BLEACH_AVAILABLE", False
        ):
            html_input = "<p>Hello <b>World</b></p>"
            result = sanitizer.sanitize_html(html_input)

            # Should escape all HTML
            assert "&lt;p&gt;" in result
            assert "&lt;b&gt;" in result

    def test_sanitize_html_length_validation(self):
        """Test HTML length validation."""
        sanitizer = InputSanitizer()

        long_html = "<p>" + "a" * SecurityConfig.MAX_TEXT_LENGTH + "</p>"
        with pytest.raises(ValidationError, match="Text too long"):
            sanitizer.sanitize_html(long_html)

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        sanitizer = InputSanitizer()

        # Normal filename
        result = sanitizer.sanitize_filename("document.pdf")
        assert result == "document.pdf"

        # Filename with dangerous characters
        result = sanitizer.sanitize_filename('document<>:"|?*.pdf')
        assert result == "document________.pdf"

        # Filename with path components
        result = sanitizer.sanitize_filename("../../etc/passwd")
        assert result == "passwd"

        result = sanitizer.sanitize_filename("..\\..\\windows\\system32")
        assert result == "system32"

    def test_sanitize_filename_edge_cases(self):
        """Test filename sanitization edge cases."""
        sanitizer = InputSanitizer()

        # Empty filename
        with pytest.raises(ValidationError, match="Invalid filename"):
            sanitizer.sanitize_filename("")

        # Filename with only dots
        with pytest.raises(ValidationError, match="Invalid filename"):
            sanitizer.sanitize_filename("...")

        # Filename too long
        long_filename = "a" * (SecurityConfig.MAX_FILENAME_LENGTH + 1) + ".txt"
        with pytest.raises(ValidationError, match="Filename too long"):
            sanitizer.sanitize_filename(long_filename)

    def test_sanitize_url_basic(self):
        """Test basic URL sanitization."""
        sanitizer = InputSanitizer()

        # Valid URLs
        result = sanitizer.sanitize_url("https://example.com")
        assert result == "https://example.com"

        result = sanitizer.sanitize_url("http://example.com/path?query=value")
        assert result.startswith("http://example.com/path")

    def test_sanitize_url_invalid_schemes(self):
        """Test URL sanitization with invalid schemes."""
        sanitizer = InputSanitizer()

        # Invalid schemes
        with pytest.raises(
            ValidationError, match="Only HTTP and HTTPS URLs are allowed"
        ):
            sanitizer.sanitize_url("ftp://example.com")

        with pytest.raises(
            ValidationError, match="Only HTTP and HTTPS URLs are allowed"
        ):
            sanitizer.sanitize_url("javascript:alert('xss')")

        with pytest.raises(
            ValidationError, match="Only HTTP and HTTPS URLs are allowed"
        ):
            sanitizer.sanitize_url("file:///etc/passwd")

    def test_sanitize_url_malformed(self):
        """Test URL sanitization with malformed URLs."""
        sanitizer = InputSanitizer()

        # Malformed URLs
        with pytest.raises(ValidationError, match="Invalid URL format"):
            sanitizer.sanitize_url("not-a-url")

        with pytest.raises(ValidationError, match="URL must be a string"):
            sanitizer.sanitize_url(123)


class TestThreatDetector:
    """Test threat detection functionality."""

    def test_detect_sql_injection_basic(self):
        """Test basic SQL injection detection."""
        detector = ThreatDetector()

        # Positive cases
        assert detector.detect_sql_injection("' OR '1'='1")
        assert detector.detect_sql_injection("'; DROP TABLE users; --")
        assert detector.detect_sql_injection("1' UNION SELECT * FROM users")
        assert detector.detect_sql_injection("admin' --")

        # Negative cases
        assert not detector.detect_sql_injection("regular text")
        assert not detector.detect_sql_injection("user@example.com")
        assert not detector.detect_sql_injection("file.txt")

    def test_detect_sql_injection_case_insensitive(self):
        """Test SQL injection detection is case insensitive."""
        detector = ThreatDetector()

        assert detector.detect_sql_injection("' or '1'='1")
        assert detector.detect_sql_injection("' OR '1'='1")
        assert detector.detect_sql_injection("' Or '1'='1")
        assert detector.detect_sql_injection("1' union select * from users")
        assert detector.detect_sql_injection("1' UNION SELECT * FROM users")

    def test_detect_xss_basic(self):
        """Test basic XSS detection."""
        detector = ThreatDetector()

        # Positive cases
        assert detector.detect_xss("<script>alert('xss')</script>")
        assert detector.detect_xss("javascript:alert('xss')")
        assert detector.detect_xss('<img src="x" onerror="alert(1)">')
        assert detector.detect_xss('<iframe src="javascript:alert(1)"></iframe>')
        assert detector.detect_xss('<object data="javascript:alert(1)"></object>')

        # Negative cases
        assert not detector.detect_xss("regular text")
        assert not detector.detect_xss("<p>Normal HTML</p>")
        assert not detector.detect_xss("user@example.com")

    def test_detect_xss_case_insensitive(self):
        """Test XSS detection is case insensitive."""
        detector = ThreatDetector()

        assert detector.detect_xss("<SCRIPT>alert('xss')</SCRIPT>")
        assert detector.detect_xss("JAVASCRIPT:alert('xss')")
        assert detector.detect_xss('<IMG SRC="x" ONERROR="alert(1)">')

    def test_detect_path_traversal_basic(self):
        """Test basic path traversal detection."""
        detector = ThreatDetector()

        # Positive cases
        assert detector.detect_path_traversal("../../../etc/passwd")
        assert detector.detect_path_traversal("..\\..\\windows\\system32")
        assert detector.detect_path_traversal("%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd")
        assert detector.detect_path_traversal("%2e%2e%5c%2e%2e%5cwindows%5csystem32")

        # Negative cases
        assert not detector.detect_path_traversal("regular/path/file.txt")
        assert not detector.detect_path_traversal("documents/report.pdf")
        assert not detector.detect_path_traversal("user@example.com")

    def test_detect_command_injection_basic(self):
        """Test basic command injection detection."""
        detector = ThreatDetector()

        # Positive cases
        assert detector.detect_command_injection("ls -la | cat")
        assert detector.detect_command_injection("rm -rf / && echo done")
        assert detector.detect_command_injection("cat /etc/passwd; whoami")
        assert detector.detect_command_injection("$(whoami)")
        assert detector.detect_command_injection("`ls -la`")
        assert detector.detect_command_injection("command > output.txt")
        assert detector.detect_command_injection("command < input.txt")
        assert detector.detect_command_injection("command\nls")

        # Negative cases
        assert not detector.detect_command_injection("regular text")
        assert not detector.detect_command_injection("user@example.com")
        assert not detector.detect_command_injection("file.txt")


class TestInputValidator:
    """Test high-level input validation."""

    def test_validate_and_sanitize_string(self):
        """Test string validation and sanitization."""
        validator = InputValidator(strict_mode=False)

        # Normal string
        result = validator.validate_and_sanitize("Hello World", "string")
        assert result == "Hello World"

        # String with HTML
        result = validator.validate_and_sanitize("<b>Hello</b>", "string")
        assert "&lt;b&gt;" in result

    def test_validate_and_sanitize_email(self):
        """Test email validation and sanitization."""
        validator = InputValidator(strict_mode=False)

        # Valid emails
        result = validator.validate_and_sanitize("user@example.com", "email")
        assert result == "user@example.com"

        result = validator.validate_and_sanitize("  USER@EXAMPLE.COM  ", "email")
        assert result == "user@example.com"

        # Invalid emails
        with pytest.raises(ValidationError, match="Invalid email format"):
            validator.validate_and_sanitize("invalid-email", "email")

        with pytest.raises(ValidationError, match="Invalid email format"):
            validator.validate_and_sanitize("user@", "email")

    def test_validate_and_sanitize_username(self):
        """Test username validation and sanitization."""
        validator = InputValidator(strict_mode=False)

        # Valid usernames
        result = validator.validate_and_sanitize("user123", "username")
        assert result == "user123"

        result = validator.validate_and_sanitize("user.name_123", "username")
        assert result == "user.name_123"

        # Invalid usernames
        with pytest.raises(
            ValidationError, match="Username must be at least 3 characters"
        ):
            validator.validate_and_sanitize("ab", "username")

        with pytest.raises(ValidationError, match="Username can only contain"):
            validator.validate_and_sanitize("user@example", "username")

    def test_validate_and_sanitize_threats_strict_mode(self):
        """Test threat detection in strict mode."""
        validator = InputValidator(strict_mode=True)

        # SQL injection
        with pytest.raises(HTTPException) as exc_info:
            validator.validate_and_sanitize("' OR '1'='1", "string")
        assert exc_info.value.status_code == 400
        assert "SQL injection" in str(exc_info.value.detail)

        # XSS
        with pytest.raises(HTTPException) as exc_info:
            validator.validate_and_sanitize("<script>alert('xss')</script>", "string")
        assert exc_info.value.status_code == 400
        assert "XSS" in str(exc_info.value.detail)

    def test_validate_and_sanitize_threats_non_strict_mode(self):
        """Test threat detection in non-strict mode."""
        validator = InputValidator(strict_mode=False)

        # Should not raise HTTPException but still detect threats
        result = validator.validate_and_sanitize("' OR '1'='1", "string")
        assert result is not None  # Should still sanitize

        result = validator.validate_and_sanitize(
            "<script>alert('xss')</script>", "string"
        )
        assert "&lt;script&gt;" in result

    def test_validate_and_sanitize_none_values(self):
        """Test handling of None values."""
        validator = InputValidator(strict_mode=False)

        result = validator.validate_and_sanitize(None, "string")
        assert result is None

        result = validator.validate_and_sanitize(None, "email")
        assert result is None

    def test_validate_and_sanitize_non_string_types(self):
        """Test handling of non-string types."""
        validator = InputValidator(strict_mode=False)

        # Numbers should pass through for non-string fields
        result = validator.validate_and_sanitize(123, "number")
        assert result == 123

        # Numbers should fail for string fields
        with pytest.raises(ValidationError, match="Expected string for string"):
            validator.validate_and_sanitize(123, "string")

    def test_validate_and_sanitize_html_allowed(self):
        """Test HTML validation when allowed."""
        validator = InputValidator(strict_mode=False)

        result = validator.validate_and_sanitize(
            "<p>Hello <b>World</b></p>", "text", allow_html=True
        )
        # Should contain HTML tags (either original or escaped)
        assert result is not None
        assert "Hello" in result
        assert "World" in result

    def test_validate_and_sanitize_filename(self):
        """Test filename validation."""
        validator = InputValidator(strict_mode=False)

        result = validator.validate_and_sanitize("document.pdf", "filename")
        assert result == "document.pdf"

        result = validator.validate_and_sanitize("../../../etc/passwd", "filename")
        assert result == "passwd"

    def test_validate_and_sanitize_url(self):
        """Test URL validation."""
        validator = InputValidator(strict_mode=False)

        result = validator.validate_and_sanitize("https://example.com", "url")
        assert result == "https://example.com"

        with pytest.raises(
            ValidationError, match="Only HTTP and HTTPS URLs are allowed"
        ):
            validator.validate_and_sanitize("javascript:alert('xss')", "url")


class TestSecureBaseModel:
    """Test secure Pydantic base model."""

    def test_secure_model_basic(self):
        """Test basic secure model functionality."""

        class TestModel(SecureBaseModel):
            name: str
            email: str
            username: str

        # Valid data
        model = TestModel(name="John Doe", email="john@example.com", username="johndoe")
        assert model.name == "John Doe"
        assert model.email == "john@example.com"
        assert model.username == "johndoe"

    def test_secure_model_sanitization(self):
        """Test automatic sanitization in secure model."""

        class TestModel(SecureBaseModel):
            name: str
            description: str

        # Data with HTML
        model = TestModel(
            name="<script>alert('xss')</script>", description="<b>Bold text</b>"
        )

        # Should be sanitized
        assert "&lt;script&gt;" in model.name
        assert "&lt;b&gt;" in model.description or "<b>" not in model.description

    def test_secure_model_threat_detection(self):
        """Test threat detection in secure model."""

        class TestModel(SecureBaseModel):
            query: str

        # Should raise HTTPException for SQL injection
        with pytest.raises(HTTPException) as exc_info:
            TestModel(query="' OR '1'='1")
        assert exc_info.value.status_code == 400
        assert "SQL injection" in str(exc_info.value.detail)


class TestGlobalValidationFunction:
    """Test global validation function."""

    def test_validate_input_function(self):
        """Test the global validate_input function."""
        # Normal string
        result = validate_input("Hello World", "string")
        assert result == "Hello World"

        # Email
        result = validate_input("user@example.com", "email")
        assert result == "user@example.com"

        # Threat in strict mode
        with pytest.raises(HTTPException):
            validate_input("' OR '1'='1", "string")


class TestSecurityConfiguration:
    """Test security configuration."""

    def test_security_config_values(self):
        """Test security configuration values."""
        # Check that configuration values are reasonable
        assert SecurityConfig.MAX_STRING_LENGTH > 0
        assert SecurityConfig.MAX_TEXT_LENGTH > SecurityConfig.MAX_STRING_LENGTH
        assert SecurityConfig.MAX_USERNAME_LENGTH > 0
        assert SecurityConfig.MAX_EMAIL_LENGTH > 0
        assert SecurityConfig.MAX_FILENAME_LENGTH > 0

        # Check patterns are compiled
        assert SecurityConfig.USERNAME_PATTERN.pattern
        assert SecurityConfig.EMAIL_PATTERN.pattern
        assert SecurityConfig.FILENAME_PATTERN.pattern

        # Check threat patterns exist
        assert len(SecurityConfig.SQL_INJECTION_PATTERNS) > 0
        assert len(SecurityConfig.XSS_PATTERNS) > 0
        assert len(SecurityConfig.PATH_TRAVERSAL_PATTERNS) > 0

    def test_security_config_patterns(self):
        """Test security configuration patterns."""
        # Test username pattern
        assert SecurityConfig.USERNAME_PATTERN.match("user123")
        assert SecurityConfig.USERNAME_PATTERN.match("user.name_123")
        assert not SecurityConfig.USERNAME_PATTERN.match("user@example")

        # Test email pattern
        assert SecurityConfig.EMAIL_PATTERN.match("user@example.com")
        assert SecurityConfig.EMAIL_PATTERN.match("user.name+tag@example.co.uk")
        assert not SecurityConfig.EMAIL_PATTERN.match("invalid-email")

        # Test filename pattern
        assert SecurityConfig.FILENAME_PATTERN.match("document.pdf")
        assert SecurityConfig.FILENAME_PATTERN.match("file_123.txt")
        assert not SecurityConfig.FILENAME_PATTERN.match("file with spaces.txt")


class TestSecurityValidationIntegration:
    """Test security validation integration scenarios."""

    def test_multiple_threats_detection(self):
        """Test detection of multiple threats in single input."""
        detector = ThreatDetector()

        # Input with multiple threats
        malicious_input = (
            "' OR '1'='1 <script>alert('xss')</script> ../../../etc/passwd"
        )

        assert detector.detect_sql_injection(malicious_input)
        assert detector.detect_xss(malicious_input)
        assert detector.detect_path_traversal(malicious_input)

    def test_validation_performance(self):
        """Test validation performance with large inputs."""
        validator = InputValidator(strict_mode=False)

        # Test with large but valid input
        large_input = "a" * 1000
        result = validator.validate_and_sanitize(large_input, "string")
        assert result == large_input

        # Test with large input containing threats
        large_threat = "normal text " * 100 + "' OR '1'='1"
        with pytest.raises(HTTPException):
            validator = InputValidator(strict_mode=True)
            validator.validate_and_sanitize(large_threat, "string")

    def test_edge_case_inputs(self):
        """Test validation with edge case inputs."""
        validator = InputValidator(strict_mode=False)

        # Empty string
        result = validator.validate_and_sanitize("", "string")
        assert result == ""

        # String with only whitespace
        result = validator.validate_and_sanitize("   ", "string")
        assert result == ""

        # String with unicode characters
        result = validator.validate_and_sanitize("café naïve résumé", "string")
        assert "café" in result

    def test_validation_with_custom_config(self):
        """Test validation with custom configuration."""
        # Test with shorter max length
        validator = InputValidator(strict_mode=False)

        with pytest.raises(ValidationError, match="String too long"):
            validator.validate_and_sanitize("a" * 100, "string", max_length=10)

        # Test with HTML allowed
        result = validator.validate_and_sanitize(
            "<p>Hello World</p>", "text", allow_html=True
        )
        assert result is not None
        assert "Hello World" in result

    def test_concurrent_validation(self):
        """Test validation under concurrent access."""
        import threading
        import time

        validator = InputValidator(strict_mode=False)
        results = []
        errors = []

        def validate_worker():
            try:
                for i in range(10):
                    result = validator.validate_and_sanitize(f"test_{i}", "string")
                    results.append(result)
                    time.sleep(0.001)  # Small delay to encourage concurrency
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=validate_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 iterations
        assert all("test_" in result for result in results)
