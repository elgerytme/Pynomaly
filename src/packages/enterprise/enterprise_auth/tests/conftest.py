"""
Pytest configuration for enterprise authentication testing.
Provides fixtures for security testing, user management, and authentication flows.
"""

import pytest
import time
from typing import Dict, Any, List
from unittest.mock import Mock
import secrets
import string


@pytest.fixture
def valid_credentials() -> Dict[str, str]:
    """Valid test credentials."""
    return {
        'username': 'admin',
        'password': 'correct_password',
        'email': 'admin@company.com'
    }


@pytest.fixture
def invalid_credentials() -> List[Dict[str, str]]:
    """Collection of invalid credentials for testing."""
    return [
        {'username': 'admin', 'password': 'wrong_password'},
        {'username': 'nonexistent', 'password': 'any_password'},
        {'username': '', 'password': 'correct_password'},
        {'username': 'admin', 'password': ''},
        {'username': None, 'password': 'correct_password'},
        {'username': 'admin', 'password': None},
    ]


@pytest.fixture
def mock_user_database():
    """Mock user database for testing."""
    return {
        'admin': {
            'user_id': 'user123',
            'username': 'admin',
            'email': 'admin@company.com',
            'password_hash': 'hashed_correct_password',
            'roles': ['admin', 'user'],
            'is_active': True,
            'failed_attempts': 0,
            'locked_until': None
        },
        'regular_user': {
            'user_id': 'user456',
            'username': 'regular_user',
            'email': 'user@company.com',
            'password_hash': 'hashed_user_password',
            'roles': ['user'],
            'is_active': True,
            'failed_attempts': 0,
            'locked_until': None
        }
    }


@pytest.fixture
def security_test_payloads():
    """Security test payloads for injection testing."""
    return {
        'sql_injection': [
            "admin'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "admin'/**/OR/**/1=1#",
        ],
        'xss': [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
        ],
        'path_traversal': [
            "../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2froot",
        ],
        'command_injection': [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& ping -c 10 127.0.0.1",
        ]
    }


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return Timer()


@pytest.fixture
def security_config():
    """Security configuration for testing."""
    return {
        'max_failed_attempts': 5,
        'lockout_duration_minutes': 15,
        'session_timeout_minutes': 30,
        'token_length': 32,
        'password_min_length': 8,
        'require_mfa': False,  # Disabled for testing
        'audit_logging': True
    }


def pytest_configure(config):
    """Configure pytest markers for security testing."""
    markers = [
        "security: Security-focused tests",
        "auth: Authentication tests",
        "injection: Injection attack prevention tests",
        "brute_force: Brute force protection tests",
        "rbac: Role-based access control tests",
        "crypto: Cryptographic security tests",
        "compliance: Security compliance tests",
        "performance: Security performance tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)