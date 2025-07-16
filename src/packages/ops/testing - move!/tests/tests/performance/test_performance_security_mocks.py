"""
Mock implementations for performance and security testing.
"""


class MockThreatDetector:
    """Mock threat detector for testing."""

    def analyze_input(self, input_data: str) -> float:
        """Analyze input and return threat level."""
        # Simple threat detection based on known patterns
        threat_level = 0.0

        if any(
            pattern in input_data.lower()
            for pattern in ["drop", "delete", "insert", "update"]
        ):
            threat_level += 0.8

        if any(
            pattern in input_data.lower()
            for pattern in ["<script>", "javascript:", "onerror"]
        ):
            threat_level += 0.7

        if any(pattern in input_data for pattern in [";", "&", "|", "`"]):
            threat_level += 0.9

        if "../" in input_data or "..\\" in input_data:
            threat_level += 0.8

        return min(threat_level, 1.0)


class MockInputSanitizer:
    """Mock input sanitizer for testing."""

    def sanitize_string(self, input_string: str) -> str:
        """Sanitize string input."""
        # Remove/escape dangerous characters
        sanitized = input_string.replace("'", "''")
        sanitized = sanitized.replace(";", "")
        sanitized = sanitized.replace("--", "")
        return sanitized

    def sanitize_html(self, html_input: str) -> str:
        """Sanitize HTML input."""
        # Remove script tags and event handlers
        sanitized = html_input.replace("<script>", "&lt;script&gt;")
        sanitized = sanitized.replace("</script>", "&lt;/script&gt;")
        sanitized = sanitized.replace("onerror=", "data-onerror=")
        sanitized = sanitized.replace("onload=", "data-onload=")
        sanitized = sanitized.replace("javascript:", "data-javascript:")
        return sanitized

    def sanitize_command(self, command: str) -> str:
        """Sanitize command input."""
        # Remove dangerous characters
        dangerous_chars = [";", "&", "|", "`", "$", "(", ")"]
        sanitized = command
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        return sanitized

    def sanitize_path(self, path: str) -> str:
        """Sanitize file path input."""
        # Remove path traversal sequences
        sanitized = path.replace("../", "")
        sanitized = sanitized.replace("..\\", "")
        sanitized = sanitized.replace("%2e%2e", "")
        return sanitized

    def validate_input(self, input_type: str, value: str) -> bool:
        """Validate input based on type."""
        if input_type == "email":
            return "@" in value and "." in value and "<" not in value
        elif input_type == "phone":
            return any(char.isdigit() for char in value) and ";" not in value
        elif input_type == "url":
            return (
                value.startswith(("http://", "https://")) and "javascript:" not in value
            )
        return True


class MockJWTAuthService:
    """Mock JWT authentication service."""

    def __init__(self):
        self.secret_key = "test-secret-key"
        self.algorithm = "HS256"

    def create_user(self, username: str, email: str, password: str, roles: list = None):
        """Create a mock user."""
        return {
            "username": username,
            "email": email,
            "roles": roles or ["user"],
            "id": "test-user-id",
        }

    def create_access_token(
        self, user: dict, ip_address: str = None, user_agent: str = None
    ):
        """Create a mock access token."""
        return {
            "access_token": "mock-access-token",
            "refresh_token": "mock-refresh-token",
            "token_type": "bearer",
            "expires_in": 3600,
        }


# Mock classes for domain entities
class MockDetector:
    """Mock detector for testing."""

    def __init__(
        self,
        id: str,
        name: str,
        algorithm: str,
        parameters: dict,
        trained: bool = False,
    ):
        self.id = id
        self.name = name
        self.algorithm = algorithm
        self.parameters = parameters
        self.trained = trained


class MockAnomalyScore:
    """Mock anomaly score for testing."""

    def __init__(self, value: float):
        self.value = value


class MockPerformanceMetrics:
    """Mock performance metrics for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Patch the imports in the test files
def patch_imports():
    """Patch imports for testing."""
    import sys
    from types import ModuleType

    # Create mock modules
    mock_threat_detection = ModuleType(
        "monorepo.infrastructure.security.threat_detection"
    )
    mock_threat_detection.ThreatDetector = MockThreatDetector

    mock_input_validation = ModuleType(
        "monorepo.infrastructure.security.input_validation"
    )
    mock_input_validation.InputSanitizer = MockInputSanitizer

    mock_jwt_auth = ModuleType("monorepo.infrastructure.auth.jwt_auth")
    mock_jwt_auth.JWTAuthService = MockJWTAuthService

    mock_detector = ModuleType("monorepo.domain.entities.detector")
    mock_detector.Detector = MockDetector

    mock_anomaly_score = ModuleType("monorepo.domain.value_objects.anomaly_score")
    mock_anomaly_score.AnomalyScore = MockAnomalyScore

    mock_performance_metrics = ModuleType(
        "monorepo.domain.value_objects.performance_metrics"
    )
    mock_performance_metrics.PerformanceMetrics = MockPerformanceMetrics

    # Add to sys.modules
    sys.modules["monorepo.infrastructure.security.threat_detection"] = (
        mock_threat_detection
    )
    sys.modules["monorepo.infrastructure.security.input_validation"] = (
        mock_input_validation
    )
    sys.modules["monorepo.infrastructure.auth.jwt_auth"] = mock_jwt_auth
    sys.modules["monorepo.domain.entities.detector"] = mock_detector
    sys.modules["monorepo.domain.value_objects.anomaly_score"] = mock_anomaly_score
    sys.modules["monorepo.domain.value_objects.performance_metrics"] = (
        mock_performance_metrics
    )


# Call patch_imports when this module is imported
patch_imports()
