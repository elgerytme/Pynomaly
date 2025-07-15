"""Application layer service protocols."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ApplicationConfigProtocol(Protocol):
    """Protocol for application configuration."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...

    def get_section(self, section: str) -> dict[str, Any]:
        """Get configuration section."""
        ...

    def has_feature_flag(self, flag_name: str) -> bool:
        """Check if feature flag is enabled."""
        ...


@runtime_checkable
class ApplicationCacheProtocol(Protocol):
    """Protocol for application caching."""

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        ...

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        ...

    def clear(self) -> None:
        """Clear all cache entries."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


@runtime_checkable
class ApplicationMetricsProtocol(Protocol):
    """Protocol for application metrics."""

    def increment_counter(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        ...

    def record_gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a gauge metric."""
        ...

    def record_histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a histogram metric."""
        ...

    def start_timer(self, name: str, tags: dict[str, str] | None = None) -> Any:
        """Start a timer for duration tracking."""
        ...

    def record_duration(self, name: str, duration_ms: float, tags: dict[str, str] | None = None) -> None:
        """Record a duration metric."""
        ...


@runtime_checkable
class ApplicationSecurityProtocol(Protocol):
    """Protocol for application security operations."""

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data."""
        ...

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        ...

    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        ...

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        ...

    def generate_token(self, payload: dict[str, Any]) -> str:
        """Generate a secure token."""
        ...

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify and decode a token."""
        ...


@runtime_checkable
class ApplicationLoggingProtocol(Protocol):
    """Protocol for application logging."""

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        ...

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        ...


@runtime_checkable
class ApplicationNotificationProtocol(Protocol):
    """Protocol for application notifications."""

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None
    ) -> bool:
        """Send email notification."""
        ...

    def send_sms(self, to: str, message: str) -> bool:
        """Send SMS notification."""
        ...

    def send_webhook(self, url: str, payload: dict[str, Any]) -> bool:
        """Send webhook notification."""
        ...


@runtime_checkable
class ApplicationEventBusProtocol(Protocol):
    """Protocol for application event bus."""

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish an event."""
        ...

    def subscribe(self, event_type: str, handler: Any) -> None:
        """Subscribe to an event type."""
        ...

    def unsubscribe(self, event_type: str, handler: Any) -> None:
        """Unsubscribe from an event type."""
        ...
