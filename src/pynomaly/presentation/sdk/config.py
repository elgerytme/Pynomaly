"""
SDK Configuration Management

Handles configuration, authentication, and connection settings for the Pynomaly SDK.
Supports multiple authentication methods and environment-based configuration.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for HTTP client behavior."""

    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff_factor: float = 0.3
    retry_on_status: tuple = (408, 429, 500, 502, 503, 504)
    verify_ssl: bool = True
    user_agent: str = "Pynomaly-SDK/1.0.0"
    max_connections: int = 10
    max_connections_per_host: int = 5
    
    # Security hardening options
    enforce_tls: bool = True
    minimum_tls_version: str = "TLSv1.2"
    enable_checksum_validation: bool = True
    enable_client_side_encryption: bool = True
    encryption_key: str | None = None


@dataclass
class SDKConfig:
    """Main SDK configuration class."""

    # Connection settings
    base_url: str = "http://localhost:8000"
    api_version: str = "v1"

    # Authentication
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    token: str | None = None
    auth_type: str = "api_key"  # api_key, basic, bearer, oauth

    # Client configuration
    client: ClientConfig = field(default_factory=ClientConfig)

    # Feature flags
    async_enabled: bool = True
    streaming_enabled: bool = True
    batch_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_requests: bool = False
    log_responses: bool = False

    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds

    @classmethod
    def from_environment(cls) -> "SDKConfig":
        """Create configuration from environment variables."""

        config = cls()

        # Connection
        if base_url := os.getenv("PYNOMALY_BASE_URL"):
            config.base_url = base_url

        if api_version := os.getenv("PYNOMALY_API_VERSION"):
            config.api_version = api_version

        # Authentication
        if api_key := os.getenv("PYNOMALY_API_KEY"):
            config.api_key = api_key
            config.auth_type = "api_key"

        if username := os.getenv("PYNOMALY_USERNAME"):
            config.username = username

        if password := os.getenv("PYNOMALY_PASSWORD"):
            config.password = password
            config.auth_type = "basic"

        if token := os.getenv("PYNOMALY_TOKEN"):
            config.token = token
            config.auth_type = "bearer"

        if auth_type := os.getenv("PYNOMALY_AUTH_TYPE"):
            config.auth_type = auth_type

        # Client settings
        if timeout := os.getenv("PYNOMALY_TIMEOUT"):
            try:
                config.client.timeout = float(timeout)
            except ValueError:
                logger.warning(f"Invalid timeout value: {timeout}")

        if verify_ssl := os.getenv("PYNOMALY_VERIFY_SSL"):
            config.client.verify_ssl = verify_ssl.lower() in ("true", "1", "yes")

        # Feature flags
        if async_enabled := os.getenv("PYNOMALY_ASYNC_ENABLED"):
            config.async_enabled = async_enabled.lower() in ("true", "1", "yes")

        # Logging
        if log_level := os.getenv("PYNOMALY_LOG_LEVEL"):
            config.log_level = log_level.upper()

        if log_requests := os.getenv("PYNOMALY_LOG_REQUESTS"):
            config.log_requests = log_requests.lower() in ("true", "1", "yes")

        return config

    @classmethod
    def from_file(cls, config_path: str | Path) -> "SDKConfig":
        """Load configuration from a JSON file."""

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                config_data = json.load(f)

            # Create client config if present
            client_data = config_data.pop("client", {})
            client_config = ClientConfig(**client_data)

            # Create main config
            config = cls(**config_data)
            config.client = client_config

            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except TypeError as e:
            raise ValueError(f"Invalid configuration format: {e}")

    def to_file(self, config_path: str | Path) -> None:
        """Save configuration to a JSON file."""

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling dataclass fields
        config_data = {
            "base_url": self.base_url,
            "api_version": self.api_version,
            "api_key": self.api_key,
            "username": self.username,
            "password": self.password,
            "token": self.token,
            "auth_type": self.auth_type,
            "async_enabled": self.async_enabled,
            "streaming_enabled": self.streaming_enabled,
            "batch_enabled": self.batch_enabled,
            "log_level": self.log_level,
            "log_requests": self.log_requests,
            "log_responses": self.log_responses,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "client": {
                "timeout": self.client.timeout,
                "max_retries": self.client.max_retries,
                "retry_backoff_factor": self.client.retry_backoff_factor,
                "retry_on_status": list(self.client.retry_on_status),
                "verify_ssl": self.client.verify_ssl,
                "user_agent": self.client.user_agent,
                "max_connections": self.client.max_connections,
                "max_connections_per_host": self.client.max_connections_per_host,
            },
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    @property
    def api_base_url(self) -> str:
        """Get the full API base URL."""
        return f"{self.base_url.rstrip('/')}/api/{self.api_version}"

    def get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers based on auth type."""

        headers = {}

        if self.auth_type == "api_key" and self.api_key:
            headers["X-API-Key"] = self.api_key

        elif self.auth_type == "bearer" and self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        elif self.auth_type == "basic" and self.username and self.password:
            import base64

            credentials = base64.b64encode(
                f"{self.username}:{self.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"

        return headers

    def validate(self) -> None:
        """Validate the configuration."""

        if not self.base_url:
            raise ValueError("base_url is required")

        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")

        if self.auth_type == "api_key" and not self.api_key:
            raise ValueError("api_key is required when auth_type is 'api_key'")

        elif self.auth_type == "basic" and (not self.username or not self.password):
            raise ValueError(
                "username and password are required when auth_type is 'basic'"
            )

        elif self.auth_type == "bearer" and not self.token:
            raise ValueError("token is required when auth_type is 'bearer'")

        if self.client.timeout <= 0:
            raise ValueError("timeout must be positive")

        if self.client.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def setup_logging(self) -> None:
        """Configure logging for the SDK."""

        level = getattr(logging, self.log_level.upper(), logging.INFO)

        # Configure SDK logger
        sdk_logger = logging.getLogger("pynomaly.presentation.sdk")
        sdk_logger.setLevel(level)

        # Add console handler if none exists
        if not sdk_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            sdk_logger.addHandler(handler)


def get_default_config_path() -> Path:
    """Get the default configuration file path."""

    # Try user config directory first
    if config_dir := os.getenv("XDG_CONFIG_HOME"):
        return Path(config_dir) / "pynomaly" / "config.json"

    # Fall back to home directory
    home = Path.home()
    return home / ".pynomaly" / "config.json"


def load_config(config_path: str | Path | None = None) -> SDKConfig:
    """Load configuration from file or environment."""

    # Try to load from file first
    if config_path:
        return SDKConfig.from_file(config_path)

    # Try default config file location
    default_path = get_default_config_path()
    if default_path.exists():
        return SDKConfig.from_file(default_path)

    # Fall back to environment variables
    return SDKConfig.from_environment()
