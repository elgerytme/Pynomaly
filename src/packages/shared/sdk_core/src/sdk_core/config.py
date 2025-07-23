"""Configuration management for SDK clients."""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from urllib.parse import urljoin


class Environment(Enum):
    """Predefined environments."""
    
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"


@dataclass
class ClientConfig:
    """Configuration for SDK clients."""
    
    # Base configuration
    base_url: str
    api_version: str = "v1"
    timeout: float = 30.0
    
    # Authentication
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    retry_on_status: tuple = (429, 500, 502, 503, 504)
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # HTTP client settings
    verify_ssl: bool = True
    connection_pool_size: int = 10
    max_keepalive_connections: int = 5
    
    # Headers
    user_agent: str = "Platform-SDK/0.1.0"
    default_headers: Dict[str, str] = field(default_factory=dict)
    
    # Logging
    log_requests: bool = False
    log_responses: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure base_url ends with /
        if not self.base_url.endswith("/"):
            self.base_url += "/"
            
        # Add default headers
        if "User-Agent" not in self.default_headers:
            self.default_headers["User-Agent"] = self.user_agent
    
    @property
    def api_base_url(self) -> str:
        """Get the base URL for API endpoints."""
        return urljoin(self.base_url, f"api/{self.api_version}/")
    
    @classmethod
    def for_environment(cls, env: Environment, **kwargs) -> "ClientConfig":
        """Create configuration for a specific environment."""
        
        env_configs = {
            Environment.LOCAL: {
                "base_url": "http://localhost:8000",
                "verify_ssl": False,
                "log_requests": True,
                "log_responses": True,
            },
            Environment.DEVELOPMENT: {
                "base_url": "https://dev-api.platform.com",
                "log_requests": True,
            },
            Environment.STAGING: {
                "base_url": "https://staging-api.platform.com",
            },
            Environment.PRODUCTION: {
                "base_url": "https://api.platform.com",
                "log_requests": False,
                "log_responses": False,
            },
        }
        
        config = env_configs.get(env, {})
        config.update(kwargs)
        
        return cls(**config)
    
    def with_auth(self, api_key: Optional[str] = None, jwt_token: Optional[str] = None) -> "ClientConfig":
        """Create a new config with authentication credentials."""
        return ClientConfig(
            **{
                **self.__dict__,
                "api_key": api_key or self.api_key,
                "jwt_token": jwt_token or self.jwt_token,
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientConfig":
        """Create config from dictionary."""
        return cls(**data)