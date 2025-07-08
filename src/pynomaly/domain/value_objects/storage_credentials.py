"""Storage credentials value object for cloud storage authentication."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class AuthenticationType(Enum):
    """Supported authentication types for cloud storage."""
    
    ACCESS_KEY = "access_key"
    TOKEN = "token"
    SERVICE_ACCOUNT = "service_account"
    ROLE_BASED = "role_based"
    OAUTH2 = "oauth2"
    SAS_TOKEN = "sas_token"
    ANONYMOUS = "anonymous"


@dataclass(frozen=True)
class StorageCredentials:
    """Value object encapsulating storage authentication credentials.
    
    This value object provides a provider-agnostic way to encapsulate
    authentication credentials for cloud storage systems while maintaining
    domain boundary integrity.
    
    Attributes:
        auth_type: Type of authentication mechanism
        credentials: Dictionary containing authentication data
        region: Optional region/zone identifier
        endpoint_url: Optional custom endpoint URL
        timeout_seconds: Connection timeout in seconds
        max_retry_attempts: Maximum number of retry attempts
        verify_ssl: Whether to verify SSL certificates
    """
    
    auth_type: AuthenticationType
    credentials: Dict[str, Any]
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    timeout_seconds: int = 30
    max_retry_attempts: int = 3
    verify_ssl: bool = True
    
    def __post_init__(self) -> None:
        """Validate credentials configuration."""
        if not isinstance(self.auth_type, AuthenticationType):
            raise TypeError(
                f"auth_type must be AuthenticationType enum, got {type(self.auth_type)}"
            )
        
        if not isinstance(self.credentials, dict):
            raise TypeError(
                f"credentials must be a dictionary, got {type(self.credentials)}"
            )
        
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        
        if self.max_retry_attempts < 0:
            raise ValueError(
                f"max_retry_attempts must be non-negative, got {self.max_retry_attempts}"
            )
        
        # Validate required fields based on auth type
        self._validate_auth_type_requirements()
    
    def _validate_auth_type_requirements(self) -> None:
        """Validate that required fields are present for the authentication type."""
        required_fields = {
            AuthenticationType.ACCESS_KEY: ["access_key_id", "secret_access_key"],
            AuthenticationType.TOKEN: ["token"],
            AuthenticationType.SERVICE_ACCOUNT: ["service_account_key"],
            AuthenticationType.ROLE_BASED: ["role_arn"],
            AuthenticationType.OAUTH2: ["client_id", "client_secret"],
            AuthenticationType.SAS_TOKEN: ["sas_token"],
            AuthenticationType.ANONYMOUS: [],
        }
        
        required = required_fields.get(self.auth_type, [])
        missing = [field for field in required if field not in self.credentials]
        
        if missing:
            raise ValueError(
                f"Missing required credentials for {self.auth_type.value}: {missing}"
            )
    
    @classmethod
    def create_access_key_credentials(
        cls,
        access_key_id: str,
        secret_access_key: str,
        region: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> StorageCredentials:
        """Create credentials using access key authentication.
        
        Args:
            access_key_id: Access key identifier
            secret_access_key: Secret access key
            region: Optional region
            session_token: Optional session token for temporary credentials
            
        Returns:
            StorageCredentials instance for access key authentication
        """
        credentials = {
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
        }
        
        if session_token:
            credentials["session_token"] = session_token
        
        return cls(
            auth_type=AuthenticationType.ACCESS_KEY,
            credentials=credentials,
            region=region,
        )
    
    @classmethod
    def create_token_credentials(
        cls,
        token: str,
        region: Optional[str] = None,
    ) -> StorageCredentials:
        """Create credentials using token authentication.
        
        Args:
            token: Authentication token
            region: Optional region
            
        Returns:
            StorageCredentials instance for token authentication
        """
        return cls(
            auth_type=AuthenticationType.TOKEN,
            credentials={"token": token},
            region=region,
        )
    
    @classmethod
    def create_service_account_credentials(
        cls,
        service_account_key: str,
        region: Optional[str] = None,
    ) -> StorageCredentials:
        """Create credentials using service account authentication.
        
        Args:
            service_account_key: Service account key (JSON string or file path)
            region: Optional region
            
        Returns:
            StorageCredentials instance for service account authentication
        """
        return cls(
            auth_type=AuthenticationType.SERVICE_ACCOUNT,
            credentials={"service_account_key": service_account_key},
            region=region,
        )
    
    @classmethod
    def create_role_based_credentials(
        cls,
        role_arn: str,
        region: Optional[str] = None,
        session_name: Optional[str] = None,
    ) -> StorageCredentials:
        """Create credentials using role-based authentication.
        
        Args:
            role_arn: Role ARN to assume
            region: Optional region
            session_name: Optional session name
            
        Returns:
            StorageCredentials instance for role-based authentication
        """
        credentials = {"role_arn": role_arn}
        
        if session_name:
            credentials["session_name"] = session_name
        
        return cls(
            auth_type=AuthenticationType.ROLE_BASED,
            credentials=credentials,
            region=region,
        )
    
    @classmethod
    def create_oauth2_credentials(
        cls,
        client_id: str,
        client_secret: str,
        region: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> StorageCredentials:
        """Create credentials using OAuth2 authentication.
        
        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            region: Optional region
            scope: Optional OAuth2 scope
            
        Returns:
            StorageCredentials instance for OAuth2 authentication
        """
        credentials = {
            "client_id": client_id,
            "client_secret": client_secret,
        }
        
        if scope:
            credentials["scope"] = scope
        
        return cls(
            auth_type=AuthenticationType.OAUTH2,
            credentials=credentials,
            region=region,
        )
    
    @classmethod
    def create_sas_token_credentials(
        cls,
        sas_token: str,
        region: Optional[str] = None,
    ) -> StorageCredentials:
        """Create credentials using SAS token authentication.
        
        Args:
            sas_token: Shared Access Signature token
            region: Optional region
            
        Returns:
            StorageCredentials instance for SAS token authentication
        """
        return cls(
            auth_type=AuthenticationType.SAS_TOKEN,
            credentials={"sas_token": sas_token},
            region=region,
        )
    
    @classmethod
    def create_anonymous_credentials(
        cls,
        region: Optional[str] = None,
    ) -> StorageCredentials:
        """Create credentials for anonymous access.
        
        Args:
            region: Optional region
            
        Returns:
            StorageCredentials instance for anonymous access
        """
        return cls(
            auth_type=AuthenticationType.ANONYMOUS,
            credentials={},
            region=region,
        )
    
    def get_credential_value(self, key: str) -> Any:
        """Get a credential value by key.
        
        Args:
            key: Credential key
            
        Returns:
            Credential value
            
        Raises:
            KeyError: If key is not found
        """
        return self.credentials[key]
    
    def has_credential(self, key: str) -> bool:
        """Check if a credential key exists.
        
        Args:
            key: Credential key
            
        Returns:
            True if key exists
        """
        return key in self.credentials
    
    def is_temporary(self) -> bool:
        """Check if credentials are temporary (have session token).
        
        Returns:
            True if credentials are temporary
        """
        return self.has_credential("session_token")
    
    def is_anonymous(self) -> bool:
        """Check if credentials are for anonymous access.
        
        Returns:
            True if credentials are anonymous
        """
        return self.auth_type == AuthenticationType.ANONYMOUS
    
    def with_timeout(self, timeout_seconds: int) -> StorageCredentials:
        """Create a copy with different timeout.
        
        Args:
            timeout_seconds: New timeout value
            
        Returns:
            New StorageCredentials instance with updated timeout
        """
        return StorageCredentials(
            auth_type=self.auth_type,
            credentials=self.credentials,
            region=self.region,
            endpoint_url=self.endpoint_url,
            timeout_seconds=timeout_seconds,
            max_retry_attempts=self.max_retry_attempts,
            verify_ssl=self.verify_ssl,
        )
    
    def with_retries(self, max_retry_attempts: int) -> StorageCredentials:
        """Create a copy with different retry configuration.
        
        Args:
            max_retry_attempts: New maximum retry attempts
            
        Returns:
            New StorageCredentials instance with updated retry config
        """
        return StorageCredentials(
            auth_type=self.auth_type,
            credentials=self.credentials,
            region=self.region,
            endpoint_url=self.endpoint_url,
            timeout_seconds=self.timeout_seconds,
            max_retry_attempts=max_retry_attempts,
            verify_ssl=self.verify_ssl,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (without sensitive data).
        
        Returns:
            Dictionary representation without sensitive credential values
        """
        return {
            "auth_type": self.auth_type.value,
            "region": self.region,
            "endpoint_url": self.endpoint_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retry_attempts": self.max_retry_attempts,
            "verify_ssl": self.verify_ssl,
            "has_credentials": len(self.credentials) > 0,
            "is_temporary": self.is_temporary(),
            "is_anonymous": self.is_anonymous(),
        }
    
    def __str__(self) -> str:
        """Human-readable representation (without sensitive data)."""
        return (
            f"StorageCredentials({self.auth_type.value}, "
            f"region={self.region}, anonymous={self.is_anonymous()})"
        )
