"""Storage credentials value object."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any


class AuthenticationType(Enum):
    """Authentication types for storage credentials."""
    
    ACCESS_KEY = "access_key"
    SERVICE_ACCOUNT = "service_account"
    ROLE_BASED = "role_based"
    SAS_TOKEN = "sas_token"
    CONNECTION_STRING = "connection_string"
    TOKEN = "token"
    OAUTH2 = "oauth2"


@dataclass(frozen=True)
class StorageCredentials:
    """Storage credentials for cloud storage authentication.
    
    This value object encapsulates authentication information needed to
    access various cloud storage services.
    
    Attributes:
        auth_type: Type of authentication mechanism
        region: Cloud region/location
        credentials: Authentication credentials dictionary
        endpoint_url: Optional custom endpoint URL
        timeout: Connection timeout in seconds
        retry_config: Retry configuration
    """
    
    auth_type: AuthenticationType
    region: str
    credentials: Dict[str, Any]
    endpoint_url: Optional[str] = None
    timeout: int = 30
    retry_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate credentials."""
        if not isinstance(self.auth_type, AuthenticationType):
            raise TypeError(f"auth_type must be AuthenticationType, got {type(self.auth_type)}")
        
        if not self.region:
            raise ValueError("region cannot be empty")
        
        if not isinstance(self.credentials, dict):
            raise TypeError(f"credentials must be dict, got {type(self.credentials)}")
        
        if not self.credentials:
            raise ValueError("credentials cannot be empty")
        
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        
        # Validate auth_type specific requirements
        self._validate_auth_type_requirements()
    
    def _validate_auth_type_requirements(self) -> None:
        """Validate auth_type specific credential requirements."""
        if self.auth_type == AuthenticationType.ACCESS_KEY:
            required_keys = ['access_key_id', 'secret_access_key']
            for key in required_keys:
                if key not in self.credentials:
                    raise ValueError(f"Missing required credential: {key}")
        
        elif self.auth_type == AuthenticationType.SERVICE_ACCOUNT:
            required_keys = ['service_account_key']
            for key in required_keys:
                if key not in self.credentials:
                    raise ValueError(f"Missing required credential: {key}")
        
        elif self.auth_type == AuthenticationType.SAS_TOKEN:
            required_keys = ['sas_token']
            for key in required_keys:
                if key not in self.credentials:
                    raise ValueError(f"Missing required credential: {key}")
        
        elif self.auth_type == AuthenticationType.CONNECTION_STRING:
            required_keys = ['connection_string']
            for key in required_keys:
                if key not in self.credentials:
                    raise ValueError(f"Missing required credential: {key}")
        
        elif self.auth_type == AuthenticationType.TOKEN:
            required_keys = ['token']
            for key in required_keys:
                if key not in self.credentials:
                    raise ValueError(f"Missing required credential: {key}")
        
        elif self.auth_type == AuthenticationType.OAUTH2:
            required_keys = ['client_id', 'client_secret']
            for key in required_keys:
                if key not in self.credentials:
                    raise ValueError(f"Missing required credential: {key}")
    
    @classmethod
    def create_access_key_credentials(
        cls,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ) -> StorageCredentials:
        """Create access key based credentials.
        
        Args:
            access_key_id: Access key ID
            secret_access_key: Secret access key
            region: Cloud region
            session_token: Optional session token
            endpoint_url: Optional custom endpoint URL
            
        Returns:
            StorageCredentials instance
        """
        credentials = {
            'access_key_id': access_key_id,
            'secret_access_key': secret_access_key
        }
        
        if session_token:
            credentials['session_token'] = session_token
        
        return cls(
            auth_type=AuthenticationType.ACCESS_KEY,
            region=region,
            credentials=credentials,
            endpoint_url=endpoint_url
        )
    
    @classmethod
    def create_service_account_credentials(
        cls,
        service_account_key: str,
        region: str,
        project_id: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ) -> StorageCredentials:
        """Create service account based credentials.
        
        Args:
            service_account_key: Service account key (JSON or file path)
            region: Cloud region
            project_id: Optional project ID
            endpoint_url: Optional custom endpoint URL
            
        Returns:
            StorageCredentials instance
        """
        credentials = {
            'service_account_key': service_account_key
        }
        
        if project_id:
            credentials['project_id'] = project_id
        
        return cls(
            auth_type=AuthenticationType.SERVICE_ACCOUNT,
            region=region,
            credentials=credentials,
            endpoint_url=endpoint_url
        )
    
    @classmethod
    def create_role_based_credentials(
        cls,
        region: str,
        role_arn: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ) -> StorageCredentials:
        """Create role-based credentials.
        
        Args:
            region: Cloud region
            role_arn: Optional role ARN
            endpoint_url: Optional custom endpoint URL
            
        Returns:
            StorageCredentials instance
        """
        credentials = {}
        
        if role_arn:
            credentials['role_arn'] = role_arn
        
        return cls(
            auth_type=AuthenticationType.ROLE_BASED,
            region=region,
            credentials=credentials or {'role_based': True},
            endpoint_url=endpoint_url
        )
    
    @classmethod
    def create_sas_token_credentials(
        cls,
        sas_token: str,
        region: str,
        account_name: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ) -> StorageCredentials:
        """Create SAS token based credentials.
        
        Args:
            sas_token: SAS token
            region: Cloud region
            account_name: Optional account name
            endpoint_url: Optional custom endpoint URL
            
        Returns:
            StorageCredentials instance
        """
        credentials = {
            'sas_token': sas_token
        }
        
        if account_name:
            credentials['account_name'] = account_name
        
        return cls(
            auth_type=AuthenticationType.SAS_TOKEN,
            region=region,
            credentials=credentials,
            endpoint_url=endpoint_url
        )
    
    @classmethod
    def create_connection_string_credentials(
        cls,
        connection_string: str,
        region: str,
        endpoint_url: Optional[str] = None
    ) -> StorageCredentials:
        """Create connection string based credentials.
        
        Args:
            connection_string: Connection string
            region: Cloud region
            endpoint_url: Optional custom endpoint URL
            
        Returns:
            StorageCredentials instance
        """
        credentials = {
            'connection_string': connection_string
        }
        
        return cls(
            auth_type=AuthenticationType.CONNECTION_STRING,
            region=region,
            credentials=credentials,
            endpoint_url=endpoint_url
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of credentials
        """
        return {
            'auth_type': self.auth_type.value,
            'region': self.region,
            'credentials': self.credentials.copy(),
            'endpoint_url': self.endpoint_url,
            'timeout': self.timeout,
            'retry_config': self.retry_config
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"StorageCredentials(auth_type={self.auth_type.value}, region={self.region})"
