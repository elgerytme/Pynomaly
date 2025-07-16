"""Storage credentials value object."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class StorageAuthType(str, Enum):
    """Storage authentication types."""

    ACCESS_KEY = "access_key"
    ROLE_BASED = "role_based"
    ANONYMOUS = "anonymous"
    TOKEN = "token"


class StorageCredentials(BaseModel):
    """Storage credentials value object."""

    auth_type: StorageAuthType
    region: str = Field(default="us-east-1")
    endpoint_url: str | None = None
    credentials: dict[str, Any] = Field(default_factory=dict)

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate region format."""
        if not v or not isinstance(v, str):
            raise ValueError("Region must be a non-empty string")
        return v

    @classmethod
    def create_access_key_credentials(
        cls,
        access_key_id: str,
        secret_access_key: str,
        region: str = "us-east-1",
        session_token: str | None = None,
        endpoint_url: str | None = None,
    ) -> StorageCredentials:
        """Create access key credentials.

        Args:
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region: AWS region
            session_token: Optional session token
            endpoint_url: Optional custom endpoint URL

        Returns:
            Storage credentials instance
        """
        credentials = {
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
        }

        if session_token:
            credentials["session_token"] = session_token

        return cls(
            auth_type=StorageAuthType.ACCESS_KEY,
            region=region,
            endpoint_url=endpoint_url,
            credentials=credentials,
        )

    @classmethod
    def create_role_based_credentials(
        cls,
        role_arn: str,
        region: str = "us-east-1",
        session_name: str | None = None,
        external_id: str | None = None,
        endpoint_url: str | None = None,
    ) -> StorageCredentials:
        """Create role-based credentials.

        Args:
            role_arn: ARN of the IAM role to assume
            region: AWS region
            session_name: Optional session name
            external_id: Optional external ID
            endpoint_url: Optional custom endpoint URL

        Returns:
            Storage credentials instance
        """
        credentials = {
            "role_arn": role_arn,
        }

        if session_name:
            credentials["session_name"] = session_name
        if external_id:
            credentials["external_id"] = external_id

        return cls(
            auth_type=StorageAuthType.ROLE_BASED,
            region=region,
            endpoint_url=endpoint_url,
            credentials=credentials,
        )

    @classmethod
    def create_anonymous_credentials(
        cls, region: str = "us-east-1", endpoint_url: str | None = None
    ) -> StorageCredentials:
        """Create anonymous credentials.

        Args:
            region: AWS region
            endpoint_url: Optional custom endpoint URL

        Returns:
            Storage credentials instance
        """
        return cls(
            auth_type=StorageAuthType.ANONYMOUS,
            region=region,
            endpoint_url=endpoint_url,
            credentials={},
        )

    @classmethod
    def create_token_credentials(
        cls, token: str, region: str = "us-east-1", endpoint_url: str | None = None
    ) -> StorageCredentials:
        """Create token-based credentials.

        Args:
            token: Authentication token
            region: AWS region
            endpoint_url: Optional custom endpoint URL

        Returns:
            Storage credentials instance
        """
        return cls(
            auth_type=StorageAuthType.TOKEN,
            region=region,
            endpoint_url=endpoint_url,
            credentials={"token": token},
        )

    def get_credential_value(self, key: str) -> Any:
        """Get a credential value by key.

        Args:
            key: Credential key

        Returns:
            Credential value or None
        """
        return self.credentials.get(key)

    def has_credential(self, key: str) -> bool:
        """Check if a credential exists.

        Args:
            key: Credential key

        Returns:
            True if credential exists
        """
        return key in self.credentials

    def to_boto3_kwargs(self) -> dict[str, Any]:
        """Convert to boto3 client kwargs.

        Returns:
            Dictionary of boto3 client arguments
        """
        kwargs = {
            "region_name": self.region,
        }

        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url

        if self.auth_type == StorageAuthType.ACCESS_KEY:
            kwargs["aws_access_key_id"] = self.credentials.get("access_key_id")
            kwargs["aws_secret_access_key"] = self.credentials.get("secret_access_key")

            if self.credentials.get("session_token"):
                kwargs["aws_session_token"] = self.credentials.get("session_token")

        elif self.auth_type == StorageAuthType.TOKEN:
            kwargs["aws_session_token"] = self.credentials.get("token")

        return kwargs

    def is_valid(self) -> bool:
        """Check if credentials are valid.

        Returns:
            True if credentials are valid
        """
        if self.auth_type == StorageAuthType.ACCESS_KEY:
            return (
                self.credentials.get("access_key_id") is not None
                and self.credentials.get("secret_access_key") is not None
            )

        elif self.auth_type == StorageAuthType.ROLE_BASED:
            return self.credentials.get("role_arn") is not None

        elif self.auth_type == StorageAuthType.TOKEN:
            return self.credentials.get("token") is not None

        elif self.auth_type == StorageAuthType.ANONYMOUS:
            return True

        return False

    def __str__(self) -> str:
        """String representation."""
        return f"StorageCredentials(auth_type={self.auth_type}, region={self.region})"

    def __repr__(self) -> str:
        """Repr representation."""
        return self.__str__()
