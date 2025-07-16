"""Tests for StorageCredentials value object."""

import pytest
from pydantic import ValidationError

from monorepo.domain.value_objects.storage_credentials import (
    StorageAuthType,
    StorageCredentials,
)


class TestStorageAuthType:
    """Test suite for StorageAuthType enum."""

    def test_enum_values(self):
        """Test StorageAuthType enum values."""
        assert StorageAuthType.ACCESS_KEY == "access_key"
        assert StorageAuthType.ROLE_BASED == "role_based"
        assert StorageAuthType.ANONYMOUS == "anonymous"
        assert StorageAuthType.TOKEN == "token"

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        values = [item.value for item in StorageAuthType]
        expected = ["access_key", "role_based", "anonymous", "token"]
        assert values == expected

    def test_enum_membership(self):
        """Test enum membership."""
        assert "access_key" in StorageAuthType
        assert "invalid_type" not in StorageAuthType


class TestStorageCredentials:
    """Test suite for StorageCredentials value object."""

    def test_basic_creation(self):
        """Test basic StorageCredentials creation."""
        credentials = StorageCredentials(
            auth_type=StorageAuthType.ACCESS_KEY, region="us-west-2"
        )

        assert credentials.auth_type == StorageAuthType.ACCESS_KEY
        assert credentials.region == "us-west-2"
        assert credentials.endpoint_url is None
        assert credentials.credentials == {}

    def test_creation_with_all_fields(self):
        """Test StorageCredentials creation with all fields."""
        creds_dict = {"access_key_id": "AKIATEST", "secret_access_key": "secret123"}

        credentials = StorageCredentials(
            auth_type=StorageAuthType.ACCESS_KEY,
            region="eu-west-1",
            endpoint_url="https://custom.endpoint.com",
            credentials=creds_dict,
        )

        assert credentials.auth_type == StorageAuthType.ACCESS_KEY
        assert credentials.region == "eu-west-1"
        assert credentials.endpoint_url == "https://custom.endpoint.com"
        assert credentials.credentials == creds_dict

    def test_default_region(self):
        """Test default region value."""
        credentials = StorageCredentials(auth_type=StorageAuthType.ANONYMOUS)
        assert credentials.region == "us-east-1"

    def test_region_validation_empty(self):
        """Test region validation with empty string."""
        with pytest.raises(ValidationError, match="Region must be a non-empty string"):
            StorageCredentials(auth_type=StorageAuthType.ANONYMOUS, region="")

    def test_region_validation_none(self):
        """Test region validation with None."""
        with pytest.raises(ValidationError):
            StorageCredentials(auth_type=StorageAuthType.ANONYMOUS, region=None)

    def test_create_access_key_credentials_basic(self):
        """Test creating access key credentials with basic parameters."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123", secret_access_key="secret123key"
        )

        assert credentials.auth_type == StorageAuthType.ACCESS_KEY
        assert credentials.region == "us-east-1"
        assert credentials.endpoint_url is None
        assert credentials.credentials["access_key_id"] == "AKIATEST123"
        assert credentials.credentials["secret_access_key"] == "secret123key"
        assert "session_token" not in credentials.credentials

    def test_create_access_key_credentials_full(self):
        """Test creating access key credentials with all parameters."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123",
            secret_access_key="secret123key",
            region="ap-southeast-2",
            session_token="session123",
            endpoint_url="https://s3.custom.com",
        )

        assert credentials.auth_type == StorageAuthType.ACCESS_KEY
        assert credentials.region == "ap-southeast-2"
        assert credentials.endpoint_url == "https://s3.custom.com"
        assert credentials.credentials["access_key_id"] == "AKIATEST123"
        assert credentials.credentials["secret_access_key"] == "secret123key"
        assert credentials.credentials["session_token"] == "session123"

    def test_create_role_based_credentials_basic(self):
        """Test creating role-based credentials with basic parameters."""
        role_arn = "arn:aws:iam::123456789012:role/TestRole"

        credentials = StorageCredentials.create_role_based_credentials(
            role_arn=role_arn
        )

        assert credentials.auth_type == StorageAuthType.ROLE_BASED
        assert credentials.region == "us-east-1"
        assert credentials.endpoint_url is None
        assert credentials.credentials["role_arn"] == role_arn
        assert "session_name" not in credentials.credentials
        assert "external_id" not in credentials.credentials

    def test_create_role_based_credentials_full(self):
        """Test creating role-based credentials with all parameters."""
        role_arn = "arn:aws:iam::123456789012:role/TestRole"

        credentials = StorageCredentials.create_role_based_credentials(
            role_arn=role_arn,
            region="eu-central-1",
            session_name="test-session",
            external_id="external123",
            endpoint_url="https://sts.custom.com",
        )

        assert credentials.auth_type == StorageAuthType.ROLE_BASED
        assert credentials.region == "eu-central-1"
        assert credentials.endpoint_url == "https://sts.custom.com"
        assert credentials.credentials["role_arn"] == role_arn
        assert credentials.credentials["session_name"] == "test-session"
        assert credentials.credentials["external_id"] == "external123"

    def test_create_anonymous_credentials_basic(self):
        """Test creating anonymous credentials with basic parameters."""
        credentials = StorageCredentials.create_anonymous_credentials()

        assert credentials.auth_type == StorageAuthType.ANONYMOUS
        assert credentials.region == "us-east-1"
        assert credentials.endpoint_url is None
        assert credentials.credentials == {}

    def test_create_anonymous_credentials_full(self):
        """Test creating anonymous credentials with all parameters."""
        credentials = StorageCredentials.create_anonymous_credentials(
            region="us-west-1", endpoint_url="https://minio.local:9000"
        )

        assert credentials.auth_type == StorageAuthType.ANONYMOUS
        assert credentials.region == "us-west-1"
        assert credentials.endpoint_url == "https://minio.local:9000"
        assert credentials.credentials == {}

    def test_create_token_credentials_basic(self):
        """Test creating token credentials with basic parameters."""
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

        credentials = StorageCredentials.create_token_credentials(token=token)

        assert credentials.auth_type == StorageAuthType.TOKEN
        assert credentials.region == "us-east-1"
        assert credentials.endpoint_url is None
        assert credentials.credentials["token"] == token

    def test_create_token_credentials_full(self):
        """Test creating token credentials with all parameters."""
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

        credentials = StorageCredentials.create_token_credentials(
            token=token,
            region="ca-central-1",
            endpoint_url="https://token.endpoint.com",
        )

        assert credentials.auth_type == StorageAuthType.TOKEN
        assert credentials.region == "ca-central-1"
        assert credentials.endpoint_url == "https://token.endpoint.com"
        assert credentials.credentials["token"] == token

    def test_get_credential_value_existing(self):
        """Test getting existing credential value."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123", secret_access_key="secret123key"
        )

        assert credentials.get_credential_value("access_key_id") == "AKIATEST123"
        assert credentials.get_credential_value("secret_access_key") == "secret123key"

    def test_get_credential_value_missing(self):
        """Test getting missing credential value."""
        credentials = StorageCredentials.create_anonymous_credentials()

        assert credentials.get_credential_value("nonexistent_key") is None

    def test_has_credential_existing(self):
        """Test checking for existing credential."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123", secret_access_key="secret123key"
        )

        assert credentials.has_credential("access_key_id") is True
        assert credentials.has_credential("secret_access_key") is True

    def test_has_credential_missing(self):
        """Test checking for missing credential."""
        credentials = StorageCredentials.create_anonymous_credentials()

        assert credentials.has_credential("access_key_id") is False
        assert credentials.has_credential("nonexistent_key") is False

    def test_to_boto3_kwargs_access_key(self):
        """Test converting access key credentials to boto3 kwargs."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123",
            secret_access_key="secret123key",
            region="us-west-2",
            endpoint_url="https://s3.custom.com",
        )

        kwargs = credentials.to_boto3_kwargs()

        assert kwargs["region_name"] == "us-west-2"
        assert kwargs["endpoint_url"] == "https://s3.custom.com"
        assert kwargs["aws_access_key_id"] == "AKIATEST123"
        assert kwargs["aws_secret_access_key"] == "secret123key"
        assert "aws_session_token" not in kwargs

    def test_to_boto3_kwargs_access_key_with_session_token(self):
        """Test converting access key credentials with session token to boto3 kwargs."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123",
            secret_access_key="secret123key",
            session_token="session123",
        )

        kwargs = credentials.to_boto3_kwargs()

        assert kwargs["aws_access_key_id"] == "AKIATEST123"
        assert kwargs["aws_secret_access_key"] == "secret123key"
        assert kwargs["aws_session_token"] == "session123"

    def test_to_boto3_kwargs_token(self):
        """Test converting token credentials to boto3 kwargs."""
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        credentials = StorageCredentials.create_token_credentials(
            token=token, region="eu-west-1"
        )

        kwargs = credentials.to_boto3_kwargs()

        assert kwargs["region_name"] == "eu-west-1"
        assert kwargs["aws_session_token"] == token
        assert "aws_access_key_id" not in kwargs
        assert "aws_secret_access_key" not in kwargs

    def test_to_boto3_kwargs_anonymous(self):
        """Test converting anonymous credentials to boto3 kwargs."""
        credentials = StorageCredentials.create_anonymous_credentials(
            region="ap-southeast-1", endpoint_url="https://minio.local:9000"
        )

        kwargs = credentials.to_boto3_kwargs()

        assert kwargs["region_name"] == "ap-southeast-1"
        assert kwargs["endpoint_url"] == "https://minio.local:9000"
        assert "aws_access_key_id" not in kwargs
        assert "aws_secret_access_key" not in kwargs
        assert "aws_session_token" not in kwargs

    def test_to_boto3_kwargs_role_based(self):
        """Test converting role-based credentials to boto3 kwargs."""
        role_arn = "arn:aws:iam::123456789012:role/TestRole"
        credentials = StorageCredentials.create_role_based_credentials(
            role_arn=role_arn, region="us-east-2"
        )

        kwargs = credentials.to_boto3_kwargs()

        # Role-based credentials don't add AWS keys directly to boto3 kwargs
        # They would be handled by the STS assume role process
        assert kwargs["region_name"] == "us-east-2"
        assert "aws_access_key_id" not in kwargs
        assert "aws_secret_access_key" not in kwargs

    def test_is_valid_access_key_valid(self):
        """Test is_valid for valid access key credentials."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123", secret_access_key="secret123key"
        )

        assert credentials.is_valid() is True

    def test_is_valid_access_key_invalid(self):
        """Test is_valid for invalid access key credentials."""
        # Missing secret key
        credentials = StorageCredentials(
            auth_type=StorageAuthType.ACCESS_KEY,
            credentials={"access_key_id": "AKIATEST123"},
        )

        assert credentials.is_valid() is False

        # Missing access key
        credentials = StorageCredentials(
            auth_type=StorageAuthType.ACCESS_KEY,
            credentials={"secret_access_key": "secret123key"},
        )

        assert credentials.is_valid() is False

    def test_is_valid_role_based_valid(self):
        """Test is_valid for valid role-based credentials."""
        role_arn = "arn:aws:iam::123456789012:role/TestRole"
        credentials = StorageCredentials.create_role_based_credentials(
            role_arn=role_arn
        )

        assert credentials.is_valid() is True

    def test_is_valid_role_based_invalid(self):
        """Test is_valid for invalid role-based credentials."""
        credentials = StorageCredentials(
            auth_type=StorageAuthType.ROLE_BASED, credentials={}
        )

        assert credentials.is_valid() is False

    def test_is_valid_token_valid(self):
        """Test is_valid for valid token credentials."""
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        credentials = StorageCredentials.create_token_credentials(token=token)

        assert credentials.is_valid() is True

    def test_is_valid_token_invalid(self):
        """Test is_valid for invalid token credentials."""
        credentials = StorageCredentials(
            auth_type=StorageAuthType.TOKEN, credentials={}
        )

        assert credentials.is_valid() is False

    def test_is_valid_anonymous(self):
        """Test is_valid for anonymous credentials."""
        credentials = StorageCredentials.create_anonymous_credentials()

        assert credentials.is_valid() is True

    def test_is_valid_unknown_auth_type(self):
        """Test is_valid for unknown auth type."""
        # This would be difficult to create due to Pydantic validation,
        # but we can test the logic path
        credentials = StorageCredentials(
            auth_type=StorageAuthType.ACCESS_KEY, credentials={}
        )
        # Manually change auth_type to bypass Pydantic validation
        credentials.__dict__["auth_type"] = "unknown_type"

        assert credentials.is_valid() is False

    def test_string_representation(self):
        """Test string representations."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123",
            secret_access_key="secret123key",
            region="us-west-2",
        )

        str_repr = str(credentials)
        repr_repr = repr(credentials)

        assert "StorageCredentials" in str_repr
        assert "ACCESS_KEY" in str_repr
        assert "us-west-2" in str_repr

        assert str_repr == repr_repr

    def test_string_representation_different_types(self):
        """Test string representations for different auth types."""
        # Anonymous
        anon_creds = StorageCredentials.create_anonymous_credentials(region="eu-west-1")
        assert "ANONYMOUS" in str(anon_creds)
        assert "eu-west-1" in str(anon_creds)

        # Token
        token_creds = StorageCredentials.create_token_credentials(
            "token123", region="ap-south-1"
        )
        assert "TOKEN" in str(token_creds)
        assert "ap-south-1" in str(token_creds)

        # Role-based
        role_creds = StorageCredentials.create_role_based_credentials(
            "arn:aws:iam::123456789012:role/TestRole", region="ca-central-1"
        )
        assert "ROLE_BASED" in str(role_creds)
        assert "ca-central-1" in str(role_creds)

    def test_pydantic_validation(self):
        """Test Pydantic model validation."""
        # Valid data
        data = {
            "auth_type": "access_key",
            "region": "us-east-1",
            "credentials": {"access_key_id": "test", "secret_access_key": "secret"},
        }

        credentials = StorageCredentials(**data)
        assert credentials.auth_type == StorageAuthType.ACCESS_KEY

        # Invalid auth_type
        with pytest.raises(ValidationError):
            StorageCredentials(auth_type="invalid_type", region="us-east-1")

    def test_credentials_modification(self):
        """Test that credentials dict can be modified after creation."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123", secret_access_key="secret123key"
        )

        # Add additional credential
        credentials.credentials["additional_key"] = "additional_value"

        assert credentials.get_credential_value("additional_key") == "additional_value"
        assert credentials.has_credential("additional_key") is True

    def test_complex_endpoint_urls(self):
        """Test with complex endpoint URLs."""
        test_urls = [
            "https://s3.amazonaws.com",
            "http://localhost:9000",
            "https://storage.googleapis.com",
            "https://blob.core.windows.net",
            "https://s3.eu-west-1.amazonaws.com",
        ]

        for url in test_urls:
            credentials = StorageCredentials.create_anonymous_credentials(
                endpoint_url=url
            )
            assert credentials.endpoint_url == url

            kwargs = credentials.to_boto3_kwargs()
            assert kwargs["endpoint_url"] == url

    def test_empty_credentials_dict(self):
        """Test behavior with empty credentials dict."""
        credentials = StorageCredentials(
            auth_type=StorageAuthType.ANONYMOUS, region="us-east-1", credentials={}
        )

        assert credentials.get_credential_value("any_key") is None
        assert credentials.has_credential("any_key") is False
        assert credentials.is_valid() is True  # Anonymous is always valid

    def test_case_sensitivity(self):
        """Test case sensitivity of credential keys."""
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id="AKIATEST123", secret_access_key="secret123key"
        )

        assert credentials.has_credential("access_key_id") is True
        assert credentials.has_credential("ACCESS_KEY_ID") is False
        assert credentials.has_credential("Access_Key_Id") is False
