"""
S3 storage adapter implementation.

This module provides a concrete implementation of the AbstractCloudStorageAdapter
for Amazon S3 and S3-compatible storage systems.
"""

from __future__ import annotations

import asyncio
import io
import os
from datetime import datetime
from typing import IO, Any, AsyncGenerator, Dict, Optional, Union

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from pynomaly.domain.exceptions.storage_exceptions import (
    EncryptionError,
    QuotaExceededError,
    StorageAuthenticationError,
    StorageConnectionError,
    StorageError,
    StorageNotFoundError,
    StoragePermissionError,
)
from pynomaly.domain.services.cloud_storage_adapter import (
    AbstractCloudStorageAdapter,
    ContentType,
    DownloadOptions,
    EncryptionType,
    ProgressInfo,
    StorageCredentials,
    StorageMetadata,
    UploadOptions,
)
from pynomaly.domain.value_objects.storage_credentials import AuthenticationType


class S3Adapter(AbstractCloudStorageAdapter):
    def __init__(self, credentials: StorageCredentials):
        super().__init__(credentials)
        self._s3_client = None
        self._s3_resource = None

    async def connect(self) -> None:
        try:
            session = self._create_session()
            self._s3_client = session.client(
                "s3",
                endpoint_url=self.credentials.endpoint_url,
                config=Config(
                    signature_version='s3v4',
                    retries={
                        'max_attempts': self.credentials.max_retry_attempts,
                        'mode': 'adaptive'
                    },
                    connect_timeout=self.credentials.timeout_seconds,
                    read_timeout=self.credentials.timeout_seconds,
                ),
            )
            self._s3_resource = session.resource("s3")
            self._is_connected = True
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            raise StorageConnectionError(f"Failed to connect to S3: {e}")
    
    def _create_session(self) -> boto3.Session:
        """Create a boto3 session based on the authentication type."""
        if self.credentials.auth_type == AuthenticationType.ACCESS_KEY:
            return boto3.Session(
                aws_access_key_id=self.credentials.get_credential_value("access_key_id"),
                aws_secret_access_key=self.credentials.get_credential_value("secret_access_key"),
                aws_session_token=self.credentials.credentials.get("session_token"),
                region_name=self.credentials.region,
            )
        elif self.credentials.auth_type == AuthenticationType.ROLE_BASED:
            # Use STS to assume role
            sts_client = boto3.client('sts')
            assumed_role = sts_client.assume_role(
                RoleArn=self.credentials.get_credential_value("role_arn"),
                RoleSessionName=self.credentials.credentials.get("session_name", "pynomaly-session")
            )
            return boto3.Session(
                aws_access_key_id=assumed_role['Credentials']['AccessKeyId'],
                aws_secret_access_key=assumed_role['Credentials']['SecretAccessKey'],
                aws_session_token=assumed_role['Credentials']['SessionToken'],
                region_name=self.credentials.region,
            )
        else:
            # Default session (uses environment variables or instance profile)
            return boto3.Session(region_name=self.credentials.region)

    async def disconnect(self) -> None:
        self._s3_client = None
        self._s3_resource = None
        self._is_connected = False

    async def upload(
        self, container: str, key: str, data: Union[bytes, IO], options: Optional[UploadOptions] = None
    ) -> str:
        try:
            extra_args = {}
            if options:
                if options.encryption_type == EncryptionType.SERVER_SIDE_KMS:
                    extra_args['ServerSideEncryption'] = 'aws:kms'
                    extra_args['SSEKMSKeyId'] = options.encryption_key_id
                elif options.encryption_type == EncryptionType.SERVER_SIDE_AES256:
                    extra_args['ServerSideEncryption'] = 'AES256'
                if options.custom_metadata:
                    extra_args['Metadata'] = options.custom_metadata
                if options.cache_control:
                    extra_args['CacheControl'] = options.cache_control
                if options.content_type:
                    extra_args['ContentType'] = options.content_type.value
                if options.expires:
                    extra_args['Expires'] = options.expires

            with io.BytesIO(data) if isinstance(data, bytes) else data as data_stream:
                self._s3_client.upload_fileobj(
                    data_stream,
                    container,
                    key,
                    ExtraArgs=extra_args,
                    Config=TransferConfig(
                        multipart_threshold=1024 * 1024 * 25,  # 25 MB
                        max_concurrency=10,
                        multipart_chunksize=options.multipart_chunk_size if options else 8 * 1024 * 1024,  # 8 MB
                        use_threads=True
                    ),
                )
            return self._s3_client.head_object(Bucket=container, Key=key)["ETag"]
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            raise StorageError(f"Failed to upload to S3: {e}")

    async def download(
        self, container: str, key: str, options: Optional[DownloadOptions] = None
    ) -> bytes:
        try:
            extra_args = {}
            if options and options.byte_range:
                extra_args['Range'] = f'bytes={options.byte_range[0]}-{options.byte_range[1]}'

            with io.BytesIO() as data:
                self._s3_client.download_fileobj(Bucket=container, Key=key, Fileobj=data, ExtraArgs=extra_args)
                return data.getvalue()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise StorageNotFoundError(f"Object {key} not found in {container}.")
            raise StorageError(f"Failed to download from S3: {e}")

    async def generate_presigned_url(
        self, container: str, key: str, operation: str, expires_in: int = 3600
    ) -> str:
        try:
            return self._s3_client.generate_presigned_url(
                ClientMethod=operation,
                Params={"Bucket": container, "Key": key},
                ExpiresIn=expires_in,
            )
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            raise StorageError(f"Failed to generate presigned URL: {e}")

    async def list_objects(
        self, container: str, prefix: Optional[str] = None, max_keys: Optional[int] = None, continuation_token: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            pagination_args = {'Bucket': container, 'Prefix': prefix, 'MaxKeys': max_keys, 'ContinuationToken': continuation_token}
            response = self._s3_client.list_objects_v2(**{k: v for k, v in pagination_args.items() if v is not None})
            return response
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            raise StorageError(f"Failed to list objects in S3: {e}")

    async def get_metadata(
        self, container: str, key: str, version_id: Optional[str] = None
    ) -> StorageMetadata:
        try:
            response = self._s3_client.head_object(Bucket=container, Key=key, VersionId=version_id)
            return StorageMetadata(
                size_bytes=response['ContentLength'],
                content_type=response['ContentType'],
                last_modified=response['LastModified'].isoformat() if 'LastModified' in response else None,
                etag=response['ETag'] if 'ETag' in response else None,
                encryption_type=EncryptionType.SERVER_SIDE_KMS if response.get('ServerSideEncryption') == 'aws:kms' else EncryptionType.NONE,
                encryption_key_id=response.get('SSEKMSKeyId'),
                custom_metadata=response['Metadata'] if 'Metadata' in response else None
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise StorageNotFoundError(f"Metadata not found for object {key} in {container}.")
            raise StorageError(f"Failed to get metadata from S3: {e}")

    async def copy_object(
        self, source_container: str, source_key: str, dest_container: str, dest_key: str, options: Optional[UploadOptions] = None
    ) -> str:
        try:
            copy_source = {'Bucket': source_container, 'Key': source_key}
            extra_args = {}
            if options:
                if options.encryption_type == EncryptionType.SERVER_SIDE_KMS:
                    extra_args['ServerSideEncryption'] = 'aws:kms'
                    extra_args['SSEKMSKeyId'] = options.encryption_key_id
                elif options.encryption_type == EncryptionType.SERVER_SIDE_AES256:
                    extra_args['ServerSideEncryption'] = 'AES256'

            self._s3_client.copy(copy_source, dest_container, dest_key, ExtraArgs=extra_args)
            return self._s3_client.head_object(Bucket=dest_container, Key=dest_key)["ETag"]
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            raise StorageError(f"Failed to copy object in S3: {e}")

    async def test_connection(self) -> bool:
        try:
            self._s3_client.list_buckets()
            return True
        except (BotoCoreError, ClientError, NoCredentialsError):
            return False

    async def upload_progressive(
        self,
        container: str,
        key: str,
        data: Union[bytes, IO],
        options: Optional[UploadOptions] = None,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Upload data with progress tracking."""
        try:
            extra_args = {}
            if options:
                if options.encryption_type == EncryptionType.SERVER_SIDE_KMS:
                    extra_args['ServerSideEncryption'] = 'aws:kms'
                    extra_args['SSEKMSKeyId'] = options.encryption_key_id
                elif options.encryption_type == EncryptionType.SERVER_SIDE_AES256:
                    extra_args['ServerSideEncryption'] = 'AES256'
                if options.custom_metadata:
                    extra_args['Metadata'] = options.custom_metadata
                if options.cache_control:
                    extra_args['CacheControl'] = options.cache_control
                if options.content_type:
                    extra_args['ContentType'] = options.content_type.value
                if options.expires:
                    extra_args['Expires'] = options.expires

            # Custom progress callback wrapper
            def progress_wrapper(bytes_transferred):
                if progress_callback:
                    # Create progress info
                    progress_info = ProgressInfo(
                        bytes_transferred=bytes_transferred,
                        total_bytes=len(data) if isinstance(data, bytes) else None,
                        percentage=(bytes_transferred / len(data) * 100) if isinstance(data, bytes) else 0,
                        transfer_rate=0,  # Not tracked in this implementation
                    )
                    progress_callback(progress_info)

            with io.BytesIO(data) if isinstance(data, bytes) else data as data_stream:
                self._s3_client.upload_fileobj(
                    data_stream,
                    container,
                    key,
                    ExtraArgs=extra_args,
                    Callback=progress_wrapper,
                    Config=TransferConfig(
                        multipart_threshold=1024 * 1024 * 25,  # 25 MB
                        max_concurrency=10,
                        multipart_chunksize=options.multipart_chunk_size if options else 8 * 1024 * 1024,
                        use_threads=True
                    ),
                )
            return self._s3_client.head_object(Bucket=container, Key=key)["ETag"]
        except (BotoCoreError, ClientError, NoCredentialsError) as e:
            raise StorageError(f"Failed to upload to S3: {e}")

    async def download_stream(
        self,
        container: str,
        key: str,
        options: Optional[DownloadOptions] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Download data as a stream."""
        try:
            extra_args = {}
            if options and options.byte_range:
                extra_args['Range'] = f'bytes={options.byte_range[0]}-{options.byte_range[1]}'

            response = self._s3_client.get_object(Bucket=container, Key=key, **extra_args)
            chunk_size = options.stream_chunk_size if options else 64 * 1024
            
            while True:
                chunk = response['Body'].read(chunk_size)
                if not chunk:
                    break
                yield chunk
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise StorageNotFoundError(f"Object {key} not found in {container}.")
            raise StorageError(f"Failed to download stream from S3: {e}")

    async def download_progressive(
        self,
        container: str,
        key: str,
        options: Optional[DownloadOptions] = None,
        progress_callback: Optional[callable] = None,
    ) -> bytes:
        """Download data with progress tracking."""
        try:
            extra_args = {}
            if options and options.byte_range:
                extra_args['Range'] = f'bytes={options.byte_range[0]}-{options.byte_range[1]}'

            # Custom progress callback wrapper
            def progress_wrapper(bytes_transferred):
                if progress_callback:
                    # Create progress info
                    progress_info = ProgressInfo(
                        bytes_transferred=bytes_transferred,
                        total_bytes=None,  # Not easily available in download
                        percentage=0,  # Not easily calculated in download
                        transfer_rate=0,  # Not tracked in this implementation
                    )
                    progress_callback(progress_info)

            with io.BytesIO() as data:
                self._s3_client.download_fileobj(
                    Bucket=container,
                    Key=key,
                    Fileobj=data,
                    ExtraArgs=extra_args,
                    Callback=progress_wrapper,
                )
                return data.getvalue()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise StorageNotFoundError(f"Object {key} not found in {container}.")
            raise StorageError(f"Failed to download from S3: {e}")

    async def delete(
        self,
        container: str,
        key: str,
        version_id: Optional[str] = None,
    ) -> bool:
        """Delete an object from storage."""
        try:
            delete_args = {'Bucket': container, 'Key': key}
            if version_id:
                delete_args['VersionId'] = version_id
            
            self._s3_client.delete_object(**delete_args)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return False
            raise StorageError(f"Failed to delete object from S3: {e}")

    async def exists(
        self,
        container: str,
        key: str,
        version_id: Optional[str] = None,
    ) -> bool:
        """Check if an object exists in storage."""
        try:
            head_args = {'Bucket': container, 'Key': key}
            if version_id:
                head_args['VersionId'] = version_id
            
            self._s3_client.head_object(**head_args)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return False
            raise StorageError(f"Failed to check object existence in S3: {e}")
    
    @classmethod
    def from_environment(
        cls,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> "S3Adapter":
        """Create S3Adapter from environment variables or AWS profile.
        
        Args:
            region: AWS region (defaults to AWS_DEFAULT_REGION env var)
            endpoint_url: Custom S3 endpoint URL
            profile_name: AWS profile name
            
        Returns:
            S3Adapter instance
        """
        # Try to get credentials from environment
        access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        session_token = os.environ.get("AWS_SESSION_TOKEN")
        
        if access_key_id and secret_access_key:
            credentials = StorageCredentials.create_access_key_credentials(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                region=region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
                session_token=session_token,
            )
            credentials = credentials.with_timeout(30)
            if endpoint_url:
                credentials = StorageCredentials(
                    auth_type=credentials.auth_type,
                    credentials=credentials.credentials,
                    region=credentials.region,
                    endpoint_url=endpoint_url,
                    timeout_seconds=credentials.timeout_seconds,
                    max_retry_attempts=credentials.max_retry_attempts,
                    verify_ssl=credentials.verify_ssl,
                )
            return cls(credentials)
        
        # If no environment variables, use default session (profile or instance role)
        credentials = StorageCredentials(
            auth_type=AuthenticationType.ANONYMOUS,
            credentials={},
            region=region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            endpoint_url=endpoint_url,
        )
        return cls(credentials)
    
    @classmethod
    def from_sts_role(
        cls,
        role_arn: str,
        region: Optional[str] = None,
        session_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> "S3Adapter":
        """Create S3Adapter using STS role assumption.
        
        Args:
            role_arn: ARN of the role to assume
            region: AWS region
            session_name: Session name for the assumed role
            endpoint_url: Custom S3 endpoint URL
            
        Returns:
            S3Adapter instance
        """
        credentials = StorageCredentials.create_role_based_credentials(
            role_arn=role_arn,
            region=region or "us-east-1",
            session_name=session_name,
        )
        
        if endpoint_url:
            credentials = StorageCredentials(
                auth_type=credentials.auth_type,
                credentials=credentials.credentials,
                region=credentials.region,
                endpoint_url=endpoint_url,
                timeout_seconds=credentials.timeout_seconds,
                max_retry_attempts=credentials.max_retry_attempts,
                verify_ssl=credentials.verify_ssl,
            )
        
        return cls(credentials)
    
    @classmethod
    def from_access_key(
        cls,
        access_key_id: str,
        secret_access_key: str,
        region: Optional[str] = None,
        session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> "S3Adapter":
        """Create S3Adapter using access key credentials.
        
        Args:
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region: AWS region
            session_token: Optional session token for temporary credentials
            endpoint_url: Custom S3 endpoint URL
            
        Returns:
            S3Adapter instance
        """
        credentials = StorageCredentials.create_access_key_credentials(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region=region or "us-east-1",
            session_token=session_token,
        )
        
        if endpoint_url:
            credentials = StorageCredentials(
                auth_type=credentials.auth_type,
                credentials=credentials.credentials,
                region=credentials.region,
                endpoint_url=endpoint_url,
                timeout_seconds=credentials.timeout_seconds,
                max_retry_attempts=credentials.max_retry_attempts,
                verify_ssl=credentials.verify_ssl,
            )
        
        return cls(credentials)

