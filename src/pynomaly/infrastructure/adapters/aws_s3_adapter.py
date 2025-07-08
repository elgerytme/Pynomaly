"""AWS S3 storage adapter implementation."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

from pynomaly.domain.services.cloud_storage_adapter import AbstractCloudStorageAdapter, UploadOptions, DownloadOptions, EncryptionType, StorageMetadata
from pynomaly.domain.value_objects.storage_credentials import StorageCredentials
from pynomaly.domain.exceptions.storage_exceptions import (
    StorageError,
    StorageAuthenticationError,
    StoragePermissionError,
    StorageConnectionError,
    StorageNotFoundError,
)


class AWSS3Adapter(AbstractCloudStorageAdapter):
    """AWS S3 implementation of CloudStorageAdapter."""

    def __init__(self, credentials: StorageCredentials):
        super().__init__(credentials)
        self._s3_client = None
        self._s3_resource = None

    async def connect(self) -> None:
        """Connect to AWS S3."""
        try:
            # Configure boto3 client
            config = Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=50,
                region_name=self.credentials.region
            )
            
            # Create S3 client based on authentication type
            if self.credentials.auth_type.value == 'access_key':
                self._s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.credentials.credentials['access_key_id'],
                    aws_secret_access_key=self.credentials.credentials['secret_access_key'],
                    config=config
                )
            elif self.credentials.auth_type.value == 'role_based':
                # Use instance profile or role-based authentication
                self._s3_client = boto3.client('s3', config=config)
            else:
                raise StorageAuthenticationError(f"Unsupported authentication type: {self.credentials.auth_type.value}")
            
            # Create resource for some operations
            self._s3_resource = boto3.resource('s3', config=config)
            
            # Test connection
            await self._test_connection()
            self._is_connected = True
            
        except NoCredentialsError:
            raise StorageAuthenticationError("AWS credentials not found")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code in ['AccessDenied', 'SignatureDoesNotMatch']:
                raise StorageAuthenticationError(f"Authentication failed: {e}")
            else:
                raise StorageConnectionError(f"Failed to connect to AWS S3: {e}")
        except Exception as e:
            raise StorageConnectionError(f"Failed to connect to AWS S3: {e}")

    async def disconnect(self) -> None:
        """Disconnect from AWS S3."""
        if self._s3_client:
            # boto3 clients don't need explicit disconnection
            self._s3_client = None
            self._s3_resource = None
        self._is_connected = False

    async def upload(
        self,
        container: str,
        key: str,
        data: bytes,
        options: Optional[UploadOptions] = None
    ) -> str:
        """Upload data to AWS S3."""
        try:
            upload_args = {
                'Bucket': container,
                'Key': key,
                'Body': data,
                'ContentType': options.content_type.value if options else 'application/octet-stream'
            }
            
            # Add encryption if specified
            if options and options.encryption_type != EncryptionType.NONE:
                if options.encryption_type == EncryptionType.SERVER_SIDE_AES256:
                    upload_args['ServerSideEncryption'] = 'AES256'
                elif options.encryption_type == EncryptionType.SERVER_SIDE_KMS:
                    upload_args['ServerSideEncryption'] = 'aws:kms'
                    if options.encryption_key_id:
                        upload_args['SSEKMSKeyId'] = options.encryption_key_id
            
            # Add custom metadata
            if options and options.custom_metadata:
                upload_args['Metadata'] = options.custom_metadata
            
            # Add storage class
            if options and options.storage_class:
                upload_args['StorageClass'] = options.storage_class
            
            # Add cache control
            if options and options.cache_control:
                upload_args['CacheControl'] = options.cache_control
            
            # Add expires
            if options and options.expires:
                upload_args['Expires'] = options.expires
            
            # Use multipart upload for large files
            if options and options.enable_multipart and len(data) > options.multipart_chunk_size:
                return await self._multipart_upload(container, key, data, options)
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self._s3_client.put_object, **upload_args
                )
                return response['ETag'].strip('"')
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchBucket':
                raise StorageNotFoundError(f"Bucket '{container}' not found")
            elif error_code == 'AccessDenied':
                raise StoragePermissionError(f"Access denied for bucket '{container}' and key '{key}'")
            else:
                raise StorageError(f"Failed to upload to S3: {e}")
        except Exception as e:
            raise StorageError(f"Failed to upload to S3: {e}")

    async def upload_progressive(
        self,
        container: str,
        key: str,
        data: bytes,
        options: Optional[UploadOptions] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """Upload data to AWS S3 with progress tracking."""
        try:
            total_size = len(data)
            chunk_size = options.multipart_chunk_size if options else 8 * 1024 * 1024  # 8MB default
            
            if total_size <= chunk_size:
                # Single part upload
                if progress_callback:
                    progress_callback(0, total_size)
                etag = await self.upload(container, key, data, options)
                if progress_callback:
                    progress_callback(total_size, total_size)
                return etag
            else:
                # Multipart upload with progress
                return await self._multipart_upload_with_progress(
                    container, key, data, options, progress_callback
                )
                
        except Exception as e:
            raise StorageError(f"Failed to upload with progress: {e}")

    async def download(
        self,
        container: str,
        key: str,
        options: Optional[DownloadOptions] = None
    ) -> bytes:
        """Download data from AWS S3."""
        try:
            download_args = {
                'Bucket': container,
                'Key': key
            }
            
            # Add version ID if specified
            if options and options.version_id:
                download_args['VersionId'] = options.version_id
            
            # Add byte range if specified
            if options and options.byte_range:
                start, end = options.byte_range
                download_args['Range'] = f'bytes={start}-{end}'
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.get_object, **download_args
            )
            
            return response['Body'].read()
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise StorageNotFoundError(f"Object '{key}' not found in bucket '{container}'")
            elif error_code == 'NoSuchBucket':
                raise StorageNotFoundError(f"Bucket '{container}' not found")
            elif error_code == 'AccessDenied':
                raise StoragePermissionError(f"Access denied for bucket '{container}' and key '{key}'")
            else:
                raise StorageError(f"Failed to download from S3: {e}")
        except Exception as e:
            raise StorageError(f"Failed to download from S3: {e}")

    async def download_stream(
        self,
        container: str,
        key: str,
        options: Optional[DownloadOptions] = None
    ) -> AsyncGenerator[bytes, None]:
        """Download data from AWS S3 as a stream."""
        try:
            download_args = {
                'Bucket': container,
                'Key': key
            }
            
            if options and options.version_id:
                download_args['VersionId'] = options.version_id
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.get_object, **download_args
            )
            
            chunk_size = options.stream_chunk_size if options else 64 * 1024  # 64KB default
            
            body = response['Body']
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise StorageNotFoundError(f"Object '{key}' not found in bucket '{container}'")
            elif error_code == 'NoSuchBucket':
                raise StorageNotFoundError(f"Bucket '{container}' not found")
            elif error_code == 'AccessDenied':
                raise StoragePermissionError(f"Access denied for bucket '{container}' and key '{key}'")
            else:
                raise StorageError(f"Failed to download stream from S3: {e}")
        except Exception as e:
            raise StorageError(f"Failed to download stream from S3: {e}")

    async def download_progressive(
        self,
        container: str,
        key: str,
        options: Optional[DownloadOptions] = None,
        progress_callback: Optional[callable] = None
    ) -> bytes:
        """Download data from AWS S3 with progress tracking."""
        try:
            # Get object metadata first to know the size
            metadata = await self.get_metadata(container, key, options.version_id if options else None)
            total_size = metadata.size_bytes
            
            if progress_callback:
                progress_callback(0, total_size)
            
            # Download with progress tracking
            chunk_size = options.stream_chunk_size if options else 64 * 1024  # 64KB default
            downloaded_data = b''
            downloaded_size = 0
            
            async for chunk in self.download_stream(container, key, options):
                downloaded_data += chunk
                downloaded_size += len(chunk)
                
                if progress_callback:
                    progress_callback(downloaded_size, total_size)
            
            return downloaded_data
            
        except Exception as e:
            raise StorageError(f"Failed to download with progress: {e}")

    async def delete(
        self,
        container: str,
        key: str,
        version_id: Optional[str] = None
    ) -> bool:
        """Delete an object from AWS S3."""
        try:
            delete_args = {
                'Bucket': container,
                'Key': key
            }
            
            if version_id:
                delete_args['VersionId'] = version_id
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.delete_object, **delete_args
            )
            
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                return False  # Object didn't exist
            elif error_code == 'NoSuchBucket':
                raise StorageNotFoundError(f"Bucket '{container}' not found")
            elif error_code == 'AccessDenied':
                raise StoragePermissionError(f"Access denied for bucket '{container}' and key '{key}'")
            else:
                raise StorageError(f"Failed to delete from S3: {e}")
        except Exception as e:
            raise StorageError(f"Failed to delete from S3: {e}")

    async def exists(
        self,
        container: str,
        key: str,
        version_id: Optional[str] = None
    ) -> bool:
        """Check if an object exists in AWS S3."""
        try:
            head_args = {
                'Bucket': container,
                'Key': key
            }
            
            if version_id:
                head_args['VersionId'] = version_id
            
            await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.head_object, **head_args
            )
            
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code in ['NoSuchKey', 'NotFound']:
                return False
            elif error_code == 'NoSuchBucket':
                raise StorageNotFoundError(f"Bucket '{container}' not found")
            elif error_code == 'AccessDenied':
                raise StoragePermissionError(f"Access denied for bucket '{container}' and key '{key}'")
            else:
                raise StorageError(f"Failed to check existence in S3: {e}")
        except Exception as e:
            raise StorageError(f"Failed to check existence in S3: {e}")

    async def get_metadata(
        self,
        container: str,
        key: str,
        version_id: Optional[str] = None
    ) -> StorageMetadata:
        """Get metadata for an object in AWS S3."""
        try:
            head_args = {
                'Bucket': container,
                'Key': key
            }
            
            if version_id:
                head_args['VersionId'] = version_id
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.head_object, **head_args
            )
            
            # Determine encryption type
            encryption_type = EncryptionType.NONE
            encryption_key_id = None
            
            if response.get('ServerSideEncryption') == 'AES256':
                encryption_type = EncryptionType.SERVER_SIDE_AES256
            elif response.get('ServerSideEncryption') == 'aws:kms':
                encryption_type = EncryptionType.SERVER_SIDE_KMS
                encryption_key_id = response.get('SSEKMSKeyId')
            
            return StorageMetadata(
                size_bytes=response['ContentLength'],
                content_type=response.get('ContentType', 'application/octet-stream'),
                last_modified=response.get('LastModified').isoformat() if response.get('LastModified') else None,
                etag=response.get('ETag', '').strip('"'),
                encryption_type=encryption_type,
                encryption_key_id=encryption_key_id,
                custom_metadata=response.get('Metadata', {})
            )
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise StorageNotFoundError(f"Object '{key}' not found in bucket '{container}'")
            elif error_code == 'NoSuchBucket':
                raise StorageNotFoundError(f"Bucket '{container}' not found")
            elif error_code == 'AccessDenied':
                raise StoragePermissionError(f"Access denied for bucket '{container}' and key '{key}'")
            else:
                raise StorageError(f"Failed to get metadata from S3: {e}")
        except Exception as e:
            raise StorageError(f"Failed to get metadata from S3: {e}")

    async def list_objects(
        self,
        container: str,
        prefix: Optional[str] = None,
        max_keys: Optional[int] = None,
        continuation_token: Optional[str] = None
    ) -> dict:
        """List objects in an AWS S3 bucket."""
        try:
            list_args = {
                'Bucket': container
            }
            
            if prefix:
                list_args['Prefix'] = prefix
            
            if max_keys:
                list_args['MaxKeys'] = max_keys
            
            if continuation_token:
                list_args['ContinuationToken'] = continuation_token
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.list_objects_v2, **list_args
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"'),
                    'storage_class': obj.get('StorageClass', 'STANDARD')
                })
            
            return {
                'objects': objects,
                'is_truncated': response.get('IsTruncated', False),
                'next_continuation_token': response.get('NextContinuationToken'),
                'key_count': response.get('KeyCount', 0)
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchBucket':
                raise StorageNotFoundError(f"Bucket '{container}' not found")
            elif error_code == 'AccessDenied':
                raise StoragePermissionError(f"Access denied for bucket '{container}'")
            else:
                raise StorageError(f"Failed to list objects in S3: {e}")
        except Exception as e:
            raise StorageError(f"Failed to list objects in S3: {e}")

    async def copy_object(
        self,
        source_container: str,
        source_key: str,
        dest_container: str,
        dest_key: str,
        options: Optional[UploadOptions] = None
    ) -> str:
        """Copy an object within or between AWS S3 buckets."""
        try:
            copy_source = {
                'Bucket': source_container,
                'Key': source_key
            }
            
            copy_args = {
                'CopySource': copy_source,
                'Bucket': dest_container,
                'Key': dest_key
            }
            
            # Add metadata and options
            if options:
                if options.custom_metadata:
                    copy_args['Metadata'] = options.custom_metadata
                    copy_args['MetadataDirective'] = 'REPLACE'
                
                if options.storage_class:
                    copy_args['StorageClass'] = options.storage_class
                
                if options.encryption_type != EncryptionType.NONE:
                    if options.encryption_type == EncryptionType.SERVER_SIDE_AES256:
                        copy_args['ServerSideEncryption'] = 'AES256'
                    elif options.encryption_type == EncryptionType.SERVER_SIDE_KMS:
                        copy_args['ServerSideEncryption'] = 'aws:kms'
                        if options.encryption_key_id:
                            copy_args['SSEKMSKeyId'] = options.encryption_key_id
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.copy_object, **copy_args
            )
            
            return response['CopyObjectResult']['ETag'].strip('"')
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchKey':
                raise StorageNotFoundError(f"Source object '{source_key}' not found in bucket '{source_container}'")
            elif error_code == 'NoSuchBucket':
                raise StorageNotFoundError(f"Bucket not found")
            elif error_code == 'AccessDenied':
                raise StoragePermissionError(f"Access denied for copy operation")
            else:
                raise StorageError(f"Failed to copy object in S3: {e}")
        except Exception as e:
            raise StorageError(f"Failed to copy object in S3: {e}")

    async def generate_presigned_url(
        self,
        container: str,
        key: str,
        operation: str,
        expires_in: int = 3600
    ) -> str:
        """Generate a presigned URL for an AWS S3 object."""
        try:
            operation_map = {
                'get': 'get_object',
                'put': 'put_object',
                'delete': 'delete_object'
            }
            
            if operation not in operation_map:
                raise ValueError(f"Unsupported operation: {operation}")
            
            url = await asyncio.get_event_loop().run_in_executor(
                None,
                self._s3_client.generate_presigned_url,
                operation_map[operation],
                {'Bucket': container, 'Key': key},
                expires_in
            )
            
            return url
            
        except ClientError as e:
            raise StorageError(f"Failed to generate presigned URL: {e}")
        except Exception as e:
            raise StorageError(f"Failed to generate presigned URL: {e}")

    async def test_connection(self) -> bool:
        """Test the connection to AWS S3."""
        try:
            # Test by listing buckets
            await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.list_buckets
            )
            return True
        except Exception:
            return False

    async def _test_connection(self) -> None:
        """Internal method to test connection during initialization."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.list_buckets
            )
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'AccessDenied':
                raise StorageAuthenticationError(f"Authentication failed: {e}")
            else:
                raise StorageConnectionError(f"Connection test failed: {e}")

    async def _multipart_upload(
        self,
        container: str,
        key: str,
        data: bytes,
        options: Optional[UploadOptions] = None
    ) -> str:
        """Perform multipart upload for large files."""
        try:
            # Initiate multipart upload
            create_args = {
                'Bucket': container,
                'Key': key,
                'ContentType': options.content_type.value if options else 'application/octet-stream'
            }
            
            if options and options.custom_metadata:
                create_args['Metadata'] = options.custom_metadata
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.create_multipart_upload, **create_args
            )
            
            upload_id = response['UploadId']
            
            # Upload parts
            chunk_size = options.multipart_chunk_size if options else 8 * 1024 * 1024  # 8MB
            parts = []
            part_number = 1
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                
                part_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._s3_client.upload_part,
                    **{
                        'Bucket': container,
                        'Key': key,
                        'PartNumber': part_number,
                        'UploadId': upload_id,
                        'Body': chunk
                    }
                )
                
                parts.append({
                    'ETag': part_response['ETag'],
                    'PartNumber': part_number
                })
                
                part_number += 1
            
            # Complete multipart upload
            complete_response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._s3_client.complete_multipart_upload,
                **{
                    'Bucket': container,
                    'Key': key,
                    'UploadId': upload_id,
                    'MultipartUpload': {'Parts': parts}
                }
            )
            
            return complete_response['ETag'].strip('"')
            
        except Exception as e:
            # Abort multipart upload on error
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._s3_client.abort_multipart_upload,
                    **{
                        'Bucket': container,
                        'Key': key,
                        'UploadId': upload_id
                    }
                )
            except:
                pass  # Ignore errors during cleanup
            
            raise StorageError(f"Multipart upload failed: {e}")

    async def _multipart_upload_with_progress(
        self,
        container: str,
        key: str,
        data: bytes,
        options: Optional[UploadOptions] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """Perform multipart upload with progress tracking."""
        try:
            total_size = len(data)
            uploaded_size = 0
            
            if progress_callback:
                progress_callback(0, total_size)
            
            # Initiate multipart upload
            create_args = {
                'Bucket': container,
                'Key': key,
                'ContentType': options.content_type.value if options else 'application/octet-stream'
            }
            
            if options and options.custom_metadata:
                create_args['Metadata'] = options.custom_metadata
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._s3_client.create_multipart_upload, **create_args
            )
            
            upload_id = response['UploadId']
            
            # Upload parts with progress tracking
            chunk_size = options.multipart_chunk_size if options else 8 * 1024 * 1024  # 8MB
            parts = []
            part_number = 1
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                
                part_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._s3_client.upload_part,
                    **{
                        'Bucket': container,
                        'Key': key,
                        'PartNumber': part_number,
                        'UploadId': upload_id,
                        'Body': chunk
                    }
                )
                
                parts.append({
                    'ETag': part_response['ETag'],
                    'PartNumber': part_number
                })
                
                uploaded_size += len(chunk)
                
                if progress_callback:
                    progress_callback(uploaded_size, total_size)
                
                part_number += 1
            
            # Complete multipart upload
            complete_response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._s3_client.complete_multipart_upload,
                **{
                    'Bucket': container,
                    'Key': key,
                    'UploadId': upload_id,
                    'MultipartUpload': {'Parts': parts}
                }
            )
            
            return complete_response['ETag'].strip('"')
            
        except Exception as e:
            # Abort multipart upload on error
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._s3_client.abort_multipart_upload,
                    **{
                        'Bucket': container,
                        'Key': key,
                        'UploadId': upload_id
                    }
                )
            except:
                pass  # Ignore errors during cleanup
            
            raise StorageError(f"Multipart upload with progress failed: {e}")

