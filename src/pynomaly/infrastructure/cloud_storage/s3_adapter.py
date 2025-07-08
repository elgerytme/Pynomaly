"""AWS S3 cloud storage adapter implementation."""

import asyncio
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, BinaryIO, Union
import aiofiles

from .base import CloudStorageAdapter, CloudStorageConfig, StorageMetadata
from ...shared.exceptions import CloudStorageError


class S3Adapter(CloudStorageAdapter):
    """AWS S3 cloud storage adapter."""
    
    def __init__(self, config: CloudStorageConfig):
        super().__init__(config)
        self._s3_client = None
        self._session = None
    
    async def connect(self) -> None:
        """Connect to AWS S3."""
        try:
            import aioboto3
            self._session = aioboto3.Session(
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name=self.config.region,
            )
            self._s3_client = self._session.client(
                's3',
                endpoint_url=self.config.endpoint_url,
                config=self._get_s3_config(),
            )
        except Exception as e:
            raise CloudStorageError(f"Failed to connect to S3: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from AWS S3."""
        if self._s3_client:
            await self._s3_client.close()
            self._s3_client = None
        self._session = None
    
    async def upload_file(
        self,
        file_path: Union[str, Path],
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Upload a file to S3."""
        if not self._s3_client:
            raise CloudStorageError("S3 client not connected")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise CloudStorageError(f"File not found: {file_path}")
        
        try:
            # Determine if multipart upload is needed
            file_size = file_path.stat().st_size
            if file_size > self.config.multipart_threshold:
                return await self._multipart_upload_file(file_path, key, metadata, content_type, encrypt)
            
            # Single upload
            extra_args = self._build_upload_args(metadata, content_type, encrypt)
            
            async with aiofiles.open(file_path, 'rb') as f:
                await self._s3_client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Body=await f.read(),
                    **extra_args
                )
            
            return await self.get_metadata(key)
            
        except Exception as e:
            raise CloudStorageError(f"Failed to upload file to S3: {e}")
    
    async def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Upload a stream to S3."""
        if not self._s3_client:
            raise CloudStorageError("S3 client not connected")
        
        try:
            extra_args = self._build_upload_args(metadata, content_type, encrypt)
            
            await self._s3_client.put_object(
                Bucket=self.config.bucket_name,
                Key=key,
                Body=stream.read(),
                **extra_args
            )
            
            return await self.get_metadata(key)
            
        except Exception as e:
            raise CloudStorageError(f"Failed to upload stream to S3: {e}")
    
    async def download_file(
        self,
        key: str,
        file_path: Union[str, Path],
        decrypt: bool = False,
    ) -> StorageMetadata:
        """Download a file from S3."""
        if not self._s3_client:
            raise CloudStorageError("S3 client not connected")
        
        file_path = Path(file_path)
        
        try:
            # Get metadata first
            metadata = await self.get_metadata(key)
            
            # Download the file
            response = await self._s3_client.get_object(
                Bucket=self.config.bucket_name,
                Key=key
            )
            
            async with aiofiles.open(file_path, 'wb') as f:
                async for chunk in response['Body'].iter_chunks():
                    await f.write(chunk)
            
            return metadata
            
        except Exception as e:
            raise CloudStorageError(f"Failed to download file from S3: {e}")
    
    async def download_stream(
        self,
        key: str,
        decrypt: bool = False,
    ) -> io.BytesIO:
        """Download a stream from S3."""
        if not self._s3_client:
            raise CloudStorageError("S3 client not connected")
        
        try:
            response = await self._s3_client.get_object(
                Bucket=self.config.bucket_name,
                Key=key
            )
            
            stream = io.BytesIO()
            async for chunk in response['Body'].iter_chunks():
                stream.write(chunk)
            
            stream.seek(0)
            return stream
            
        except Exception as e:
            raise CloudStorageError(f"Failed to download stream from S3: {e}")
    
    async def get_metadata(self, key: str) -> StorageMetadata:
        """Get metadata for an S3 object."""
        if not self._s3_client:
            raise CloudStorageError("S3 client not connected")
        
        try:
            response = await self._s3_client.head_object(
                Bucket=self.config.bucket_name,
                Key=key
            )
            
            return StorageMetadata(
                size=response['ContentLength'],
                content_type=response.get('ContentType', 'application/octet-stream'),
                etag=response['ETag'].strip('"'),
                last_modified=response['LastModified'],
                custom_metadata=response.get('Metadata', {}),
                encryption_info=self._extract_encryption_info(response),
                storage_class=response.get('StorageClass'),
            )
            
        except Exception as e:
            raise CloudStorageError(f"Failed to get metadata from S3: {e}")
    
    async def delete_object(self, key: str) -> bool:
        """Delete an object from S3."""
        if not self._s3_client:
            raise CloudStorageError("S3 client not connected")
        
        try:
            await self._s3_client.delete_object(
                Bucket=self.config.bucket_name,
                Key=key
            )
            return True
            
        except Exception as e:
            raise CloudStorageError(f"Failed to delete object from S3: {e}")
    
    async def list_objects(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List objects in S3."""
        if not self._s3_client:
            raise CloudStorageError("S3 client not connected")
        
        try:
            paginator = self._s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config.bucket_name,
                Prefix=prefix or '',
                MaxKeys=limit or 1000,
            )
            
            objects = []
            async for page in pages:
                contents = page.get('Contents', [])
                for obj in contents:
                    objects.append(obj['Key'])
                    if limit and len(objects) >= limit:
                        break
                if limit and len(objects) >= limit:
                    break
            
            return objects
            
        except Exception as e:
            raise CloudStorageError(f"Failed to list objects in S3: {e}")
    
    async def object_exists(self, key: str) -> bool:
        """Check if an object exists in S3."""
        if not self._s3_client:
            raise CloudStorageError("S3 client not connected")
        
        try:
            await self._s3_client.head_object(
                Bucket=self.config.bucket_name,
                Key=key
            )
            return True
            
        except Exception:
            return False
    
    def _get_s3_config(self):
        """Get S3 client configuration."""
        from botocore.config import Config
        
        return Config(
            retries={
                'max_attempts': self.config.max_retries,
                'mode': 'adaptive'
            },
            max_pool_connections=50,
        )
    
    def _build_upload_args(
        self,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encrypt: bool = False,
    ) -> Dict[str, Any]:
        """Build upload arguments for S3."""
        args = {}
        
        if metadata:
            args['Metadata'] = metadata
        
        if content_type:
            args['ContentType'] = content_type
        
        if encrypt and self.config.enable_encryption:
            args['ServerSideEncryption'] = 'AES256'
            if self.config.encryption_key:
                args['SSEKMSKeyId'] = self.config.encryption_key
        
        return args
    
    def _extract_encryption_info(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract encryption information from S3 response."""
        encryption_info = {}
        
        if 'ServerSideEncryption' in response:
            encryption_info['algorithm'] = response['ServerSideEncryption']
        
        if 'SSEKMSKeyId' in response:
            encryption_info['key_id'] = response['SSEKMSKeyId']
        
        return encryption_info if encryption_info else None
    
    async def _multipart_upload_file(
        self,
        file_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Perform multipart upload for large files."""
        try:
            # Initialize multipart upload
            extra_args = self._build_upload_args(metadata, content_type, encrypt)
            
            response = await self._s3_client.create_multipart_upload(
                Bucket=self.config.bucket_name,
                Key=key,
                **extra_args
            )
            
            upload_id = response['UploadId']
            
            # Upload parts
            parts = []
            part_size = 100 * 1024 * 1024  # 100MB per part
            part_number = 1
            
            async with aiofiles.open(file_path, 'rb') as f:
                while True:
                    chunk = await f.read(part_size)
                    if not chunk:
                        break
                    
                    part_response = await self._s3_client.upload_part(
                        Bucket=self.config.bucket_name,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk,
                    )
                    
                    parts.append({
                        'ETag': part_response['ETag'],
                        'PartNumber': part_number,
                    })
                    
                    part_number += 1
            
            # Complete multipart upload
            await self._s3_client.complete_multipart_upload(
                Bucket=self.config.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            return await self.get_metadata(key)
            
        except Exception as e:
            # Abort multipart upload on error
            try:
                await self._s3_client.abort_multipart_upload(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    UploadId=upload_id
                )
            except:
                pass
            
            raise CloudStorageError(f"Failed to perform multipart upload: {e}")
