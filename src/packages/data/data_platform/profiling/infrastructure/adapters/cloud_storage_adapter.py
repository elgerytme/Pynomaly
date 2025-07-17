import pandas as pd
import io
from typing import Dict, Any, Optional, List
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CloudStorageAdapter(ABC):
    """Abstract base class for cloud storage adapters."""
    
    def __init__(self, credentials: Dict[str, Any], **kwargs):
        self.credentials = credentials
        self.kwargs = kwargs
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to cloud storage."""
        pass
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in the storage."""
        pass
    
    @abstractmethod
    def download_file(self, file_path: str) -> bytes:
        """Download file content as bytes."""
        pass
    
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from cloud storage file."""
        pass


class S3Adapter(CloudStorageAdapter):
    """Amazon S3 storage adapter."""
    
    def __init__(self, credentials: Dict[str, Any], bucket_name: str, **kwargs):
        super().__init__(credentials, **kwargs)
        self.bucket_name = bucket_name
        self.client = None
        self.s3_resource = None
    
    def connect(self) -> None:
        """Establish connection to S3."""
        try:
            import boto3
            
            # Create S3 client and resource
            self.client = boto3.client(
                's3',
                aws_access_key_id=self.credentials.get('access_key_id'),
                aws_secret_access_key=self.credentials.get('secret_access_key'),
                aws_session_token=self.credentials.get('session_token'),
                region_name=self.credentials.get('region', 'us-east-1')
            )
            
            self.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=self.credentials.get('access_key_id'),
                aws_secret_access_key=self.credentials.get('secret_access_key'),
                aws_session_token=self.credentials.get('session_token'),
                region_name=self.credentials.get('region', 'us-east-1')
            )
            
            # Test connection
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Connected to S3 bucket: {self.bucket_name}")
            
        except ImportError:
            raise ImportError("boto3 is required for S3 connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to S3: {e}")
            raise
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3 bucket."""
        if not self.client:
            raise RuntimeError("Not connected to S3")
        
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            raise
    
    def download_file(self, file_path: str) -> bytes:
        """Download file from S3."""
        if not self.client:
            raise RuntimeError("Not connected to S3")
        
        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            
            return response['Body'].read()
            
        except Exception as e:
            logger.error(f"Failed to download file from S3: {e}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from S3 file."""
        try:
            # Download file content
            file_content = self.download_file(file_path)
            
            # Determine file format from extension
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return pd.read_csv(io.BytesIO(file_content), **self.kwargs)
            elif file_extension == 'json':
                return pd.read_json(io.BytesIO(file_content), **self.kwargs)
            elif file_extension in ['parquet', 'pq']:
                return pd.read_parquet(io.BytesIO(file_content), **self.kwargs)
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(io.BytesIO(file_content), **self.kwargs)
            else:
                # Default to CSV
                return pd.read_csv(io.BytesIO(file_content), **self.kwargs)
                
        except Exception as e:
            logger.error(f"Failed to load data from S3: {e}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get metadata about S3 file."""
        if not self.client:
            raise RuntimeError("Not connected to S3")
        
        try:
            response = self.client.head_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            
            return {
                'file_path': file_path,
                'bucket': self.bucket_name,
                'size_bytes': response['ContentLength'],
                'size_mb': response['ContentLength'] / (1024 * 1024),
                'last_modified': response['LastModified'],
                'etag': response['ETag'],
                'content_type': response.get('ContentType', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Failed to get S3 file info: {e}")
            return {}


class AzureBlobAdapter(CloudStorageAdapter):
    """Azure Blob Storage adapter."""
    
    def __init__(self, credentials: Dict[str, Any], container_name: str, **kwargs):
        super().__init__(credentials, **kwargs)
        self.container_name = container_name
        self.blob_service_client = None
        self.container_client = None
    
    def connect(self) -> None:
        """Establish connection to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Create blob service client
            if 'connection_string' in self.credentials:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.credentials['connection_string']
                )
            else:
                account_url = f"https://{self.credentials['account_name']}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.credentials.get('account_key')
                )
            
            # Get container client
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            
            # Test connection
            _ = self.container_client.get_container_properties()
            logger.info(f"Connected to Azure Blob container: {self.container_name}")
            
        except ImportError:
            raise ImportError("azure-storage-blob is required for Azure Blob connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Azure Blob: {e}")
            raise
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in Azure Blob container."""
        if not self.container_client:
            raise RuntimeError("Not connected to Azure Blob")
        
        try:
            blob_list = self.container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blob_list]
            
        except Exception as e:
            logger.error(f"Failed to list Azure Blob files: {e}")
            raise
    
    def download_file(self, file_path: str) -> bytes:
        """Download file from Azure Blob."""
        if not self.container_client:
            raise RuntimeError("Not connected to Azure Blob")
        
        try:
            blob_client = self.container_client.get_blob_client(file_path)
            return blob_client.download_blob().readall()
            
        except Exception as e:
            logger.error(f"Failed to download file from Azure Blob: {e}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from Azure Blob file."""
        try:
            # Download file content
            file_content = self.download_file(file_path)
            
            # Determine file format from extension
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return pd.read_csv(io.BytesIO(file_content), **self.kwargs)
            elif file_extension == 'json':
                return pd.read_json(io.BytesIO(file_content), **self.kwargs)
            elif file_extension in ['parquet', 'pq']:
                return pd.read_parquet(io.BytesIO(file_content), **self.kwargs)
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(io.BytesIO(file_content), **self.kwargs)
            else:
                # Default to CSV
                return pd.read_csv(io.BytesIO(file_content), **self.kwargs)
                
        except Exception as e:
            logger.error(f"Failed to load data from Azure Blob: {e}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get metadata about Azure Blob file."""
        if not self.container_client:
            raise RuntimeError("Not connected to Azure Blob")
        
        try:
            blob_client = self.container_client.get_blob_client(file_path)
            properties = blob_client.get_blob_properties()
            
            return {
                'file_path': file_path,
                'container': self.container_name,
                'size_bytes': properties.size,
                'size_mb': properties.size / (1024 * 1024),
                'last_modified': properties.last_modified,
                'etag': properties.etag,
                'content_type': properties.content_settings.content_type
            }
            
        except Exception as e:
            logger.error(f"Failed to get Azure Blob file info: {e}")
            return {}


class GCSAdapter(CloudStorageAdapter):
    """Google Cloud Storage adapter."""
    
    def __init__(self, credentials: Dict[str, Any], bucket_name: str, **kwargs):
        super().__init__(credentials, **kwargs)
        self.bucket_name = bucket_name
        self.client = None
        self.bucket = None
    
    def connect(self) -> None:
        """Establish connection to Google Cloud Storage."""
        try:
            from google.cloud import storage
            
            # Create client
            if 'service_account_path' in self.credentials:
                self.client = storage.Client.from_service_account_json(
                    self.credentials['service_account_path']
                )
            else:
                self.client = storage.Client()
            
            # Get bucket
            self.bucket = self.client.bucket(self.bucket_name)
            
            # Test connection
            _ = self.bucket.exists()
            logger.info(f"Connected to GCS bucket: {self.bucket_name}")
            
        except ImportError:
            raise ImportError("google-cloud-storage is required for GCS connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to GCS: {e}")
            raise
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in GCS bucket."""
        if not self.bucket:
            raise RuntimeError("Not connected to GCS")
        
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
            
        except Exception as e:
            logger.error(f"Failed to list GCS files: {e}")
            raise
    
    def download_file(self, file_path: str) -> bytes:
        """Download file from GCS."""
        if not self.bucket:
            raise RuntimeError("Not connected to GCS")
        
        try:
            blob = self.bucket.blob(file_path)
            return blob.download_as_bytes()
            
        except Exception as e:
            logger.error(f"Failed to download file from GCS: {e}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from GCS file."""
        try:
            # Download file content
            file_content = self.download_file(file_path)
            
            # Determine file format from extension
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                return pd.read_csv(io.BytesIO(file_content), **self.kwargs)
            elif file_extension == 'json':
                return pd.read_json(io.BytesIO(file_content), **self.kwargs)
            elif file_extension in ['parquet', 'pq']:
                return pd.read_parquet(io.BytesIO(file_content), **self.kwargs)
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(io.BytesIO(file_content), **self.kwargs)
            else:
                # Default to CSV
                return pd.read_csv(io.BytesIO(file_content), **self.kwargs)
                
        except Exception as e:
            logger.error(f"Failed to load data from GCS: {e}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get metadata about GCS file."""
        if not self.bucket:
            raise RuntimeError("Not connected to GCS")
        
        try:
            blob = self.bucket.blob(file_path)
            blob.reload()
            
            return {
                'file_path': file_path,
                'bucket': self.bucket_name,
                'size_bytes': blob.size,
                'size_mb': blob.size / (1024 * 1024),
                'last_modified': blob.time_created,
                'etag': blob.etag,
                'content_type': blob.content_type
            }
            
        except Exception as e:
            logger.error(f"Failed to get GCS file info: {e}")
            return {}


def get_cloud_storage_adapter(provider: str, credentials: Dict[str, Any], 
                             container_name: str, **kwargs) -> CloudStorageAdapter:
    """Factory function to get the appropriate cloud storage adapter."""
    providers = {
        'aws': S3Adapter,
        's3': S3Adapter,
        'azure': AzureBlobAdapter,
        'gcs': GCSAdapter,
        'google': GCSAdapter
    }
    
    provider_lower = provider.lower()
    if provider_lower not in providers:
        raise ValueError(f"Unsupported cloud provider: {provider}. Supported: {list(providers.keys())}")
    
    if provider_lower in ['aws', 's3']:
        return providers[provider_lower](credentials, container_name, **kwargs)
    elif provider_lower == 'azure':
        return providers[provider_lower](credentials, container_name, **kwargs)
    elif provider_lower in ['gcs', 'google']:
        return providers[provider_lower](credentials, container_name, **kwargs)


class CloudDataProfiler:
    """Helper class for profiling cloud storage data sources."""
    
    def __init__(self, adapter: CloudStorageAdapter):
        self.adapter = adapter
    
    def profile_file(self, file_path: str) -> pd.DataFrame:
        """Profile a specific cloud storage file."""
        return self.adapter.load_data(file_path)
    
    def profile_multiple_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Profile multiple cloud storage files."""
        dataframes = []
        
        for file_path in file_paths:
            try:
                df = self.adapter.load_data(file_path)
                df['_source_file'] = file_path
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to load file {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No files could be loaded successfully")
        
        return pd.concat(dataframes, ignore_index=True, sort=False)
    
    def get_storage_info(self, prefix: str = "") -> Dict[str, Any]:
        """Get information about cloud storage."""
        files = self.adapter.list_files(prefix)
        
        total_size = 0
        file_types = {}
        
        for file_path in files[:100]:  # Limit to first 100 files for performance
            try:
                file_info = self.adapter.get_file_info(file_path)
                total_size += file_info.get('size_bytes', 0)
                
                # Count file types
                extension = file_path.split('.')[-1].lower()
                file_types[extension] = file_types.get(extension, 0) + 1
                
            except Exception as e:
                logger.warning(f"Failed to get info for {file_path}: {e}")
                continue
        
        return {
            'total_files': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_types': file_types,
            'sample_files': files[:10]
        }