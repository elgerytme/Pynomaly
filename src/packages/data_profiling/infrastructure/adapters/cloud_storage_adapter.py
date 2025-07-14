"""Cloud storage adapters for data profiling."""

import pandas as pd
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


class CloudStorageAdapter(ABC):
    """Abstract base class for cloud storage adapters."""
    
    def __init__(self, connection_config: Dict[str, Any]):
        self.connection_config = connection_config
        self._client = None
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to cloud storage."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to cloud storage."""
        pass
    
    @abstractmethod
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in the storage."""
        pass
    
    @abstractmethod
    def download_object(self, object_key: str, local_path: str) -> None:
        """Download object to local path."""
        pass
    
    @abstractmethod
    def load_object(self, object_key: str, **kwargs) -> pd.DataFrame:
        """Load object directly into DataFrame."""
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class S3Adapter(CloudStorageAdapter):
    """Amazon S3 storage adapter."""
    
    def connect(self) -> None:
        """Establish S3 connection."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Extract connection parameters
            aws_access_key_id = self.connection_config.get('aws_access_key_id')
            aws_secret_access_key = self.connection_config.get('aws_secret_access_key')
            region_name = self.connection_config.get('region_name', 'us-east-1')
            
            if aws_access_key_id and aws_secret_access_key:
                self._client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name
                )
            else:
                # Use default credentials (IAM role, env vars, etc.)
                self._client = boto3.client('s3', region_name=region_name)
            
            # Test connection
            self._client.list_buckets()
            logger.info("Connected to S3 successfully")
            
        except ImportError:
            raise ImportError("boto3 is required for S3 connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to S3: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close S3 connection."""
        self._client = None
        logger.info("Disconnected from S3")
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in S3 bucket."""
        if not self._client:
            raise RuntimeError("Not connected to S3")
        
        bucket_name = self.connection_config.get('bucket_name')
        if not bucket_name:
            raise ValueError("bucket_name is required for S3")
        
        try:
            response = self._client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise
    
    def download_object(self, object_key: str, local_path: str) -> None:
        """Download S3 object to local path."""
        if not self._client:
            raise RuntimeError("Not connected to S3")
        
        bucket_name = self.connection_config.get('bucket_name')
        if not bucket_name:
            raise ValueError("bucket_name is required for S3")
        
        try:
            self._client.download_file(bucket_name, object_key, local_path)
            logger.info(f"Downloaded S3 object {object_key} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download S3 object {object_key}: {e}")
            raise
    
    def load_object(self, object_key: str, **kwargs) -> pd.DataFrame:
        """Load S3 object directly into DataFrame."""
        bucket_name = self.connection_config.get('bucket_name')
        if not bucket_name:
            raise ValueError("bucket_name is required for S3")
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.download_object(object_key, temp_path)
            
            # Determine file format from extension
            file_extension = object_key.lower().split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(temp_path, **kwargs)
            elif file_extension in ['json', 'jsonl']:
                df = pd.read_json(temp_path, **kwargs)
            elif file_extension == 'parquet':
                df = pd.read_parquet(temp_path, **kwargs)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(temp_path, **kwargs)
            else:
                # Try CSV as default
                df = pd.read_csv(temp_path, **kwargs)
            
            logger.info(f"Loaded S3 object {object_key}: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class AzureBlobAdapter(CloudStorageAdapter):
    """Azure Blob Storage adapter."""
    
    def connect(self) -> None:
        """Establish Azure Blob connection."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            connection_string = self.connection_config.get('connection_string')
            account_name = self.connection_config.get('account_name')
            account_key = self.connection_config.get('account_key')
            
            if connection_string:
                self._client = BlobServiceClient.from_connection_string(connection_string)
            elif account_name and account_key:
                account_url = f"https://{account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(account_url=account_url, credential=account_key)
            else:
                raise ValueError("Either connection_string or account_name/account_key required")
            
            # Test connection
            self._client.get_account_information()
            logger.info("Connected to Azure Blob Storage successfully")
            
        except ImportError:
            raise ImportError("azure-storage-blob is required for Azure Blob connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Azure Blob Storage: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Azure Blob connection."""
        self._client = None
        logger.info("Disconnected from Azure Blob Storage")
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List blobs in Azure container."""
        if not self._client:
            raise RuntimeError("Not connected to Azure Blob Storage")
        
        container_name = self.connection_config.get('container_name')
        if not container_name:
            raise ValueError("container_name is required for Azure Blob")
        
        try:
            container_client = self._client.get_container_client(container_name)
            return [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]
        except Exception as e:
            logger.error(f"Failed to list Azure blobs: {e}")
            raise
    
    def download_object(self, object_key: str, local_path: str) -> None:
        """Download Azure blob to local path."""
        if not self._client:
            raise RuntimeError("Not connected to Azure Blob Storage")
        
        container_name = self.connection_config.get('container_name')
        if not container_name:
            raise ValueError("container_name is required for Azure Blob")
        
        try:
            blob_client = self._client.get_blob_client(container=container_name, blob=object_key)
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            logger.info(f"Downloaded Azure blob {object_key} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download Azure blob {object_key}: {e}")
            raise
    
    def load_object(self, object_key: str, **kwargs) -> pd.DataFrame:
        """Load Azure blob directly into DataFrame."""
        container_name = self.connection_config.get('container_name')
        if not container_name:
            raise ValueError("container_name is required for Azure Blob")
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.download_object(object_key, temp_path)
            
            # Determine file format from extension
            file_extension = object_key.lower().split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(temp_path, **kwargs)
            elif file_extension in ['json', 'jsonl']:
                df = pd.read_json(temp_path, **kwargs)
            elif file_extension == 'parquet':
                df = pd.read_parquet(temp_path, **kwargs)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(temp_path, **kwargs)
            else:
                # Try CSV as default
                df = pd.read_csv(temp_path, **kwargs)
            
            logger.info(f"Loaded Azure blob {object_key}: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class GCSAdapter(CloudStorageAdapter):
    """Google Cloud Storage adapter."""
    
    def connect(self) -> None:
        """Establish GCS connection."""
        try:
            from google.cloud import storage
            
            # Use service account key if provided
            credentials_path = self.connection_config.get('credentials_path')
            project_id = self.connection_config.get('project_id')
            
            if credentials_path:
                self._client = storage.Client.from_service_account_json(
                    credentials_path, project=project_id
                )
            else:
                # Use default credentials (environment, metadata server, etc.)
                self._client = storage.Client(project=project_id)
            
            # Test connection
            list(self._client.list_buckets(max_results=1))
            logger.info("Connected to Google Cloud Storage successfully")
            
        except ImportError:
            raise ImportError("google-cloud-storage is required for GCS connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Google Cloud Storage: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close GCS connection."""
        self._client = None
        logger.info("Disconnected from Google Cloud Storage")
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in GCS bucket."""
        if not self._client:
            raise RuntimeError("Not connected to Google Cloud Storage")
        
        bucket_name = self.connection_config.get('bucket_name')
        if not bucket_name:
            raise ValueError("bucket_name is required for GCS")
        
        try:
            bucket = self._client.bucket(bucket_name)
            return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
        except Exception as e:
            logger.error(f"Failed to list GCS objects: {e}")
            raise
    
    def download_object(self, object_key: str, local_path: str) -> None:
        """Download GCS object to local path."""
        if not self._client:
            raise RuntimeError("Not connected to Google Cloud Storage")
        
        bucket_name = self.connection_config.get('bucket_name')
        if not bucket_name:
            raise ValueError("bucket_name is required for GCS")
        
        try:
            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(object_key)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded GCS object {object_key} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download GCS object {object_key}: {e}")
            raise
    
    def load_object(self, object_key: str, **kwargs) -> pd.DataFrame:
        """Load GCS object directly into DataFrame."""
        bucket_name = self.connection_config.get('bucket_name')
        if not bucket_name:
            raise ValueError("bucket_name is required for GCS")
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.download_object(object_key, temp_path)
            
            # Determine file format from extension
            file_extension = object_key.lower().split('.')[-1]
            
            if file_extension == 'csv':
                df = pd.read_csv(temp_path, **kwargs)
            elif file_extension in ['json', 'jsonl']:
                df = pd.read_json(temp_path, **kwargs)
            elif file_extension == 'parquet':
                df = pd.read_parquet(temp_path, **kwargs)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(temp_path, **kwargs)
            else:
                # Try CSV as default
                df = pd.read_csv(temp_path, **kwargs)
            
            logger.info(f"Loaded GCS object {object_key}: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def get_cloud_storage_adapter(provider: str, connection_config: Dict[str, Any]) -> CloudStorageAdapter:
    """Factory function to get the appropriate cloud storage adapter."""
    adapters = {
        's3': S3Adapter,
        'aws': S3Adapter,
        'azure': AzureBlobAdapter,
        'azure_blob': AzureBlobAdapter,
        'gcs': GCSAdapter,
        'google_cloud': GCSAdapter,
        'gcp': GCSAdapter
    }
    
    provider_lower = provider.lower()
    if provider_lower not in adapters:
        raise ValueError(f"Unsupported cloud storage provider: {provider}. Supported: {list(adapters.keys())}")
    
    return adapters[provider_lower](connection_config)


class CloudStorageProfiler:
    """Helper class for profiling cloud storage sources."""
    
    def __init__(self, adapter: CloudStorageAdapter):
        self.adapter = adapter
    
    def profile_objects(self, prefix: str = "", limit: int = 10) -> Dict[str, Any]:
        """Profile objects in cloud storage."""
        try:
            objects = self.adapter.list_objects(prefix)
            
            # Sample some objects for analysis
            sample_objects = objects[:limit]
            profiles = {}
            
            for obj_key in sample_objects:
                try:
                    df = self.adapter.load_object(obj_key)
                    profiles[obj_key] = {
                        'row_count': len(df),
                        'column_count': len(df.columns),
                        'columns': list(df.columns),
                        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                        'data_types': df.dtypes.to_dict(),
                        'null_counts': df.isnull().sum().to_dict(),
                        'success': True
                    }
                except Exception as e:
                    profiles[obj_key] = {
                        'error': str(e),
                        'success': False
                    }
            
            return {
                'total_objects': len(objects),
                'sampled_objects': len(sample_objects),
                'object_profiles': profiles,
                'object_list': objects
            }
            
        except Exception as e:
            logger.error(f"Failed to profile cloud storage objects: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def estimate_storage_size(self, prefix: str = "") -> Dict[str, Any]:
        """Estimate total storage size and object count."""
        try:
            objects = self.adapter.list_objects(prefix)
            
            # Sample a few objects to estimate average size
            sample_size = min(10, len(objects))
            total_sample_size = 0
            successful_samples = 0
            
            for obj_key in objects[:sample_size]:
                try:
                    df = self.adapter.load_object(obj_key)
                    obj_size = df.memory_usage(deep=True).sum()
                    total_sample_size += obj_size
                    successful_samples += 1
                except Exception:
                    continue
            
            if successful_samples > 0:
                avg_object_size = total_sample_size / successful_samples
                estimated_total_size_mb = (avg_object_size * len(objects)) / (1024 * 1024)
            else:
                estimated_total_size_mb = 0
            
            return {
                'total_objects': len(objects),
                'sampled_objects': successful_samples,
                'estimated_total_size_mb': estimated_total_size_mb,
                'average_object_size_mb': (total_sample_size / successful_samples) / (1024 * 1024) if successful_samples > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate storage size: {e}")
            return {
                'error': str(e),
                'success': False
            }