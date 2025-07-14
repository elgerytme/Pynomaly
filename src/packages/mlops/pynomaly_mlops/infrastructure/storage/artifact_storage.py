"""Artifact Storage Service

Model artifact storage with S3-compatible backends and local storage.
"""

import hashlib
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
from uuid import UUID
import pickle
import joblib
import json

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from pynomaly_mlops.domain.entities.model import Model
from pynomaly_mlops.domain.value_objects.semantic_version import SemanticVersion


class ArtifactStorageService(ABC):
    """Abstract base class for model artifact storage."""
    
    @abstractmethod
    async def store_model(self, model: Model, model_artifact: Any) -> str:
        """Store a model artifact.
        
        Args:
            model: Model entity
            model_artifact: The trained model object
            
        Returns:
            URI where the model was stored
        """
        pass
    
    @abstractmethod
    async def load_model(self, artifact_uri: str) -> Any:
        """Load a model artifact.
        
        Args:
            artifact_uri: URI where the model is stored
            
        Returns:
            The loaded model object
        """
        pass
    
    @abstractmethod
    async def delete_model(self, artifact_uri: str) -> bool:
        """Delete a model artifact.
        
        Args:
            artifact_uri: URI where the model is stored
            
        Returns:
            True if deleted successfully, False if not found
        """
        pass
    
    @abstractmethod
    async def get_model_info(self, artifact_uri: str) -> Optional[Dict[str, Any]]:
        """Get model artifact metadata.
        
        Args:
            artifact_uri: URI where the model is stored
            
        Returns:
            Dictionary with size, checksum, modified time, etc.
        """
        pass
    
    @abstractmethod
    async def model_exists(self, artifact_uri: str) -> bool:
        """Check if model artifact exists.
        
        Args:
            artifact_uri: URI where the model should be stored
            
        Returns:
            True if model exists, False otherwise
        """
        pass


class S3ArtifactStorage(ArtifactStorageService):
    """S3-compatible storage for model artifacts."""
    
    def __init__(
        self,
        bucket_name: str,
        region_name: str = 'us-east-1',
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        prefix: str = 'mlops/models'
    ):
        """Initialize S3 artifact storage.
        
        Args:
            bucket_name: S3 bucket name
            region_name: AWS region
            endpoint_url: Custom S3 endpoint (for S3-compatible services)
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            prefix: Prefix for model storage in bucket
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3ArtifactStorage. Install with: pip install boto3")
        
        self.bucket_name = bucket_name
        self.prefix = prefix.strip('/')
        
        # Initialize S3 client
        session_kwargs = {}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })
        
        self.session = boto3.Session(**session_kwargs)
        
        client_kwargs = {'region_name': region_name}
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
        
        self.s3_client = self.session.client('s3', **client_kwargs)
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                except ClientError as create_error:
                    raise RuntimeError(f"Failed to create bucket {self.bucket_name}: {create_error}")
            else:
                raise RuntimeError(f"Failed to access bucket {self.bucket_name}: {e}")
    
    def _get_model_key(self, model: Model) -> str:
        """Generate S3 key for model artifact.
        
        Args:
            model: Model entity
            
        Returns:
            S3 key for the model
        """
        return f"{self.prefix}/{model.name}/{model.version}/{model.id}.pkl"
    
    def _get_metadata_key(self, model: Model) -> str:
        """Generate S3 key for model metadata.
        
        Args:
            model: Model entity
            
        Returns:
            S3 key for the model metadata
        """
        return f"{self.prefix}/{model.name}/{model.version}/{model.id}_metadata.json"
    
    async def store_model(self, model: Model, model_artifact: Any) -> str:
        """Store a model artifact in S3.
        
        Args:
            model: Model entity
            model_artifact: The trained model object
            
        Returns:
            S3 URI where the model was stored
        """
        model_key = self._get_model_key(model)
        metadata_key = self._get_metadata_key(model)
        
        try:
            # Serialize model
            model_bytes = self._serialize_model(model_artifact)
            
            # Calculate checksum
            checksum = hashlib.sha256(model_bytes).hexdigest()
            
            # Store model artifact
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=model_key,
                Body=model_bytes,
                Metadata={
                    'model-id': str(model.id),
                    'model-name': model.name,
                    'model-version': str(model.version),
                    'checksum': checksum,
                    'stored-at': datetime.utcnow().isoformat()
                }
            )
            
            # Store metadata
            metadata = {
                'model_id': str(model.id),
                'model_name': model.name,
                'model_version': str(model.version),
                'model_type': model.model_type.value,
                'checksum': checksum,
                'size_bytes': len(model_bytes),
                'stored_at': datetime.utcnow().isoformat(),
                'serialization_method': 'pickle'
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            return f"s3://{self.bucket_name}/{model_key}"
            
        except Exception as e:
            raise RuntimeError(f"Failed to store model in S3: {e}")
    
    async def load_model(self, artifact_uri: str) -> Any:
        """Load a model artifact from S3.
        
        Args:
            artifact_uri: S3 URI where the model is stored
            
        Returns:
            The loaded model object
        """
        try:
            # Parse S3 URI
            bucket, key = self._parse_s3_uri(artifact_uri)
            
            # Download model bytes
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            model_bytes = response['Body'].read()
            
            # Deserialize model
            return self._deserialize_model(model_bytes)
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Model not found at {artifact_uri}")
            raise RuntimeError(f"Failed to load model from S3: {e}")
    
    async def delete_model(self, artifact_uri: str) -> bool:
        """Delete a model artifact from S3.
        
        Args:
            artifact_uri: S3 URI where the model is stored
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            # Parse S3 URI
            bucket, key = self._parse_s3_uri(artifact_uri)
            
            # Delete model file
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            
            # Delete metadata file
            metadata_key = key.replace('.pkl', '_metadata.json')
            try:
                self.s3_client.delete_object(Bucket=bucket, Key=metadata_key)
            except ClientError:
                pass  # Metadata file might not exist
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return False
            raise RuntimeError(f"Failed to delete model from S3: {e}")
    
    async def get_model_info(self, artifact_uri: str) -> Optional[Dict[str, Any]]:
        """Get model artifact metadata from S3.
        
        Args:
            artifact_uri: S3 URI where the model is stored
            
        Returns:
            Dictionary with size, checksum, modified time, etc.
        """
        try:
            # Parse S3 URI
            bucket, key = self._parse_s3_uri(artifact_uri)
            
            # Get object metadata
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            
            return {
                'size_bytes': response['ContentLength'],
                'last_modified': response['LastModified'],
                'checksum': response.get('Metadata', {}).get('checksum'),
                'stored_at': response.get('Metadata', {}).get('stored-at'),
                'etag': response['ETag'].strip('\"')
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise RuntimeError(f"Failed to get model info from S3: {e}")
    
    async def model_exists(self, artifact_uri: str) -> bool:
        """Check if model artifact exists in S3.
        
        Args:
            artifact_uri: S3 URI where the model should be stored
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            # Parse S3 URI
            bucket, key = self._parse_s3_uri(artifact_uri)
            
            # Check if object exists
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return False
            raise RuntimeError(f"Failed to check model existence in S3: {e}")
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key.
        
        Args:
            s3_uri: S3 URI (s3://bucket/key)
            
        Returns:
            Tuple of (bucket, key)
        """
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        
        path = s3_uri[5:]  # Remove 's3://'
        parts = path.split('/', 1)
        
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        
        return parts[0], parts[1]
    
    def _serialize_model(self, model_artifact: Any) -> bytes:
        """Serialize model to bytes.
        
        Args:
            model_artifact: Model object to serialize
            
        Returns:
            Serialized model bytes
        """
        # Try different serialization methods
        try:
            # First try joblib (better for sklearn models)
            return joblib.dump(model_artifact, compress=3)
        except Exception:
            # Fall back to pickle
            return pickle.dumps(model_artifact)
    
    def _deserialize_model(self, model_bytes: bytes) -> Any:
        """Deserialize model from bytes.
        
        Args:
            model_bytes: Serialized model bytes
            
        Returns:
            Deserialized model object
        """
        try:
            # Try joblib first
            return joblib.load(model_bytes)
        except Exception:
            # Fall back to pickle
            return pickle.loads(model_bytes)


class LocalArtifactStorage(ArtifactStorageService):
    """Local filesystem storage for model artifacts."""
    
    def __init__(self, base_path: str = "./mlops_models"):
        """Initialize local artifact storage.
        
        Args:
            base_path: Base directory for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_model_path(self, model: Model) -> Path:
        """Generate local path for model artifact.
        
        Args:
            model: Model entity
            
        Returns:
            Path for the model file
        """
        model_dir = self.base_path / model.name / str(model.version)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{model.id}.pkl"
    
    def _get_metadata_path(self, model: Model) -> Path:
        """Generate local path for model metadata.
        
        Args:
            model: Model entity
            
        Returns:
            Path for the metadata file
        """
        model_dir = self.base_path / model.name / str(model.version)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / f"{model.id}_metadata.json"
    
    async def store_model(self, model: Model, model_artifact: Any) -> str:
        """Store a model artifact locally.
        
        Args:
            model: Model entity
            model_artifact: The trained model object
            
        Returns:
            File URI where the model was stored
        """
        model_path = self._get_model_path(model)
        metadata_path = self._get_metadata_path(model)
        
        try:
            # Serialize and save model
            with open(model_path, 'wb') as f:
                joblib.dump(model_artifact, f, compress=3)
            
            # Calculate checksum
            with open(model_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Save metadata
            metadata = {
                'model_id': str(model.id),
                'model_name': model.name,
                'model_version': str(model.version),
                'model_type': model.model_type.value,
                'checksum': checksum,
                'size_bytes': model_path.stat().st_size,
                'stored_at': datetime.utcnow().isoformat(),
                'serialization_method': 'joblib'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return f"file://{model_path.absolute()}"
            
        except Exception as e:
            raise RuntimeError(f"Failed to store model locally: {e}")
    
    async def load_model(self, artifact_uri: str) -> Any:
        """Load a model artifact from local storage.
        
        Args:
            artifact_uri: File URI where the model is stored
            
        Returns:
            The loaded model object
        """
        try:
            # Parse file URI
            file_path = self._parse_file_uri(artifact_uri)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Model not found at {artifact_uri}")
            
            # Load model
            with open(file_path, 'rb') as f:
                return joblib.load(f)
                
        except Exception as e:
            if "No such file" in str(e):
                raise FileNotFoundError(f"Model not found at {artifact_uri}")
            raise RuntimeError(f"Failed to load model from local storage: {e}")
    
    async def delete_model(self, artifact_uri: str) -> bool:
        """Delete a model artifact from local storage.
        
        Args:
            artifact_uri: File URI where the model is stored
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            # Parse file URI
            file_path = self._parse_file_uri(artifact_uri)
            
            if not file_path.exists():
                return False
            
            # Delete model file
            file_path.unlink()
            
            # Delete metadata file
            metadata_path = file_path.with_suffix('.json').with_name(
                file_path.stem + '_metadata.json'
            )
            if metadata_path.exists():
                metadata_path.unlink()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete model from local storage: {e}")
    
    async def get_model_info(self, artifact_uri: str) -> Optional[Dict[str, Any]]:
        """Get model artifact metadata from local storage.
        
        Args:
            artifact_uri: File URI where the model is stored
            
        Returns:
            Dictionary with size, checksum, modified time, etc.
        """
        try:
            # Parse file URI
            file_path = self._parse_file_uri(artifact_uri)
            
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            
            # Try to load metadata file
            metadata_path = file_path.with_suffix('.json').with_name(
                file_path.stem + '_metadata.json'
            )
            
            checksum = None
            stored_at = None
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    checksum = metadata.get('checksum')
                    stored_at = metadata.get('stored_at')
            
            return {
                'size_bytes': stat.st_size,
                'last_modified': datetime.fromtimestamp(stat.st_mtime),
                'checksum': checksum,
                'stored_at': stored_at
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get model info from local storage: {e}")
    
    async def model_exists(self, artifact_uri: str) -> bool:
        """Check if model artifact exists in local storage.
        
        Args:
            artifact_uri: File URI where the model should be stored
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            # Parse file URI
            file_path = self._parse_file_uri(artifact_uri)
            return file_path.exists()
            
        except Exception:
            return False
    
    def _parse_file_uri(self, file_uri: str) -> Path:
        """Parse file URI into Path.
        
        Args:
            file_uri: File URI (file://path)
            
        Returns:
            Path object
        """
        if file_uri.startswith('file://'):
            return Path(file_uri[7:])
        else:
            # Assume it's already a path
            return Path(file_uri)