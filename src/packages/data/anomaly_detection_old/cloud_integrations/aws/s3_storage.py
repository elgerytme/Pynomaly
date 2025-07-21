"""
AWS S3 Storage Manager for Pynomaly Detection
==============================================

Manages S3 storage for data, models, and artifacts with optimized performance.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass
from datetime import datetime, timedelta
import tempfile

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class S3StorageConfig:
    """S3 storage configuration."""
    bucket_name: str
    region: str = "us-east-1"
    data_prefix: str = "data/"
    model_prefix: str = "models/"
    artifact_prefix: str = "artifacts/"
    cache_prefix: str = "cache/"
    backup_prefix: str = "backups/"
    encryption: str = "AES256"
    versioning: bool = True
    lifecycle_enabled: bool = True
    
class S3StorageManager:
    """AWS S3 storage management for Pynomaly Detection."""
    
    def __init__(self, config: S3StorageConfig, profile_name: Optional[str] = None):
        """Initialize S3 storage manager.
        
        Args:
            config: S3 storage configuration
            profile_name: AWS profile name (optional)
        """
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK (boto3) is required for S3 integration")
        
        self.config = config
        self.profile_name = profile_name
        
        # Initialize AWS clients
        session = boto3.Session(profile_name=profile_name)
        self.s3_client = session.client('s3', region_name=config.region)
        self.s3_resource = session.resource('s3', region_name=config.region)
        
        # Initialize bucket
        self._ensure_bucket_exists()
        
        logger.info(f"S3 Storage Manager initialized for bucket: {config.bucket_name}")
    
    def save_model(self, model_data: Any, model_name: str, 
                   version: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Save model to S3 with versioning and metadata.
        
        Args:
            model_data: Model object to save
            model_name: Model name
            version: Model version (auto-generated if None)
            metadata: Additional metadata
            
        Returns:
            S3 key of saved model
        """
        try:
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare model data
            model_key = f"{self.config.model_prefix}{model_name}/v{version}/model.pkl"
            metadata_key = f"{self.config.model_prefix}{model_name}/v{version}/metadata.json"
            
            # Save model
            with tempfile.NamedTemporaryFile() as tmp_file:
                pickle.dump(model_data, tmp_file)
                tmp_file.flush()
                
                self.s3_client.upload_file(
                    tmp_file.name,
                    self.config.bucket_name,
                    model_key,
                    ExtraArgs={
                        'ServerSideEncryption': self.config.encryption,
                        'Metadata': {
                            'model_name': model_name,
                            'version': version,
                            'saved_at': datetime.now().isoformat(),
                            'type': 'pynomaly_model'
                        }
                    }
                )
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'model_name': model_name,
                'version': version,
                'saved_at': datetime.now().isoformat(),
                'model_key': model_key,
                'type': 'pynomaly_model'
            })
            
            self.s3_client.put_object(
                Bucket=self.config.bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json',
                ServerSideEncryption=self.config.encryption
            )
            
            logger.info(f"Model saved to S3: {model_key}")
            return model_key
            
        except Exception as e:
            logger.error(f"Failed to save model to S3: {e}")
            raise
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Load model from S3.
        
        Args:
            model_name: Model name
            version: Model version (latest if None)
            
        Returns:
            Loaded model object
        """
        try:
            if version is None:
                version = self._get_latest_model_version(model_name)
            
            model_key = f"{self.config.model_prefix}{model_name}/v{version}/model.pkl"
            
            with tempfile.NamedTemporaryFile() as tmp_file:
                self.s3_client.download_file(
                    self.config.bucket_name,
                    model_key,
                    tmp_file.name
                )
                
                with open(tmp_file.name, 'rb') as f:
                    model_data = pickle.load(f)
            
            logger.info(f"Model loaded from S3: {model_key}")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load model from S3: {e}")
            raise
    
    def save_data(self, data: Union[np.ndarray, pd.DataFrame, Dict], 
                  data_name: str, format: str = "parquet") -> str:
        """Save data to S3 in optimized format.
        
        Args:
            data: Data to save
            data_name: Data name
            format: Storage format (parquet, csv, json, pickle)
            
        Returns:
            S3 key of saved data
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "parquet" and NUMPY_AVAILABLE:
                data_key = f"{self.config.data_prefix}{data_name}/{timestamp}.parquet"
                
                with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
                    if isinstance(data, pd.DataFrame):
                        data.to_parquet(tmp_file.name, compression='snappy')
                    elif isinstance(data, np.ndarray):
                        pd.DataFrame(data).to_parquet(tmp_file.name, compression='snappy')
                    else:
                        raise ValueError(f"Unsupported data type for parquet: {type(data)}")
                    
                    self.s3_client.upload_file(
                        tmp_file.name,
                        self.config.bucket_name,
                        data_key,
                        ExtraArgs={
                            'ServerSideEncryption': self.config.encryption,
                            'Metadata': {
                                'data_name': data_name,
                                'format': format,
                                'saved_at': timestamp,
                                'type': 'pynomaly_data'
                            }
                        }
                    )
            
            elif format == "csv" and NUMPY_AVAILABLE:
                data_key = f"{self.config.data_prefix}{data_name}/{timestamp}.csv"
                
                with tempfile.NamedTemporaryFile(mode='w', suffix=".csv") as tmp_file:
                    if isinstance(data, pd.DataFrame):
                        data.to_csv(tmp_file.name, index=False)
                    elif isinstance(data, np.ndarray):
                        pd.DataFrame(data).to_csv(tmp_file.name, index=False)
                    else:
                        raise ValueError(f"Unsupported data type for CSV: {type(data)}")
                    
                    self.s3_client.upload_file(
                        tmp_file.name,
                        self.config.bucket_name,
                        data_key,
                        ExtraArgs={'ServerSideEncryption': self.config.encryption}
                    )
            
            elif format == "json":
                data_key = f"{self.config.data_prefix}{data_name}/{timestamp}.json"
                
                if isinstance(data, dict):
                    json_data = json.dumps(data, indent=2)
                elif NUMPY_AVAILABLE and isinstance(data, (np.ndarray, pd.DataFrame)):
                    json_data = data.to_json() if hasattr(data, 'to_json') else json.dumps(data.tolist())
                else:
                    json_data = json.dumps(data)
                
                self.s3_client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=data_key,
                    Body=json_data,
                    ContentType='application/json',
                    ServerSideEncryption=self.config.encryption
                )
            
            elif format == "pickle":
                data_key = f"{self.config.data_prefix}{data_name}/{timestamp}.pkl"
                
                with tempfile.NamedTemporaryFile() as tmp_file:
                    pickle.dump(data, tmp_file)
                    tmp_file.flush()
                    
                    self.s3_client.upload_file(
                        tmp_file.name,
                        self.config.bucket_name,
                        data_key,
                        ExtraArgs={'ServerSideEncryption': self.config.encryption}
                    )
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data saved to S3: {data_key}")
            return data_key
            
        except Exception as e:
            logger.error(f"Failed to save data to S3: {e}")
            raise
    
    def load_data(self, data_key: str, format: str = None) -> Any:
        """Load data from S3.
        
        Args:
            data_key: S3 key of data
            format: Data format (auto-detected if None)
            
        Returns:
            Loaded data
        """
        try:
            if format is None:
                format = self._detect_format(data_key)
            
            if format == "parquet" and NUMPY_AVAILABLE:
                with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp_file:
                    self.s3_client.download_file(
                        self.config.bucket_name,
                        data_key,
                        tmp_file.name
                    )
                    data = pd.read_parquet(tmp_file.name)
            
            elif format == "csv" and NUMPY_AVAILABLE:
                with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv") as tmp_file:
                    self.s3_client.download_file(
                        self.config.bucket_name,
                        data_key,
                        tmp_file.name
                    )
                    data = pd.read_csv(tmp_file.name)
            
            elif format == "json":
                response = self.s3_client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=data_key
                )
                data = json.loads(response['Body'].read().decode('utf-8'))
            
            elif format == "pickle":
                with tempfile.NamedTemporaryFile() as tmp_file:
                    self.s3_client.download_file(
                        self.config.bucket_name,
                        data_key,
                        tmp_file.name
                    )
                    
                    with open(tmp_file.name, 'rb') as f:
                        data = pickle.load(f)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data loaded from S3: {data_key}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from S3: {e}")
            raise
    
    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available models in S3.
        
        Args:
            model_name: Filter by model name (optional)
            
        Returns:
            List of model information
        """
        try:
            prefix = self.config.model_prefix
            if model_name:
                prefix += f"{model_name}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )
            
            models = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('metadata.json'):
                    try:
                        metadata_response = self.s3_client.get_object(
                            Bucket=self.config.bucket_name,
                            Key=obj['Key']
                        )
                        metadata = json.loads(metadata_response['Body'].read().decode('utf-8'))
                        
                        models.append({
                            'model_name': metadata.get('model_name'),
                            'version': metadata.get('version'),
                            'saved_at': metadata.get('saved_at'),
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'key': obj['Key']
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {obj['Key']}: {e}")
            
            return sorted(models, key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete model from S3.
        
        Args:
            model_name: Model name
            version: Model version (all versions if None)
            
        Returns:
            True if deletion successful
        """
        try:
            if version:
                # Delete specific version
                prefix = f"{self.config.model_prefix}{model_name}/v{version}/"
            else:
                # Delete all versions
                prefix = f"{self.config.model_prefix}{model_name}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=prefix
            )
            
            objects_to_delete = []
            for obj in response.get('Contents', []):
                objects_to_delete.append({'Key': obj['Key']})
            
            if objects_to_delete:
                self.s3_client.delete_objects(
                    Bucket=self.config.bucket_name,
                    Delete={'Objects': objects_to_delete}
                )
                
                logger.info(f"Deleted {len(objects_to_delete)} objects for model: {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def create_backup(self, backup_name: str, include_models: bool = True, 
                     include_data: bool = True) -> str:
        """Create backup of S3 data.
        
        Args:
            backup_name: Backup name
            include_models: Include models in backup
            include_data: Include data in backup
            
        Returns:
            Backup S3 key
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_key = f"{self.config.backup_prefix}{backup_name}/{timestamp}/"
            
            # Create backup manifest
            manifest = {
                'backup_name': backup_name,
                'created_at': timestamp,
                'include_models': include_models,
                'include_data': include_data,
                'objects': []
            }
            
            # Copy models
            if include_models:
                models_response = self.s3_client.list_objects_v2(
                    Bucket=self.config.bucket_name,
                    Prefix=self.config.model_prefix
                )
                
                for obj in models_response.get('Contents', []):
                    source_key = obj['Key']
                    dest_key = f"{backup_key}models/{source_key[len(self.config.model_prefix):]}"
                    
                    self.s3_client.copy_object(
                        CopySource={'Bucket': self.config.bucket_name, 'Key': source_key},
                        Bucket=self.config.bucket_name,
                        Key=dest_key
                    )
                    
                    manifest['objects'].append({
                        'source': source_key,
                        'destination': dest_key,
                        'type': 'model'
                    })
            
            # Copy data
            if include_data:
                data_response = self.s3_client.list_objects_v2(
                    Bucket=self.config.bucket_name,
                    Prefix=self.config.data_prefix
                )
                
                for obj in data_response.get('Contents', []):
                    source_key = obj['Key']
                    dest_key = f"{backup_key}data/{source_key[len(self.config.data_prefix):]}"
                    
                    self.s3_client.copy_object(
                        CopySource={'Bucket': self.config.bucket_name, 'Key': source_key},
                        Bucket=self.config.bucket_name,
                        Key=dest_key
                    )
                    
                    manifest['objects'].append({
                        'source': source_key,
                        'destination': dest_key,
                        'type': 'data'
                    })
            
            # Save manifest
            manifest_key = f"{backup_key}manifest.json"
            self.s3_client.put_object(
                Bucket=self.config.bucket_name,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType='application/json',
                ServerSideEncryption=self.config.encryption
            )
            
            logger.info(f"Backup created: {backup_key}")
            return backup_key
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage metrics and usage statistics.
        
        Returns:
            Storage metrics
        """
        try:
            metrics = {
                'bucket_name': self.config.bucket_name,
                'region': self.config.region,
                'total_objects': 0,
                'total_size': 0,
                'by_prefix': {}
            }
            
            # Get bucket statistics
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.bucket_name
            )
            
            for obj in response.get('Contents', []):
                key = obj['Key']
                size = obj['Size']
                
                metrics['total_objects'] += 1
                metrics['total_size'] += size
                
                # Categorize by prefix
                prefix = key.split('/')[0] + '/'
                if prefix not in metrics['by_prefix']:
                    metrics['by_prefix'][prefix] = {'objects': 0, 'size': 0}
                
                metrics['by_prefix'][prefix]['objects'] += 1
                metrics['by_prefix'][prefix]['size'] += size
            
            # Format sizes
            metrics['total_size_mb'] = round(metrics['total_size'] / (1024 * 1024), 2)
            metrics['total_size_gb'] = round(metrics['total_size'] / (1024 * 1024 * 1024), 2)
            
            for prefix_data in metrics['by_prefix'].values():
                prefix_data['size_mb'] = round(prefix_data['size'] / (1024 * 1024), 2)
                prefix_data['size_gb'] = round(prefix_data['size'] / (1024 * 1024 * 1024), 2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get storage metrics: {e}")
            return {}
    
    def _ensure_bucket_exists(self):
        """Ensure S3 bucket exists with proper configuration."""
        try:
            self.s3_client.head_bucket(Bucket=self.config.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Create bucket
                create_config = {}
                if self.config.region != 'us-east-1':
                    create_config['CreateBucketConfiguration'] = {
                        'LocationConstraint': self.config.region
                    }
                
                self.s3_client.create_bucket(
                    Bucket=self.config.bucket_name,
                    **create_config
                )
                
                # Enable versioning
                if self.config.versioning:
                    self.s3_client.put_bucket_versioning(
                        Bucket=self.config.bucket_name,
                        VersioningConfiguration={'Status': 'Enabled'}
                    )
                
                # Configure lifecycle
                if self.config.lifecycle_enabled:
                    self._configure_lifecycle()
                
                logger.info(f"Created S3 bucket: {self.config.bucket_name}")
            else:
                raise
    
    def _configure_lifecycle(self):
        """Configure S3 lifecycle rules."""
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'Pynomaly-DataLifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': self.config.data_prefix},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                },
                {
                    'ID': 'Pynomaly-CacheLifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': self.config.cache_prefix},
                    'Expiration': {'Days': 7}
                }
            ]
        }
        
        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket=self.config.bucket_name,
            LifecycleConfiguration=lifecycle_config
        )
    
    def _get_latest_model_version(self, model_name: str) -> str:
        """Get latest version of a model."""
        models = self.list_models(model_name)
        if not models:
            raise ValueError(f"No models found for: {model_name}")
        
        return models[0]['version']
    
    def _detect_format(self, key: str) -> str:
        """Detect file format from S3 key."""
        extension = key.split('.')[-1].lower()
        
        format_map = {
            'parquet': 'parquet',
            'csv': 'csv', 
            'json': 'json',
            'pkl': 'pickle',
            'pickle': 'pickle'
        }
        
        return format_map.get(extension, 'pickle')