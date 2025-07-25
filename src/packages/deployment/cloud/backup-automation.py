#!/usr/bin/env python3
"""
Cloud Backup Automation System
Comprehensive backup and disaster recovery for cloud environments
"""

import asyncio
import boto3
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml
from botocore.exceptions import ClientError, NoCredentialsError


@dataclass
class BackupConfig:
    """Backup configuration"""
    name: str
    source_type: str  # kubernetes, database, filesystem, application
    source_path: str
    destination: str
    schedule: str  # cron format
    retention_days: int = 30
    compression: bool = True
    encryption: bool = True
    enabled: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BackupResult:
    """Backup operation result"""
    backup_id: str
    config_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, success, failed
    size_bytes: int = 0
    location: str = ""
    error_message: Optional[str] = None
    checksum: Optional[str] = None


class CloudBackupAutomation:
    """Main cloud backup automation system"""
    
    def __init__(self, config_path: str = "config/backup-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.backup_configs: Dict[str, BackupConfig] = {}
        self.backup_results: Dict[str, BackupResult] = {}
        
        # Initialize cloud clients
        self.s3_client = None
        self.ec2_client = None
        self.rds_client = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'/tmp/backup-automation-{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self._initialize_cloud_clients()
        self._initialize_backup_configs()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load backup configuration"""
        default_config = {
            "aws": {
                "region": "us-west-2",
                "s3_bucket": "hexagonal-backups",
                "encryption_key_id": "",
                "storage_class": "STANDARD_IA"
            },
            "kubernetes": {
                "namespace": "production",
                "backup_pvcs": True,
                "backup_secrets": True,
                "backup_configmaps": True
            },
            "database": {
                "postgresql": {
                    "host": "prod-db.company.com",
                    "databases": ["hexagonal_prod", "analytics_prod"],
                    "backup_format": "custom"
                },
                "redis": {
                    "host": "prod-redis.company.com",
                    "backup_rdb": True
                }
            },
            "retention": {
                "daily_backups": 7,
                "weekly_backups": 4,
                "monthly_backups": 12,
                "yearly_backups": 3
            },
            "monitoring": {
                "slack_webhook": os.getenv("SLACK_WEBHOOK_URL", ""),
                "email_recipients": ["devops@company.com"]
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return {**default_config, **user_config}
        
        return default_config
    
    def _initialize_cloud_clients(self):
        """Initialize cloud service clients"""
        try:
            # AWS clients
            self.s3_client = boto3.client('s3', region_name=self.config["aws"]["region"])
            self.ec2_client = boto3.client('ec2', region_name=self.config["aws"]["region"])
            self.rds_client = boto3.client('rds', region_name=self.config["aws"]["region"])
            
            # Verify S3 bucket exists
            self._ensure_s3_bucket()
            
            self.logger.info("✅ Cloud clients initialized successfully")
            
        except NoCredentialsError:
            self.logger.error("❌ AWS credentials not configured")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize cloud clients: {e}")
    
    def _ensure_s3_bucket(self):
        """Ensure S3 backup bucket exists"""
        bucket_name = self.config["aws"]["s3_bucket"]
        
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"S3 bucket verified: {bucket_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Create bucket
                try:
                    if self.config["aws"]["region"] == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.config["aws"]["region"]}
                        )
                    
                    # Enable versioning
                    self.s3_client.put_bucket_versioning(
                        Bucket=bucket_name,
                        VersioningConfiguration={'Status': 'Enabled'}
                    )
                    
                    # Set lifecycle policy
                    self._set_s3_lifecycle_policy(bucket_name)
                    
                    self.logger.info(f"Created S3 bucket: {bucket_name}")
                    
                except ClientError as create_error:
                    self.logger.error(f"Failed to create S3 bucket: {create_error}")
            else:
                self.logger.error(f"S3 bucket access error: {e}")
    
    def _set_s3_lifecycle_policy(self, bucket_name: str):
        """Set S3 lifecycle policy for backup retention"""
        lifecycle_policy = {
            'Rules': [
                {
                    'ID': 'backup-retention-policy',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'backups/'},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        },
                        {
                            'Days': 365,
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ],
                    'Expiration': {
                        'Days': self.config["retention"]["yearly_backups"] * 365
                    }
                }
            ]
        }
        
        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_policy
        )
    
    def _initialize_backup_configs(self):
        """Initialize backup configurations"""
        
        # Kubernetes backup
        self.backup_configs["kubernetes_manifests"] = BackupConfig(
            name="kubernetes_manifests",
            source_type="kubernetes",
            source_path="all-namespaces",
            destination=f"s3://{self.config['aws']['s3_bucket']}/kubernetes/",
            schedule="0 2 * * *",  # Daily at 2 AM
            retention_days=30,
            metadata={"backup_type": "manifests"}
        )
        
        # Database backups
        for db_name in self.config["database"]["postgresql"]["databases"]:
            self.backup_configs[f"postgresql_{db_name}"] = BackupConfig(
                name=f"postgresql_{db_name}",
                source_type="database",
                source_path=f"postgresql://{self.config['database']['postgresql']['host']}/{db_name}",
                destination=f"s3://{self.config['aws']['s3_bucket']}/databases/postgresql/",
                schedule="0 1 * * *",  # Daily at 1 AM
                retention_days=90,
                metadata={"database_type": "postgresql", "database_name": db_name}
            )
        
        # Redis backup
        self.backup_configs["redis_backup"] = BackupConfig(
            name="redis_backup",
            source_type="database",
            source_path=f"redis://{self.config['database']['redis']['host']}",
            destination=f"s3://{self.config['aws']['s3_bucket']}/databases/redis/",
            schedule="0 3 * * *",  # Daily at 3 AM
            retention_days=30,
            metadata={"database_type": "redis"}
        )
        
        # Application data backup
        self.backup_configs["application_logs"] = BackupConfig(
            name="application_logs",
            source_type="filesystem",
            source_path="/var/log/applications/",
            destination=f"s3://{self.config['aws']['s3_bucket']}/logs/",
            schedule="0 4 * * *",  # Daily at 4 AM
            retention_days=90,
            metadata={"backup_type": "logs"}
        )
        
        # Persistent volume backups
        self.backup_configs["persistent_volumes"] = BackupConfig(
            name="persistent_volumes",
            source_type="kubernetes",
            source_path="pvc",
            destination=f"s3://{self.config['aws']['s3_bucket']}/volumes/",
            schedule="0 0 * * 0",  # Weekly on Sunday
            retention_days=180,
            metadata={"backup_type": "persistent_volumes"}
        )
    
    async def create_backup(self, config_name: str) -> BackupResult:
        """Create backup based on configuration"""
        if config_name not in self.backup_configs:
            raise ValueError(f"Backup configuration not found: {config_name}")
        
        config = self.backup_configs[config_name]
        
        if not config.enabled:
            self.logger.info(f"Backup {config_name} is disabled, skipping")
            return None
        
        backup_id = f"{config_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        result = BackupResult(
            backup_id=backup_id,
            config_name=config_name,
            start_time=datetime.now()
        )
        
        self.logger.info(f"Starting backup: {backup_id}")
        
        try:
            # Dispatch to appropriate backup method
            if config.source_type == "kubernetes":
                await self._backup_kubernetes(config, result)
            elif config.source_type == "database":
                await self._backup_database(config, result)
            elif config.source_type == "filesystem":
                await self._backup_filesystem(config, result)
            elif config.source_type == "application":
                await self._backup_application(config, result)
            else:
                raise ValueError(f"Unknown backup source type: {config.source_type}")
            
            result.status = "success"
            result.end_time = datetime.now()
            
            # Calculate backup metrics
            await self._calculate_backup_metrics(result)
            
            # Upload backup manifest
            await self._upload_backup_manifest(result, config)
            
            self.logger.info(f"✅ Backup completed successfully: {backup_id}")
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.end_time = datetime.now()
            
            self.logger.error(f"❌ Backup failed: {backup_id} - {e}")
        
        # Store result
        self.backup_results[backup_id] = result
        
        # Send notification
        await self._send_backup_notification(result, config)
        
        return result
    
    async def _backup_kubernetes(self, config: BackupConfig, result: BackupResult):
        """Backup Kubernetes resources"""
        backup_dir = f"/tmp/k8s-backup-{result.backup_id}"
        os.makedirs(backup_dir, exist_ok=True)
        
        try:
            if config.metadata.get("backup_type") == "manifests":
                # Backup all Kubernetes manifests
                await self._backup_k8s_manifests(backup_dir)
            elif config.metadata.get("backup_type") == "persistent_volumes":
                # Backup persistent volumes using snapshots
                await self._backup_k8s_volumes(backup_dir)
            
            # Compress backup
            archive_path = f"{backup_dir}.tar.gz"
            subprocess.run([
                "tar", "-czf", archive_path, "-C", backup_dir, "."
            ], check=True)
            
            # Upload to S3
            s3_key = f"kubernetes/{result.backup_id}.tar.gz"
            await self._upload_to_s3(archive_path, s3_key)
            
            result.location = f"s3://{self.config['aws']['s3_bucket']}/{s3_key}"
            result.size_bytes = os.path.getsize(archive_path)
            
        finally:
            # Cleanup temporary files
            subprocess.run(["rm", "-rf", backup_dir, f"{backup_dir}.tar.gz"], check=False)
    
    async def _backup_k8s_manifests(self, backup_dir: str):
        """Backup Kubernetes manifests"""
        namespace = self.config["kubernetes"]["namespace"]
        
        # Backup deployments
        subprocess.run([
            "kubectl", "get", "deployments", "-n", namespace, "-o", "yaml"
        ], stdout=open(f"{backup_dir}/deployments.yaml", "w"), check=True)
        
        # Backup services
        subprocess.run([
            "kubectl", "get", "services", "-n", namespace, "-o", "yaml"
        ], stdout=open(f"{backup_dir}/services.yaml", "w"), check=True)
        
        # Backup configmaps
        if self.config["kubernetes"]["backup_configmaps"]:
            subprocess.run([
                "kubectl", "get", "configmaps", "-n", namespace, "-o", "yaml"
            ], stdout=open(f"{backup_dir}/configmaps.yaml", "w"), check=True)
        
        # Backup secrets (excluding default service account tokens)
        if self.config["kubernetes"]["backup_secrets"]:
            subprocess.run([
                "kubectl", "get", "secrets", "-n", namespace, 
                "--field-selector", "type!=kubernetes.io/service-account-token",
                "-o", "yaml"
            ], stdout=open(f"{backup_dir}/secrets.yaml", "w"), check=True)
        
        # Backup persistent volume claims
        if self.config["kubernetes"]["backup_pvcs"]:
            subprocess.run([
                "kubectl", "get", "pvc", "-n", namespace, "-o", "yaml"
            ], stdout=open(f"{backup_dir}/pvcs.yaml", "w"), check=True)
    
    async def _backup_k8s_volumes(self, backup_dir: str):
        """Backup Kubernetes persistent volumes using volume snapshots"""
        namespace = self.config["kubernetes"]["namespace"]
        
        # Get all PVCs
        result = subprocess.run([
            "kubectl", "get", "pvc", "-n", namespace, "-o", "json"
        ], capture_output=True, text=True, check=True)
        
        pvcs = json.loads(result.stdout)
        
        snapshots = []
        
        for pvc in pvcs.get("items", []):
            pvc_name = pvc["metadata"]["name"]
            snapshot_name = f"{pvc_name}-snapshot-{int(time.time())}"
            
            # Create volume snapshot
            snapshot_manifest = {
                "apiVersion": "snapshot.storage.k8s.io/v1",
                "kind": "VolumeSnapshot",
                "metadata": {
                    "name": snapshot_name,
                    "namespace": namespace
                },
                "spec": {
                    "source": {
                        "persistentVolumeClaimName": pvc_name
                    }
                }
            }
            
            # Apply snapshot manifest
            with open(f"{backup_dir}/{snapshot_name}.yaml", "w") as f:
                yaml.dump(snapshot_manifest, f)
            
            subprocess.run([
                "kubectl", "apply", "-f", f"{backup_dir}/{snapshot_name}.yaml"
            ], check=True)
            
            snapshots.append(snapshot_name)
        
        # Wait for snapshots to complete
        for snapshot_name in snapshots:
            await self._wait_for_snapshot_ready(snapshot_name, namespace)
        
        # Save snapshot manifest
        with open(f"{backup_dir}/snapshots.json", "w") as f:
            json.dump(snapshots, f)
    
    async def _wait_for_snapshot_ready(self, snapshot_name: str, namespace: str, timeout: int = 300):
        """Wait for volume snapshot to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = subprocess.run([
                "kubectl", "get", "volumesnapshot", snapshot_name, "-n", namespace, "-o", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                snapshot = json.loads(result.stdout)
                if snapshot.get("status", {}).get("readyToUse"):
                    return True
            
            await asyncio.sleep(10)
        
        raise Exception(f"Snapshot {snapshot_name} not ready within timeout")
    
    async def _backup_database(self, config: BackupConfig, result: BackupResult):
        """Backup database"""
        if config.metadata["database_type"] == "postgresql":
            await self._backup_postgresql(config, result)
        elif config.metadata["database_type"] == "redis":
            await self._backup_redis(config, result)
        else:
            raise ValueError(f"Unsupported database type: {config.metadata['database_type']}")
    
    async def _backup_postgresql(self, config: BackupConfig, result: BackupResult):
        """Backup PostgreSQL database"""
        db_name = config.metadata["database_name"]
        backup_file = f"/tmp/{result.backup_id}.sql"
        
        # Create database dump
        env = os.environ.copy()
        env["PGPASSWORD"] = os.getenv("POSTGRES_PASSWORD", "")
        
        subprocess.run([
            "pg_dump",
            "-h", self.config["database"]["postgresql"]["host"],
            "-U", os.getenv("POSTGRES_USER", "postgres"),
            "-d", db_name,
            "-f", backup_file,
            "--format=custom",
            "--no-owner",
            "--no-privileges"
        ], env=env, check=True)
        
        # Compress if enabled
        if config.compression:
            compressed_file = f"{backup_file}.gz"
            subprocess.run(["gzip", backup_file], check=True)
            backup_file = compressed_file
        
        # Upload to S3
        s3_key = f"databases/postgresql/{db_name}/{result.backup_id}.sql{'gz' if config.compression else ''}"
        await self._upload_to_s3(backup_file, s3_key)
        
        result.location = f"s3://{self.config['aws']['s3_bucket']}/{s3_key}"
        result.size_bytes = os.path.getsize(backup_file)
        
        # Cleanup
        os.remove(backup_file)
    
    async def _backup_redis(self, config: BackupConfig, result: BackupResult):
        """Backup Redis database"""
        backup_file = f"/tmp/{result.backup_id}.rdb"
        
        # Create Redis backup using BGSAVE
        subprocess.run([
            "redis-cli",
            "-h", self.config["database"]["redis"]["host"],
            "--rdb", backup_file
        ], check=True)
        
        # Compress if enabled
        if config.compression:
            compressed_file = f"{backup_file}.gz"
            subprocess.run(["gzip", backup_file], check=True)
            backup_file = compressed_file
        
        # Upload to S3
        s3_key = f"databases/redis/{result.backup_id}.rdb{'gz' if config.compression else ''}"
        await self._upload_to_s3(backup_file, s3_key)
        
        result.location = f"s3://{self.config['aws']['s3_bucket']}/{s3_key}"
        result.size_bytes = os.path.getsize(backup_file)
        
        # Cleanup
        os.remove(backup_file)
    
    async def _backup_filesystem(self, config: BackupConfig, result: BackupResult):
        """Backup filesystem directory"""
        if not os.path.exists(config.source_path):
            raise ValueError(f"Source path does not exist: {config.source_path}")
        
        backup_file = f"/tmp/{result.backup_id}.tar.gz"
        
        # Create compressed archive
        subprocess.run([
            "tar", "-czf", backup_file, "-C", os.path.dirname(config.source_path),
            os.path.basename(config.source_path)
        ], check=True)
        
        # Upload to S3
        s3_key = f"filesystem/{result.backup_id}.tar.gz"
        await self._upload_to_s3(backup_file, s3_key)
        
        result.location = f"s3://{self.config['aws']['s3_bucket']}/{s3_key}"
        result.size_bytes = os.path.getsize(backup_file)
        
        # Cleanup
        os.remove(backup_file)
    
    async def _backup_application(self, config: BackupConfig, result: BackupResult):
        """Backup application-specific data"""
        # This would be customized based on application requirements
        self.logger.info(f"Application backup not implemented for: {config.name}")
        result.status = "skipped"
    
    async def _upload_to_s3(self, file_path: str, s3_key: str):
        """Upload file to S3 with encryption"""
        extra_args = {}
        
        # Add encryption if enabled
        if self.config["aws"].get("encryption_key_id"):
            extra_args["ServerSideEncryption"] = "aws:kms"
            extra_args["SSEKMSKeyId"] = self.config["aws"]["encryption_key_id"]
        else:
            extra_args["ServerSideEncryption"] = "AES256"
        
        # Set storage class
        extra_args["StorageClass"] = self.config["aws"].get("storage_class", "STANDARD")
        
        # Upload file
        self.s3_client.upload_file(
            file_path,
            self.config["aws"]["s3_bucket"],
            s3_key,
            ExtraArgs=extra_args
        )
    
    async def _calculate_backup_metrics(self, result: BackupResult):
        """Calculate backup metrics"""
        if result.start_time and result.end_time:
            duration = (result.end_time - result.start_time).total_seconds()
            result.metadata = result.metadata or {}
            result.metadata["duration_seconds"] = duration
            
            if result.size_bytes > 0:
                result.metadata["throughput_mbps"] = (result.size_bytes / 1024 / 1024) / duration
    
    async def _upload_backup_manifest(self, result: BackupResult, config: BackupConfig):
        """Upload backup manifest with metadata"""
        manifest = {
            "backup_id": result.backup_id,
            "config_name": result.config_name,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "status": result.status,
            "size_bytes": result.size_bytes,
            "location": result.location,
            "checksum": result.checksum,
            "config": asdict(config),
            "metadata": getattr(result, 'metadata', {})
        }
        
        manifest_json = json.dumps(manifest, indent=2)
        manifest_key = f"manifests/{result.backup_id}.json"
        
        # Upload manifest
        self.s3_client.put_object(
            Bucket=self.config["aws"]["s3_bucket"],
            Key=manifest_key,
            Body=manifest_json,
            ContentType="application/json"
        )
    
    async def _send_backup_notification(self, result: BackupResult, config: BackupConfig):
        """Send backup notification"""
        if result.status == "success":
            message = f"✅ Backup successful: {result.backup_id}"
            color = "good"
        else:
            message = f"❌ Backup failed: {result.backup_id}"
            color = "danger"
        
        # Slack notification
        if self.config["monitoring"]["slack_webhook"]:
            await self._send_slack_notification(message, result, color)
        
        # Email notification (for failures)
        if result.status == "failed" and self.config["monitoring"]["email_recipients"]:
            await self._send_email_notification(result, config)
    
    async def _send_slack_notification(self, message: str, result: BackupResult, color: str):
        """Send Slack notification"""
        import aiohttp
        
        webhook_url = self.config["monitoring"]["slack_webhook"]
        
        payload = {
            "attachments": [{
                "color": color,
                "title": message,
                "fields": [
                    {"title": "Backup ID", "value": result.backup_id, "short": True},
                    {"title": "Status", "value": result.status, "short": True},
                    {"title": "Size", "value": f"{result.size_bytes / 1024 / 1024:.2f} MB", "short": True},
                    {"title": "Location", "value": result.location, "short": False}
                ],
                "footer": "Cloud Backup System"
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=payload)
    
    async def _send_email_notification(self, result: BackupResult, config: BackupConfig):
        """Send email notification for backup failures"""
        # Email notification implementation would go here
        self.logger.info(f"Would send email notification for failed backup: {result.backup_id}")
    
    async def restore_backup(self, backup_id: str, target_location: str = None) -> bool:
        """Restore backup from cloud storage"""
        self.logger.info(f"Starting restore for backup: {backup_id}")
        
        try:
            # Download backup manifest
            manifest = await self._download_backup_manifest(backup_id)
            
            if not manifest:
                raise ValueError(f"Backup manifest not found: {backup_id}")
            
            # Download backup file
            backup_file = await self._download_backup_file(manifest)
            
            # Restore based on backup type
            config = BackupConfig(**manifest["config"])
            
            if config.source_type == "kubernetes":
                await self._restore_kubernetes(backup_file, config, target_location)
            elif config.source_type == "database":
                await self._restore_database(backup_file, config, target_location)
            elif config.source_type == "filesystem":
                await self._restore_filesystem(backup_file, config, target_location)
            
            self.logger.info(f"✅ Restore completed successfully: {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Restore failed: {backup_id} - {e}")
            return False
    
    async def _download_backup_manifest(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Download backup manifest"""
        try:
            manifest_key = f"manifests/{backup_id}.json"
            
            response = self.s3_client.get_object(
                Bucket=self.config["aws"]["s3_bucket"],
                Key=manifest_key
            )
            
            manifest_data = response["Body"].read().decode('utf-8')
            return json.loads(manifest_data)
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise
    
    async def _download_backup_file(self, manifest: Dict[str, Any]) -> str:
        """Download backup file from S3"""
        location = manifest["location"]
        if not location.startswith("s3://"):
            raise ValueError(f"Unsupported backup location: {location}")
        
        # Parse S3 location
        s3_path = location[5:]  # Remove 's3://'
        bucket, key = s3_path.split('/', 1)
        
        # Download file
        local_file = f"/tmp/restore-{manifest['backup_id']}"
        
        self.s3_client.download_file(bucket, key, local_file)
        
        return local_file
    
    async def _restore_kubernetes(self, backup_file: str, config: BackupConfig, target_location: str):
        """Restore Kubernetes resources"""
        # Extract backup
        extract_dir = f"/tmp/k8s-restore-{int(time.time())}"
        os.makedirs(extract_dir, exist_ok=True)
        
        subprocess.run([
            "tar", "-xzf", backup_file, "-C", extract_dir
        ], check=True)
        
        # Apply Kubernetes manifests
        for yaml_file in os.listdir(extract_dir):
            if yaml_file.endswith('.yaml'):
                subprocess.run([
                    "kubectl", "apply", "-f", os.path.join(extract_dir, yaml_file)
                ], check=True)
        
        # Cleanup
        subprocess.run(["rm", "-rf", extract_dir], check=False)
        os.remove(backup_file)
    
    async def _restore_database(self, backup_file: str, config: BackupConfig, target_location: str):
        """Restore database"""
        if config.metadata["database_type"] == "postgresql":
            await self._restore_postgresql(backup_file, config, target_location)
        elif config.metadata["database_type"] == "redis":
            await self._restore_redis(backup_file, config, target_location)
    
    async def _restore_postgresql(self, backup_file: str, config: BackupConfig, target_location: str):
        """Restore PostgreSQL database"""
        # Decompress if needed
        if backup_file.endswith('.gz'):
            subprocess.run(["gunzip", backup_file], check=True)
            backup_file = backup_file[:-3]  # Remove .gz extension
        
        # Restore database
        db_name = target_location or config.metadata["database_name"]
        
        env = os.environ.copy()
        env["PGPASSWORD"] = os.getenv("POSTGRES_PASSWORD", "")
        
        subprocess.run([
            "pg_restore",
            "-h", self.config["database"]["postgresql"]["host"],
            "-U", os.getenv("POSTGRES_USER", "postgres"),
            "-d", db_name,
            "--clean",
            "--if-exists",
            backup_file
        ], env=env, check=True)
        
        # Cleanup
        os.remove(backup_file)
    
    async def _restore_redis(self, backup_file: str, config: BackupConfig, target_location: str):
        """Restore Redis database"""
        # Implementation for Redis restore
        self.logger.info("Redis restore not fully implemented")
    
    async def _restore_filesystem(self, backup_file: str, config: BackupConfig, target_location: str):
        """Restore filesystem"""
        restore_path = target_location or config.source_path
        
        # Extract backup
        subprocess.run([
            "tar", "-xzf", backup_file, "-C", os.path.dirname(restore_path)
        ], check=True)
        
        # Cleanup
        os.remove(backup_file)
    
    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        self.logger.info("Starting backup cleanup...")
        
        try:
            # List all backup manifests
            response = self.s3_client.list_objects_v2(
                Bucket=self.config["aws"]["s3_bucket"],
                Prefix="manifests/"
            )
            
            current_time = datetime.now()
            deleted_count = 0
            
            for obj in response.get('Contents', []):
                manifest_key = obj['Key']
                
                # Download and parse manifest
                try:
                    manifest_response = self.s3_client.get_object(
                        Bucket=self.config["aws"]["s3_bucket"],
                        Key=manifest_key
                    )
                    
                    manifest_data = manifest_response["Body"].read().decode('utf-8')
                    manifest = json.loads(manifest_data)
                    
                    # Check if backup is older than retention period
                    backup_time = datetime.fromisoformat(manifest["start_time"].replace('Z', '+00:00'))
                    config_name = manifest["config_name"]
                    
                    if config_name in self.backup_configs:
                        retention_days = self.backup_configs[config_name].retention_days
                        
                        if (current_time - backup_time).days > retention_days:
                            # Delete backup and manifest
                            await self._delete_backup(manifest)
                            deleted_count += 1
                
                except Exception as e:
                    self.logger.warning(f"Failed to process manifest {manifest_key}: {e}")
            
            self.logger.info(f"✅ Cleanup completed: {deleted_count} old backups deleted")
            
        except Exception as e:
            self.logger.error(f"❌ Backup cleanup failed: {e}")
    
    async def _delete_backup(self, manifest: Dict[str, Any]):
        """Delete backup and its manifest"""
        backup_id = manifest["backup_id"]
        
        # Delete backup file
        location = manifest["location"]
        if location.startswith("s3://"):
            s3_path = location[5:]
            bucket, key = s3_path.split('/', 1)
            
            self.s3_client.delete_object(Bucket=bucket, Key=key)
        
        # Delete manifest
        manifest_key = f"manifests/{backup_id}.json"
        self.s3_client.delete_object(
            Bucket=self.config["aws"]["s3_bucket"],
            Key=manifest_key
        )
        
        self.logger.info(f"Deleted backup: {backup_id}")
    
    def list_backups(self, config_name: str = None) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.config["aws"]["s3_bucket"],
                Prefix="manifests/"
            )
            
            for obj in response.get('Contents', []):
                manifest_key = obj['Key']
                
                try:
                    manifest_response = self.s3_client.get_object(
                        Bucket=self.config["aws"]["s3_bucket"],
                        Key=manifest_key
                    )
                    
                    manifest_data = manifest_response["Body"].read().decode('utf-8')
                    manifest = json.loads(manifest_data)
                    
                    # Filter by config name if specified
                    if config_name and manifest["config_name"] != config_name:
                        continue
                    
                    backups.append(manifest)
                
                except Exception as e:
                    self.logger.warning(f"Failed to read manifest {manifest_key}: {e}")
        
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["start_time"], reverse=True)
        
        return backups
    
    def generate_backup_report(self) -> str:
        """Generate backup status report"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("CLOUD BACKUP STATUS REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Configuration summary
        report_lines.append("Backup Configurations:")
        for name, config in self.backup_configs.items():
            status = "✅ Enabled" if config.enabled else "❌ Disabled"
            report_lines.append(f"  {name:<25} {status:<12} {config.schedule}")
        
        report_lines.append("")
        
        # Recent backup results
        recent_results = sorted(
            self.backup_results.values(),
            key=lambda x: x.start_time,
            reverse=True
        )[:10]
        
        if recent_results:
            report_lines.append("Recent Backup Results:")
            for result in recent_results:
                status_symbol = "✅" if result.status == "success" else "❌"
                size_mb = result.size_bytes / 1024 / 1024 if result.size_bytes else 0
                report_lines.append(f"  {status_symbol} {result.backup_id:<40} {size_mb:>8.2f} MB")
        
        report_lines.append("")
        
        # Storage summary
        try:
            bucket_size = self._get_bucket_size()
            report_lines.append(f"Total Storage Used: {bucket_size / 1024 / 1024 / 1024:.2f} GB")
        except Exception:
            report_lines.append("Total Storage Used: Unable to calculate")
        
        return "\n".join(report_lines)
    
    def _get_bucket_size(self) -> int:
        """Get total size of backup bucket"""
        total_size = 0
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.config["aws"]["s3_bucket"]):
            for obj in page.get('Contents', []):
                total_size += obj['Size']
        
        return total_size


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cloud Backup Automation")
    parser.add_argument("--config", default="config/backup-config.yaml", help="Configuration file")
    parser.add_argument("--backup", help="Create backup for specific configuration")
    parser.add_argument("--backup-all", action="store_true", help="Create all scheduled backups")
    parser.add_argument("--restore", help="Restore backup by ID")
    parser.add_argument("--target", help="Target location for restore")
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old backups")
    parser.add_argument("--report", action="store_true", help="Generate backup status report")
    args = parser.parse_args()
    
    backup_system = CloudBackupAutomation(args.config)
    
    if args.report:
        report = backup_system.generate_backup_report()
        print(report)
        return
    
    if args.list:
        backups = backup_system.list_backups()
        print("Available Backups:")
        print("-" * 80)
        for backup in backups:
            print(f"{backup['backup_id']:<40} {backup['status']:<10} {backup['start_time']}")
        return
    
    if args.cleanup:
        await backup_system.cleanup_old_backups()
        return
    
    if args.restore:
        success = await backup_system.restore_backup(args.restore, args.target)
        exit(0 if success else 1)
    
    if args.backup:
        result = await backup_system.create_backup(args.backup)
        if result and result.status == "success":
            print(f"✅ Backup completed: {result.backup_id}")
        else:
            print(f"❌ Backup failed: {result.error_message if result else 'Unknown error'}")
            exit(1)
    
    if args.backup_all:
        for config_name in backup_system.backup_configs:
            result = await backup_system.create_backup(config_name)
            if result:
                status_symbol = "✅" if result.status == "success" else "❌"
                print(f"{status_symbol} {config_name}: {result.status}")


if __name__ == "__main__":
    asyncio.run(main())