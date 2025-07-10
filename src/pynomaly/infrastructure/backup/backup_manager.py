#!/usr/bin/env python3
"""
Backup and Recovery Manager for Pynomaly

This module provides comprehensive backup and recovery capabilities
for databases, files, configurations, and application state.
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import boto3
import paramiko
import yaml
from cryptography.fernet import Fernet


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CompressionType(Enum):
    """Compression types for backups."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    TAR_XZ = "tar.xz"


@dataclass
class BackupMetadata:
    """Metadata for backup operations."""
    backup_id: str
    backup_type: BackupType
    source_path: str
    destination_path: str
    timestamp: datetime
    status: BackupStatus
    size_bytes: int = 0
    checksum: str = ""
    compression: CompressionType = CompressionType.GZIP
    encryption_enabled: bool = False
    retention_days: int = 30
    tags: dict[str, str] = field(default_factory=dict)
    error_message: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "compression": self.compression.value,
            "encryption_enabled": self.encryption_enabled,
            "retention_days": self.retention_days,
            "tags": self.tags,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds
        }


class BackupProvider(ABC):
    """Base class for backup providers."""

    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def upload_backup(self, local_path: str, remote_path: str, metadata: BackupMetadata) -> bool:
        """Upload backup to remote storage."""
        pass

    @abstractmethod
    async def download_backup(self, remote_path: str, local_path: str) -> bool:
        """Download backup from remote storage."""
        pass

    @abstractmethod
    async def list_backups(self, prefix: str = "") -> list[dict[str, Any]]:
        """List available backups."""
        pass

    @abstractmethod
    async def delete_backup(self, remote_path: str) -> bool:
        """Delete backup from remote storage."""
        pass


class LocalBackupProvider(BackupProvider):
    """Local filesystem backup provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("local", config)
        self.backup_root = Path(config.get("backup_directory", "/var/backups/pynomaly"))
        self.backup_root.mkdir(parents=True, exist_ok=True)

    async def upload_backup(self, local_path: str, remote_path: str, metadata: BackupMetadata) -> bool:
        """Copy backup to local backup directory."""
        try:
            destination = self.backup_root / remote_path
            destination.parent.mkdir(parents=True, exist_ok=True)

            if Path(local_path).is_dir():
                shutil.copytree(local_path, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(local_path, destination)

            # Save metadata
            metadata_path = destination.with_suffix(destination.suffix + ".meta")
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)

            return True

        except Exception as e:
            self.logger.error(f"Failed to upload backup: {e}")
            return False

    async def download_backup(self, remote_path: str, local_path: str) -> bool:
        """Copy backup from local backup directory."""
        try:
            source = self.backup_root / remote_path

            if not source.exists():
                self.logger.error(f"Backup not found: {source}")
                return False

            if source.is_dir():
                shutil.copytree(source, local_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source, local_path)

            return True

        except Exception as e:
            self.logger.error(f"Failed to download backup: {e}")
            return False

    async def list_backups(self, prefix: str = "") -> list[dict[str, Any]]:
        """List local backups."""
        backups = []

        try:
            search_path = self.backup_root / prefix if prefix else self.backup_root

            for backup_file in search_path.rglob("*.meta"):
                try:
                    with open(backup_file) as f:
                        metadata = json.load(f)
                    backups.append(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata {backup_file}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")

        return backups

    async def delete_backup(self, remote_path: str) -> bool:
        """Delete local backup."""
        try:
            backup_path = self.backup_root / remote_path
            metadata_path = backup_path.with_suffix(backup_path.suffix + ".meta")

            if backup_path.exists():
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()

            if metadata_path.exists():
                metadata_path.unlink()

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete backup: {e}")
            return False


class S3BackupProvider(BackupProvider):
    """Amazon S3 backup provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("s3", config)
        self.bucket_name = config["bucket_name"]
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=config.get("access_key_id"),
            aws_secret_access_key=config.get("secret_access_key"),
            region_name=config.get("region", "us-east-1")
        )

    async def upload_backup(self, local_path: str, remote_path: str, metadata: BackupMetadata) -> bool:
        """Upload backup to S3."""
        try:
            # Upload file
            self.s3_client.upload_file(local_path, self.bucket_name, remote_path)

            # Upload metadata
            metadata_key = f"{remote_path}.meta"
            metadata_content = json.dumps(metadata.to_dict(), indent=2, default=str)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=metadata_content,
                ContentType="application/json"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to upload to S3: {e}")
            return False

    async def download_backup(self, remote_path: str, local_path: str) -> bool:
        """Download backup from S3."""
        try:
            self.s3_client.download_file(self.bucket_name, remote_path, local_path)
            return True

        except Exception as e:
            self.logger.error(f"Failed to download from S3: {e}")
            return False

    async def list_backups(self, prefix: str = "") -> list[dict[str, Any]]:
        """List S3 backups."""
        backups = []

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            for obj in response.get("Contents", []):
                if obj["Key"].endswith(".meta"):
                    try:
                        # Get metadata
                        metadata_response = self.s3_client.get_object(
                            Bucket=self.bucket_name,
                            Key=obj["Key"]
                        )
                        metadata = json.loads(metadata_response["Body"].read())
                        backups.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"Failed to read metadata {obj['Key']}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to list S3 backups: {e}")

        return backups

    async def delete_backup(self, remote_path: str) -> bool:
        """Delete backup from S3."""
        try:
            # Delete backup file
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=remote_path)

            # Delete metadata
            metadata_key = f"{remote_path}.meta"
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=metadata_key)

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete from S3: {e}")
            return False


class SFTPBackupProvider(BackupProvider):
    """SFTP backup provider."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("sftp", config)
        self.hostname = config["hostname"]
        self.port = config.get("port", 22)
        self.username = config["username"]
        self.password = config.get("password")
        self.private_key_path = config.get("private_key_path")
        self.remote_directory = config.get("remote_directory", "/backups")

    def _get_ssh_client(self):
        """Get SSH client connection."""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if self.private_key_path:
            key = paramiko.RSAKey.from_private_key_file(self.private_key_path)
            ssh.connect(self.hostname, port=self.port, username=self.username, pkey=key)
        else:
            ssh.connect(self.hostname, port=self.port, username=self.username, password=self.password)

        return ssh

    async def upload_backup(self, local_path: str, remote_path: str, metadata: BackupMetadata) -> bool:
        """Upload backup via SFTP."""
        try:
            ssh = self._get_ssh_client()
            sftp = ssh.open_sftp()

            remote_full_path = f"{self.remote_directory}/{remote_path}"

            # Create remote directory
            try:
                sftp.makedirs(str(Path(remote_full_path).parent))
            except OSError:
                pass  # Directory might already exist

            # Upload file
            sftp.put(local_path, remote_full_path)

            # Upload metadata
            metadata_path = f"{remote_full_path}.meta"
            metadata_content = json.dumps(metadata.to_dict(), indent=2, default=str)

            with sftp.open(metadata_path, "w") as f:
                f.write(metadata_content)

            sftp.close()
            ssh.close()

            return True

        except Exception as e:
            self.logger.error(f"Failed to upload via SFTP: {e}")
            return False

    async def download_backup(self, remote_path: str, local_path: str) -> bool:
        """Download backup via SFTP."""
        try:
            ssh = self._get_ssh_client()
            sftp = ssh.open_sftp()

            remote_full_path = f"{self.remote_directory}/{remote_path}"
            sftp.get(remote_full_path, local_path)

            sftp.close()
            ssh.close()

            return True

        except Exception as e:
            self.logger.error(f"Failed to download via SFTP: {e}")
            return False

    async def list_backups(self, prefix: str = "") -> list[dict[str, Any]]:
        """List SFTP backups."""
        backups = []

        try:
            ssh = self._get_ssh_client()
            sftp = ssh.open_sftp()

            # List files recursively
            def list_files(path):
                try:
                    for item in sftp.listdir_attr(path):
                        item_path = f"{path}/{item.filename}"
                        if item.filename.endswith(".meta"):
                            try:
                                with sftp.open(item_path, "r") as f:
                                    metadata = json.loads(f.read())
                                backups.append(metadata)
                            except Exception as e:
                                self.logger.warning(f"Failed to read metadata {item_path}: {e}")
                        elif sftp.stat(item_path).st_mode & 0o040000:  # Directory
                            list_files(item_path)
                except OSError:
                    pass

            search_path = f"{self.remote_directory}/{prefix}" if prefix else self.remote_directory
            list_files(search_path)

            sftp.close()
            ssh.close()

        except Exception as e:
            self.logger.error(f"Failed to list SFTP backups: {e}")

        return backups

    async def delete_backup(self, remote_path: str) -> bool:
        """Delete backup via SFTP."""
        try:
            ssh = self._get_ssh_client()
            sftp = ssh.open_sftp()

            remote_full_path = f"{self.remote_directory}/{remote_path}"
            metadata_path = f"{remote_full_path}.meta"

            # Delete files
            try:
                sftp.remove(remote_full_path)
            except OSError:
                pass

            try:
                sftp.remove(metadata_path)
            except OSError:
                pass

            sftp.close()
            ssh.close()

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete via SFTP: {e}")
            return False


class DatabaseBackupHandler:
    """Handle database-specific backup operations."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def backup_postgresql(self, backup_path: str, metadata: BackupMetadata) -> bool:
        """Backup PostgreSQL database."""
        try:
            db_config = self.config.get("postgresql", {})

            cmd = [
                "pg_dump",
                f"--host={db_config.get('host', 'localhost')}",
                f"--port={db_config.get('port', 5432)}",
                f"--username={db_config.get('username', 'postgres')}",
                f"--dbname={db_config.get('database')}",
                "--no-password",
                "--verbose",
                f"--file={backup_path}"
            ]

            # Set password environment variable
            env = os.environ.copy()
            if db_config.get("password"):
                env["PGPASSWORD"] = db_config["password"]

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                metadata.size_bytes = Path(backup_path).stat().st_size
                metadata.checksum = self._calculate_checksum(backup_path)
                return True
            else:
                metadata.error_message = result.stderr
                return False

        except Exception as e:
            metadata.error_message = str(e)
            self.logger.error(f"PostgreSQL backup failed: {e}")
            return False

    async def backup_mysql(self, backup_path: str, metadata: BackupMetadata) -> bool:
        """Backup MySQL database."""
        try:
            db_config = self.config.get("mysql", {})

            cmd = [
                "mysqldump",
                f"--host={db_config.get('host', 'localhost')}",
                f"--port={db_config.get('port', 3306)}",
                f"--user={db_config.get('username', 'root')}",
                "--single-transaction",
                "--routines",
                "--triggers",
                db_config.get("database")
            ]

            if db_config.get("password"):
                cmd.append(f"--password={db_config['password']}")

            with open(backup_path, "w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                metadata.size_bytes = Path(backup_path).stat().st_size
                metadata.checksum = self._calculate_checksum(backup_path)
                return True
            else:
                metadata.error_message = result.stderr
                return False

        except Exception as e:
            metadata.error_message = str(e)
            self.logger.error(f"MySQL backup failed: {e}")
            return False

    async def backup_mongodb(self, backup_path: str, metadata: BackupMetadata) -> bool:
        """Backup MongoDB database."""
        try:
            db_config = self.config.get("mongodb", {})

            # Create backup directory
            backup_dir = Path(backup_path).parent / "mongodb_backup"
            backup_dir.mkdir(exist_ok=True)

            cmd = [
                "mongodump",
                f"--host={db_config.get('host', 'localhost')}:{db_config.get('port', 27017)}",
                f"--db={db_config.get('database')}",
                f"--out={backup_dir}"
            ]

            if db_config.get("username"):
                cmd.extend([
                    f"--username={db_config['username']}",
                    f"--password={db_config.get('password', '')}"
                ])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Create archive
                shutil.make_archive(backup_path.replace(".tar.gz", ""), "gztar", backup_dir)
                shutil.rmtree(backup_dir)

                metadata.size_bytes = Path(backup_path).stat().st_size
                metadata.checksum = self._calculate_checksum(backup_path)
                return True
            else:
                metadata.error_message = result.stderr
                return False

        except Exception as e:
            metadata.error_message = str(e)
            self.logger.error(f"MongoDB backup failed: {e}")
            return False

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()


class FileBackupHandler:
    """Handle file and directory backup operations."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize encryption if configured
        self.encryption_key = None
        if config.get("encryption", {}).get("enabled"):
            key_file = config["encryption"].get("key_file")
            if key_file and Path(key_file).exists():
                with open(key_file, "rb") as f:
                    self.encryption_key = f.read()
            else:
                self.encryption_key = Fernet.generate_key()
                if key_file:
                    with open(key_file, "wb") as f:
                        f.write(self.encryption_key)

    async def backup_directory(self, source_path: str, backup_path: str,
                             metadata: BackupMetadata) -> bool:
        """Backup directory with compression and encryption."""
        try:
            source = Path(source_path)
            backup_file = Path(backup_path)

            if not source.exists():
                metadata.error_message = f"Source path does not exist: {source_path}"
                return False

            # Create temporary archive
            temp_path = backup_file.with_suffix(".tmp")

            if metadata.compression == CompressionType.ZIP:
                success = await self._create_zip_archive(source, temp_path)
            elif metadata.compression == CompressionType.TAR_GZ:
                success = await self._create_tar_archive(source, temp_path, "gz")
            elif metadata.compression == CompressionType.TAR_XZ:
                success = await self._create_tar_archive(source, temp_path, "xz")
            else:
                success = await self._create_tar_archive(source, temp_path, "gz")

            if not success:
                metadata.error_message = "Failed to create archive"
                return False

            # Encrypt if enabled
            if metadata.encryption_enabled and self.encryption_key:
                encrypted_path = await self._encrypt_file(temp_path)
                temp_path.unlink()
                temp_path = encrypted_path

            # Move to final location
            shutil.move(str(temp_path), str(backup_file))

            # Update metadata
            metadata.size_bytes = backup_file.stat().st_size
            metadata.checksum = self._calculate_checksum(str(backup_file))

            return True

        except Exception as e:
            metadata.error_message = str(e)
            self.logger.error(f"Directory backup failed: {e}")
            return False

    async def _create_zip_archive(self, source: Path, archive_path: Path) -> bool:
        """Create ZIP archive."""
        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                if source.is_file():
                    zipf.write(source, source.name)
                else:
                    for file_path in source.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(source.parent)
                            zipf.write(file_path, arcname)
            return True

        except Exception as e:
            self.logger.error(f"Failed to create ZIP archive: {e}")
            return False

    async def _create_tar_archive(self, source: Path, archive_path: Path, compression: str) -> bool:
        """Create TAR archive with compression."""
        try:
            import tarfile

            mode = f"w:{compression}" if compression else "w"

            with tarfile.open(archive_path, mode) as tar:
                if source.is_file():
                    tar.add(source, arcname=source.name)
                else:
                    tar.add(source, arcname=source.name)

            return True

        except Exception as e:
            self.logger.error(f"Failed to create TAR archive: {e}")
            return False

    async def _encrypt_file(self, file_path: Path) -> Path:
        """Encrypt file using Fernet encryption."""
        encrypted_path = file_path.with_suffix(file_path.suffix + ".enc")
        fernet = Fernet(self.encryption_key)

        with open(file_path, "rb") as f:
            encrypted_data = fernet.encrypt(f.read())

        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)

        return encrypted_path

    async def decrypt_file(self, encrypted_path: str, output_path: str) -> bool:
        """Decrypt encrypted backup file."""
        try:
            if not self.encryption_key:
                self.logger.error("No encryption key available for decryption")
                return False

            fernet = Fernet(self.encryption_key)

            with open(encrypted_path, "rb") as f:
                decrypted_data = fernet.decrypt(f.read())

            with open(output_path, "wb") as f:
                f.write(decrypted_data)

            return True

        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            return False

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()


class BackupManager:
    """Main backup and recovery manager."""

    def __init__(self, config_path: str = "config/backup/backup_config.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize providers
        self.providers = {}
        self._initialize_providers()

        # Initialize handlers
        self.db_handler = DatabaseBackupHandler(self.config.get("databases", {}))
        self.file_handler = FileBackupHandler(self.config.get("files", {}))

        # Backup tracking
        self.active_backups = {}
        self.backup_history = []

    def _load_config(self) -> dict[str, Any]:
        """Load backup configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _initialize_providers(self):
        """Initialize backup providers."""
        providers_config = self.config.get("providers", {})

        # Local provider
        if "local" in providers_config:
            self.providers["local"] = LocalBackupProvider(providers_config["local"])

        # S3 provider
        if "s3" in providers_config:
            try:
                self.providers["s3"] = S3BackupProvider(providers_config["s3"])
            except Exception as e:
                self.logger.warning(f"Failed to initialize S3 provider: {e}")

        # SFTP provider
        if "sftp" in providers_config:
            try:
                self.providers["sftp"] = SFTPBackupProvider(providers_config["sftp"])
            except Exception as e:
                self.logger.warning(f"Failed to initialize SFTP provider: {e}")

        if not self.providers:
            # Default to local provider
            self.providers["local"] = LocalBackupProvider({"backup_directory": "/tmp/pynomaly_backups"})

    async def create_backup(self, backup_name: str, source_type: str,
                          source_path: str, backup_type: BackupType = BackupType.FULL,
                          provider_name: str = "local", **kwargs) -> str:
        """Create a new backup."""
        backup_id = f"{backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create backup metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            source_path=source_path,
            destination_path=f"{backup_name}/{backup_id}",
            timestamp=datetime.now(),
            status=BackupStatus.PENDING,
            compression=CompressionType(kwargs.get("compression", "gzip")),
            encryption_enabled=kwargs.get("encryption", False),
            retention_days=kwargs.get("retention_days", 30),
            tags=kwargs.get("tags", {})
        )

        self.active_backups[backup_id] = metadata

        try:
            metadata.status = BackupStatus.RUNNING
            start_time = time.time()

            # Create temporary backup file
            temp_dir = Path("/tmp/pynomaly_backups")
            temp_dir.mkdir(exist_ok=True)

            if source_type == "database":
                backup_file = temp_dir / f"{backup_id}.sql"
                success = await self._backup_database(source_path, str(backup_file), metadata)
            elif source_type == "directory":
                backup_file = temp_dir / f"{backup_id}.tar.gz"
                success = await self.file_handler.backup_directory(source_path, str(backup_file), metadata)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            if success:
                # Upload to provider
                provider = self.providers.get(provider_name)
                if provider:
                    remote_path = f"{metadata.destination_path}/{backup_file.name}"
                    upload_success = await provider.upload_backup(str(backup_file), remote_path, metadata)

                    if upload_success:
                        metadata.status = BackupStatus.COMPLETED
                        self.logger.info(f"Backup completed successfully: {backup_id}")
                    else:
                        metadata.status = BackupStatus.FAILED
                        metadata.error_message = "Failed to upload backup"
                else:
                    metadata.status = BackupStatus.FAILED
                    metadata.error_message = f"Provider not found: {provider_name}"
            else:
                metadata.status = BackupStatus.FAILED

            # Clean up temporary file
            if backup_file.exists():
                backup_file.unlink()

            metadata.duration_seconds = time.time() - start_time

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            self.logger.error(f"Backup failed: {e}")

        finally:
            # Move to history
            self.backup_history.append(metadata)
            if backup_id in self.active_backups:
                del self.active_backups[backup_id]

        return backup_id

    async def _backup_database(self, database_name: str, backup_path: str, metadata: BackupMetadata) -> bool:
        """Backup database based on configuration."""
        db_config = self.config.get("databases", {})

        if database_name in db_config.get("postgresql", {}):
            return await self.db_handler.backup_postgresql(backup_path, metadata)
        elif database_name in db_config.get("mysql", {}):
            return await self.db_handler.backup_mysql(backup_path, metadata)
        elif database_name in db_config.get("mongodb", {}):
            return await self.db_handler.backup_mongodb(backup_path, metadata)
        else:
            metadata.error_message = f"Database configuration not found: {database_name}"
            return False

    async def restore_backup(self, backup_id: str, restore_path: str,
                           provider_name: str = "local") -> bool:
        """Restore backup from storage."""
        try:
            provider = self.providers.get(provider_name)
            if not provider:
                self.logger.error(f"Provider not found: {provider_name}")
                return False

            # Find backup metadata
            backups = await provider.list_backups()
            backup_metadata = None

            for backup in backups:
                if backup.get("backup_id") == backup_id:
                    backup_metadata = backup
                    break

            if not backup_metadata:
                self.logger.error(f"Backup not found: {backup_id}")
                return False

            # Download backup
            remote_path = backup_metadata["destination_path"]
            temp_file = f"/tmp/{backup_id}_restore"

            download_success = await provider.download_backup(remote_path, temp_file)
            if not download_success:
                self.logger.error(f"Failed to download backup: {backup_id}")
                return False

            # Decrypt if necessary
            if backup_metadata.get("encryption_enabled"):
                decrypted_file = f"{temp_file}.dec"
                decrypt_success = await self.file_handler.decrypt_file(temp_file, decrypted_file)
                if decrypt_success:
                    os.rename(decrypted_file, temp_file)
                else:
                    os.unlink(temp_file)
                    return False

            # Extract/restore
            success = await self._extract_backup(temp_file, restore_path, backup_metadata)

            # Clean up
            if Path(temp_file).exists():
                os.unlink(temp_file)

            return success

        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False

    async def _extract_backup(self, backup_file: str, restore_path: str, metadata: dict[str, Any]) -> bool:
        """Extract backup to restore location."""
        try:
            compression = metadata.get("compression", "gzip")

            if compression == "zip":
                with zipfile.ZipFile(backup_file, "r") as zipf:
                    zipf.extractall(restore_path)
            else:
                import tarfile
                with tarfile.open(backup_file, f"r:{compression}") as tar:
                    tar.extractall(restore_path)

            return True

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return False

    async def list_backups(self, provider_name: str = "local") -> list[dict[str, Any]]:
        """List available backups."""
        provider = self.providers.get(provider_name)
        if provider:
            return await provider.list_backups()
        return []

    async def delete_backup(self, backup_id: str, provider_name: str = "local") -> bool:
        """Delete backup from storage."""
        try:
            provider = self.providers.get(provider_name)
            if not provider:
                return False

            # Find backup
            backups = await provider.list_backups()
            for backup in backups:
                if backup.get("backup_id") == backup_id:
                    remote_path = backup["destination_path"]
                    return await provider.delete_backup(remote_path)

            return False

        except Exception as e:
            self.logger.error(f"Delete backup failed: {e}")
            return False

    async def cleanup_old_backups(self, provider_name: str = "local"):
        """Clean up expired backups."""
        try:
            provider = self.providers.get(provider_name)
            if not provider:
                return

            backups = await provider.list_backups()
            now = datetime.now()

            for backup in backups:
                backup_time = datetime.fromisoformat(backup["timestamp"])
                retention_days = backup.get("retention_days", 30)

                if (now - backup_time).days > retention_days:
                    backup_id = backup["backup_id"]
                    await self.delete_backup(backup_id, provider_name)
                    self.logger.info(f"Deleted expired backup: {backup_id}")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def get_backup_stats(self) -> dict[str, Any]:
        """Get backup statistics."""
        total_backups = len(self.backup_history)
        successful = len([b for b in self.backup_history if b.status == BackupStatus.COMPLETED])
        failed = len([b for b in self.backup_history if b.status == BackupStatus.FAILED])

        total_size = sum(b.size_bytes for b in self.backup_history if b.status == BackupStatus.COMPLETED)

        return {
            "total_backups": total_backups,
            "successful_backups": successful,
            "failed_backups": failed,
            "success_rate": (successful / total_backups * 100) if total_backups > 0 else 0,
            "total_size_bytes": total_size,
            "active_backups": len(self.active_backups),
            "providers": list(self.providers.keys())
        }


async def main():
    """Main function for testing."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)
    logger.info("Testing backup and recovery system")

    # Create backup manager
    backup_manager = BackupManager()

    # Create test directory
    test_dir = Path("/tmp/test_backup_source")
    test_dir.mkdir(exist_ok=True)

    # Create test files
    (test_dir / "file1.txt").write_text("Test content 1")
    (test_dir / "file2.txt").write_text("Test content 2")
    (test_dir / "subdir").mkdir(exist_ok=True)
    (test_dir / "subdir" / "file3.txt").write_text("Test content 3")

    # Create backup
    logger.info("Creating test backup...")
    backup_id = await backup_manager.create_backup(
        backup_name="test_backup",
        source_type="directory",
        source_path=str(test_dir),
        compression="gzip",
        tags={"test": "true", "environment": "dev"}
    )

    logger.info(f"Backup created: {backup_id}")

    # List backups
    backups = await backup_manager.list_backups()
    logger.info(f"Available backups: {len(backups)}")

    # Test restore
    restore_dir = Path("/tmp/test_restore")
    restore_dir.mkdir(exist_ok=True)

    logger.info("Testing restore...")
    restore_success = await backup_manager.restore_backup(backup_id, str(restore_dir))
    logger.info(f"Restore successful: {restore_success}")

    # Print statistics
    stats = backup_manager.get_backup_stats()
    logger.info(f"Backup statistics: {json.dumps(stats, indent=2)}")

    # Clean up
    shutil.rmtree(test_dir)
    shutil.rmtree(restore_dir)


if __name__ == "__main__":
    asyncio.run(main())
