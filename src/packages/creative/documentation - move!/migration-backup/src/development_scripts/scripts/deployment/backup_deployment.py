#!/usr/bin/env python3
"""
Backup and Disaster Recovery for Production Deployment
Comprehensive backup automation and disaster recovery procedures
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups"""

    DATABASE = "database"
    CONFIGURATION = "configuration"
    MODEL = "model"
    APPLICATION = "application"
    FULL = "full"


class BackupStatus(Enum):
    """Backup status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class BackupConfig:
    """Backup configuration"""

    backup_type: BackupType
    retention_days: int = 30
    compression: bool = True
    encryption: bool = True
    storage_location: str = "s3://pynomaly-backups"
    schedule: str | None = None
    notification_channels: list[str] = field(default_factory=list)


@dataclass
class BackupResult:
    """Backup operation result"""

    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    start_time: datetime
    end_time: datetime | None = None
    file_path: str | None = None
    file_size: int | None = None
    checksum: str | None = None
    error_message: str | None = None


class BackupManager:
    """Manages backup and disaster recovery operations"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.backup_dir = self.project_root / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Load configuration
        self.config = self._load_backup_config()

        # AWS configuration
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.s3_bucket = os.getenv("BACKUP_S3_BUCKET", "pynomaly-backups")

    def _load_backup_config(self) -> dict:
        """Load backup configuration"""
        config_file = self.project_root / "config" / "backup" / "backup_config.yml"

        default_config = {
            "databases": {
                "postgres": {
                    "enabled": True,
                    "retention_days": 30,
                    "schedule": "0 2 * * *",
                },
                "redis": {
                    "enabled": True,
                    "retention_days": 7,
                    "schedule": "0 3 * * *",
                },
                "mongodb": {
                    "enabled": True,
                    "retention_days": 30,
                    "schedule": "0 4 * * *",
                },
            },
            "storage": {
                "local_path": str(self.backup_dir),
                "s3_bucket": self.s3_bucket,
                "encryption_key": os.getenv("BACKUP_ENCRYPTION_KEY"),
            },
            "notifications": {
                "email": {"enabled": True, "recipients": ["ops@monorepo.com"]},
                "slack": {
                    "enabled": True,
                    "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                },
            },
        }

        if config_file.exists():
            try:
                with open(config_file) as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load backup config: {e}")

        return default_config

    def create_backup_id(self, backup_type: BackupType) -> str:
        """Create unique backup ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{backup_type.value}_{timestamp}"

    def backup_database(self, db_type: str) -> BackupResult:
        """Backup database"""
        backup_id = self.create_backup_id(BackupType.DATABASE)
        result = BackupResult(
            backup_id=backup_id,
            backup_type=BackupType.DATABASE,
            status=BackupStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            logger.info(f"Starting {db_type} database backup")

            if db_type == "postgres":
                result = self._backup_postgres(result)
            elif db_type == "redis":
                result = self._backup_redis(result)
            elif db_type == "mongodb":
                result = self._backup_mongodb(result)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            # Upload to S3
            if result.status == BackupStatus.COMPLETED:
                self._upload_to_s3(result)

            # Verify backup
            if result.status == BackupStatus.COMPLETED:
                self._verify_backup(result)

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            result.status = BackupStatus.FAILED
            result.error_message = str(e)
        finally:
            result.end_time = datetime.now()

        return result

    def _backup_postgres(self, result: BackupResult) -> BackupResult:
        """Backup PostgreSQL database"""
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        backup_file = self.backup_dir / f"{result.backup_id}.sql"

        # Use pg_dump to create backup
        cmd = [
            "pg_dump",
            database_url,
            "--no-password",
            "--verbose",
            "--format=custom",
            "--compress=9",
            f"--file={backup_file}",
        ]

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode == 0:
            result.file_path = str(backup_file)
            result.file_size = backup_file.stat().st_size
            result.checksum = self._calculate_checksum(backup_file)
            result.status = BackupStatus.COMPLETED
            logger.info(f"PostgreSQL backup completed: {backup_file}")
        else:
            raise Exception(f"pg_dump failed: {process.stderr}")

        return result

    def _backup_redis(self, result: BackupResult) -> BackupResult:
        """Backup Redis database"""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        backup_file = self.backup_dir / f"{result.backup_id}.rdb"

        # Use redis-cli to create backup
        cmd = ["redis-cli", "-u", redis_url, "--rdb", str(backup_file)]

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode == 0:
            result.file_path = str(backup_file)
            result.file_size = backup_file.stat().st_size
            result.checksum = self._calculate_checksum(backup_file)
            result.status = BackupStatus.COMPLETED
            logger.info(f"Redis backup completed: {backup_file}")
        else:
            raise Exception(f"redis-cli backup failed: {process.stderr}")

        return result

    def _backup_mongodb(self, result: BackupResult) -> BackupResult:
        """Backup MongoDB database"""
        mongodb_url = os.getenv("MONGODB_URL")
        if not mongodb_url:
            raise ValueError("MONGODB_URL environment variable not set")

        backup_dir = self.backup_dir / result.backup_id
        backup_dir.mkdir(exist_ok=True)

        # Use mongodump to create backup
        cmd = ["mongodump", "--uri", mongodb_url, "--gzip", "--out", str(backup_dir)]

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode == 0:
            # Create tar archive
            archive_file = self.backup_dir / f"{result.backup_id}.tar.gz"
            subprocess.run(
                [
                    "tar",
                    "-czf",
                    str(archive_file),
                    "-C",
                    str(self.backup_dir),
                    result.backup_id,
                ],
                check=True,
            )

            # Remove directory
            subprocess.run(["rm", "-rf", str(backup_dir)], check=True)

            result.file_path = str(archive_file)
            result.file_size = archive_file.stat().st_size
            result.checksum = self._calculate_checksum(archive_file)
            result.status = BackupStatus.COMPLETED
            logger.info(f"MongoDB backup completed: {archive_file}")
        else:
            raise Exception(f"mongodump failed: {process.stderr}")

        return result

    def backup_configuration(self) -> BackupResult:
        """Backup configuration files"""
        backup_id = self.create_backup_id(BackupType.CONFIGURATION)
        result = BackupResult(
            backup_id=backup_id,
            backup_type=BackupType.CONFIGURATION,
            status=BackupStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            logger.info("Starting configuration backup")

            config_backup_file = self.backup_dir / f"{backup_id}.tar.gz"

            # Create tar archive of configuration files
            cmd = [
                "tar",
                "-czf",
                str(config_backup_file),
                "-C",
                str(self.project_root),
                "config/",
                ".env.example",
                "docker-compose.production.yml",
                "k8s/",
                "deploy/",
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode == 0:
                result.file_path = str(config_backup_file)
                result.file_size = config_backup_file.stat().st_size
                result.checksum = self._calculate_checksum(config_backup_file)
                result.status = BackupStatus.COMPLETED

                # Upload to S3
                self._upload_to_s3(result)

                logger.info(f"Configuration backup completed: {config_backup_file}")
            else:
                raise Exception(f"Configuration backup failed: {process.stderr}")

        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            result.status = BackupStatus.FAILED
            result.error_message = str(e)
        finally:
            result.end_time = datetime.now()

        return result

    def backup_models(self) -> BackupResult:
        """Backup ML models"""
        backup_id = self.create_backup_id(BackupType.MODEL)
        result = BackupResult(
            backup_id=backup_id,
            backup_type=BackupType.MODEL,
            status=BackupStatus.RUNNING,
            start_time=datetime.now(),
        )

        try:
            logger.info("Starting model backup")

            models_dir = self.project_root / "storage" / "models"
            if not models_dir.exists():
                logger.warning("Models directory not found, creating empty backup")
                models_dir.mkdir(parents=True, exist_ok=True)

            model_backup_file = self.backup_dir / f"{backup_id}.tar.gz"

            # Create tar archive of models
            cmd = [
                "tar",
                "-czf",
                str(model_backup_file),
                "-C",
                str(models_dir.parent),
                "models/",
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode == 0:
                result.file_path = str(model_backup_file)
                result.file_size = model_backup_file.stat().st_size
                result.checksum = self._calculate_checksum(model_backup_file)
                result.status = BackupStatus.COMPLETED

                # Upload to S3
                self._upload_to_s3(result)

                logger.info(f"Model backup completed: {model_backup_file}")
            else:
                raise Exception(f"Model backup failed: {process.stderr}")

        except Exception as e:
            logger.error(f"Model backup failed: {e}")
            result.status = BackupStatus.FAILED
            result.error_message = str(e)
        finally:
            result.end_time = datetime.now()

        return result

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        import hashlib

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _upload_to_s3(self, result: BackupResult) -> bool:
        """Upload backup to S3"""
        try:
            if not result.file_path:
                return False

            s3_key = f"backups/{result.backup_type.value}/{result.backup_id}/{Path(result.file_path).name}"

            cmd = [
                "aws",
                "s3",
                "cp",
                result.file_path,
                f"s3://{self.s3_bucket}/{s3_key}",
                "--region",
                self.aws_region,
            ]

            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode == 0:
                logger.info(f"Backup uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
                return True
            else:
                logger.error(f"S3 upload failed: {process.stderr}")
                return False

        except Exception as e:
            logger.error(f"S3 upload error: {e}")
            return False

    def _verify_backup(self, result: BackupResult) -> bool:
        """Verify backup integrity"""
        try:
            if not result.file_path or not Path(result.file_path).exists():
                return False

            # Recalculate checksum
            current_checksum = self._calculate_checksum(Path(result.file_path))

            if current_checksum == result.checksum:
                result.status = BackupStatus.VERIFIED
                logger.info(f"Backup verified: {result.backup_id}")
                return True
            else:
                logger.error(f"Backup verification failed: {result.backup_id}")
                result.status = BackupStatus.FAILED
                result.error_message = "Checksum verification failed"
                return False

        except Exception as e:
            logger.error(f"Backup verification error: {e}")
            result.status = BackupStatus.FAILED
            result.error_message = f"Verification error: {e}"
            return False

    def cleanup_old_backups(self, retention_days: int = 30):
        """Clean up old backup files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            for backup_file in self.backup_dir.glob("*"):
                if backup_file.is_file():
                    file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        backup_file.unlink()
                        logger.info(f"Removed old backup: {backup_file}")

            # Also cleanup S3
            self._cleanup_s3_backups(retention_days)

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    def _cleanup_s3_backups(self, retention_days: int):
        """Clean up old S3 backups"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")

            # List and delete old S3 objects
            cmd = [
                "aws",
                "s3api",
                "list-objects-v2",
                "--bucket",
                self.s3_bucket,
                "--prefix",
                "backups/",
                "--query",
                f"Contents[?LastModified<='{cutoff_str}'].Key",
                "--output",
                "text",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                old_keys = result.stdout.strip().split()

                for key in old_keys:
                    delete_cmd = ["aws", "s3", "rm", f"s3://{self.s3_bucket}/{key}"]
                    subprocess.run(delete_cmd, capture_output=True)
                    logger.info(f"Removed old S3 backup: {key}")

        except Exception as e:
            logger.error(f"S3 cleanup failed: {e}")

    def create_full_backup(self) -> list[BackupResult]:
        """Create full system backup"""
        logger.info("Starting full system backup")

        results = []

        # Backup databases
        for db_type in ["postgres", "redis", "mongodb"]:
            if self.config["databases"][db_type]["enabled"]:
                result = self.backup_database(db_type)
                results.append(result)

        # Backup configuration
        config_result = self.backup_configuration()
        results.append(config_result)

        # Backup models
        model_result = self.backup_models()
        results.append(model_result)

        # Generate backup report
        self._generate_backup_report(results)

        # Send notifications
        self._send_backup_notifications(results)

        return results

    def _generate_backup_report(self, results: list[BackupResult]):
        """Generate backup report"""
        try:
            report = {
                "backup_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "total_backups": len(results),
                    "successful_backups": len(
                        [
                            r
                            for r in results
                            if r.status
                            in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
                        ]
                    ),
                    "failed_backups": len(
                        [r for r in results if r.status == BackupStatus.FAILED]
                    ),
                },
                "backup_details": [
                    {
                        "backup_id": result.backup_id,
                        "type": result.backup_type.value,
                        "status": result.status.value,
                        "file_size_mb": round(result.file_size / 1024 / 1024, 2)
                        if result.file_size
                        else 0,
                        "duration_seconds": (
                            result.end_time - result.start_time
                        ).total_seconds()
                        if result.end_time
                        else 0,
                        "error": result.error_message,
                    }
                    for result in results
                ],
            }

            report_file = (
                self.backup_dir
                / f"backup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Backup report generated: {report_file}")

        except Exception as e:
            logger.error(f"Failed to generate backup report: {e}")

    def _send_backup_notifications(self, results: list[BackupResult]):
        """Send backup completion notifications"""
        try:
            success_count = len(
                [
                    r
                    for r in results
                    if r.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
                ]
            )
            failed_count = len([r for r in results if r.status == BackupStatus.FAILED])

            if failed_count > 0:
                status = "⚠️ BACKUP COMPLETED WITH ERRORS"
                color = "warning"
            else:
                status = "✅ BACKUP COMPLETED SUCCESSFULLY"
                color = "good"

            message = f"""
{status}
• Successful backups: {success_count}
• Failed backups: {failed_count}
• Total backups: {len(results)}
• Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """

            # Send to configured notification channels
            for channel in self.config.get("notifications", {}).keys():
                if (
                    channel == "slack"
                    and self.config["notifications"]["slack"]["enabled"]
                ):
                    self._send_slack_notification(message, color)

        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")

    def _send_slack_notification(self, message: str, color: str):
        """Send Slack notification"""
        try:
            webhook_url = self.config["notifications"]["slack"].get("webhook_url")
            if not webhook_url:
                return

            import requests

            payload = {
                "attachments": [
                    {"color": color, "text": message, "mrkdwn_in": ["text"]}
                ]
            }

            response = requests.post(webhook_url, json=payload)
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.error(
                    f"Failed to send Slack notification: {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Slack notification error: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Backup and disaster recovery")
    parser.add_argument(
        "--type",
        choices=["database", "config", "models", "full"],
        default="full",
        help="Type of backup to perform",
    )
    parser.add_argument(
        "--db-type",
        choices=["postgres", "redis", "mongodb"],
        help="Database type for database backup",
    )
    parser.add_argument("--cleanup", action="store_true", help="Clean up old backups")
    parser.add_argument(
        "--retention-days", type=int, default=30, help="Retention period in days"
    )

    args = parser.parse_args()

    backup_manager = BackupManager()

    try:
        if args.cleanup:
            backup_manager.cleanup_old_backups(args.retention_days)
            logger.info("Backup cleanup completed")
            return

        if args.type == "full":
            results = backup_manager.create_full_backup()
        elif args.type == "database":
            if not args.db_type:
                logger.error("--db-type required for database backup")
                sys.exit(1)
            results = [backup_manager.backup_database(args.db_type)]
        elif args.type == "config":
            results = [backup_manager.backup_configuration()]
        elif args.type == "models":
            results = [backup_manager.backup_models()]

        # Check results
        failed_backups = [r for r in results if r.status == BackupStatus.FAILED]
        if failed_backups:
            logger.error(f"Some backups failed: {len(failed_backups)}")
            sys.exit(1)
        else:
            logger.info("All backups completed successfully")

    except Exception as e:
        logger.error(f"Backup operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
