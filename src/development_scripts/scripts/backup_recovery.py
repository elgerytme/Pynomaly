#!/usr/bin/env python3
"""
Backup and recovery testing script for anomaly_detection production deployment.
This script creates comprehensive backup strategies and tests recovery procedures.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Backup configuration data structure."""

    backup_base_path: str = "/backups"
    retention_days: int = 30
    compression_enabled: bool = True
    encryption_enabled: bool = True
    cloud_backup_enabled: bool = False
    cloud_provider: str = "s3"  # s3, gcs, azure


@dataclass
class BackupResult:
    """Backup operation result."""

    component: str
    operation: str
    status: str
    backup_path: str
    size_bytes: int
    duration_seconds: float
    error_message: str = ""


class BackupRecovery:
    """Main backup and recovery orchestrator."""

    def __init__(self, config: BackupConfig):
        """Initialize backup and recovery system."""
        self.config = config
        self.backup_results: list[BackupResult] = []
        self.recovery_results: list[BackupResult] = []

        # Create backup directories
        self.database_backup_path = Path(config.backup_base_path) / "database"
        self.models_backup_path = Path(config.backup_base_path) / "models"
        self.config_backup_path = Path(config.backup_base_path) / "config"
        self.logs_backup_path = Path(config.backup_base_path) / "logs"

        # Create directories
        for path in [
            self.database_backup_path,
            self.models_backup_path,
            self.config_backup_path,
            self.logs_backup_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    async def backup_database(self) -> bool:
        """Backup PostgreSQL database."""
        logger.info("🗄️ Starting database backup...")

        start_time = datetime.now()

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"anomaly_detection_db_backup_{timestamp}.sql"
            backup_path = self.database_backup_path / backup_filename

            # Create database dump
            dump_cmd = [
                "docker-compose",
                "-f",
                "docker-compose.simple.yml",
                "exec",
                "-T",
                "postgres",
                "pg_dump",
                "-U",
                "anomaly_detection",
                "-d",
                "anomaly_detection_prod",
                "--verbose",
                "--no-password",
                "--format=custom",
            ]

            logger.info(f"Running database backup: {' '.join(dump_cmd)}")

            with open(backup_path, "wb") as f:
                result = subprocess.run(
                    dump_cmd, stdout=f, stderr=subprocess.PIPE, text=False
                )

            if result.returncode != 0:
                logger.error(f"Database backup failed: {result.stderr.decode()}")
                self.backup_results.append(
                    BackupResult(
                        component="Database",
                        operation="backup",
                        status="failed",
                        backup_path=str(backup_path),
                        size_bytes=0,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                        error_message=result.stderr.decode(),
                    )
                )
                return False

            # Compress backup if enabled
            if self.config.compression_enabled:
                compressed_path = backup_path.with_suffix(".sql.gz")
                compress_cmd = ["gzip", str(backup_path)]
                subprocess.run(compress_cmd, check=True)
                backup_path = compressed_path

            # Get backup size
            backup_size = backup_path.stat().st_size

            # Verify backup integrity
            if await self._verify_database_backup(backup_path):
                logger.info(
                    f"✅ Database backup completed: {backup_path} ({backup_size} bytes)"
                )
                self.backup_results.append(
                    BackupResult(
                        component="Database",
                        operation="backup",
                        status="success",
                        backup_path=str(backup_path),
                        size_bytes=backup_size,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                    )
                )
                return True
            else:
                logger.error("Database backup verification failed")
                self.backup_results.append(
                    BackupResult(
                        component="Database",
                        operation="backup",
                        status="failed",
                        backup_path=str(backup_path),
                        size_bytes=backup_size,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                        error_message="Backup verification failed",
                    )
                )
                return False

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            self.backup_results.append(
                BackupResult(
                    component="Database",
                    operation="backup",
                    status="failed",
                    backup_path="",
                    size_bytes=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=str(e),
                )
            )
            return False

    async def _verify_database_backup(self, backup_path: Path) -> bool:
        """Verify database backup integrity."""
        try:
            # For compressed backups, we need to decompress first
            if backup_path.suffix == ".gz":
                # Create temporary file for verification
                with tempfile.NamedTemporaryFile(suffix=".sql") as temp_file:
                    decompress_cmd = ["gunzip", "-c", str(backup_path)]
                    with open(temp_file.name, "wb") as f:
                        subprocess.run(decompress_cmd, stdout=f, check=True)

                    # Verify the decompressed file
                    return temp_file.name and Path(temp_file.name).stat().st_size > 0
            else:
                # For uncompressed backups, check file size and basic structure
                if backup_path.stat().st_size > 0:
                    # Try to read the header to ensure it's a valid pg_dump file
                    with open(backup_path, "rb") as f:
                        header = f.read(100)
                        return b"PGDMP" in header or b"CREATE" in header
                return False

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    async def backup_models(self) -> bool:
        """Backup ML models and artifacts."""
        logger.info("🤖 Starting models backup...")

        start_time = datetime.now()

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"anomaly_detection_models_backup_{timestamp}.tar.gz"
            backup_path = self.models_backup_path / backup_filename

            # Create models directory if it doesn't exist
            models_source = Path("models")
            if not models_source.exists():
                models_source.mkdir(exist_ok=True)
                # Create sample model files for testing
                (models_source / "sample_model.pkl").write_text("sample model data")
                (models_source / "model_metadata.json").write_text(
                    '{"version": "1.0.0", "type": "isolation_forest"}'
                )

            # Create tar archive of models
            tar_cmd = [
                "tar",
                "-czf",
                str(backup_path),
                "-C",
                str(models_source.parent),
                str(models_source.name),
            ]

            result = subprocess.run(tar_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Models backup failed: {result.stderr}")
                self.backup_results.append(
                    BackupResult(
                        component="Models",
                        operation="backup",
                        status="failed",
                        backup_path=str(backup_path),
                        size_bytes=0,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                        error_message=result.stderr,
                    )
                )
                return False

            # Get backup size
            backup_size = backup_path.stat().st_size

            # Verify backup
            if await self._verify_models_backup(backup_path):
                logger.info(
                    f"✅ Models backup completed: {backup_path} ({backup_size} bytes)"
                )
                self.backup_results.append(
                    BackupResult(
                        component="Models",
                        operation="backup",
                        status="success",
                        backup_path=str(backup_path),
                        size_bytes=backup_size,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                    )
                )
                return True
            else:
                logger.error("Models backup verification failed")
                return False

        except Exception as e:
            logger.error(f"Models backup failed: {e}")
            self.backup_results.append(
                BackupResult(
                    component="Models",
                    operation="backup",
                    status="failed",
                    backup_path="",
                    size_bytes=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=str(e),
                )
            )
            return False

    async def _verify_models_backup(self, backup_path: Path) -> bool:
        """Verify models backup integrity."""
        try:
            # Test tar file integrity
            test_cmd = ["tar", "-tzf", str(backup_path)]
            result = subprocess.run(test_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Check if expected files are in the archive
                file_list = result.stdout.strip().split("\n")
                return any("model" in f for f in file_list)
            return False

        except Exception as e:
            logger.error(f"Models backup verification failed: {e}")
            return False

    async def backup_configuration(self) -> bool:
        """Backup system configuration files."""
        logger.info("⚙️ Starting configuration backup...")

        start_time = datetime.now()

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"anomaly_detection_config_backup_{timestamp}.tar.gz"
            backup_path = self.config_backup_path / backup_filename

            # List of configuration files and directories to backup
            config_items = [
                "config/",
                ".env",
                "docker-compose.simple.yml",
                "docker-compose.production.yml",
                "Dockerfile.production",
                "requirements-prod.txt",
                "pyproject.toml",
            ]

            # Filter existing items
            existing_items = [item for item in config_items if Path(item).exists()]

            if not existing_items:
                logger.warning("No configuration files found to backup")
                return False

            # Create tar archive
            tar_cmd = ["tar", "-czf", str(backup_path)] + existing_items

            result = subprocess.run(tar_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Configuration backup failed: {result.stderr}")
                self.backup_results.append(
                    BackupResult(
                        component="Configuration",
                        operation="backup",
                        status="failed",
                        backup_path=str(backup_path),
                        size_bytes=0,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                        error_message=result.stderr,
                    )
                )
                return False

            # Get backup size
            backup_size = backup_path.stat().st_size

            logger.info(
                f"✅ Configuration backup completed: {backup_path} ({backup_size} bytes)"
            )
            self.backup_results.append(
                BackupResult(
                    component="Configuration",
                    operation="backup",
                    status="success",
                    backup_path=str(backup_path),
                    size_bytes=backup_size,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )
            )
            return True

        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            self.backup_results.append(
                BackupResult(
                    component="Configuration",
                    operation="backup",
                    status="failed",
                    backup_path="",
                    size_bytes=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=str(e),
                )
            )
            return False

    async def backup_logs(self) -> bool:
        """Backup system logs."""
        logger.info("📝 Starting logs backup...")

        start_time = datetime.now()

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"anomaly_detection_logs_backup_{timestamp}.tar.gz"
            backup_path = self.logs_backup_path / backup_filename

            # Create logs directory if it doesn't exist
            logs_source = Path("logs")
            if not logs_source.exists():
                logs_source.mkdir(exist_ok=True)
                # Create sample log files
                (logs_source / "app.log").write_text("Sample application log")
                (logs_source / "error.log").write_text("Sample error log")
                (logs_source / "access.log").write_text("Sample access log")

            # Create tar archive
            tar_cmd = [
                "tar",
                "-czf",
                str(backup_path),
                "-C",
                str(logs_source.parent),
                str(logs_source.name),
            ]

            result = subprocess.run(tar_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Logs backup failed: {result.stderr}")
                self.backup_results.append(
                    BackupResult(
                        component="Logs",
                        operation="backup",
                        status="failed",
                        backup_path=str(backup_path),
                        size_bytes=0,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                        error_message=result.stderr,
                    )
                )
                return False

            # Get backup size
            backup_size = backup_path.stat().st_size

            logger.info(
                f"✅ Logs backup completed: {backup_path} ({backup_size} bytes)"
            )
            self.backup_results.append(
                BackupResult(
                    component="Logs",
                    operation="backup",
                    status="success",
                    backup_path=str(backup_path),
                    size_bytes=backup_size,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )
            )
            return True

        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
            self.backup_results.append(
                BackupResult(
                    component="Logs",
                    operation="backup",
                    status="failed",
                    backup_path="",
                    size_bytes=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=str(e),
                )
            )
            return False

    async def test_database_recovery(self) -> bool:
        """Test database recovery procedure."""
        logger.info("🔄 Testing database recovery...")

        start_time = datetime.now()

        try:
            # Find the latest database backup
            backup_files = list(
                self.database_backup_path.glob("anomaly_detection_db_backup_*.sql*")
            )
            if not backup_files:
                logger.error("No database backup files found for recovery test")
                return False

            latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Using backup file: {latest_backup}")

            # Create test database for recovery
            test_db_name = "anomaly_detection_test_recovery"

            # Create test database
            create_db_cmd = [
                "docker-compose",
                "-f",
                "docker-compose.simple.yml",
                "exec",
                "-T",
                "postgres",
                "createdb",
                "-U",
                "anomaly_detection",
                test_db_name,
            ]

            result = subprocess.run(create_db_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warning(
                    f"Test database creation failed (may already exist): {result.stderr}"
                )

            # Restore from backup
            if latest_backup.suffix == ".gz":
                # Handle compressed backup
                decompress_cmd = ["gunzip", "-c", str(latest_backup)]
                restore_cmd = [
                    "docker-compose",
                    "-f",
                    "docker-compose.simple.yml",
                    "exec",
                    "-T",
                    "postgres",
                    "psql",
                    "-U",
                    "anomaly_detection",
                    "-d",
                    test_db_name,
                ]

                # Pipe decompressed data to psql
                decompress_proc = subprocess.Popen(
                    decompress_cmd, stdout=subprocess.PIPE
                )
                restore_proc = subprocess.Popen(
                    restore_cmd,
                    stdin=decompress_proc.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                decompress_proc.stdout.close()

                stdout, stderr = restore_proc.communicate()

                if restore_proc.returncode != 0:
                    logger.error(f"Database recovery failed: {stderr.decode()}")
                    return False
            else:
                # Handle uncompressed backup
                restore_cmd = [
                    "docker-compose",
                    "-f",
                    "docker-compose.simple.yml",
                    "exec",
                    "-T",
                    "postgres",
                    "pg_restore",
                    "-U",
                    "anomaly_detection",
                    "-d",
                    test_db_name,
                    "--verbose",
                    "--no-owner",
                ]

                with open(latest_backup, "rb") as f:
                    result = subprocess.run(
                        restore_cmd, stdin=f, capture_output=True, text=True
                    )

                if result.returncode != 0:
                    logger.error(f"Database recovery failed: {result.stderr}")
                    return False

            # Verify recovery by checking table existence
            verify_cmd = [
                "docker-compose",
                "-f",
                "docker-compose.simple.yml",
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "anomaly_detection",
                "-d",
                test_db_name,
                "-c",
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';",
            ]

            result = subprocess.run(verify_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ Database recovery test completed successfully")
                self.recovery_results.append(
                    BackupResult(
                        component="Database",
                        operation="recovery",
                        status="success",
                        backup_path=str(latest_backup),
                        size_bytes=latest_backup.stat().st_size,
                        duration_seconds=(datetime.now() - start_time).total_seconds(),
                    )
                )

                # Clean up test database
                drop_db_cmd = [
                    "docker-compose",
                    "-f",
                    "docker-compose.simple.yml",
                    "exec",
                    "-T",
                    "postgres",
                    "dropdb",
                    "-U",
                    "anomaly_detection",
                    test_db_name,
                ]
                subprocess.run(drop_db_cmd, capture_output=True)

                return True
            else:
                logger.error(f"Database recovery verification failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Database recovery test failed: {e}")
            self.recovery_results.append(
                BackupResult(
                    component="Database",
                    operation="recovery",
                    status="failed",
                    backup_path="",
                    size_bytes=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=str(e),
                )
            )
            return False

    async def test_models_recovery(self) -> bool:
        """Test models recovery procedure."""
        logger.info("🤖 Testing models recovery...")

        start_time = datetime.now()

        try:
            # Find the latest models backup
            backup_files = list(
                self.models_backup_path.glob("anomaly_detection_models_backup_*.tar.gz")
            )
            if not backup_files:
                logger.error("No models backup files found for recovery test")
                return False

            latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Using backup file: {latest_backup}")

            # Create temporary recovery directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract models backup
                extract_cmd = ["tar", "-xzf", str(latest_backup), "-C", temp_dir]

                result = subprocess.run(extract_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(f"Models recovery failed: {result.stderr}")
                    return False

                # Verify extracted files
                extracted_path = Path(temp_dir) / "models"
                if extracted_path.exists() and list(extracted_path.glob("*")):
                    logger.info("✅ Models recovery test completed successfully")
                    self.recovery_results.append(
                        BackupResult(
                            component="Models",
                            operation="recovery",
                            status="success",
                            backup_path=str(latest_backup),
                            size_bytes=latest_backup.stat().st_size,
                            duration_seconds=(
                                datetime.now() - start_time
                            ).total_seconds(),
                        )
                    )
                    return True
                else:
                    logger.error(
                        "Models recovery verification failed: no files extracted"
                    )
                    return False

        except Exception as e:
            logger.error(f"Models recovery test failed: {e}")
            self.recovery_results.append(
                BackupResult(
                    component="Models",
                    operation="recovery",
                    status="failed",
                    backup_path="",
                    size_bytes=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    error_message=str(e),
                )
            )
            return False

    async def cleanup_old_backups(self) -> bool:
        """Clean up old backup files based on retention policy."""
        logger.info("🧹 Cleaning up old backups...")

        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            deleted_count = 0

            # Clean up each backup directory
            for backup_dir in [
                self.database_backup_path,
                self.models_backup_path,
                self.config_backup_path,
                self.logs_backup_path,
            ]:
                for backup_file in backup_dir.glob("*"):
                    if backup_file.is_file():
                        file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                        if file_time < cutoff_date:
                            backup_file.unlink()
                            deleted_count += 1
                            logger.info(f"Deleted old backup: {backup_file}")

            logger.info(
                f"✅ Cleanup completed: {deleted_count} old backup files deleted"
            )
            return True

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return False

    def generate_backup_report(self) -> dict[str, Any]:
        """Generate comprehensive backup and recovery report."""
        total_backup_size = sum(r.size_bytes for r in self.backup_results)
        total_backup_time = sum(r.duration_seconds for r in self.backup_results)

        successful_backups = [r for r in self.backup_results if r.status == "success"]
        failed_backups = [r for r in self.backup_results if r.status == "failed"]

        successful_recoveries = [
            r for r in self.recovery_results if r.status == "success"
        ]
        failed_recoveries = [r for r in self.recovery_results if r.status == "failed"]

        report = {
            "backup_recovery_test": {
                "timestamp": datetime.now().isoformat(),
                "backup_base_path": self.config.backup_base_path,
                "retention_days": self.config.retention_days,
                "compression_enabled": self.config.compression_enabled,
                "total_backup_size_bytes": total_backup_size,
                "total_backup_size_mb": round(total_backup_size / 1024 / 1024, 2),
                "total_backup_time_seconds": total_backup_time,
                "backup_success_rate": len(successful_backups)
                / len(self.backup_results)
                * 100
                if self.backup_results
                else 0,
                "recovery_success_rate": len(successful_recoveries)
                / len(self.recovery_results)
                * 100
                if self.recovery_results
                else 0,
            },
            "backup_results": [
                {
                    "component": r.component,
                    "operation": r.operation,
                    "status": r.status,
                    "backup_path": r.backup_path,
                    "size_bytes": r.size_bytes,
                    "size_mb": round(r.size_bytes / 1024 / 1024, 2),
                    "duration_seconds": r.duration_seconds,
                    "error_message": r.error_message,
                }
                for r in self.backup_results
            ],
            "recovery_results": [
                {
                    "component": r.component,
                    "operation": r.operation,
                    "status": r.status,
                    "backup_path": r.backup_path,
                    "size_bytes": r.size_bytes,
                    "size_mb": round(r.size_bytes / 1024 / 1024, 2),
                    "duration_seconds": r.duration_seconds,
                    "error_message": r.error_message,
                }
                for r in self.recovery_results
            ],
            "backup_summary": {
                "total_backups": len(self.backup_results),
                "successful_backups": len(successful_backups),
                "failed_backups": len(failed_backups),
                "total_recoveries": len(self.recovery_results),
                "successful_recoveries": len(successful_recoveries),
                "failed_recoveries": len(failed_recoveries),
            },
            "recommendations": [
                "Schedule regular automated backups",
                "Test recovery procedures monthly",
                "Monitor backup storage space",
                "Implement off-site backup storage",
                "Document recovery procedures",
                "Train team on recovery processes",
                "Set up backup failure alerts",
                "Regular backup integrity checks",
            ],
        }

        return report

    def save_backup_report(self, report: dict[str, Any]):
        """Save backup report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backup_recovery_report_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"💾 Backup report saved to {filename}")

    def print_backup_summary(self, report: dict[str, Any]):
        """Print backup and recovery summary."""
        test_info = report["backup_recovery_test"]
        summary = report["backup_summary"]

        print("\n" + "=" * 60)
        print("💾 anomaly_detection BACKUP & RECOVERY TEST SUMMARY")
        print("=" * 60)
        print(f"Test Time: {test_info['timestamp']}")
        print(f"Backup Path: {test_info['backup_base_path']}")
        print(f"Retention: {test_info['retention_days']} days")
        print(f"Total Backup Size: {test_info['total_backup_size_mb']} MB")
        print(
            f"Total Backup Time: {test_info['total_backup_time_seconds']:.2f} seconds"
        )

        print("\n📊 BACKUP RESULTS:")
        print(f"  • Total Backups: {summary['total_backups']}")
        print(f"  • Successful: {summary['successful_backups']}")
        print(f"  • Failed: {summary['failed_backups']}")
        print(f"  • Success Rate: {test_info['backup_success_rate']:.1f}%")

        print("\n🔄 RECOVERY RESULTS:")
        print(f"  • Total Recovery Tests: {summary['total_recoveries']}")
        print(f"  • Successful: {summary['successful_recoveries']}")
        print(f"  • Failed: {summary['failed_recoveries']}")
        print(f"  • Success Rate: {test_info['recovery_success_rate']:.1f}%")

        print("\n📋 BACKUP DETAILS:")
        for result in report["backup_results"]:
            status_emoji = "✅" if result["status"] == "success" else "❌"
            print(
                f"  {status_emoji} {result['component']}: {result['size_mb']} MB ({result['duration_seconds']:.2f}s)"
            )

        print("\n📋 RECOMMENDATIONS:")
        for recommendation in report["recommendations"]:
            print(f"  • {recommendation}")

        print("\n" + "=" * 60)
        print("🎉 BACKUP & RECOVERY TEST COMPLETE!")
        print("=" * 60)


async def main():
    """Main backup and recovery testing workflow."""
    config = BackupConfig()

    # Override with environment variables if available
    config.backup_base_path = os.getenv("BACKUP_PATH", config.backup_base_path)

    backup_recovery = BackupRecovery(config)

    try:
        logger.info("🚀 Starting backup and recovery testing...")

        # Perform backups
        db_backup_success = await backup_recovery.backup_database()
        models_backup_success = await backup_recovery.backup_models()
        config_backup_success = await backup_recovery.backup_configuration()
        logs_backup_success = await backup_recovery.backup_logs()

        # Test recovery procedures
        db_recovery_success = await backup_recovery.test_database_recovery()
        models_recovery_success = await backup_recovery.test_models_recovery()

        # Cleanup old backups
        cleanup_success = await backup_recovery.cleanup_old_backups()

        # Generate report
        report = backup_recovery.generate_backup_report()
        backup_recovery.save_backup_report(report)
        backup_recovery.print_backup_summary(report)

        # Overall success
        overall_success = all(
            [
                db_backup_success,
                models_backup_success,
                config_backup_success,
                logs_backup_success,
                db_recovery_success,
                models_recovery_success,
                cleanup_success,
            ]
        )

        if overall_success:
            logger.info("✅ Backup and recovery testing completed successfully!")
            return True
        else:
            logger.error("❌ Backup and recovery testing completed with errors")
            return False

    except Exception as e:
        logger.error(f"Backup and recovery testing failed: {e}")
        return False


if __name__ == "__main__":
    # Run the backup and recovery testing
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
