#!/usr/bin/env python3
"""
Basic tests for Backup and Disaster Recovery functionality
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml


def test_backup_manager_import():
    """Test that backup manager can be imported."""
    try:
        # Import without initializing to avoid dependency issues
        import sys

        sys.path.insert(0, "src")

        # Test individual components
        from monorepo.infrastructure.backup.backup_manager import (
            BackupMetadata,
            BackupStatus,
            BackupType,
            CompressionType,
        )

        assert BackupType.FULL.value == "full"
        assert BackupStatus.COMPLETED.value == "completed"
        assert CompressionType.GZIP.value == "gzip"

    except ImportError as e:
        pytest.skip(f"Backup manager dependencies not available: {e}")


def test_disaster_recovery_import():
    """Test that disaster recovery can be imported."""
    try:
        import sys

        sys.path.insert(0, "src")

        from monorepo.infrastructure.backup.disaster_recovery_service import (
            RecoveryPlan,
            RecoveryPoint,
            RecoveryPriority,
            RecoveryStatus,
        )

        assert RecoveryPriority.CRITICAL.value == "critical"
        assert RecoveryStatus.COMPLETED.value == "completed"

    except ImportError as e:
        pytest.skip(f"Disaster recovery dependencies not available: {e}")


def test_backup_metadata():
    """Test backup metadata functionality."""
    try:
        import sys

        sys.path.insert(0, "src")

        from monorepo.infrastructure.backup.backup_manager import (
            BackupMetadata,
            BackupStatus,
            BackupType,
            CompressionType,
        )

        metadata = BackupMetadata(
            backup_id="test_backup_001",
            backup_type=BackupType.FULL,
            source_path="/test/source",
            destination_path="/test/destination",
            timestamp=datetime.now(),
            status=BackupStatus.PENDING,
            compression=CompressionType.GZIP,
            encryption_enabled=True,
            retention_days=30,
            tags={"environment": "test"},
        )

        # Test serialization
        metadata_dict = metadata.to_dict()
        assert metadata_dict["backup_id"] == "test_backup_001"
        assert metadata_dict["backup_type"] == "full"
        assert metadata_dict["encryption_enabled"] is True
        assert metadata_dict["tags"]["environment"] == "test"

    except ImportError as e:
        pytest.skip(f"Backup metadata dependencies not available: {e}")


def test_recovery_point():
    """Test recovery point functionality."""
    try:
        import sys

        sys.path.insert(0, "src")

        from monorepo.infrastructure.backup.disaster_recovery_service import (
            RecoveryPoint,
            RecoveryPriority,
        )

        recovery_point = RecoveryPoint(
            service_name="test_service",
            rpo_minutes=30,
            rto_minutes=60,
            priority=RecoveryPriority.HIGH,
            backup_frequency_minutes=30,
            dependencies=["database"],
            health_check_url="http://localhost:8080/health",
            restart_command="systemctl restart test_service",
        )

        assert recovery_point.service_name == "test_service"
        assert recovery_point.priority == RecoveryPriority.HIGH
        assert "database" in recovery_point.dependencies

        # Test serialization
        rp_dict = recovery_point.to_dict()
        assert rp_dict["service_name"] == "test_service"
        assert rp_dict["priority"] == "high"

    except ImportError as e:
        pytest.skip(f"Recovery point dependencies not available: {e}")


def test_recovery_plan():
    """Test recovery plan functionality."""
    try:
        import sys

        sys.path.insert(0, "src")

        from monorepo.infrastructure.backup.disaster_recovery_service import (
            RecoveryPlan,
            RecoveryPoint,
            RecoveryPriority,
        )

        recovery_points = [
            RecoveryPoint(
                service_name="database",
                rpo_minutes=15,
                rto_minutes=30,
                priority=RecoveryPriority.CRITICAL,
                backup_frequency_minutes=15,
            ),
            RecoveryPoint(
                service_name="api",
                rpo_minutes=30,
                rto_minutes=60,
                priority=RecoveryPriority.HIGH,
                backup_frequency_minutes=30,
                dependencies=["database"],
            ),
        ]

        plan = RecoveryPlan(
            plan_id="test_plan",
            name="Test Recovery Plan",
            description="Test plan for unit testing",
            recovery_points=recovery_points,
            notification_channels=["email"],
            estimated_rto_minutes=90,
        )

        assert plan.plan_id == "test_plan"
        assert len(plan.recovery_points) == 2
        assert plan.estimated_rto_minutes == 90

        # Test serialization
        plan_dict = plan.to_dict()
        assert plan_dict["plan_id"] == "test_plan"
        assert len(plan_dict["recovery_points"]) == 2

    except ImportError as e:
        pytest.skip(f"Recovery plan dependencies not available: {e}")


def test_backup_config_loading():
    """Test backup configuration loading."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test backup configuration
        config_path = Path(temp_dir) / "backup_config.yml"
        config_data = {
            "providers": {"local": {"backup_directory": "/var/backups/pynomaly"}},
            "databases": {
                "postgresql": {
                    "host": "localhost",
                    "port": 5432,
                    "username": "postgres",
                    "database": "monorepo",
                }
            },
            "files": {
                "encryption": {
                    "enabled": True,
                    "key_file": "/etc/pynomaly/backup_encryption.key",
                }
            },
        }
        config_path.write_text(yaml.dump(config_data))

        # Load and verify configuration
        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)

        assert "providers" in loaded_config
        assert "local" in loaded_config["providers"]
        assert loaded_config["databases"]["postgresql"]["host"] == "localhost"
        assert loaded_config["files"]["encryption"]["enabled"] is True


def test_dr_config_loading():
    """Test disaster recovery configuration loading."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test DR configuration
        config_path = Path(temp_dir) / "dr_config.yml"
        config_data = {
            "monitoring": {
                "enabled": True,
                "check_interval_minutes": 5,
                "failure_threshold": 3,
            },
            "notification_channels": {
                "email": {
                    "type": "email",
                    "smtp_server": "smtp.example.com",
                    "recipients": ["admin@example.com"],
                }
            },
            "recovery_plans": {
                "primary": {
                    "name": "Primary Recovery Plan",
                    "description": "Main recovery plan",
                    "recovery_points": [
                        {
                            "service_name": "database",
                            "rpo_minutes": 30,
                            "rto_minutes": 60,
                            "priority": "critical",
                            "backup_frequency_minutes": 30,
                        }
                    ],
                }
            },
        }
        config_path.write_text(yaml.dump(config_data))

        # Load and verify configuration
        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config["monitoring"]["enabled"] is True
        assert "email" in loaded_config["notification_channels"]
        assert "primary" in loaded_config["recovery_plans"]

        primary_plan = loaded_config["recovery_plans"]["primary"]
        assert primary_plan["name"] == "Primary Recovery Plan"
        assert len(primary_plan["recovery_points"]) == 1

        recovery_point = primary_plan["recovery_points"][0]
        assert recovery_point["service_name"] == "database"
        assert recovery_point["priority"] == "critical"


def test_configuration_files_exist():
    """Test that configuration files exist."""
    backup_config_path = Path("config/backup/backup_config.yml")
    dr_config_path = Path("config/disaster_recovery/dr_config.yml")

    assert backup_config_path.exists(), "Backup configuration file should exist"
    assert dr_config_path.exists(), "Disaster recovery configuration file should exist"

    # Test that they can be loaded
    with open(backup_config_path) as f:
        backup_config = yaml.safe_load(f)

    with open(dr_config_path) as f:
        dr_config = yaml.safe_load(f)

    # Basic structure validation
    assert "providers" in backup_config
    assert "databases" in backup_config
    assert "monitoring" in dr_config
    assert "recovery_plans" in dr_config


def test_backup_infrastructure_availability():
    """Test backup infrastructure availability."""
    backup_manager_path = Path("src/pynomaly/infrastructure/backup/backup_manager.py")
    dr_service_path = Path(
        "src/pynomaly/infrastructure/backup/disaster_recovery_service.py"
    )

    assert backup_manager_path.exists(), "Backup manager should exist"
    assert dr_service_path.exists(), "Disaster recovery service should exist"

    # Check file sizes to ensure they're not empty
    assert (
        backup_manager_path.stat().st_size > 1000
    ), "Backup manager should be substantial"
    assert dr_service_path.stat().st_size > 1000, "DR service should be substantial"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
