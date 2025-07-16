# Pynomaly Backup and Recovery Procedures

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 💾 Backup & Recovery

This comprehensive guide covers backup and recovery procedures for Pynomaly production deployments, including automated backup strategies, disaster recovery planning, and step-by-step recovery procedures.

## 📋 Table of Contents

- [Overview](#overview)
- [Backup Strategy](#backup-strategy)
- [Database Backups](#database-backups)
- [Model Registry Backups](#model-registry-backups)
- [Configuration Backups](#configuration-backups)
- [File System Backups](#file-system-backups)
- [Recovery Procedures](#recovery-procedures)
- [Disaster Recovery](#disaster-recovery)
- [Testing & Validation](#testing--validation)
- [Monitoring & Alerting](#monitoring--alerting)

## 🎯 Overview

### Backup Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Production Environment                   │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐     │
│  │   PostgreSQL  │ │ Model Registry│ │ Configuration │     │
│  │   Database    │ │   (S3/MinIO)  │ │   Files       │     │
│  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘     │
│          │                 │                 │             │
└──────────┼─────────────────┼─────────────────┼─────────────┘
           │                 │                 │
    ┌──────▼─────────────────▼─────────────────▼──────┐
    │              Backup Controller                   │
    │    (Automated Backup Orchestration)              │
    └──────┬─────────────────┬─────────────────┬──────┘
           │                 │                 │
    ┌──────▼──────┐ ┌────────▼────────┐ ┌──────▼──────┐
    │   Primary   │ │   Secondary     │ │   Archive   │
    │   Backup    │ │   Backup        │ │   Storage   │
    │  (Daily)    │ │  (Real-time)    │ │ (Long-term) │
    └─────────────┘ └─────────────────┘ └─────────────┘
```

### Backup Objectives

- **Recovery Time Objective (RTO)**: < 30 minutes for critical systems
- **Recovery Point Objective (RPO)**: < 15 minutes data loss maximum
- **Backup Frequency**: Multiple tiers (Real-time, Hourly, Daily, Weekly, Monthly)
- **Retention Policy**: 30 days daily, 12 weeks weekly, 12 months monthly
- **Geographic Distribution**: Multi-region backup storage

## 📊 Backup Strategy

### Backup Tiers

#### Tier 1: Real-time Replication
- **PostgreSQL Streaming Replication**: Real-time standby replica
- **Model Registry Sync**: Continuous synchronization to secondary storage
- **Configuration Sync**: Real-time configuration replication

#### Tier 2: Frequent Backups
- **Database Snapshots**: Every 15 minutes
- **Transaction Log Shipping**: Continuous
- **Model Versioning**: Every model update

#### Tier 3: Scheduled Backups
- **Full Database Backup**: Daily at 2:00 AM UTC
- **Incremental Backups**: Every 4 hours
- **System Configuration**: Daily
- **Application Logs**: Hourly rotation and backup

#### Tier 4: Archive Storage
- **Monthly Full Backups**: Long-term retention
- **Compliance Backups**: Regulatory requirements
- **Historical Data**: Data warehouse snapshots

### Backup Retention Policy

```bash
#!/bin/bash
# backup-retention-policy.sh

# Retention periods
DAILY_RETENTION_DAYS=30
WEEKLY_RETENTION_WEEKS=12
MONTHLY_RETENTION_MONTHS=12
YEARLY_RETENTION_YEARS=7

# Cleanup old backups
cleanup_old_backups() {
    local backup_type=$1
    local retention_period=$2
    
    echo "🧹 Cleaning up old $backup_type backups..."
    
    case $backup_type in
        "daily")
            find /backup/daily -name "*.sql.gz" -mtime +$retention_period -delete
            ;;
        "weekly")
            find /backup/weekly -name "*.sql.gz" -mtime +$((retention_period * 7)) -delete
            ;;
        "monthly")
            find /backup/monthly -name "*.sql.gz" -mtime +$((retention_period * 30)) -delete
            ;;
        "yearly")
            find /backup/yearly -name "*.sql.gz" -mtime +$((retention_period * 365)) -delete
            ;;
    esac
}

# Execute cleanup
cleanup_old_backups "daily" $DAILY_RETENTION_DAYS
cleanup_old_backups "weekly" $WEEKLY_RETENTION_WEEKS
cleanup_old_backups "monthly" $MONTHLY_RETENTION_MONTHS
cleanup_old_backups "yearly" $YEARLY_RETENTION_YEARS
```

## 🗄️ Database Backups

### PostgreSQL Backup Configuration

#### Continuous Archiving Setup

```bash
#!/bin/bash
# setup-postgres-backup.sh

set -e

echo "🔧 Setting up PostgreSQL continuous archiving..."

# Create backup directories
mkdir -p /backup/postgres/{daily,hourly,wal}
chown postgres:postgres /backup/postgres -R

# Configure PostgreSQL for archiving
cat >> /etc/postgresql/15/main/postgresql.conf << 'EOF'
# Archiving settings
archive_mode = on
archive_command = 'cp %p /backup/postgres/wal/%f'
archive_timeout = 300  # 5 minutes

# WAL settings
wal_level = replica
max_wal_senders = 3
wal_keep_size = 1GB

# Backup settings
full_page_writes = on
wal_log_hints = on
EOF

# Restart PostgreSQL
systemctl restart postgresql

echo "✅ PostgreSQL backup configuration complete"
```

#### Automated Database Backup Script

```bash
#!/bin/bash
# postgres-backup.sh

set -e

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-pynomaly}"
DB_USER="${DB_USER:-postgres}"
BACKUP_DIR="/backup/postgres"
S3_BUCKET="${S3_BUCKET:-pynomaly-backups}"
RETENTION_DAYS=30

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="pynomaly_backup_${TIMESTAMP}.sql.gz"
BACKUP_PATH="${BACKUP_DIR}/daily/${BACKUP_FILE}"

echo "🚀 Starting PostgreSQL backup: ${BACKUP_FILE}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/daily"

# Perform database backup
pg_dump \
    --host="$DB_HOST" \
    --port="$DB_PORT" \
    --username="$DB_USER" \
    --dbname="$DB_NAME" \
    --format=custom \
    --compress=9 \
    --no-password \
    --verbose \
    --file="${BACKUP_PATH%.gz}"

# Compress backup
gzip "${BACKUP_PATH%.gz}"

# Verify backup integrity
echo "🔍 Verifying backup integrity..."
pg_restore --list "${BACKUP_PATH}" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Backup verification successful"
else
    echo "❌ Backup verification failed"
    exit 1
fi

# Upload to S3 (if configured)
if [ -n "$S3_BUCKET" ]; then
    echo "☁️ Uploading backup to S3..."
    aws s3 cp "${BACKUP_PATH}" "s3://${S3_BUCKET}/postgres/daily/" \
        --storage-class STANDARD_IA \
        --server-side-encryption AES256
    
    if [ $? -eq 0 ]; then
        echo "✅ S3 upload successful"
    else
        echo "❌ S3 upload failed"
        exit 1
    fi
fi

# Calculate backup size and duration
BACKUP_SIZE=$(du -h "${BACKUP_PATH}" | cut -f1)
echo "📊 Backup completed: ${BACKUP_SIZE}"

# Log backup completion
echo "$(date): Backup ${BACKUP_FILE} completed successfully (${BACKUP_SIZE})" >> "${BACKUP_DIR}/backup.log"

# Cleanup old backups
echo "🧹 Cleaning up old backups..."
find "${BACKUP_DIR}/daily" -name "pynomaly_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete

echo "✅ PostgreSQL backup process completed"
```

#### Point-in-Time Recovery Setup

```sql
-- point-in-time-recovery.sql
-- Setup for PostgreSQL Point-in-Time Recovery

-- Create backup user with minimal privileges
CREATE USER backup_user WITH PASSWORD 'secure_backup_password';
GRANT CONNECT ON DATABASE pynomaly TO backup_user;
GRANT USAGE ON SCHEMA public TO backup_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO backup_user;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO backup_user;

-- Create backup metadata table
CREATE TABLE IF NOT EXISTS backup_metadata (
    id SERIAL PRIMARY KEY,
    backup_type VARCHAR(50) NOT NULL,
    backup_file VARCHAR(255) NOT NULL,
    backup_size BIGINT,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'in_progress',
    checksum VARCHAR(64),
    wal_start_lsn pg_lsn,
    wal_end_lsn pg_lsn
);

-- Function to log backup operations
CREATE OR REPLACE FUNCTION log_backup_operation(
    p_backup_type VARCHAR(50),
    p_backup_file VARCHAR(255),
    p_status VARCHAR(20) DEFAULT 'started'
) RETURNS INTEGER AS $$
DECLARE
    backup_id INTEGER;
BEGIN
    INSERT INTO backup_metadata (backup_type, backup_file, start_time, status, wal_start_lsn)
    VALUES (p_backup_type, p_backup_file, NOW(), p_status, pg_current_wal_lsn())
    RETURNING id INTO backup_id;
    
    RETURN backup_id;
END;
$$ LANGUAGE plpgsql;
```

### Database Replication Setup

```bash
#!/bin/bash
# setup-postgres-replication.sh

set -e

echo "🔧 Setting up PostgreSQL streaming replication..."

# Primary server configuration
setup_primary() {
    echo "Configuring primary server..."
    
    # Create replication user
    sudo -u postgres psql -c "CREATE USER replicator REPLICATION LOGIN PASSWORD 'replication_password';"
    
    # Configure pg_hba.conf
    echo "host replication replicator 10.0.0.0/24 md5" >> /etc/postgresql/15/main/pg_hba.conf
    
    # Configure postgresql.conf
    cat >> /etc/postgresql/15/main/postgresql.conf << 'EOF'
listen_addresses = '*'
max_wal_senders = 3
wal_level = replica
hot_standby = on
EOF
    
    systemctl restart postgresql
}

# Standby server configuration
setup_standby() {
    echo "Configuring standby server..."
    
    # Stop PostgreSQL
    systemctl stop postgresql
    
    # Remove existing data
    rm -rf /var/lib/postgresql/15/main/*
    
    # Create base backup from primary
    sudo -u postgres pg_basebackup -h primary_server_ip -D /var/lib/postgresql/15/main -U replicator -P -W -R
    
    # Start PostgreSQL
    systemctl start postgresql
}

# Check if this is primary or standby
if [ "$1" = "primary" ]; then
    setup_primary
elif [ "$1" = "standby" ]; then
    setup_standby
else
    echo "Usage: $0 {primary|standby}"
    exit 1
fi

echo "✅ PostgreSQL replication setup complete"
```

## 🤖 Model Registry Backups

### Model Backup Strategy

```python
#!/usr/bin/env python3
# model-backup.py

import os
import boto3
import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional

class ModelRegistryBackup:
    """Automated backup system for ML model registry."""
    
    def __init__(self, 
                 local_registry_path: str = "/data/models",
                 s3_bucket: str = "pynomaly-model-backups",
                 backup_frequency: str = "daily"):
        self.local_registry_path = Path(local_registry_path)
        self.s3_bucket = s3_bucket
        self.backup_frequency = backup_frequency
        self.s3_client = boto3.client('s3')
        
    def create_model_manifest(self) -> Dict:
        """Create manifest of all models in registry."""
        manifest = {
            "backup_timestamp": datetime.datetime.utcnow().isoformat(),
            "registry_path": str(self.local_registry_path),
            "models": []
        }
        
        for model_dir in self.local_registry_path.iterdir():
            if model_dir.is_dir():
                model_info = self._get_model_info(model_dir)
                if model_info:
                    manifest["models"].append(model_info)
        
        return manifest
    
    def _get_model_info(self, model_dir: Path) -> Optional[Dict]:
        """Extract model information and calculate checksums."""
        try:
            model_info = {
                "model_id": model_dir.name,
                "path": str(model_dir),
                "files": [],
                "total_size": 0
            }
            
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_checksum = self._calculate_checksum(file_path)
                    
                    model_info["files"].append({
                        "name": file_path.name,
                        "relative_path": str(file_path.relative_to(model_dir)),
                        "size": file_size,
                        "checksum": file_checksum,
                        "modified": file_path.stat().st_mtime
                    })
                    model_info["total_size"] += file_size
            
            return model_info
        except Exception as e:
            print(f"Error processing model {model_dir}: {e}")
            return None
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def backup_models(self) -> bool:
        """Backup all models to S3."""
        try:
            print("🚀 Starting model registry backup...")
            
            # Create manifest
            manifest = self.create_model_manifest()
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Upload models
            for model in manifest["models"]:
                model_backup_key = f"models/{timestamp}/{model['model_id']}"
                self._backup_model_directory(
                    Path(model["path"]), 
                    model_backup_key
                )
            
            # Upload manifest
            manifest_key = f"manifests/manifest_{timestamp}.json"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType='application/json'
            )
            
            print(f"✅ Model backup completed: {len(manifest['models'])} models")
            return True
            
        except Exception as e:
            print(f"❌ Model backup failed: {e}")
            return False
    
    def _backup_model_directory(self, model_dir: Path, s3_prefix: str):
        """Backup entire model directory to S3."""
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(model_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                self.s3_client.upload_file(
                    str(file_path),
                    self.s3_bucket,
                    s3_key,
                    ExtraArgs={
                        'StorageClass': 'STANDARD_IA',
                        'ServerSideEncryption': 'AES256'
                    }
                )
    
    def restore_models(self, backup_timestamp: str, 
                      target_path: Optional[str] = None) -> bool:
        """Restore models from backup."""
        try:
            restore_path = Path(target_path) if target_path else self.local_registry_path
            restore_path.mkdir(parents=True, exist_ok=True)
            
            print(f"🔄 Restoring models from backup {backup_timestamp}...")
            
            # Download manifest
            manifest_key = f"manifests/manifest_{backup_timestamp}.json"
            manifest_obj = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=manifest_key
            )
            manifest = json.loads(manifest_obj['Body'].read())
            
            # Restore each model
            for model in manifest["models"]:
                model_restore_path = restore_path / model["model_id"]
                model_restore_path.mkdir(parents=True, exist_ok=True)
                
                self._restore_model_directory(
                    f"models/{backup_timestamp}/{model['model_id']}",
                    model_restore_path
                )
            
            print(f"✅ Model restore completed: {len(manifest['models'])} models")
            return True
            
        except Exception as e:
            print(f"❌ Model restore failed: {e}")
            return False
    
    def _restore_model_directory(self, s3_prefix: str, local_dir: Path):
        """Restore model directory from S3."""
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    local_file_path = local_dir / s3_key.replace(f"{s3_prefix}/", "")
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.s3_client.download_file(
                        self.s3_bucket,
                        s3_key,
                        str(local_file_path)
                    )

if __name__ == "__main__":
    backup_system = ModelRegistryBackup()
    backup_system.backup_models()
```

## ⚙️ Configuration Backups

### Configuration Backup Script

```bash
#!/bin/bash
# config-backup.sh

set -e

# Configuration
CONFIG_DIRS=(
    "/etc/pynomaly"
    "/opt/pynomaly/config"
    "/etc/nginx/sites-available"
    "/etc/systemd/system"
    "/etc/prometheus"
    "/etc/grafana"
)
BACKUP_DIR="/backup/config"
S3_BUCKET="${S3_BUCKET:-pynomaly-backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🚀 Starting configuration backup..."

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Create configuration archive
CONFIG_ARCHIVE="${BACKUP_DIR}/config_backup_${TIMESTAMP}.tar.gz"

echo "📦 Creating configuration archive..."
tar -czf "${CONFIG_ARCHIVE}" \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='*.cache' \
    "${CONFIG_DIRS[@]}" 2>/dev/null || true

# Verify archive
if [ -f "${CONFIG_ARCHIVE}" ]; then
    ARCHIVE_SIZE=$(du -h "${CONFIG_ARCHIVE}" | cut -f1)
    echo "✅ Configuration archive created: ${ARCHIVE_SIZE}"
else
    echo "❌ Failed to create configuration archive"
    exit 1
fi

# Upload to S3
if [ -n "$S3_BUCKET" ]; then
    echo "☁️ Uploading configuration backup to S3..."
    aws s3 cp "${CONFIG_ARCHIVE}" "s3://${S3_BUCKET}/config/" \
        --storage-class STANDARD_IA \
        --server-side-encryption AES256
    
    if [ $? -eq 0 ]; then
        echo "✅ S3 upload successful"
    else
        echo "❌ S3 upload failed"
        exit 1
    fi
fi

# Cleanup old backups
echo "🧹 Cleaning up old configuration backups..."
find "${BACKUP_DIR}" -name "config_backup_*.tar.gz" -mtime +30 -delete

echo "✅ Configuration backup completed"
```

## 🔄 Recovery Procedures

### Complete System Recovery

```bash
#!/bin/bash
# system-recovery.sh

set -e

BACKUP_DATE="$1"
RECOVERY_MODE="$2"  # full|database|models|config

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date> [recovery_mode]"
    echo "Example: $0 20231215_143000 full"
    exit 1
fi

echo "🔄 Starting system recovery for backup: $BACKUP_DATE"

# Full system recovery
recover_full_system() {
    echo "📋 Performing full system recovery..."
    
    # Stop all services
    echo "🛑 Stopping all services..."
    docker-compose down || true
    systemctl stop postgresql || true
    systemctl stop redis || true
    
    # Recover database
    recover_database
    
    # Recover models
    recover_models
    
    # Recover configuration
    recover_configuration
    
    # Start services
    echo "🚀 Starting services..."
    systemctl start postgresql
    systemctl start redis
    docker-compose up -d
    
    # Verify recovery
    verify_recovery
    
    echo "✅ Full system recovery completed"
}

# Database recovery
recover_database() {
    echo "🗄️ Recovering database..."
    
    # Stop PostgreSQL
    systemctl stop postgresql
    
    # Backup current data (if exists)
    if [ -d "/var/lib/postgresql/15/main" ]; then
        mv /var/lib/postgresql/15/main "/var/lib/postgresql/15/main.backup.$(date +%s)"
    fi
    
    # Download backup from S3
    BACKUP_FILE="pynomaly_backup_${BACKUP_DATE}.sql.gz"
    aws s3 cp "s3://${S3_BUCKET}/postgres/daily/${BACKUP_FILE}" "/tmp/"
    
    # Initialize new database
    sudo -u postgres initdb -D /var/lib/postgresql/15/main
    systemctl start postgresql
    
    # Create database
    sudo -u postgres createdb pynomaly
    
    # Restore from backup
    gunzip -c "/tmp/${BACKUP_FILE}" | sudo -u postgres pg_restore -d pynomaly -v
    
    echo "✅ Database recovery completed"
}

# Model registry recovery
recover_models() {
    echo "🤖 Recovering model registry..."
    
    # Backup current models
    if [ -d "/data/models" ]; then
        mv /data/models "/data/models.backup.$(date +%s)"
    fi
    
    # Create models directory
    mkdir -p /data/models
    
    # Restore models using Python script
    python3 /scripts/model-backup.py restore "$BACKUP_DATE" "/data/models"
    
    echo "✅ Model registry recovery completed"
}

# Configuration recovery
recover_configuration() {
    echo "⚙️ Recovering configuration..."
    
    # Download configuration backup
    CONFIG_ARCHIVE="config_backup_${BACKUP_DATE}.tar.gz"
    aws s3 cp "s3://${S3_BUCKET}/config/${CONFIG_ARCHIVE}" "/tmp/"
    
    # Extract configuration
    cd /
    tar -xzf "/tmp/${CONFIG_ARCHIVE}"
    
    # Reload systemd
    systemctl daemon-reload
    
    echo "✅ Configuration recovery completed"
}

# Verify recovery
verify_recovery() {
    echo "🔍 Verifying recovery..."
    
    # Check database connectivity
    sudo -u postgres psql -d pynomaly -c "SELECT COUNT(*) FROM pg_tables;"
    
    # Check model registry
    ls -la /data/models/
    
    # Check API health
    sleep 30  # Wait for services to start
    curl -f http://localhost:8000/health || echo "⚠️ API health check failed"
    
    echo "✅ Recovery verification completed"
}

# Execute recovery based on mode
case "$RECOVERY_MODE" in
    "full"|"")
        recover_full_system
        ;;
    "database")
        recover_database
        ;;
    "models")
        recover_models
        ;;
    "config")
        recover_configuration
        ;;
    *)
        echo "Invalid recovery mode: $RECOVERY_MODE"
        echo "Valid modes: full, database, models, config"
        exit 1
        ;;
esac

echo "🎉 Recovery process completed successfully"
```

### Point-in-Time Recovery

```bash
#!/bin/bash
# point-in-time-recovery.sh

set -e

TARGET_TIME="$1"  # Format: 2023-12-15 14:30:00

if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 '<target_time>'"
    echo "Example: $0 '2023-12-15 14:30:00'"
    exit 1
fi

echo "🕐 Starting point-in-time recovery to: $TARGET_TIME"

# Stop PostgreSQL
systemctl stop postgresql

# Backup current data
mv /var/lib/postgresql/15/main "/var/lib/postgresql/15/main.backup.$(date +%s)"

# Find appropriate base backup
BASE_BACKUP=$(aws s3 ls s3://${S3_BUCKET}/postgres/daily/ \
    | awk '$1 <= "'$(date -d "$TARGET_TIME" +%Y-%m-%d)'"' \
    | tail -1 \
    | awk '{print $4}')

echo "📦 Using base backup: $BASE_BACKUP"

# Download and restore base backup
aws s3 cp "s3://${S3_BUCKET}/postgres/daily/${BASE_BACKUP}" /tmp/
gunzip -c "/tmp/${BASE_BACKUP}" | sudo -u postgres pg_restore -C -d template1

# Create recovery configuration
cat > /var/lib/postgresql/15/main/recovery.conf << EOF
restore_command = 'aws s3 cp s3://${S3_BUCKET}/postgres/wal/%f %p'
recovery_target_time = '$TARGET_TIME'
recovery_target_action = 'promote'
EOF

# Start PostgreSQL in recovery mode
systemctl start postgresql

echo "✅ Point-in-time recovery initiated"
echo "🔍 Monitor PostgreSQL logs for recovery progress"
```

## 🚨 Disaster Recovery

### Multi-Region Disaster Recovery

```yaml
# disaster-recovery-plan.yml
disaster_recovery:
  rto_target: "30 minutes"
  rpo_target: "15 minutes"
  
  regions:
    primary:
      region: "us-east-1"
      availability_zones: ["us-east-1a", "us-east-1b", "us-east-1c"]
      backup_frequency: "continuous"
      
    secondary:
      region: "us-west-2"
      availability_zones: ["us-west-2a", "us-west-2b"]
      backup_frequency: "4 hours"
      sync_delay: "< 15 minutes"
      
    archive:
      region: "eu-west-1"
      storage_class: "GLACIER"
      retention: "7 years"

  failover_triggers:
    - primary_region_unavailable
    - database_corruption
    - security_incident
    - planned_maintenance

  automated_failover:
    enabled: true
    health_check_interval: "30 seconds"
    failure_threshold: 3
    recovery_verification: true
    
  manual_procedures:
    - executive_approval_required: true
    - communication_plan: true
    - customer_notification: true
    - service_status_updates: true
```

### Disaster Recovery Automation

```python
#!/usr/bin/env python3
# disaster-recovery.py

import boto3
import json
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class DisasterRecoveryConfig:
    primary_region: str = "us-east-1"
    secondary_region: str = "us-west-2"
    rto_minutes: int = 30
    rpo_minutes: int = 15
    health_check_interval: int = 30
    failure_threshold: int = 3

class DisasterRecoveryOrchestrator:
    """Automated disaster recovery orchestration."""
    
    def __init__(self, config: DisasterRecoveryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # AWS clients for different regions
        self.primary_session = boto3.Session(region_name=config.primary_region)
        self.secondary_session = boto3.Session(region_name=config.secondary_region)
        
        self.primary_rds = self.primary_session.client('rds')
        self.secondary_rds = self.secondary_session.client('rds')
        
        self.primary_ecs = self.primary_session.client('ecs')
        self.secondary_ecs = self.secondary_session.client('ecs')
        
        self.route53 = boto3.client('route53')
        
    def monitor_primary_health(self) -> bool:
        """Monitor primary region health status."""
        try:
            # Check RDS instance health
            rds_health = self._check_rds_health()
            
            # Check ECS service health
            ecs_health = self._check_ecs_health()
            
            # Check API endpoint health
            api_health = self._check_api_health()
            
            overall_health = rds_health and ecs_health and api_health
            
            self.logger.info(f"Primary region health: RDS={rds_health}, ECS={ecs_health}, API={api_health}")
            
            return overall_health
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _check_rds_health(self) -> bool:
        """Check RDS instance availability."""
        try:
            response = self.primary_rds.describe_db_instances(
                DBInstanceIdentifier='pynomaly-production'
            )
            
            instance = response['DBInstances'][0]
            return instance['DBInstanceStatus'] == 'available'
            
        except Exception as e:
            self.logger.error(f"RDS health check failed: {e}")
            return False
    
    def _check_ecs_health(self) -> bool:
        """Check ECS service health."""
        try:
            response = self.primary_ecs.describe_services(
                cluster='pynomaly-production',
                services=['pynomaly-api']
            )
            
            service = response['services'][0]
            return service['runningCount'] > 0 and service['status'] == 'ACTIVE'
            
        except Exception as e:
            self.logger.error(f"ECS health check failed: {e}")
            return False
    
    def _check_api_health(self) -> bool:
        """Check API endpoint health."""
        try:
            import requests
            response = requests.get(
                'https://api.pynomaly.com/health',
                timeout=10
            )
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"API health check failed: {e}")
            return False
    
    def initiate_failover(self) -> bool:
        """Initiate disaster recovery failover."""
        try:
            self.logger.info("🚨 Initiating disaster recovery failover...")
            
            # Step 1: Promote secondary RDS instance
            self._promote_secondary_rds()
            
            # Step 2: Scale up secondary ECS services
            self._scale_up_secondary_services()
            
            # Step 3: Update DNS routing
            self._update_dns_routing()
            
            # Step 4: Verify secondary region health
            secondary_healthy = self._verify_secondary_health()
            
            if secondary_healthy:
                self.logger.info("✅ Disaster recovery failover completed successfully")
                self._send_failover_notification("SUCCESS")
                return True
            else:
                self.logger.error("❌ Secondary region health verification failed")
                self._send_failover_notification("FAILED")
                return False
                
        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            self._send_failover_notification("ERROR", str(e))
            return False
    
    def _promote_secondary_rds(self):
        """Promote secondary RDS read replica."""
        try:
            self.logger.info("📊 Promoting secondary RDS instance...")
            
            self.secondary_rds.promote_read_replica(
                DBInstanceIdentifier='pynomaly-production-replica'
            )
            
            # Wait for promotion to complete
            waiter = self.secondary_rds.get_waiter('db_instance_available')
            waiter.wait(
                DBInstanceIdentifier='pynomaly-production-replica',
                WaiterConfig={'Delay': 30, 'MaxAttempts': 20}
            )
            
            self.logger.info("✅ RDS promotion completed")
            
        except Exception as e:
            self.logger.error(f"RDS promotion failed: {e}")
            raise
    
    def _scale_up_secondary_services(self):
        """Scale up ECS services in secondary region."""
        try:
            self.logger.info("🚀 Scaling up secondary ECS services...")
            
            self.secondary_ecs.update_service(
                cluster='pynomaly-production',
                service='pynomaly-api',
                desiredCount=4  # Scale to production capacity
            )
            
            # Wait for services to be stable
            waiter = self.secondary_ecs.get_waiter('services_stable')
            waiter.wait(
                cluster='pynomaly-production',
                services=['pynomaly-api'],
                WaiterConfig={'Delay': 30, 'MaxAttempts': 20}
            )
            
            self.logger.info("✅ ECS scaling completed")
            
        except Exception as e:
            self.logger.error(f"ECS scaling failed: {e}")
            raise
    
    def _update_dns_routing(self):
        """Update Route53 DNS to point to secondary region."""
        try:
            self.logger.info("🌐 Updating DNS routing...")
            
            # Get hosted zone
            response = self.route53.list_hosted_zones_by_name(
                DNSName='pynomaly.com'
            )
            hosted_zone_id = response['HostedZones'][0]['Id']
            
            # Update A record to point to secondary region ALB
            self.route53.change_resource_record_sets(
                HostedZoneId=hosted_zone_id,
                ChangeBatch={
                    'Changes': [{
                        'Action': 'UPSERT',
                        'ResourceRecordSet': {
                            'Name': 'api.pynomaly.com',
                            'Type': 'A',
                            'AliasTarget': {
                                'DNSName': 'secondary-alb.us-west-2.elb.amazonaws.com',
                                'EvaluateTargetHealth': True,
                                'HostedZoneId': 'Z1D633PJN98FT9'  # ALB hosted zone
                            }
                        }
                    }]
                }
            )
            
            self.logger.info("✅ DNS routing updated")
            
        except Exception as e:
            self.logger.error(f"DNS update failed: {e}")
            raise
    
    def _verify_secondary_health(self) -> bool:
        """Verify secondary region health after failover."""
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                import requests
                response = requests.get(
                    'https://api.pynomaly.com/health',
                    timeout=10
                )
                if response.status_code == 200:
                    return True
            except:
                pass
            
            if attempt < max_attempts - 1:
                time.sleep(30)  # Wait before retry
        
        return False
    
    def _send_failover_notification(self, status: str, error: Optional[str] = None):
        """Send failover notification to operations team."""
        try:
            sns = boto3.client('sns')
            
            message = f"""
🚨 DISASTER RECOVERY FAILOVER - {status}

Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}
Primary Region: {self.config.primary_region}
Secondary Region: {self.config.secondary_region}
"""
            
            if error:
                message += f"\nError Details: {error}"
            
            sns.publish(
                TopicArn='arn:aws:sns:us-east-1:123456789012:pynomaly-alerts',
                Subject=f'🚨 DR Failover {status}',
                Message=message
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")

def main():
    """Main disaster recovery monitoring loop."""
    config = DisasterRecoveryConfig()
    dr_orchestrator = DisasterRecoveryOrchestrator(config)
    
    consecutive_failures = 0
    
    while True:
        try:
            is_healthy = dr_orchestrator.monitor_primary_health()
            
            if is_healthy:
                consecutive_failures = 0
                logging.info("✅ Primary region is healthy")
            else:
                consecutive_failures += 1
                logging.warning(f"⚠️ Primary region unhealthy (attempt {consecutive_failures}/{config.failure_threshold})")
                
                if consecutive_failures >= config.failure_threshold:
                    logging.error("🚨 Failure threshold reached, initiating failover...")
                    dr_orchestrator.initiate_failover()
                    break  # Exit monitoring after failover
            
            time.sleep(config.health_check_interval)
            
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
            break
        except Exception as e:
            logging.error(f"Monitoring error: {e}")
            time.sleep(config.health_check_interval)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

## ✅ Testing & Validation

### Backup Testing Automation

```bash
#!/bin/bash
# test-backup-restore.sh

set -e

echo "🧪 Starting backup and restore testing..."

# Test configuration
TEST_DB="pynomaly_test_restore"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
TEST_DATA_SIZE=1000

# Create test database with sample data
create_test_data() {
    echo "📊 Creating test data..."
    
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS $TEST_DB;"
    sudo -u postgres psql -c "CREATE DATABASE $TEST_DB;"
    
    # Insert test data
    sudo -u postgres psql -d $TEST_DB << EOF
CREATE TABLE test_anomalies (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    value FLOAT,
    is_anomaly BOOLEAN,
    model_name VARCHAR(100)
);

INSERT INTO test_anomalies (value, is_anomaly, model_name)
SELECT 
    random() * 100,
    random() > 0.9,
    'test_model_' || (random() * 10)::int
FROM generate_series(1, $TEST_DATA_SIZE);
EOF
    
    echo "✅ Test data created: $TEST_DATA_SIZE records"
}

# Test backup creation
test_backup_creation() {
    echo "💾 Testing backup creation..."
    
    # Create backup
    pg_dump \
        --host=localhost \
        --port=5432 \
        --username=postgres \
        --dbname=$TEST_DB \
        --format=custom \
        --compress=9 \
        --file="/tmp/test_backup_${BACKUP_DATE}.sql"
    
    # Verify backup file exists and has content
    if [ -f "/tmp/test_backup_${BACKUP_DATE}.sql" ] && [ -s "/tmp/test_backup_${BACKUP_DATE}.sql" ]; then
        BACKUP_SIZE=$(du -h "/tmp/test_backup_${BACKUP_DATE}.sql" | cut -f1)
        echo "✅ Backup created successfully: $BACKUP_SIZE"
        return 0
    else
        echo "❌ Backup creation failed"
        return 1
    fi
}

# Test backup restoration
test_backup_restoration() {
    echo "🔄 Testing backup restoration..."
    
    # Drop and recreate database
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS ${TEST_DB}_restored;"
    sudo -u postgres psql -c "CREATE DATABASE ${TEST_DB}_restored;"
    
    # Restore from backup
    pg_restore \
        --host=localhost \
        --port=5432 \
        --username=postgres \
        --dbname="${TEST_DB}_restored" \
        --verbose \
        "/tmp/test_backup_${BACKUP_DATE}.sql"
    
    # Verify restored data
    ORIGINAL_COUNT=$(sudo -u postgres psql -d $TEST_DB -t -c "SELECT COUNT(*) FROM test_anomalies;" | xargs)
    RESTORED_COUNT=$(sudo -u postgres psql -d "${TEST_DB}_restored" -t -c "SELECT COUNT(*) FROM test_anomalies;" | xargs)
    
    if [ "$ORIGINAL_COUNT" = "$RESTORED_COUNT" ]; then
        echo "✅ Backup restoration successful: $RESTORED_COUNT records restored"
        return 0
    else
        echo "❌ Backup restoration failed: $ORIGINAL_COUNT != $RESTORED_COUNT"
        return 1
    fi
}

# Test data integrity
test_data_integrity() {
    echo "🔍 Testing data integrity..."
    
    # Compare checksums
    ORIGINAL_CHECKSUM=$(sudo -u postgres psql -d $TEST_DB -t -c "SELECT md5(string_agg(id::text || value::text, '' ORDER BY id)) FROM test_anomalies;" | xargs)
    RESTORED_CHECKSUM=$(sudo -u postgres psql -d "${TEST_DB}_restored" -t -c "SELECT md5(string_agg(id::text || value::text, '' ORDER BY id)) FROM test_anomalies;" | xargs)
    
    if [ "$ORIGINAL_CHECKSUM" = "$RESTORED_CHECKSUM" ]; then
        echo "✅ Data integrity verified: checksums match"
        return 0
    else
        echo "❌ Data integrity check failed: checksums don't match"
        return 1
    fi
}

# Cleanup test resources
cleanup_test_resources() {
    echo "🧹 Cleaning up test resources..."
    
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS $TEST_DB;" 2>/dev/null || true
    sudo -u postgres psql -c "DROP DATABASE IF EXISTS ${TEST_DB}_restored;" 2>/dev/null || true
    rm -f "/tmp/test_backup_${BACKUP_DATE}.sql" 2>/dev/null || true
    
    echo "✅ Cleanup completed"
}

# Run all tests
run_backup_tests() {
    local test_results=()
    
    # Run tests
    create_test_data && test_results+=("✅ Test data creation: PASSED") || test_results+=("❌ Test data creation: FAILED")
    test_backup_creation && test_results+=("✅ Backup creation: PASSED") || test_results+=("❌ Backup creation: FAILED")
    test_backup_restoration && test_results+=("✅ Backup restoration: PASSED") || test_results+=("❌ Backup restoration: FAILED")
    test_data_integrity && test_results+=("✅ Data integrity: PASSED") || test_results+=("❌ Data integrity: FAILED")
    
    # Print results
    echo
    echo "📋 Test Results Summary:"
    echo "========================"
    for result in "${test_results[@]}"; do
        echo "$result"
    done
    
    # Cleanup
    cleanup_test_resources
    
    # Check if all tests passed
    if echo "${test_results[@]}" | grep -q "FAILED"; then
        echo
        echo "❌ Some tests failed. Please review backup procedures."
        return 1
    else
        echo
        echo "✅ All backup tests passed successfully!"
        return 0
    fi
}

# Execute tests
run_backup_tests
```

## 📊 Monitoring & Alerting

### Backup Monitoring Dashboard

```python
#!/usr/bin/env python3
# backup-monitoring.py

import boto3
import json
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BackupMetrics:
    """Backup metrics data structure."""
    backup_type: str
    last_backup_time: datetime.datetime
    backup_size: int
    duration_seconds: int
    success_rate: float
    error_count: int

class BackupMonitor:
    """Monitor backup operations and generate alerts."""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.sns = boto3.client('sns')
        self.s3 = boto3.client('s3')
        
    def collect_backup_metrics(self) -> List[BackupMetrics]:
        """Collect backup metrics from various sources."""
        metrics = []
        
        # Database backup metrics
        db_metrics = self._get_database_backup_metrics()
        if db_metrics:
            metrics.append(db_metrics)
        
        # Model backup metrics
        model_metrics = self._get_model_backup_metrics()
        if model_metrics:
            metrics.append(model_metrics)
        
        # Configuration backup metrics
        config_metrics = self._get_config_backup_metrics()
        if config_metrics:
            metrics.append(config_metrics)
        
        return metrics
    
    def _get_database_backup_metrics(self) -> Optional[BackupMetrics]:
        """Get database backup metrics."""
        try:
            # Query CloudWatch for backup metrics
            response = self.cloudwatch.get_metric_statistics(
                Namespace='Pynomaly/Backups',
                MetricName='DatabaseBackupDuration',
                Dimensions=[
                    {'Name': 'BackupType', 'Value': 'Database'}
                ],
                StartTime=datetime.datetime.utcnow() - datetime.timedelta(days=1),
                EndTime=datetime.datetime.utcnow(),
                Period=3600,
                Statistics=['Average', 'Maximum']
            )
            
            if response['Datapoints']:
                latest = max(response['Datapoints'], key=lambda x: x['Timestamp'])
                
                return BackupMetrics(
                    backup_type='Database',
                    last_backup_time=latest['Timestamp'],
                    backup_size=0,  # Will be populated from S3
                    duration_seconds=int(latest['Average']),
                    success_rate=0.0,  # Will be calculated
                    error_count=0
                )
            
        except Exception as e:
            print(f"Error getting database backup metrics: {e}")
        
        return None
    
    def send_backup_alerts(self, metrics: List[BackupMetrics]):
        """Send alerts for backup issues."""
        current_time = datetime.datetime.utcnow()
        
        for metric in metrics:
            # Check if backup is overdue
            time_since_backup = current_time - metric.last_backup_time
            
            if time_since_backup.total_seconds() > 86400:  # 24 hours
                self._send_alert(
                    severity='CRITICAL',
                    message=f"Backup overdue for {metric.backup_type}: {time_since_backup}"
                )
            
            # Check success rate
            if metric.success_rate < 0.95:  # Less than 95% success rate
                self._send_alert(
                    severity='WARNING',
                    message=f"Low backup success rate for {metric.backup_type}: {metric.success_rate:.2%}"
                )
            
            # Check error count
            if metric.error_count > 5:
                self._send_alert(
                    severity='WARNING',
                    message=f"High error count for {metric.backup_type}: {metric.error_count} errors"
                )
    
    def _send_alert(self, severity: str, message: str):
        """Send alert notification."""
        try:
            self.sns.publish(
                TopicArn='arn:aws:sns:us-east-1:123456789012:pynomaly-backup-alerts',
                Subject=f'🚨 Backup Alert - {severity}',
                Message=f"""
Backup Alert - {severity}

Message: {message}
Timestamp: {datetime.datetime.utcnow().isoformat()}
Environment: Production

Please investigate backup procedures immediately.
"""
            )
        except Exception as e:
            print(f"Failed to send alert: {e}")

if __name__ == "__main__":
    monitor = BackupMonitor()
    metrics = monitor.collect_backup_metrics()
    monitor.send_backup_alerts(metrics)
```

---

## 📚 Related Documentation

- **[Production Deployment Guide](deployment.md)**: Complete deployment procedures
- **[Security Hardening Guide](SECURITY_HARDENING_GUIDE.md)**: Production security measures
- **[Monitoring Setup Guide](MONITORING_SETUP_GUIDE.md)**: Observability and alerting
- **[Production Checklist](PRODUCTION_CHECKLIST.md)**: Pre-deployment validation

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-15  
**Next Review**: 2024-04-15  

This backup and recovery guide ensures comprehensive data protection and rapid disaster recovery capabilities for Pynomaly production deployments.