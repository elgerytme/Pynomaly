# Backup and Recovery Procedures

## 📋 Overview

This runbook covers comprehensive backup and recovery procedures for Pynomaly production systems, including databases, application data, configurations, and complete disaster recovery scenarios.

**Frequency**: Daily automated, weekly manual verification  
**Retention**: 30 days local, 90 days remote, 1 year archives  
**Recovery Time Objective (RTO)**: 4 hours  
**Recovery Point Objective (RPO)**: 1 hour  

## 🗂️ Backup Categories

### 1. Database Backups

- **PostgreSQL**: Full daily, incremental hourly
- **Redis**: Snapshot every 6 hours
- **Configuration databases**: Daily

### 2. Application Data

- **User uploads**: Continuous sync to S3
- **Logs**: Real-time shipping to centralized logging
- **Configuration files**: Version controlled + daily backup

### 3. System Configuration

- **Infrastructure as Code**: Git repository
- **Application configurations**: Encrypted backups
- **Secrets and certificates**: Secure vault backups

## 🔄 Automated Backup Schedule

### Daily Backups (2:00 AM UTC)

```bash
# Database backup
0 2 * * * /opt/pynomaly/scripts/backup/backup-database.sh

# Application data backup  
15 2 * * * /opt/pynomaly/scripts/backup/backup-application-data.sh

# Configuration backup
30 2 * * * /opt/pynomaly/scripts/backup/backup-configurations.sh

# Log backup
45 2 * * * /opt/pynomaly/scripts/backup/backup-logs.sh
```

### Hourly Incremental Backups

```bash
# Database incremental
0 * * * * /opt/pynomaly/scripts/backup/backup-database-incremental.sh

# Critical application data
30 * * * * /opt/pynomaly/scripts/backup/backup-critical-data.sh
```

## 💾 Database Backup Procedures

### PostgreSQL Full Backup

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup/backup-database.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/pynomaly/database"
S3_BUCKET="pynomaly-backups-prod"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Database credentials from environment
source /opt/pynomaly/config/backup.env

# Perform database backup
echo "Starting PostgreSQL backup at $(date)"
pg_dump \
  --host="$DB_HOST" \
  --port="$DB_PORT" \
  --username="$DB_USER" \
  --dbname="$DB_NAME" \
  --no-password \
  --verbose \
  --compress=9 \
  --file="$BACKUP_DIR/pynomaly_db_$BACKUP_DATE.sql.gz"

# Verify backup integrity
if [ $? -eq 0 ]; then
    echo "Database backup completed successfully"
    
    # Upload to S3
    aws s3 cp "$BACKUP_DIR/pynomaly_db_$BACKUP_DATE.sql.gz" \
        "s3://$S3_BUCKET/database/daily/" \
        --storage-class STANDARD_IA
    
    # Update backup metadata
    echo "database_backup_$BACKUP_DATE.sql.gz,$(date -u +%Y-%m-%dT%H:%M:%SZ),$(stat -c%s $BACKUP_DIR/pynomaly_db_$BACKUP_DATE.sql.gz)" >> \
        "$BACKUP_DIR/backup_catalog.csv"
    
    # Clean up local backups older than 7 days
    find "$BACKUP_DIR" -name "pynomaly_db_*.sql.gz" -mtime +7 -delete
    
else
    echo "Database backup failed!"
    # Send alert
    curl -X POST "$SLACK_WEBHOOK" \
        -H 'Content-type: application/json' \
        --data '{"text":"🚨 Database backup failed for Pynomaly production"}'
    exit 1
fi
```

### PostgreSQL Incremental Backup

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup/backup-database-incremental.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/pynomaly/database/incremental"
WAL_DIR="/var/lib/postgresql/12/main/pg_wal"

# Create incremental backup using pg_basebackup
echo "Starting incremental backup at $(date)"

# Use WAL-E for continuous archiving
envdir /etc/wal-e.d/env wal-e backup-push /var/lib/postgresql/12/main

if [ $? -eq 0 ]; then
    echo "Incremental backup completed successfully"
else
    echo "Incremental backup failed!"
    exit 1
fi
```

### Redis Backup

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup/backup-redis.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/pynomaly/redis"
REDIS_HOST="prod-redis.amazonaws.com"

mkdir -p "$BACKUP_DIR"

# Create Redis snapshot
echo "Starting Redis backup at $(date)"
redis-cli -h "$REDIS_HOST" BGSAVE

# Wait for background save to complete
while [ $(redis-cli -h "$REDIS_HOST" LASTSAVE) -eq $(redis-cli -h "$REDIS_HOST" LASTSAVE) ]; do
    sleep 5
done

# Copy dump file
scp "redis@$REDIS_HOST:/var/lib/redis/dump.rdb" \
    "$BACKUP_DIR/redis_dump_$BACKUP_DATE.rdb"

# Compress and upload
gzip "$BACKUP_DIR/redis_dump_$BACKUP_DATE.rdb"
aws s3 cp "$BACKUP_DIR/redis_dump_$BACKUP_DATE.rdb.gz" \
    "s3://pynomaly-backups-prod/redis/"

echo "Redis backup completed at $(date)"
```

## 📁 Application Data Backup

### User Data and File Uploads

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup/backup-application-data.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
DATA_DIR="/opt/pynomaly/data"
BACKUP_DIR="/var/backups/pynomaly/application"

echo "Starting application data backup at $(date)"

# Create incremental backup using rsync
rsync -avz \
    --backup \
    --backup-dir="$BACKUP_DIR/incremental/$BACKUP_DATE" \
    "$DATA_DIR/" \
    "$BACKUP_DIR/current/"

# Sync to S3
aws s3 sync "$DATA_DIR/" "s3://pynomaly-data-prod/" \
    --exclude "*.tmp" \
    --exclude "*.log" \
    --storage-class STANDARD_IA

echo "Application data backup completed at $(date)"
```

### Configuration Backup

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup/backup-configurations.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
CONFIG_DIR="/opt/pynomaly/config"
BACKUP_DIR="/var/backups/pynomaly/config"

mkdir -p "$BACKUP_DIR"

echo "Starting configuration backup at $(date)"

# Create encrypted tar archive
tar -czf "$BACKUP_DIR/config_$BACKUP_DATE.tar.gz" \
    -C "/opt/pynomaly" \
    config/ \
    --exclude="*.tmp" \
    --exclude="*.pid"

# Encrypt sensitive configuration
gpg --symmetric \
    --cipher-algo AES256 \
    --compress-algo 1 \
    --s2k-mode 3 \
    --s2k-digest-algo SHA512 \
    --s2k-count 65536 \
    --passphrase-file /opt/pynomaly/backup.key \
    --output "$BACKUP_DIR/config_$BACKUP_DATE.tar.gz.gpg" \
    "$BACKUP_DIR/config_$BACKUP_DATE.tar.gz"

# Upload encrypted backup
aws s3 cp "$BACKUP_DIR/config_$BACKUP_DATE.tar.gz.gpg" \
    "s3://pynomaly-backups-prod/config/"

# Clean up unencrypted file
rm "$BACKUP_DIR/config_$BACKUP_DATE.tar.gz"

echo "Configuration backup completed at $(date)"
```

## 🔍 Backup Verification

### Daily Backup Verification Script

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup/verify-backups.sh

VERIFICATION_DATE=$(date +%Y%m%d)
BACKUP_DIR="/var/backups/pynomaly"
LOG_FILE="/var/log/pynomaly/backup-verification.log"

echo "Starting backup verification at $(date)" >> "$LOG_FILE"

# Verify database backup
echo "Verifying database backup..." >> "$LOG_FILE"
LATEST_DB_BACKUP=$(ls -t "$BACKUP_DIR"/database/pynomaly_db_*.sql.gz | head -1)

if [ -f "$LATEST_DB_BACKUP" ]; then
    # Test backup integrity
    gzip -t "$LATEST_DB_BACKUP"
    if [ $? -eq 0 ]; then
        echo "✅ Database backup integrity verified" >> "$LOG_FILE"
    else
        echo "❌ Database backup integrity check failed" >> "$LOG_FILE"
        exit 1
    fi
    
    # Test restore to temporary database
    TEMP_DB="pynomaly_backup_test_$VERIFICATION_DATE"
    createdb "$TEMP_DB"
    gunzip -c "$LATEST_DB_BACKUP" | psql "$TEMP_DB"
    
    if [ $? -eq 0 ]; then
        echo "✅ Database backup restore test successful" >> "$LOG_FILE"
        dropdb "$TEMP_DB"
    else
        echo "❌ Database backup restore test failed" >> "$LOG_FILE"
        dropdb "$TEMP_DB"
        exit 1
    fi
else
    echo "❌ No database backup found for verification" >> "$LOG_FILE"
    exit 1
fi

# Verify S3 backups
echo "Verifying S3 backups..." >> "$LOG_FILE"
S3_BACKUP_COUNT=$(aws s3 ls s3://pynomaly-backups-prod/database/daily/ | wc -l)

if [ "$S3_BACKUP_COUNT" -gt 0 ]; then
    echo "✅ S3 backups verified ($S3_BACKUP_COUNT files)" >> "$LOG_FILE"
else
    echo "❌ No S3 backups found" >> "$LOG_FILE"
    exit 1
fi

echo "Backup verification completed successfully at $(date)" >> "$LOG_FILE"
```

## 🔄 Recovery Procedures

### Database Recovery

#### Point-in-Time Recovery

```bash
#!/bin/bash
# /opt/pynomaly/scripts/recovery/restore-database-pit.sh

RESTORE_DATE="$1"  # Format: YYYY-MM-DD HH:MM:SS
BACKUP_DIR="/var/backups/pynomaly/database"

if [ -z "$RESTORE_DATE" ]; then
    echo "Usage: $0 'YYYY-MM-DD HH:MM:SS'"
    exit 1
fi

echo "Starting point-in-time recovery to $RESTORE_DATE"

# Stop application
systemctl stop pynomaly

# Stop PostgreSQL
systemctl stop postgresql

# Backup current data directory
mv /var/lib/postgresql/12/main /var/lib/postgresql/12/main.backup.$(date +%Y%m%d_%H%M%S)

# Find latest base backup before restore point
LATEST_BACKUP=$(find "$BACKUP_DIR" -name "pynomaly_db_*.sql.gz" \
    -newermt "$(date -d "$RESTORE_DATE - 24 hours" +%Y-%m-%d)" \
    ! -newermt "$RESTORE_DATE" | sort | tail -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "No suitable backup found for restore point $RESTORE_DATE"
    exit 1
fi

echo "Using backup: $LATEST_BACKUP"

# Restore base backup
mkdir -p /var/lib/postgresql/12/main
chown postgres:postgres /var/lib/postgresql/12/main

# Initialize new cluster
sudo -u postgres /usr/lib/postgresql/12/bin/initdb \
    -D /var/lib/postgresql/12/main \
    --locale=en_US.UTF-8

# Start PostgreSQL in recovery mode
systemctl start postgresql

# Create database and restore
sudo -u postgres createdb pynomaly_prod
gunzip -c "$LATEST_BACKUP" | sudo -u postgres psql pynomaly_prod

# Apply WAL files for point-in-time recovery
sudo -u postgres /usr/lib/postgresql/12/bin/pg_ctl \
    -D /var/lib/postgresql/12/main \
    stop

# Configure recovery
cat > /var/lib/postgresql/12/main/recovery.conf << EOF
restore_command = 'envdir /etc/wal-e.d/env wal-e wal-fetch %f %p'
recovery_target_time = '$RESTORE_DATE'
recovery_target_timeline = 'latest'
EOF

chown postgres:postgres /var/lib/postgresql/12/main/recovery.conf

# Start PostgreSQL for recovery
systemctl start postgresql

# Wait for recovery to complete
while [ -f /var/lib/postgresql/12/main/recovery.conf ]; do
    echo "Recovery in progress..."
    sleep 10
done

echo "Point-in-time recovery completed to $RESTORE_DATE"

# Start application
systemctl start pynomaly

# Verify application
curl -f http://localhost:8000/health
```

#### Full Database Restore

```bash
#!/bin/bash
# /opt/pynomaly/scripts/recovery/restore-database-full.sh

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "Starting full database restore from $BACKUP_FILE"

# Stop application
systemctl stop pynomaly

# Create restore database
sudo -u postgres createdb pynomaly_restore_$(date +%Y%m%d_%H%M%S)

# Restore backup
if [[ "$BACKUP_FILE" == *.gz ]]; then
    gunzip -c "$BACKUP_FILE" | sudo -u postgres psql pynomaly_restore_$(date +%Y%m%d_%H%M%S)
else
    sudo -u postgres psql pynomaly_restore_$(date +%Y%m%d_%H%M%S) < "$BACKUP_FILE"
fi

if [ $? -eq 0 ]; then
    echo "Database restore completed successfully"
    
    # Rename databases
    sudo -u postgres psql -c "ALTER DATABASE pynomaly_prod RENAME TO pynomaly_prod_backup_$(date +%Y%m%d_%H%M%S);"
    sudo -u postgres psql -c "ALTER DATABASE pynomaly_restore_$(date +%Y%m%d_%H%M%S) RENAME TO pynomaly_prod;"
    
    # Start application
    systemctl start pynomaly
    
    # Verify
    curl -f http://localhost:8000/health
    
    echo "Full database restore completed"
else
    echo "Database restore failed"
    exit 1
fi
```

### Application Data Recovery

```bash
#!/bin/bash
# /opt/pynomaly/scripts/recovery/restore-application-data.sh

RESTORE_DATE="$1"  # Format: YYYY-MM-DD
DATA_DIR="/opt/pynomaly/data"
BACKUP_DIR="/var/backups/pynomaly/application"

if [ -z "$RESTORE_DATE" ]; then
    echo "Usage: $0 YYYY-MM-DD"
    exit 1
fi

echo "Starting application data restore for $RESTORE_DATE"

# Stop application
systemctl stop pynomaly

# Backup current data
mv "$DATA_DIR" "$DATA_DIR.backup.$(date +%Y%m%d_%H%M%S)"

# Restore from S3
mkdir -p "$DATA_DIR"
aws s3 sync "s3://pynomaly-backups-prod/application/$RESTORE_DATE/" "$DATA_DIR/"

# Restore file permissions
chown -R pynomaly:pynomaly "$DATA_DIR"
chmod -R 755 "$DATA_DIR"

# Start application
systemctl start pynomaly

echo "Application data restore completed"
```

## 🚨 Disaster Recovery

### Complete System Recovery

```bash
#!/bin/bash
# /opt/pynomaly/scripts/recovery/disaster-recovery.sh

RECOVERY_DATE="$1"
S3_BUCKET="pynomaly-backups-prod"

echo "Starting complete disaster recovery for $RECOVERY_DATE"

# 1. Provision new infrastructure
terraform -chdir=/opt/pynomaly/infrastructure apply -auto-approve

# 2. Install application
ansible-playbook -i inventory/production site.yml

# 3. Restore database
LATEST_DB_BACKUP=$(aws s3 ls "s3://$S3_BUCKET/database/daily/" | 
    grep "$RECOVERY_DATE" | tail -1 | awk '{print $4}')

if [ -n "$LATEST_DB_BACKUP" ]; then
    aws s3 cp "s3://$S3_BUCKET/database/daily/$LATEST_DB_BACKUP" /tmp/
    ./restore-database-full.sh "/tmp/$LATEST_DB_BACKUP"
fi

# 4. Restore application data
aws s3 sync "s3://pynomaly-data-prod/" "/opt/pynomaly/data/"

# 5. Restore configurations
LATEST_CONFIG_BACKUP=$(aws s3 ls "s3://$S3_BUCKET/config/" | 
    grep "$RECOVERY_DATE" | tail -1 | awk '{print $4}')

if [ -n "$LATEST_CONFIG_BACKUP" ]; then
    aws s3 cp "s3://$S3_BUCKET/config/$LATEST_CONFIG_BACKUP" /tmp/
    gpg --decrypt --passphrase-file /opt/pynomaly/backup.key \
        "/tmp/$LATEST_CONFIG_BACKUP" | tar -xzf - -C /opt/pynomaly/
fi

# 6. Start services
systemctl enable pynomaly
systemctl start pynomaly

# 7. Verify recovery
./scripts/testing/smoke-test.sh

echo "Disaster recovery completed"
```

## 📊 Backup Monitoring

### Backup Health Dashboard

```bash
#!/bin/bash
# /opt/pynomaly/scripts/backup/backup-health-check.sh

echo "Pynomaly Backup Health Report - $(date)"
echo "=========================================="

# Check last backup dates
echo "Last Backup Dates:"
echo "- Database: $(ls -t /var/backups/pynomaly/database/pynomaly_db_*.sql.gz | head -1 | xargs stat -c %y)"
echo "- Redis: $(ls -t /var/backups/pynomaly/redis/redis_dump_*.rdb.gz | head -1 | xargs stat -c %y)"
echo "- Config: $(ls -t /var/backups/pynomaly/config/config_*.tar.gz.gpg | head -1 | xargs stat -c %y)"

# Check S3 backup status
echo ""
echo "S3 Backup Status:"
echo "- Database backups: $(aws s3 ls s3://pynomaly-backups-prod/database/daily/ | wc -l) files"
echo "- Config backups: $(aws s3 ls s3://pynomaly-backups-prod/config/ | wc -l) files"
echo "- Total size: $(aws s3 ls s3://pynomaly-backups-prod/ --recursive --summarize | grep "Total Size" | awk '{print $3 $4}')"

# Check backup integrity
echo ""
echo "Backup Integrity:"
LATEST_DB_BACKUP=$(ls -t /var/backups/pynomaly/database/pynomaly_db_*.sql.gz | head -1)
if gzip -t "$LATEST_DB_BACKUP" 2>/dev/null; then
    echo "✅ Latest database backup integrity: GOOD"
else
    echo "❌ Latest database backup integrity: FAILED"
fi

# Check disk space
echo ""
echo "Backup Storage:"
echo "- Local backup disk usage: $(df -h /var/backups | tail -1 | awk '{print $5}')"
echo "- Available space: $(df -h /var/backups | tail -1 | awk '{print $4}')"
```

## 📋 Backup Checklist

### Daily Backup Verification

- [ ] Database backup completed successfully
- [ ] Database backup uploaded to S3
- [ ] Application data synchronized
- [ ] Configuration backup encrypted and stored
- [ ] Backup integrity verified
- [ ] No backup alerts in monitoring
- [ ] Backup catalog updated

### Weekly Recovery Testing

- [ ] Test database restore to staging environment
- [ ] Verify application data restore
- [ ] Test configuration restore
- [ ] Validate backup decryption
- [ ] Document any issues found
- [ ] Update recovery procedures if needed

### Monthly Disaster Recovery Test

- [ ] Complete infrastructure recovery test
- [ ] Full system restore validation
- [ ] Network connectivity verification
- [ ] Application functionality testing
- [ ] Performance baseline verification
- [ ] Documentation updates

---

**Last Updated**: 2024-12-10  
**Next Review**: 2025-01-10  
**Document Owner**: DevOps Team
