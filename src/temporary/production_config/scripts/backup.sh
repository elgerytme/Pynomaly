#!/bin/bash
# Pynomaly Production Backup Script

set -e

# Configuration
BACKUP_DIR="/app/backups"
DB_NAME="pynomaly_prod"
DB_USER="pynomaly_user"
DATA_DIR="/app/data"
RETENTION_DAYS=7

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Database backup
echo "Starting database backup..."
pg_dump -h localhost -U "$DB_USER" -d "$DB_NAME" | gzip > "$BACKUP_DIR/db_$TIMESTAMP.sql.gz"

# Data backup
echo "Starting data backup..."
tar czf "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" -C "$DATA_DIR" .

# Upload to S3 (optional)
if [[ -n "$AWS_S3_BUCKET" ]]; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_DIR/db_$TIMESTAMP.sql.gz" "s3://$AWS_S3_BUCKET/backups/"
    aws s3 cp "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" "s3://$AWS_S3_BUCKET/backups/"
fi

# Cleanup old backups
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully!"
