#!/bin/bash

# PostgreSQL Backup Script for Pynomaly
# This script creates encrypted backups of the PostgreSQL database and uploads them to S3

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-pynomaly}"
DB_HOST="${DB_HOST:-pynomaly-postgresql}"
DB_NAME="${DB_NAME:-pynomaly}"
DB_USER="${DB_USER:-pynomaly}"
S3_BUCKET="${S3_BUCKET:-pynomaly-backups}"
S3_PREFIX="${S3_PREFIX:-postgresql}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
ENCRYPTION_KEY="${ENCRYPTION_KEY:-}"

# Derived variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="pynomaly_backup_${TIMESTAMP}.sql"
ENCRYPTED_FILE="${BACKUP_FILE}.gpg"
LOCAL_BACKUP_DIR="/tmp/backups"
S3_KEY="${S3_PREFIX}/${ENCRYPTED_FILE}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Error handling
cleanup() {
    local exit_code=$?
    if [[ -f "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}" ]]; then
        rm -f "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}"
    fi
    if [[ -f "${LOCAL_BACKUP_DIR}/${ENCRYPTED_FILE}" ]]; then
        rm -f "${LOCAL_BACKUP_DIR}/${ENCRYPTED_FILE}"
    fi
    exit $exit_code
}

trap cleanup EXIT ERR

# Validate requirements
validate_requirements() {
    log "Validating requirements..."
    
    # Check if running in Kubernetes
    if [[ ! -f /var/run/secrets/kubernetes.io/serviceaccount/token ]]; then
        log "ERROR: This script must run in a Kubernetes pod"
        exit 1
    fi
    
    # Check required tools
    for tool in kubectl pg_dump aws gpg; do
        if ! command -v "$tool" &> /dev/null; then
            log "ERROR: Required tool '$tool' not found"
            exit 1
        fi
    done
    
    # Check environment variables
    if [[ -z "$ENCRYPTION_KEY" ]]; then
        log "ERROR: ENCRYPTION_KEY environment variable is required"
        exit 1
    fi
    
    log "Requirements validation passed"
}

# Test database connectivity
test_database_connection() {
    log "Testing database connection..."
    
    if ! kubectl exec -n "$NAMESPACE" deployment/pynomaly-api -- \
        pg_isready -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -q; then
        log "ERROR: Cannot connect to database"
        exit 1
    fi
    
    log "Database connection successful"
}

# Create database backup
create_backup() {
    log "Creating database backup..."
    
    mkdir -p "$LOCAL_BACKUP_DIR"
    
    # Create backup with compression
    kubectl exec -n "$NAMESPACE" deployment/pynomaly-api -- \
        pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
        --verbose \
        --format=custom \
        --compress=9 \
        --no-owner \
        --no-privileges \
        --exclude-table-data='audit_logs' \
        --exclude-table-data='session_data' > "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}"
    
    # Verify backup file was created and has content
    if [[ ! -f "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}" ]] || [[ ! -s "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}" ]]; then
        log "ERROR: Backup file was not created or is empty"
        exit 1
    fi
    
    local backup_size=$(du -h "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
    log "Backup created successfully: ${BACKUP_FILE} (${backup_size})"
}

# Encrypt backup
encrypt_backup() {
    log "Encrypting backup..."
    
    # Create GPG key from environment variable
    echo "$ENCRYPTION_KEY" | gpg --batch --yes --passphrase-fd 0 \
        --symmetric --cipher-algo AES256 \
        --output "${LOCAL_BACKUP_DIR}/${ENCRYPTED_FILE}" \
        "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}"
    
    # Verify encrypted file was created
    if [[ ! -f "${LOCAL_BACKUP_DIR}/${ENCRYPTED_FILE}" ]]; then
        log "ERROR: Encrypted backup file was not created"
        exit 1
    fi
    
    # Remove unencrypted backup
    rm -f "${LOCAL_BACKUP_DIR}/${BACKUP_FILE}"
    
    local encrypted_size=$(du -h "${LOCAL_BACKUP_DIR}/${ENCRYPTED_FILE}" | cut -f1)
    log "Backup encrypted successfully: ${ENCRYPTED_FILE} (${encrypted_size})"
}

# Upload to S3
upload_to_s3() {
    log "Uploading backup to S3..."
    
    # Upload with metadata
    aws s3 cp "${LOCAL_BACKUP_DIR}/${ENCRYPTED_FILE}" "s3://${S3_BUCKET}/${S3_KEY}" \
        --metadata "timestamp=${TIMESTAMP},database=${DB_NAME},namespace=${NAMESPACE}" \
        --storage-class STANDARD_IA
    
    # Verify upload
    if ! aws s3 ls "s3://${S3_BUCKET}/${S3_KEY}" &> /dev/null; then
        log "ERROR: Failed to upload backup to S3"
        exit 1
    fi
    
    log "Backup uploaded successfully to s3://${S3_BUCKET}/${S3_KEY}"
}

# Clean up old backups
cleanup_old_backups() {
    log "Cleaning up old backups (older than ${RETENTION_DAYS} days)..."
    
    # Calculate cutoff date
    local cutoff_date
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        cutoff_date=$(date -j -v-${RETENTION_DAYS}d '+%Y-%m-%d')
    else
        # Linux
        cutoff_date=$(date -d "${RETENTION_DAYS} days ago" '+%Y-%m-%d')
    fi
    
    # List and delete old backups
    aws s3api list-objects-v2 --bucket "$S3_BUCKET" --prefix "$S3_PREFIX/" \
        --query "Contents[?LastModified<='${cutoff_date}'].Key" --output text | \
    while read -r key; do
        if [[ -n "$key" && "$key" != "None" ]]; then
            log "Deleting old backup: $key"
            aws s3 rm "s3://${S3_BUCKET}/${key}"
        fi
    done
    
    log "Old backup cleanup completed"
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    log "Sending notification: $status - $message"
    
    # Send to Slack if webhook URL is configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Database Backup $status: $message\"}" \
            "$SLACK_WEBHOOK_URL" || true
    fi
    
    # Send to CloudWatch if configured
    if command -v aws &> /dev/null; then
        aws cloudwatch put-metric-data \
            --namespace "Pynomaly/Backup" \
            --metric-data "MetricName=BackupStatus,Value=$([ "$status" = "SUCCESS" ] && echo 1 || echo 0),Unit=Count" || true
    fi
}

# Validate backup integrity
validate_backup() {
    log "Validating backup integrity..."
    
    # Download and decrypt backup for validation
    local temp_validation_file="/tmp/validation_${TIMESTAMP}.sql"
    
    aws s3 cp "s3://${S3_BUCKET}/${S3_KEY}" - | \
        gpg --batch --yes --passphrase-fd 0 --decrypt > "$temp_validation_file" <<< "$ENCRYPTION_KEY"
    
    # Check if decrypted file is a valid PostgreSQL dump
    if ! file "$temp_validation_file" | grep -q "PostgreSQL custom database dump"; then
        log "ERROR: Backup validation failed - not a valid PostgreSQL dump"
        rm -f "$temp_validation_file"
        exit 1
    fi
    
    # Clean up validation file
    rm -f "$temp_validation_file"
    
    log "Backup validation successful"
}

# Main execution
main() {
    log "Starting PostgreSQL backup process..."
    log "Namespace: $NAMESPACE"
    log "Database: $DB_NAME"
    log "S3 Bucket: $S3_BUCKET"
    log "Retention: $RETENTION_DAYS days"
    
    validate_requirements
    test_database_connection
    create_backup
    encrypt_backup
    upload_to_s3
    validate_backup
    cleanup_old_backups
    
    local backup_info="Backup completed: s3://${S3_BUCKET}/${S3_KEY}"
    log "$backup_info"
    send_notification "SUCCESS" "$backup_info"
    
    log "Backup process completed successfully"
}

# Handle script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi