#!/bin/bash

# PostgreSQL Restore Script for Pynomaly
# This script restores encrypted database backups from S3

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-pynomaly}"
DB_HOST="${DB_HOST:-pynomaly-postgresql}"
DB_NAME="${DB_NAME:-pynomaly}"
DB_USER="${DB_USER:-pynomaly}"
S3_BUCKET="${S3_BUCKET:-pynomaly-backups}"
S3_PREFIX="${S3_PREFIX:-postgresql}"
ENCRYPTION_KEY="${ENCRYPTION_KEY:-}"
BACKUP_FILE="${BACKUP_FILE:-}"
RESTORE_MODE="${RESTORE_MODE:-full}" # full, data-only, schema-only

# Derived variables
LOCAL_RESTORE_DIR="/tmp/restore"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Error handling
cleanup() {
    local exit_code=$?
    if [[ -d "$LOCAL_RESTORE_DIR" ]]; then
        rm -rf "$LOCAL_RESTORE_DIR"
    fi
    exit $exit_code
}

trap cleanup EXIT ERR

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Restore PostgreSQL database from encrypted S3 backup

OPTIONS:
    -f, --file FILENAME     Specific backup file to restore (default: latest)
    -m, --mode MODE         Restore mode: full, data-only, schema-only (default: full)
    -n, --namespace NS      Kubernetes namespace (default: pynomaly)
    -d, --database NAME     Database name (default: pynomaly)
    -u, --user USER         Database user (default: pynomaly)
    -b, --bucket BUCKET     S3 bucket name (default: pynomaly-backups)
    -p, --prefix PREFIX     S3 prefix (default: postgresql)
    -h, --help              Show this help message
    --dry-run               Show what would be done without executing
    --list-backups          List available backups

EXAMPLES:
    # Restore latest backup
    $0

    # Restore specific backup
    $0 -f pynomaly_backup_20240101_120000.sql.gpg

    # Data-only restore (useful for upgrading)
    $0 -m data-only

    # List available backups
    $0 --list-backups

ENVIRONMENT VARIABLES:
    ENCRYPTION_KEY          Required: GPG passphrase for decryption
    SLACK_WEBHOOK_URL       Optional: Slack notification webhook
    
EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--file)
                BACKUP_FILE="$2"
                shift 2
                ;;
            -m|--mode)
                RESTORE_MODE="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--database)
                DB_NAME="$2"
                shift 2
                ;;
            -u|--user)
                DB_USER="$2"
                shift 2
                ;;
            -b|--bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            -p|--prefix)
                S3_PREFIX="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --list-backups)
                list_backups
                exit 0
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log "ERROR: Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# List available backups
list_backups() {
    log "Available backups in s3://${S3_BUCKET}/${S3_PREFIX}/:"
    echo
    echo "Filename                                    Size      Last Modified"
    echo "------------------------------------------ --------- -------------------"
    
    aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" --human-readable | \
        grep "\.sql\.gpg$" | \
        sort -k1,2 -r | \
        while read -r date time size file; do
            printf "%-42s %-9s %s %s\n" "$file" "$size" "$date" "$time"
        done
}

# Validate requirements
validate_requirements() {
    log "Validating requirements..."
    
    # Check if running in Kubernetes
    if [[ ! -f /var/run/secrets/kubernetes.io/serviceaccount/token ]]; then
        log "ERROR: This script must run in a Kubernetes pod"
        exit 1
    fi
    
    # Check required tools
    for tool in kubectl pg_restore aws gpg; do
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
    
    # Validate restore mode
    if [[ ! "$RESTORE_MODE" =~ ^(full|data-only|schema-only)$ ]]; then
        log "ERROR: Invalid restore mode: $RESTORE_MODE"
        exit 1
    fi
    
    log "Requirements validation passed"
}

# Get latest backup if not specified
get_backup_file() {
    if [[ -z "$BACKUP_FILE" ]]; then
        log "Finding latest backup..."
        
        BACKUP_FILE=$(aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" | \
            grep "\.sql\.gpg$" | \
            sort -k1,2 -r | \
            head -n1 | \
            awk '{print $4}')
        
        if [[ -z "$BACKUP_FILE" ]]; then
            log "ERROR: No backup files found in s3://${S3_BUCKET}/${S3_PREFIX}/"
            exit 1
        fi
        
        log "Latest backup found: $BACKUP_FILE"
    fi
}

# Download and decrypt backup
download_backup() {
    log "Downloading and decrypting backup: $BACKUP_FILE"
    
    mkdir -p "$LOCAL_RESTORE_DIR"
    
    local encrypted_file="${LOCAL_RESTORE_DIR}/${BACKUP_FILE}"
    local decrypted_file="${LOCAL_RESTORE_DIR}/restore_${TIMESTAMP}.sql"
    
    # Download backup
    aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_FILE}" "$encrypted_file"
    
    # Decrypt backup
    echo "$ENCRYPTION_KEY" | gpg --batch --yes --passphrase-fd 0 \
        --decrypt "$encrypted_file" > "$decrypted_file"
    
    # Verify decrypted file
    if [[ ! -f "$decrypted_file" ]] || [[ ! -s "$decrypted_file" ]]; then
        log "ERROR: Failed to decrypt backup file"
        exit 1
    fi
    
    # Check if it's a valid PostgreSQL dump
    if ! file "$decrypted_file" | grep -q "PostgreSQL custom database dump"; then
        log "ERROR: Decrypted file is not a valid PostgreSQL dump"
        exit 1
    fi
    
    DECRYPTED_BACKUP="$decrypted_file"
    local backup_size=$(du -h "$DECRYPTED_BACKUP" | cut -f1)
    log "Backup decrypted successfully: ${backup_size}"
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

# Create database backup before restore
backup_current_database() {
    log "Creating backup of current database before restore..."
    
    local pre_restore_backup="pre_restore_backup_${TIMESTAMP}.sql"
    
    kubectl exec -n "$NAMESPACE" deployment/pynomaly-api -- \
        pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
        --format=custom \
        --compress=9 > "${LOCAL_RESTORE_DIR}/${pre_restore_backup}"
    
    log "Current database backed up to: $pre_restore_backup"
}

# Stop application services
stop_services() {
    log "Scaling down application services..."
    
    kubectl scale deployment pynomaly-api --replicas=0 -n "$NAMESPACE"
    kubectl scale deployment pynomaly-worker --replicas=0 -n "$NAMESPACE"
    
    # Wait for pods to terminate
    kubectl wait --for=delete pod -l app.kubernetes.io/name=pynomaly -n "$NAMESPACE" --timeout=300s
    
    log "Application services stopped"
}

# Start application services
start_services() {
    log "Scaling up application services..."
    
    kubectl scale deployment pynomaly-api --replicas=3 -n "$NAMESPACE"
    kubectl scale deployment pynomaly-worker --replicas=2 -n "$NAMESPACE"
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=pynomaly -n "$NAMESPACE" --timeout=600s
    
    log "Application services started"
}

# Restore database
restore_database() {
    log "Restoring database from backup..."
    log "Restore mode: $RESTORE_MODE"
    
    local restore_args=()
    
    case "$RESTORE_MODE" in
        "data-only")
            restore_args+=(--data-only)
            ;;
        "schema-only")
            restore_args+=(--schema-only)
            ;;
        "full")
            # Drop and recreate database for full restore
            kubectl exec -n "$NAMESPACE" deployment/pynomaly-postgresql-0 -- \
                psql -U postgres -c "DROP DATABASE IF EXISTS ${DB_NAME};"
            kubectl exec -n "$NAMESPACE" deployment/pynomaly-postgresql-0 -- \
                psql -U postgres -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"
            ;;
    esac
    
    # Perform restore
    kubectl exec -i -n "$NAMESPACE" deployment/pynomaly-postgresql-0 -- \
        pg_restore -h localhost -U "$DB_USER" -d "$DB_NAME" \
        --verbose \
        --clean \
        --if-exists \
        --no-owner \
        --no-privileges \
        "${restore_args[@]}" < "$DECRYPTED_BACKUP"
    
    log "Database restore completed"
}

# Verify restore
verify_restore() {
    log "Verifying database restore..."
    
    # Test basic connectivity
    kubectl exec -n "$NAMESPACE" deployment/pynomaly-api -- \
        psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();"
    
    # Check table count
    local table_count
    table_count=$(kubectl exec -n "$NAMESPACE" deployment/pynomaly-api -- \
        psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
    
    log "Database verification completed. Tables found: $table_count"
    
    if [[ $table_count -eq 0 ]]; then
        log "WARNING: No tables found in restored database"
        return 1
    fi
    
    return 0
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    log "Sending notification: $status - $message"
    
    # Send to Slack if webhook URL is configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Database Restore $status: $message\"}" \
            "$SLACK_WEBHOOK_URL" || true
    fi
}

# Main execution
main() {
    log "Starting PostgreSQL restore process..."
    log "Namespace: $NAMESPACE"
    log "Database: $DB_NAME"
    log "Restore mode: $RESTORE_MODE"
    log "S3 Bucket: $S3_BUCKET"
    
    validate_requirements
    get_backup_file
    
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log "DRY RUN: Would restore backup: $BACKUP_FILE"
        log "DRY RUN: Would use restore mode: $RESTORE_MODE"
        exit 0
    fi
    
    # Confirmation prompt
    echo
    echo "⚠️  WARNING: This will restore the database from backup!"
    echo "Backup file: $BACKUP_FILE"
    echo "Restore mode: $RESTORE_MODE"
    echo "Database: $DB_NAME"
    echo
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Restore cancelled by user"
        exit 1
    fi
    
    test_database_connection
    download_backup
    backup_current_database
    stop_services
    restore_database
    
    if verify_restore; then
        start_services
        local restore_info="Database restored successfully from: $BACKUP_FILE"
        log "$restore_info"
        send_notification "SUCCESS" "$restore_info"
    else
        log "ERROR: Database verification failed"
        send_notification "FAILED" "Database restore verification failed"
        exit 1
    fi
    
    log "Restore process completed successfully"
}

# Handle script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_arguments "$@"
    main
fi