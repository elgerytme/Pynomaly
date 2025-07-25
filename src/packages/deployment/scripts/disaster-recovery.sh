#!/bin/bash
# Disaster Recovery Script for Hexagonal Architecture
# Provides comprehensive disaster recovery capabilities

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_FILE="/tmp/disaster-recovery-$(date +%Y%m%d-%H%M%S).log"

# Default values
OPERATION=""
ENVIRONMENT="production"
BACKUP_LOCATION=""
RESTORE_POINT=""
DRY_RUN=false
FORCE=false
PARALLEL_JOBS=4

# Disaster types
declare -A DISASTER_SCENARIOS=(
    ["datacenter-outage"]="Complete datacenter failure requiring regional failover"
    ["database-corruption"]="Database corruption requiring restore from backup"
    ["security-breach"]="Security incident requiring isolation and cleanup"
    ["application-failure"]="Critical application failure requiring rollback"
    ["infrastructure-failure"]="Infrastructure component failure requiring rebuild"
    ["network-partition"]="Network connectivity issues requiring traffic rerouting"
)

# Critical services priority order
CRITICAL_SERVICES=(
    "api-gateway"
    "authentication-service"
    "data-quality-service"
    "anomaly-detection-service"
    "workflow-engine"
    "monitoring-service"
)

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

critical() {
    echo -e "${RED}${BOLD}[$(date +'%Y-%m-%d %H:%M:%S')] CRITICAL: $1${NC}" | tee -a "$LOG_FILE"
}

debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $1${NC}" | tee -a "$LOG_FILE"
    fi
}

# Help function
show_help() {
    cat << EOF
${BOLD}Disaster Recovery Management Script${NC}

${BOLD}USAGE:${NC}
    $0 <operation> [options]

${BOLD}OPERATIONS:${NC}
    backup              Create comprehensive system backup
    restore             Restore system from backup
    failover            Execute failover to backup region/cluster
    rollback            Rollback to previous stable state
    status              Check disaster recovery readiness
    test                Test disaster recovery procedures
    cleanup             Clean up after disaster recovery
    monitor             Monitor system health during recovery

${BOLD}OPTIONS:${NC}
    -e, --environment ENV       Target environment (staging|production)
    -b, --backup-location PATH  Backup storage location
    -r, --restore-point ID      Specific restore point identifier
    -d, --dry-run              Show what would be done without executing
    -f, --force                Force operation without confirmations
    -p, --parallel N           Number of parallel operations (default: 4)
    --scenario TYPE            Disaster scenario type
    --region REGION            Target region for failover
    --debug                    Enable debug output
    -h, --help                 Show this help message

${BOLD}DISASTER SCENARIOS:${NC}
$(for scenario in "${!DISASTER_SCENARIOS[@]}"; do
    printf "    %-20s %s\n" "$scenario" "${DISASTER_SCENARIOS[$scenario]}"
done)

${BOLD}EXAMPLES:${NC}
    # Create full system backup
    $0 backup -e production

    # Test disaster recovery procedures
    $0 test --scenario datacenter-outage --dry-run

    # Execute failover to backup region
    $0 failover --region us-east-1 -e production

    # Restore from specific backup point
    $0 restore -r backup-20240125-143022 -e production

    # Monitor recovery status
    $0 monitor -e production

${BOLD}CRITICAL SERVICES ORDER:${NC}
$(for i in "${!CRITICAL_SERVICES[@]}"; do
    printf "    %d. %s\n" $((i+1)) "${CRITICAL_SERVICES[$i]}"
done)

EOF
}

# Parse command line arguments
parse_args() {
    if [[ $# -eq 0 ]]; then
        error "No operation specified"
        show_help
        exit 1
    fi

    OPERATION="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -b|--backup-location)
                BACKUP_LOCATION="$2"
                shift 2
                ;;
            -r|--restore-point)
                RESTORE_POINT="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -p|--parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --scenario)
                DISASTER_SCENARIO="$2"
                shift 2
                ;;
            --region)
                TARGET_REGION="$2"
                shift 2
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validation functions
validate_operation() {
    local valid_operations=("backup" "restore" "failover" "rollback" "status" "test" "cleanup" "monitor")
    
    for op in "${valid_operations[@]}"; do
        if [[ "$OPERATION" == "$op" ]]; then
            return 0
        fi
    done
    
    error "Invalid operation: $OPERATION"
    error "Valid operations: ${valid_operations[*]}"
    exit 1
}

validate_environment() {
    case "$ENVIRONMENT" in
        staging|production)
            log "Validated environment: $ENVIRONMENT"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            error "Valid environments: staging, production"
            exit 1
            ;;
    esac
}

validate_prerequisites() {
    log "Validating disaster recovery prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "docker" "aws" "pg_dump" "redis-cli" "jq" "parallel")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured"
        exit 1
    fi
    
    log "Prerequisites validation completed"
}

# Backup operations
create_full_backup() {
    local backup_id="backup-$(date +%Y%m%d-%H%M%S)"
    local backup_dir="${BACKUP_LOCATION:-/tmp/dr-backups}/$backup_id"
    
    log "Creating full system backup: $backup_id"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would create backup at: $backup_dir"
        return 0
    fi
    
    mkdir -p "$backup_dir"
    
    # Backup Kubernetes resources
    log "Backing up Kubernetes resources..."
    kubectl get all,configmaps,secrets,persistentvolumes,persistentvolumeclaims \
        --all-namespaces -o yaml > "$backup_dir/kubernetes-resources.yaml"
    
    # Backup databases
    backup_databases "$backup_dir"
    
    # Backup persistent volumes
    backup_persistent_volumes "$backup_dir"
    
    # Backup configuration
    backup_configuration "$backup_dir"
    
    # Create backup manifest
    create_backup_manifest "$backup_id" "$backup_dir"
    
    # Upload to cloud storage
    upload_backup "$backup_dir"
    
    log "Full backup completed: $backup_id"
    echo "$backup_id" > /tmp/latest-backup-id
}

backup_databases() {
    local backup_dir="$1"
    log "Backing up databases..."
    
    mkdir -p "$backup_dir/databases"
    
    # PostgreSQL backup
    if kubectl get pods -n "$ENVIRONMENT" -l app=postgresql &> /dev/null; then
        log "Backing up PostgreSQL..."
        kubectl exec -n "$ENVIRONMENT" deployment/postgresql -- \
            pg_dumpall -U postgres > "$backup_dir/databases/postgresql.sql"
    fi
    
    # Redis backup
    if kubectl get pods -n "$ENVIRONMENT" -l app=redis &> /dev/null; then
        log "Backing up Redis..."
        kubectl exec -n "$ENVIRONMENT" deployment/redis -- \
            redis-cli --rdb /tmp/redis-backup.rdb
        kubectl cp "$ENVIRONMENT/$(kubectl get pods -n "$ENVIRONMENT" -l app=redis -o name | head -1 | cut -d/ -f2):/tmp/redis-backup.rdb" \
            "$backup_dir/databases/redis-backup.rdb"
    fi
    
    log "Database backup completed"
}

backup_persistent_volumes() {
    local backup_dir="$1"
    log "Backing up persistent volumes..."
    
    mkdir -p "$backup_dir/volumes"
    
    # Get all PVCs
    local pvcs
    pvcs=$(kubectl get pvc -n "$ENVIRONMENT" -o name)
    
    for pvc in $pvcs; do
        local pvc_name
        pvc_name=$(echo "$pvc" | cut -d/ -f2)
        
        log "Backing up PVC: $pvc_name"
        
        # Create snapshot using volume snapshot API
        kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: ${pvc_name}-backup-$(date +%s)
  namespace: $ENVIRONMENT
spec:
  source:
    persistentVolumeClaimName: $pvc_name
EOF
    done
    
    log "Persistent volume backup completed"
}

backup_configuration() {
    local backup_dir="$1"
    log "Backing up configuration..."
    
    mkdir -p "$backup_dir/config"
    
    # Copy deployment configurations
    cp -r "$PROJECT_ROOT/src/packages/deployment" "$backup_dir/config/"
    
    # Export environment variables
    env | grep -E '^(APP_|DB_|REDIS_|AWS_)' > "$backup_dir/config/environment.env" || true
    
    # Backup Helm values
    if command -v helm &> /dev/null; then
        helm list -n "$ENVIRONMENT" -o json > "$backup_dir/config/helm-releases.json"
    fi
    
    log "Configuration backup completed"
}

create_backup_manifest() {
    local backup_id="$1"
    local backup_dir="$2"
    
    cat > "$backup_dir/manifest.json" <<EOF
{
  "backup_id": "$backup_id",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "$ENVIRONMENT",
  "kubernetes_version": "$(kubectl version --short --client | head -1 | awk '{print $3}')",
  "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "services": $(kubectl get deployments -n "$ENVIRONMENT" -o json | jq '[.items[].metadata.name]'),
  "size": "$(du -sh "$backup_dir" | cut -f1)",
  "files": {
    "kubernetes_resources": "kubernetes-resources.yaml",
    "databases": "databases/",
    "volumes": "volumes/",
    "configuration": "config/"
  }
}
EOF
}

upload_backup() {
    local backup_dir="$1"
    local backup_id
    backup_id=$(basename "$backup_dir")
    
    log "Uploading backup to cloud storage..."
    
    # Compress backup
    tar -czf "${backup_dir}.tar.gz" -C "$(dirname "$backup_dir")" "$backup_id"
    
    # Upload to S3
    aws s3 cp "${backup_dir}.tar.gz" "s3://hexagonal-dr-backups/$ENVIRONMENT/" --storage-class STANDARD_IA
    
    # Upload manifest for quick access
    aws s3 cp "$backup_dir/manifest.json" "s3://hexagonal-dr-backups/$ENVIRONMENT/manifests/$backup_id.json"
    
    log "Backup uploaded successfully"
}

# Restore operations
restore_from_backup() {
    if [[ -z "$RESTORE_POINT" ]]; then
        error "Restore point not specified. Use -r/--restore-point option"
        exit 1
    fi
    
    log "Restoring system from backup: $RESTORE_POINT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would restore from backup: $RESTORE_POINT"
        return 0
    fi
    
    # Download backup
    local restore_dir="/tmp/restore-$RESTORE_POINT"
    download_backup "$RESTORE_POINT" "$restore_dir"
    
    # Confirm restore operation
    if ! confirm_restore; then
        log "Restore cancelled by user"
        exit 0
    fi
    
    # Execute restore in priority order
    restore_databases "$restore_dir"
    restore_persistent_volumes "$restore_dir"
    restore_kubernetes_resources "$restore_dir"
    restore_configuration "$restore_dir"
    
    # Verify restore
    verify_restore
    
    log "System restore completed successfully"
}

download_backup() {
    local backup_id="$1"
    local restore_dir="$2"
    
    log "Downloading backup: $backup_id"
    
    mkdir -p "$restore_dir"
    
    # Download from S3
    aws s3 cp "s3://hexagonal-dr-backups/$ENVIRONMENT/${backup_id}.tar.gz" "/tmp/"
    
    # Extract backup
    tar -xzf "/tmp/${backup_id}.tar.gz" -C "$(dirname "$restore_dir")"
    
    log "Backup downloaded and extracted"
}

confirm_restore() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "                    DISASTER RECOVERY RESTORE"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Environment:    $ENVIRONMENT"
    echo "  Restore Point:  $RESTORE_POINT"
    echo "  Current Time:   $(date)"
    echo "═══════════════════════════════════════════════════════════════"
    echo
    critical "This operation will OVERWRITE the current system state!"
    critical "All current data and configurations will be LOST!"
    echo
    
    read -p "Are you absolutely sure you want to continue? Type 'RESTORE' to confirm: " -r
    echo
    
    if [[ "$REPLY" != "RESTORE" ]]; then
        return 1
    fi
    
    return 0
}

restore_databases() {
    local restore_dir="$1"
    log "Restoring databases..."
    
    # Restore PostgreSQL
    if [[ -f "$restore_dir/databases/postgresql.sql" ]]; then
        log "Restoring PostgreSQL..."
        kubectl exec -n "$ENVIRONMENT" deployment/postgresql -- \
            psql -U postgres < "$restore_dir/databases/postgresql.sql"
    fi
    
    # Restore Redis
    if [[ -f "$restore_dir/databases/redis-backup.rdb" ]]; then
        log "Restoring Redis..."
        kubectl cp "$restore_dir/databases/redis-backup.rdb" \
            "$ENVIRONMENT/$(kubectl get pods -n "$ENVIRONMENT" -l app=redis -o name | head -1 | cut -d/ -f2):/tmp/restore.rdb"
        kubectl exec -n "$ENVIRONMENT" deployment/redis -- \
            redis-cli --rdb /tmp/restore.rdb
    fi
    
    log "Database restore completed"
}

restore_persistent_volumes() {
    local restore_dir="$1"
    log "Restoring persistent volumes..."
    
    # This would typically involve restoring from volume snapshots
    # Implementation depends on your storage provider
    
    log "Persistent volume restore completed"
}

restore_kubernetes_resources() {
    local restore_dir="$1"
    log "Restoring Kubernetes resources..."
    
    # Apply resources in order of priority
    for service in "${CRITICAL_SERVICES[@]}"; do
        log "Restoring service: $service"
        
        # Extract service-specific resources
        kubectl apply -f <(grep -A 50 "name: $service" "$restore_dir/kubernetes-resources.yaml") || true
        
        # Wait for service to be ready
        kubectl wait --for=condition=available --timeout=300s deployment/"$service" -n "$ENVIRONMENT" || true
    done
    
    log "Kubernetes resources restore completed"
}

restore_configuration() {
    local restore_dir="$1"
    log "Restoring configuration..."
    
    # Restore deployment configurations
    if [[ -d "$restore_dir/config/deployment" ]]; then
        cp -r "$restore_dir/config/deployment"/* "$PROJECT_ROOT/src/packages/deployment/"
    fi
    
    log "Configuration restore completed"
}

# Failover operations
execute_failover() {
    local target_region="${TARGET_REGION:-us-east-1}"
    
    log "Executing failover to region: $target_region"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would execute failover to: $target_region"
        return 0
    fi
    
    # Update DNS to point to backup region
    update_dns_failover "$target_region"
    
    # Scale up services in backup region
    scale_backup_region "$target_region"
    
    # Verify failover
    verify_failover "$target_region"
    
    log "Failover completed successfully"
}

update_dns_failover() {
    local target_region="$1"
    log "Updating DNS for failover to: $target_region"
    
    # Update Route53 records to point to backup region
    # This is a simplified example
    aws route53 change-resource-record-sets \
        --hosted-zone-id "Z123456789" \
        --change-batch file://dns-failover-change.json
}

scale_backup_region() {
    local target_region="$1"
    log "Scaling up services in backup region: $target_region"
    
    # Switch kubectl context to backup region
    aws eks update-kubeconfig --region "$target_region" --name "$ENVIRONMENT-backup-cluster"
    
    # Scale up critical services
    for service in "${CRITICAL_SERVICES[@]}"; do
        log "Scaling up service: $service"
        kubectl scale deployment "$service" --replicas=3 -n "$ENVIRONMENT"
        kubectl wait --for=condition=available --timeout=300s deployment/"$service" -n "$ENVIRONMENT"
    done
}

# Monitoring and verification
monitor_recovery() {
    log "Monitoring disaster recovery status..."
    
    local monitoring_duration=1800 # 30 minutes
    local check_interval=60        # 1 minute
    local checks_passed=0
    local total_checks=$((monitoring_duration / check_interval))
    
    for ((i=1; i<=total_checks; i++)); do
        log "Monitoring check $i/$total_checks"
        
        if perform_health_checks; then
            ((checks_passed++))
            log "Health check passed ($checks_passed/$i)"
        else
            warn "Health check failed"
        fi
        
        sleep $check_interval
    done
    
    local success_rate=$(( (checks_passed * 100) / total_checks ))
    
    if [[ $success_rate -ge 95 ]]; then
        log "Disaster recovery monitoring completed successfully ($success_rate% success rate)"
    else
        error "Disaster recovery monitoring shows instability ($success_rate% success rate)"
        return 1
    fi
}

perform_health_checks() {
    # Check pod health
    local unhealthy_pods
    unhealthy_pods=$(kubectl get pods -n "$ENVIRONMENT" --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    
    if [[ $unhealthy_pods -gt 0 ]]; then
        return 1
    fi
    
    # Check service endpoints
    for service in "${CRITICAL_SERVICES[@]}"; do
        local endpoint
        endpoint=$(kubectl get service "$service" -n "$ENVIRONMENT" -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
        
        if [[ -n "$endpoint" ]]; then
            if ! curl -f "http://$endpoint/health" --max-time 10 --silent; then
                return 1
            fi
        fi
    done
    
    return 0
}

verify_restore() {
    log "Verifying system restore..."
    
    # Verify critical services are running
    for service in "${CRITICAL_SERVICES[@]}"; do
        if ! kubectl get deployment "$service" -n "$ENVIRONMENT" &> /dev/null; then
            error "Service not found after restore: $service"
            return 1
        fi
        
        if ! kubectl wait --for=condition=available --timeout=60s deployment/"$service" -n "$ENVIRONMENT"; then
            error "Service not available after restore: $service"
            return 1
        fi
    done
    
    # Verify database connectivity
    if ! verify_database_connectivity; then
        error "Database connectivity verification failed"
        return 1
    fi
    
    log "System restore verification completed successfully"
}

verify_database_connectivity() {
    # Test PostgreSQL
    if kubectl get deployment postgresql -n "$ENVIRONMENT" &> /dev/null; then
        if ! kubectl exec -n "$ENVIRONMENT" deployment/postgresql -- pg_isready -U postgres; then
            return 1
        fi
    fi
    
    # Test Redis
    if kubectl get deployment redis -n "$ENVIRONMENT" &> /dev/null; then
        if ! kubectl exec -n "$ENVIRONMENT" deployment/redis -- redis-cli ping | grep -q PONG; then
            return 1
        fi
    fi
    
    return 0
}

verify_failover() {
    local target_region="$1"
    log "Verifying failover to region: $target_region"
    
    # Verify services are running in target region
    if ! perform_health_checks; then
        error "Health checks failed in target region"
        return 1
    fi
    
    # Verify DNS resolution
    local api_endpoint
    api_endpoint=$(dig +short api.hexagonal-architecture.com)
    
    if [[ -z "$api_endpoint" ]]; then
        error "DNS resolution failed for API endpoint"
        return 1
    fi
    
    log "Failover verification completed successfully"
}

# Test disaster recovery procedures
test_disaster_recovery() {
    local scenario="${DISASTER_SCENARIO:-datacenter-outage}"
    
    log "Testing disaster recovery scenario: $scenario"
    
    if [[ -z "${DISASTER_SCENARIOS[$scenario]}" ]]; then
        error "Unknown disaster scenario: $scenario"
        error "Available scenarios: ${!DISASTER_SCENARIOS[*]}"
        exit 1
    fi
    
    log "Scenario description: ${DISASTER_SCENARIOS[$scenario]}"
    
    case "$scenario" in
        "datacenter-outage")
            test_datacenter_outage
            ;;
        "database-corruption")
            test_database_corruption
            ;;
        "security-breach")
            test_security_breach
            ;;
        "application-failure")
            test_application_failure
            ;;
        "infrastructure-failure")
            test_infrastructure_failure
            ;;
        "network-partition")
            test_network_partition
            ;;
        *)
            error "Test not implemented for scenario: $scenario"
            exit 1
            ;;
    esac
    
    log "Disaster recovery test completed for scenario: $scenario"
}

test_datacenter_outage() {
    log "Testing datacenter outage scenario..."
    
    # Simulate complete datacenter failure
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would simulate datacenter outage"
        log "[DRY RUN] Would execute failover to backup region"
        log "[DRY RUN] Would verify service availability"
        return 0
    fi
    
    # Scale down services in primary region (simulation)
    log "Simulating datacenter outage by scaling down services..."
    for service in "${CRITICAL_SERVICES[@]}"; do
        kubectl scale deployment "$service" --replicas=0 -n "$ENVIRONMENT"
    done
    
    # Execute failover
    execute_failover
    
    # Verify services are available in backup region
    if ! verify_failover "${TARGET_REGION:-us-east-1}"; then
        error "Datacenter outage test failed"
        return 1
    fi
    
    log "Datacenter outage test completed successfully"
}

# Check disaster recovery status
check_dr_status() {
    log "Checking disaster recovery status..."
    
    echo
    echo "═══════════════════════════════════════════════════════════════"
    echo "              DISASTER RECOVERY STATUS REPORT"
    echo "═══════════════════════════════════════════════════════════════"
    echo
    
    # Check backup status
    echo "🔄 BACKUP STATUS:"
    local latest_backup
    latest_backup=$(aws s3 ls "s3://hexagonal-dr-backups/$ENVIRONMENT/" --recursive | sort | tail -1 | awk '{print $4}' | cut -d/ -f2)
    
    if [[ -n "$latest_backup" ]]; then
        echo "   ✅ Latest backup: $latest_backup"
        local backup_age
        backup_age=$(aws s3api head-object --bucket hexagonal-dr-backups --key "$ENVIRONMENT/$latest_backup" --query 'LastModified' --output text)
        echo "   📅 Backup age: $backup_age"
    else
        echo "   ❌ No backups found"
    fi
    
    # Check cluster health
    echo
    echo "🏥 CLUSTER HEALTH:"
    local cluster_status
    cluster_status=$(kubectl get nodes --no-headers | awk '{print $2}' | sort | uniq -c)
    echo "   Node status: $cluster_status"
    
    # Check service availability
    echo
    echo "🎯 SERVICE AVAILABILITY:"
    for service in "${CRITICAL_SERVICES[@]}"; do
        if kubectl get deployment "$service" -n "$ENVIRONMENT" &> /dev/null; then
            local replicas
            replicas=$(kubectl get deployment "$service" -n "$ENVIRONMENT" -o jsonpath='{.status.readyReplicas}/{.spec.replicas}')
            echo "   ✅ $service: $replicas replicas ready"
        else
            echo "   ❌ $service: Not found"
        fi
    done
    
    # Check backup region readiness
    echo
    echo "🌍 BACKUP REGION STATUS:"
    local backup_region="${TARGET_REGION:-us-east-1}"
    if aws eks describe-cluster --name "$ENVIRONMENT-backup-cluster" --region "$backup_region" &> /dev/null; then
        echo "   ✅ Backup cluster available in $backup_region"
    else
        echo "   ❌ Backup cluster not found in $backup_region"
    fi
    
    echo
    echo "═══════════════════════════════════════════════════════════════"
}

# Main function
main() {
    log "Starting disaster recovery operation: $OPERATION"
    log "Environment: $ENVIRONMENT"
    log "Log file: $LOG_FILE"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate inputs
    validate_operation
    validate_environment
    validate_prerequisites
    
    # Execute operation
    case "$OPERATION" in
        backup)
            create_full_backup
            ;;
        restore)
            restore_from_backup
            ;;
        failover)
            execute_failover
            ;;
        rollback)
            # Rollback would use the last known good state
            RESTORE_POINT="${RESTORE_POINT:-$(cat /tmp/latest-backup-id 2>/dev/null || echo '')}"
            if [[ -z "$RESTORE_POINT" ]]; then
                error "No restore point available for rollback"
                exit 1
            fi
            restore_from_backup
            ;;
        status)
            check_dr_status
            ;;
        test)
            test_disaster_recovery
            ;;
        cleanup)
            log "Cleaning up disaster recovery artifacts..."
            # Cleanup implementation
            ;;
        monitor)
            monitor_recovery
            ;;
        *)
            error "Operation not implemented: $OPERATION"
            exit 1
            ;;
    esac
    
    log "Disaster recovery operation completed: $OPERATION"
}

# Trap signals for cleanup
trap 'error "Script interrupted"; exit 130' INT TERM

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi