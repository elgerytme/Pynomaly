#!/bin/bash

# Pynomaly Isolation Manager
# Main script for creating and managing isolated development environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/.project-rules/isolation-config.yaml"
ISOLATION_LOG="$PROJECT_ROOT/.isolation.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$ISOLATION_LOG"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
    log "INFO: $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    log "SUCCESS: $1"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    log "WARNING: $1"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    log "ERROR: $1"
}

# Check if isolation is enabled
check_isolation_enabled() {
    if ! grep -q "enabled: true" "$CONFIG_FILE" 2>/dev/null; then
        warn "Isolation is currently disabled in configuration"
        echo "To enable isolation, set 'enabled: true' in $CONFIG_FILE"
        exit 1
    fi
}

# Generate unique isolation ID
generate_isolation_id() {
    local branch_name
    branch_name=$(git branch --show-current 2>/dev/null || echo "unknown")
    local timestamp
    timestamp=$(date '+%Y%m%d-%H%M%S')
    local user
    user=$(whoami)
    echo "${user}-${branch_name}-${timestamp}" | sed 's/[^a-zA-Z0-9-]/-/g' | tr '[:upper:]' '[:lower:]'
}

# Check prerequisites
check_prerequisites() {
    local missing_deps=()

    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
    fi

    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        echo "Please install the missing dependencies and try again."
        exit 1
    fi
}

# Create container-based isolation
create_container_isolation() {
    local isolation_id="$1"
    local profile="${2:-development}"

    info "Creating container-based isolation: $isolation_id"

    # Create isolation directory
    local isolation_dir="$PROJECT_ROOT/.isolated-work/container-$isolation_id"
    mkdir -p "$isolation_dir"

    # Copy docker-compose template
    cp "$PROJECT_ROOT/.project-rules/templates/docker-compose.isolation.yml" "$isolation_dir/docker-compose.yml"

    # Create environment file
    cat > "$isolation_dir/.env" <<EOF
ISOLATION_ID=$isolation_id
PYTHON_VERSION=3.11
LOG_LEVEL=DEBUG
HOST_API_PORT=$((8000 + $(shuf -i 1-999 -n 1)))
HOST_DB_PORT=$((5432 + $(shuf -i 1-999 -n 1)))
HOST_REDIS_PORT=$((6379 + $(shuf -i 1-999 -n 1)))
HOST_MONITOR_PORT=$((9100 + $(shuf -i 1-999 -n 1)))
HOST_JUPYTER_PORT=$((8888 + $(shuf -i 1-999 -n 1)))
HOST_DOCS_PORT=$((8080 + $(shuf -i 1-999 -n 1)))
EOF

    # Build and start the isolation environment
    cd "$isolation_dir"

    info "Building isolation container..."
    docker-compose build pynomaly-isolated

    info "Starting isolation environment..."
    if [ "$profile" = "testing" ]; then
        docker-compose --profile testing up -d
    elif [ "$profile" = "experimentation" ]; then
        docker-compose --profile experimentation up -d
    elif [ "$profile" = "documentation" ]; then
        docker-compose --profile documentation up -d
    else
        docker-compose up -d
    fi

    # Wait for services to be ready
    info "Waiting for services to be ready..."
    sleep 10

    # Check if services are healthy
    if docker-compose ps | grep -q "Up (healthy)"; then
        success "Container isolation created successfully: $isolation_id"
        info "Access your isolated environment:"
        info "  - Shell: docker-compose exec pynomaly-isolated /bin/bash"
        info "  - API: http://localhost:$(grep HOST_API_PORT .env | cut -d= -f2)"
        info "  - Database: localhost:$(grep HOST_DB_PORT .env | cut -d= -f2)"

        # Save isolation metadata
        save_isolation_metadata "$isolation_id" "container" "$isolation_dir"
    else
        error "Failed to start isolation environment"
        docker-compose logs
        exit 1
    fi
}

# Create virtual environment isolation
create_venv_isolation() {
    local isolation_id="$1"

    info "Creating virtual environment isolation: $isolation_id"

    # Create isolation directory
    local isolation_dir="$PROJECT_ROOT/.isolated-work/venv-$isolation_id"
    mkdir -p "$isolation_dir"

    # Copy project files (exclude .git and large directories)
    rsync -av --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
          --exclude='node_modules' --exclude='*.pyc' \
          "$PROJECT_ROOT/" "$isolation_dir/"

    # Create virtual environment
    cd "$isolation_dir"
    python3 -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    pip install --upgrade pip
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi
    if [ -f "pyproject.toml" ]; then
        pip install -e .
    fi

    # Create activation script
    cat > "$isolation_dir/activate-isolation.sh" <<EOF
#!/bin/bash
cd "$isolation_dir"
source .venv/bin/activate
export PYNOMALY_ENV=isolated
export PYTHONPATH="$isolation_dir/src"
echo "🔒 Pynomaly Virtual Environment Isolation Activated"
echo "📁 Workspace: \$(pwd)"
echo "🐍 Python: \$(python --version)"
exec /bin/bash
EOF
    chmod +x "$isolation_dir/activate-isolation.sh"

    success "Virtual environment isolation created: $isolation_id"
    info "Activate with: $isolation_dir/activate-isolation.sh"

    # Save isolation metadata
    save_isolation_metadata "$isolation_id" "venv" "$isolation_dir"
}

# Create folder-based isolation
create_folder_isolation() {
    local isolation_id="$1"

    info "Creating folder-based isolation: $isolation_id"

    # Create isolation directory
    local isolation_dir="$PROJECT_ROOT/.isolated-work/folder-$isolation_id"
    mkdir -p "$isolation_dir"

    # Copy only specific files (for documentation/config changes)
    rsync -av --include='docs/' --include='*.md' --include='*.yaml' \
          --include='*.yml' --include='*.json' --include='*.toml' \
          --exclude='*' "$PROJECT_ROOT/" "$isolation_dir/"

    success "Folder isolation created: $isolation_id"
    info "Work directory: $isolation_dir"

    # Save isolation metadata
    save_isolation_metadata "$isolation_id" "folder" "$isolation_dir"
}

# Save isolation metadata
save_isolation_metadata() {
    local isolation_id="$1"
    local strategy="$2"
    local path="$3"

    local metadata_file="$PROJECT_ROOT/.isolated-work/.metadata.json"
    local current_time
    current_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Initialize metadata file if it doesn't exist
    if [ ! -f "$metadata_file" ]; then
        echo '{"isolations": []}' > "$metadata_file"
    fi

    # Add isolation metadata
    local temp_file
    temp_file=$(mktemp)
    jq --arg id "$isolation_id" \
       --arg strategy "$strategy" \
       --arg path "$path" \
       --arg created "$current_time" \
       '.isolations += [{
         "id": $id,
         "strategy": $strategy,
         "path": $path,
         "created": $created,
         "active": true
       }]' "$metadata_file" > "$temp_file" && mv "$temp_file" "$metadata_file"
}

# List active isolations
list_isolations() {
    local metadata_file="$PROJECT_ROOT/.isolated-work/.metadata.json"

    if [ ! -f "$metadata_file" ]; then
        info "No isolations found"
        return
    fi

    echo "Active Isolations:"
    echo "=================="

    jq -r '.isolations[] | select(.active == true) |
           "ID: \(.id)\nStrategy: \(.strategy)\nPath: \(.path)\nCreated: \(.created)\n---"' \
           "$metadata_file"
}

# Clean up isolation
cleanup_isolation() {
    local isolation_id="$1"
    local metadata_file="$PROJECT_ROOT/.isolated-work/.metadata.json"

    if [ ! -f "$metadata_file" ]; then
        warn "No isolation metadata found"
        return
    fi

    # Get isolation info
    local isolation_info
    isolation_info=$(jq -r --arg id "$isolation_id" \
                     '.isolations[] | select(.id == $id)' "$metadata_file")

    if [ -z "$isolation_info" ]; then
        warn "Isolation not found: $isolation_id"
        return
    fi

    local strategy
    strategy=$(echo "$isolation_info" | jq -r '.strategy')
    local path
    path=$(echo "$isolation_info" | jq -r '.path')

    info "Cleaning up isolation: $isolation_id (strategy: $strategy)"

    # Strategy-specific cleanup
    case "$strategy" in
        "container")
            if [ -d "$path" ]; then
                cd "$path"
                docker-compose down -v 2>/dev/null || true
                docker-compose rm -f 2>/dev/null || true
                # Remove volumes
                docker volume rm $(docker volume ls -q | grep "$isolation_id") 2>/dev/null || true
            fi
            ;;
        "venv"|"folder")
            # For venv and folder, just remove the directory
            ;;
    esac

    # Remove isolation directory
    if [ -d "$path" ]; then
        rm -rf "$path"
        success "Removed isolation directory: $path"
    fi

    # Update metadata
    local temp_file
    temp_file=$(mktemp)
    jq --arg id "$isolation_id" \
       '.isolations = (.isolations | map(if .id == $id then .active = false else . end))' \
       "$metadata_file" > "$temp_file" && mv "$temp_file" "$metadata_file"

    success "Isolation cleaned up: $isolation_id"
}

# Auto-cleanup old isolations
auto_cleanup() {
    local metadata_file="$PROJECT_ROOT/.isolated-work/.metadata.json"
    local max_age_days="${1:-7}"

    if [ ! -f "$metadata_file" ]; then
        return
    fi

    info "Running auto-cleanup (max age: $max_age_days days)..."

    local cutoff_date
    cutoff_date=$(date -u -d "$max_age_days days ago" +"%Y-%m-%dT%H:%M:%SZ")

    # Find old isolations
    local old_isolations
    old_isolations=$(jq -r --arg cutoff "$cutoff_date" \
                     '.isolations[] | select(.active == true and .created < $cutoff) | .id' \
                     "$metadata_file")

    if [ -n "$old_isolations" ]; then
        while IFS= read -r isolation_id; do
            warn "Auto-cleaning old isolation: $isolation_id"
            cleanup_isolation "$isolation_id"
        done <<< "$old_isolations"
    else
        info "No old isolations found for cleanup"
    fi
}

# Show help
show_help() {
    cat <<EOF
Pynomaly Isolation Manager

Usage: $0 <command> [options]

Commands:
  create [strategy] [profile]  Create new isolation environment
                              Strategies: container (default), venv, folder
                              Profiles: development, testing, experimentation, documentation

  list                        List all active isolations

  cleanup <isolation-id>      Clean up specific isolation

  auto-cleanup [days]         Clean up isolations older than N days (default: 7)

  status                      Show isolation system status

  help                        Show this help message

Examples:
  $0 create container development    # Create container isolation for development
  $0 create venv                     # Create virtual environment isolation
  $0 create folder                   # Create folder isolation for docs/config
  $0 list                            # List active isolations
  $0 cleanup user-feature-123        # Clean up specific isolation
  $0 auto-cleanup 14                 # Clean up isolations older than 14 days

Configuration:
  Edit $CONFIG_FILE to modify isolation settings.

Logs:
  Isolation logs are written to $ISOLATION_LOG
EOF
}

# Main command handler
main() {
    # Ensure isolation work directory exists
    mkdir -p "$PROJECT_ROOT/.isolated-work"

    case "${1:-help}" in
        "create")
            check_isolation_enabled
            check_prerequisites

            local strategy="${2:-container}"
            local profile="${3:-development}"
            local isolation_id
            isolation_id=$(generate_isolation_id)

            case "$strategy" in
                "container")
                    create_container_isolation "$isolation_id" "$profile"
                    ;;
                "venv")
                    create_venv_isolation "$isolation_id"
                    ;;
                "folder")
                    create_folder_isolation "$isolation_id"
                    ;;
                *)
                    error "Unknown strategy: $strategy"
                    echo "Available strategies: container, venv, folder"
                    exit 1
                    ;;
            esac
            ;;
        "list")
            list_isolations
            ;;
        "cleanup")
            if [ -z "${2:-}" ]; then
                error "Please specify isolation ID to cleanup"
                exit 1
            fi
            cleanup_isolation "$2"
            ;;
        "auto-cleanup")
            auto_cleanup "${2:-7}"
            ;;
        "status")
            echo "Isolation System Status"
            echo "======================="
            echo "Configuration: $CONFIG_FILE"
            echo "Enabled: $(grep -o 'enabled: [^[:space:]]*' "$CONFIG_FILE" || echo 'unknown')"
            echo "Work Directory: $PROJECT_ROOT/.isolated-work"
            echo "Active Isolations: $(jq -r '.isolations | map(select(.active == true)) | length' "$PROJECT_ROOT/.isolated-work/.metadata.json" 2>/dev/null || echo '0')"
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
