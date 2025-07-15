#!/bin/bash
# Prerequisites setup for advanced monitoring deployment

set -euo pipefail

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

log_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

# Function to install jq
install_jq() {
    log_info "Installing jq..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y jq
    elif command -v yum &> /dev/null; then
        sudo yum install -y jq
    elif command -v brew &> /dev/null; then
        brew install jq
    else
        log_error "Package manager not found. Please install jq manually."
        return 1
    fi
}

# Function to setup Docker on WSL
setup_docker_wsl() {
    log_info "Setting up Docker for WSL..."
    
    cat << 'EOF'
Docker Desktop is not available in this WSL environment.
For development purposes, we'll create a mock deployment script.

To properly deploy the monitoring stack:
1. Install Docker Desktop on Windows
2. Enable WSL integration in Docker Desktop settings
3. Restart your WSL session

Alternatively, install Docker directly in WSL:
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER

For now, we'll proceed with simulation mode.
EOF
}

main() {
    log_info "Setting up prerequisites for advanced monitoring deployment..."
    
    # Check and install jq
    if ! command -v jq &> /dev/null; then
        install_jq
    else
        log_success "jq is already installed"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        setup_docker_wsl
    else
        log_success "Docker is available"
    fi
    
    log_success "Prerequisites setup completed"
}

main "$@"