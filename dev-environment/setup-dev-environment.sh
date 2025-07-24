#!/bin/bash

# Development Environment Setup Script
# This script sets up a complete development environment for the anomaly detection platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    log_success "Detected OS: $OS"
    
    # Check required tools
    local required_tools=("git" "curl" "wget")
    for tool in "${required_tools[@]}"; do
        if ! command_exists "$tool"; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    log_success "System requirements check passed"
}

# Install Python and pyenv
install_python() {
    log_info "Setting up Python environment..."
    
    if ! command_exists pyenv; then
        log_info "Installing pyenv..."
        if [[ "$OS" == "macos" ]]; then
            if command_exists brew; then
                brew install pyenv
            else
                log_error "Homebrew is required on macOS. Please install it first."
                exit 1
            fi
        else
            curl https://pyenv.run | bash
        fi
        
        # Add pyenv to PATH
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
    fi
    
    # Install Python versions
    local python_versions=("3.11.7" "3.12.1")
    for version in "${python_versions[@]}"; do
        if ! pyenv versions | grep -q "$version"; then
            log_info "Installing Python $version..."
            pyenv install "$version"
        fi
    done
    
    # Set global Python version
    pyenv global 3.11.7
    
    log_success "Python environment setup complete"
}

# Install Poetry
install_poetry() {
    log_info "Installing Poetry..."
    
    if ! command_exists poetry; then
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Configure Poetry
    poetry config virtualenvs.in-project true
    poetry config virtualenvs.prefer-active-python true
    
    log_success "Poetry installation complete"
}

# Install Docker
install_docker() {
    log_info "Checking Docker installation..."
    
    if ! command_exists docker; then
        log_warning "Docker is not installed. Please install Docker manually:"
        if [[ "$OS" == "macos" ]]; then
            log_info "  - Download Docker Desktop for Mac from https://docker.com"
        elif [[ "$OS" == "linux" ]]; then
            log_info "  - Run: curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
        else
            log_info "  - Download Docker Desktop from https://docker.com"
        fi
        return 1
    fi
    
    if ! command_exists docker-compose; then
        log_warning "Docker Compose is not installed. Installing..."
        if [[ "$OS" == "linux" ]]; then
            sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
        fi
    fi
    
    log_success "Docker setup complete"
}

# Install development tools
install_dev_tools() {
    log_info "Installing development tools..."
    
    # Install Node.js and npm (for pre-commit and other tools)
    if ! command_exists node; then
        if command_exists curl; then
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
            export NVM_DIR="$HOME/.nvm"
            [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
            nvm install --lts
            nvm use --lts
        fi
    fi
    
    # Install system packages based on OS
    if [[ "$OS" == "macos" ]] && command_exists brew; then
        brew install git-lfs hadolint shellcheck
    elif [[ "$OS" == "linux" ]] && command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y git-lfs shellcheck
        
        # Install Hadolint
        wget -O /tmp/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64
        chmod +x /tmp/hadolint
        sudo mv /tmp/hadolint /usr/local/bin/hadolint
    fi
    
    log_success "Development tools installation complete"
}

# Install Buck2 build system
install_buck2() {
    log_info "Installing Buck2 build system..."
    
    if ! command_exists buck2; then
        # Download and install Buck2
        local buck2_url
        if [[ "$OS" == "macos" ]]; then
            buck2_url="https://github.com/facebook/buck2/releases/download/latest/buck2-x86_64-apple-darwin.zst"
        elif [[ "$OS" == "linux" ]]; then
            buck2_url="https://github.com/facebook/buck2/releases/download/latest/buck2-x86_64-unknown-linux-gnu.zst"
        else
            log_warning "Buck2 automatic installation not supported on $OS"
            return 1
        fi
        
        wget -O /tmp/buck2.zst "$buck2_url"
        zstd -d /tmp/buck2.zst -o /tmp/buck2
        chmod +x /tmp/buck2
        sudo mv /tmp/buck2 /usr/local/bin/buck2
    fi
    
    log_success "Buck2 installation complete"
}

# Setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    # Install pre-commit
    pip install pre-commit
    
    # Install pre-commit hooks
    pre-commit install
    pre-commit install --hook-type commit-msg
    pre-commit install --hook-type pre-push
    
    log_success "Pre-commit hooks setup complete"
}

# Install project dependencies
install_project_dependencies() {
    log_info "Installing project dependencies..."
    
    # Install main project dependencies
    if [[ -f "requirements-prod.txt" ]]; then
        pip install -r requirements-prod.txt
    fi
    
    # Install development dependencies
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
    fi
    
    # Install anomaly detection package in development mode
    if [[ -d "src/packages/data/anomaly_detection" ]]; then
        cd src/packages/data/anomaly_detection
        pip install -e .
        cd - > /dev/null
    fi
    
    log_success "Project dependencies installation complete"
}

# Setup IDE configurations
setup_ide() {
    log_info "Setting up IDE configurations..."
    
    # VS Code extensions (if VS Code is installed)
    if command_exists code; then
        log_info "Installing VS Code extensions..."
        local extensions=(
            "ms-python.python"
            "ms-python.black-formatter"
            "ms-python.mypy-type-checker"
            "charliermarsh.ruff"
            "ms-azuretools.vscode-docker"
            "eamodio.gitlens"
            "github.vscode-pull-request-github"
        )
        
        for ext in "${extensions[@]}"; do
            code --install-extension "$ext" --force
        done
        
        log_success "VS Code extensions installed"
    fi
}

# Validate installation
validate_installation() {
    log_info "Validating installation..."
    
    local errors=0
    
    # Check Python
    if ! python --version | grep -q "3.1[1-2]"; then
        log_error "Python 3.11 or 3.12 not found"
        ((errors++))
    fi
    
    # Check essential tools
    local tools=("git" "docker" "pre-commit")
    for tool in "${tools[@]}"; do
        if ! command_exists "$tool"; then
            log_error "$tool not found"
            ((errors++))
        fi
    done
    
    # Test pre-commit
    if ! pre-commit run --all-files > /dev/null 2>&1; then
        log_warning "Pre-commit hooks failed - this is normal for first run"
    fi
    
    # Test Docker
    if ! docker run --rm hello-world > /dev/null 2>&1; then
        log_error "Docker test failed"
        ((errors++))
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "All validation checks passed!"
        return 0
    else
        log_error "$errors validation checks failed"
        return 1
    fi
}

# Print usage information
print_next_steps() {
    log_success "Development environment setup complete!"
    echo
    echo "🚀 Next steps:"
    echo "1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo "2. Activate Python environment: pyenv activate 3.11.7"
    echo "3. Run tests: pytest"
    echo "4. Start development server: python -m uvicorn anomaly_detection.api.main:app --reload"
    echo "5. Open VS Code: code ."
    echo
    echo "📚 Useful commands:"
    echo "- Run pre-commit: pre-commit run --all-files"
    echo "- Build with Buck2: buck2 build //..."
    echo "- Run Docker stack: docker-compose up -d"
    echo "- Check code quality: ruff check src/"
    echo
    echo "🔗 Resources:"
    echo "- Documentation: docs/"
    echo "- VS Code workspace: open monorepo.code-workspace"
    echo "- Development scripts: src/development_scripts/"
}

# Main function
main() {
    log_info "Starting development environment setup..."
    
    # Change to script directory
    cd "$(dirname "$0")/.."
    
    # Run setup steps
    check_system_requirements
    install_python
    install_poetry
    install_docker || log_warning "Docker setup skipped - install manually if needed"
    install_dev_tools
    install_buck2 || log_warning "Buck2 installation skipped - install manually if needed"
    setup_pre_commit
    install_project_dependencies
    setup_ide
    
    # Validate installation
    if validate_installation; then
        print_next_steps
    else
        log_error "Setup completed with errors. Please check the logs above."
        exit 1
    fi
}

# Handle command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "validate")
        validate_installation
        ;;
    "help")
        echo "Usage: $0 [setup|validate|help]"
        echo "  setup    - Run full development environment setup (default)"
        echo "  validate - Validate current installation"
        echo "  help     - Show this help message"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac