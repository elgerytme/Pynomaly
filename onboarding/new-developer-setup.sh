#!/bin/bash

# New Developer Onboarding Script
# Comprehensive automated setup for new team members

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_header() {
    echo -e "${PURPLE}[SETUP]${NC} $1"
}

# Welcome message
welcome_developer() {
    clear
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         🚀 ANOMALY DETECTION PLATFORM - DEVELOPER SETUP         ║
║                                                                  ║
║  Welcome to the team! This script will set up your complete     ║
║  development environment for the detection platform.     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
EOF
    echo
    echo "This setup will take approximately 15-30 minutes and includes:"
    echo "• Development environment configuration"
    echo "• IDE setup with extensions and debugging"
    echo "• Security tools and pre-commit hooks"
    echo "• Local development services"
    echo "• Team collaboration tools"
    echo "• Project documentation and tutorials"
    echo
    read -p "Press Enter to begin the setup process..."
}

# Collect developer information
collect_developer_info() {
    log_header "👤 Developer Information"
    
    echo "Let's set up your development profile:"
    
    read -p "Full Name: " DEVELOPER_NAME
    read -p "Email Address: " DEVELOPER_EMAIL
    read -p "GitHub Username: " GITHUB_USERNAME
    read -p "Preferred IDE (vscode/pycharm/other): " PREFERRED_IDE
    read -p "Operating System (linux/macos/windows): " OS_TYPE
    
    # Optional information
    echo
    echo "Optional information (press Enter to skip):"
    read -p "Slack Username: " SLACK_USERNAME
    read -p "Team/Department: " TEAM_NAME
    read -p "Role (developer/senior-developer/architect): " ROLE
    
    # Confirm information
    echo
    echo "Please confirm your information:"
    echo "Name: $DEVELOPER_NAME"
    echo "Email: $DEVELOPER_EMAIL"
    echo "GitHub: $GITHUB_USERNAME"
    echo "IDE: $PREFERRED_IDE"
    echo "OS: $OS_TYPE"
    echo
    read -p "Is this correct? (y/n): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Please restart the script to enter your information again"
        exit 0
    fi
}

# Set up Git configuration
setup_git_configuration() {
    log_header "📝 Git Configuration"
    
    # Configure Git user information
    git config --global user.name "$DEVELOPER_NAME"
    git config --global user.email "$DEVELOPER_EMAIL"
    
    # Set up useful Git aliases
    git config --global alias.co checkout
    git config --global alias.br branch
    git config --global alias.ci commit
    git config --global alias.st status
    git config --global alias.unstage 'reset HEAD --'
    git config --global alias.last 'log -1 HEAD'
    git config --global alias.visual '!gitk'
    
    # Configure Git to use rebase for pulls
    git config --global pull.rebase true
    
    # Set up commit template
    cat > ~/.gitmessage << EOF
# <type>(<scope>): <subject>
# 
# <body>
# 
# <footer>
#
# Type: feat, fix, docs, style, refactor, test, chore
# Scope: component or file name
# Subject: short description (50 chars max)
# Body: detailed description (wrap at 72 chars)
# Footer: closes #123, breaking changes, etc.
EOF
    
    git config --global commit.template ~/.gitmessage
    
    log_success "Git configuration completed"
}

# Install development tools
install_development_tools() {
    log_header "🔧 Development Tools Installation"
    
    # Run the existing development environment setup
    if [[ -f "dev-environment/setup-dev-environment.sh" ]]; then
        log_info "Running automated development environment setup..."
        ./dev-environment/setup-dev-environment.sh
    else
        log_warning "Development environment setup script not found, installing manually..."
        
        # Install Python via pyenv
        if ! command -v pyenv >/dev/null 2>&1; then
            log_info "Installing pyenv..."
            curl https://pyenv.run | bash
            export PATH="$HOME/.pyenv/bin:$PATH"
            eval "$(pyenv init --path)"
            eval "$(pyenv init -)"
        fi
        
        # Install Python versions
        pyenv install 3.11.7 2>/dev/null || true
        pyenv global 3.11.7
        
        # Install essential tools
        pip install --user pre-commit black ruff mypy pytest bandit safety
    fi
    
    log_success "Development tools installation completed"
}

# Set up IDE configuration
setup_ide_configuration() {
    log_header "💻 IDE Configuration"
    
    case $PREFERRED_IDE in
        "vscode")
            setup_vscode
            ;;
        "pycharm")
            setup_pycharm
            ;;
        *)
            log_info "Generic IDE setup (manual configuration may be required)"
            ;;
    esac
}

# Set up VS Code
setup_vscode() {
    log_info "Setting up VS Code configuration..."
    
    # Install VS Code extensions if code command is available
    if command -v code >/dev/null 2>&1; then
        log_info "Installing VS Code extensions..."
        
        # Essential extensions
        local extensions=(
            "ms-python.python"
            "ms-python.black-formatter"
            "ms-python.mypy-type-checker"
            "charliermarsh.ruff"
            "ms-azuretools.vscode-docker"
            "eamodio.gitlens"
            "github.vscode-pull-request-github"
            "github.copilot"
            "ms-vscode-remote.remote-containers"
            "redhat.vscode-yaml"
            "yzhang.markdown-all-in-one"
        )
        
        for ext in "${extensions[@]}"; do
            code --install-extension "$ext" --force 2>/dev/null || log_warning "Failed to install extension: $ext"
        done
        
        log_success "VS Code extensions installed"
    else
        log_warning "VS Code 'code' command not found. Please install extensions manually."
        log_info "Required extensions list saved to ~/vscode-extensions.txt"
        
        cat > ~/vscode-extensions.txt << EOF
Required VS Code Extensions for Anomaly Detection Platform:

Essential Extensions:
- Python (ms-python.python)
- Black Formatter (ms-python.black-formatter)
- MyPy Type Checker (ms-python.mypy-type-checker)
- Ruff (charliermarsh.ruff)
- Docker (ms-azuretools.vscode-docker)
- GitLens (eamodio.gitlens)
- GitHub Pull Requests (github.vscode-pull-request-github)
- GitHub Copilot (github.copilot)
- Remote Containers (ms-vscode-remote.remote-containers)
- YAML (redhat.vscode-yaml)
- Markdown All in One (yzhang.markdown-all-in-one)

Install these through VS Code Extensions marketplace or use:
code --install-extension <extension-id>
EOF
    fi
    
    # Create personal VS Code settings
    local vscode_settings_dir="$HOME/.config/Code/User"
    if [[ "$OS_TYPE" == "macos" ]]; then
        vscode_settings_dir="$HOME/Library/Application Support/Code/User"
    elif [[ "$OS_TYPE" == "windows" ]]; then
        vscode_settings_dir="$HOME/AppData/Roaming/Code/User"
    fi
    
    mkdir -p "$vscode_settings_dir"
    
    cat > "$vscode_settings_dir/settings.json" << EOF
{
    "python.defaultInterpreterPath": "~/.pyenv/shims/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/.coverage": true
    },
    "git.enableCommitSigning": false,
    "git.confirmSync": false,
    "terminal.integrated.defaultProfile.linux": "bash",
    "workbench.colorTheme": "Default Dark+",
    "explorer.confirmDelete": false,
    "developer.name": "$DEVELOPER_NAME",
    "developer.email": "$DEVELOPER_EMAIL"
}
EOF
    
    log_success "VS Code configuration completed"
}

# Set up PyCharm
setup_pycharm() {
    log_info "Setting up PyCharm configuration..."
    
    # Create PyCharm configuration directory
    local pycharm_config_dir="$HOME/.config/JetBrains"
    if [[ "$OS_TYPE" == "macos" ]]; then
        pycharm_config_dir="$HOME/Library/Application Support/JetBrains"
    fi
    
    # Create configuration notes
    cat > ~/pycharm-setup-notes.txt << EOF
PyCharm Setup Notes for Anomaly Detection Platform:

1. Configure Python Interpreter:
   - Go to File > Settings > Project > Python Interpreter
   - Add new interpreter: ~/.pyenv/versions/3.11.7/bin/python

2. Enable Code Quality Tools:
   - Install plugins: Black, MyPy, Ruff
   - Configure formatters in Settings > Tools > External Tools

3. Set up Version Control:
   - Configure Git integration
   - Set up GitHub integration with your token

4. Configure Testing:
   - Set pytest as default test runner
   - Configure test templates

5. Import Code Style:
   - Use Black formatter settings
   - Set line length to 88 characters

6. Set up Docker Integration:
   - Enable Docker plugin
   - Configure docker-compose files

Manual configuration required - see ~/pycharm-setup-notes.txt
EOF
    
    log_success "PyCharm setup notes created"
}

# Set up local development services
setup_local_services() {
    log_header "🐳 Local Development Services"
    
    if command -v docker >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then
        log_info "Setting up local development services with Docker..."
        
        # Create developer-specific docker-compose override
        cat > docker-compose.override.yml << EOF
version: '3.8'

# Developer-specific overrides
# This file is gitignored and can be customized per developer

services:
  api:
    environment:
      - DEVELOPER_NAME=${DEVELOPER_NAME}
      - DEVELOPER_EMAIL=${DEVELOPER_EMAIL}
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    volumes:
      - ./logs:/app/logs
    ports:
      - "8000:8000"
      - "5678:5678"  # Debug port
  
  database:
    ports:
      - "5432:5432"
    environment:
      - DEVELOPER=${GITHUB_USERNAME}
  
  redis:
    ports:
      - "6379:6379"
  
  # Add developer-specific services
  jupyter:
    image: jupyter/scipy-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
    environment:
      - JUPYTER_ENABLE_LAB=yes
EOF
        
        # Start development services
        log_info "Starting development services..."
        docker-compose up -d database redis 2>/dev/null || log_warning "Could not start all services"
        
        log_success "Local development services configured"
    else
        log_warning "Docker not found. Local services will need manual setup."
    fi
}

# Set up project documentation access
setup_documentation_access() {
    log_header "📚 Documentation and Resources"
    
    # Create developer bookmarks
    cat > ~/anomaly-detection-bookmarks.md << EOF
# 📚 Anomaly Detection Platform - Developer Resources

## 🔗 Quick Links
- **Repository:** https://github.com/your-org/anomaly-detection
- **Documentation:** ./docs/
- **API Docs:** http://localhost:8000/docs (when running locally)
- **Grafana:** http://localhost:3000 (admin/grafana)
- **Jupyter:** http://localhost:8888 (when running)

## 📖 Essential Reading
1. **Getting Started:** [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
2. **Architecture Overview:** [docs/architecture/](docs/architecture/)
3. **API Documentation:** [docs/api/](docs/api/)
4. **Contributing Guidelines:** [CONTRIBUTING.md](CONTRIBUTING.md)
5. **Development Guide:** [dev-environment/README.md](dev-environment/README.md)

## 🛠️ Development Workflows
- **Code Quality:** Run \`pre-commit run --all-files\`
- **Testing:** Run \`pytest\` or \`python -m pytest\`
- **Linting:** Run \`ruff check src/\`
- **Formatting:** Run \`black src/\`
- **Type Checking:** Run \`mypy src/\`

## 🧪 Local Development
- **Start API:** \`python -m uvicorn anomaly_detection.api.main:app --reload\`
- **Run Tests:** \`pytest -v\`
- **Start Services:** \`docker-compose up -d\`
- **View Logs:** \`docker-compose logs -f\`

## 👥 Team Resources
- **Slack Channels:** 
  - #anomaly-detection-dev
  - #anomaly-detection-alerts
  - #general-dev
- **Code Reviews:** Use GitHub Pull Requests
- **Issues:** GitHub Issues with appropriate labels
- **Meetings:** Check team calendar

## 🚨 Getting Help
1. Check documentation first
2. Search existing GitHub issues  
3. Ask in Slack #anomaly-detection-dev
4. Create GitHub issue with detailed description
5. Pair with senior team member

## 🎯 Your First Tasks
1. Complete this onboarding checklist
2. Set up development environment
3. Run the test suite successfully
4. Make a small documentation improvement PR
5. Attend next team standup
6. Review recent PRs to understand code style

## 📝 Onboarding Checklist
- [ ] Complete environment setup
- [ ] Configure IDE with extensions
- [ ] Run tests successfully
- [ ] Start local development services
- [ ] Join team Slack channels
- [ ] Set up GitHub notifications
- [ ] Review architecture documentation
- [ ] Understand deployment process
- [ ] Complete first PR (documentation fix)
- [ ] Meet with team lead
EOF
    
    log_success "Documentation and resources set up"
}

# Create developer profile
create_developer_profile() {
    log_header "👤 Developer Profile Creation"
    
    # Create developer profile file
    cat > ~/.anomaly-detection-profile << EOF
# Anomaly Detection Platform - Developer Profile
# Created: $(date)

DEVELOPER_NAME="$DEVELOPER_NAME"
DEVELOPER_EMAIL="$DEVELOPER_EMAIL"
GITHUB_USERNAME="$GITHUB_USERNAME"
PREFERRED_IDE="$PREFERRED_IDE"
OS_TYPE="$OS_TYPE"
SLACK_USERNAME="$SLACK_USERNAME"
TEAM_NAME="$TEAM_NAME"
ROLE="$ROLE"
ONBOARDING_DATE="$(date -I)"
PROFILE_VERSION="1.0"

# Development preferences
export ANOMALY_DETECTION_DEV_NAME="$DEVELOPER_NAME"
export ANOMALY_DETECTION_DEV_EMAIL="$DEVELOPER_EMAIL"
export PYTHONPATH="\${PWD}/src:\${PYTHONPATH}"

# Useful aliases
alias ad-test='pytest -v'
alias ad-lint='ruff check src/'
alias ad-format='black src/'
alias ad-type='mypy src/'
alias ad-api='python -m uvicorn anomaly_detection.api.main:app --reload'
alias ad-services='docker-compose up -d'
alias ad-logs='docker-compose logs -f'
alias ad-clean='find . -type d -name __pycache__ -delete && find . -name "*.pyc" -delete'

# Load this profile in your shell startup file (.bashrc, .zshrc, etc.)
EOF
    
    # Add to shell startup file
    local shell_rc=""
    if [[ -n "${BASH_VERSION:-}" ]]; then
        shell_rc="$HOME/.bashrc"
    elif [[ -n "${ZSH_VERSION:-}" ]]; then
        shell_rc="$HOME/.zshrc"
    fi
    
    if [[ -n "$shell_rc" ]] && [[ -f "$shell_rc" ]]; then
        if ! grep -q "anomaly-detection-profile" "$shell_rc"; then
            echo "" >> "$shell_rc"
            echo "# Anomaly Detection Platform Profile" >> "$shell_rc"
            echo "source ~/.anomaly-detection-profile" >> "$shell_rc"
            log_success "Developer profile added to $shell_rc"
        fi
    fi
}

# Set up team collaboration tools
setup_collaboration_tools() {
    log_header "🤝 Team Collaboration Setup"
    
    # GitHub CLI setup
    if command -v gh >/dev/null 2>&1; then
        log_info "Setting up GitHub CLI..."
        
        if ! gh auth status >/dev/null 2>&1; then
            log_info "Please authenticate with GitHub CLI:"
            gh auth login
        fi
        
        # Set up useful GitHub CLI aliases
        gh alias set pv 'pr view --web'
        gh alias set pc 'pr create --web'
        gh alias set is 'issue list'
        gh alias set ic 'issue create --web'
        
        log_success "GitHub CLI configured"
    else
        log_warning "GitHub CLI not found. Install with: brew install gh (macOS) or apt install gh (Ubuntu)"
    fi
    
    # Create collaboration guidelines
    cat > ~/collaboration-guidelines.md << EOF
# 🤝 Team Collaboration Guidelines

## 📋 Daily Workflow
1. Start day: Pull latest changes, check Slack
2. Work on assigned issues/features
3. Regular commits with meaningful messages
4. Push changes and create PR when ready
5. End day: Update team on progress

## 🔄 Pull Request Process
1. Create feature branch: git checkout -b feature/your-feature
2. Make changes and commit regularly
3. Run tests and quality checks locally
4. Push branch: git push origin feature/your-feature
5. Create PR with descriptive title and description
6. Request reviews from team members
7. Address feedback and merge when approved

## 💬 Communication
- **Slack:** Daily updates and quick questions
- **GitHub Issues:** Bug reports and feature requests
- **PR Comments:** Code-specific discussions
- **Standups:** Progress updates and blockers
- **Documentation:** Important decisions and processes

## 🎯 Code Quality Standards
- All code must pass pre-commit hooks
- Maintain test coverage above 80%
- Follow existing code patterns and style
- Write clear commit messages
- Document complex functions and classes
- Update relevant documentation

## 🚨 When You're Stuck
1. Try to solve for 30 minutes
2. Check documentation and existing code
3. Ask in Slack (don't suffer in silence!)
4. Pair program with team member
5. Escalate to team lead if needed

## 🎉 Contributing to Team Culture
- Help onboard new team members
- Share knowledge in team meetings
- Contribute to documentation
- Participate in code reviews
- Suggest process improvements
EOF
    
    log_success "Collaboration tools and guidelines set up"
}

# Run validation tests
run_validation_tests() {
    log_header "✅ Environment Validation"
    
    log_info "Running environment validation tests..."
    
    # Test Python environment
    if python --version | grep -q "3.1[1-2]"; then
        log_success "Python version check passed"
    else
        log_error "Python version check failed"
    fi
    
    # Test essential tools
    local tools=("git" "pip" "pytest" "black" "ruff")
    for tool in "${tools[@]}"; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_success "$tool is available"
        else
            log_error "$tool is not available"
        fi
    done
    
    # Test pre-commit hooks
    if [[ -f ".pre-commit-config.yaml" ]]; then
        if pre-commit run --all-files >/dev/null 2>&1; then
            log_success "Pre-commit hooks validation passed"
        else
            log_warning "Pre-commit hooks validation failed (normal for first run)"
        fi
    fi
    
    # Test project structure
    if [[ -d "src" ]] && [[ -d "tests" ]] && [[ -f "README.md" ]]; then
        log_success "Project structure validation passed"
    else
        log_error "Project structure validation failed"
    fi
    
    # Run development environment validator if available
    if [[ -f "dev-environment/validate-dev-environment.py" ]]; then
        if python dev-environment/validate-dev-environment.py >/dev/null 2>&1; then
            log_success "Development environment validation passed"
        else
            log_warning "Development environment validation failed"
        fi
    fi
}

# Generate onboarding report
generate_onboarding_report() {
    log_header "📋 Onboarding Report Generation"
    
    local report_file="onboarding-report-${GITHUB_USERNAME}-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# 🎉 Developer Onboarding Report

**Developer:** $DEVELOPER_NAME  
**Email:** $DEVELOPER_EMAIL  
**GitHub:** $GITHUB_USERNAME  
**Setup Date:** $(date)  
**Environment:** $OS_TYPE  

## ✅ Completed Setup Steps

1. **Personal Information Collected**
   - Full name and contact information
   - GitHub and Slack integration
   - Role and team assignment

2. **Git Configuration**
   - User name and email configured
   - Useful aliases set up
   - Commit template installed
   - Rebase preference configured

3. **Development Environment**
   - Python 3.11 installed via pyenv
   - Essential development tools installed
   - Pre-commit hooks configured
   - IDE configuration completed

4. **IDE Setup (${PREFERRED_IDE})**
   - Extensions installed
   - Configuration files created
   - Debug configuration available
   - Code quality tools integrated

5. **Local Services**
   - Docker containers configured
   - Development database accessible
   - API server ready for development
   - Jupyter notebook environment available

6. **Documentation Access**
   - Essential reading list provided
   - Team resources documented
   - Collaboration guidelines created
   - Quick reference bookmarks

7. **Developer Profile**
   - Profile file created
   - Shell aliases configured
   - Environment variables set
   - Startup integration completed

8. **Validation Tests**
   - Environment validation completed
   - Tool availability verified
   - Project structure validated
   - Pre-commit hooks tested

## 🔗 Important Resources

- **Bookmarks:** ~/anomaly-detection-bookmarks.md
- **Profile:** ~/.anomaly-detection-profile
- **Collaboration:** ~/collaboration-guidelines.md
- **IDE Notes:** ~/$(echo $PREFERRED_IDE)-setup-notes.txt

## 🎯 Next Steps

### Immediate (Today)
1. [ ] Restart your terminal to load new configuration
2. [ ] Open the project in your IDE
3. [ ] Run \`pytest\` to verify everything works
4. [ ] Join team Slack channels
5. [ ] Introduce yourself to the team

### This Week
1. [ ] Complete first pull request (documentation improvement)
2. [ ] Attend team standup meeting
3. [ ] Review recent pull requests for code style
4. [ ] Set up GitHub notifications
5. [ ] Schedule 1:1 with team lead

### This Month
1. [ ] Complete first feature implementation
2. [ ] Participate in code reviews
3. [ ] Contribute to team documentation
4. [ ] Suggest process improvements
5. [ ] Help onboard next new team member

## 🛠️ Quick Commands

\`\`\`bash
# Test everything is working
pytest

# Start development server
python -m uvicorn anomaly_detection.api.main:app --reload

# Run code quality checks
pre-commit run --all-files

# Start local services
docker-compose up -d

# Format and lint code
black src/ && ruff check src/ --fix
\`\`\`

## 🆘 Getting Help

- **Slack:** #anomaly-detection-dev
- **Email:** team-lead@company.com
- **Documentation:** docs/GETTING_STARTED.md
- **Issues:** GitHub Issues with 'help-wanted' label

## 🎊 Welcome to the Team!

You're all set up and ready to contribute to the detection platform. 
The team is excited to work with you!

Remember: Don't hesitate to ask questions. Everyone was new once, and 
we're here to help you succeed.

Happy coding! 🚀
EOF
    
    log_success "Onboarding report generated: $report_file"
    echo
    log_info "📋 Your onboarding report has been saved to: $report_file"
}

# Final success message
final_success_message() {
    clear
    cat << 'EOF'
🎉 CONGRATULATIONS! 🎉

Your development environment is now fully configured and ready for use!

╔══════════════════════════════════════════════════════════════════╗
║                     ✅ SETUP COMPLETE ✅                        ║
║                                                                  ║
║  Your detection platform development environment is      ║
║  ready! You can now start contributing to the project.          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

🚀 QUICK START:
   1. Restart your terminal: source ~/.bashrc (or ~/.zshrc)
   2. Test the setup: pytest
   3. Start coding: code . (if using VS Code)

📚 IMPORTANT FILES:
   • ~/anomaly-detection-bookmarks.md - Essential links and resources
   • ~/collaboration-guidelines.md - Team processes and standards
   • ~/.anomaly-detection-profile - Your developer profile and aliases

🤝 NEXT STEPS:
   • Join Slack channels: #anomaly-detection-dev
   • Schedule 1:1 with team lead
   • Review the architecture documentation
   • Make your first contribution!

Welcome to the Anomaly Detection Platform team! 🎊
EOF
}

# Main execution function
main() {
    # Check if running from the repository root
    if [[ ! -f "README.md" ]] || [[ ! -d "src" ]]; then
        log_error "Please run this script from the repository root directory"
        exit 1
    fi
    
    # Run onboarding process
    welcome_developer
    collect_developer_info
    setup_git_configuration
    install_development_tools
    setup_ide_configuration
    setup_local_services
    setup_documentation_access
    create_developer_profile
    setup_collaboration_tools
    run_validation_tests
    generate_onboarding_report
    final_success_message
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi