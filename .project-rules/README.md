# anomaly_detection Isolation System

A comprehensive isolation framework for safe development without conflicts with staging, commits, merging, and checkout operations.

## 🎯 Purpose

The isolation system provides three strategies for working on changes in complete isolation:

- **Container isolation**: Full Docker-based environments with services
- **Virtual environment isolation**: Python venv with dependency isolation  
- **Folder isolation**: Simple file-based isolation for docs/config

## 🚀 Quick Start

### 1. Enable Isolation (Currently Disabled for Testing)

```bash
# Edit configuration to enable
vim .project-rules/isolation-config.yaml
# Set: enabled: true

# Install Git hooks for automation
.project-rules/automation/install-hooks.sh
```

### 2. Create Isolation Environment

```bash
# Container isolation (recommended for complex changes)
.project-rules/scripts/isolate.sh create container development

# Virtual environment isolation (lightweight)
.project-rules/scripts/isolate.sh create venv

# Folder isolation (docs/config only)
.project-rules/scripts/isolate.sh create folder
```

### 3. Work in Isolation

```bash
# For container isolation
cd .isolated-work/container-*/
docker-compose exec anomaly_detection-isolated /bin/bash

# For venv isolation
./activate-isolation.sh

# For folder isolation
cd .isolated-work/folder-*/
```

## 📋 Available Commands

```bash
# Create isolation
.project-rules/scripts/isolate.sh create [strategy] [profile]

# List active isolations
.project-rules/scripts/isolate.sh list

# Clean up specific isolation
.project-rules/scripts/isolate.sh cleanup <isolation-id>

# Auto-cleanup old isolations
.project-rules/scripts/isolate.sh auto-cleanup [days]

# Show system status
.project-rules/scripts/isolate.sh status
```

## 🔧 Configuration

Edit `.project-rules/isolation-config.yaml` to customize:

- Isolation strategies and profiles
- Automatic triggers based on branch/file patterns
- Resource limits and cleanup policies
- Git integration settings

## 🛡️ Safety Features

- **Automatic cleanup**: Removes old isolation environments
- **Resource monitoring**: Prevents excessive disk/memory usage
- **Git integration**: Prevents dangerous operations on main branch
- **Conflict prevention**: Isolates changes from main workspace

## 📊 Isolation Strategies

### Container Strategy

- Full Docker environment with PostgreSQL, Redis
- Complete service isolation
- Persistent volumes for data
- Multiple profiles (development, testing, experimentation)

### Virtual Environment Strategy  

- Python venv with isolated dependencies
- Lightweight and fast
- Good for Python-only changes
- Automatic cleanup

### Folder Strategy

- Simple file copying for docs/config
- No dependency management
- Fastest setup
- Good for documentation changes

## 🔄 Automation Features

When Git hooks are installed:

- **Auto-trigger**: Creates isolation for qualifying file changes
- **Branch protection**: Prevents direct commits to main
- **Commit context**: Adds isolation metadata to commits
- **Push validation**: Validates isolation state before pushing

## 🧪 Testing the System

```bash
# Test container isolation
.project-rules/scripts/isolate.sh create container testing

# Test virtual environment
.project-rules/scripts/isolate.sh create venv

# Test folder isolation
.project-rules/scripts/isolate.sh create folder

# List all test isolations
.project-rules/scripts/isolate.sh list

# Clean up all test isolations
.project-rules/scripts/isolate.sh auto-cleanup 0
```

## 📁 Directory Structure

```
.project-rules/
├── isolation-config.yaml          # Main configuration
├── templates/                     # Docker templates
│   ├── Dockerfile.isolation
│   └── docker-compose.isolation.yml
├── scripts/                       # Management scripts
│   ├── isolate.sh                # Main isolation manager
│   ├── help.sh                   # Help and status info
│   ├── start-dev.sh              # Development server
│   └── test.sh                   # Test runner
├── hooks/                         # Git hooks
│   └── pre-commit-isolation
├── automation/                    # Setup scripts
│   └── install-hooks.sh
└── README.md                      # This file

.isolated-work/                    # Isolation workspaces
├── container-*/                   # Container isolations
├── venv-*/                       # Virtual env isolations
├── folder-*/                     # Folder isolations
└── .metadata.json                # Isolation tracking
```

## ⚠️ Important Notes

- **Currently disabled**: System starts disabled for testing
- **Testing required**: Thoroughly test before enabling in production
- **Resource usage**: Monitor disk/memory usage with multiple isolations
- **Git integration**: Hooks provide automation but can be bypassed
- **Cleanup**: Regular cleanup prevents resource exhaustion

## 🔗 Integration

The isolation system integrates with:

- **Git hooks**: Automatic triggering and validation
- **CI/CD**: Can be disabled in CI environments  
- **Docker**: Container-based isolation with services
- **Testing**: Isolated test environments with coverage
- **Development**: Hot-reload development servers

## 🆘 Troubleshooting

```bash
# Check system status
.project-rules/scripts/isolate.sh status

# View isolation logs
tail -f .isolation.log

# Clean up all isolations
.project-rules/scripts/isolate.sh auto-cleanup 0

# Uninstall Git hooks
rm .git/hooks/pre-commit .git/hooks/post-commit .git/hooks/prepare-commit-msg .git/hooks/pre-push
```
