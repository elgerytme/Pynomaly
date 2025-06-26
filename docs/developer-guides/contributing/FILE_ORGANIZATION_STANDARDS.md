# File Organization Standards

## 📁 Overview

This document defines the strict file organization standards for the Pynomaly project to maintain a clean, navigable, and professional repository structure.

## 🎯 Core Principles

1. **Clean Root Directory**: Only essential project files in the root
2. **Logical Categorization**: Files grouped by purpose and function
3. **Predictable Structure**: Consistent directory naming and organization
4. **Automated Enforcement**: Rules enforced through tooling and CI/CD

## 📂 Mandatory Directory Structure

### Root Directory (/) - RESTRICTED
**Only these files are allowed in the project root:**

#### Essential Project Files
- `README.md` - Primary project documentation
- `LICENSE` - Project license
- `CHANGELOG.md` - Version history
- `TODO.md` - Project roadmap and tasks
- `CLAUDE.md` - AI assistant instructions
- `CONTRIBUTING.md` - Contribution guidelines

#### Package Configuration
- `pyproject.toml` - Python package configuration (primary)
- `setup.py` - Legacy Python setup (if needed)
- `setup.cfg` - Additional setup configuration
- `MANIFEST.in` - Package manifest

#### Requirements Files
- `requirements.txt` - Core dependencies
- `requirements-minimal.txt` - Minimal installation
- `requirements-server.txt` - Server/API dependencies
- `requirements-production.txt` - Production dependencies
- `requirements-test.txt` - Testing dependencies

#### Build & Development Tools
- `Makefile` - Build automation
- `package.json` - Node.js dependencies (for web UI)
- `package-lock.json` - Node.js lock file
- `.gitignore` - Git ignore patterns
- `.gitattributes` - Git attributes
- `.pre-commit-config.yaml` - Pre-commit hooks

#### IDE Configuration (Optional)
- `Pynomaly.code-workspace` - VS Code workspace

### Source Code (/src)
```
src/
├── pynomaly/                    # Main package
│   ├── domain/                 # Business logic (no external deps)
│   ├── application/            # Use cases and services
│   ├── infrastructure/         # External integrations
│   ├── presentation/           # APIs, CLI, Web UI
│   └── shared/                 # Common utilities
└── storage/                    # Runtime data storage
```

### Testing (/tests)
```
tests/
├── unit/                       # Unit tests
├── integration/                # Integration tests
├── e2e/                       # End-to-end tests
├── performance/               # Performance tests
├── security/                  # Security tests
├── conftest.py               # Pytest configuration
├── pytest.ini               # Pytest settings
└── reports/                  # Test reports and results
```

### Documentation (/docs)
```
docs/
├── api/                      # API documentation
├── architecture/             # Architecture decisions and diagrams
├── deployment/               # Deployment guides
├── development/              # Development guides
├── guides/                   # User guides
├── tutorials/                # Step-by-step tutorials
├── reference/                # Reference documentation
└── index.md                  # Documentation index
```

### Scripts (/scripts)
```
scripts/
├── development/              # Development utilities
├── deployment/               # Deployment scripts
├── testing/                  # Test automation scripts
├── maintenance/              # Maintenance and cleanup
└── setup/                    # Installation and setup scripts
```

### Examples (/examples)
```
examples/
├── basic/                    # Basic usage examples
├── advanced/                 # Advanced usage examples
├── notebooks/                # Jupyter notebooks
├── datasets/                 # Sample datasets
└── configs/                  # Example configurations
```

### Deployment (/deploy)
```
deploy/
├── docker/                   # Docker files and configs
├── kubernetes/               # Kubernetes manifests
├── terraform/                # Infrastructure as code
└── artifacts/                # Build artifacts
```

### Configuration (/config)
```
config/
├── development/              # Development configurations
├── production/               # Production configurations
├── testing/                  # Testing configurations
└── templates/                # Configuration templates
```

### Reports (/reports)
```
reports/
├── coverage/                 # Test coverage reports
├── security/                 # Security audit reports
├── performance/              # Performance benchmarks
└── quality/                  # Code quality reports
```

### Templates (/templates)
```
templates/
├── scripts/                  # Script templates
├── documentation/            # Documentation templates
├── testing/                  # Testing templates
└── experiments/              # Experiment templates
```

## 🚫 Prohibited in Root Directory

### Files That Must NOT Be in Root
- `test_*.py` - Testing files → `tests/`
- `test_*.sh` - Testing scripts → `tests/`
- `test_*.ps1` - PowerShell test scripts → `tests/`
- `*_test.py` - Test files → `tests/`
- `fix_*.py` - Fix scripts → `scripts/`
- `setup_*.py` - Setup scripts → `scripts/`
- `run_*.py` - Run scripts → `scripts/`
- `*_REPORT.md` - Reports → `reports/` or `docs/`
- `*_SUMMARY.md` - Summaries → `docs/`
- `*_GUIDE.md` - Guides → `docs/guides/`
- `TESTING_*.md` - Testing docs → `tests/` or `docs/`
- `*.backup` - Backup files → DELETE
- `*.tmp` - Temporary files → DELETE
- `=*` - Version artifacts → DELETE
- Virtual environments → DELETE

### Directories That Must NOT Be in Root
- `test_*` - Testing directories → `tests/`
- `venv*` - Virtual environments → DELETE
- `.venv*` - Virtual environments → DELETE
- `env*` - Environment directories → DELETE
- `temp*` - Temporary directories → DELETE
- `backup*` - Backup directories → DELETE
- `scratch*` - Scratch directories → DELETE

## 📋 File Naming Conventions

### General Rules
- Use lowercase with underscores: `my_script.py`
- Be descriptive: `analyze_anomalies.py` not `analyze.py`
- Include purpose: `test_integration_api.py`

### Specific Patterns
- **Tests**: `test_<module>_<function>.py`
- **Scripts**: `<action>_<purpose>.py` (e.g., `setup_database.py`)
- **Documentation**: `<TOPIC>_<TYPE>.md` (e.g., `API_REFERENCE.md`)
- **Configuration**: `<environment>_<service>.yaml`

## 🔧 Enforcement Mechanisms

### 1. .gitignore Rules
Comprehensive patterns prevent stray files from being committed:
```gitignore
# Testing files belong in tests/
/test_*.py
/test_*.sh
/*_test.py

# Scripts belong in scripts/
/fix_*.py
/setup_*.py
/run_*.py
```

### 2. Pre-commit Hooks
Automated validation before commits:
- File location validation
- Naming convention checks
- Directory structure verification

### 3. CI/CD Validation
GitHub Actions workflow validates:
- Repository structure compliance
- File organization standards
- Automated cleanup suggestions

### 4. Development Tools
- `scripts/organize_files.py` - Automated file organization
- `scripts/validate_structure.py` - Structure validation
- `scripts/cleanup_repository.py` - Repository cleanup

## 🛠️ File Organization Commands

### Manual Organization
```bash
# Analyze current structure
python scripts/analyze_project_structure.py

# Organize files automatically
python scripts/organize_files.py --dry-run
python scripts/organize_files.py --execute

# Validate structure
python scripts/validate_structure.py
```

### Pre-commit Validation
```bash
# Install pre-commit hooks
pre-commit install

# Run manual validation
pre-commit run file-organization --all-files
```

## 📊 Structure Validation

### Automated Checks
1. **Root Directory Compliance**: Only allowed files in root
2. **Naming Convention Adherence**: Proper file naming
3. **Logical Categorization**: Files in correct directories
4. **Dependency Isolation**: No circular dependencies
5. **Documentation Completeness**: Required docs present

### Quality Gates
- **Pre-commit**: Basic validation and auto-fixes
- **CI/CD Pipeline**: Comprehensive structure validation
- **Release Process**: Final compliance verification

## 🔄 Migration Process

### For Existing Files
1. **Identification**: Use analysis script to identify stray files
2. **Categorization**: Determine appropriate target directory
3. **Migration**: Move files with history preservation
4. **Validation**: Verify new structure compliance
5. **Cleanup**: Remove temporary and obsolete files

### Migration Script
```bash
# Generate migration plan
python scripts/analyze_project_structure.py

# Execute migration
python scripts/migrate_files.py --from-analysis reports/project_structure_analysis.json
```

## 📈 Benefits

### For Developers
- **Faster Navigation**: Predictable file locations
- **Reduced Cognitive Load**: Clear organization patterns
- **Better Collaboration**: Consistent structure for all contributors

### for Project Management
- **Professional Appearance**: Clean, organized repository
- **Easier Maintenance**: Automated organization and validation
- **Scalability**: Structure supports project growth

### For Documentation
- **Logical Documentation Structure**: Docs organized by purpose
- **Easy Reference**: Predictable locations for guides and references
- **Comprehensive Coverage**: Required documentation enforced

## ⚠️ Violations and Remediation

### Common Violations
1. **Test files in root** → Move to `tests/`
2. **Scripts in root** → Move to `scripts/`
3. **Documentation in root** → Move to `docs/`
4. **Temporary files committed** → Delete and add to .gitignore
5. **Multiple virtual environments** → Delete, use single .venv

### Remediation Process
1. **Detection**: Automated via pre-commit or CI/CD
2. **Notification**: Clear error messages with guidance
3. **Auto-fix**: Automated resolution where possible
4. **Manual Review**: Complex cases require developer intervention

## 🎯 Compliance Checklist

- [ ] Only essential files in project root
- [ ] All tests in `tests/` directory
- [ ] All scripts in `scripts/` directory
- [ ] All documentation in `docs/` directory
- [ ] No temporary files committed
- [ ] No virtual environments in repository
- [ ] Consistent naming conventions
- [ ] Proper directory structure
- [ ] .gitignore patterns updated
- [ ] Pre-commit hooks installed

## 📚 Related Documentation

- [Contributing Guidelines](../CONTRIBUTING.md)
- [Development Setup](./development-setup.md)
- [CI/CD Pipeline](./cicd-pipeline.md)
- [Code Quality Standards](./code-quality.md)