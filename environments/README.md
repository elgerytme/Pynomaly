# Python Environments Directory

This directory contains all Python virtual environments for the Pynomaly project. All environments follow the dot-prefix naming convention to distinguish them from regular directories and to ensure they are hidden from most tools and listings.

## 🏗️ Directory Structure

```
environments/
├── README.md              # This documentation file
├── .venv/                 # Main development environment (Python 3.11+)
├── .venv_testing/         # Testing environment with extended dependencies
├── .test_env/             # Minimal test environment
├── .test_env_check/       # Environment validation testing
├── .test_venv/            # General testing virtual environment
├── .test_venv_bash/       # Bash scripting test environment
├── .test_venv_fresh/      # Fresh installation testing environment
└── .test_environments/    # Multi-environment testing directory
```

## 🎯 Environment Purposes

### Development Environments
- **`.venv/`** - Primary development environment with full dependencies
  - Used for day-to-day development work
  - Contains all core and optional dependencies
  - Python 3.11+ with Poetry management

### Testing Environments  
- **`.venv_testing/`** - Extended testing environment
  - Additional testing frameworks and tools
  - Performance testing dependencies
  - UI automation dependencies

- **`.test_env/`** - Minimal testing environment
  - Core dependencies only
  - Used for basic functionality validation

- **`.test_env_check/`** - Environment validation
  - Used to test installation procedures
  - Validates package integrity

- **`.test_venv/`** - General testing virtual environment
  - Cross-platform testing scenarios

- **`.test_venv_bash/`** - Bash-specific testing
  - Shell script testing environment
  - Unix/Linux environment validation

- **`.test_venv_fresh/`** - Fresh installation testing
  - Clean environment for testing setup procedures
  - Installation validation

- **`.test_environments/`** - Multi-environment testing
  - Contains multiple sub-environments for comprehensive testing

## 📋 Project Rules and Conventions

### Environment Naming Convention
**RULE**: All Python environments MUST use dot-prefix naming (`.env_name`)

✅ **Correct Examples**:
- `.venv`
- `.venv_testing`
- `.test_env`
- `.dev_environment`

❌ **Incorrect Examples**:
- `venv` (no dot prefix)
- `test_env` (no dot prefix)  
- `my_environment` (no dot prefix)

### Environment Location
**RULE**: All environments MUST be placed in the `environments/` directory

✅ **Correct Location**: `environments/.venv/`
❌ **Incorrect Location**: `.venv/` (project root)

### Git Management
- All environment directories are automatically excluded from Git (except this README.md)
- Dot-prefixed directories are inherently treated as hidden
- Environment contents should never be committed to version control

### Tool Integration
All code quality tools, formatters, and static analysis tools are configured to ignore environment directories:
- **Formatters**: Black, YAPF, isort
- **Linters**: Ruff, Flake8, Pylint, PyLyzer  
- **Type Checkers**: MyPy, PyRight
- **Security**: Bandit, Safety
- **Coverage**: pytest-cov, coverage.py

## 🚀 Usage Guidelines

### Creating New Environments
When creating a new environment, always:

1. **Use dot-prefix naming**:
   ```bash
   python -m venv environments/.my_new_env
   ```

2. **Document the purpose**: Add entry to this README
3. **Update tool configurations** if needed

### Activating Environments
```bash
# Development environment
source environments/.venv/bin/activate  # Linux/macOS
# or
environments\.venv\Scripts\activate     # Windows

# Testing environment  
source environments/.venv_testing/bin/activate  # Linux/macOS
# or
environments\.venv_testing\Scripts\activate     # Windows
```

### Environment Management Commands
```bash
# List all environments
ls -la environments/

# Check active environment
which python

# Deactivate any environment
deactivate

# Remove environment (be careful!)
rm -rf environments/.old_env_name
```

## 🔧 Tool Configuration

### Poetry Integration
Poetry is configured to create environments in this directory:
```toml
[tool.poetry]
# Poetry will respect the environments/ directory structure
```

### IDE Integration
Configure your IDE to recognize environment locations:
- **VS Code**: Update `python.pythonPath` settings
- **PyCharm**: Configure Python interpreter paths
- **Vim/Neovim**: Update environment detection plugins

## 🚨 Important Notes

### Never Commit Environments
- Virtual environments contain platform-specific binaries
- Large file sizes impact repository performance  
- Environment contents change frequently
- Use `requirements.txt` or `pyproject.toml` for dependency tracking

### Cross-Platform Compatibility
- Environment activation scripts differ between Windows/Unix
- Use appropriate activation commands for your platform
- Test installation procedures on multiple platforms when possible

### Cleanup Recommendations
- Remove unused environments periodically
- Recreate environments when dependency conflicts arise
- Use fresh environments for release testing

## 📚 Related Documentation

- [Installation Guide](../docs/getting-started/installation.md)
- [Development Setup](../docs/development/README.md)
- [Testing Procedures](../docs/development/COMPREHENSIVE_TEST_ANALYSIS.md)
- [Poetry Guide](../docs/development/HATCH_GUIDE.md)

## 🔄 Migration History

This directory was created to consolidate all Python environments from the project root directory. Previous environment locations have been moved and renamed according to the new conventions.

**Migration Date**: Current session
**Previous Locations**: Various root-level directories
**New Convention**: Centralized in `environments/` with dot-prefix naming
