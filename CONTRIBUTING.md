# Contributing to Pynomaly
## Container Environment Validation

Before starting development, ensure your container environment is properly configured:

```bash
# Verify Docker is installed and running
docker --version
docker-compose --version

# Test container connectivity
make dev-clean  # Clean any existing containers
make dev        # Start development environment
```

**Prerequisites:**
- Docker Engine 20.10+ 
- Docker Compose 2.0+
- Make utility (cross-platform)

## Environment Recreation

**All development commands must be executed within containers.** This ensures consistency across all development environments and prevents local environment conflicts.

### Container-Based Development Setup

```bash
# Quick setup - all dependencies installed in container
make dev-install

# Start development environment (recommended)
make dev-shell

# Alternative: Start full development stack
make dev
```

### Legacy Installation (Not Recommended)

*Note: Direct installation is maintained for legacy support only. Use containers for all development work.*

```bash
# Install Hatch (legacy)
pip install hatch
hatch env create

# Or using Poetry (legacy support)
poetry install
```

**Important:** The container environment is the only supported development method. Local installations may cause inconsistencies.

## Build Hook Safety

Build hooks are configured with `pre-commit` to ensure code quality and secure development practices. Run the following command to set up the pre-commit hooks:

```bash
pre-commit install
```

You can check all hooks by running:

```bash
pre-commit run --all-files
```

## Branch Compliance and CI/CD

### Branch Naming Convention

All branches must follow the established naming convention to pass CI/CD validation:

```bash
# Valid branch name formats:
feature/your-feature-name
bugfix/issue-description
hotfix/critical-fix
release/version-number
chore/maintenance-task
docs/documentation-update

# Protected branches (PR-only):
main
develop
```

### Branch Validation

The CI/CD pipeline includes automatic branch compliance checks:

```bash
# Validate current branch name
make branch-validate

# Create new branch with validation
make branch-new TYPE=feature NAME=my-feature

# Switch branches safely
make branch-switch NAME=feature/my-feature
```

### CI/CD Pipeline Integration

- **Branch Compliance Job**: Runs first in CI pipeline
- **Direct Push Protection**: Prevents direct pushes to `main`/`develop`
- **Automated Validation**: Uses `make branch-validate` command
- **Early Failure**: Stops pipeline if branch name is invalid

**Note**: All other CI jobs depend on branch compliance passing first.

## Running Tests

**All tests must be executed within containers** to ensure consistency and isolation. The test environment is pre-configured with all dependencies and test databases.

### Container-Based Testing

```bash
# Run all tests in container environment
make dev-test

# Run tests with coverage report
make test-cov

# Run specific test suites
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test              # Core tests (domain + application)
```

### Interactive Testing

```bash
# Start test environment shell
make dev-shell

# Within container shell:
pytest tests/unit/ -v
pytest tests/integration/ -v --tb=short
pytest tests/domain/ tests/application/ -v
```

### Legacy Testing (Not Recommended)

*Note: Direct test execution is maintained for legacy support only. Use containers for all testing.*

```bash
# Using Hatch (legacy)
hatch env run test:run
hatch env run test:run-cov

# Specific test types (legacy)
hatch env run test:run tests/unit/
hatch env run test:run tests/integration/
```

## Recommended PowerShell Commands

The following PowerShell commands are provided to help manage development tasks:

- **Build the project**:
  ```powershell
  Build
  ```

- **Clean build artifacts**:
  ```powershell
  Clean
  ```

- **Start development server**:
  ```powershell
  Start-Dev
  ```

Ensure you have imported `tasks.ps1` in your PowerShell session like so:

```powershell
. .\tasks.ps1
```
## Code Style

**All code formatting and linting must be performed within containers** to ensure consistency across development environments.

### Container-Based Code Quality

```bash
# Format code automatically
make format

# Run all code quality checks
make lint

# Individual quality checks
make style    # Check code style without fixing
make typing   # Run type checking with mypy
```

### Interactive Code Quality

```bash
# Start development shell for manual formatting
make dev-shell

# Within container shell:
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/
```

### Pre-commit Hooks (Container-based)

```bash
# Install pre-commit hooks (runs in container)
make pre-commit

# Run pre-commit manually
pre-commit run --all-files
```

### Legacy Code Quality (Not Recommended)

*Note: Direct execution is maintained for legacy support only. Use containers for all code quality checks.*

```bash
# Using Hatch (legacy)
hatch env run lint:fmt
hatch env run lint:style
hatch env run lint:typing
```

## Development Setup

**All development must be performed within containers.** This ensures consistent environments and eliminates "works on my machine" issues.

## PowerShell Build Tasks

For Windows users, we provide a `tasks.ps1` PowerShell script that proxies common make targets. **All PowerShell tasks execute within containers** to ensure consistency.

### Container-Based PowerShell Workflow

```powershell
# Import the container-aware tasks
. .\tasks.ps1

# Available container-based tasks:
Build        # Build project in container
Clean        # Clean artifacts in container
Start-Dev    # Start development server in container
Test         # Run tests in container
Format       # Format code in container
Lint         # Run quality checks in container
```

### Available Tasks

- **Build**: Build the project using containerized environment
  ```powershell
  Build
  ```
  This runs `make build` which executes the build process in a container.

- **Clean**: Remove build artifacts using containerized cleanup
  ```powershell
  Clean
  ```
  This runs `make clean` which removes artifacts within the container environment.

- **Start-Dev**: Start the development server in container
  ```powershell
  Start-Dev
  ```
  This runs `make dev` which starts the containerized development environment.

## Maintenance Guidelines

### Adding Optional Dependencies

When adding new optional dependencies, always implement graceful fallbacks to prevent breaking existing functionality:

```python
try:
    import heavy_library
    HAS_HEAVY_LIBRARY = True
except ImportError:
    HAS_HEAVY_LIBRARY = False
    heavy_library = None

def feature_requiring_heavy_library():
    if not HAS_HEAVY_LIBRARY:
        raise ImportError(
            "heavy_library is required for this feature. "
            "Install it with: pip install heavy_library"
        )
    return heavy_library.do_something()
```

### Exposing Use-Case Aliases

To make the library more user-friendly, expose common use-case aliases in the main module:

```python
# In __init__.py
from .core import ComplexClassName
from .utils import helper_function

# Create user-friendly aliases
Detector = ComplexClassName  # Common use case alias
detect_anomalies = helper_function  # Simplified function name
```

### Creating Stub Modules for Testing

For heavy libraries in tests, create stub modules to improve test performance and reduce dependencies:

```python
# In tests/stubs/heavy_library.py
class MockHeavyClass:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    def expensive_method(self):
        return "mocked_result"

# In test files
import sys
from unittest.mock import MagicMock

# Mock the heavy library before importing your module
sys.modules['heavy_library'] = MagicMock()
sys.modules['heavy_library'].HeavyClass = MockHeavyClass

# Now import and test your module
from pynomaly import your_module
```

These practices help maintain backward compatibility and reduce test execution time while preventing future breakage.

## Example Workflow

### Container-Based Development Workflow

```bash
# 1. Container Environment Validation
make dev-clean
make dev

# 2. Start development shell
make dev-shell

# 3. Run tests in container
make dev-test

# 4. Format and lint code in container
make format
make lint

# 5. Build project in container
make build
```

### PowerShell Container Workflow

```powershell
# Import the container-aware tasks
. .\tasks.ps1

# Clean previous builds (containerized)
Clean

# Build the project (containerized)
Build

# Start development server (containerized)
Start-Dev
```

### Daily Development Routine

```bash
# Start your development session
make dev-shell

# Within container shell:
# - Edit code
# - Run tests: pytest tests/unit/ -v
# - Format code: black src/ tests/
# - Check style: ruff check src/ tests/

# Exit container and run full checks
exit
make ci  # Run full CI pipeline locally
```
