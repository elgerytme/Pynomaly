# Contributing to Pynomaly

We welcome contributions to Pynomaly! This document outlines the process for contributing to the project.

## Git Workflow

### Branch Naming Convention

We follow a structured branch naming convention based on the type of work being done:

- **Feature branches for tests**: `feat/tests/<module>-<description>`
  - Example: `feat/tests/domain-entity-validation`
  - Example: `feat/tests/api-endpoint-coverage`
  - Example: `feat/tests/infrastructure-persistence`

- **General feature branches**: `feat/<description>`
  - Example: `feat/anomaly-detection-algorithms`
  - Example: `feat/web-interface-improvements`

- **Bug fixes**: `fix/<description>`
  - Example: `fix/memory-leak-training-service`
  - Example: `fix/api-authentication-issues`

- **Documentation**: `docs/<description>`
  - Example: `docs/api-reference-update`
  - Example: `docs/user-guide-improvements`

- **Chores and maintenance**: `chore/<description>`
  - Example: `chore/dependency-updates`
  - Example: `chore/code-cleanup`

### Rebase Policy

We use a **rebase-first** policy to maintain a clean commit history:

1. **Before pushing**: Always rebase your feature branch on the latest `main` branch
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Interactive rebase**: Clean up your commit history before submitting a PR
   ```bash
   git rebase -i HEAD~n  # where n is the number of commits to review
   ```

3. **Squash commits**: Related commits should be squashed into logical units
   - Each commit should represent a complete, working change
   - Commit messages should be descriptive and follow [Conventional Commits](https://www.conventionalcommits.org/)

4. **Force push carefully**: When rebasing, use `--force-with-lease` to safely update your branch
   ```bash
   git push --force-with-lease origin your-branch-name
   ```

### CI Requirements

**All pull requests must have 100% passing tests and ≥95% code coverage before merge.**

#### Required CI Checks

Before any PR can be merged to `main`, the following checks must pass:

1. **✅ Build & Package** - Successful Hatch build and package creation
2. **✅ Test Suite** - All unit and integration tests pass (100% pass rate required)
3. **✅ Code Coverage** - Minimum 95% code coverage across all modules
4. **✅ Code Quality** - Linting, formatting, and type checking pass
5. **✅ Security Scan** - No high-severity security issues found
6. **✅ Documentation** - All documentation builds successfully

#### Coverage Requirements by Module

- **Domain Layer**: ≥98% coverage
- **Application Layer**: ≥95% coverage  
- **Infrastructure Layer**: ≥90% coverage
- **Presentation Layer**: ≥85% coverage
- **Overall Project**: ≥95% coverage

#### Continuous Integration Pipeline

Our CI pipeline runs automatically on:
- Every push to feature branches
- Every pull request to `main`
- Daily scheduled runs at 2 AM UTC

The pipeline includes:
- Multi-platform testing (Ubuntu, Windows, macOS)
- Multi-version Python testing (3.11, 3.12, 3.13)
- Performance benchmarking
- Security scanning
- Documentation validation

## Development Process

### 1. Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/elgerytme/Pynomaly.git
cd Pynomaly

# Install Hatch (our build system)
pip install hatch

# Create and activate development environment
hatch env create
hatch shell

# Install pre-commit hooks
pre-commit install
```

### 2. Creating a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create new feature branch
git checkout -b feat/tests/your-module-description

# Make your changes
# ... develop, test, commit ...

# Rebase before pushing
git fetch origin
git rebase origin/main

# Push feature branch
git push origin feat/tests/your-module-description
```

### 3. Testing Your Changes

```bash
# Run all tests
hatch run test:run

# Run tests with coverage
hatch run test:run-cov

# Run specific test modules
hatch run test:run tests/domain/
hatch run test:run tests/application/

# Run linting and formatting
hatch run lint:style
hatch run lint:typing
hatch run lint:fmt
```

### 4. Submitting a Pull Request

1. **Ensure your branch is up to date**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Verify all tests pass locally**:
   ```bash
   hatch run test:run-cov
   ```

3. **Push your changes**:
   ```bash
   git push origin feat/tests/your-module-description
   ```

4. **Create a pull request** on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable
   - Confirmation that all CI checks pass

### 5. Code Review Process

- All PRs require at least one approved review
- PR author should respond to review comments promptly
- Use GitHub's suggestion feature for minor fixes
- Re-request review after addressing feedback

### 6. Merge Requirements

**Before merge, ensure:**
- ✅ All CI checks pass (100% test pass rate)
- ✅ Code coverage ≥95%
- ✅ At least one approved review
- ✅ Branch is up to date with main
- ✅ No merge conflicts
- ✅ All conversations resolved

## Code Standards

### Python Code Quality

We maintain high code quality standards:

- **Type hints**: All functions must have proper type annotations
- **Documentation**: All public functions/classes must have docstrings
- **Testing**: All new code must have comprehensive tests
- **Linting**: Code must pass Ruff linting checks
- **Formatting**: Code must be formatted with Ruff formatter

### Test Writing Guidelines

1. **Test Coverage**: Aim for 100% line coverage on new code
2. **Test Types**:
   - Unit tests for individual functions/classes
   - Integration tests for component interactions
   - End-to-end tests for complete workflows
3. **Test Structure**: Follow the Arrange-Act-Assert pattern
4. **Test Naming**: Use descriptive test names that explain what is being tested

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or modifying tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```
feat(domain): add anomaly detection algorithm validation
fix(api): resolve authentication timeout issues
test(infrastructure): add persistence layer integration tests
docs(contributing): update branch naming conventions
```

## Getting Help

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Check the `docs/` directory for detailed guides
- **CI Failures**: Check the Actions tab for detailed error logs

## Branch Protection

The `main` branch is protected with the following rules:
- Require pull request reviews before merging
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Require 100% passing tests
- Require ≥95% code coverage
- Restrict pushes that create merge commits

Thank you for contributing to Pynomaly! 🚀
