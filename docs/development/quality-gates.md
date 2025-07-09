# Quality Gates and Continuous Integration

This document describes the comprehensive quality gates and CI/CD pipeline setup for the Pynomaly project.

## Overview

The CI/CD pipeline enforces strict quality gates that must pass before any code can be merged into the main branches. This ensures consistent code quality, security, and maintainability across the project.

## Quality Gates

### 1. Linting & Formatting

**Tools Used:**
- **Ruff**: Fast Python linter and formatter
- **Black**: Python code formatter
- **MyPy**: Static type checker
- **isort**: Import statement organizer

**Requirements:**
- All ruff linting checks must pass
- Code must be formatted with both ruff and black
- All type annotations must pass mypy strict mode
- Import statements must be properly sorted

**Commands:**
```bash
# Run locally
hatch run lint:ruff check src/ tests/
hatch run lint:ruff format --check src/ tests/
hatch run lint:black --check --diff src/ tests/
hatch run lint:mypy src/pynomaly/ --strict
hatch run lint:isort --check-only --diff src/ tests/

# Auto-fix issues
hatch run lint:fmt
```

### 2. Unit Tests

**Requirements:**
- All unit tests must pass
- Minimum code coverage: 85%
- Tests must run on Python 3.11 and 3.12
- Coverage reports uploaded to Codecov

**Commands:**
```bash
# Run locally
hatch run test:run-cov --cov-report=term --cov-report=html
coverage report --show-missing --fail-under=85
```

### 3. Documentation Build

**Requirements:**
- MkDocs documentation must build successfully
- All documentation links must be valid
- No broken documentation references

**Commands:**
```bash
# Run locally
hatch run docs:build
python scripts/analysis/check_documentation_links.py
```

### 4. Security Scan

**Tools Used:**
- **Bandit**: Security linter for Python code
- **Safety**: Dependency vulnerability scanner

**Requirements:**
- No high-severity security issues from Bandit
- No known vulnerabilities in dependencies
- Security scan must pass completely

**Commands:**
```bash
# Run locally
bandit -r src/ -ll --severity-level medium
safety check
```

### 5. Build & Package

**Requirements:**
- Package must build successfully with Hatch
- Built package must be installable
- All build artifacts must be valid

**Commands:**
```bash
# Run locally
hatch build --clean
pip install dist/*.whl
python -c \"import pynomaly; print('Package works!')\"
```

### 6. Integration Tests

**Requirements:**
- Integration tests must pass
- CLI functionality must work
- API startup must succeed

**Commands:**
```bash
# Run locally
hatch run test:run tests/infrastructure/
hatch run cli:test-cli
```

## Workflows

### Primary Quality Gate Workflow

**File:** `.github/workflows/ci-enhanced.yml`

This is the main quality gate workflow that enforces all requirements:

```yaml
name: Enhanced CI with Quality Gates

on:
  pull_request:
    branches: [ main, develop ]
  push:
    branches: [ main, develop ]

jobs:
  lint-and-format:         # Quality Gate 1
  unit-tests:              # Quality Gate 2  
  documentation:           # Quality Gate 3
  security-scan:           # Quality Gate 4
  build-and-package:       # Quality Gate 5
  integration-tests:       # Quality Gate 6
  quality-gate-summary:    # Final gate evaluation
```

### Supporting Workflows

1. **`.github/workflows/quality.yml`** - Additional code quality checks
2. **`.github/workflows/quality-gates.yml`** - Comprehensive quality metrics
3. **`.github/workflows/test.yml`** - Extended test matrix
4. **`.github/workflows/ci.yml`** - Original CI pipeline

## Branch Protection

### Configuration

Branch protection settings are configured to require:

1. **Required Status Checks:**
   - Quality Gate: Linting & Formatting
   - Quality Gate: Unit Tests
   - Quality Gate: Documentation
   - Quality Gate: Security Scan
   - Quality Gate: Build & Package
   - Quality Gate: Integration Tests
   - 📊 Quality Gate Summary

2. **Pull Request Requirements:**
   - At least 1 approving review
   - Dismiss stale reviews when new commits are pushed
   - Require code owner reviews (for main branch)
   - Require up-to-date branches before merging

3. **Additional Protections:**
   - Enforce rules for administrators
   - Require linear history (no merge commits)
   - Prevent force pushes to main
   - Require conversation resolution

### Setup Instructions

#### Manual Setup (GitHub UI)

1. Go to repository Settings → Branches
2. Add rule for \"main\" branch
3. Configure protection settings:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1)
   - ✅ Dismiss stale reviews
   - ✅ Require review from code owners
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date
   - ✅ Require linear history
   - ✅ Require conversation resolution
   - ✅ Include administrators

#### Automated Setup (GitHub API)

Use the provided script to configure branch protection:

```bash
# Set your GitHub token
export GITHUB_TOKEN=\"your_token_here\"

# Run the configuration script
python scripts/setup/configure_branch_protection.py

# Or with explicit token
python scripts/setup/configure_branch_protection.py --token YOUR_TOKEN
```

## Local Development

### Pre-commit Hooks

Set up pre-commit hooks to run quality checks locally:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Development Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Run quality checks locally:**
   ```bash
   # Lint and format
   hatch run lint:fmt
   
   # Run tests
   hatch run test:run-cov
   
   # Build documentation
   hatch run docs:build
   
   # Security scan
   bandit -r src/
   safety check
   ```

3. **Commit changes:**
   ```bash
   git add .
   git commit -m \"feat: add new feature\"
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/my-feature
   # Create PR via GitHub UI
   ```

5. **Monitor quality gates:**
   - Check CI status in PR
   - Address any failing quality gates
   - Request review once all gates pass

### Quality Gate Failures

If quality gates fail:

1. **Check the failing job in GitHub Actions**
2. **Run the same check locally:**
   ```bash
   # For linting issues
   hatch run lint:ruff check src/ tests/
   hatch run lint:black --check src/ tests/
   
   # For test failures
   hatch run test:run
   
   # For documentation issues
   hatch run docs:build
   ```

3. **Fix the issues:**
   ```bash
   # Auto-fix formatting
   hatch run lint:fmt
   
   # Fix tests, documentation, security issues manually
   ```

4. **Commit and push fixes:**
   ```bash
   git add .
   git commit -m \"fix: address quality gate failures\"
   git push
   ```

## Monitoring and Reporting

### Artifacts

The CI pipeline generates several artifacts:

- **Code Coverage Reports** (HTML and XML)
- **Security Scan Results** (JSON and SARIF)
- **Documentation Build** (Static site)
- **Quality Reports** (HTML summary)
- **Build Artifacts** (Wheel and source distributions)

### Codecov Integration

Code coverage is automatically uploaded to Codecov:

- **Coverage threshold:** 85%
- **Branch coverage:** Tracked
- **Pull request comments:** Enabled
- **Coverage diff:** Shown in PRs

### Security Scanning

Security issues are tracked in:

- **GitHub Security Tab** (SARIF uploads)
- **Bandit reports** (JSON and text)
- **Safety reports** (Dependency vulnerabilities)

## Customization

### Adjusting Thresholds

Edit workflow environment variables:

```yaml
env:
  MIN_COVERAGE: 85          # Minimum code coverage
  MAX_COMPLEXITY: 10        # Maximum cyclomatic complexity
  FAIL_ON_QUALITY_GATE: true  # Fail PR on quality issues
```

### Adding New Quality Gates

1. Add new job to `.github/workflows/ci-enhanced.yml`
2. Update branch protection requirements
3. Document the new gate in this file

### Tool Configuration

Tool configurations are in `pyproject.toml`:

```toml
[tool.ruff]
target-version = \"py311\"
line-length = 88

[tool.black]
target-version = [\"py311\"]
line-length = 88

[tool.mypy]
python_version = \"3.11\"
strict = true

[tool.coverage.run]
source = [\"src\"]
branch = true
```

## Troubleshooting

### Common Issues

1. **Quality gate timeout:**
   - Increase timeout in workflow
   - Optimize test performance
   - Run tests in parallel

2. **Coverage below threshold:**
   - Add more tests
   - Remove dead code
   - Exclude test files from coverage

3. **Type checking failures:**
   - Add type annotations
   - Update mypy configuration
   - Use type: ignore comments sparingly

4. **Security scan failures:**
   - Fix security issues immediately
   - Update dependencies
   - Add security exceptions if needed

### Getting Help

- **GitHub Issues:** Report problems
- **Documentation:** Check existing docs
- **CI Logs:** Review detailed failure logs
- **Local Testing:** Reproduce issues locally

## Best Practices

1. **Run quality checks locally** before pushing
2. **Keep PRs small** for easier review
3. **Write comprehensive tests** for new features
4. **Update documentation** with code changes
5. **Address security issues** immediately
6. **Monitor coverage trends** over time
7. **Review quality gate failures** thoroughly
8. **Use meaningful commit messages** for tracking

This quality gate system ensures that the Pynomaly project maintains high standards for code quality, security, and maintainability throughout its development lifecycle.
