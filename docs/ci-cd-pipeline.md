# Enhanced CI/CD Pipeline Documentation

This document describes the comprehensive CI/CD pipeline setup for the Pynomaly project, featuring matrix testing, Hatch environment management, and production-ready deployment workflows.

## 🚀 Pipeline Overview

The enhanced CI/CD pipeline (`enhanced-cicd.yml`) provides:

- **Matrix Testing**: Python 3.11 & 3.12 compatibility testing
- **Hatch Environment Management**: Modern Python project tooling
- **Coverage Reporting**: Codecov integration
- **Production Smoke Tests**: API testing with `uvicorn --workers 4`
- **Documentation Deployment**: Automatic GitHub Pages deployment
- **Security Scanning**: Bandit and Safety security analysis
- **Comprehensive Caching**: Faster builds with environment caching

## 📋 Pipeline Jobs

### 1. Lint & Quality (`lint-and-quality`)
- **Purpose**: Code quality checks and linting
- **Matrix**: Python 3.11, 3.12
- **Command**: `hatch run lint:all`
- **Features**:
  - Ruff linting
  - Black code formatting
  - isort import sorting
  - MyPy type checking
  - Environment caching for faster builds

### 2. Test Suite (`test-suite`)
- **Purpose**: Run comprehensive test suite with coverage
- **Matrix**: Python 3.11, 3.12
- **Command**: `hatch run test:run-cov`
- **Features**:
  - Full test suite execution
  - Coverage report generation
  - Codecov upload
  - Test result artifacts

### 3. Build Documentation (`build-docs`)
- **Purpose**: Build MkDocs documentation
- **Command**: `hatch run docs:build`
- **Features**:
  - MkDocs build with Material theme
  - GitHub Pages artifact generation
  - Documentation link validation

### 4. Deploy Documentation (`deploy-docs`)
- **Purpose**: Deploy docs to GitHub Pages
- **Trigger**: Only on `main` branch push
- **Features**:
  - Automatic GitHub Pages deployment
  - Environment protection

### 5. Production Smoke Tests (`production-smoke-tests`)
- **Purpose**: Test API in production-like environment
- **Command**: `hatch run prod:uvicorn --workers 4`
- **Features**:
  - Multi-worker API server testing
  - Health check validation
  - API endpoint testing
  - Metrics endpoint validation

### 6. Build Package (`build-package`)
- **Purpose**: Build Python package
- **Command**: `hatch build --clean`
- **Features**:
  - Wheel and source distribution creation
  - Installation testing
  - Build artifact upload

### 7. Security Scan (`security-scan`)
- **Purpose**: Security vulnerability scanning
- **Tools**: Bandit, Safety
- **Features**:
  - Source code security analysis
  - Dependency vulnerability checking
  - Security report generation

### 8. Pipeline Summary (`pipeline-summary`)
- **Purpose**: Generate comprehensive pipeline report
- **Features**:
  - Job status summary
  - PR comment generation
  - Pipeline artifact upload
  - Exit code determination

## 🔧 Required Hatch Environment Configuration

The pipeline expects the following Hatch environments to be configured in `pyproject.toml`:

```toml
[tool.hatch.envs.lint]
dependencies = [
    "ruff>=0.8.0",
    "black>=24.0.0",
    "isort>=5.13.0",
    "mypy>=1.13.0",
]

[tool.hatch.envs.lint.scripts]
all = [
    "ruff check {args:.}",
    "black --check --diff {args:.}",
    "isort --check-only --diff {args:.}",
    "mypy {args:src/pynomaly tests}",
]

[tool.hatch.envs.test]
dependencies = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-xdist>=3.6.0",
]

[tool.hatch.envs.test.scripts]
run-cov = "pytest --cov=pynomaly --cov-report=html --cov-report=xml {args:tests}"

[tool.hatch.envs.docs]
dependencies = [
    "mkdocs>=1.6.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.27.0",
]

[tool.hatch.envs.docs.scripts]
build = "mkdocs build --clean --strict"

[tool.hatch.envs.prod]
extra-dependencies = ["pynomaly[production]"]

[tool.hatch.envs.prod.scripts]
serve-api = "uvicorn pynomaly.presentation.api.app:app --host 0.0.0.0 --port 8000"
```

## 🔐 Required GitHub Secrets

Configure these secrets in your GitHub repository:

```bash
# Codecov token for coverage reporting
CODECOV_TOKEN=your_codecov_token_here

# Optional: Slack webhook for notifications
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
```

## 📊 Matrix Testing Details

The pipeline runs tests across multiple Python versions:

### Python Version Matrix
- **3.11**: Primary development version
- **3.12**: Latest stable version

### Matrix Benefits
- **Compatibility**: Ensures code works across Python versions
- **Early Detection**: Catches version-specific issues
- **Confidence**: Higher reliability for production deployments

## 🏭 Production Smoke Tests

The production smoke tests simulate a real production environment:

```bash
# API server startup
hatch run prod:uvicorn pynomaly.presentation.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Health checks
curl -f http://localhost:8000/health
curl -f http://localhost:8000/docs
curl -f http://localhost:8000/api/v1/health
curl -f http://localhost:8000/metrics
```

### Test Coverage
- **Health Endpoints**: Basic service health
- **API Documentation**: OpenAPI/Swagger docs
- **Core API**: Main API endpoints
- **Metrics**: Monitoring endpoints

## 🎯 Performance Optimizations

### Environment Caching
```yaml
- name: Cache Hatch environments
  uses: actions/cache@v4
  with:
    path: |
      ~/.local/share/hatch
      ~/.cache/hatch
    key: hatch-${{ runner.os }}-py${{ matrix.python-version }}-${{ hashFiles('**/pyproject.toml') }}
```

### Concurrency Control
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

### Parallel Execution
- Matrix jobs run in parallel
- Independent job execution
- Optimized dependency chains

## 📈 Artifacts Generated

### Quality Reports
- **Location**: `quality-reports-py{version}`
- **Content**: Code quality analysis
- **Retention**: 30 days

### Test Results
- **Location**: `test-results-py{version}`
- **Content**: Coverage reports, test outputs
- **Retention**: 30 days

### Documentation
- **Location**: `documentation-build`
- **Content**: Built documentation site
- **Retention**: 30 days

### Build Artifacts
- **Location**: `build-artifacts`
- **Content**: Python wheels and source distributions
- **Retention**: 30 days

### Security Reports
- **Location**: `security-reports`
- **Content**: Bandit and Safety scan results
- **Retention**: 30 days

### Pipeline Summary
- **Location**: `pipeline-summary-report`
- **Content**: Comprehensive pipeline analysis
- **Retention**: 90 days

## 🔄 Workflow Triggers

### Push Events
```yaml
on:
  push:
    branches: [main, develop]
```

### Pull Request Events
```yaml
on:
  pull_request:
    branches: [main, develop]
```

### Manual Trigger
```yaml
on:
  workflow_dispatch:
```

## 📝 PR Comments

The pipeline automatically generates PR comments with:
- Matrix testing results
- Job status summaries
- Pipeline features overview
- Artifact information

## 🛠️ Troubleshooting

### Common Issues

#### 1. Hatch Environment Not Found
```bash
Error: Environment 'lint' not found
```
**Solution**: Ensure all required Hatch environments are configured in `pyproject.toml`

#### 2. Coverage Upload Failed
```bash
Error: Failed to upload coverage to Codecov
```
**Solution**: Check `CODECOV_TOKEN` secret is configured correctly

#### 3. API Smoke Test Timeout
```bash
Error: API server failed to start within timeout
```
**Solution**: Check API dependencies are installed in `prod` environment

#### 4. Documentation Build Failed
```bash
Error: MkDocs build failed
```
**Solution**: Verify `mkdocs.yml` configuration and documentation source files

### Debug Steps

1. **Check Hatch Configuration**:
   ```bash
   hatch env show
   hatch version
   ```

2. **Verify Dependencies**:
   ```bash
   hatch run lint:python -m pip list
   hatch run test:python -m pip list
   ```

3. **Test Commands Locally**:
   ```bash
   hatch run lint:all
   hatch run test:run-cov
   hatch run docs:build
   ```

## 📚 Additional Resources

- [Hatch Documentation](https://hatch.pypa.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Codecov Documentation](https://docs.codecov.io/)
- [MkDocs Documentation](https://www.mkdocs.org/)

## 🔄 Pipeline Maintenance

### Regular Updates
- Update action versions monthly
- Review security scan results
- Monitor pipeline performance
- Update Python version matrix as needed

### Performance Monitoring
- Track pipeline execution times
- Monitor cache hit rates
- Review artifact sizes
- Optimize job dependencies

### Security Considerations
- Regular dependency updates
- Security scan threshold reviews
- Secret rotation
- Access control reviews

## 📋 Checklist for New Projects

- [ ] Configure Hatch environments in `pyproject.toml`
- [ ] Set up GitHub secrets (`CODECOV_TOKEN`)
- [ ] Create `mkdocs.yml` configuration
- [ ] Set up GitHub Pages deployment
- [ ] Configure branch protection rules
- [ ] Test pipeline on feature branch
- [ ] Verify all artifacts are generated
- [ ] Review security scan results
- [ ] Set up monitoring/notifications

This enhanced CI/CD pipeline provides a robust, scalable foundation for modern Python project development with comprehensive testing, quality assurance, and deployment capabilities.
