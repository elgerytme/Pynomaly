# Package-Specific CI/CD System

## Overview

This directory contains a comprehensive CI/CD system designed for our monorepo architecture. Each package operates as an independent unit with its own testing, security scanning, and deployment pipelines while maintaining integration within the larger monorepo ecosystem.

## Architecture

### 🏗️ System Components

1. **Reusable Workflow Templates** (`_reusable-python-ci.yml`)
   - Common CI/CD patterns for Python packages
   - Standardized quality gates, testing, and deployment
   - Cross-platform testing (Ubuntu, Windows, macOS)
   - Security scanning and performance benchmarks

2. **Custom GitHub Actions** (`.github/actions/`)
   - `detect-package-changes/`: Smart package change detection
   - `setup-python-package/`: Optimized Python setup with caching
   - `security-scan/`: Comprehensive security scanning

3. **Package-Specific Workflows**
   - Each package has tailored CI/CD workflows
   - Package-specific testing strategies
   - Domain-specific validations

4. **Monorepo Orchestrator** (`monorepo-ci.yml`)
   - Central coordination of package workflows
   - Change detection and selective execution
   - Cross-package dependency management

## Package Workflows

### 📦 Supported Packages

| Package | Path | Workflow Features |
|---------|------|-------------------|
| **Anomaly Detection** | `data/anomaly_detection` | Algorithm validation, AutoML testing, performance benchmarks |
| **MLOps** | `ai/mlops` | API testing, container validation, cloud integration |
| **Infrastructure** | `ops/infrastructure` | Terraform validation, Kubernetes testing, security scanning |
| **Core Software** | `software/core` | Clean architecture validation, domain testing |
| **Interfaces** | `software/interfaces` | API/CLI testing, SDK validation, compatibility checks |
| **Data Observability** | `data/data_observability` | Data quality monitoring, lineage tracking |
| **Mathematics** | `formal_sciences/mathematics` | Mathematical correctness, precision testing, property-based tests |

### 🔄 Workflow Triggers

#### Automatic Triggers
- **Push to `main`/`develop`**: Runs CI for changed packages
- **Pull Requests**: Validates changes before merge
- **Scheduled Runs**: Weekly dependency updates and security scans

#### Manual Triggers
- **Workflow Dispatch**: Manual control with options:
  - Force run all packages
  - Select specific packages
  - Skip certain packages
  - Control test types (performance, integration, etc.)

## Features

### 🛡️ Security & Quality

#### Multi-Layer Security Scanning
- **Bandit**: Python code security analysis
- **Safety**: Dependency vulnerability scanning
- **pip-audit**: Advanced dependency auditing
- **Semgrep**: Static analysis with security patterns
- **Secret Detection**: Prevents credential leaks

#### Quality Gates
- **Code Coverage**: Package-specific thresholds (75%-95%)
- **Type Checking**: MyPy static analysis
- **Linting**: Ruff for code quality
- **Format Checking**: Black and isort

### 🧪 Testing Strategies

#### Comprehensive Test Coverage
- **Unit Tests**: Package-specific functionality
- **Integration Tests**: Service interactions
- **Property-Based Tests**: Mathematical correctness (Hypothesis)
- **Performance Tests**: Benchmark regression detection
- **Security Tests**: Vulnerability assessments

#### Package-Specific Testing
- **Anomaly Detection**: Algorithm accuracy, data handling
- **MLOps**: API endpoints, model lifecycle, monitoring
- **Infrastructure**: Terraform validation, container security
- **Mathematics**: Numerical precision, mathematical properties

### 📊 Performance & Monitoring

#### Performance Benchmarking
- **Automated Benchmarks**: Track performance regressions
- **Cross-Platform Testing**: Ensure consistent performance
- **Load Testing**: API and service capacity validation
- **Memory Profiling**: Resource usage optimization

#### Monitoring & Alerting
- **Build Status Tracking**: Real-time CI/CD monitoring
- **Failure Notifications**: Critical issue alerts
- **Performance Tracking**: Benchmark trend analysis
- **Security Alerts**: Vulnerability notifications

### 🚀 Deployment & Release

#### Automated Release Management
- **Version Management**: Semantic versioning
- **Release Notes**: Auto-generated from commits
- **PyPI Publishing**: Secure package distribution
- **Container Building**: Docker image creation
- **Deployment Validation**: Post-deployment checks

#### Environment Management
- **Development**: Continuous integration
- **Staging**: Pre-production validation
- **Production**: Controlled releases with rollback

## Usage

### 🎯 Running CI/CD

#### For Developers

```bash
# Trigger CI for all packages
gh workflow run monorepo-ci.yml

# Run specific packages
gh workflow run monorepo-ci.yml \
  -f packages_to_run="data/anomaly_detection,ai/mlops"

# Skip packages
gh workflow run monorepo-ci.yml \
  -f skip_packages="ops/infrastructure"

# Force all packages
gh workflow run monorepo-ci.yml \
  -f force_run_all=true
```

#### For Package Maintainers

```bash
# Run package-specific CI
cd src/packages/data/anomaly_detection
gh workflow run ci.yml

# Run with options
gh workflow run ci.yml \
  -f run_performance_tests=true \
  -f python_version=3.11
```

### 📋 Development Workflow

1. **Feature Development**
   ```bash
   # Create feature branch
   git checkout -b feature/new-algorithm
   
   # Make changes to package
   # CI automatically runs on push
   git push origin feature/new-algorithm
   ```

2. **Pull Request Process**
   - CI runs automatically for changed packages
   - Security scans and quality gates must pass
   - Performance benchmarks are compared
   - Manual review and approval required

3. **Release Process**
   ```bash
   # Create release
   gh release create v1.0.0 \
     --title "Anomaly Detection v1.0.0" \
     --notes "Major release with new algorithms"
   
   # Automated release workflow handles:
   # - PyPI publishing
   # - Container building
   # - Documentation updates
   ```

## Configuration

### 🔧 Package Configuration

Each package requires:

```toml
# pyproject.toml
[project]
name = "package-name"
version = "1.0.0"
dependencies = [...]

[project.optional-dependencies]
test = ["pytest>=8.0.0", "pytest-cov>=6.0.0"]
dev = ["ruff>=0.8.0", "mypy>=1.13.0"]
```

### ⚙️ Workflow Configuration

```yaml
# .github/workflows/ci.yml
env:
  PACKAGE_NAME: your-package
  PACKAGE_PATH: src/packages/your/package

jobs:
  ci-cd:
    uses: ./.github/workflows/_reusable-python-ci.yml
    with:
      package-name: your-package
      package-path: src/packages/your/package
      coverage-threshold: 85
```

### 🔒 Security Configuration

```yaml
# Security scanning options
- name: Run security scan
  uses: ./.github/actions/security-scan
  with:
    package-path: ${{ env.PACKAGE_PATH }}
    fail-on-high: true
    fail-on-medium: false
    upload-sarif: true
```

## Monitoring & Maintenance

### 📈 Metrics & Reporting

- **Build Success Rate**: Package CI/CD success metrics
- **Test Coverage**: Coverage trends and improvements
- **Performance Benchmarks**: Performance regression tracking
- **Security Scores**: Vulnerability and risk assessments

### 🔄 Maintenance Tasks

#### Weekly Automated Tasks
- Dependency updates with compatibility testing
- Security vulnerability scanning
- Performance benchmark baseline updates
- Documentation link validation

#### Monthly Manual Tasks
- Review and update CI/CD configurations
- Analyze performance trends and optimize
- Security audit and policy updates
- Package dependency cleanup

## Troubleshooting

### 🔍 Common Issues

#### CI Failures
```bash
# Check logs
gh run list --workflow=monorepo-ci.yml --limit=1
gh run view <run-id> --log

# Re-run failed jobs
gh run rerun <run-id> --failed
```

#### Package Detection Issues
```bash
# Manual package detection
gh workflow run monorepo-ci.yml \
  -f force_run_all=true
```

#### Security Scan Failures
```bash
# Review security reports
gh workflow run <package>-ci.yml
# Download artifacts from workflow run
```

### 🆘 Emergency Procedures

#### Critical Security Vulnerability
1. Stop all deployments
2. Run emergency security scan
3. Apply patches immediately
4. Validate fixes in staging
5. Deploy hotfix to production

#### CI/CD System Failure
1. Switch to manual deployment process
2. Investigate and fix CI/CD issues
3. Validate system with test packages
4. Resume automated operations

## Best Practices

### 📝 Development Guidelines

1. **Package Independence**: Each package should be deployable independently
2. **Test Coverage**: Maintain high test coverage (>80%)
3. **Security First**: Regular security scanning and updates
4. **Performance Monitoring**: Track and improve performance metrics
5. **Documentation**: Keep CI/CD documentation current

### 🔐 Security Guidelines

1. **Secret Management**: Use GitHub secrets for sensitive data
2. **Dependency Updates**: Regular automated dependency updates
3. **Vulnerability Response**: Quick response to security alerts
4. **Access Control**: Proper permission management
5. **Audit Trails**: Comprehensive logging and monitoring

### 🚀 Performance Guidelines

1. **Benchmark Everything**: Track performance regressions
2. **Optimize Builds**: Use caching and parallel execution
3. **Resource Management**: Monitor CI/CD resource usage
4. **Test Efficiency**: Fast feedback loops
5. **Scalability**: Design for package growth

## Support & Contributing

### 📞 Getting Help

- **CI/CD Issues**: Create issue with `ci/cd` label
- **Security Concerns**: Use security issue template
- **Performance Problems**: Include benchmark data
- **Documentation Updates**: Submit PR with changes

### 🤝 Contributing

1. Follow existing patterns and conventions
2. Test changes thoroughly before submission
3. Update documentation for new features
4. Consider backward compatibility
5. Review security implications

---

**Last Updated**: July 2025  
**Version**: 1.0.0  
**Maintainers**: DevOps Team