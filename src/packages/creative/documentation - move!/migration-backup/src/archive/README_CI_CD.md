# 🚀 Pynomaly CI/CD Pipeline

This document describes the comprehensive CI/CD pipeline for Pynomaly, designed to ensure code quality, security, and reliable deployments.

## 📋 Overview

The Pynomaly CI/CD pipeline consists of:

1. **Unified CI Pipeline** - Automated quality checks, testing, and validation
2. **Unified CD Pipeline** - Automated deployment to staging and production
3. **Custom CI Scripts** - Standalone scripts for specific CI tasks
4. **GitHub Actions Integration** - Full automation with GitHub workflows

## 🔧 Pipeline Components

### 1. Unified CI Pipeline (`ci-unified.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Weekly scheduled runs
- Manual dispatch

**Jobs:**
- **Quality Check**: Code quality, security scanning, documentation validation
- **Build**: Package building and verification
- **Test**: Comprehensive test suite (unit, integration, security, API)
- **Docker Build**: Container building and security scanning
- **CI Summary**: Consolidated reporting and status updates

**Key Features:**
- Parallel execution with optimized caching
- Multi-Python version testing (3.11, 3.12)
- Comprehensive test coverage reporting
- Security scanning with Trivy
- Automated PR comments with results

### 2. Unified CD Pipeline (`cd-unified.yml`)

**Triggers:**
- Push to `main` branch
- Release tags (`v*`)
- Manual dispatch with environment selection

**Jobs:**
- **Prepare Deployment**: Environment detection and setup
- **Build & Push**: Docker image building and registry push
- **Deploy Staging**: Staging environment deployment
- **Deploy Production**: Production environment deployment
- **Notify**: Slack notifications and deployment reports

**Key Features:**
- Environment-specific deployment logic
- Automated backup before production deployments
- Comprehensive health checks and smoke tests
- Security scanning of deployed images
- Rollback capabilities

### 3. Custom CI Scripts

#### Test Runner (`scripts/ci/test-runner.py`)
- Comprehensive test suite execution
- Multiple test types: unit, integration, API, security, performance, e2e
- Parallel test execution with pytest-xdist
- Detailed HTML and JSON reporting
- Coverage analysis and reporting

#### Quality Checker (`scripts/ci/quality-check.py`)
- Code quality analysis with multiple tools
- Ruff linting and formatting checks
- MyPy type checking
- Security analysis with Bandit and Safety
- Code complexity and duplication detection
- Comprehensive quality scoring

#### Deployment Manager (`scripts/ci/deploy.py`)
- Multi-environment deployment support
- Automated backup and recovery
- Health checks and smoke tests
- Configuration management
- Monitoring setup
- Notification system

#### Pipeline Runner (`scripts/ci/run-ci-pipeline.sh`)
- Unified CI pipeline orchestration
- Prerequisite checking
- Dependency management
- Service orchestration
- Comprehensive reporting

## 🎯 Key Benefits

### 1. **Reduced Complexity**
- Consolidated from 33 to 3 main workflows
- Unified pipeline with clear dependencies
- Simplified maintenance and debugging

### 2. **Enhanced Quality**
- Comprehensive code quality checks
- Multi-tool security scanning
- Automated testing across environments
- Performance and load testing

### 3. **Improved Reliability**
- Automated backup and recovery
- Health checks and smoke tests
- Rollback capabilities
- Environment-specific validation

### 4. **Better Visibility**
- Unified reporting and dashboards
- Clear status indicators
- Automated notifications
- Comprehensive artifact management

## 🚀 Getting Started

### Prerequisites

1. **Python 3.11+** - Required for development
2. **Docker** - For containerization and testing
3. **Git** - Version control
4. **Node.js** - For frontend components (if applicable)

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly
```

2. **Install dependencies:**
```bash
pip install -e ".[dev,test,lint]"
```

3. **Run quality checks:**
```bash
./scripts/ci/quality-check.py --install-deps
```

4. **Run tests:**
```bash
./scripts/ci/test-runner.py --suites unit integration
```

5. **Run full CI pipeline:**
```bash
./scripts/ci/run-ci-pipeline.sh
```

### GitHub Actions Setup

The CI/CD pipeline is automatically triggered by GitHub Actions. No additional setup is required for basic functionality.

**Environment Variables:**
- `GITHUB_TOKEN` - Automatically provided by GitHub
- `SLACK_WEBHOOK_URL` - For deployment notifications (optional)

**Secrets:**
- `DOCKER_REGISTRY_TOKEN` - For container registry access
- `DEPLOY_SSH_KEY` - For deployment access (if using SSH)

## 📊 Pipeline Stages

### CI Pipeline Stages

1. **Quality Check** (5-10 minutes)
   - Ruff linting and formatting
   - MyPy type checking
   - Bandit security analysis
   - Safety vulnerability scanning
   - Documentation validation

2. **Build** (2-5 minutes)
   - Package building with Hatch
   - Wheel installation testing
   - Artifact generation

3. **Test** (10-20 minutes)
   - Unit tests with coverage
   - Integration tests
   - Security tests
   - API tests
   - Performance tests

4. **Docker Build** (5-10 minutes)
   - Multi-stage Docker building
   - Security scanning with Trivy
   - Image testing

### CD Pipeline Stages

1. **Prepare Deployment** (1-2 minutes)
   - Environment detection
   - Image tag generation
   - Deployment setup

2. **Build & Push** (3-5 minutes)
   - Docker image building
   - Registry push
   - Security scanning

3. **Deploy Staging** (10-15 minutes)
   - Staging deployment
   - Health checks
   - Smoke tests

4. **Deploy Production** (15-20 minutes)
   - Database backup
   - Production deployment
   - Comprehensive validation

## 📈 Monitoring and Reporting

### Test Reports
- **HTML Reports**: Interactive test results with coverage
- **JSON Reports**: Machine-readable test data
- **Coverage Reports**: Detailed coverage analysis
- **Performance Reports**: Benchmark results

### Quality Reports
- **Quality Score**: Overall code quality percentage
- **Compliance Reports**: Security and style compliance
- **Trend Analysis**: Quality metrics over time

### Deployment Reports
- **Deployment Status**: Success/failure tracking
- **Performance Metrics**: Response times and resource usage
- **Security Scan Results**: Vulnerability assessments
- **Health Check Results**: System status validation

## 🔒 Security Features

### Code Security
- **Static Analysis**: Bandit security scanning
- **Dependency Scanning**: Safety vulnerability checks
- **Secret Detection**: Automated secret scanning
- **License Compliance**: License validation

### Container Security
- **Image Scanning**: Trivy vulnerability scanning
- **Base Image Updates**: Automated base image updates
- **Runtime Security**: Security monitoring
- **Access Controls**: RBAC and permissions

### Deployment Security
- **Secure Secrets**: Encrypted secret management
- **Network Security**: Secure communications
- **Audit Logging**: Comprehensive audit trails
- **Compliance**: GDPR, HIPAA, SOX compliance

## 🛠️ Customization

### Adding New Tests
1. Create test files in appropriate directories
2. Use pytest markers for test categorization
3. Update test runner configuration
4. Add to CI pipeline if needed

### Custom Quality Checks
1. Add new tools to quality checker
2. Configure tool-specific settings
3. Update reporting templates
4. Test with sample codebase

### Environment Configuration
1. Create environment-specific configs
2. Update deployment scripts
3. Configure monitoring
4. Test deployment process

## 🔧 Troubleshooting

### Common Issues

1. **Test Failures**
   - Check test logs in CI artifacts
   - Run tests locally with same configuration
   - Verify test data and fixtures

2. **Quality Check Failures**
   - Review quality reports
   - Run quality checks locally
   - Update code to meet standards

3. **Deployment Issues**
   - Check deployment logs
   - Verify environment configuration
   - Test deployment scripts locally

4. **Performance Issues**
   - Monitor resource usage
   - Check for memory leaks
   - Optimize test execution

### Debug Commands

```bash
# Run specific test suite
./scripts/ci/test-runner.py --suites unit --output-dir debug-output

# Run quality checks only
./scripts/ci/quality-check.py --checks ruff_lint mypy --output-dir debug-output

# Test deployment locally
./scripts/ci/deploy.py development --steps validate_environment health_check

# Run CI pipeline with debug output
DEBUG=1 ./scripts/ci/run-ci-pipeline.sh
```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [PyTest Documentation](https://docs.pytest.org/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run full CI pipeline locally
4. Submit pull request with clear description
5. Address any CI/CD feedback

## 📄 License

This CI/CD pipeline is part of the Pynomaly project and is licensed under the MIT License.

---

*This CI/CD pipeline ensures reliable, secure, and efficient software delivery for Pynomaly. For questions or support, please contact the development team.*