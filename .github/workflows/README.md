# 🚀 CI/CD Workflows

This directory contains automated workflows for building, testing, and publishing the anomaly detection platform and its SDKs.

## 📋 Available Workflows

### Core Workflows
- **`release.yml`** - Complete release management with validation, building, and deployment
- **`publish-pypi.yml`** - Dedicated Python package publishing to PyPI
- **`sdk-publishing.yml`** - Multi-language SDK publishing (Python, TypeScript, Java)

### Workflow Triggers
- **Release:** Triggered when a new release is published
- **Manual:** Can be triggered manually via workflow_dispatch
- **Push:** Some workflows trigger on specific tag pushes

## 🔧 Required Secrets Configuration

For the workflows to function properly, the following secrets must be configured in your GitHub repository:

### Python Publishing (PyPI)
```bash
# Production PyPI
PYPI_API_TOKEN=pypi-your-production-token-here

# Test PyPI (for testing releases)
TESTPYPI_API_TOKEN=pypi-your-test-token-here
```

### TypeScript Publishing (NPM)
```bash
# NPM registry token
NPM_TOKEN=npm_your-npm-token-here
```

### Java Publishing (Maven Central)
```bash
# OSSRH (Sonatype) credentials
OSSRH_USERNAME=your-sonatype-username
OSSRH_TOKEN=your-sonatype-token

# GPG signing (required for Maven Central)
MAVEN_GPG_PRIVATE_KEY=-----BEGIN PGP PRIVATE KEY BLOCK-----...
MAVEN_GPG_PASSPHRASE=your-gpg-passphrase
```

### GitHub (Automatic)
```bash
# Automatically provided by GitHub Actions
GITHUB_TOKEN=ghp_automatically-provided
```

## 🎯 Workflow Usage

### 1. Publishing All SDKs (Release)
```bash
# Create and publish a release - this will automatically publish all SDKs
gh release create v1.2.3 --title "Release v1.2.3" --notes "Release notes..."
```

### 2. Manual SDK Publishing
```bash
# Publish specific SDKs manually
gh workflow run sdk-publishing.yml \
  -f sdks_to_publish="python,typescript,java" \
  -f environment="production"

# Test publishing to staging registries
gh workflow run sdk-publishing.yml \
  -f sdks_to_publish="python" \
  -f environment="test"
```

### 3. Python Only Publishing
```bash
# Use the dedicated Python workflow
gh workflow run publish-pypi.yml \
  -f environment="pypi"
```

## 🏗️ SDK Build & Publishing Process

### 1. Change Detection
- Automatically detects changes in SDK source code
- Only builds and publishes changed SDKs (unless manual override)
- Supports selective publishing via manual triggers

### 2. Validation Phase
- Starts API server for integration testing
- Runs comprehensive SDK validation suite
- Validates all SDKs against real endpoints
- Can be skipped for emergency releases

### 3. Build Phase (Per SDK)

#### Python SDK
- Uses `hatch` for building
- Builds both wheel and source distributions
- Validates package metadata with `twine`
- Publishes to PyPI or TestPyPI

#### TypeScript SDK
- Uses npm with Rollup for building
- Generates CommonJS and ESM bundles
- Includes TypeScript declarations
- Publishes to npm registry

#### Java SDK
- Uses Maven for building
- Generates JAR with sources and JavaDocs
- Signs artifacts with GPG for Maven Central
- Deploys to OSSRH (Sonatype)

### 4. Verification Phase
- Waits for packages to become available
- Tests installation from each registry
- Verifies basic functionality
- Generates comprehensive reports

## 📊 Environment Configuration

### Production Environment
- **Python:** Published to PyPI
- **TypeScript:** Published to npm (public)
- **Java:** Published to Maven Central
- **Validation:** Full validation required

### Test Environment
- **Python:** Published to TestPyPI
- **TypeScript:** Published to npm with `beta` tag
- **Java:** Deployed to Maven Central staging (not released)
- **Validation:** Can be skipped

### Staging Environment
- Used for internal testing before production
- Same as test but with different deployment targets

## 🔍 Monitoring & Debugging

### Workflow Logs
- All workflows provide detailed logging
- Each step includes clear success/failure indicators
- Artifacts are stored for 30-90 days

### Common Issues

#### Authentication Errors
```bash
# Check if secrets are properly configured
gh secret list

# Update secrets if needed
gh secret set PYPI_API_TOKEN --body "your-token"
```

#### Build Failures
```bash
# Check the specific SDK build logs
# Look for dependency issues or test failures
# Verify local build works before pushing
```

#### Publishing Errors
```bash
# Version conflicts - ensure version numbers are unique
# Registry timeouts - workflows include retry logic
# Authentication - verify tokens have correct permissions
```

### Artifact Storage
- Build artifacts stored for 30 days
- Release reports stored for 90 days
- Logs available for configured retention period

## 🚀 Quick Setup Checklist

1. **Configure Secrets**
   - [ ] Set up PyPI API tokens
   - [ ] Configure NPM token
   - [ ] Set up Maven Central credentials
   - [ ] Generate and configure GPG key

2. **Test Workflows**
   - [ ] Run test publication to staging registries
   - [ ] Verify SDK validation works
   - [ ] Test manual workflow triggers

3. **First Release**
   - [ ] Create release with proper version tag
   - [ ] Monitor all publishing workflows
   - [ ] Verify packages are available
   - [ ] Test installation from registries

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [NPM Publishing Guide](https://docs.npmjs.com/using-npm/publishing-packages-to-npm)
- [Maven Central Publishing Guide](https://central.sonatype.org/publish/publish-guide/)

## 🆘 Support

If you encounter issues with the CI/CD workflows:

1. Check the workflow logs in GitHub Actions
2. Verify all required secrets are configured
3. Test SDK builds locally first
4. Review the validation reports and artifacts

For additional support, create an issue in the repository with:
- Workflow run URL
- Error messages
- Steps to reproduce the issue