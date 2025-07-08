# Container Migration Audit

**Date**: 2025-01-08  
**Status**: In Progress  
**Purpose**: Audit and checklist for migrating from local virtual environment workflows to container usage

## Executive Summary

This document identifies all references to local virtual environment workflows within the Pynomaly repository and provides a comprehensive checklist of files/sections that must be updated or deprecated in favor of container usage. The migration aims to standardize development, testing, and deployment environments across all platforms.

## Current State Analysis

### Virtual Environment References Found

#### Root Level Files
- `README.md` - Extensive virtual environment setup instructions
- `pyproject.toml` - Hatch environment configurations
- `requirements.txt` - Legacy pip requirements
- `Makefile` - Virtual environment targets and workflows

#### Configuration Files
- `config/.env.example` - Environment variable template
- `config/environments/` - Multiple requirements files for different environments
- `config/backup_poetry_config/` - Legacy Poetry configuration files

#### Scripts Directory
- `scripts/setup/` - 15+ setup scripts for virtual environments
- `scripts/testing/` - Environment-specific testing scripts
- `scripts/development/` - Development environment setup utilities

#### CI/CD Workflows
- `.github/workflows/` - 24 workflow files with virtual environment setups
- Multiple Python version matrix testing with venv creation

#### Documentation
- `docs/getting-started/installation.md` - Detailed virtual environment instructions
- `docs/getting-started/FEATURE_INSTALLATION_GUIDE.md` - Feature-specific pip installations
- `docs/windows_setup.md` - Platform-specific virtual environment guidance

## Migration Checklist

### 🔴 Critical Priority - Must Update

#### 1. **README.md**
- **Current**: Extensive virtual environment setup (lines 62-76, 97-147)
- **Action**: Replace with Docker-first approach
- **Impact**: Primary user entry point
- **Effort**: High
- **Files to modify**:
  - Update installation sections to prioritize Docker
  - Add container-based quick start
  - Move virtual environment instructions to "Alternative Setup"

#### 2. **pyproject.toml**
- **Current**: Hatch environment configurations (lines 408-534)
- **Action**: Simplify to focus on package metadata, remove environment-specific settings
- **Impact**: Build system configuration
- **Effort**: Medium
- **Files to modify**:
  - Remove `[tool.hatch.envs.*]` sections
  - Keep build configuration only
  - Update dependencies to be container-friendly

#### 3. **CI/CD Workflows (.github/workflows/)**
- **Current**: 24 workflow files with virtual environment setup
- **Action**: Migrate to container-based CI
- **Impact**: All automated testing and deployment
- **Effort**: High
- **Key files**:
  - `ci.yml` - Lines 97-106, 222-226 (Hatch and Python setup)
  - `multi-python-testing.yml` - Lines 132-157 (Virtual environment creation)
  - `comprehensive-testing.yml`
  - `production-cicd.yml`

### 🟡 High Priority - Should Update

#### 4. **Makefile**
- **Current**: Virtual environment targets (lines 123-140, 174-189)
- **Action**: Replace with Docker-based workflows
- **Impact**: Development workflow automation
- **Effort**: Medium
- **Targets to replace**:
  - `setup`, `dev-install`, `env-clean`
  - Test execution targets
  - Development workflow commands

#### 5. **Setup Scripts (scripts/setup/)**
- **Current**: 15+ virtual environment setup scripts
- **Action**: Deprecate or containerize
- **Impact**: Developer onboarding
- **Effort**: Medium
- **Scripts to deprecate**:
  - `setup_simple.py` (268 lines)
  - `setup_fresh_environment.sh` (143 lines)
  - `install_features.py` (191 lines)
  - `setup_dev_environment.py`
  - `setup_multi_python.py`

#### 6. **Documentation (docs/getting-started/)**
- **Current**: Detailed virtual environment instructions
- **Action**: Rewrite for container-first approach
- **Impact**: User onboarding experience
- **Effort**: High
- **Files to update**:
  - `installation.md` (434 lines) - Primary installation guide
  - `FEATURE_INSTALLATION_GUIDE.md` (195 lines) - Feature installation
  - `README_SIMPLE_SETUP.md` - Simple setup instructions
  - `WINDOWS_SETUP_GUIDE.md` - Windows-specific guidance

### 🟢 Medium Priority - Consider Updating

#### 7. **Environment Configuration Files**
- **Current**: Multiple requirements files and configurations
- **Action**: Consolidate into Docker configurations
- **Impact**: Environment management
- **Effort**: Low
- **Files to review**:
  - `config/environments/requirements-*.txt` (4 files)
  - `config/backup_poetry_config/` (5 files)
  - Environment-specific configurations

#### 8. **Testing Scripts (scripts/testing/)**
- **Current**: Environment-specific test scripts
- **Action**: Containerize test execution
- **Impact**: Testing workflows
- **Effort**: Medium
- **Scripts to update**:
  - `test_fresh_installation.sh`
  - `test_setup_simple_*.sh|ps1`
  - `test_setup_with_poetry.sh`
  - Environment matrix testing scripts

#### 9. **Development Scripts (scripts/development/)**
- **Current**: Development environment utilities
- **Action**: Replace with container-based alternatives
- **Impact**: Developer workflows
- **Effort**: Low
- **Scripts to review**:
  - `setup_multi_python.py`
  - `setup_standalone.py`
  - Environment-specific development tools

### 🔵 Low Priority - Optional Updates

#### 10. **Legacy Configuration Files**
- **Current**: Poetry and legacy pip configurations
- **Action**: Archive or remove
- **Impact**: Minimal
- **Effort**: Low
- **Files to archive**:
  - `poetry.lock` (if exists)
  - `config/backup_poetry_config/`
  - Legacy requirements files

## Container Migration Strategy

### Phase 1: Foundation (Week 1)
1. **Enhance Docker Configurations**
   - Update `deploy/docker/Dockerfile` with all environment variants
   - Create development-focused `Dockerfile.dev`
   - Enhance `docker-compose.yml` with development services
   - Add multi-stage builds for different feature sets

2. **Create Container Scripts**
   - `scripts/docker/setup.sh` - Container-based setup
   - `scripts/docker/dev.sh` - Development environment
   - `scripts/docker/test.sh` - Test execution
   - `scripts/docker/build.sh` - Build workflows

### Phase 2: Documentation (Week 2)
1. **Update Primary Documentation**
   - Rewrite `README.md` with Docker-first approach
   - Update `docs/getting-started/installation.md`
   - Create `docs/deployment/CONTAINER_GUIDE.md`
   - Add container-based feature installation guide

2. **Migration Guides**
   - Create migration guide for existing developers
   - Document container equivalents for common workflows
   - Provide troubleshooting for container issues

### Phase 3: CI/CD Migration (Week 3)
1. **Update Workflows**
   - Migrate `.github/workflows/ci.yml` to container-based
   - Update testing workflows to use Docker
   - Enhance production deployment workflows
   - Add container security scanning

2. **Performance Optimization**
   - Implement Docker layer caching
   - Optimize build times with multi-stage builds
   - Add container registry integration

### Phase 4: Cleanup (Week 4)
1. **Deprecate Legacy Files**
   - Archive virtual environment scripts
   - Update Makefile targets
   - Clean up configuration files
   - Remove redundant documentation

2. **Validation**
   - Test all container workflows
   - Validate feature parity with virtual environments
   - Performance testing and optimization
   - Documentation review and updates

## Container Advantages

### Consistency Benefits
- **Environment Parity**: Identical environments across development, testing, and production
- **Dependency Isolation**: No conflicts with system packages or other projects
- **Version Control**: Infrastructure as code with Dockerfile versioning
- **Platform Independence**: Works identically on Windows, macOS, and Linux

### Developer Experience
- **Faster Onboarding**: Single `docker-compose up` command to start
- **Reduced Setup Issues**: Eliminate Python version and dependency conflicts
- **Easy Feature Testing**: Spin up containers with specific feature sets
- **Simplified CI/CD**: Consistent environments eliminate "works on my machine" issues

### Operational Benefits
- **Scalability**: Easy horizontal scaling with container orchestration
- **Security**: Isolated execution environments
- **Monitoring**: Better observability with container metrics
- **Deployment**: Simplified deployment pipeline with container registries

## Existing Container Infrastructure

### Current Docker Assets
- `deploy/docker/Dockerfile` - Multi-stage production build
- `deploy/docker/docker-compose.yml` - Service orchestration
- `deploy/docker/Dockerfile.hardened` - Security-focused build
- `deploy/docker/Dockerfile.production` - Production-optimized build
- `deploy/docker/Dockerfile.testing` - Testing environment
- `config/.env.example` - Environment variable template

### Container Support Matrix
| Feature Set | Container Support | Status |
|-------------|-------------------|--------|
| Core API | ✅ Dockerfile | Complete |
| Web UI | ✅ docker-compose.yml | Complete |
| Database | ✅ PostgreSQL service | Complete |
| Cache | ✅ Redis service | Complete |
| Monitoring | ⚠️ Partial | Needs work |
| Development | ❌ Missing | To implement |
| Testing | ⚠️ Basic | Needs enhancement |
| ML Features | ❌ Missing | To implement |

## Risk Assessment

### High Risk Areas
1. **Developer Adoption**: Resistance to changing established workflows
2. **Learning Curve**: Docker/container knowledge requirements
3. **Performance**: Potential overhead compared to native virtual environments
4. **Windows Compatibility**: Docker Desktop requirements and limitations

### Mitigation Strategies
1. **Gradual Migration**: Maintain virtual environment support during transition
2. **Documentation**: Comprehensive guides and troubleshooting
3. **Training**: Docker workshops and knowledge sharing sessions
4. **Fallback Options**: Keep virtual environment instructions as "Alternative Setup"

## Success Metrics

### Quantitative Measures
- Developer setup time: Target <10 minutes from clone to running
- CI/CD reliability: >95% consistent build success rate
- Environment-related issues: Reduce by 80%
- Cross-platform compatibility: 100% feature parity

### Qualitative Measures
- Developer satisfaction with onboarding experience
- Reduced support requests for environment setup
- Simplified deployment processes
- Improved collaboration between development and operations

## Timeline and Milestones

| Phase | Duration | Key Deliverables | Success Criteria |
|-------|----------|------------------|------------------|
| 1. Foundation | 1 week | Enhanced Dockerfiles, container scripts | All features available in containers |
| 2. Documentation | 1 week | Updated docs, migration guides | Clear container-first documentation |
| 3. CI/CD Migration | 1 week | Container-based workflows | All CI/CD using containers |
| 4. Cleanup | 1 week | Deprecated legacy files | No virtual environment references in main docs |

## Next Steps

1. **Immediate Actions**
   - Create `docs/internal/` directory structure
   - Begin Phase 1: Foundation work
   - Set up container development environment
   - Start documenting container equivalents

2. **Week 1 Deliverables**
   - Enhanced Docker configurations
   - Container-based development scripts
   - Initial documentation updates
   - Migration strategy validation

3. **Stakeholder Communication**
   - Share audit results with development team
   - Gather feedback on migration approach
   - Identify potential blockers or concerns
   - Plan knowledge transfer sessions

---

**Document Status**: Draft v1.0  
**Last Updated**: 2025-01-08  
**Next Review**: Weekly during migration phases  
**Owner**: Infrastructure Team  
**Stakeholders**: Development Team, DevOps, QA
