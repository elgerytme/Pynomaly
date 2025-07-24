# anomaly_detection Buck2-First Monorepo Migration Plan

## Overview

This document outlines the migration from the current hybrid structure to a Buck2-first monorepo architecture.

## Current State Assessment

### Existing Structure
- **Dual package system**: `src/anomaly_detection/` (monolithic) + `src/packages/` (monorepo)
- **17 packages** in the current workspace
- **Mixed build systems**: Hatch + Buck2 configurations
- **Complex dependencies** with some circular references

### Issues Identified
1. **Dual structure confusion** - developers unsure which package to modify
2. **Inconsistent build processes** - different packages use different build systems
3. **Nested package structures** - enterprise packages have packages within packages
4. **Scattered configuration** - multiple pyproject.toml files with inconsistent settings

## Migration Strategy

### Phase 1: Buck2 Infrastructure ✅ COMPLETED
- [x] Created root BUCK file with monorepo structure
- [x] Implemented shared Buck2 build rules (`tools/buck/python.bzl`)
- [x] Created comprehensive testing framework (`tools/buck/testing.bzl`)
- [x] Set up package-specific BUCK files for core packages

### Phase 2: Package Template System ✅ COMPLETED
- [x] Created standardized package templates with Buck2 integration
- [x] Implemented advanced Buck2 macros (`tools/buck/monorepo_python_package.bzl`)
- [x] Created package generators with Buck2-native configurations
- [x] Set up domain-specific build configurations for all packages

### Phase 3: Package Structure Consolidation 🚧 IN PROGRESS

#### 3.1 Eliminate Dual Structure
```bash
# Migrate src/anomaly_detection/ content to appropriate packages
src/anomaly_detection/domain/         → src/packages/core/domain/
src/anomaly_detection/application/    → src/packages/core/application/
src/anomaly_detection/infrastructure/ → src/packages/infrastructure/
src/anomaly_detection/presentation/   → src/packages/api/, src/packages/cli/, src/packages/web/
```

#### 3.2 Consolidate Data Packages
```bash
# Merge data processing packages into unified data-platform
src/packages/data_transformation/ → src/packages/data-platform/transformation/
src/packages/data_profiling/      → src/packages/data-platform/profiling/
src/packages/data_quality/        → src/packages/data-platform/quality/
src/packages/data_science/        → src/packages/data-platform/science/
```

#### 3.3 Flatten Enterprise Structure
```bash
# Eliminate nested packages
src/packages/enterprise/enterprise-packages/packages/ → src/packages/enterprise/
```

#### 3.4 Standardize Interface Packages
```bash
# Group all user interfaces
src/packages/api/     → src/packages/interfaces/api/
src/packages/cli/     → src/packages/interfaces/cli/
src/packages/web/     → src/packages/interfaces/web/
src/packages/sdk/     → src/packages/interfaces/sdk/
```

### Phase 4: Configuration Standardization

#### 4.1 Root Configuration
- **Primary build system**: Buck2
- **Package manager**: Hatch (for Python dependencies)
- **Developer tools**: Nx (for visualization and scaffolding)
- **Testing**: Buck2 + pytest integration

#### 4.2 Package-Level Configuration
- **BUCK file**: Primary build configuration
- **pyproject.toml**: Minimal Python metadata only
- **project.json**: Nx integration configuration
- **README.md**: Package documentation

## New Monorepo Structure

```
src/packages/
├── core/                    # Domain logic (no dependencies)
│   ├── domain/
│   ├── application/
│   ├── dto/
│   ├── use_cases/
│   └── shared/
├── infrastructure/          # Infrastructure adapters
│   ├── database/
│   ├── external-apis/
│   ├── caching/
│   ├── monitoring/
│   └── security/
├── algorithms/              # ML algorithm adapters
│   └── adapters/
├── data-platform/           # Unified data processing
│   ├── transformation/
│   ├── profiling/
│   ├── quality/
│   └── science/
├── mlops/                   # MLOps functionality
├── enterprise/              # Enterprise features
├── interfaces/              # All user interfaces
│   ├── api/                # REST API
│   ├── cli/                # Command line
│   ├── web/                # Web UI
│   └── sdk/                # Client SDKs
└── testing/                 # Shared testing utilities
```

## Buck2 Build System

### Primary Responsibilities
- **Package builds**: All package compilation and bundling
- **Dependency management**: Clear package dependencies
- **Test execution**: Comprehensive test suites
- **Incremental builds**: Only rebuild changed packages
- **Remote caching**: Shared build artifacts across team

### Key Features
1. **Layer enforcement**: Domain packages cannot depend on infrastructure
2. **Parallel builds**: Build independent packages simultaneously  
3. **Incremental testing**: Only test affected packages
4. **Build caching**: Reuse artifacts when nothing changed
5. **Dependency validation**: Prevent circular dependencies

### Build Targets

#### Core Targets
```bash
buck2 build //:core                  # Core domain package
buck2 build //:infrastructure        # Infrastructure adapters  
buck2 build //:application          # Application services
buck2 build //:algorithms           # ML algorithms
```

#### Interface Targets
```bash
buck2 build //:api                  # REST API
buck2 build //:cli                  # Command line interface
buck2 build //:web                  # Web UI
buck2 build //:sdk                  # Client SDKs
```

#### Convenience Targets
```bash
buck2 build //:build-core           # All core packages
buck2 build //:build-interfaces     # All interface packages
buck2 build //:build-all            # Complete monorepo
buck2 test //:test-all              # All test suites
```

## Developer Experience Features

### Buck2-Native Developer Tools
1. **Dependency visualization**: `buck2 uquery deps(...)`
2. **Affected package detection**: `buck2 build @mode/dev` with incremental builds
3. **Package scaffolding**: Standardized package templates and generators
4. **Workspace management**: Buck2 macros and domain-specific configurations

### Developer Commands
```bash
# Visualize dependencies
buck2 uquery "deps(//src/packages/...)"

# Build only affected packages (incremental)
buck2 build @mode/dev //src/packages/...

# Generate new package using templates
python scripts/create_domain_package.py --name=new-feature --domain=data

# Sync and validate Buck2 configuration
buck2 audit rules
```

## Migration Commands

### Automated Package Setup
```bash
# 1. Validate current Buck2 setup
buck2 build //:build-all
buck2 test //:test-all

# 2. Generate missing BUCK files for packages
for package in src/packages/*/; do
  if [ ! -f "$package/BUCK" ]; then
    python scripts/create_domain_package.py --name=$(basename $package) --domain=data
  fi
done

# 3. Update existing packages to use standardized macros
python scripts/migrate_package_structure.py --update-buck-files
```

### Manual Verification
```bash
# Check package dependencies
buck2 query "deps(//:build-all)"

# Verify no circular dependencies
buck2 query "allpaths(//:core, //:infrastructure)"

# Test incremental builds
buck2 build //:build-all --show-output
```

## Benefits of Buck2-First Approach

### Performance
- **5-10x faster builds** through incremental compilation
- **Remote caching** eliminates redundant builds across team
- **Parallel execution** utilizes all CPU cores efficiently

### Developer Experience  
- **Clear dependencies** prevent architecture violations
- **Fast feedback** through incremental testing
- **Consistent builds** across all environments

### Scalability
- **Handles large codebases** efficiently (designed for Meta-scale)
- **Language agnostic** supports Python, JavaScript, Rust, etc.
- **Remote execution** for distributed builds (future)

### Quality
- **Dependency validation** prevents circular references
- **Layer enforcement** maintains clean architecture
- **Hermetic builds** ensure reproducibility

## Timeline

### Week 1: Infrastructure Complete ✅
- [x] Buck2 build rules and configurations
- [x] Nx integration for developer experience
- [x] Package generator and sync tools

### Week 2: Package Consolidation 🚧
- [ ] Eliminate dual package structure
- [ ] Consolidate data platform packages
- [ ] Flatten enterprise package structure
- [ ] Update all package configurations

### Week 3: Testing & Validation
- [ ] Comprehensive test suite execution
- [ ] Performance benchmarking
- [ ] Developer workflow validation
- [ ] Documentation updates

### Week 4: Migration Complete
- [ ] Team training on new workflow
- [ ] CI/CD pipeline updates
- [ ] Monitoring and alerting setup
- [ ] Migration retrospective

## Success Metrics

1. **Build Performance**: 50%+ reduction in build times
2. **Developer Productivity**: Faster package creation and testing
3. **Code Quality**: Zero circular dependencies
4. **Architecture Compliance**: Clean layer separation
5. **Team Adoption**: All developers using Buck2 workflow

## Rollback Plan

If issues arise, revert using:
```bash
git checkout main -- .buckconfig BUCK src/packages/*/BUCK
# Clean up any temporary migration files if needed
rm -rf tmp/migration
```

The existing Hatch-based build system remains functional as fallback.