# Buck2 Monorepo Developer Guide

## Overview

Pynomaly uses a **Buck2-first monorepo architecture** with selective Nx integration for the best of both worlds: high-performance builds with excellent developer experience.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for web assets and Nx)
- Buck2 installed (pre-configured in this repository)

### Basic Commands

```bash
# Build everything
buck2 build //:build-all

# Run all tests  
buck2 test //:test-all

# Build specific package
buck2 build //src/packages/core:core

# Test specific package
buck2 test //src/packages/core:test-all

# Development mode for a package
buck2 build //src/packages/api:dev
```

## Repository Structure

```
src/packages/
├── core/                    # Domain logic (no dependencies)
├── infrastructure/          # Infrastructure adapters  
├── algorithms/              # ML algorithm adapters
├── data-platform/           # Unified data processing
├── mlops/                   # MLOps functionality
├── enterprise/              # Enterprise features
├── interfaces/              # User interfaces
│   ├── api/                # REST API
│   ├── cli/                # Command line
│   ├── web/                # Web UI
│   └── sdk/                # Client SDKs
└── testing/                 # Shared testing utilities
```

## Package Dependencies

### Clean Architecture Layers
1. **Domain** (`core`): No dependencies
2. **Infrastructure**: Depends on `core` only
3. **Application** (`services`): Depends on `core` + `infrastructure`
4. **Presentation** (`interfaces`): Depends on all lower layers

### Dependency Rules
- ✅ Higher layers can depend on lower layers
- ❌ Lower layers cannot depend on higher layers
- ❌ Circular dependencies are prevented by Buck2

## Working with Packages

### Building Packages

```bash
# Build single package
buck2 build //src/packages/core:core

# Build with dependencies
buck2 build //src/packages/api:api

# Build all core packages
buck2 build //:build-core

# Build all interface packages  
buck2 build //:build-interfaces
```

### Testing Packages

```bash
# Run all tests for a package
buck2 test //src/packages/core:test-all

# Run specific test types
buck2 test //src/packages/core:test-unit
buck2 test //src/packages/core:test-integration
buck2 test //src/packages/api:test-security

# Run tests with coverage
buck2 test //src/packages/core:test-all --coverage
```

### Development Workflow

```bash
# 1. Create new feature branch
git checkout -b feature/my-feature

# 2. Make changes to packages
# Edit files in src/packages/...

# 3. Build affected packages only
buck2 build $(buck2 query "allpaths(set(//src/packages/...), $(buck2 query "owner('$(git diff --name-only HEAD~1)')"))")

# 4. Run affected tests
buck2 test $(buck2 query "tests(allpaths(set(//src/packages/...), $(buck2 query "owner('$(git diff --name-only HEAD~1)')"))")

# 5. Check for issues
buck2 build //:ci-tests
```

## Creating New Packages

### Using Nx Generator (Recommended)

```bash
# Generate new core package
nx generate package --name=my-feature --type=core

# Generate new interface package  
nx generate package --name=my-api --type=interface

# Generate new infrastructure package
nx generate package --name=my-adapter --type=infrastructure
```

### Manual Package Creation

1. **Create directory structure**:
```bash
mkdir -p src/packages/my-package/{domain,application,infrastructure,tests}
```

2. **Create BUCK file**:
```python
load("//tools/buck:python.bzl", "pynomaly_workspace_package")

pynomaly_workspace_package(
    name = "my-package",
    layer = "domain",  # or infrastructure, application, presentation
    package_type = "library",
    dependencies = [
        "//src/packages/core:core",
    ],
)
```

3. **Create pyproject.toml**:
```toml
[project]
name = "pynomaly-my-package"
description = "My package description"
dependencies = ["pynomaly-core"]
```

4. **Update workspace.json**:
```bash
python tools/nx/sync-nx-buck2.py
```

## Nx Integration

### Visualization

```bash
# View dependency graph
nx graph

# View affected packages
nx affected:graph

# Show project information
nx show projects
```

### Development Tools

```bash
# Generate package
nx generate package --name=feature --type=application

# Sync Nx with Buck2
nx run workspace-tools:buck2-sync

# Run command on all packages
nx run-many --target=test
```

## Build Performance

### Incremental Builds
Buck2 only rebuilds changed packages and their dependents:

```bash
# First build (full)
buck2 build //:build-all  # ~2-3 minutes

# Subsequent builds (incremental)  
buck2 build //:build-all  # ~5-10 seconds if no changes
```

### Remote Caching
Enable remote caching for team collaboration:

```bash
# Enable remote cache (when available)
buck2 build //:build-all --remote-cache

# Check cache status
buck2 status
```

### Parallel Execution
Buck2 automatically uses all CPU cores:

```bash
# Build packages in parallel
buck2 build //:build-all  # Uses all cores

# Limit parallelism if needed
buck2 build //:build-all --num-threads=4
```

## Testing Strategy

### Test Types per Package

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test package interactions
3. **Property Tests**: Property-based testing with Hypothesis
4. **Performance Tests**: Benchmarking critical paths
5. **Security Tests**: Security validation for interfaces

### Running Test Suites

```bash
# Fast feedback loop (unit tests only)
buck2 test $(buck2 query "filter('test-unit', //src/packages/...)")

# Full test suite
buck2 test //:test-all

# Test specific layer
buck2 test $(buck2 query "filter('test-.*', deps(//src/packages/core/...))")
```

## Debugging

### Build Issues

```bash
# Verbose build output
buck2 build //:build-all --verbose

# Show build steps
buck2 build //src/packages/core:core --show-output

# Debug dependency issues
buck2 query "deps(//src/packages/api:api)"
```

### Test Issues

```bash
# Run tests with verbose output
buck2 test //src/packages/core:test-unit --verbose

# Debug specific test
buck2 test //src/packages/core:test-unit -- -k test_specific_function

# Test with debugger
buck2 test //src/packages/core:test-unit -- --pdb
```

### Dependency Issues

```bash
# Check for circular dependencies
buck2 query "allpaths(//src/packages/core:core, //src/packages/infrastructure:infrastructure)"

# Find all dependents of a package
buck2 query "rdeps(//src/packages/..., //src/packages/core:core)"

# Show dependency graph
buck2 query "deps(//src/packages/api:api)" --output-attribute=buck.type
```

## Performance Optimization

### Build Optimization

1. **Use specific targets**: Build only what you need
```bash
# Instead of building everything
buck2 build //:build-all

# Build specific packages
buck2 build //src/packages/core:core //src/packages/api:api
```

2. **Leverage caching**: Keep build cache warm
```bash
# Regular cache maintenance
buck2 clean --keep-cache
```

3. **Parallel development**: Work on independent packages
```bash
# Check package independence
buck2 query "allpaths(//src/packages/feature-a:..., //src/packages/feature-b:...)"
```

### Test Optimization

1. **Test affected packages only**:
```bash
# After making changes
buck2 test $(buck2 query "affected(//src/packages/..., HEAD~1)")
```

2. **Use test filtering**:
```bash
# Run fast tests first
buck2 test $(buck2 query "filter('test-unit', //src/packages/...)")

# Then integration tests
buck2 test $(buck2 query "filter('test-integration', //src/packages/...)")
```

## Common Patterns

### Adding Dependencies

1. **Update BUCK file**:
```python
python_library(
    name = "my-package",
    deps = [
        "//src/packages/core:core",
        "//src/packages/infrastructure:database",  # Add new dependency
    ],
)
```

2. **Verify dependency is allowed**:
```bash
buck2 build //src/packages/my-package:my-package
```

### Package Refactoring

1. **Move code between packages**:
```bash
# Update BUCK files to reflect new structure
# Run sync tool to update Nx
python tools/nx/sync-nx-buck2.py
```

2. **Verify builds still work**:
```bash
buck2 build //:build-all
buck2 test //:test-all
```

### Adding New Interfaces

1. **Create interface package**:
```bash
nx generate package --name=graphql-api --type=interface
```

2. **Add appropriate dependencies**:
```python
# In src/packages/interfaces/graphql-api/BUCK
deps = [
    "//src/packages/core:core",
    "//src/packages/application:application",
]
```

## Troubleshooting

### Common Issues

1. **Circular dependency error**:
```
ERROR: Cycle detected in deps
```
**Solution**: Review package dependencies and remove circular references.

2. **Missing dependency error**:
```
ERROR: Unable to find target
```
**Solution**: Add missing dependency to BUCK file.

3. **Build cache issues**:
```bash
# Clear cache and rebuild
buck2 clean
buck2 build //:build-all
```

### Getting Help

1. **Check Buck2 status**:
```bash
buck2 status
```

2. **Query dependency information**:
```bash
buck2 query --help
```

3. **View Buck2 logs**:
```bash
tail -f ~/.buckd/buckd.log
```

## Best Practices

1. **Keep packages focused**: One domain per package
2. **Minimize dependencies**: Only depend on what you need
3. **Test at package boundaries**: Integration tests for package interfaces
4. **Use layer enforcement**: Respect clean architecture boundaries
5. **Leverage incremental builds**: Structure code for fast builds
6. **Document package purpose**: Clear README for each package
7. **Regular dependency audits**: Review and minimize dependencies