# Pynomaly Buck2-First Monorepo

## 🚀 Overview

Pynomaly now uses a **Buck2-first monorepo architecture** with selective Nx integration, providing:

- **⚡ High-performance builds** with incremental compilation and remote caching
- **🎯 Clean architecture enforcement** through dependency validation
- **📊 Excellent developer experience** with dependency visualization and scaffolding
- **🔧 Powerful tooling** for package management and testing

## 🏗️ Architecture

### Build System Strategy
- **Primary**: Buck2 for all builds, tests, and dependency management
- **Secondary**: Nx for visualization, scaffolding, and affected package detection
- **Package Manager**: Hatch for Python dependency management

### Package Structure
```
src/packages/
├── core/                    # 🎯 Domain logic (no dependencies)
├── infrastructure/          # 🔌 Infrastructure adapters
├── algorithms/              # 🤖 ML algorithm adapters
├── data-platform/           # 📊 Unified data processing
├── mlops/                   # 🔄 MLOps functionality
├── enterprise/              # 🏢 Enterprise features
├── interfaces/              # 💻 User interfaces
│   ├── api/                # REST API
│   ├── cli/                # Command line
│   ├── web/                # Web UI
│   └── sdk/                # Client SDKs
└── testing/                 # 🧪 Shared testing utilities
```

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure Buck2 is available (already configured)
buck2 --version

# Install Node.js for Nx integration
node --version  # Should be 18+
```

### Essential Commands

```bash
# 🏗️ Build everything
buck2 build //:build-all

# 🧪 Run all tests
buck2 test //:test-all

# 🔍 Build specific package
buck2 build //src/packages/core:core

# 📊 View dependency graph
nx graph

# ⚡ Test only affected packages
nx affected:test
```

## 📦 Working with Packages

### Building Packages

```bash
# Core packages (most frequently used)
buck2 build //:build-core

# Interface packages (user-facing)
buck2 build //:build-interfaces

# Specific package
buck2 build //src/packages/api:api

# Development mode (with hot reload)
buck2 build //src/packages/api:dev
```

### Testing Packages

```bash
# Comprehensive test suite
buck2 test //:test-all

# Package-specific tests
buck2 test //src/packages/core:test-all

# Test by type
buck2 test //src/packages/core:test-unit
buck2 test //src/packages/api:test-integration
buck2 test //src/packages/api:test-security
```

### Creating New Packages

```bash
# Generate new package with Buck2 + Nx integration
nx generate package --name=my-feature --type=application

# Available types: core, infrastructure, application, interface, library
nx generate package --name=my-adapter --type=infrastructure
```

## 🎯 Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes to packages
# Edit src/packages/...

# Build affected packages only
buck2 build $(python tools/scripts/get-affected-packages.py)

# Test affected packages
buck2 test $(python tools/scripts/get-affected-packages.py)
```

### 2. Continuous Integration
```bash
# Fast feedback (unit tests)
buck2 test //:ci-tests

# Full validation
buck2 build //:build-all
buck2 test //:test-all
```

### 3. Performance Optimization
```bash
# Incremental builds (only changed packages)
buck2 build //:build-all  # ~5-10 seconds for no changes

# Parallel execution (uses all CPU cores)
buck2 test //:test-all --num-threads=auto

# Remote caching (when available)
buck2 build //:build-all --remote-cache
```

## 📊 Developer Experience with Nx

### Visualization
```bash
# Interactive dependency graph
nx graph

# Show affected packages
nx affected:graph

# Project information
nx show projects
```

### Package Management
```bash
# Sync Nx with Buck2 configuration
nx run workspace-tools:buck2-sync

# Generate package with templates
nx generate package --name=feature --type=application

# Run commands across packages
nx run-many --target=test
```

## 🧪 Testing Strategy

### Test Types
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Package interaction testing  
3. **Property Tests**: Property-based testing with Hypothesis
4. **Performance Tests**: Benchmarking and performance validation
5. **Security Tests**: Security validation for interfaces

### Test Execution
```bash
# Fast feedback loop (unit tests only)
buck2 test $(buck2 query "filter('test-unit', //src/packages/...)")

# Layer-specific testing
buck2 test $(buck2 query "filter('test-.*', deps(//src/packages/core/...))")

# Comprehensive testing
buck2 test //:test-all
```

## 🔧 Migration from Previous Structure

### Automated Migration
```bash
# Dry run to see what would change
python tools/scripts/migrate-to-buck2.py --dry-run

# Perform migration
python tools/scripts/migrate-to-buck2.py

# Validate migration
buck2 build //:build-all
buck2 test //:test-all
```

### Manual Steps (if needed)
1. **Backup current structure**: `cp -r src migration-backup/`
2. **Run migration script**: `python tools/scripts/migrate-to-buck2.py`
3. **Update configurations**: `python tools/nx/sync-nx-buck2.py`
4. **Validate builds**: `buck2 build //:build-all`

## 📈 Performance Benefits

### Build Performance
- **5-10x faster builds** through incremental compilation
- **Remote caching** eliminates redundant builds
- **Parallel execution** utilizes all CPU cores
- **Dependency optimization** builds only what changed

### Developer Productivity
- **Clear dependencies** prevent architecture violations
- **Fast feedback** through incremental testing  
- **Package scaffolding** reduces boilerplate
- **Visual tools** improve understanding

### Code Quality
- **Dependency validation** prevents circular references
- **Layer enforcement** maintains clean architecture
- **Hermetic builds** ensure reproducibility
- **Comprehensive testing** at package boundaries

## 🛠️ Advanced Usage

### Custom Build Rules
```python
# In tools/buck/python.bzl
load("//tools/buck:python.bzl", "pynomaly_workspace_package")

pynomaly_workspace_package(
    name = "my-package",
    layer = "application",
    package_type = "service",
    dependencies = ["//src/packages/core:core"],
)
```

### Dependency Analysis
```bash
# Check for circular dependencies
buck2 query "allpaths(//src/packages/core:core, //src/packages/infrastructure:infrastructure)"

# Find all dependents
buck2 query "rdeps(//src/packages/..., //src/packages/core:core)"

# Dependency graph visualization
buck2 query "deps(//src/packages/api:api)" --output-attribute=buck.type
```

### Remote Caching Setup
```bash
# Configure remote cache (when available)
# Edit .buckconfig to add remote cache URL
buck2 build //:build-all --remote-cache
```

## 📚 Documentation

- **[Developer Guide](docs/developer-guides/BUCK2_MONOREPO_GUIDE.md)**: Comprehensive development guide
- **[Migration Plan](MONOREPO_MIGRATION_PLAN.md)**: Detailed migration strategy
- **[Architecture Decisions](docs/architecture/adr/)**: Architectural decision records

## 🚨 Troubleshooting

### Common Issues

1. **Build cache issues**:
```bash
buck2 clean
buck2 build //:build-all
```

2. **Dependency conflicts**:
```bash
buck2 query "deps(//src/packages/my-package:my-package)"
```

3. **Test failures**:
```bash
buck2 test //src/packages/core:test-unit --verbose
```

### Getting Help
- **Buck2 documentation**: https://buck2.build/
- **Nx documentation**: https://nx.dev/
- **Project issues**: Create GitHub issue with `[Buck2]` prefix

## 🎉 Benefits Summary

✅ **Performance**: 5-10x faster builds with incremental compilation
✅ **Quality**: Dependency validation prevents architecture violations  
✅ **Productivity**: Visual tools and scaffolding reduce development time
✅ **Scalability**: Designed for large-scale monorepo management
✅ **Consistency**: Unified build system across all packages
✅ **Developer Experience**: Best-in-class tooling for monorepo development

---

**Ready to build faster? Start with `buck2 build //:build-all` and experience the difference!** 🚀