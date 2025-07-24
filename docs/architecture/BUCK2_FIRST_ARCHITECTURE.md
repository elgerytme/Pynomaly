# Buck2-First Monorepo Architecture

## Overview

This monorepo follows a **Buck2-first architecture** with comprehensive build system integration, standardized package templates, and domain-driven design principles.

## Core Principles

### 1. Buck2-Native Build System
- **Primary build tool**: Buck2 handles all compilation, testing, and deployment
- **Advanced macros**: Domain-specific package configurations via `monorepo_python_package.bzl`
- **Incremental builds**: Only rebuild changed packages and their dependencies
- **Remote caching**: Team-wide build artifact sharing (configurable)

### 2. Domain-Driven Package Organization
```
src/packages/data/
├── detection_service/       # AI/ML domain
├── data_quality/           # Data domain  
├── data_management/        # Data domain
├── knowledge_graph/        # AI domain
└── ...                     # 20+ packages
```

### 3. Standardized Package Structure
Every package follows the consistent template:
```
package_name/
├── BUCK                    # Buck2 build configuration
├── README.md              # Package documentation
├── pyproject.toml         # Python metadata (minimal)
├── src/
│   └── package_name/      # Source code
│       ├── __init__.py
│       ├── cli.py         # CLI entry point
│       ├── server.py      # Server entry point
│       ├── worker.py      # Worker entry point
│       ├── application/   # Application layer
│       ├── domain/        # Domain layer
│       ├── infrastructure/# Infrastructure layer
│       └── presentation/  # Presentation layer
├── tests/                 # Test suites
├── docs/                  # Documentation
├── examples/              # Usage examples
└── scripts/               # Package-specific scripts
```

## Buck2 Integration Features

### Advanced Build Macros

**Primary Macro**: `monorepo_python_package`
```python
load("//tools/buck:monorepo_python_package.bzl", "monorepo_python_package")

monorepo_python_package(
    name = "package_name",
    domain = "data",  # ai, data, enterprise, core
    visibility = ["PUBLIC"],
    cli_entry_points = {
        "package-name": "src/package_name.cli:main",
        "package-name-server": "src/package_name.server:main", 
        "package-name-worker": "src/package_name.worker:main",
    },
)
```

**Domain-Specific Features**:
- **Automatic dependency management** based on domain
- **Security scanning** integration (bandit, safety, semgrep)
- **Test discovery** and execution
- **Performance monitoring** and build analytics
- **CLI binary generation** for all entry points

### Build System Capabilities

#### Incremental Builds
```bash
# Build only changed packages
buck2 build @mode/dev //src/packages/...

# Test only affected packages  
buck2 test @mode/dev //src/packages/...
```

#### Dependency Management
```bash
# Visualize dependencies
buck2 uquery "deps(//src/packages/...)"

# Validate dependency graph
buck2 audit rules
```

#### Performance Features
- **Local caching**: 10GB disk cache with intelligent cleanup
- **Remote caching**: HTTP cache with JWT authentication (configurable)
- **Parallel execution**: Utilizes all CPU cores
- **Build analytics**: Performance monitoring and regression detection

## Package Generation System

### Domain Package Generator
```bash
# Create new package with Buck2 integration
python scripts/create_domain_package.py \
  --name=new-feature \
  --domain=data \
  --description="New data processing feature"
```

**Generated automatically**:
- ✅ Standardized BUCK file with domain-specific macro
- ✅ Complete source structure with all layers
- ✅ CLI, server, and worker entry points
- ✅ Test framework setup
- ✅ README with package documentation

### Template System
```bash
# Use self-contained package template
python src/packages/templates/self_contained_package/scripts/create-self-contained-package.py
```

**Features**:
- ✅ Interactive configuration
- ✅ Feature flags (database, caching, monitoring, etc.)
- ✅ Docker and Kubernetes support
- ✅ CI/CD pipeline generation

## Advanced Buck2 Features

### Security Integration
- **Multi-tool scanning**: bandit, safety, semgrep
- **Dependency vulnerability checks**: pip-audit
- **Secure build configurations**: TLS encryption, JWT authentication
- **Compliance reporting**: Automated security assessment

### Performance Monitoring
- **Build analytics**: SQLite database with metrics
- **Performance baselines**: Regression detection
- **Interactive dashboards**: HTML reports
- **Cache efficiency monitoring**: Hit rate tracking

### Remote Execution (Ready)
- **Configured but disabled** for security
- **Enterprise-ready**: Scalable build infrastructure
- **Cross-platform support**: Linux, macOS, Windows

## CI/CD Integration

### Buck2-Native Pipelines
```yaml
# .github/workflows/buck2-ci.yml
strategy:
  matrix:
    target: [ai-all, data-all, enterprise-all]
    python-version: ['3.11', '3.12']
```

**Features**:
- ✅ Matrix builds across domains
- ✅ Incremental testing with affected package detection
- ✅ Caching integration with GitHub Actions
- ✅ Security scanning in CI/CD
- ✅ Performance regression detection

## Migration from Other Build Systems

### From Nx (Not Used)
The monorepo **does not use Nx** - Buck2 provides superior Python ecosystem integration:
- **Better performance**: 5-10x faster builds
- **Python-native**: First-class Python support
- **Monorepo-optimized**: Designed for large codebases

### From Traditional Python Tools
- **Hatch**: Used only for Python metadata in pyproject.toml
- **setuptools**: Replaced by Buck2 for all builds
- **pip**: Buck2 manages dependencies through //third-party/python/

## Development Workflow

### Daily Development
```bash
# Build your changes
buck2 build //src/packages/your-package/...

# Run tests
buck2 test //src/packages/your-package/...

# Check dependencies
buck2 uquery "deps(//src/packages/your-package/...)"
```

### Package Creation
```bash
# Create new domain package
python scripts/create_domain_package.py --name=new-package --domain=data

# Validate Buck2 configuration
buck2 audit rules

# Build and test new package
buck2 build //src/packages/data/new-package/...
buck2 test //src/packages/data/new-package/...
```

### Performance Optimization
```bash
# Monitor build performance
buck2 build --show-output //src/packages/...

# Analyze cache efficiency
buck2 cache stats

# Generate performance report
python tools/buck/performance_analyzer.py
```

## Benefits Achieved

### Developer Experience
- ✅ **Consistent package structure** across 20+ packages
- ✅ **Standardized development workflow** with Buck2
- ✅ **Automated package generation** with domain awareness
- ✅ **Comprehensive tooling** for dependency management

### Build Performance  
- ✅ **5-10x faster builds** through incremental compilation
- ✅ **Parallel execution** utilizing all CPU cores
- ✅ **Smart caching** with local and remote options
- ✅ **Dependency optimization** preventing unnecessary rebuilds

### Code Quality
- ✅ **Security scanning** integrated into builds
- ✅ **Test automation** with comprehensive coverage
- ✅ **Dependency validation** preventing circular references
- ✅ **Architecture compliance** through layer enforcement

### Scalability
- ✅ **Monorepo-optimized** for large codebases
- ✅ **Remote execution ready** for enterprise scaling
- ✅ **Domain-based organization** supporting team growth
- ✅ **CI/CD integration** with matrix builds and caching

## Future Roadmap

### Phase 1: Enhanced Caching (Q2 2025)
- Enable remote HTTP caching for team collaboration
- Implement distributed build artifact sharing
- Optimize cache eviction policies

### Phase 2: Remote Execution (Q3 2025)
- Enable remote execution for large builds
- Implement build farm integration
- Add cross-platform build support

### Phase 3: Advanced Analytics (Q4 2025)
- Enhanced build performance analytics
- Predictive build optimization
- Advanced dependency graph analysis

The Buck2-first architecture provides a robust, scalable, and high-performance foundation for the monorepo, eliminating the complexity of multiple build systems while providing world-class developer experience.