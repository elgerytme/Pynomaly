# Buck2 + Hatch Integration Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 📄 Buck2_Integration_Guide

---


## Overview

Pynomaly includes a comprehensive Buck2 + Hatch integration that provides high-performance builds with intelligent caching while maintaining Hatch's excellent Python packaging capabilities.

## Current Status

✅ **Phase 1 Complete**: Core Configuration Setup
- Buck2 configuration files (`.buckconfig`, `BUCK`, `toolchains/BUCK`)
- Hatch integration hooks and build configuration
- Buck2 build hook plugin (`hatch_buck2_plugin/`)
- Web assets pipeline configuration
- Layer-specific build targets for clean architecture

## Setup Requirements

### 1. Buck2 Installation

```bash
# Install Buck2 (example for Linux/macOS)
curl -L -o buck2 https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-unknown-linux-gnu
chmod +x buck2
sudo mv buck2 /usr/local/bin/

# Verify installation
buck2 --version
```

### 2. Enable Buck2 Integration

Once Buck2 is installed, uncomment the following sections in `pyproject.toml`:

```toml
[build-system]
requires = [
    "hatchling", 
    "hatch-vcs",
    "hatch-buck2-plugin @ file:///absolute/path/to/pynomaly/hatch_buck2_plugin",
]

[tool.hatch.build.hooks.buck2]
executable = "buck2"
targets = [
    "//:pynomaly-lib",
    "//:pynomaly-cli", 
    "//:pynomaly-api",
    "//:pynomaly-web",
    "//:web-assets",
    "//:tailwind-build"
]
web_assets = true
artifacts_dir = "buck-out"
```

## Build Targets

### Core Architecture Targets

```bash
# Domain layer (pure business logic)
buck2 build //:domain

# Application layer (use cases and services)
buck2 build //:application

# Infrastructure layer (external integrations)
buck2 build //:infrastructure

# Presentation layer (API, CLI, Web UI)
buck2 build //:presentation

# Complete library
buck2 build //:pynomaly-lib
```

### Application Binaries

```bash
# CLI application
buck2 build //:pynomaly-cli

# API server
buck2 build //:pynomaly-api

# Web UI server
buck2 build //:pynomaly-web
```

### Web Assets

```bash
# Tailwind CSS compilation
buck2 build //:tailwind-build

# JavaScript bundling
buck2 build //:pynomaly-js

# Complete web assets
buck2 build //:web-assets
```

### Testing Targets

```bash
# Layer-specific tests
buck2 test //:test-domain
buck2 test //:test-application
buck2 test //:test-infrastructure
buck2 test //:test-presentation

# Integration and E2E tests
buck2 test //:test-integration

# Performance benchmarks
buck2 test //:benchmarks

# Property-based tests
buck2 test //:property-tests

# Mutation tests
buck2 test //:mutation-tests

# Security tests
buck2 test //:security-tests
```

### Convenience Targets

```bash
# All tests
buck2 build //:test-all

# Complete build
buck2 build //:build-all

# Development environment
buck2 build //:dev

# CI test suite
buck2 build //:ci-tests

# Release artifacts
buck2 build //:release
```

## Hatch Integration Benefits

### 1. Hybrid Build System
- **Buck2**: Fast incremental builds, remote caching, parallel execution
- **Hatch**: Python packaging expertise, PyPI publishing, environment management

### 2. Development Workflow

```bash
# Standard Hatch commands continue to work
hatch build                    # Uses Buck2 for acceleration when available
hatch publish                  # Standard Python packaging
hatch run test:run            # Test with Hatch environments
hatch run lint:all            # Linting and formatting

# Buck2 commands for fast development
buck2 build //:dev            # Fast development setup
buck2 test //:test-all        # Fast test execution
buck2 run //:pynomaly-cli     # Run CLI directly from build
```

### 3. Fallback Behavior
- If Buck2 is not available, Hatch uses standard build process
- No breaking changes to existing workflows
- Gradual adoption possible

## Performance Benefits

### Build Speed
- **Incremental builds**: Only rebuild changed components
- **Remote caching**: Share build artifacts across team/CI
- **Parallel execution**: Utilize all CPU cores effectively

### Development Experience
- **Fast iteration**: Sub-second builds for small changes
- **Layer isolation**: Clean architecture enforced at build level
- **Web asset optimization**: Integrated Tailwind/JS compilation

## Configuration Details

### .buckconfig
```ini
[python]
interpreter = python3
pex_extension = .pex

[cache]
mode = dir
dir = .buck-cache

[build]
execution_platforms = prelude//platforms:default
```

### BUCK Build Targets
The root `BUCK` file defines 40+ build targets including:
- **Architecture layers**: domain, application, infrastructure, presentation
- **Binary targets**: CLI, API, Web UI applications
- **Web assets**: Tailwind CSS, JavaScript bundling
- **Test suites**: Comprehensive testing across all layers
- **Quality gates**: Performance, security, mutation testing
- **Distribution**: Wheel and source distribution generation

### Web Assets Pipeline
```python
# Tailwind CSS Build
genrule(
    name = "tailwind-build",
    srcs = ["config/web/tailwind.config.js"] + glob([
        "src/pynomaly/presentation/web/templates/**/*.html",
        "src/pynomaly/presentation/web/static/css/**/*.css",
    ]),
    out = "static/css/tailwind.css",
    cmd = "npm run build-css && cp src/pynomaly/presentation/web/static/css/tailwind.css $OUT",
)
```

## Migration Strategy

### Phase 1: Core Configuration ✅ COMPLETE
- [x] Buck2 configuration files
- [x] Hatch build hooks
- [x] Layer-specific targets
- [x] Web assets pipeline

### Phase 2: Build Integration (Next)
- [ ] Install Buck2 in development environments
- [ ] Enable Buck2 build hooks
- [ ] Test complete build workflow
- [ ] Performance benchmarking

### Phase 3: CI/CD Integration
- [ ] GitHub Actions Buck2 installation
- [ ] Remote caching setup
- [ ] Multi-platform builds
- [ ] Performance monitoring

### Phase 4: Team Adoption
- [ ] Developer onboarding documentation
- [ ] Build performance analytics
- [ ] Optimization recommendations
- [ ] Full Poetry deprecation

## Troubleshooting

### Buck2 Not Found
```bash
# Check Buck2 installation
which buck2
buck2 --version

# Verify PATH configuration
echo $PATH
```

### Build Failures
```bash
# Clean Buck2 cache
buck2 clean

# Verbose build output
buck2 build --verbose //:target-name

# Check build logs
cat buck-out/log/buck.log
```

### Hatch Integration Issues
```bash
# Test without Buck2 hooks
hatch build --clean

# Check build hook status
hatch build --debug
```

## Performance Metrics

Expected performance improvements with Buck2:
- **Clean builds**: 2-5x faster than standard Python builds
- **Incremental builds**: 10-50x faster for small changes
- **Test execution**: 3-10x faster with intelligent caching
- **Web assets**: 5-15x faster compilation

## Next Steps

1. **Install Buck2** in your development environment
2. **Uncomment Buck2 configuration** in `pyproject.toml`
3. **Test basic builds**: `buck2 build //:pynomaly-lib`
4. **Benchmark performance** against standard builds
5. **Integrate into CI/CD** for team-wide benefits

The Buck2 + Hatch integration represents a state-of-the-art build system that scales from individual development to large team deployments while maintaining the simplicity and reliability of Python packaging standards.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
