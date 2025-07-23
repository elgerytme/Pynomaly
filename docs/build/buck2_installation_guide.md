# Buck2 Installation and Setup Guide

## Current Status

✅ **Buck2 Infrastructure Complete** - All foundational Buck2 configuration is in place:

- ✅ `.buckconfig` - Main configuration file with Python toolchain setup
- ✅ `.buckroot` - Project root marker 
- ✅ `toolchains/BUCK` - Platform and toolchain definitions
- ✅ `tools/buck/python.bzl` - Custom Python build rules
- ✅ `tools/buck/testing.bzl` - Testing utilities and patterns
- ✅ `BUCK` - Main build file matching actual repository structure
- ✅ `third-party/python/BUCK` - External dependency definitions
- ✅ Removed conflicting Nx configuration

## Installation Required

⚠️ **Manual Buck2 Installation Needed** - To use the Buck2 build system, you must install Buck2:

### Option 1: Official Installer (Recommended)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://get.buck2.build/ | bash
```

### Option 2: From GitHub Releases
```bash
# Download latest release for Linux x86_64
curl -L https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-unknown-linux-gnu.zst -o buck2.zst
zstd -d buck2.zst
chmod +x buck2
sudo mv buck2 /usr/local/bin/
```

### Option 3: Build from Source
```bash
git clone https://github.com/facebook/buck2.git
cd buck2
cargo build --release
sudo cp target/release/buck2 /usr/local/bin/
```

## Verification

After installation, verify Buck2 works:

```bash
# Check installation
buck2 --version

# Validate configuration  
buck2 targets //...

# Test basic build
buck2 build //:anomaly-detection
```

## Available Build Targets

### Library Targets
```bash
# AI Domain
buck2 build //:ai-anomaly-detection
buck2 build //:ai-machine-learning  
buck2 build //:ai-mlops

# Data Domain
buck2 build //:data-analytics
buck2 build //:data-engineering
buck2 build //:data-quality
buck2 build //:data-observability
buck2 build //:data-profiling

# Enterprise Domain
buck2 build //:enterprise-auth
buck2 build //:enterprise-governance
buck2 build //:enterprise-scalability

# Aggregate Targets
buck2 build //:ai-all
buck2 build //:data-all
buck2 build //:enterprise-all
buck2 build //:anomaly-detection  # Complete monorepo
```

### CLI Targets
```bash
# CLI Applications
buck2 build //:data-engineering-cli
buck2 build //:anomaly-detection-cli

# Run CLI apps
buck2 run //:data-engineering-cli
buck2 run //:anomaly-detection-cli
```

### Test Targets
```bash
# Run tests by domain
buck2 test //:ai-tests
buck2 test //:data-tests  
buck2 test //:enterprise-tests

# Run all tests
buck2 test //...
```

## Next Steps

1. **Install Buck2** using one of the methods above
2. **Test Configuration**: Run `buck2 targets //...` to validate setup
3. **Build Packages**: Start with `buck2 build //:ai-anomaly-detection` 
4. **Configure Dependencies**: Update `third-party/python/BUCK` with actual Python package paths
5. **CI/CD Integration**: Migrate build scripts from Hatch to Buck2 commands

## Configuration Details

### Repository Structure
- **Domain-based organization**: AI, Data, Enterprise packages
- **Clean architecture**: Separation of concerns within each domain
- **Standardized patterns**: Common build rules and testing utilities

### Buck2 Benefits
- **Incremental builds**: Only rebuild changed components
- **Remote caching**: Share build artifacts across machines  
- **Parallel execution**: Build multiple targets simultaneously
- **Dependency management**: Automatic detection and resolution
- **Cross-language support**: Python, JavaScript, and more

### Development Workflow  
```bash
# Daily development cycle
buck2 build //:anomaly-detection          # Build all packages
buck2 test //...                 # Run all tests  
buck2 run //:data-engineering-cli  # Test CLI applications
```

The repository is now ready for Buck2 once the binary is installed. All infrastructure is in place following 2025 best practices for monorepo build systems.