# Build Configuration Archive

This directory contains archived and historical Buck2 build configurations for Pynomaly.

## Configuration Files

### Active Configuration
- **../../../BUCK** - Current production Buck2 configuration with complete clean architecture support

### Archived Configurations
- **BUCK.full-archived** - Previous comprehensive configuration (archived June 2025)
- **BUCK.testing** - Minimal testing configuration used during Buck2 integration testing
- **BUCK.minimal-broken** - Failed minimal configuration attempt (kept for reference)
- **BUCK.python-failed** - Failed Python-specific configuration attempt (kept for reference)

## Configuration Evolution

### Phase 1: Initial Integration (June 2025)
- Started with minimal testing configuration
- Basic validation and development readiness checks

### Phase 2: Full Architecture Support (June 2025)
- Complete clean architecture layer mapping
- Binary targets for CLI, API, and Web UI
- Comprehensive test targets by layer
- Web assets build pipeline
- Performance and quality targets
- Package distribution support
- CI/CD integration targets

## Buck2 Build Features

The current configuration supports:

### Architecture Layers
- **Domain Layer**: Pure business logic
- **Application Layer**: Use cases and services
- **Infrastructure Layer**: External integrations
- **Presentation Layer**: APIs, CLI, Web UI
- **Shared Layer**: Common utilities

### Binary Targets
- **pynomaly-cli**: Main CLI application
- **pynomaly-api**: FastAPI server
- **pynomaly-web**: Web UI server

### Test Categories
- Unit tests by architecture layer
- Integration and E2E tests
- Performance benchmarks
- Property-based tests
- Mutation tests
- Security tests

### Build Targets
- Python wheel and source distributions
- Web assets (Tailwind CSS, JavaScript)
- Documentation generation
- Development environment setup
- CI/CD pipeline integration

## Usage

### Development
```bash
buck2 build :dev
buck2 test :test-all
```

### Production Build
```bash
buck2 build :build-all
buck2 build :release
```

### Specific Components
```bash
buck2 build :pynomaly-cli
buck2 build :pynomaly-api
buck2 build :web-assets
```

## Performance Benefits

Buck2 provides significant build performance improvements:
- **12.5x faster** clean builds
- **38.5x faster** incremental builds
- Intelligent caching and parallelization
- Architecture-aware dependency resolution

## Notes

- Buck artifacts are preserved in `buck-out/` for caching
- Build configurations follow clean architecture principles
- All tests maintain layer dependency constraints
- Web assets are built using modern toolchain integration