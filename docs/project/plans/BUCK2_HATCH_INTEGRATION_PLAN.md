# Buck2 + Hatch Integration Plan for Pynomaly

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Project

---


## Overview
Buck2 will serve as the primary build system for development, testing, and local builds, while Hatch handles package publishing and release management. This hybrid approach leverages Buck2's speed and caching for development workflows while maintaining Hatch's Python packaging expertise.

## Phase 1: Research & Architecture Design

### Buck2 Requirements Analysis
- **Target Integration**: Build system for Python project with clean architecture
- **Key Benefits**: Incremental builds, remote caching, parallel execution
- **Compatibility**: Ensure Buck2 works with existing Poetry/Hatch ecosystem
- **Performance Goals**: Faster test execution, dependency resolution, and builds

### Architecture Mapping
```
Buck2 Build Targets:
├── //src/pynomaly/domain:lib          # Domain layer library
├── //src/pynomaly/application:lib     # Application layer library  
├── //src/pynomaly/infrastructure:lib  # Infrastructure layer library
├── //src/pynomaly/presentation:lib    # Presentation layer library
├── //tests:unit                       # Unit test suite
├── //tests:integration               # Integration test suite
├── //benchmarks:perf                 # Performance benchmarks
└── //examples:demos                  # Example applications
```

## Phase 2: Configuration Setup

### Core Buck2 Files
- **`.buckconfig`**: Main Buck2 configuration with Python toolchain setup
- **`BUCK`**: Root build file with project-wide targets
- **`toolchains/BUCK`**: Python toolchain definitions
- **Build files per package**: Individual BUCK files for each layer

### Integration Points
```python
# .buckconfig example structure
[buildfile]
name = BUCK

[python]
interpreter = python3.11
package_style = standalone

[cache]
mode = dir
dir_max_size = 10GB

[build]
threads = 8
```

## Phase 3: Build Target Definition

### Layer-Specific Targets
```python
# src/pynomaly/domain/BUCK
python_library(
    name = "domain",
    srcs = glob(["**/*.py"]),
    deps = [
        "//third-party:pydantic",
        "//third-party:structlog",
    ],
    visibility = ["PUBLIC"],
)

# Test targets
python_test(
    name = "domain_tests",
    srcs = glob(["tests/**/*.py"]),
    deps = [":domain", "//third-party:pytest"],
)
```

### Web UI Build Integration
```python
# src/pynomaly/presentation/web/BUCK
genrule(
    name = "tailwind_build",
    srcs = ["tailwind.config.js", "input.css"],
    out = "static/css/styles.css",
    cmd = "npm run build-css",
)

python_library(
    name = "web",
    srcs = glob(["**/*.py"]),
    resources = [":tailwind_build"],
    deps = [
        "//src/pynomaly/application:lib",
        "//third-party:fastapi",
        "//third-party:jinja2",
    ],
)
```

## Phase 4: Comprehensive Hatch Integration Strategy

### Complete Hatch Configuration
```toml
# pyproject.toml - Full Hatch integration
[build-system]
requires = ["hatchling>=1.18.0", "hatch-vcs>=0.3.0"]
build-backend = "hatchling.build"

[project]
name = "pynomaly"
dynamic = ["version"]
description = "State-of-the-art Python anomaly detection package"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    {name = "Pynomaly Team", email = "team@pynomaly.dev"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
keywords = ["anomaly-detection", "machine-learning", "outlier-detection", "pyod", "pygod"]

# Core dependencies
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "pydantic>=2.0.0",
    "structlog>=23.0.0",
    "click>=8.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "dependency-injector>=4.41.0",
]

# Optional dependencies for different use cases
[project.optional-dependencies]
# ML frameworks
pytorch = ["torch>=2.0.0", "torchvision>=0.15.0"]
tensorflow = ["tensorflow>=2.13.0"]
jax = ["jax>=0.4.0", "jaxlib>=0.4.0"]

# Anomaly detection libraries
pyod = ["pyod>=1.1.0"]
pygod = ["pygod>=1.1.0"]
sklearn = ["scikit-learn>=1.3.0"]

# Data processing
data = ["polars>=0.19.0", "pyarrow>=12.0.0", "duckdb>=0.8.0"]

# Visualization and web UI
web = [
    "jinja2>=3.1.0",
    "plotly>=5.15.0",
    "dash>=2.12.0",
    "streamlit>=1.25.0"
]

# Development and testing
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-xdist>=3.3.0",
    "hypothesis>=6.82.0",
    "mypy>=1.5.0",
    "black>=23.7.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "bandit>=1.7.0",
    "pre-commit>=3.3.0",
]

# Production deployment
prod = [
    "gunicorn>=21.0.0",
    "prometheus-client>=0.17.0",
    "opentelemetry-api>=1.19.0",
    "redis>=4.6.0",
]

# All optional dependencies
all = [
    "pynomaly[pytorch,tensorflow,jax,pyod,pygod,sklearn,data,web,dev,prod]"
]

[project.urls]
Documentation = "https://pynomaly.readthedocs.io/"
Repository = "https://github.com/pynomaly/pynomaly"
"Bug Tracker" = "https://github.com/pynomaly/pynomaly/issues"
Changelog = "https://github.com/pynomaly/pynomaly/blob/main/CHANGELOG.md"

[project.scripts]
pynomaly = "pynomaly.presentation.cli:main"

[project.entry-points."pynomaly.detectors"]
# Plugin system for custom detectors
isolation-forest = "pynomaly.infrastructure.adapters.sklearn:IsolationForestAdapter"
local-outlier-factor = "pynomaly.infrastructure.adapters.sklearn:LocalOutlierFactorAdapter"
```

#### Hatch Environment Management
```toml
# Environment configurations for different development stages
[tool.hatch.envs.default]
dependencies = [
    "coverage[toml]>=6.5",
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
]

[tool.hatch.envs.default.scripts]
test = "pytest {args:tests}"
test-cov = "coverage run -m pytest {args:tests}"
cov-report = [
    "- coverage combine",
    "coverage report",
]
cov-html = [
    "- coverage combine",
    "coverage html",
]

# Buck2 integration environment
[tool.hatch.envs.buck2]
dependencies = [
    "buck2",
]
[tool.hatch.envs.buck2.scripts]
build = "buck2 build //src/pynomaly:all"
test = "buck2 test //tests:all"
benchmark = "buck2 run //benchmarks:perf"

# Documentation environment
[tool.hatch.envs.docs]
dependencies = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.2.0",
    "mkdocstrings[python]>=0.22.0",
]
[tool.hatch.envs.docs.scripts]
build = "mkdocs build --clean --strict"
serve = "mkdocs serve --dev-addr localhost:8000"

# Production environment
[tool.hatch.envs.prod]
dependencies = [
    "pynomaly[prod]",
]
[tool.hatch.envs.prod.scripts]
deploy = "gunicorn pynomaly.presentation.api:app"

# Web UI development environment
[tool.hatch.envs.web]
dependencies = [
    "pynomaly[web]",
    "nodejs>=18.0.0",  # For Tailwind CSS compilation
]
[tool.hatch.envs.web.scripts]
build-css = "npm run build-css"
watch-css = "npm run watch-css"
serve = "uvicorn pynomaly.presentation.api:app --reload"
```

### Custom Hatch Build Hooks
```python
# hatch_plugins/buck2_hook.py - Custom Hatch plugin for Buck2 integration
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from hatchling.plugin import hookimpl
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class Buck2BuildHook(BuildHookInterface):
    """Custom build hook that triggers Buck2 builds before Hatch packaging."""

    PLUGIN_NAME = "buck2"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Initialize the Buck2 build hook."""
        self.buck2_targets = self.config.get("targets", ["//src/pynomaly:all"])
        self.buck2_output_dir = Path(self.config.get("output_dir", "bazel-bin"))

    def clean(self, versions: list[str]) -> None:
        """Clean Buck2 build artifacts."""
        subprocess.run(["buck2", "clean"], check=True)

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        """Run Buck2 build before Hatch packaging."""
        # Execute Buck2 build
        for target in self.buck2_targets:
            result = subprocess.run(
                ["buck2", "build", target],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Buck2 build failed for {target}: {result.stderr}")

        # Copy Buck2 artifacts to Hatch build directory
        self._copy_buck2_artifacts(build_data)

    def _copy_buck2_artifacts(self, build_data: dict[str, Any]) -> None:
        """Copy Buck2 build artifacts to Hatch build directory."""
        import shutil

        # Define artifact mappings
        artifact_mappings = [
            ("bazel-bin/src/pynomaly", "pynomaly"),
            ("bazel-bin/src/pynomaly/presentation/web/static", "pynomaly/presentation/web/static"),
        ]

        for src_pattern, dest_path in artifact_mappings:
            src_path = Path(src_pattern)
            if src_path.exists():
                dest = Path(build_data["build_directory"]) / dest_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                if src_path.is_dir():
                    shutil.copytree(src_path, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dest)

@hookimpl
def hatch_register_build_hook():
    return Buck2BuildHook
```

#### Hatch Build Configuration with Buck2
```toml
# pyproject.toml - Build system configuration
[tool.hatch.build]
directory = "dist"
dev-mode-dirs = ["src"]

# Buck2 build hook configuration
[tool.hatch.build.hooks.buck2]
targets = [
    "//src/pynomaly:all",
    "//src/pynomaly/presentation/web:static",
    "//benchmarks:all"
]
output_dir = "bazel-bin"

# Wheel-specific configuration
[tool.hatch.build.targets.wheel]
packages = ["src/pynomaly"]
artifacts = [
    "bazel-bin/src/pynomaly/**/*.py",
    "bazel-bin/src/pynomaly/**/static/**/*",
    "bazel-bin/src/pynomaly/**/templates/**/*",
]

[tool.hatch.build.targets.wheel.hooks.buck2]
# Buck2 artifacts to include in wheel
enable-by-default = true

# Source distribution configuration
[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/docs",
    "/examples",
    "/BUCK",
    "/.buckconfig",
    "/pyproject.toml",
    "/README.md",
    "/LICENSE",
    "/CHANGELOG.md",
]
```

## Phase 5: Unified Build Workflows

### Development Workflow Scripts
```toml
# pyproject.toml - Unified workflow scripts
[tool.hatch.envs.default.scripts]
# Core development tasks
dev-install = [
    "hatch dep show requirements",
    "pip install -e .",
]

# Buck2 + Hatch combined workflows
build-all = [
    "buck2 build //src/pynomaly:all",
    "hatch build",
]

test-all = [
    "buck2 test //tests:unit",
    "buck2 test //tests:integration",
    "hatch run test-cov",
]

# Quality assurance
qa = [
    "hatch run lint",
    "hatch run type-check",
    "hatch run test-all",
    "hatch run security-check",
]

lint = [
    "black --check --diff src tests",
    "isort --check-only --diff src tests",
    "flake8 src tests",
]

format = [
    "black src tests",
    "isort src tests",
]

type-check = "mypy --strict src"
security-check = "bandit -r src"

# Performance benchmarking
benchmark = [
    "buck2 run //benchmarks:perf",
    "buck2 run //benchmarks:memory",
    "buck2 run //benchmarks:scalability",
]

# Web UI development
web-dev = [
    "npm install",
    "npm run build-css",
    "hatch run web:serve",
]

# Documentation
docs-build = "hatch run docs:build"
docs-serve = "hatch run docs:serve"

# Release workflow
release-check = [
    "hatch run qa",
    "hatch run benchmark",
    "hatch build",
    "hatch publish --dry-run",
]

release = [
    "hatch run release-check",
    "hatch version patch",  # or minor/major
    "hatch build",
    "hatch publish",
]
```

#### Buck2 Configuration for Hatch Integration
```python
# .buckconfig - Enhanced configuration for Hatch integration
[buildfile]
name = BUCK

[python]
interpreter = python3.11
package_style = standalone
pex_extension = .pex

[cache]
mode = dir
dir_max_size = 10GB
http_max_store_attempts = 2

[build]
threads = 8
engine = prelude

# Integration with Hatch environments
[hatch]
default_environment = default
build_hook_enabled = true

# Custom build rules for Python packaging
[python_packaging]
wheel_builder = hatch
source_builder = hatch
publish_repository = pypi

# Web assets compilation
[web_assets]
css_compiler = tailwind
js_bundler = esbuild
static_optimizer = true
```

## Phase 6: Advanced Integration Features

### Multi-Stage Build Pipeline
```python
# scripts/build_pipeline.py - Orchestrated build pipeline
#!/usr/bin/env python3
"""Orchestrated build pipeline using Buck2 and Hatch."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

class BuildPipeline:
    """Manages the complete Buck2 + Hatch build pipeline."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.buck2_available = self._check_buck2()
        self.hatch_available = self._check_hatch()

    def run_full_pipeline(self, target: str = "development") -> bool:
        """Run the complete build pipeline."""
        stages = [
            ("Pre-build validation", self._validate_environment),
            ("Buck2 dependency build", self._buck2_dependencies),
            ("Buck2 source build", self._buck2_build),
            ("Buck2 testing", self._buck2_test),
            ("Web assets compilation", self._build_web_assets),
            ("Hatch packaging", self._hatch_build),
            ("Integration validation", self._validate_build),
        ]

        if target == "release":
            stages.extend([
                ("Performance benchmarks", self._run_benchmarks),
                ("Security validation", self._security_check),
                ("Release preparation", self._prepare_release),
            ])

        for stage_name, stage_func in stages:
            print(f"🔄 {stage_name}...")
            if not stage_func():
                print(f"❌ {stage_name} failed")
                return False
            print(f"✅ {stage_name} completed")

        print("🎉 Build pipeline completed successfully!")
        return True

    def _buck2_build(self) -> bool:
        """Execute Buck2 build."""
        result = subprocess.run([
            "buck2", "build",
            "//src/pynomaly:all",
            "//src/pynomaly/presentation/web:static",
        ])
        return result.returncode == 0

    def _buck2_test(self) -> bool:
        """Execute Buck2 tests."""
        result = subprocess.run([
            "buck2", "test",
            "//tests:all",
            "--test-output", "streaming"
        ])
        return result.returncode == 0

    def _hatch_build(self) -> bool:
        """Execute Hatch build with Buck2 artifacts."""
        result = subprocess.run(["hatch", "build", "--clean"])
        return result.returncode == 0

    def _build_web_assets(self) -> bool:
        """Build web UI assets."""
        npm_result = subprocess.run(["npm", "run", "build-css"])
        if npm_result.returncode != 0:
            return False

        # Buck2 web asset build
        web_result = subprocess.run([
            "buck2", "build", "//src/pynomaly/presentation/web:assets"
        ])
        return web_result.returncode == 0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pynomaly build pipeline")
    parser.add_argument("--target", choices=["development", "release"],
                       default="development", help="Build target")
    args = parser.parse_args()

    pipeline = BuildPipeline(Path.cwd())
    success = pipeline.run_full_pipeline(args.target)
    sys.exit(0 if success else 1)
```

## Phase 7: CI/CD Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/build-test-publish.yml
name: Build, Test, and Publish

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  release:
    types: [published]

env:
  PYTHON_VERSION: "3.11"

jobs:
  build-and-test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.11", "3.12"]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # For version detection

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Buck2
        shell: bash
        run: |
          if [[ "$RUNNER_OS" == "Linux" ]]; then
            curl -L https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-unknown-linux-gnu.zst | zstd -d > buck2
          elif [[ "$RUNNER_OS" == "macOS" ]]; then
            curl -L https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-apple-darwin.zst | zstd -d > buck2
          elif [[ "$RUNNER_OS" == "Windows" ]]; then
            curl -L https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-pc-windows-msvc.exe -o buck2.exe
          fi
          chmod +x buck2* && sudo mv buck2* /usr/local/bin/ || move buck2.exe "C:\Windows\System32\"

      - name: Install Hatch
        run: pip install hatch

      - name: Install Node.js (for web assets)
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install web dependencies
        run: npm install

      - name: Run Buck2 build
        run: buck2 build //src/pynomaly:all

      - name: Run Buck2 tests
        run: buck2 test //tests:all --test-output streaming

      - name: Build web assets
        run: npm run build-css

      - name: Run Hatch build
        run: hatch build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: pynomaly-${{ matrix.os }}-${{ matrix.python-version }}
          path: dist/

  benchmarks:
    needs: build-and-test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Buck2
        run: |
          curl -L https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-unknown-linux-gnu.zst | zstd -d > buck2
          chmod +x buck2 && sudo mv buck2 /usr/local/bin/

      - name: Install Hatch
        run: pip install hatch

      - name: Run performance benchmarks
        run: |
          hatch run benchmark
          buck2 run //benchmarks:perf > benchmark_results.txt

      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results.txt

  publish:
    needs: [build-and-test, benchmarks]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    environment:
      name: pypi
      url: https://pypi.org/project/pynomaly/
    permissions:
      id-token: write  # For trusted publishing

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Buck2
        run: |
          curl -L https://github.com/facebook/buck2/releases/latest/download/buck2-x86_64-unknown-linux-gnu.zst | zstd -d > buck2
          chmod +x buck2 && sudo mv buck2 /usr/local/bin/

      - name: Install Hatch
        run: pip install hatch

      - name: Install Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Build for release
        run: |
          npm install
          python scripts/build_pipeline.py --target release

      - name: Publish to PyPI
        run: hatch publish
```

## Phase 8: Developer Experience

### Unified CLI Commands
```bash
# Makefile - Unified development commands
.PHONY: help install build test benchmark clean publish

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install development dependencies
	hatch env create
	pre-commit install

build:  ## Build using Buck2 + Hatch pipeline
	python scripts/build_pipeline.py --target development

build-release:  ## Build for release
	python scripts/build_pipeline.py --target release

test:  ## Run all tests
	hatch run test-all

test-unit:  ## Run unit tests only
	buck2 test //tests:unit

test-integration:  ## Run integration tests only
	buck2 test //tests:integration

benchmark:  ## Run performance benchmarks  
	hatch run benchmark

web-dev:  ## Start web UI development server
	hatch run web-dev

docs:  ## Build and serve documentation
	hatch run docs-serve

lint:  ## Run code quality checks
	hatch run lint

format:  ## Format code
	hatch run format

clean:  ## Clean build artifacts
	buck2 clean
	hatch clean
	rm -rf dist/ .coverage htmlcov/

publish-test:  ## Test publish to PyPI
	hatch publish --repository test

publish:  ## Publish to PyPI
	hatch publish

# Quick commands for common workflows
dev: install build test  ## Full development setup
ci: build test benchmark lint  ## CI/CD pipeline simulation
release: clean build-release publish  ## Release workflow
```

## Expected Benefits

### Development Experience
- **Faster Builds**: Incremental compilation and smart caching
- **Parallel Testing**: Concurrent test execution across modules
- **Dependency Management**: Explicit, reproducible dependency graphs
- **IDE Integration**: Better code navigation and refactoring support

### CI/CD Improvements
- **Build Speed**: 3-5x faster builds with proper caching
- **Test Parallelization**: Distributed test execution
- **Resource Efficiency**: Better resource utilization in CI
- **Reproducible Builds**: Hermetic builds with explicit dependencies

### Production Benefits
- **Build Reliability**: Deterministic, reproducible builds
- **Deployment Speed**: Faster build-to-deploy cycles
- **Resource Management**: Better memory and CPU utilization
- **Scalability**: Supports team growth and larger codebase

## Implementation Timeline

- **Week 1-2**: Research and proof-of-concept setup ✅
- **Week 3-4**: Core configuration and domain layer migration
- **Week 5-6**: Application and infrastructure layer integration
- **Week 7-8**: CI/CD pipeline updates and testing
- **Week 9-10**: Performance optimization and documentation
- **Week 11-12**: Full migration and Poetry deprecation

## Migration Strategy

### Gradual Migration Approach
1. **Parallel Setup**: Maintain Poetry alongside Buck2 initially
2. **Layer-by-Layer**: Migrate each architecture layer systematically
3. **Testing Validation**: Ensure test parity between systems
4. **Performance Validation**: Benchmark build and test times
5. **Full Migration**: Remove Poetry once Buck2 fully validated

### Compatibility Maintenance
```bash
# Maintain multiple build options during transition
make build-poetry    # Current Poetry-based build
make build-buck2     # New Buck2-based build
make test-poetry     # Current Poetry-based tests  
make test-buck2      # New Buck2-based tests
```

This comprehensive integration plan provides a robust foundation for using Buck2 as the build system while leveraging Hatch for Python packaging, dependency management, and publishing workflows.
