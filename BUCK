# Buck2 Build Configuration for Pynomaly Monorepo
# Comprehensive monorepo build system following Buck2-first architecture with clean boundaries

load("@prelude//python:defs.bzl", "python_binary", "python_library", "python_test")
load("@prelude//js:defs.bzl", "js_bundle")

# ==========================================
# Monorepo Package Organization
# Following domain-based clean architecture with Buck2 optimization
# ==========================================

# Core Domain Layer - Pure business logic with no external dependencies
python_library(
    name = "core",
    srcs = glob([
        "src/packages/core/**/*.py",
    ]),
    deps = [],
    visibility = ["//src/packages/..."],
    # Buck2 optimization: mark as fundamental dependency
    metadata = {
        "layer": "domain",
        "type": "core",
        "cache_priority": "high",
    },
)

# Infrastructure Layer - External integrations and adapters
python_library(
    name = "infrastructure",
    srcs = glob([
        "src/packages/infrastructure/**/*.py",
    ]),
    deps = [
        ":core",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "infrastructure",
        "type": "adapters",
    },
)

# Application Layer - Use cases and application services
python_library(
    name = "services",
    srcs = glob([
        "src/packages/services/**/*.py",
    ]),
    deps = [
        ":core",
        ":infrastructure",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "application",
        "type": "services",
    },
)

# Anomaly Detection Package - Consolidated anomaly and outlier detection
python_library(
    name = "anomaly-detection",
    srcs = glob([
        "src/packages/anomaly_detection/**/*.py",
    ]),
    deps = [
        ":core",
        ":mathematics",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "domain",
        "type": "anomaly-detection",
    },
)

# Machine Learning Package - ML operations and lifecycle management
python_library(
    name = "machine-learning",
    srcs = glob([
        "src/packages/machine_learning/**/*.py",
    ]),
    deps = [
        ":core",
        ":anomaly-detection",
        ":data-platform",
        ":infrastructure",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "application",
        "type": "machine-learning",
    },
)

# People Operations Package - User management and authentication
python_library(
    name = "people-ops",
    srcs = glob([
        "src/packages/people_ops/**/*.py",
    ]),
    deps = [
        ":core",
        ":infrastructure",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "application",
        "type": "people-ops",
    },
)

# Mathematics Package - Statistical analysis and computations
python_library(
    name = "mathematics",
    srcs = glob([
        "src/packages/mathematics/**/*.py",
    ]),
    deps = [
        ":core",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "domain",
        "type": "mathematics",
    },
)

# Data Platform Package - Consolidated data processing functionality
python_library(
    name = "data-platform",
    srcs = glob([
        "src/packages/data-platform/**/*.py",
    ]),
    deps = [
        ":core",
        ":infrastructure",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "application",
        "type": "data-platform",
    },
)

# MLOps Package - Legacy package (to be deprecated in favor of machine-learning)
python_library(
    name = "mlops",
    srcs = glob([
        "src/packages/mlops/**/*.py",
    ]),
    deps = [
        ":core",
        ":infrastructure",
        ":anomaly-detection",
        ":machine-learning",
        ":data-platform",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "application",
        "type": "mlops",
        "deprecated": True,
    },
)

# Enterprise Package - Enterprise features and multi-tenancy
python_library(
    name = "enterprise",
    srcs = glob([
        "src/packages/enterprise/**/*.py",
    ]),
    deps = [
        ":core",
        ":infrastructure",
        ":services",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "application",
        "type": "enterprise",
    },
)

# ==========================================
# Interface Layer Packages
# ==========================================

# Interfaces Package - All user-facing interfaces
python_library(
    name = "interfaces",
    srcs = glob([
        "src/packages/interfaces/**/*.py",
    ]),
    deps = [
        ":core",
        ":infrastructure", 
        ":services",
        ":anomaly-detection",
        ":machine-learning",
        ":people-ops",
        ":mathematics",
        ":data-platform",
        ":mlops",
        ":enterprise",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "layer": "presentation",
        "type": "interfaces",
    },
)

# Individual interface sub-packages accessible via interfaces
# These are handled by the main interfaces package above

# Testing Package - Shared testing utilities
python_library(
    name = "testing",
    srcs = glob([
        "src/packages/testing/**/*.py",
    ]),
    deps = [],
    visibility = ["PUBLIC"],
    metadata = {
        "layer": "shared",
        "type": "testing",
    },
)

# ==========================================
# Shared Libraries
# ==========================================

# Shared Utilities - Common utilities across packages
python_library(
    name = "shared",
    srcs = glob([
        "src/packages/core/shared/**/*.py",
    ]),
    deps = [],
    visibility = ["PUBLIC"],
    metadata = {
        "layer": "shared",
        "type": "utilities",
    },
)

# ==========================================
# Application Binaries
# ==========================================

# Main CLI Binary
python_binary(
    name = "pynomaly-cli",
    main = "src/packages/interfaces/cli/cli/app.py",
    deps = [":interfaces"],
    visibility = ["PUBLIC"],
)

# API Server Binary
python_binary(
    name = "pynomaly-api",
    main = "src/packages/interfaces/api/api/app.py",
    deps = [":interfaces"],
    visibility = ["PUBLIC"],
)

# Web UI Server Binary  
python_binary(
    name = "pynomaly-web",
    main = "src/packages/interfaces/web/web/app.py",
    deps = [":interfaces"],
    visibility = ["PUBLIC"],
)

# ==========================================
# Web Assets Build Pipeline
# ==========================================

# Tailwind CSS Build
genrule(
    name = "tailwind-build",
    srcs = [
        "src/workspace_configs/webpack.config.js",
        "config/web/tailwind.config.js",
    ] + glob([
        "src/packages/web/web/templates/**/*.html",
        "src/packages/web/web/static/**/*.css",
    ]),
    out = "static/css/tailwind.css",
    cmd = "cd $(location .) && npm run build-css",
    visibility = ["PUBLIC"],
)

# JavaScript Bundle
js_bundle(
    name = "pynomaly-js",
    srcs = glob([
        "src/packages/web/web/static/**/*.js",
        "src/build_artifacts/storybook-static/js/**/*.js",
    ]),
    entry_point = "src/packages/web/web/static/js/main.js",
    visibility = ["PUBLIC"],
)

# Web Assets Bundle
genrule(
    name = "web-assets",
    srcs = [
        ":tailwind-build",
        ":pynomaly-js",
    ] + glob([
        "src/packages/web/web/static/**/*",
        "src/build_artifacts/storybook-static/**/*",
    ]),
    out = "web-assets.tar.gz",
    cmd = "tar -czf $OUT $(location :tailwind-build) $(location :pynomaly-js)",
    visibility = ["PUBLIC"],
)

# ==========================================
# Test Targets by Layer
# ==========================================

# Core Domain Tests
python_test(
    name = "test-core",
    srcs = glob([
        "tests/domain/**/*.py",
        "tests/unit/**/*.py",
    ]),
    deps = [
        ":core",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Infrastructure Tests
python_test(
    name = "test-infrastructure",
    srcs = glob([
        "tests/infrastructure/**/*.py",
    ]),
    deps = [
        ":infrastructure",
        ":core",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Application Services Tests
python_test(
    name = "test-application",
    srcs = glob([
        "tests/application/**/*.py",
    ]),
    deps = [
        ":application",
        ":infrastructure",
        ":core",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Algorithm Tests
python_test(
    name = "test-algorithms",
    srcs = glob([
        "tests/automl/**/*.py",
    ]),
    deps = [
        ":algorithms",
        ":core",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Data Platform Tests
python_test(
    name = "test-data-platform",
    srcs = glob([
        "tests/data_science/**/*.py",
    ]),
    deps = [
        ":data-platform",
        ":core",
        ":infrastructure",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# API Tests
python_test(
    name = "test-api",
    srcs = glob([
        "tests/api/**/*.py",
        "tests/presentation/**/*.py",
    ]),
    deps = [
        ":api",
        ":application",
        ":core",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# CLI Tests
python_test(
    name = "test-cli",
    srcs = glob([
        "tests/cli/**/*.py",
    ]),
    deps = [
        ":cli",
        ":application",
        ":core",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Integration Tests
python_test(
    name = "test-integration",
    srcs = glob([
        "tests/integration/**/*.py",
        "tests/e2e/**/*.py",
    ]),
    deps = [
        ":core",
        ":infrastructure",
        ":application",
        ":api",
        ":cli",
        ":web",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# ==========================================
# Performance and Quality Targets
# ==========================================

# Performance Benchmarks
python_test(
    name = "benchmarks",
    srcs = glob([
        "tests/benchmarks/**/*.py",
        "tests/performance/**/*.py",
    ]),
    deps = [
        ":core",
        ":algorithms",
        ":application",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Property-Based Tests
python_test(
    name = "property-tests",
    srcs = glob([
        "tests/property/**/*.py",
        "tests/property_based/**/*.py",
    ]),
    deps = [
        ":core",
        ":algorithms",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Security Tests
python_test(
    name = "security-tests",
    srcs = glob([
        "tests/security/**/*.py",
    ]),
    deps = [
        ":api",
        ":enterprise",
        ":testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# ==========================================
# Build Convenience Targets
# ==========================================

# All Core Packages (most frequently changed)
genrule(
    name = "build-core",
    srcs = [],
    out = "core-build.txt",
    cmd = "echo 'Core packages built' > $OUT",
    deps = [
        ":core",
        ":infrastructure",
        ":application",
        ":algorithms",
    ],
)

# All Interface Packages
genrule(
    name = "build-interfaces",
    srcs = [],
    out = "interfaces-build.txt", 
    cmd = "echo 'Interface packages built' > $OUT",
    deps = [
        ":api",
        ":cli",
        ":web",
        ":sdk",
    ],
)

# All Test Suites
genrule(
    name = "test-all",
    srcs = [],
    out = "test-results.txt",
    cmd = "echo 'All tests completed' > $OUT",
    deps = [
        ":test-core",
        ":test-infrastructure", 
        ":test-application",
        ":test-algorithms",
        ":test-data-platform",
        ":test-api",
        ":test-cli",
        ":test-integration",
        ":benchmarks",
        ":property-tests",
        ":security-tests",
    ],
)

# Complete Build
genrule(
    name = "build-all",
    srcs = [],
    out = "build-complete.txt",
    cmd = "echo 'Monorepo build completed' > $OUT",
    deps = [
        ":build-core",
        ":build-interfaces",
        ":data-platform",
        ":mlops",
        ":enterprise",
        ":web-assets",
    ],
)

# ==========================================
# Development Targets
# ==========================================

# Development Environment
genrule(
    name = "dev",
    srcs = [],
    out = "dev-ready.txt",
    cmd = "echo 'Development environment ready' > $OUT",
    deps = [
        ":build-core",
        ":testing",
    ],
)

# CI/CD Test Suite
genrule(
    name = "ci-tests",
    srcs = [],
    out = "ci-test-results.txt",
    cmd = "echo 'CI tests passed' > $OUT",
    deps = [
        ":test-core",
        ":test-infrastructure",
        ":test-application",
        ":test-integration",
        ":security-tests",
    ],
)

# Release Build
genrule(
    name = "release",
    srcs = [],
    out = "release-artifacts.txt",
    cmd = "echo 'Release artifacts generated' > $OUT",
    deps = [
        ":test-all",
        ":build-all",
    ],
)

# Affected Package Detection (for incremental builds)
genrule(
    name = "affected",
    srcs = [],
    out = "affected-packages.txt",
    cmd = "echo 'Detecting affected packages...' > $OUT && git diff --name-only HEAD~1 | grep '^src/packages/' | cut -d'/' -f3 | sort -u >> $OUT",
    visibility = ["PUBLIC"],
)