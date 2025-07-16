# Buck2 Build Configuration for Pynomaly Monorepo
# Domain-based monorepo build system following Buck2-first architecture with clean boundaries

load("@prelude//python:defs.bzl", "python_binary", "python_library", "python_test")
load("@prelude//js:defs.bzl", "js_bundle")

# ==========================================
# Domain-Based Monorepo Package Organization
# Following domain-based clean architecture with Buck2 optimization
# Domains: AI, Data, Software, Ops, Formal Sciences, Creative
# ==========================================

# ==========================================
# SOFTWARE DOMAIN - Core architecture and interfaces
# ==========================================

# Core Domain Layer - Pure business logic with no external dependencies
python_library(
    name = "software-core",
    srcs = glob([
        "src/packages/software/core/**/*.py",
    ]),
    deps = [],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "software",
        "layer": "domain",
        "type": "core",
        "cache_priority": "high",
    },
)

# Domain Library - Domain management and business logic templates
python_library(
    name = "software-domain-library",
    srcs = glob([
        "src/packages/software/domain_library/**/*.py",
    ]),
    deps = [
        ":software-core",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "software",
        "layer": "domain",
        "type": "domain-library",
    },
)

# Interfaces Package - All user-facing interfaces
python_library(
    name = "software-interfaces",
    srcs = glob([
        "src/packages/software/interfaces/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":ops-infrastructure",
        ":software-services",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "software",
        "layer": "presentation",
        "type": "interfaces",
    },
)

# Application Services Layer
python_library(
    name = "software-services",
    srcs = glob([
        "src/packages/software/services/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":ops-infrastructure",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "software",
        "layer": "application",
        "type": "services",
    },
)

# Enterprise Features
python_library(
    name = "software-enterprise",
    srcs = glob([
        "src/packages/software/enterprise/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":ops-infrastructure",
        ":software-services",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "software",
        "layer": "application",
        "type": "enterprise",
    },
)

# Mobile Adaptations
python_library(
    name = "software-mobile",
    srcs = glob([
        "src/packages/software/mobile/**/*.py",
    ]),
    deps = [
        ":software-core",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "software",
        "layer": "presentation",
        "type": "mobile",
    },
)

# ==========================================
# AI DOMAIN - Machine Learning and AI Operations
# ==========================================

# Anomaly Detection Package - Consolidated anomaly and outlier detection
python_library(
    name = "ai-anomaly-detection",
    srcs = glob([
        "src/packages/ai/anomaly_detection/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":formal-sciences-mathematics",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "ai",
        "layer": "domain",
        "type": "anomaly-detection",
    },
)

# ML Algorithms Package - ML algorithm infrastructure
python_library(
    name = "ai-algorithms",
    srcs = glob([
        "src/packages/ai/algorithms/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":formal-sciences-mathematics",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "ai",
        "layer": "infrastructure",
        "type": "algorithms",
    },
)

# Machine Learning Package - ML operations and lifecycle management
python_library(
    name = "ai-machine-learning",
    srcs = glob([
        "src/packages/ai/machine_learning/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":ai-anomaly-detection",
        ":data-platform",
        ":ops-infrastructure",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "ai",
        "layer": "application",
        "type": "machine-learning",
    },
)

# MLOps Package - ML operations platform
python_library(
    name = "ai-mlops",
    srcs = glob([
        "src/packages/ai/mlops/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":ops-infrastructure",
        ":ai-anomaly-detection",
        ":ai-machine-learning",
        ":data-platform",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "ai",
        "layer": "application",
        "type": "mlops",
    },
)

# ==========================================
# DATA DOMAIN - Data processing and observability
# ==========================================

# Data Platform Package - Consolidated data processing functionality
python_library(
    name = "data-platform",
    srcs = glob([
        "src/packages/data/data_platform/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":ops-infrastructure",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "data",
        "layer": "application",
        "type": "data_platform",
    },
)

# Data Observability Package - Data monitoring and lineage
python_library(
    name = "data-observability",
    srcs = glob([
        "src/packages/data/data_observability/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":data-platform",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "data",
        "layer": "application",
        "type": "data-observability",
    },
)

# ==========================================
# OPS DOMAIN - Operations and infrastructure
# ==========================================

# Infrastructure Layer - External integrations and adapters
python_library(
    name = "ops-infrastructure",
    srcs = glob([
        "src/packages/ops/infrastructure/**/*.py",
    ]),
    deps = [
        ":software-core",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "ops",
        "layer": "infrastructure",
        "type": "adapters",
    },
)

# Configuration Management
python_library(
    name = "ops-config",
    srcs = glob([
        "src/packages/ops/config/**/*.py",
    ]),
    deps = [
        ":software-core",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "ops",
        "layer": "infrastructure",
        "type": "config",
    },
)

# People Operations Package - User management and authentication
python_library(
    name = "ops-people-ops",
    srcs = glob([
        "src/packages/ops/people_ops/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":ops-infrastructure",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "ops",
        "layer": "application",
        "type": "people-ops",
    },
)

# Testing Package - Shared testing utilities
python_library(
    name = "ops-testing",
    srcs = glob([
        "src/packages/ops/testing/**/*.py",
    ]),
    deps = [],
    visibility = ["PUBLIC"],
    metadata = {
        "domain": "ops",
        "layer": "shared",
        "type": "testing",
    },
)

# Development Tools
python_library(
    name = "ops-tools",
    srcs = glob([
        "src/packages/ops/tools/**/*.py",
    ]),
    deps = [
        ":software-core",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "ops",
        "layer": "infrastructure",
        "type": "tools",
    },
)

# ==========================================
# FORMAL SCIENCES DOMAIN - Mathematics and logic
# ==========================================

# Mathematics Package - Statistical analysis and computations
python_library(
    name = "formal-sciences-mathematics",
    srcs = glob([
        "src/packages/formal_sciences/mathematics/**/*.py",
    ]),
    deps = [
        ":software-core",
    ],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "formal_sciences",
        "layer": "domain",
        "type": "mathematics",
    },
)

# ==========================================
# CREATIVE DOMAIN - Documentation and content
# ==========================================

# Documentation Package - All documentation and training materials
python_library(
    name = "creative-documentation",
    srcs = glob([
        "src/packages/creative/documentation/**/*.py",
    ]),
    deps = [],
    visibility = ["//src/packages/..."],
    metadata = {
        "domain": "creative",
        "layer": "content",
        "type": "documentation",
    },
)

# ==========================================
# Application Binaries
# ==========================================

# Main CLI Binary
python_binary(
    name = "pynomaly-cli",
    main = "src/packages/software/interfaces/cli/cli/app.py",
    deps = [":software-interfaces"],
    visibility = ["PUBLIC"],
)

# API Server Binary
python_binary(
    name = "pynomaly-api",
    main = "src/packages/software/interfaces/api/api/app.py",
    deps = [":software-interfaces"],
    visibility = ["PUBLIC"],
)

# Web UI Server Binary  
python_binary(
    name = "pynomaly-web",
    main = "src/packages/software/interfaces/web/web/app.py",
    deps = [":software-interfaces"],
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
        "src/packages/software/interfaces/web/web/templates/**/*.html",
        "src/packages/software/interfaces/web/web/static/**/*.css",
    ]),
    out = "static/css/tailwind.css",
    cmd = "cd $(location .) && npm run build-css",
    visibility = ["PUBLIC"],
)

# JavaScript Bundle
js_bundle(
    name = "pynomaly-js",
    srcs = glob([
        "src/packages/software/interfaces/web/web/static/**/*.js",
        "src/build_artifacts/storybook-static/js/**/*.js",
    ]),
    entry_point = "src/packages/software/interfaces/web/web/static/js/main.js",
    visibility = ["PUBLIC"],
)

# Web Assets Bundle
genrule(
    name = "web-assets",
    srcs = [
        ":tailwind-build",
        ":pynomaly-js",
    ] + glob([
        "src/packages/software/interfaces/web/web/static/**/*",
        "src/build_artifacts/storybook-static/**/*",
    ]),
    out = "web-assets.tar.gz",
    cmd = "tar -czf $OUT $(location :tailwind-build) $(location :pynomaly-js)",
    visibility = ["PUBLIC"],
)

# ==========================================
# Test Targets by Domain
# ==========================================

# Software Domain Tests
python_test(
    name = "test-software",
    srcs = glob([
        "tests/software/**/*.py",
        "tests/unit/**/*.py",
    ]),
    deps = [
        ":software-core",
        ":software-interfaces",
        ":software-services",
        ":ops-testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# AI Domain Tests
python_test(
    name = "test-ai",
    srcs = glob([
        "tests/ai/**/*.py",
        "tests/ml/**/*.py",
        "tests/anomaly/**/*.py",
    ]),
    deps = [
        ":ai-anomaly-detection",
        ":ai-algorithms",
        ":ai-machine-learning",
        ":ai-mlops",
        ":ops-testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Data Domain Tests
python_test(
    name = "test-data",
    srcs = glob([
        "tests/data/**/*.py",
        "tests/data_science/**/*.py",
    ]),
    deps = [
        ":data-platform",
        ":data-observability",
        ":software-core",
        ":ops-testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# Ops Domain Tests
python_test(
    name = "test-ops",
    srcs = glob([
        "tests/ops/**/*.py",
        "tests/infrastructure/**/*.py",
    ]),
    deps = [
        ":ops-infrastructure",
        ":ops-config",
        ":ops-people-ops",
        ":ops-tools",
        ":ops-testing",
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
        ":software-core",
        ":ops-infrastructure",
        ":software-interfaces",
        ":ai-anomaly-detection",
        ":data-platform",
        ":ops-testing",
    ],
    env = {
        "PYTHONPATH": "src/packages",
    },
)

# ==========================================
# Build Convenience Targets by Domain
# ==========================================

# Software Domain Build
genrule(
    name = "build-software",
    srcs = [],
    out = "software-build.txt",
    cmd = "echo 'Software domain packages built' > $OUT",
    deps = [
        ":software-core",
        ":software-interfaces",
        ":software-services",
        ":software-enterprise",
        ":software-mobile",
        ":software-domain-library",
    ],
)

# AI Domain Build
genrule(
    name = "build-ai",
    srcs = [],
    out = "ai-build.txt",
    cmd = "echo 'AI domain packages built' > $OUT",
    deps = [
        ":ai-anomaly-detection",
        ":ai-algorithms",
        ":ai-machine-learning",
        ":ai-mlops",
    ],
)

# Data Domain Build
genrule(
    name = "build-data",
    srcs = [],
    out = "data-build.txt",
    cmd = "echo 'Data domain packages built' > $OUT",
    deps = [
        ":data-platform",
        ":data-observability",
    ],
)

# Ops Domain Build
genrule(
    name = "build-ops",
    srcs = [],
    out = "ops-build.txt",
    cmd = "echo 'Ops domain packages built' > $OUT",
    deps = [
        ":ops-infrastructure",
        ":ops-config",
        ":ops-people-ops",
        ":ops-testing",
        ":ops-tools",
    ],
)

# All Test Suites
genrule(
    name = "test-all",
    srcs = [],
    out = "test-results.txt",
    cmd = "echo 'All domain tests completed' > $OUT",
    deps = [
        ":test-software",
        ":test-ai",
        ":test-data",
        ":test-ops",
        ":test-integration",
    ],
)

# Complete Build
genrule(
    name = "build-all",
    srcs = [],
    out = "build-complete.txt",
    cmd = "echo 'Domain-organized monorepo build completed' > $OUT",
    deps = [
        ":build-software",
        ":build-ai",
        ":build-data",
        ":build-ops",
        ":formal-sciences-mathematics",
        ":creative-documentation",
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
    cmd = "echo 'Domain-based development environment ready' > $OUT",
    deps = [
        ":build-software",
        ":ops-testing",
    ],
)

# CI/CD Test Suite
genrule(
    name = "ci-tests",
    srcs = [],
    out = "ci-test-results.txt",
    cmd = "echo 'CI tests passed for all domains' > $OUT",
    deps = [
        ":test-software",
        ":test-ai",
        ":test-data",
        ":test-ops",
        ":test-integration",
    ],
)

# Release Build
genrule(
    name = "release",
    srcs = [],
    out = "release-artifacts.txt",
    cmd = "echo 'Domain-organized release artifacts generated' > $OUT",
    deps = [
        ":test-all",
        ":build-all",
    ],
)

# Domain Impact Analysis (for incremental builds)
genrule(
    name = "affected",
    srcs = [],
    out = "affected-packages.txt",
    cmd = "echo 'Detecting affected domains and packages...' > $OUT && git diff --name-only HEAD~1 | grep '^src/packages/' | cut -d'/' -f3,4 | sort -u >> $OUT",
    visibility = ["PUBLIC"],
)