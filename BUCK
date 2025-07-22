# Buck2 Build Configuration for Pynomaly Monorepo
# Matches actual repository structure as of 2025-01-21

load("@prelude//python:defs.bzl", "python_binary", "python_library", "python_test")
load("//tools/buck:python.bzl", "monorepo_workspace_package")  
load("//tools/buck:testing.bzl", "python_test_suite")
load("//tools/buck:advanced_features.bzl", "openapi_python_client", "ml_model_artifacts", "docker_image_build", "documentation_site")
load("//tools/buck:monitoring.bzl", "build_metrics_collector", "performance_dashboard", "build_performance_alerts")
load("//tools/buck:import_validation.bzl", "create_import_validation_suite", "create_import_fix_suite")

# ==========================================
# AI DOMAIN - Machine Learning and Anomaly Detection
# ==========================================

# Reference to self-contained package-level targets (moved to data domain)
alias(
    name = "data-anomaly-detection",
    actual = "//src/packages/data/anomaly_detection:anomaly_detection",
    visibility = ["PUBLIC"],
)

alias(
    name = "ai-machine-learning", 
    actual = "//src/packages/ai/machine_learning:machine_learning",
    visibility = ["PUBLIC"],
)

alias(
    name = "ai-mlops",
    actual = "//src/packages/ai/mlops:mlops",
    visibility = ["PUBLIC"],
)

alias(
    name = "ai-neuro-symbolic",
    actual = "//src/packages/ai/neuro_symbolic:neuro_symbolic",
    visibility = ["PUBLIC"],
)

# ==========================================
# DATA DOMAIN - Data Processing and Analytics
# ==========================================

# Reference to self-contained package-level targets
alias(
    name = "data-analytics",
    actual = "//src/packages/data/data_analytics:data_analytics",
    visibility = ["PUBLIC"],
)

alias(
    name = "data-engineering",
    actual = "//src/packages/data/data_engineering:data_engineering",
    visibility = ["PUBLIC"],
)

alias(
    name = "data-quality",
    actual = "//src/packages/data/data_quality:data_quality", 
    visibility = ["PUBLIC"],
)

alias(
    name = "data-observability",
    actual = "//src/packages/data/observability:data_observability",
    visibility = ["PUBLIC"],
)

alias(
    name = "data-profiling", 
    actual = "//src/packages/data/profiling:profiling",
    visibility = ["PUBLIC"],
)

alias(
    name = "data-architecture",
    actual = "//src/packages/data/data_architecture:data_architecture",
    visibility = ["PUBLIC"],
)

alias(
    name = "data-statistics",
    actual = "//src/packages/data/statistics:statistics",
    visibility = ["PUBLIC"],
)

# ==========================================
# ENTERPRISE DOMAIN - Authentication and Governance  
# ==========================================

# Reference to self-contained package-level targets
alias(
    name = "enterprise-auth",
    actual = "//src/packages/enterprise/enterprise_auth:enterprise_auth",
    visibility = ["PUBLIC"],
)

alias(
    name = "enterprise-governance",
    actual = "//src/packages/enterprise/enterprise_governance:enterprise_governance",
    visibility = ["PUBLIC"],
)

alias(
    name = "enterprise-scalability", 
    actual = "//src/packages/enterprise/enterprise_scalability:enterprise_scalability",
    visibility = ["PUBLIC"],
)

# ==========================================
# CLI APPLICATIONS - Command Line Interfaces
# ==========================================

# Data Engineering CLI
python_binary(
    name = "data-engineering-cli",
    main = "src/packages/data/data_engineering/presentation/cli/app.py",
    deps = [
        ":data-engineering",
        "//third-party/python:click",
    ],
    visibility = ["PUBLIC"],
)

# Anomaly Detection CLI  
python_binary(
    name = "anomaly-detection-cli",
    main = "src/packages/data/anomaly_detection/cli/__init__.py", 
    deps = [
        ":data-anomaly-detection",
        "//third-party/python:click",
    ],
    visibility = ["PUBLIC"],
)

# ==========================================
# TEST TARGETS - Automated Testing
# ==========================================

# AI Domain Tests
python_test_suite(
    name = "ai-tests",
    srcs = glob([
        "src/packages/ai/**/*_test.py",
        "src/packages/ai/**/test_*.py",
        "src/packages/ai/**/tests/**/*.py",
    ]),
    deps = [
        ":ai-machine-learning", 
        ":ai-mlops",
        ":ai-neuro-symbolic",
    ],
    visibility = ["PUBLIC"],
)

# Data Domain Tests
python_test_suite(
    name = "data-tests",
    srcs = glob([
        "src/packages/data/**/*_test.py",
        "src/packages/data/**/test_*.py", 
        "src/packages/data/**/tests/**/*.py",
    ]),
    deps = [
        ":data-anomaly-detection",
        ":data-analytics",
        ":data-engineering",
        ":data-quality",
        ":data-observability",
        ":data-profiling",
    ],
    visibility = ["PUBLIC"],
)

# Enterprise Domain Tests
python_test_suite(
    name = "enterprise-tests", 
    srcs = glob([
        "src/packages/enterprise/**/*_test.py",
        "src/packages/enterprise/**/test_*.py",
        "src/packages/enterprise/**/tests/**/*.py", 
    ]),
    deps = [
        ":enterprise-auth",
        ":enterprise-governance",
        ":enterprise-scalability",
    ],
    visibility = ["PUBLIC"],
)

# ==========================================
# AGGREGATE TARGETS - Convenient Build Groups
# ==========================================

# All AI Packages
python_library(
    name = "ai-all",
    deps = [
        ":ai-machine-learning",
        ":ai-mlops",
        ":ai-neuro-symbolic",
    ],
    visibility = ["PUBLIC"],
)

# All Data Packages
python_library(
    name = "data-all",
    deps = [
        ":data-anomaly-detection",
        ":data-analytics",
        ":data-engineering", 
        ":data-quality",
        ":data-observability",
        ":data-profiling",
    ],
    visibility = ["PUBLIC"],
)

# All Enterprise Packages
python_library(
    name = "enterprise-all",
    deps = [
        ":enterprise-auth",
        ":enterprise-governance",
        ":enterprise-scalability",
    ],
    visibility = ["PUBLIC"],
)

# Complete Monorepo
python_library(
    name = "pynomaly",
    deps = [
        ":ai-all",
        ":data-all", 
        ":enterprise-all",
    ],
    visibility = ["PUBLIC"],
)

# ==========================================
# ADVANCED FEATURES - Code Generation & Automation
# ==========================================

# API Client Generation (example)
# openapi_python_client(
#     name = "anomaly-detection-api-client",
#     openapi_spec = "docs/api/openapi.yaml",
#     client_name = "AnomalyDetectionClient",
#     visibility = ["PUBLIC"],
# )

# ML Model Artifacts (example)
# ml_model_artifacts(
#     name = "anomaly-detection-models",
#     model_script = "src/packages/data/anomaly_detection/scripts/train_model.py",
#     training_data = [
#         "data/training/anomaly_dataset.csv",
#     ],
#     model_config = "configs/model_config.yaml",
#     output_formats = ["pkl", "onnx", "tensorflow"],
#     visibility = ["PUBLIC"],
# )

# Docker Images
docker_image_build(
    name = "pynomaly-api-image",
    dockerfile = "docker/api/Dockerfile",
    srcs = [
        ":pynomaly",
    ],
    tags = [
        "pynomaly/api:latest",
        "pynomaly/api:v0.1.0",
    ],
    build_args = {
        "PYTHON_VERSION": "3.11",
        "APP_ENV": "production",
    },
    visibility = ["PUBLIC"],
)

# Documentation Site
documentation_site(
    name = "pynomaly-docs",
    markdown_files = glob([
        "docs/**/*.md",
        "README.md",
    ]),
    config_file = "docs/mkdocs.yml",
    theme = "material",
    visibility = ["PUBLIC"],
)

# ==========================================
# BUILD PERFORMANCE MONITORING
# ==========================================

# Metrics Collection
build_metrics_collector(
    name = "build-metrics-collector",
    targets = [
        "//:pynomaly",
        "//:ai-all",
        "//:data-all", 
        "//:enterprise-all"
    ],
    metrics_output = "metrics/build_metrics.json",
    enable_profiling = True,
    visibility = ["PUBLIC"],
)

# Performance Dashboard
performance_dashboard(
    name = "performance-dashboard",
    metrics_files = [
        "metrics/build_metrics.json",
    ],
    dashboard_port = 8080,
    visibility = ["PUBLIC"],
)

# Performance Alerts
build_performance_alerts(
    name = "build-alerts",
    metrics_file = "metrics/build_metrics.json",
    thresholds = {
        "max_duration": 300.0,  # 5 minutes
        "min_success_rate": 0.95,  # 95%
        "min_cache_hit_rate": 0.8,  # 80%
    },
    visibility = ["PUBLIC"],
)

# ==========================================
# IMPORT CONSOLIDATION VALIDATION
# ==========================================

# Create import validation suite for all packages
create_import_validation_suite(
    name = "import-validation-all",
    packages = {
        "ai.machine_learning": [":ai-machine-learning"],
        "ai.mlops": [":ai-mlops"],
        "ai.neuro_symbolic": [":ai-neuro-symbolic"],
        "data.anomaly_detection": [":data-anomaly-detection"],
        "data.analytics": [":data-analytics"],
        "data.engineering": [":data-engineering"],
        "data.quality": [":data-quality"],
        "data.observability": [":data-observability"],
        "data.profiling": [":data-profiling"],
        "data.architecture": [":data-architecture"],
        "data.statistics": [":data-statistics"],
        "enterprise.auth": [":enterprise-auth"],
        "enterprise.governance": [":enterprise-governance"],
        "enterprise.scalability": [":enterprise-scalability"],
    }
)

# Create import fix suite for all packages  
create_import_fix_suite(
    name = "import-fix-all",
    packages = {
        "ai.machine_learning": [":ai-machine-learning"],
        "ai.mlops": [":ai-mlops"],
        "ai.neuro_symbolic": [":ai-neuro-symbolic"],
        "data.anomaly_detection": [":data-anomaly-detection"],
        "data.analytics": [":data-analytics"],
        "data.engineering": [":data-engineering"],
        "data.quality": [":data-quality"],
        "data.observability": [":data-observability"],
        "data.profiling": [":data-profiling"],
        "data.architecture": [":data-architecture"],
        "data.statistics": [":data-statistics"],
        "enterprise.auth": [":enterprise-auth"],
        "enterprise.governance": [":enterprise-governance"],
        "enterprise.scalability": [":enterprise-scalability"],
    }
)