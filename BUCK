# Buck2 Build Configuration for Data Intelligence Monorepo
# Matches actual repository structure as of 2025-01-21

load("@prelude//python:defs.bzl", "python_binary", "python_library", "python_test")
load("//tools/buck:python.bzl", "monorepo_workspace_package")  
load("//tools/buck:testing.bzl", "python_test_suite")
load("//tools/buck:advanced_features.bzl", "openapi_python_client", "ml_model_artifacts", "docker_image_build", "documentation_site")
load("//tools/buck:monitoring.bzl", "build_metrics_collector", "performance_dashboard", "build_performance_alerts")
load("//tools/buck:import_validation.bzl", "create_import_validation_suite", "create_import_fix_suite")
load("//tools/buck:tox_integration.bzl", "create_standard_tox_integration")
load("//tools/buck:performance_baselines.bzl", "create_performance_monitoring_suite")
load("//tools/buck:security_compliance.bzl", "create_security_suite", "create_compliance_report")
load("//tools/buck:analytics_dashboard.bzl", "create_analytics_suite")

# ==========================================
# AI DOMAIN - Machine Learning
# ==========================================

# Reference to self-contained package-level targets

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

# Data Engineering CLI applications are provided by individual packages

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
    name = "data-intelligence",
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
#     name = "data-intelligence-api-client",
#     openapi_spec = "docs/api/openapi.yaml",
#     client_name = "DataIntelligenceClient",
#     visibility = ["PUBLIC"],
# )

# ML Model Artifacts (example)
# ml_model_artifacts(
#     name = "data-intelligence-models",
#     model_script = "src/packages/data/data_analytics/scripts/train_model.py",
#     training_data = [
#         "data/training/analytics_dataset.csv",
#     ],
#     model_config = "configs/model_config.yaml",
#     output_formats = ["pkl", "onnx", "tensorflow"],
#     visibility = ["PUBLIC"],
# )

# Docker Images
docker_image_build(
    name = "data-intelligence-api-image",
    dockerfile = "docker/api/Dockerfile",
    srcs = [
        ":data-intelligence",
    ],
    tags = [
        "data-intelligence/api:latest",
        "data-intelligence/api:v0.1.0",
    ],
    build_args = {
        "PYTHON_VERSION": "3.11",
        "APP_ENV": "production",
    },
    visibility = ["PUBLIC"],
)

# Documentation Site
documentation_site(
    name = "data-intelligence-docs",
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
        "//:data-intelligence",
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
# DOCUMENTATION VALIDATION
# ==========================================

load("//tools/buck:report_validation.bzl", "create_report_validation_rule")
load("//tools/buck:root_documentation_validation.bzl", "create_root_documentation_validation_rule", "create_strict_root_documentation_validation_rule")

# Validate report locations across repository
create_report_validation_rule(
    name = "validate-report-locations",
    report_files = glob(["docs/reports/**/*"]),
    package_docs_paths = [
        "src/packages/*/docs/reports/",
        "src/packages/*/*/docs/reports/"
    ],
    visibility = ["PUBLIC"],
)

# Prevent new documentation files in repository root
create_root_documentation_validation_rule(
    name = "validate-root-documentation",
    visibility = ["PUBLIC"],
)

# Strict validation (also fails on grandfathered files)
create_strict_root_documentation_validation_rule(
    name = "validate-root-documentation-strict",
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
# ==========================================
# TOX INTEGRATION  
# ==========================================

# Create standard tox integration with comprehensive test environments
create_standard_tox_integration(
    name = "tox"
)

# ==========================================
# PERFORMANCE BASELINE MONITORING
# ==========================================

# Create performance monitoring suite for critical build targets
create_performance_monitoring_suite(
    name = "performance-monitoring",
    targets = [
        "//:data-intelligence",
        "//:ai-all",
        "//:data-all", 
        "//:enterprise-all"
    ],
    tolerance = 0.15,  # 15% regression tolerance
)

# ==========================================
# SECURITY AND COMPLIANCE
# ==========================================

# Create comprehensive security suite for all domains
create_security_suite(
    name = "security-suite",
    targets = [
        "//:ai-all",
        "//:data-all",
        "//:enterprise-all"
    ]
)

# Create compliance report for the entire monorepo
create_compliance_report(
    name = "compliance-report",
    packages = [
        "//:ai-all",
        "//:data-all", 
        "//:enterprise-all"
    ]
)

# ==========================================
# BXL INTEGRATION - IDE SUPPORT
# ==========================================

# Reference to BXL tools for IDE integration and development workflows
alias(
    name = "bxl-tools",
    actual = "//tools/bxl:all_bxl_tools",
    visibility = ["PUBLIC"],
)

# ==========================================
# ENHANCED BUILD ANALYTICS & VISUALIZATION
# ==========================================

# Create comprehensive analytics suite with interactive dashboard
create_analytics_suite(
    name = "build-analytics",
    targets = [
        "//:data-intelligence",
        "//:ai-all",
        "//:data-all",
        "//:enterprise-all"
    ]
)

# ==========================================
# SECURITY AND COMPLIANCE
# ==========================================

# Create comprehensive security and compliance suite for critical targets
create_security_compliance_suite(
    name = "security-compliance",
    targets = [
        "//:data-intelligence",
        "//:ai-all",
        "//:data-all", 
        "//:enterprise-all"
    ],
    scan_types = ["bandit", "safety", "semgrep", "dependency_check", "license_check"],
    fail_threshold = "high",
)
