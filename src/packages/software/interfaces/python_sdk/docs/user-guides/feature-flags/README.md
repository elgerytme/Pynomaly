# Feature Flags Guide

Pynomaly uses a comprehensive feature flag system to enable controlled rollout of advanced features while maintaining system stability and simplicity for basic use cases.

## Overview

Feature flags allow you to:

- Enable/disable specific features at runtime
- Control system complexity based on your needs
- Manage experimental features safely
- Configure enterprise vs basic functionality

## Quick Start

Set environment variables to enable features:

```bash
# Enable AutoML features
export PYNOMALY_AUTOML=true

# Enable advanced monitoring
export PYNOMALY_PERFORMANCE_MONITORING=true

# Enable enterprise authentication
export PYNOMALY_JWT_AUTHENTICATION=true
```

## Environment Variables

All feature flags use the `PYNOMALY_` prefix and can be set as environment variables:

| Feature | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| AutoML | `PYNOMALY_AUTOML` | `true` | Enable AutoML hyperparameter optimization |
| Advanced AutoML | `PYNOMALY_ADVANCED_AUTOML` | `true` | Enable advanced AutoML with multi-objective optimization |
| Performance Monitoring | `PYNOMALY_PERFORMANCE_MONITORING` | `true` | Enable real-time performance tracking |
| JWT Authentication | `PYNOMALY_JWT_AUTHENTICATION` | `false` | Enable JWT-based authentication |
| Data Encryption | `PYNOMALY_DATA_ENCRYPTION` | `false` | Enable data encryption at rest and in transit |
| Deep Learning | `PYNOMALY_DEEP_LEARNING` | `true` | Enable PyTorch/TensorFlow deep learning features |
| Explainability | `PYNOMALY_ADVANCED_EXPLAINABILITY` | `true` | Enable SHAP/LIME explainability features |

## Feature Categories

### Core Enhancement Features (Default: Enabled)

- **Algorithm Optimization**: Performance improvements for pattern algorithms
- **Memory Efficiency**: Memory-efficient streaming processing
- **Performance Monitoring**: Real-time system monitoring
- **CLI Simplification**: Simplified command-line workflows

### Enterprise Features (Default: Disabled)

- **JWT Authentication**: Enterprise-grade authentication
- **Data Encryption**: End-to-end data encryption
- **Audit Logging**: Comprehensive audit trails
- **SSO Integration**: Single Sign-On support

### Advanced ML Features (Default: Enabled)

- **AutoML**: Automated hyperparameter optimization
- **Deep Learning**: Neural network-based anomaly detection
- **Explainability**: Model interpretation and explanation
- **Ensemble Optimization**: Advanced ensemble methods

### Integration Features (Default: Disabled)

- **Database Connectivity**: PostgreSQL/MySQL integration
- **Cloud Storage**: S3/Azure Blob storage
- **Monitoring Integration**: Prometheus/Grafana

## Feature Stages

Features are categorized by maturity:

- **Stable**: Production-ready features (enabled by default)
- **Beta**: Well-tested features (selectively enabled)
- **Experimental**: Early-stage features (disabled by default)
- **Deprecated**: Legacy features (disabled by default)

## Usage Examples

### Enabling AutoML Features

```bash
# Set environment variable
export PYNOMALY_AUTOML=true

# Run AutoML optimization
pynomaly automl optimize data.csv IsolationForest --max-time 1800
```

### Configuration File

Create a `.env` file for persistent configuration:

```bash
# Core features
PYNOMALY_AUTOML=true
PYNOMALY_PERFORMANCE_MONITORING=true
PYNOMALY_DEEP_LEARNING=true

# Enterprise features (enable as needed)
PYNOMALY_JWT_AUTHENTICATION=false
PYNOMALY_DATA_ENCRYPTION=false
PYNOMALY_AUDIT_LOGGING=false

# Integration features
PYNOMALY_DATABASE_CONNECTIVITY=false
PYNOMALY_CLOUD_STORAGE=false
```

### Programmatic Configuration

```python
from pynomaly.infrastructure.config.feature_flags import feature_flags

# Check if a feature is enabled
if feature_flags.is_enabled("automl"):
    print("AutoML is enabled")

# Get all enabled features
enabled_features = feature_flags.get_enabled_features()
print(f"Enabled features: {enabled_features}")

# Validate feature compatibility
issues = feature_flags.validate_feature_compatibility()
if issues:
    print(f"Configuration issues: {issues}")
```

### Decorators for Feature-Gated Functions

```python
from pynomaly.infrastructure.config.feature_flags import require_feature, require_automl

@require_feature("advanced_automl")
def advanced_optimization():
    """Function that requires advanced AutoML features."""
    pass

@require_automl
def any_automl_function():
    """Function that requires any AutoML features."""
    pass
```

## Feature Dependencies

Some features depend on others:

- **Advanced AutoML** requires **Algorithm Optimization** and **Performance Monitoring**
- **Meta Learning** requires **Advanced AutoML**
- **Mobile Interface** requires **Progressive Web App**
- **Business Intelligence** requires **Real-time Dashboards**

The system automatically validates dependencies and reports conflicts.

## Best Practices

### Development Environment

```bash
# Enable all development features
export PYNOMALY_AUTOML=true
export PYNOMALY_DEEP_LEARNING=true
export PYNOMALY_PERFORMANCE_MONITORING=true
export PYNOMALY_ADVANCED_EXPLAINABILITY=true
```

### Production Environment

```bash
# Enable only stable features
export PYNOMALY_AUTOML=true
export PYNOMALY_PERFORMANCE_MONITORING=true
export PYNOMALY_HEALTH_MONITORING=true

# Enterprise features as needed
export PYNOMALY_JWT_AUTHENTICATION=true
export PYNOMALY_AUDIT_LOGGING=true
```

### Testing Environment

```bash
# Enable experimental features for testing
export PYNOMALY_STREAMING_ANALYTICS=true
export PYNOMALY_DISTRIBUTED_COMPUTING=true
export PYNOMALY_GRAPH_ANALYTICS=true
```

## Troubleshooting

### Feature Not Available Error

```
RuntimeError: Feature 'automl' is not enabled. Enable it by setting PYNOMALY_AUTOML=true
```

**Solution**: Set the required environment variable:

```bash
export PYNOMALY_AUTOML=true
```

### Dependency Conflicts

```
Configuration issues: {'missing_dependencies': ['advanced_automl requires algorithm_optimization']}
```

**Solution**: Enable required dependencies:

```bash
export PYNOMALY_ALGORITHM_OPTIMIZATION=true
export PYNOMALY_ADVANCED_AUTOML=true
```

### Missing Package Requirements

```
Configuration issues: {'missing_packages': ['automl requires package: optuna']}
```

**Solution**: Install required packages:

```bash
pip install optuna
```

## Feature Flag API Reference

### Core Functions

- `feature_flags.is_enabled(feature_name)`: Check if feature is enabled
- `feature_flags.get_enabled_features()`: Get all enabled features
- `feature_flags.validate_feature_compatibility()`: Validate configuration
- `feature_flags.get_feature_info(feature_name)`: Get feature metadata

### Convenience Functions

- `is_automl_enabled()`: Check AutoML availability
- `is_enterprise_features_enabled()`: Check enterprise features
- `is_advanced_analytics_enabled()`: Check analytics features
- `is_algorithm_optimization_enabled()`: Check optimization features

### Decorators

- `@require_feature(feature_name)`: Require specific feature
- `@require_automl`: Require any AutoML feature
- `@conditional_import(feature_name, module_name)`: Conditional imports

## Configuration Examples

### Minimal Configuration (Basic Usage)

```bash
# Only core detection features
PYNOMALY_AUTOML=false
PYNOMALY_DEEP_LEARNING=false
PYNOMALY_PERFORMANCE_MONITORING=true
```

### Standard Configuration (Recommended)

```bash
# Balanced feature set
PYNOMALY_AUTOML=true
PYNOMALY_DEEP_LEARNING=true
PYNOMALY_PERFORMANCE_MONITORING=true
PYNOMALY_ADVANCED_EXPLAINABILITY=true
```

### Full Configuration (All Features)

```bash
# Enable everything (development/testing)
PYNOMALY_AUTOML=true
PYNOMALY_ADVANCED_AUTOML=true
PYNOMALY_DEEP_LEARNING=true
PYNOMALY_PERFORMANCE_MONITORING=true
PYNOMALY_ADVANCED_EXPLAINABILITY=true
PYNOMALY_JWT_AUTHENTICATION=true
PYNOMALY_DATA_ENCRYPTION=true
PYNOMALY_REAL_TIME_DASHBOARDS=true
PYNOMALY_PROGRESSIVE_WEB_APP=true
```

## Support

For feature flag related issues:

1. Check feature dependencies and conflicts
2. Verify required packages are installed
3. Validate environment variable syntax
4. Review feature maturity (experimental features may be unstable)

See [Configuration Reference](../../reference/configuration/README.md) for detailed configuration options.
