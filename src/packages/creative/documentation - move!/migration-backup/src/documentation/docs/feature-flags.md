# Pynomaly Feature Flags

This document provides comprehensive documentation for Pynomaly's feature flag system, which enables controlled rollout of features and functionality.

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Feature Categories](#feature-categories)
4. [Development Stages](#development-stages)
5. [Feature Definitions](#feature-definitions)
6. [Usage Examples](#usage-examples)
7. [Environment Variables](#environment-variables)
8. [CLI Commands](#cli-commands)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Overview

Pynomaly's feature flag system provides:

- **Controlled Rollout**: Enable/disable features without code changes
- **Dependency Management**: Automatic validation of feature dependencies
- **Conflict Resolution**: Prevention of incompatible feature combinations
- **Environment Awareness**: Different configurations for dev/staging/production
- **Validation**: Runtime checks for required packages and dependencies

### Key Benefits

- **Risk Mitigation**: Test features in controlled environments
- **Performance Control**: Enable expensive features only when needed
- **Backward Compatibility**: Maintain stable APIs while adding functionality
- **Resource Management**: Control resource-intensive features
- **Gradual Migration**: Smooth transition between legacy and new implementations

## Configuration

### Basic Setup

Feature flags are configured through environment variables with the prefix `PYNOMALY_`:

```bash
# Enable AutoML features
export PYNOMALY_AUTOML=true

# Enable performance monitoring
export PYNOMALY_PERFORMANCE_MONITORING=true

# Enable advanced explainability
export PYNOMALY_ADVANCED_EXPLAINABILITY=true
```

### Programmatic Configuration

```python
from pynomaly.infrastructure.config.feature_flags import FeatureFlags

# Initialize with custom values
flags = FeatureFlags(
    automl=True,
    performance_monitoring=True,
    jwt_authentication=False
)

# Check if a feature is enabled
if flags.automl:
    # AutoML functionality
    pass
```

## Feature Categories

Features are organized into logical categories for better management:

### Core
Essential platform functionality that forms the foundation of Pynomaly.

**Features:**
- `mobile_interface` - Mobile-responsive interface

### Analytics
Data analysis, monitoring, and reporting capabilities.

**Features:**
- `performance_monitoring` - Real-time performance tracking
- `explainability_integration` - SHAP/LIME model explainability
- `advanced_explainability` - Advanced explainable AI features
- `real_time_dashboards` - Real-time analytics dashboards
- `business_intelligence` - Advanced BI reporting and analytics
- `automl_experiment_tracking` - Comprehensive experiment tracking

### Automation
Automated processes and intelligent decision-making systems.

**Features:**
- `ensemble_intelligence` - Smart ensemble composition
- `advanced_automl` - Advanced AutoML with multi-objective optimization
- `meta_learning` - Meta-learning from optimization history
- `ensemble_optimization` - Advanced ensemble optimization
- All `automl_*` features - Various AutoML capabilities

### Performance
System optimization and resource management features.

**Features:**
- `algorithm_optimization` - Algorithm performance optimization
- `memory_efficiency` - Memory-efficient data processing
- `automl_distributed_search` - Distributed hyperparameter search
- `automl_early_stopping` - Early stopping in optimization
- `automl_resource_management` - Resource-aware optimization
- `distributed_computing` - Distributed processing capabilities

### Enterprise
Business-grade features for production deployments.

**Features:**
- `jwt_authentication` - JWT-based authentication
- `data_encryption` - Data encryption at rest and in transit
- `audit_logging` - Comprehensive audit logging
- `sso_integration` - SSO (SAML/OAuth) integration
- `compliance_features` - GDPR/HIPAA compliance features
- `multi_tenancy` - Multi-tenant isolation

### Integrations
External system connections and third-party integrations.

**Features:**
- `database_connectivity` - PostgreSQL/MySQL connectivity
- `cloud_storage` - S3/Azure Blob storage integration
- `monitoring_integration` - Prometheus/Grafana integration
- `deep_learning_adapters` - PyTorch/TensorFlow deep learning adapters
- `deep_learning` - Deep learning anomaly detection
- `progressive_web_app` - PWA features with offline capabilities
- `streaming_analytics` - Real-time streaming data processing

## Development Stages

Features progress through defined maturity stages:

### Experimental
Early-stage features with potential breaking changes.

**Characteristics:**
- May have incomplete functionality
- API may change without notice
- Suitable for development/testing only
- Not recommended for production

**Examples:**
- `ensemble_intelligence`
- `jwt_authentication`
- `automl_distributed_search`
- `automl_neural_architecture_search`

### Beta
Stable features ready for testing in controlled environments.

**Characteristics:**
- Feature-complete but may have minor issues
- API is relatively stable
- Suitable for staging environments
- Use with caution in production

**Examples:**
- `algorithm_optimization`
- `memory_efficiency`
- `advanced_automl`
- `meta_learning`

### Stable
Production-ready features with guaranteed stability.

**Characteristics:**
- Fully tested and documented
- Stable API with backward compatibility
- Default enabled for essential features
- Safe for all environments

**Examples:**
- `automl_hyperparameter_optimization`
- `automl_feature_engineering`
- `automl_model_selection`
- `automl_early_stopping`

### Deprecated
Legacy features being phased out.

**Characteristics:**
- No longer actively maintained
- Will be removed in future versions
- Migration path to newer alternatives available

**Examples:**
- `automl_framework` (superseded by `advanced_automl`)

## Feature Definitions

### Core Enhancement Features (Phase 2)

#### algorithm_optimization
- **Category**: Performance
- **Stage**: Beta
- **Default**: `true`
- **Description**: Algorithm performance optimization
- **Dependencies**: None
- **Required Packages**: `pyod`, `scikit-learn`

#### memory_efficiency
- **Category**: Performance  
- **Stage**: Beta
- **Default**: `true`
- **Description**: Memory-efficient data processing
- **Dependencies**: `algorithm_optimization`

#### performance_monitoring
- **Category**: Analytics
- **Stage**: Beta
- **Default**: `true`
- **Description**: Real-time performance tracking
- **Required Packages**: `prometheus-client`

#### ensemble_intelligence
- **Category**: Automation
- **Stage**: Experimental
- **Default**: `false`
- **Description**: Smart ensemble composition
- **Dependencies**: `algorithm_optimization`

### AutoML Features (Phase 3A)

#### advanced_automl
- **Category**: Automation
- **Stage**: Beta
- **Default**: `true`
- **Description**: Advanced AutoML with multi-objective optimization
- **Dependencies**: `algorithm_optimization`, `performance_monitoring`
- **Required Packages**: `optuna`

#### automl_hyperparameter_optimization
- **Category**: Automation
- **Stage**: Stable
- **Default**: `true`
- **Description**: Hyperparameter optimization in AutoML
- **Dependencies**: `advanced_automl`
- **Required Packages**: `optuna`, `scikit-optimize`

#### automl_feature_engineering
- **Category**: Automation
- **Stage**: Stable
- **Default**: `true`
- **Description**: Automated feature engineering in AutoML
- **Dependencies**: `advanced_automl`
- **Required Packages**: `scikit-learn`, `pandas`

#### automl_model_selection
- **Category**: Automation
- **Stage**: Stable
- **Default**: `true`
- **Description**: Automated model selection in AutoML
- **Dependencies**: `advanced_automl`

#### automl_ensemble_creation
- **Category**: Automation
- **Stage**: Beta
- **Default**: `true`
- **Description**: Ensemble model creation in AutoML
- **Dependencies**: `automl_model_selection`, `ensemble_optimization`

#### automl_pipeline_optimization
- **Category**: Automation
- **Stage**: Beta
- **Default**: `true`
- **Description**: End-to-end pipeline optimization
- **Dependencies**: `automl_feature_engineering`, `automl_hyperparameter_optimization`

#### automl_distributed_search
- **Category**: Performance
- **Stage**: Experimental
- **Default**: `false`
- **Description**: Distributed hyperparameter search
- **Dependencies**: `automl_hyperparameter_optimization`
- **Required Packages**: `dask`, `ray`
- **Conflicts**: `automl_neural_architecture_search`

#### automl_neural_architecture_search
- **Category**: Automation
- **Stage**: Experimental
- **Default**: `false`
- **Description**: Neural architecture search (NAS)
- **Dependencies**: `deep_learning`, `automl_hyperparameter_optimization`
- **Required Packages**: `torch`, `tensorflow`
- **Conflicts**: `automl_distributed_search`

### Enterprise Features (Phase 3)

#### jwt_authentication
- **Category**: Enterprise
- **Stage**: Experimental
- **Default**: `false`
- **Description**: JWT-based authentication
- **Required Packages**: `pyjwt`, `passlib`

#### data_encryption
- **Category**: Enterprise
- **Stage**: Experimental
- **Default**: `false`
- **Description**: Data encryption at rest and in transit

#### audit_logging
- **Category**: Enterprise
- **Stage**: Experimental
- **Default**: `false`
- **Description**: Comprehensive audit logging

### Deep Learning Features

#### deep_learning
- **Category**: Integrations
- **Stage**: Beta
- **Default**: `true`
- **Description**: Deep learning anomaly detection with PyTorch, TensorFlow, and JAX
- **Dependencies**: `algorithm_optimization`

#### deep_learning_adapters
- **Category**: Integrations
- **Stage**: Experimental
- **Default**: `false`
- **Description**: PyTorch/TensorFlow deep learning adapters
- **Dependencies**: `algorithm_optimization`
- **Required Packages**: `torch`, `tensorflow`

### User Experience Features (Phase 3C)

#### real_time_dashboards
- **Category**: Analytics
- **Stage**: Beta
- **Default**: `true`
- **Description**: Real-time analytics dashboards
- **Dependencies**: `performance_monitoring`

#### progressive_web_app
- **Category**: Integrations
- **Stage**: Beta
- **Default**: `true`
- **Description**: PWA features with offline capabilities

#### mobile_interface
- **Category**: Core
- **Stage**: Beta
- **Default**: `true`
- **Description**: Mobile-responsive interface
- **Dependencies**: `progressive_web_app`

## Usage Examples

### Basic Feature Checking

```python
from pynomaly.infrastructure.config.feature_flags import feature_flags

# Check if AutoML is enabled
if feature_flags.is_enabled('automl'):
    print("AutoML features are available")

# Get all enabled features
enabled = feature_flags.get_enabled_features()
print(f"Enabled features: {enabled}")
```

### Using Decorators

```python
from pynomaly.infrastructure.config.feature_flags import require_feature, require_automl

@require_feature('advanced_explainability')
def generate_explanations(model, data):
    """Generate model explanations using SHAP/LIME."""
    # Implementation only runs if feature is enabled
    pass

@require_automl
def run_automl_optimization(dataset):
    """Run AutoML optimization pipeline."""
    # Implementation only runs if any AutoML feature is enabled
    pass
```

### Conditional Imports

```python
from pynomaly.infrastructure.config.feature_flags import conditional_import

# Import TensorFlow only if deep learning is enabled
tf = conditional_import('deep_learning', 'tensorflow')

if tf:
    # Use TensorFlow functionality
    pass
```

### Feature Validation

```python
from pynomaly.infrastructure.config.feature_flags import feature_flags

# Validate feature compatibility
issues = feature_flags.validate_feature_compatibility()

if issues:
    print("Feature compatibility issues:")
    for category, problems in issues.items():
        print(f"  {category}: {problems}")
```

### AutoML Configuration

```python
from pynomaly.infrastructure.config.feature_flags import (
    get_automl_configuration,
    validate_automl_environment
)

# Get comprehensive AutoML configuration
config = get_automl_configuration()
print(f"AutoML features enabled: {config['feature_count']}")
print(f"Environment valid: {config['configuration_valid']}")

if config['warnings']:
    print("Warnings:", config['warnings'])

# Detailed environment validation
validation = validate_automl_environment()
if not validation['valid']:
    print("AutoML environment issues:")
    print("Errors:", validation['errors'])
    print("Missing packages:", validation['missing_packages'])
```

## Environment Variables

All feature flags can be configured via environment variables with the `PYNOMALY_` prefix:

### Core Features
```bash
export PYNOMALY_ALGORITHM_OPTIMIZATION=true
export PYNOMALY_MEMORY_EFFICIENCY=true
export PYNOMALY_PERFORMANCE_MONITORING=true
export PYNOMALY_ENSEMBLE_INTELLIGENCE=false
```

### AutoML Features
```bash
export PYNOMALY_AUTOML=true
export PYNOMALY_ADVANCED_AUTOML=true
export PYNOMALY_AUTOML_HYPERPARAMETER_OPTIMIZATION=true
export PYNOMALY_AUTOML_FEATURE_ENGINEERING=true
export PYNOMALY_AUTOML_MODEL_SELECTION=true
export PYNOMALY_AUTOML_ENSEMBLE_CREATION=true
export PYNOMALY_AUTOML_DISTRIBUTED_SEARCH=false
export PYNOMALY_AUTOML_NEURAL_ARCHITECTURE_SEARCH=false
```

### Enterprise Features
```bash
export PYNOMALY_JWT_AUTHENTICATION=false
export PYNOMALY_DATA_ENCRYPTION=false
export PYNOMALY_AUDIT_LOGGING=false
export PYNOMALY_SSO_INTEGRATION=false
export PYNOMALY_COMPLIANCE_FEATURES=false
```

### Deep Learning Features
```bash
export PYNOMALY_DEEP_LEARNING=true
export PYNOMALY_DEEP_LEARNING_ADAPTERS=false
export PYNOMALY_ADVANCED_EXPLAINABILITY=true
```

### User Experience Features
```bash
export PYNOMALY_REAL_TIME_DASHBOARDS=true
export PYNOMALY_PROGRESSIVE_WEB_APP=true
export PYNOMALY_MOBILE_INTERFACE=true
export PYNOMALY_BUSINESS_INTELLIGENCE=true
```

## CLI Commands

### Check Feature Status
```bash
# Check specific feature
pynomaly config check-feature automl

# List all enabled features
pynomaly config list-features --enabled

# List all features with details
pynomaly config list-features --verbose

# Check AutoML configuration
pynomaly config automl-status
```

### Validate Configuration
```bash
# Validate all feature dependencies
pynomaly config validate

# Validate AutoML environment
pynomaly config validate-automl

# Check for missing packages
pynomaly config check-packages
```

### Feature Management
```bash
# Enable a feature
pynomaly config enable automl_distributed_search

# Disable a feature
pynomaly config disable automl_neural_architecture_search

# Reset to defaults
pynomaly config reset-defaults
```

## Troubleshooting

### Common Issues

#### Missing Dependencies
```
Error: Feature 'automl_ensemble_creation' requires 'automl_model_selection'
```

**Solution**: Enable the required dependency:
```bash
export PYNOMALY_AUTOML_MODEL_SELECTION=true
```

#### Package Not Found
```
Warning: Package 'optuna' required for automl_hyperparameter_optimization not found
```

**Solution**: Install the required package:
```bash
pip install optuna
```

#### Feature Conflicts
```
Error: automl_distributed_search conflicts with automl_neural_architecture_search
```

**Solution**: Disable one of the conflicting features:
```bash
export PYNOMALY_AUTOML_NEURAL_ARCHITECTURE_SEARCH=false
```

### Debugging

#### Check Feature Status
```python
from pynomaly.infrastructure.config.feature_flags import feature_flags

# Get detailed feature information
info = feature_flags.get_feature_info('automl_distributed_search')
print(f"Stage: {info.stage}")
print(f"Dependencies: {info.dependencies}")
print(f"Conflicts: {info.conflicts}")
print(f"Required packages: {info.required_packages}")
```

#### Validate Environment
```python
from pynomaly.infrastructure.config.feature_flags import validate_automl_environment

validation = validate_automl_environment()
if not validation['valid']:
    print("Issues found:")
    for category, issues in validation.items():
        if issues:
            print(f"  {category}: {issues}")
```

### Performance Considerations

#### Resource-Intensive Features
These features may require significant computational resources:

- `automl_distributed_search` - Requires cluster setup
- `automl_neural_architecture_search` - GPU-intensive
- `distributed_computing` - Multi-node processing
- `streaming_analytics` - High memory usage

#### Memory Usage
Monitor memory usage when enabling multiple features:

```python
import psutil

# Check memory usage
memory = psutil.virtual_memory()
if memory.percent > 80:
    print("High memory usage detected - consider disabling resource-intensive features")
```

## Best Practices

### Development Workflow

1. **Start Simple**: Begin with core features only
2. **Gradual Enablement**: Add features incrementally
3. **Test Thoroughly**: Validate each feature addition
4. **Monitor Resources**: Watch memory and CPU usage
5. **Document Changes**: Keep track of enabled features

### Environment Configuration

#### Development
```bash
# Enable all development features
export PYNOMALY_AUTOML=true
export PYNOMALY_DEEP_LEARNING=true
export PYNOMALY_ADVANCED_EXPLAINABILITY=true
export PYNOMALY_PERFORMANCE_MONITORING=true
```

#### Staging
```bash
# Enable stable and beta features
export PYNOMALY_AUTOML=true
export PYNOMALY_AUTOML_HYPERPARAMETER_OPTIMIZATION=true
export PYNOMALY_AUTOML_FEATURE_ENGINEERING=true
export PYNOMALY_PERFORMANCE_MONITORING=true
export PYNOMALY_REAL_TIME_DASHBOARDS=true
```

#### Production
```bash
# Enable only stable, essential features
export PYNOMALY_AUTOML=true
export PYNOMALY_AUTOML_HYPERPARAMETER_OPTIMIZATION=true
export PYNOMALY_AUTOML_FEATURE_ENGINEERING=true
export PYNOMALY_PERFORMANCE_MONITORING=true
export PYNOMALY_HEALTH_MONITORING=true
```

### Security Considerations

- **Enterprise Features**: Only enable in secure environments
- **Authentication**: Use JWT auth for production deployments
- **Encryption**: Enable for sensitive data processing
- **Audit Logging**: Enable for compliance requirements

### Performance Optimization

1. **Profile Usage**: Monitor feature performance impact
2. **Selective Enabling**: Only enable needed features
3. **Resource Limits**: Set appropriate memory/CPU limits
4. **Caching**: Leverage caching features when available

### Monitoring and Alerting

```python
# Set up feature usage monitoring
from pynomaly.infrastructure.config.feature_flags import feature_flags

enabled_features = feature_flags.get_enabled_features()
feature_count = len(enabled_features)

if feature_count > 20:
    print("Warning: High number of features enabled - monitor performance")

# Check for experimental features in production
experimental_features = feature_flags.get_features_by_stage(FeatureStage.EXPERIMENTAL)
enabled_experimental = [
    name for name in experimental_features 
    if feature_flags.is_enabled(name)
]

if enabled_experimental and ENVIRONMENT == 'production':
    print(f"Warning: Experimental features enabled in production: {enabled_experimental}")
```

### Migration Strategies

When migrating from legacy features:

1. **Parallel Enablement**: Run old and new features side by side
2. **Gradual Transition**: Migrate components incrementally  
3. **Validation**: Verify equivalent functionality
4. **Rollback Plan**: Maintain ability to revert changes
5. **Documentation**: Update configuration and procedures

This comprehensive documentation ensures teams can effectively use Pynomaly's feature flag system for controlled, safe feature rollout and management.