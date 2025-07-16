# Pynomaly Feature Flags Documentation

## Overview

Pynomaly uses a comprehensive feature flag system to control the availability of various features and components. This allows for gradual rollout of new functionality while maintaining system stability and simplicity for basic use cases.

## Feature Flag System

### Configuration

Feature flags are configured through environment variables with the prefix `PYNOMALY_`. They can be set to `true` or `false`.

### Core Feature Flag Manager

The feature flag system is centralized in `src/pynomaly/infrastructure/config/feature_flags.py` and provides:

- **Centralized Configuration**: All feature flags in one place
- **Dependency Management**: Features can depend on other features
- **Conflict Resolution**: Features can conflict with each other
- **Package Requirements**: Features can require specific packages
- **Development Stages**: Features are categorized by maturity (experimental, beta, stable, deprecated)

## Available Feature Flags

### Phase 2: Core Enhancement Features

#### `algorithm_optimization` (Default: `true`)

- **Environment Variable**: `PYNOMALY_ALGORITHM_OPTIMIZATION`
- **Description**: Enable algorithm performance optimization
- **Category**: Performance
- **Stage**: Beta
- **Required Packages**: `pyod`, `scikit-learn`

#### `memory_efficiency` (Default: `true`)

- **Environment Variable**: `PYNOMALY_MEMORY_EFFICIENCY`
- **Description**: Enable memory-efficient streaming processing
- **Category**: Performance
- **Stage**: Beta
- **Dependencies**: `algorithm_optimization`

#### `performance_monitoring` (Default: `true`)

- **Environment Variable**: `PYNOMALY_PERFORMANCE_MONITORING`
- **Description**: Enable real-time performance monitoring
- **Category**: Analytics
- **Stage**: Beta
- **Required Packages**: `prometheus-client`

#### `ensemble_intelligence` (Default: `false`)

- **Environment Variable**: `PYNOMALY_ENSEMBLE_INTELLIGENCE`
- **Description**: Enable smart ensemble composition
- **Category**: Automation
- **Stage**: Experimental
- **Dependencies**: `algorithm_optimization`

### Phase 3: Enterprise Features (Disabled by default)

#### `jwt_authentication` (Default: `false`)

- **Environment Variable**: `PYNOMALY_JWT_AUTHENTICATION`
- **Description**: Enable JWT-based authentication
- **Category**: Enterprise
- **Stage**: Experimental
- **Required Packages**: `pyjwt`, `passlib`

#### `data_encryption` (Default: `false`)

- **Environment Variable**: `PYNOMALY_DATA_ENCRYPTION`
- **Description**: Enable data encryption at rest and in transit
- **Category**: Enterprise
- **Stage**: Experimental

#### `audit_logging` (Default: `false`)

- **Environment Variable**: `PYNOMALY_AUDIT_LOGGING`
- **Description**: Enable comprehensive audit logging
- **Category**: Enterprise
- **Stage**: Experimental

#### `explainability_integration` (Default: `false`)

- **Environment Variable**: `PYNOMALY_EXPLAINABILITY_INTEGRATION`
- **Description**: Enable SHAP/LIME explainability features
- **Category**: Analytics
- **Stage**: Experimental
- **Required Packages**: `shap`, `lime`

### Phase 3A: Advanced ML/AI Features (Enabled for development)

#### `advanced_automl` (Default: `true`)

- **Environment Variable**: `PYNOMALY_ADVANCED_AUTOML`
- **Description**: Enable advanced AutoML with multi-objective optimization
- **Category**: Automation
- **Stage**: Beta
- **Dependencies**: `algorithm_optimization`, `performance_monitoring`
- **Required Packages**: `optuna`

#### `automl` (Default: `true`)

- **Environment Variable**: `PYNOMALY_AUTOML`
- **Description**: Enable AutoML features (alias for advanced_automl)
- **Category**: Automation
- **Stage**: Stable
- **Dependencies**: `algorithm_optimization`
- **Required Packages**: `optuna`

#### `meta_learning` (Default: `true`)

- **Environment Variable**: `PYNOMALY_META_LEARNING`
- **Description**: Enable meta-learning from optimization history
- **Category**: Automation
- **Stage**: Beta
- **Dependencies**: `advanced_automl`

#### `ensemble_optimization` (Default: `true`)

- **Environment Variable**: `PYNOMALY_ENSEMBLE_OPTIMIZATION`
- **Description**: Enable advanced ensemble optimization
- **Category**: Automation
- **Stage**: Beta
- **Dependencies**: `algorithm_optimization`

#### `deep_learning` (Default: `true`)

- **Environment Variable**: `PYNOMALY_DEEP_LEARNING`
- **Description**: Enable deep learning anomaly detection with PyTorch, TensorFlow, and JAX
- **Category**: Integrations
- **Stage**: Beta
- **Dependencies**: `algorithm_optimization`

#### `advanced_explainability` (Default: `true`)

- **Environment Variable**: `PYNOMALY_ADVANCED_EXPLAINABILITY`
- **Description**: Enable advanced explainable AI features
- **Category**: Analytics
- **Stage**: Beta
- **Required Packages**: `shap`, `lime`

#### `intelligent_selection` (Default: `true`)

- **Environment Variable**: `PYNOMALY_INTELLIGENT_SELECTION`
- **Description**: Enable intelligent algorithm selection with learning capabilities
- **Category**: Automation
- **Stage**: Beta

### Phase 3C: User Experience Features

#### `real_time_dashboards` (Default: `true`)

- **Environment Variable**: `PYNOMALY_REAL_TIME_DASHBOARDS`
- **Description**: Enable real-time analytics dashboards
- **Category**: Analytics
- **Stage**: Beta
- **Dependencies**: `performance_monitoring`

#### `progressive_web_app` (Default: `true`)

- **Environment Variable**: `PYNOMALY_PROGRESSIVE_WEB_APP`
- **Description**: Enable PWA features with offline capabilities
- **Category**: Integrations
- **Stage**: Beta

#### `mobile_interface` (Default: `true`)

- **Environment Variable**: `PYNOMALY_MOBILE_INTERFACE`
- **Description**: Enable mobile-responsive interface
- **Category**: Core
- **Stage**: Beta
- **Dependencies**: `progressive_web_app`

#### `business_intelligence` (Default: `true`)

- **Environment Variable**: `PYNOMALY_BUSINESS_INTELLIGENCE`
- **Description**: Enable advanced BI reporting and analytics
- **Category**: Analytics
- **Stage**: Beta
- **Dependencies**: `real_time_dashboards`

### Phase 4: Advanced Capabilities (Disabled by default)

#### `streaming_analytics` (Default: `false`)

- **Environment Variable**: `PYNOMALY_STREAMING_ANALYTICS`
- **Description**: Enable real-time streaming analytics
- **Category**: Integrations
- **Stage**: Experimental
- **Dependencies**: `memory_efficiency`, `performance_monitoring`
- **Conflicts**: `distributed_computing`
- **Required Packages**: `kafka-python`, `redis`

#### `distributed_computing` (Default: `false`)

- **Environment Variable**: `PYNOMALY_DISTRIBUTED_COMPUTING`
- **Description**: Enable distributed computing capabilities
- **Category**: Performance
- **Stage**: Experimental
- **Dependencies**: `algorithm_optimization`, `performance_monitoring`
- **Conflicts**: `streaming_analytics`
- **Required Packages**: `dask`, `ray`

#### `graph_analytics` (Default: `false`)

- **Environment Variable**: `PYNOMALY_GRAPH_ANALYTICS`
- **Description**: Enable graph anomaly detection
- **Category**: Analytics
- **Stage**: Experimental

#### `multi_modal_detection` (Default: `false`)

- **Environment Variable**: `PYNOMALY_MULTI_MODAL_DETECTION`
- **Description**: Enable text/image anomaly detection
- **Category**: Analytics
- **Stage**: Experimental

### Phase 5: Production Hardening

#### `health_monitoring` (Default: `true`)

- **Environment Variable**: `PYNOMALY_HEALTH_MONITORING`
- **Description**: Enable comprehensive health monitoring
- **Category**: Analytics
- **Stage**: Beta

#### `performance_optimization` (Default: `false`)

- **Environment Variable**: `PYNOMALY_PERFORMANCE_OPTIMIZATION`
- **Description**: Enable production performance optimization
- **Category**: Performance
- **Stage**: Experimental

#### `backup_recovery` (Default: `false`)

- **Environment Variable**: `PYNOMALY_BACKUP_RECOVERY`
- **Description**: Enable automated backup and recovery
- **Category**: Enterprise
- **Stage**: Experimental

#### `sso_integration` (Default: `false`)

- **Environment Variable**: `PYNOMALY_SSO_INTEGRATION`
- **Description**: Enable SSO (SAML/OAuth) integration
- **Category**: Enterprise
- **Stage**: Experimental

#### `compliance_features` (Default: `false`)

- **Environment Variable**: `PYNOMALY_COMPLIANCE_FEATURES`
- **Description**: Enable GDPR/HIPAA compliance features
- **Category**: Enterprise
- **Stage**: Experimental

#### `multi_tenancy` (Default: `false`)

- **Environment Variable**: `PYNOMALY_MULTI_TENANCY`
- **Description**: Enable multi-tenant isolation
- **Category**: Enterprise
- **Stage**: Experimental

## Using Feature Flags

### Environment Configuration

Set feature flags using environment variables:

```bash
# Enable AutoML features
export PYNOMALY_AUTOML=true
export PYNOMALY_ADVANCED_AUTOML=true

# Enable enterprise features
export PYNOMALY_JWT_AUTHENTICATION=true
export PYNOMALY_AUDIT_LOGGING=true

# Enable experimental features
export PYNOMALY_STREAMING_ANALYTICS=true
```

Or in a `.env` file:

```env
PYNOMALY_AUTOML=true
PYNOMALY_ADVANCED_AUTOML=true
PYNOMALY_DEEP_LEARNING=true
PYNOMALY_EXPLAINABILITY_INTEGRATION=true
```

### Programmatic Usage

#### Checking Feature Flags

```python
from pynomaly.infrastructure.config.feature_flags import feature_flags

# Check if a feature is enabled
if feature_flags.is_enabled("automl"):
    # AutoML functionality
    pass

# Get all enabled features
enabled_features = feature_flags.get_enabled_features()
print(f"Enabled features: {enabled_features}")
```

#### Using Decorators

```python
from pynomaly.infrastructure.config.feature_flags import require_feature, require_automl

@require_feature("advanced_explainability")
def generate_explanation(model, data):
    # This function only runs if advanced_explainability is enabled
    pass

@require_automl
def run_automl_optimization(dataset):
    # This function only runs if AutoML features are enabled
    pass
```

#### Conditional Imports

```python
from pynomaly.infrastructure.config.feature_flags import conditional_import

# Conditionally import based on feature flag
shap = conditional_import("explainability_integration", "shap", fallback=None)
if shap:
    # Use SHAP functionality
    pass
```

### API Endpoints

AutoML API endpoints automatically check for AutoML feature flags and return appropriate error messages when disabled:

```json
{
  "error": "AutoML features are not enabled",
  "message": "Enable AutoML by setting PYNOMALY_AUTOML=true or PYNOMALY_ADVANCED_AUTOML=true",
  "automl_enabled": false
}
```

## Feature Categories

### Core

Essential system functionality that should remain stable.

### Analytics

Features related to data analysis, monitoring, and insights.

### Integrations

Features that integrate with external systems or libraries.

### Enterprise

Advanced features typically required in enterprise environments.

### Automation

Features that provide automated functionality and intelligence.

### Performance

Features focused on system performance and optimization.

## Development Stages

### Experimental

New features under active development. May be unstable or incomplete.

- Default: Usually `false`
- Use with caution in production

### Beta

Features that are feature-complete but may need refinement.

- Default: Usually `true` for core features, `false` for advanced features
- Generally safe for development and testing

### Stable

Mature features that are production-ready.

- Default: `true`
- Safe for all environments

### Deprecated

Features that are being phased out.

- Default: `false`
- Should not be used in new implementations

## Validation and Compatibility

The feature flag system includes automatic validation:

```python
# Validate feature compatibility
issues = feature_flags.validate_feature_compatibility()
if issues:
    print("Feature compatibility issues:")
    for issue_type, problems in issues.items():
        print(f"  {issue_type}: {problems}")
```

### Dependency Validation

Features can depend on other features. The system will warn if dependencies are not met.

### Conflict Detection

Some features conflict with each other. The system will detect and report conflicts.

### Package Requirements

Features can require specific Python packages. The system will check for missing packages.

## Best Practices

### Development

1. **Start Simple**: Begin with core features enabled and gradually enable advanced features as needed.
2. **Test Incrementally**: Test each feature flag individually to understand its impact.
3. **Use Defaults**: The default settings are chosen to provide a good balance of functionality and stability.

### Production

1. **Gradual Rollout**: Enable new features gradually in production environments.
2. **Monitor Impact**: Use performance monitoring to track the impact of enabled features.
3. **Document Changes**: Keep track of which features are enabled in each environment.

### Configuration Management

1. **Environment-Specific**: Use different feature flag configurations for different environments.
2. **Version Control**: Keep feature flag configurations in version control.
3. **Documentation**: Document why specific features are enabled or disabled.

## Troubleshooting

### Common Issues

#### AutoML Not Working

```bash
# Check if AutoML is enabled
export PYNOMALY_AUTOML=true
# or
export PYNOMALY_ADVANCED_AUTOML=true
```

#### Missing Dependencies

If you see package-related errors, install the required packages:

```bash
# For AutoML features
pip install optuna

# For explainability features
pip install shap lime

# For authentication features
pip install pyjwt passlib
```

#### Feature Conflicts

If features conflict, disable one of the conflicting features:

```bash
# Disable distributed computing if using streaming analytics
export PYNOMALY_DISTRIBUTED_COMPUTING=false
export PYNOMALY_STREAMING_ANALYTICS=true
```

### Debugging

Enable debug logging to see feature flag status:

```python
import logging
logging.getLogger('pynomaly.infrastructure.config.feature_flags').setLevel(logging.DEBUG)
```

## Future Considerations

The feature flag system is designed to be extensible. New features can be added by:

1. Adding the feature flag to the `FeatureFlags` class
2. Creating a `FeatureDefinition` with appropriate metadata
3. Using the `@require_feature` decorator in code
4. Updating this documentation

## Related Documentation

- [Architecture Overview](../developer-guides/architecture/overview.md)
- [Configuration Guide](../getting-started/installation.md)
- [API Reference](../api/README.md)
- [Development Setup](../developer-guides/DEVELOPMENT_SETUP.md)
