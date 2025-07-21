# Migration Notice: Enterprise MLOps Package Consolidated

## ⚠️ Important: Package Moved

The standalone `enterprise_mlops` package has been **consolidated** into the main AI MLOps package located at:

**New Location:** `@src/packages/ai/machine_learning/mlops/`

## Why This Change?

1. **Reduced Complexity**: Single package for all MLOps needs
2. **Better Integration**: Core and enterprise features work together seamlessly  
3. **Simplified Dependencies**: Optional enterprise dependencies don't affect basic usage
4. **Unified API**: Single import path for all MLOps functionality

## Migration Guide

### Before (Old)
```python
from enterprise_mlops import MLOpsOrchestrationService
from enterprise_mlops.infrastructure.mlops.mlflow import MLflowIntegration
from enterprise_mlops.infrastructure.monitoring.datadog import DatadogIntegration
```

### After (New)
```python
from pynomaly_mlops import (
    EnterpriseMlopsService,
    MLflowIntegration, 
    DatadogIntegration
)
```

### Installation Changes

**Before:**
```bash
pip install pynomaly-enterprise-mlops[all]
```

**After:**
```bash
pip install pynomaly-mlops[enterprise]
```

## Feature Parity

All features from the standalone enterprise package are now available in the consolidated package:

✅ **MLflow Integration** - `pip install pynomaly-mlops[mlflow]`  
✅ **Kubeflow Integration** - `pip install pynomaly-mlops[kubeflow]`  
✅ **Datadog Monitoring** - `pip install pynomaly-mlops[datadog]`  
✅ **New Relic Monitoring** - `pip install pynomaly-mlops[newrelic]`  
✅ **All Enterprise Features** - `pip install pynomaly-mlops[enterprise]`

## Backward Compatibility

- **Entity Models**: Enhanced with optional enterprise fields
- **Service APIs**: Unified interface with enterprise capabilities
- **Optional Dependencies**: Enterprise features remain optional

## Timeline

- **Immediate**: This standalone package is deprecated
- **Next Release**: Will be removed from the codebase
- **Recommendation**: Migrate to consolidated package now

## Need Help?

See the updated documentation in the main MLOps package:
- `@src/packages/ai/machine_learning/mlops/README_ENTERPRISE.md`
- Examples and migration guides included

## Cleanup

This directory and its contents will be removed in the next cleanup phase.