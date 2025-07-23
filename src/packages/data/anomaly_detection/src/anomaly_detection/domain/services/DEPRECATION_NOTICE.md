# 📋 Service Deprecation Notice

## Overview

As part of the service consolidation initiative to reduce architectural complexity, several legacy services have been **deprecated** in favor of new consolidated services that provide the same functionality with better performance and maintainability.

## Deprecated Services

The following services are now **deprecated** and will be removed in a future release:

### Data Processing Services (Consolidated into `DataProcessingService`)
- ❌ `DataValidationService` → ✅ Use `DataProcessingService.validate_file()`
- ❌ `DataProfilingService` → ✅ Use `DataProcessingService.profile_file()`
- ❌ `DataSamplingService` → ✅ Use `DataProcessingService.sample_file()`
- ❌ `DataConversionService` → ✅ Use `DataProcessingService.convert_file()`

### Processing Services (Consolidated into `ProcessingService`)
- ❌ `BatchProcessingService` → ✅ Use `ProcessingService.process_file()` or `ProcessingService.process_batch()`
- ❌ `StreamingService` → ✅ Use `ProcessingService.start_streaming()` and related methods

### Analytics & Monitoring Services (Consolidated into `EnhancedAnalyticsService`)
- ❌ `AnalyticsService` → ✅ Use `EnhancedAnalyticsService` (all methods preserved)  
- ❌ `HealthMonitoringService` → ✅ Use `EnhancedAnalyticsService` health monitoring methods

## Services Kept As-Is

The following core services remain unchanged:
- ✅ `DetectionService` - Core anomaly detection functionality
- ✅ `DetectionServiceSimple` - Lightweight detection alternative
- ✅ `EnsembleService` - Algorithm combination capabilities
- ✅ `ExplainabilityService` - Model interpretation features

## Migration Timeline

### Phase 1: Deprecation Warning (Current)
- Legacy services are marked as deprecated but remain functional
- New consolidated services are available and recommended
- Migration guide provided for smooth transition

### Phase 2: Deprecation Enforcement (Next Release)
- Warnings will be logged when using legacy services
- Documentation will redirect to consolidated services
- New features will only be added to consolidated services

### Phase 3: Removal (Future Release)
- Legacy services will be completely removed
- Only consolidated services will be available
- Breaking change will be clearly communicated

## Migration Resources

1. **Migration Guide**: See `SERVICE_CONSOLIDATION_MIGRATION_GUIDE.md` for detailed migration instructions
2. **API Documentation**: All consolidated services have comprehensive docstrings
3. **Example Code**: Migration guide includes before/after code examples
4. **Support**: Legacy imports remain available during transition period

## Benefits of Migration

### Performance Improvements
- **50% fewer service instances** to manage
- **Batch operations** reduce processing overhead
- **Unified interfaces** eliminate duplicate code paths
- **Better resource utilization** through consolidated operations

### Developer Experience
- **Simpler imports** - fewer services to remember
- **Consistent APIs** - unified method signatures across related operations
- **Better error handling** - centralized error management
- **Enhanced logging** - structured logging across all operations

### Maintainability
- **Reduced codebase** - from 12 services to 6 core services
- **Lower complexity** - fewer interdependencies to manage
- **Easier testing** - consolidated test suites
- **Clearer responsibility** - well-defined service boundaries

## Quick Migration Examples

### Data Operations
```python
# OLD (Deprecated)
from anomaly_detection.domain.services import DataValidationService
validator = DataValidationService()
result = await validator.validate_file(file_path)

# NEW (Recommended) 
from anomaly_detection.domain.services import DataProcessingService
processor = DataProcessingService()
result = await processor.validate_file(file_path)
```

### Processing Operations
```python
# OLD (Deprecated)
from anomaly_detection.domain.services import BatchProcessingService
batch_service = BatchProcessingService()
result = await batch_service.process_file(file_path, output_dir, config)

# NEW (Recommended)
from anomaly_detection.domain.services import ProcessingService  
processor = ProcessingService()
result = await processor.process_file(file_path, output_dir, config)
```

### Analytics & Health Monitoring
```python
# OLD (Deprecated)
from anomaly_detection.domain.services import AnalyticsService, HealthMonitoringService
analytics = AnalyticsService()
health_monitor = HealthMonitoringService()

# NEW (Recommended)
from anomaly_detection.domain.services import EnhancedAnalyticsService
analytics = EnhancedAnalyticsService()  # Includes health monitoring
```

## Deprecation Warnings

When using deprecated services, you will see warnings like:

```
DeprecationWarning: DataValidationService is deprecated. 
Use DataProcessingService.validate_file() instead. 
See SERVICE_CONSOLIDATION_MIGRATION_GUIDE.md for migration instructions.
This service will be removed in version 2.0.0.
```

## Action Required

1. **Review your code** for usage of deprecated services
2. **Plan migration** using the provided migration guide
3. **Test thoroughly** after migrating to consolidated services
4. **Update documentation** and examples in your projects
5. **Train team members** on the new consolidated architecture

## Support & Questions

- **Migration Issues**: See the migration guide or create an issue
- **Feature Requests**: Request features for consolidated services only
- **Bug Reports**: Bugs in deprecated services have lower priority
- **Documentation**: Updated docs focus on consolidated services

---

**Start your migration today to ensure smooth transition and benefit from improved performance and reduced complexity!**