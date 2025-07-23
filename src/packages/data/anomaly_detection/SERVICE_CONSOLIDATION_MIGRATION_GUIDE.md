# Service Consolidation Migration Guide

## Overview

This guide helps you migrate from the legacy service architecture to the new consolidated services. The service count has been reduced from 12 to 6 core services while maintaining all functionality.

## Architecture Changes

### Before (12 Services)
```
Domain Services:
├── analytics_service.py (433 lines)
├── batch_processing_service.py (538 lines)
├── data_conversion_service.py (373 lines)
├── data_profiling_service.py (615 lines)
├── data_sampling_service.py (443 lines)
├── data_validation_service.py (495 lines)
├── detection_service.py (562 lines)
├── detection_service_simple.py (223 lines)
├── ensemble_service.py (333 lines)
├── explainability_service.py (502 lines)
├── health_monitoring_service.py (671 lines)
└── streaming_service.py (460 lines)
```

### After (6 Core Services)
```
Core Services:
├── DetectionService (kept as-is)
├── EnsembleService (kept as-is)
├── DataProcessingService (consolidates 4 services)
├── ProcessingService (consolidates 2 services)
├── EnhancedAnalyticsService (enhanced + health monitoring)
└── ExplainabilityService (kept as-is)
```

## Migration Mappings

### 1. Data Operations → DataProcessingService

**Old Usage:**
```python
from anomaly_detection.domain.services import (
    DataValidationService,
    DataProfilingService,
    DataSamplingService,
    DataConversionService
)

# Multiple service instances
validator = DataValidationService()
profiler = DataProfilingService()
sampler = DataSamplingService()
converter = DataConversionService()

# Multiple calls
validation_result = await validator.validate_file(file_path)
profile_result = await profiler.profile_file(file_path)
sample_result = await sampler.sample_file(file_path, 1000)
converted_file = await converter.convert_file(file_path, 'parquet', output_dir)
```

**New Usage:**
```python
from anomaly_detection.domain.services import DataProcessingService

# Single service instance
data_processor = DataProcessingService()

# All operations through unified interface
validation_result = await data_processor.validate_file(file_path)
profile_result = await data_processor.profile_file(file_path)
sample_result = await data_processor.sample_file(file_path, 1000)
converted_file = await data_processor.convert_file(file_path, 'parquet', output_dir)

# Batch operations (new capability)
batch_result = await data_processor.process_batch(
    operation='validate',
    file_paths=[file1, file2, file3],
    output_dir=output_dir
)
```

### 2. Processing Operations → ProcessingService

**Old Usage:**
```python
from anomaly_detection.domain.services import (
    BatchProcessingService,
    StreamingService
)

# Separate services for batch and streaming
batch_service = BatchProcessingService(detection_service)
streaming_service = StreamingService(detection_service)

# Batch processing
batch_result = await batch_service.process_file(input_file, output_dir, config)

# Streaming processing
streaming_service.start_streaming()
result = streaming_service.process_stream_sample(sample)
```

**New Usage:**
```python
from anomaly_detection.domain.services import ProcessingService

# Unified service for both modes
processor = ProcessingService()

# Batch processing (same interface)
batch_result = await processor.process_file(input_file, output_dir, config)
batch_results = await processor.process_batch(file_list, output_dir, config)

# Streaming processing (same interface)
processor.start_streaming()
result = processor.process_stream_sample(sample)
stats = processor.get_streaming_stats()
processor.stop_streaming()

# Auto-detection mode (new capability)
result = await processor.auto_process(data_source, config, mode="auto")
```

### 3. Analytics + Health Monitoring → EnhancedAnalyticsService

**Old Usage:**
```python
from anomaly_detection.domain.services import (
    AnalyticsService,
    HealthMonitoringService
)

# Separate services
analytics = AnalyticsService()
health_monitor = HealthMonitoringService()

# Record metrics separately
analytics.record_detection(result, processing_time, data_size)
await health_monitor.start_monitoring()

# Get data from different sources
dashboard_data = analytics.get_dashboard_data()
health_status = await health_monitor.get_health_status()
```

**New Usage:**
```python
from anomaly_detection.domain.services import EnhancedAnalyticsService

# Single service with both capabilities
analytics = EnhancedAnalyticsService()

# Start integrated monitoring
await analytics.start_health_monitoring()

# Record metrics (same interface)
analytics.record_detection(result, processing_time, data_size)

# Get unified dashboard data (includes health)
dashboard_data = analytics.get_dashboard_data()  # Now includes health metrics
health_summary = analytics.get_health_summary()
alerts = analytics.get_active_alerts()

# Context manager support (new)
async with EnhancedAnalyticsService() as analytics:
    # Automatic monitoring lifecycle management
    analytics.record_detection(result, processing_time, data_size)
```

## Breaking Changes

### 1. Import Changes

**Old Imports:**
```python
from anomaly_detection.domain.services import (
    DataValidationService,
    DataProfilingService,
    DataSamplingService,
    DataConversionService,
    BatchProcessingService,
    StreamingService,
    AnalyticsService,
    HealthMonitoringService
)
```

**New Imports:**
```python
from anomaly_detection.domain.services import (
    DataProcessingService,
    ProcessingService,
    EnhancedAnalyticsService,
    # Core services remain the same
    DetectionService,
    EnsembleService,
    ExplainabilityService
)
```

### 2. Initialization Changes

**Old Initialization:**
```python
# Multiple service instances
services = {
    'validator': DataValidationService(),
    'profiler': DataProfilingService(), 
    'sampler': DataSamplingService(),
    'converter': DataConversionService(),
    'batch_processor': BatchProcessingService(detection_service),
    'streaming': StreamingService(detection_service),
    'analytics': AnalyticsService(),
    'health_monitor': HealthMonitoringService()
}
```

**New Initialization:**
```python
# Fewer service instances
services = {
    'data_processor': DataProcessingService(),
    'processor': ProcessingService(detection_service),
    'analytics': EnhancedAnalyticsService(),
    # Core services
    'detector': DetectionService(),
    'ensemble': EnsembleService(),
    'explainer': ExplainabilityService()
}
```

## New Capabilities

### 1. Batch Operations
```python
# Process multiple files with unified operations
data_processor = DataProcessingService()

# Batch validation
validation_results = await data_processor.process_batch(
    operation='validate',
    file_paths=[file1, file2, file3],
    output_dir=output_dir,
    check_types=True,
    check_missing=True
)

# Batch profiling
profiling_results = await data_processor.process_batch(
    operation='profile', 
    file_paths=file_list,
    output_dir=output_dir,
    include_correlations=True
)
```

### 2. Auto-Processing Mode
```python
# Automatically choose best processing approach
processor = ProcessingService()

result = await processor.auto_process(
    data_source=file_list,
    config={'algorithm': 'isolation_forest'},
    mode='auto'  # Automatically selects batch vs streaming
)
```

### 3. Integrated Health Monitoring
```python
# Analytics with built-in health monitoring
async with EnhancedAnalyticsService() as analytics:
    # Health monitoring starts automatically
    
    # Record detection events
    analytics.record_detection(result, processing_time, data_size)
    
    # Get comprehensive dashboard (includes health)
    dashboard = analytics.get_dashboard_data()
    # dashboard now includes:
    # - analytics metrics
    # - health status
    # - system alerts
    # - performance trends
```

## Migration Steps

### Step 1: Update Imports
Replace legacy service imports with consolidated service imports.

### Step 2: Update Service Initialization  
Replace multiple service instances with fewer consolidated instances.

### Step 3: Update Method Calls
Most method calls remain the same, but now go through consolidated services.

### Step 4: Leverage New Capabilities
Use batch operations, auto-processing, and integrated health monitoring.

### Step 5: Remove Legacy Code
Remove old service instances and update configuration.

## Example: Complete Migration

### Before (Legacy Code)
```python
import asyncio
from anomaly_detection.domain.services import (
    DataValidationService,
    DataProfilingService,
    BatchProcessingService,
    AnalyticsService,
    HealthMonitoringService
)

class LegacyAnomalyDetectionPipeline:
    def __init__(self):
        self.validator = DataValidationService()
        self.profiler = DataProfilingService()
        self.batch_processor = BatchProcessingService()
        self.analytics = AnalyticsService()
        self.health_monitor = HealthMonitoringService()
    
    async def process_data(self, files):
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        results = []
        for file_path in files:
            # Validate data
            validation = await self.validator.validate_file(file_path)
            if not validation['is_valid']:
                continue
            
            # Profile data  
            profile = await self.profiler.profile_file(file_path)
            
            # Process file
            result = await self.batch_processor.process_file(
                file_path, Path('./output'), {}
            )
            
            # Record analytics
            self.analytics.record_detection(
                result, result['processing_time'], result['total_samples']
            )
            
            results.append(result)
        
        # Get dashboard data and health status separately
        dashboard = self.analytics.get_dashboard_data()
        health = await self.health_monitor.get_health_status()
        
        return {
            'results': results,
            'dashboard': dashboard, 
            'health': health
        }
```

### After (Consolidated Code)
```python
import asyncio
from anomaly_detection.domain.services import (
    DataProcessingService,
    ProcessingService,
    EnhancedAnalyticsService
)

class ModernAnomalyDetectionPipeline:
    def __init__(self):
        self.data_processor = DataProcessingService()
        self.processor = ProcessingService()
    
    async def process_data(self, files):
        # Use integrated analytics with health monitoring
        async with EnhancedAnalyticsService() as analytics:
            
            # Batch validation (more efficient)
            validation_results = await self.data_processor.process_batch(
                operation='validate',
                file_paths=files,
                output_dir=Path('./output')
            )
            
            # Filter valid files
            valid_files = [
                files[i] for i, result in enumerate(validation_results['results'])
                if result['success']
            ]
            
            # Batch processing (more efficient)
            processing_results = await self.processor.process_batch(
                input_files=valid_files,
                output_dir=Path('./output'),
                config={'algorithm': 'isolation_forest'}
            )
            
            # Record analytics for all successful detections
            for result in processing_results['results']:
                if result['success']:
                    analytics.record_detection(
                        result['result'], 
                        result['result']['processing_time'],
                        result['result']['total_samples']
                    )
            
            # Get unified dashboard data (includes health and analytics)
            dashboard = analytics.get_dashboard_data()
            
            return {
                'processing_results': processing_results,
                'dashboard': dashboard  # Now includes health metrics
            }
```

## Benefits of Migration

1. **Reduced Complexity**: 12 → 6 services (-50% reduction)
2. **Better Performance**: Batch operations reduce overhead
3. **Unified Interface**: Consistent API across related operations
4. **Enhanced Monitoring**: Integrated health monitoring with analytics
5. **Easier Maintenance**: Fewer dependencies and interfaces to manage
6. **New Capabilities**: Auto-processing, batch operations, integrated health

## Support

- **Legacy services remain available** during transition period
- **Gradual migration** is supported - update one component at a time
- **Full backward compatibility** for existing method signatures
- **Enhanced error handling** and logging in consolidated services

## Timeline

- **Phase 1** (Immediate): New consolidated services available alongside legacy
- **Phase 2** (Next release): Legacy services marked as deprecated 
- **Phase 3** (Future release): Legacy services removed

Start migration today to benefit from improved performance and reduced complexity!