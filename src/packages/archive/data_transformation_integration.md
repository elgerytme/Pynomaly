# Data Transformation Package Integration Guide

This document outlines the integration of the `data_transformation` package with other Pynomaly packages.

## Integration Overview

The `data_transformation` package has been successfully integrated with the following components:

### 1. Infrastructure Package Integration

**File**: `infrastructure/infrastructure/data_loaders/enhanced_data_loader_factory.py`

**Features Added**:
- Enhanced data loading with automatic preprocessing
- Intelligent transformation recommendations
- Quality assessment and validation
- Algorithm-specific optimization

**Usage**:
```python
from infrastructure.data_loaders.enhanced_data_loader_factory import EnhancedDataLoaderFactory

factory = EnhancedDataLoaderFactory(enable_auto_preprocessing=True)
transformed_data = factory.load_and_transform("data.csv")
recommendations = factory.get_transformation_recommendations("data.csv")
```

### 2. Services Package Integration

**File**: `services/services/enhanced_data_preprocessing_service.py`

**Features Added**:
- Advanced data quality assessment
- Anomaly detection optimized preprocessing
- Algorithm-specific optimizations
- Comprehensive quality reporting

**Usage**:
```python
from services.services.enhanced_data_preprocessing_service import EnhancedDataPreprocessingService

service = EnhancedDataPreprocessingService()
processed_data, quality_report = await service.preprocess_for_anomaly_detection(dataset)
recommendations = await service.get_preprocessing_recommendations(dataset)
```

### 3. API Package Integration

**File**: `api/api/endpoints/enhanced_datasets.py`

**New Endpoints Added**:
- `POST /{dataset_id}/advanced-transform` - Apply advanced transformations
- `GET /{dataset_id}/transformation-recommendations` - Get intelligent recommendations
- `POST /{dataset_id}/quality-assessment` - Comprehensive quality assessment
- `POST /{dataset_id}/optimize-for-algorithm` - Algorithm-specific optimization
- `POST /upload-and-transform` - Upload and transform in one step

**Usage**:
```bash
# Apply advanced transformations
curl -X POST "/datasets/{id}/advanced-transform" \
  -F "cleaning_strategy=auto" \
  -F "feature_engineering=true"

# Get recommendations
curl -X GET "/datasets/{id}/transformation-recommendations?anomaly_detection_type=unsupervised"
```

### 4. CLI Package Integration

**File**: `cli/cli/advanced_preprocessing.py`

**New Commands Added**:
- `transform` - Apply advanced data transformations
- `analyze` - Analyze dataset and get recommendations
- `quality-check` - Perform comprehensive quality assessment
- `optimize` - Optimize for specific algorithms

**Usage**:
```bash
# Transform dataset
pynomaly advanced-preprocessing transform data.csv --output clean_data.csv --verbose

# Analyze dataset
pynomaly advanced-preprocessing analyze data.csv --format table

# Quality check
pynomaly advanced-preprocessing quality-check data.csv --threshold 0.8

# Optimize for algorithm
pynomaly advanced-preprocessing optimize data.csv isolation_forest --output optimized_data.csv
```

## Dependencies Updated

### Infrastructure Package
Added `pynomaly-data-transformation>=0.1.0` to dependencies in `infrastructure/pyproject.toml`

## Integration Benefits

1. **Enhanced Data Quality**: Advanced cleaning and validation capabilities
2. **Improved Performance**: GPU acceleration and distributed processing support
3. **Better Feature Engineering**: Automated feature generation and selection
4. **Streaming Support**: Real-time data transformation capabilities
5. **Algorithm Optimization**: Tailored preprocessing for specific anomaly detection algorithms
6. **Clean Architecture**: Better separation of concerns and maintainability

## Configuration

### Environment Variables
```bash
# Enable advanced features
PYNOMALY_ENABLE_DATA_TRANSFORMATION=true

# Configure processing
PYNOMALY_DEFAULT_CLEANING_STRATEGY=auto
PYNOMALY_ENABLE_PARALLEL_PROCESSING=true
```

### Service Configuration
```yaml
# config/data_transformation.yml
data_transformation:
  enabled: true
  default_config:
    cleaning_strategy: "auto"
    scaling_method: "robust"
    encoding_strategy: "onehot"
    feature_engineering: true
    parallel_processing: true
```

## Error Handling

All integration points include graceful fallbacks when the `data_transformation` package is not available:

1. **Import Errors**: Graceful degradation to basic preprocessing
2. **Feature Flags**: Ability to disable advanced features
3. **Fallback Methods**: Basic preprocessing when advanced features fail

## Testing

### Unit Tests
- All integration points include comprehensive unit tests
- Mock objects for data_transformation components when not available
- Test both enabled and disabled modes

### Integration Tests
- End-to-end testing with real data transformation workflows
- Performance testing with large datasets
- Algorithm-specific optimization validation

## Migration Guide

### Existing Code Migration

1. **Replace Basic Data Loaders**:
```python
# Old
loader = DataLoaderFactory()
data = loader.load_data("data.csv")

# New
loader = EnhancedDataLoaderFactory()
data = loader.load_and_transform("data.csv")
```

2. **Upgrade Preprocessing Services**:
```python
# Old
preprocessor = DataPreprocessingService()
cleaned_data = preprocessor.clean_data(data)

# New
preprocessor = EnhancedDataPreprocessingService()
cleaned_data, report = await preprocessor.preprocess_for_anomaly_detection(data)
```

3. **Enhanced API Usage**:
```python
# Old
response = await client.post(f"/datasets/{id}/clean")

# New
response = await client.post(f"/datasets/{id}/advanced-transform")
```

## Performance Considerations

- **Memory Usage**: Enhanced processing may use more memory for large datasets
- **Processing Time**: Initial processing may be slower but results in better quality
- **Caching**: Transformation results are cached to improve subsequent performance
- **Parallel Processing**: Enabled by default for datasets > 10,000 rows

## Security

- **Input Validation**: All input data is validated before processing
- **Resource Limits**: Memory and CPU usage limits to prevent DoS
- **Access Control**: Enhanced endpoints respect existing authentication/authorization
- **Data Privacy**: No data is logged or stored beyond necessary caching

## Monitoring

Integration includes comprehensive monitoring:

- **Metrics**: Processing time, memory usage, success rates
- **Logging**: Structured logging for all transformation operations
- **Health Checks**: Service health endpoints for monitoring systems
- **Alerts**: Configurable alerts for processing failures or performance issues

## Future Enhancements

Planned improvements to the integration:

1. **Real-time Streaming**: Enhanced streaming data processing
2. **Distributed Processing**: Multi-node processing for large datasets
3. **AutoML Integration**: Automated preprocessing pipeline selection
4. **Custom Transformations**: User-defined transformation plugins
5. **Data Lineage**: Complete transformation history tracking

## Support

For issues or questions regarding the data_transformation integration:

1. Check the integration logs for error messages
2. Verify the data_transformation package is properly installed
3. Review the configuration settings
4. Consult the API documentation for endpoint specifications
5. Contact the development team for advanced support

## Version Compatibility

- **data_transformation**: >= 0.1.0
- **pynomaly-core**: >= 0.1.1
- **pynomaly-infrastructure**: Compatible with enhanced loader factory
- **Python**: >= 3.11

---

*Last updated: January 2025*
*Integration Version: 1.0.0*