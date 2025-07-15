# Data Validation Pipeline

Comprehensive data validation and processing infrastructure for Pynomaly.

## Overview

The data validation pipeline provides:

- **Schema Validation**: JSON Schema-based validation
- **Type Checking**: Automatic type conversion and validation
- **Range Validation**: Numeric range constraints
- **Pattern Matching**: Regular expression validation
- **Outlier Detection**: Statistical outlier identification
- **Data Cleaning**: Automatic data cleaning and transformation
- **Error Handling**: Comprehensive error reporting and quarantine
- **Performance Monitoring**: Real-time processing statistics

## Components

### Validators

- `TypeValidator`: Data type validation and conversion
- `RangeValidator`: Numeric range constraints
- `PatternValidator`: Regular expression matching
- `LengthValidator`: String/collection length validation
- `NullValidator`: Null value handling
- `UniqueValidator`: Uniqueness constraints
- `OutlierValidator`: Statistical outlier detection
- `SchemaValidator`: JSON Schema validation

### Pipeline Processors

- `ValidationProcessor`: Data validation processing
- `TransformationProcessor`: Data transformation
- `EnrichmentProcessor`: Data enrichment
- `OutputProcessor`: Data output handling

### Pipeline Management

- `DataPipeline`: Main pipeline orchestration
- `PipelineManager`: Multiple pipeline management
- `DataRecord`: Data record with metadata

## Configuration

Validation pipelines are configured using YAML files:

```yaml
pipelines:
  user_input:
    validators:
      - type: "null"
        params:
          allow_null: false
          default_value: ""
      - type: "type"
        params:
          expected_type: "str"
      - type: "length"
        params:
          min_length: 1
          max_length: 100
```

## Usage

### Command Line

```bash
# Validate data file
python3 scripts/data/validate_data.py \
    --config config/data/validation_config.yml \
    --data data/input/sample.json \
    --pipeline user_input

# Run full pipeline
python3 scripts/data/run_pipeline.py \
    --config config/data/validation_config.yml \
    --input data/input/sample.json \
    --output data/output/processed.json
```

### Python API

```python
from pynomaly.infrastructure.data.validation import ValidationConfigLoader
from pynomaly.infrastructure.data.pipeline import DataPipeline, ValidationProcessor

# Load configuration
pipelines = ValidationConfigLoader.load_from_yaml("config.yml")

# Create pipeline
pipeline = DataPipeline("my_pipeline")
pipeline.add_processor(ValidationProcessor("validation", pipelines["user_input"]))

# Process data
pipeline.add_data({"name": "John", "email": "john@example.com"})
await pipeline.start_processing()
```

## File Structure

```
data/
├── input/          # Input data files
├── output/         # Processed output files
├── errors/         # Error records
├── quarantine/     # Quarantined records
└── backups/        # Data backups

config/data/
├── validation_config.yml    # Validation configuration
└── schemas/                 # JSON schemas
    ├── user_data.json
    ├── anomaly_data.json
    └── timeseries_data.json

scripts/data/
├── validate_data.py         # Data validation script
├── run_pipeline.py          # Pipeline runner
├── test_validation.sh       # Test script
└── performance_test.py      # Performance testing
```

## Validation Actions

When validation fails, the following actions can be taken:

- `SKIP`: Skip processing the record
- `CLEAN`: Clean/transform the data and continue
- `REJECT`: Reject the record completely
- `QUARANTINE`: Move to quarantine for manual review
- `TRANSFORM`: Apply transformation and continue

## Performance

The pipeline supports:

- **Parallel Processing**: Multiple workers for concurrent processing
- **Batch Processing**: Efficient batch operations
- **Async Operations**: Asynchronous processing support
- **Memory Management**: Configurable memory limits
- **Caching**: Validation result caching

## Monitoring

Real-time monitoring includes:

- Processing throughput (records/second)
- Success/failure rates
- Error distribution
- Processing latencies
- Queue sizes
- Memory usage

## Error Handling

Comprehensive error handling:

- Detailed error messages
- Error categorization
- Automatic retry logic
- Dead letter queues
- Error aggregation
- Alert integration

## Testing

Run tests:

```bash
# Basic validation tests
./scripts/data/test_validation.sh

# Performance tests
python3 scripts/data/performance_test.py
```

## Examples

See `data/input/` for sample data files demonstrating various validation scenarios.
