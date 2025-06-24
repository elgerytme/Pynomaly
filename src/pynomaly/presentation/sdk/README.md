# Pynomaly Python SDK

A comprehensive Python SDK for programmatic access to the Pynomaly anomaly detection platform. The SDK provides both synchronous and asynchronous interfaces for all major operations including dataset management, detector training, anomaly detection, and experiment tracking.

## Features

- 🚀 **Dual Interface**: Both synchronous and asynchronous clients
- 🔒 **Type Safe**: Full type hints and Pydantic model validation  
- 🛡️ **Error Handling**: Comprehensive exception hierarchy
- ⚡ **High Performance**: Async support with concurrent processing
- 📊 **Streaming**: Real-time anomaly detection capabilities
- 🔧 **Configurable**: Multiple configuration methods (file, environment, direct)
- 📖 **Well Documented**: Comprehensive examples and documentation

## Installation

The SDK is included with Pynomaly. Install Pynomaly to get the SDK:

```bash
pip install pynomaly
```

For async support, install with async extras:

```bash
pip install pynomaly[async]
```

## Quick Start

### Synchronous Client

```python
from pynomaly.presentation.sdk import PynomaliClient, AlgorithmType

# Initialize client
client = PynomaliClient(
    base_url="http://localhost:8000",
    api_key="your-api-key-here"
)

# Create dataset
dataset = client.create_dataset(
    data_source="data.csv",  # or numpy array, list, etc.
    name="My Dataset"
)

# Train detector
detector = client.train_detector(
    dataset_id=dataset.id,
    algorithm=AlgorithmType.ISOLATION_FOREST,
    parameters={"n_estimators": 100}
)

# Detect anomalies
result = client.detect_anomalies(
    detector_id=detector.id,
    data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
)

print(f"Found {result.num_anomalies} anomalies out of {result.num_samples} samples")

client.close()
```

### Asynchronous Client

```python
import asyncio
from pynomaly.presentation.sdk import AsyncPynomaliClient, AlgorithmType

async def main():
    async with AsyncPynomaliClient(
        base_url="http://localhost:8000",
        api_key="your-api-key-here"
    ) as client:
        
        # Create dataset
        dataset = await client.create_dataset(
            data_source=[[1, 2], [3, 4], [5, 6]],
            name="Async Dataset"
        )
        
        # Train detector
        detector = await client.train_detector(
            dataset_id=dataset.id,
            algorithm=AlgorithmType.LOCAL_OUTLIER_FACTOR
        )
        
        # Concurrent detection on multiple datasets
        data_batches = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]]
        ]
        
        results = await client.batch_detect_concurrent(
            detector_id=detector.id,
            data_sources=data_batches,
            max_concurrent=3
        )
        
        for i, result in enumerate(results):
            print(f"Batch {i+1}: {result.num_anomalies} anomalies")

asyncio.run(main())
```

## Configuration

The SDK supports multiple configuration methods:

### 1. Direct Parameters

```python
client = PynomaliClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)
```

### 2. Configuration Object

```python
from pynomaly.presentation.sdk import SDKConfig

config = SDKConfig(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    auth_type="api_key"
)

# Customize client settings
config.client.timeout = 60.0
config.client.max_retries = 5
config.client.verify_ssl = True

client = PynomaliClient(config=config)
```

### 3. Environment Variables

Set environment variables:

```bash
export PYNOMALY_BASE_URL="http://localhost:8000"
export PYNOMALY_API_KEY="your-api-key"
export PYNOMALY_TIMEOUT="30"
export PYNOMALY_LOG_LEVEL="INFO"
```

Then create client:

```python
client = PynomaliClient()  # Loads from environment
```

### 4. Configuration File

Create `config.json`:

```json
{
  "base_url": "http://localhost:8000",
  "api_key": "your-api-key",
  "auth_type": "api_key",
  "client": {
    "timeout": 30.0,
    "max_retries": 3,
    "verify_ssl": true
  }
}
```

Load configuration:

```python
client = PynomaliClient(config_path="config.json")
```

## Authentication

The SDK supports multiple authentication methods:

### API Key Authentication

```python
config = SDKConfig(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    auth_type="api_key"
)
```

### Basic Authentication

```python
config = SDKConfig(
    base_url="http://localhost:8000",
    username="your-username",
    password="your-password",
    auth_type="basic"
)
```

### Bearer Token Authentication

```python
config = SDKConfig(
    base_url="http://localhost:8000",
    token="your-jwt-token",
    auth_type="bearer"
)
```

## Advanced Usage

### Streaming Detection

Process continuous data streams with automatic batching:

```python
async def data_stream():
    """Generate streaming data."""
    while True:
        batch = generate_data_batch()
        yield batch
        await asyncio.sleep(1)

async with AsyncPynomaliClient(...) as client:
    async for result in client.stream_detection(
        detector_id="detector-123",
        data_stream=data_stream(),
        buffer_size=100
    ):
        print(f"Processed batch: {result.num_anomalies} anomalies")
```

### Experiment Comparison

Compare multiple algorithms on the same dataset:

```python
experiment = client.create_experiment(
    name="Algorithm Comparison",
    dataset_id="dataset-123",
    algorithms=[
        AlgorithmType.ISOLATION_FOREST,
        AlgorithmType.LOCAL_OUTLIER_FACTOR,
        AlgorithmType.ONE_CLASS_SVM
    ],
    parameters={
        "IsolationForest": {"n_estimators": 100},
        "LocalOutlierFactor": {"n_neighbors": 20},
        "OneClassSVM": {"nu": 0.05}
    }
)

print(f"Best algorithm: {experiment.best_algorithm}")
for algorithm, metrics in experiment.results.items():
    print(f"{algorithm}: F1={metrics.f1_score:.3f}")
```

### Concurrent Processing

Process multiple datasets simultaneously:

```python
async with AsyncPynomaliClient(...) as client:
    # Train multiple detectors concurrently
    tasks = [
        client.train_detector(dataset_id, AlgorithmType.ISOLATION_FOREST)
        for dataset_id in dataset_ids
    ]
    detectors = await asyncio.gather(*tasks)
    
    # Perform concurrent detection
    results = await client.batch_detect_concurrent(
        detector_id=detectors[0].id,
        data_sources=data_batches,
        max_concurrent=10
    )
```

## Error Handling

The SDK provides a comprehensive exception hierarchy:

```python
from pynomaly.presentation.sdk.exceptions import (
    PynomaliSDKError,
    AuthenticationError,
    ValidationError,
    ResourceNotFoundError,
    TimeoutError
)

try:
    detector = client.get_detector("invalid-id")
except ResourceNotFoundError as e:
    print(f"Detector not found: {e.resource_id}")
except AuthenticationError as e:
    print(f"Authentication failed: {e.message}")
except ValidationError as e:
    print(f"Validation error: {e.validation_errors}")
except TimeoutError as e:
    print(f"Request timed out: {e.message}")
except PynomaliSDKError as e:
    print(f"SDK error: {e.message} (status: {e.status_code})")
```

## Data Types and Models

The SDK uses Pydantic models for type safety and validation:

```python
from pynomaly.presentation.sdk.models import (
    Dataset, Detector, DetectionResult,
    AlgorithmType, DataFormat, DetectorStatus
)

# Type-safe model access
dataset: Dataset = client.get_dataset("dataset-123")
print(f"Dataset has {dataset.num_samples} samples")
print(f"Feature names: {dataset.feature_names}")

detector: Detector = client.get_detector("detector-456") 
print(f"Algorithm: {detector.algorithm}")
print(f"Status: {detector.status}")
print(f"Performance: {detector.performance_metrics}")

result: DetectionResult = client.detect_anomalies(...)
print(f"Anomaly rate: {result.anomaly_rate:.2%}")
for i, score in enumerate(result.anomaly_scores):
    print(f"Sample {i}: {score.value:.3f}")
```

## Examples

The SDK includes comprehensive examples:

```python
# Run all examples
from pynomaly.presentation.sdk.examples import main
main()

# Or run specific examples
from pynomaly.presentation.sdk.examples import (
    basic_synchronous_example,
    basic_asynchronous_example,
    experiment_comparison_example,
    streaming_detection_example
)
```

## Best Practices

### 1. Use Context Managers

Always use context managers to ensure proper cleanup:

```python
# Synchronous
with PynomaliClient(...) as client:
    # Your code here
    pass

# Asynchronous  
async with AsyncPynomaliClient(...) as client:
    # Your code here
    pass
```

### 2. Handle Errors Gracefully

Implement proper error handling for production code:

```python
try:
    result = client.detect_anomalies(detector_id, data)
except AuthenticationError:
    # Handle auth issues
    refresh_credentials()
except TimeoutError:
    # Handle timeouts
    retry_with_backoff()
except ValidationError as e:
    # Handle validation errors
    fix_data_format(e.validation_errors)
```

### 3. Use Async for High Throughput

For high-throughput scenarios, use the async client:

```python
async def process_many_datasets(dataset_ids):
    async with AsyncPynomaliClient(...) as client:
        tasks = [
            client.detect_anomalies(detector_id, dataset_id)
            for dataset_id in dataset_ids
        ]
        return await asyncio.gather(*tasks)
```

### 4. Configure Appropriate Timeouts

Set timeouts based on your data size and network conditions:

```python
config = SDKConfig()
config.client.timeout = 300.0  # 5 minutes for large datasets
config.client.max_retries = 5
```

### 5. Monitor Resource Usage

For streaming applications, monitor memory usage:

```python
async for result in client.stream_detection(...):
    process_result(result)
    
    # Periodic cleanup
    if should_cleanup():
        gc.collect()
```

## API Reference

### PynomaliClient

Main synchronous client class.

**Methods:**
- `create_dataset()` - Create a new dataset
- `train_detector()` - Train an anomaly detector  
- `detect_anomalies()` - Detect anomalies in data
- `create_experiment()` - Run algorithm comparison experiments
- `health_check()` - Check API health

### AsyncPynomaliClient

Main asynchronous client class with the same interface as `PynomaliClient` but with async/await support.

**Additional Methods:**
- `batch_detect_concurrent()` - Concurrent detection
- `stream_detection()` - Streaming detection

### Configuration Classes

- `SDKConfig` - Main configuration class
- `ClientConfig` - HTTP client configuration

### Model Classes

- `Dataset` - Dataset representation
- `Detector` - Detector representation  
- `DetectionResult` - Detection results
- `TrainingJob` - Training job status
- `ExperimentResult` - Experiment results

### Exception Classes

- `PynomaliSDKError` - Base SDK exception
- `AuthenticationError` - Authentication failures
- `ValidationError` - Data validation errors
- `ResourceNotFoundError` - Resource not found
- `TimeoutError` - Request timeouts

## Support

For SDK support and questions:

1. Check the [main Pynomaly documentation](../../docs/)
2. Review the [examples](examples.py) 
3. Open an issue on the Pynomaly GitHub repository
4. Contact the development team

## License

The Pynomaly SDK is licensed under the same license as the main Pynomaly project.