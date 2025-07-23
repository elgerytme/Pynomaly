# Anomaly Detection Python Client

Official Python client for the platform's anomaly detection service.

## Features

- 🔍 **Multiple Algorithms**: Support for isolation forest, one-class SVM, and more
- 🔄 **Async/Sync API**: Both async and synchronous interfaces
- 🔐 **JWT Authentication**: Automatic token management and refresh
- 🛡️ **Error Handling**: Comprehensive exception handling with retry logic
- 📊 **Ensemble Detection**: Multi-algorithm ensemble detection
- 📈 **Model Management**: Save, load, and manage trained models
- ⚡ **Performance**: Connection pooling and request optimization

## Installation

```bash
pip install anomaly-detection-client
```

## Quick Start

### Async Usage

```python
import asyncio
from anomaly_detection_client import AnomalyDetectionClient

async def main():
    async with AnomalyDetectionClient(api_key="your-api-key") as client:
        # Simple detection
        result = await client.detect(
            data=[[1.0, 2.0], [2.0, 3.0], [100.0, 200.0]],  # 2D data points
            algorithm="isolation_forest"
        )
        
        print(f"Anomalies found at indices: {result.anomalies}")
        print(f"Anomaly scores: {result.scores}")

asyncio.run(main())
```

### Sync Usage

```python
from anomaly_detection_client import AnomalyDetectionSyncClient

with AnomalyDetectionSyncClient(api_key="your-api-key") as client:
    result = client.detect(
        data=[[1.0, 2.0], [2.0, 3.0], [100.0, 200.0]],
        algorithm="isolation_forest"
    )
    
    print(f"Anomalies: {result.anomalies}")
```

## Advanced Usage

### Ensemble Detection

```python
async with AnomalyDetectionClient(api_key="your-api-key") as client:
    result = await client.detect_ensemble(
        data=data,
        algorithms=["isolation_forest", "one_class_svm", "local_outlier_factor"],
        voting_strategy="majority"
    )
```

### Model Management

```python
async with AnomalyDetectionClient(api_key="your-api-key") as client:
    # Train and save model
    model = await client.train_model(
        data=training_data,
        algorithm="isolation_forest",
        name="production_model_v1"
    )
    
    # Use saved model for prediction
    result = await client.predict(
        data=new_data,
        model_id=model.id
    )
```

### Configuration

```python
from anomaly_detection_client import AnomalyDetectionClient, ClientConfig, Environment

config = ClientConfig.for_environment(
    Environment.PRODUCTION,
    api_key="your-api-key",
    timeout=60.0,
    max_retries=5
)

client = AnomalyDetectionClient(config=config)
```

## API Reference

### Detection Methods

- `detect()` - Single algorithm detection
- `detect_ensemble()` - Multi-algorithm ensemble detection  
- `predict()` - Prediction using saved models
- `batch_detect()` - Batch processing for large datasets

### Model Management

- `train_model()` - Train and save a model
- `get_model()` - Retrieve model information
- `list_models()` - List available models
- `delete_model()` - Delete a model

### Utilities

- `get_algorithms()` - List available algorithms
- `health_check()` - Service health status
- `get_metrics()` - Service metrics

## Error Handling

```python
from anomaly_detection_client import (
    AnomalyDetectionClient,
    ValidationError,
    RateLimitError,
    ServerError
)

try:
    result = await client.detect(data=invalid_data)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    print(f"Details: {e.details}")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after} seconds")
except ServerError as e:
    print(f"Server error: {e.message}")
```

## Environment Variables

```bash
# Optional: Set default configuration via environment
ANOMALY_DETECTION_API_KEY=your-api-key
ANOMALY_DETECTION_BASE_URL=https://api.platform.com
ANOMALY_DETECTION_TIMEOUT=30.0
```

## Examples

See the `examples/` directory for complete usage examples:
- Basic detection
- Ensemble methods
- Model management
- Error handling
- Batch processing