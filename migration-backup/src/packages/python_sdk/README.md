# Pynomaly Python SDK - Data Science Client Library

A comprehensive Python SDK for data science packages with intuitive APIs, async support, and seamless integration with popular data science libraries including pandas, numpy, and scikit-learn.

## 🚀 Features

- **Pythonic API Design**: Clean, intuitive interfaces with full type hints
- **Async/Await Support**: Non-blocking operations for high-performance applications
- **Popular Library Integration**: Native support for pandas, numpy, scikit-learn
- **Automatic Retry Logic**: Built-in error handling and retry mechanisms
- **Connection Management**: Efficient connection pooling and session management
- **Intelligent Caching**: Response caching with configurable TTL
- **Comprehensive Testing**: 100% test coverage with unit and integration tests
- **Extensive Documentation**: Complete API documentation and examples

## 📦 Installation

```bash
pip install pynomaly-python-sdk
```

## 🔧 Quick Start

### Basic Usage

```python
import asyncio
import pandas as pd
from pynomaly_sdk import create_client

async def main():
    # Create client with API key
    client = create_client(api_key="your-api-key")
    
    async with client:
        # Check API health
        health = await client.health_check()
        print(f"API Status: {health['status']}")
        
        # Create sample data
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'feature2': [2, 4, 6, 8, 200]   # 200 is an outlier
        })
        
        # Detect anomalies
        result = await client.detect_anomalies(
            data=data,
            algorithm="isolation_forest",
            algorithm_params={"contamination": 0.1}
        )
        
        print(f"Anomaly Scores: {result.anomaly_scores}")
        print(f"Anomaly Labels: {result.anomaly_labels}")
        print(f"Algorithm Used: {result.algorithm_used}")
        print(f"Execution Time: {result.execution_time}s")

# Run the example
asyncio.run(main())
```

### Synchronous Usage

```python
import asyncio
import pandas as pd
from pynomaly_sdk import create_client

def detect_anomalies_sync(data, algorithm="isolation_forest"):
    """Synchronous wrapper for anomaly detection."""
    
    async def _detect():
        client = create_client(api_key="your-api-key")
        async with client:
            return await client.detect_anomalies(data, algorithm)
    
    return asyncio.run(_detect())

# Usage
data = pd.DataFrame({'x': [1, 2, 3, 4, 100], 'y': [2, 4, 6, 8, 200]})
result = detect_anomalies_sync(data)
print(f"Detected {sum(result.anomaly_labels)} anomalies")
```

## 🏗️ Advanced Usage

### Batch Processing

```python
import asyncio
from pynomaly_sdk import create_client

async def batch_detection():
    client = create_client(api_key="your-api-key")
    
    # Define multiple datasets for processing
    datasets = [
        {
            "data": [[1, 2], [3, 4], [5, 6], [100, 200]],
            "algorithm": "isolation_forest",
            "algorithm_params": {"contamination": 0.25}
        },
        {
            "data": [[10, 20], [30, 40], [50, 60], [1000, 2000]],
            "algorithm": "lof",
            "algorithm_params": {"n_neighbors": 3}
        }
    ]
    
    # Progress callback
    def progress_callback(completed, total, result):
        print(f"Progress: {completed}/{total} - {result.algorithm_used}")
    
    async with client:
        results = await client.detect_anomalies_batch(
            datasets=datasets,
            max_concurrent=2,
            progress_callback=progress_callback
        )
        
        for i, result in enumerate(results):
            print(f"Dataset {i+1}: {sum(result.anomaly_labels)} anomalies detected")

asyncio.run(batch_detection())
```

### Custom Configuration

```python
from pynomaly_sdk import PynomagyClient, ClientConfig

# Custom configuration
config = ClientConfig(
    base_url="https://your-custom-api.com/v1",
    api_key="your-api-key",
    timeout=60,
    max_retries=5,
    backoff_factor=1.5,
    connection_pool_size=20,
    max_concurrent_requests=15,
    enable_caching=True,
    cache_ttl=600,  # 10 minutes
    debug=True,
    custom_headers={"X-Custom-Header": "value"}
)

client = PynomagyClient(config)
```

### Data Quality Assessment

```python
import asyncio
import pandas as pd
from pynomaly_sdk import create_client

async def assess_quality():
    client = create_client(api_key="your-api-key")
    
    # Sample data with quality issues
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', None, 'Diana', 'Eve'],  # Missing value
        'score': [95, 87, 92, 88, 95],  # Duplicate value
        'email': ['alice@test.com', 'bob@invalid', 'charlie@test.com', 
                 'diana@test.com', 'eve@test.com']  # Invalid email
    })
    
    async with client:
        quality_report = await client.assess_data_quality(
            data=data,
            quality_metrics=["completeness", "uniqueness", "validity", "consistency"]
        )
        
        print("Data Quality Report:")
        for metric, score in quality_report.items():
            print(f"  {metric.capitalize()}: {score:.2%}")

asyncio.run(assess_quality())
```

### Model Performance Evaluation

```python
import asyncio
import pandas as pd
from pynomaly_sdk import create_client

async def evaluate_model():
    client = create_client(api_key="your-api-key")
    
    # Test data for model evaluation
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'actual': [0, 0, 0, 1, 1, 0, 1, 1, 0, 1]
    })
    
    async with client:
        performance = await client.evaluate_model_performance(
            model_id="my-model-v1.0",
            test_data=test_data,
            metrics=["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        )
        
        print("Model Performance Metrics:")
        for metric, value in performance.items():
            print(f"  {metric.upper()}: {value:.3f}")

asyncio.run(evaluate_model())
```

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/packages/python_sdk --cov-report=html tests/

# Run specific test file
pytest tests/packages/python_sdk/test_client.py -v
```

### Test Structure

```
tests/
├── packages/
│   └── python_sdk/
│       ├── test_client.py           # Main client tests
│       ├── test_domain/             # Domain layer tests
│       ├── test_application/        # Application layer tests
│       ├── test_infrastructure/     # Infrastructure tests
│       └── test_integration/        # Integration tests
```

## 📚 API Reference

### PynomagyClient

The main client class for interacting with Pynomaly services.

#### Methods

##### `detect_anomalies(data, algorithm, algorithm_params, metadata, use_cache)`

Detect anomalies in data using specified algorithm.

**Parameters:**
- `data` (Union[pd.DataFrame, np.ndarray, List[List[float]]]): Input data
- `algorithm` (str): Algorithm name (default: "isolation_forest")
- `algorithm_params` (Optional[Dict]): Algorithm parameters
- `metadata` (Optional[Dict]): Additional metadata
- `use_cache` (bool): Whether to use caching (default: True)

**Returns:**
- `DetectionResponseDTO`: Detection results with scores and labels

##### `detect_anomalies_batch(datasets, max_concurrent, progress_callback)`

Detect anomalies in multiple datasets concurrently.

**Parameters:**
- `datasets` (List[Dict]): List of dataset configurations
- `max_concurrent` (Optional[int]): Maximum concurrent requests
- `progress_callback` (Optional[Callable]): Progress callback function

**Returns:**
- `List[DetectionResponseDTO]`: List of detection results

##### `assess_data_quality(data, quality_metrics, use_cache)`

Assess data quality using specified metrics.

**Parameters:**
- `data` (Union[pd.DataFrame, np.ndarray]): Input data
- `quality_metrics` (Optional[List[str]]): Quality metrics to compute
- `use_cache` (bool): Whether to use caching (default: True)

**Returns:**
- `Dict`: Quality assessment results

##### `evaluate_model_performance(model_id, test_data, metrics, use_cache)`

Evaluate model performance on test data.

**Parameters:**
- `model_id` (str): Model identifier
- `test_data` (Union[pd.DataFrame, np.ndarray]): Test data
- `metrics` (Optional[List[str]]): Metrics to compute
- `use_cache` (bool): Whether to use caching (default: True)

**Returns:**
- `Dict`: Performance metrics

##### `list_available_algorithms()`

Get list of available algorithms.

**Returns:**
- `List[Dict]`: List of algorithm information

##### `get_algorithm_info(algorithm_name)`

Get detailed information about a specific algorithm.

**Parameters:**
- `algorithm_name` (str): Name of the algorithm

**Returns:**
- `Dict`: Algorithm information and parameters

##### `health_check()`

Check API health status.

**Returns:**
- `Dict`: Health status information

##### `clear_cache()`

Clear the request cache.

##### `get_cache_stats()`

Get cache statistics.

**Returns:**
- `Dict`: Cache statistics

### Configuration Classes

#### `ClientConfig`

Configuration for the Pynomaly client.

**Attributes:**
- `base_url` (str): API base URL
- `api_key` (Optional[str]): API key for authentication
- `timeout` (int): Request timeout in seconds
- `max_retries` (int): Maximum retry attempts
- `backoff_factor` (float): Backoff factor for retries
- `verify_ssl` (bool): SSL verification enabled
- `connection_pool_size` (int): Connection pool size
- `max_concurrent_requests` (int): Maximum concurrent requests
- `enable_caching` (bool): Enable response caching
- `cache_ttl` (int): Cache TTL in seconds
- `debug` (bool): Debug mode enabled
- `custom_headers` (Dict[str, str]): Custom HTTP headers

#### `RetryConfig`

Configuration for retry behavior.

**Attributes:**
- `max_attempts` (int): Maximum retry attempts
- `backoff_factor` (float): Backoff factor
- `max_backoff` (float): Maximum backoff time
- `retry_on_status` (List[int]): HTTP status codes to retry on
- `retry_on_exceptions` (List[type]): Exception types to retry on

### Data Transfer Objects

#### `DetectionResponseDTO`

Response object for anomaly detection operations.

**Attributes:**
- `request_id` (str): Request identifier
- `anomaly_scores` (List[float]): Anomaly scores for each data point
- `anomaly_labels` (List[int]): Binary labels (0=normal, 1=anomaly)
- `algorithm_used` (str): Algorithm used for detection
- `execution_time` (float): Execution time in seconds
- `metadata` (Dict): Additional metadata

## 🔧 Error Handling

The SDK provides comprehensive error handling with custom exceptions:

```python
from pynomaly_sdk.domain.exceptions.validation_exceptions import ValidationError
from aiohttp import ClientError
import asyncio

async def error_handling_example():
    client = create_client(api_key="your-api-key")
    
    try:
        async with client:
            # This will raise ValidationError for invalid data
            await client.detect_anomalies("invalid_data")
            
    except ValidationError as e:
        print(f"Validation error: {e}")
    except ClientError as e:
        print(f"HTTP error: {e}")
    except asyncio.TimeoutError:
        print("Request timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")

asyncio.run(error_handling_example())
```

## 🚀 Performance Optimization

### Connection Pooling

```python
from pynomaly_sdk import ClientConfig, PynomagyClient

# Optimized for high-throughput applications
config = ClientConfig(
    connection_pool_size=50,
    max_concurrent_requests=20,
    timeout=30,
    enable_caching=True,
    cache_ttl=300
)

client = PynomagyClient(config)
```

### Batch Processing

```python
# Process large datasets efficiently
async def process_large_dataset():
    client = create_client(api_key="your-api-key")
    
    # Split large dataset into chunks
    chunk_size = 1000
    large_data = generate_large_dataset()  # Your data generation function
    
    chunks = [large_data[i:i+chunk_size] for i in range(0, len(large_data), chunk_size)]
    
    datasets = [
        {"data": chunk, "algorithm": "isolation_forest"}
        for chunk in chunks
    ]
    
    async with client:
        results = await client.detect_anomalies_batch(
            datasets=datasets,
            max_concurrent=10
        )
        
        # Combine results
        all_scores = []
        all_labels = []
        
        for result in results:
            all_scores.extend(result.anomaly_scores)
            all_labels.extend(result.anomaly_labels)
        
        return all_scores, all_labels
```

## 📈 Monitoring and Observability

### Request Logging

```python
import logging
from pynomaly_sdk import create_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def monitored_detection():
    client = create_client(api_key="your-api-key", debug=True)
    
    async with client:
        # Log cache stats
        logger.info(f"Cache stats: {client.get_cache_stats()}")
        
        # Perform detection
        result = await client.detect_anomalies(your_data)
        
        # Log results
        logger.info(f"Detection completed: {result.request_id}")
        logger.info(f"Anomalies found: {sum(result.anomaly_labels)}")
        logger.info(f"Execution time: {result.execution_time}s")
```

### Performance Metrics

```python
import time
from pynomaly_sdk import create_client

async def performance_monitoring():
    client = create_client(api_key="your-api-key")
    
    start_time = time.time()
    
    async with client:
        # Measure API health check
        health_start = time.time()
        health = await client.health_check()
        health_time = time.time() - health_start
        
        # Measure detection time
        detection_start = time.time()
        result = await client.detect_anomalies(your_data)
        detection_time = time.time() - detection_start
        
        total_time = time.time() - start_time
        
        print(f"Performance Metrics:")
        print(f"  Health check: {health_time:.3f}s")
        print(f"  Detection: {detection_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Cache stats: {client.get_cache_stats()}")
```

## 🔐 Security

### API Key Management

```python
import os
from pynomaly_sdk import create_client

# Use environment variables for API keys
api_key = os.getenv('PYNOMALY_API_KEY')
if not api_key:
    raise ValueError("PYNOMALY_API_KEY environment variable not set")

client = create_client(api_key=api_key)
```

### SSL Configuration

```python
from pynomaly_sdk import ClientConfig, PynomagyClient

# Custom SSL configuration
config = ClientConfig(
    api_key="your-api-key",
    verify_ssl=True,  # Always verify SSL in production
    custom_headers={
        "X-API-Version": "v1",
        "X-Client-Version": "1.0.0"
    }
)

client = PynomagyClient(config)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.pynomaly.com](https://docs.pynomaly.com)
- **Issues**: [GitHub Issues](https://github.com/pynomaly/python-sdk/issues)
- **Email**: support@pynomaly.com
- **Community**: [Discord Server](https://discord.gg/pynomaly)

## 🗺️ Roadmap

- [ ] GraphQL API support
- [ ] Real-time streaming detection
- [ ] Advanced visualization tools
- [ ] MLOps pipeline integration
- [ ] Custom algorithm plugins
- [ ] Distributed processing support

---

**Made with ❤️ by the Pynomaly Team**