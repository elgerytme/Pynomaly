# Anomaly Detection Python SDK

A comprehensive Python SDK for the Anomaly Detection service, providing both synchronous and asynchronous clients for seamless integration.

## Installation

```bash
pip install anomaly-detection-sdk
```

## Quick Start

### Synchronous Client

```python
from anomaly_detection_sdk import AnomalyDetectionClient

# Initialize client
client = AnomalyDetectionClient(base_url="http://localhost:8000")

# Detect anomalies
data = [[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]]  # Last point is anomalous
result = client.detect_anomalies(data, algorithm="isolation_forest")
print(f"Anomalies detected: {result.anomalies}")
```

### Asynchronous Client

```python
import asyncio
from anomaly_detection_sdk import AsyncAnomalyDetectionClient

async def main():
    client = AsyncAnomalyDetectionClient(base_url="http://localhost:8000")
    
    data = [[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]]
    result = await client.detect_anomalies(data, algorithm="isolation_forest")
    print(f"Anomalies detected: {result.anomalies}")
    
    await client.close()

asyncio.run(main())
```

### Streaming Detection

```python
from anomaly_detection_sdk import StreamingClient

# Real-time anomaly detection
streaming_client = StreamingClient("ws://localhost:8000/ws/stream")

@streaming_client.on_anomaly
def handle_anomaly(anomaly_data):
    print(f"Anomaly detected: {anomaly_data}")

# Start streaming
streaming_client.start()
streaming_client.send_data([2.5, 3.1])
```

## Features

- **Dual Interface**: Both sync and async clients
- **Real-time Streaming**: WebSocket-based streaming detection
- **Comprehensive Models**: Support for all detection algorithms
- **Robust Error Handling**: Automatic retries and detailed error messages
- **Type Safety**: Full type hints and Pydantic models
- **Easy Integration**: Simple, intuitive API design

## Documentation

For detailed documentation, examples, and API reference, visit: [SDK Documentation](../../../docs/)

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/anomaly-detection/issues)
- Documentation: [Full API documentation](../../../docs/api.md)
- Examples: [Usage examples](../../../examples/)