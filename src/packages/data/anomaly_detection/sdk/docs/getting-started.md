# Getting Started with Anomaly Detection SDKs

This guide will help you get up and running with the Anomaly Detection SDKs quickly. Choose your preferred programming language and follow the installation and setup instructions.

## Prerequisites

Before you begin, ensure you have:

1. **Anomaly Detection Service**: A running instance of the anomaly detection service
   - Default HTTP endpoint: `http://localhost:8000`
   - Default WebSocket endpoint: `ws://localhost:8000/ws/stream`

2. **Development Environment**: 
   - Python 3.8+ (for Python SDK)
   - Node.js 14+ (for JavaScript SDK)
   - Go 1.21+ (for Go SDK)

## Installation

### Python SDK

```bash
# Install from PyPI
pip install anomaly-detection-sdk

# Or install with development dependencies
pip install anomaly-detection-sdk[dev]

# Verify installation
python -c "import anomaly_detection_sdk; print('SDK installed successfully')"
```

### JavaScript/TypeScript SDK

```bash
# Install from npm
npm install @anomaly-detection/sdk

# Or with yarn
yarn add @anomaly-detection/sdk

# Verify installation (Node.js)
node -e "const sdk = require('@anomaly-detection/sdk'); console.log('SDK installed successfully');"
```

### Go SDK

```bash
# Initialize Go module (if not already done)
go mod init your-project-name

# Install SDK
go get github.com/anomaly-detection/go-sdk

# Verify installation
go run -c "import _ \"github.com/anomaly-detection/go-sdk\""
```

## Basic Configuration

### Service Connection

All SDKs need to connect to the anomaly detection service. The basic configuration includes:

```python
# Python
from anomaly_detection_sdk import AnomalyDetectionClient

client = AnomalyDetectionClient(
    base_url="http://localhost:8000",  # Service endpoint
    api_key="your-api-key",           # Optional authentication
    timeout=30.0,                     # Request timeout in seconds
    max_retries=3                     # Retry attempts
)
```

```javascript
// JavaScript/TypeScript
import { AnomalyDetectionClient } from '@anomaly-detection/sdk';

const client = new AnomalyDetectionClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your-api-key',      // Optional
    timeout: 30000,              // Timeout in milliseconds
    maxRetries: 3
});
```

```go
// Go
import anomaly "github.com/anomaly-detection/go-sdk"

client := anomaly.NewClient(anomaly.ClientConfig{
    BaseURL:    "http://localhost:8000",
    APIKey:     &apiKey,              // Optional
    Timeout:    30 * time.Second,
    MaxRetries: 3,
})
```

## Your First Anomaly Detection

### Step 1: Prepare Your Data

Data should be formatted as a 2D array where each row represents a data point and each column represents a feature:

```python
# Python - Example data with 2 features
data = [
    [1.0, 2.0],   # Normal point
    [1.1, 2.1],   # Normal point  
    [1.2, 1.9],   # Normal point
    [10.0, 20.0], # Anomalous point (far from others)
    [0.9, 2.2],   # Normal point
]
```

```javascript
// JavaScript - Same data format
const data = [
    [1.0, 2.0],
    [1.1, 2.1],
    [1.2, 1.9],
    [10.0, 20.0], // Anomalous point
    [0.9, 2.2],
];
```

```go
// Go - Same data format
data := [][]float64{
    {1.0, 2.0},
    {1.1, 2.1},
    {1.2, 1.9},
    {10.0, 20.0}, // Anomalous point
    {0.9, 2.2},
}
```

### Step 2: Detect Anomalies

```python
# Python
from anomaly_detection_sdk import AlgorithmType

result = client.detect_anomalies(
    data=data,
    algorithm=AlgorithmType.ISOLATION_FOREST,
    parameters={'contamination': 0.2},  # Expect 20% anomalies
    return_explanations=False
)

print(f"Found {result.anomaly_count} anomalies out of {result.total_points} points")
for anomaly in result.anomalies:
    print(f"  - Point {anomaly.index}: score={anomaly.score:.3f}")
```

```javascript
// JavaScript
import { AlgorithmType } from '@anomaly-detection/sdk';

const result = await client.detectAnomalies(
    data,
    AlgorithmType.ISOLATION_FOREST,
    { contamination: 0.2 },
    false // return explanations
);

console.log(`Found ${result.anomalyCount} anomalies out of ${result.totalPoints} points`);
result.anomalies.forEach(anomaly => {
    console.log(`  - Point ${anomaly.index}: score=${anomaly.score.toFixed(3)}`);
});
```

```go
// Go
import "context"

ctx := context.Background()
result, err := client.DetectAnomalies(
    ctx,
    data,
    anomaly.IsolationForest,
    map[string]interface{}{"contamination": 0.2},
    false, // return explanations
)

if err != nil {
    log.Fatal(err)
}

fmt.Printf("Found %d anomalies out of %d points\n", result.AnomalyCount, result.TotalPoints)
for _, anomaly := range result.Anomalies {
    fmt.Printf("  - Point %d: score=%.3f\n", anomaly.Index, anomaly.Score)
}
```

### Expected Output

```
Found 1 anomalies out of 5 points
  - Point 3: score=0.847
```

## Understanding the Results

The detection result contains:

- **`anomalies`**: List of detected anomalies with indices and scores
- **`anomaly_count`**: Total number of anomalies found
- **`total_points`**: Total number of data points analyzed
- **`algorithm_used`**: The algorithm that was used
- **`execution_time`**: Time taken for detection (in seconds)

### Anomaly Scores

- **Higher scores** indicate more anomalous points
- Scores typically range from 0 to 1, but can vary by algorithm
- The **`contamination`** parameter controls how many points are considered anomalous

## Available Algorithms

All SDKs support these algorithms:

| Algorithm | Use Case | Best For |
|-----------|----------|----------|
| **Isolation Forest** | General purpose | Mixed data types, scalable |
| **Local Outlier Factor** | Density-based | Clusters with varying densities |
| **One-Class SVM** | Boundary detection | High-dimensional data |
| **Elliptic Envelope** | Statistical | Gaussian-distributed data |
| **Autoencoder** | Neural networks | Complex patterns, large datasets |
| **Ensemble** | Combined approach | Maximum accuracy |

### Algorithm Selection Guide

```python
# Python - Choose algorithm based on your data
if data_size > 10000:
    algorithm = AlgorithmType.ISOLATION_FOREST  # Fast and scalable
elif data_has_clusters:
    algorithm = AlgorithmType.LOCAL_OUTLIER_FACTOR  # Good for clustered data
elif data_is_high_dimensional:
    algorithm = AlgorithmType.ONE_CLASS_SVM  # Handles high dimensions well
else:
    algorithm = AlgorithmType.ENSEMBLE  # Best overall performance
```

## Health Check

Before processing data, verify the service is healthy:

```python
# Python
health = client.get_health()
print(f"Service status: {health.status}")
print(f"Version: {health.version}")
```

```javascript
// JavaScript
const health = await client.getHealth();
console.log(`Service status: ${health.status}`);
console.log(`Version: ${health.version}`);
```

```go
// Go
health, err := client.GetHealth(ctx)
if err != nil {
    log.Fatal("Service health check failed:", err)
}
fmt.Printf("Service status: %s\n", health.Status)
```

## Error Handling

All SDKs provide comprehensive error handling:

```python
# Python
from anomaly_detection_sdk.exceptions import ValidationError, APIError, ConnectionError

try:
    result = client.detect_anomalies(data, AlgorithmType.ISOLATION_FOREST)
except ValidationError as e:
    print(f"Data validation error: {e.message}")
except APIError as e:
    print(f"API error {e.status_code}: {e.message}")
except ConnectionError as e:
    print(f"Connection error: {e.message}")
```

```javascript
// JavaScript
import { ValidationError, APIError, ConnectionError } from '@anomaly-detection/sdk';

try {
    const result = await client.detectAnomalies(data, AlgorithmType.ISOLATION_FOREST);
} catch (error) {
    if (error instanceof ValidationError) {
        console.log(`Data validation error: ${error.message}`);
    } else if (error instanceof APIError) {
        console.log(`API error ${error.statusCode}: ${error.message}`);
    } else if (error instanceof ConnectionError) {
        console.log(`Connection error: ${error.message}`);
    }
}
```

```go
// Go
result, err := client.DetectAnomalies(ctx, data, anomaly.IsolationForest, nil, false)
if err != nil {
    switch e := err.(type) {
    case *anomaly.ValidationError:
        fmt.Printf("Data validation error: %s\n", e.Message)
    case *anomaly.APIError:
        fmt.Printf("API error %d: %s\n", e.StatusCode, e.Message)
    case *anomaly.ConnectionError:
        fmt.Printf("Connection error: %s\n", e.Message)
    default:
        fmt.Printf("Unknown error: %v\n", err)
    }
}
```

## Common Configuration Options

### Authentication

If your service requires authentication:

```python
# Python - API Key
client = AnomalyDetectionClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Python - Custom headers
client = AnomalyDetectionClient(
    base_url="http://localhost:8000",
    headers={"Authorization": "Bearer your-token"}
)
```

### Timeouts and Retries

```python
# Python
client = AnomalyDetectionClient(
    base_url="http://localhost:8000",
    timeout=60.0,      # 60 second timeout
    max_retries=5      # Retry up to 5 times
)
```

### Custom Endpoints

```python
# Python - Custom service endpoint
client = AnomalyDetectionClient(
    base_url="https://anomaly-api.yourcompany.com"
)
```

## Next Steps

Now that you have basic anomaly detection working, explore more advanced features:

1. **[Streaming Detection](streaming-guide.md)** - Real-time anomaly detection
2. **[Model Management](model-management.md)** - Train and manage custom models
3. **[API Reference](api-reference.md)** - Complete method documentation
4. **[Examples](../examples/)** - Working code examples
5. **[SDK-Specific Guides](sdk-guides/)** - Language-specific advanced features

## Troubleshooting

### Common Issues

**Connection Refused**
```
Error: Failed to connect to http://localhost:8000
Solution: Ensure the anomaly detection service is running
```

**Data Format Error**
```
Error: Data must be a 2D array
Solution: Format data as [[x1,y1], [x2,y2], ...]
```

**Authentication Error**
```
Error: 401 Unauthorized
Solution: Check your API key or authentication headers
```

**Timeout Error**
```
Error: Request timed out
Solution: Increase timeout or check service performance
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
# Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

```javascript
// JavaScript (Node.js)
process.env.DEBUG = 'anomaly-detection-sdk:*'
```

```go
// Go - errors include detailed context
if err != nil {
    log.Printf("Detection failed: %+v", err)
}
```

## Support

- **Documentation**: [Full documentation](README.md)
- **Examples**: [Working examples](../examples/)
- **Issues**: [GitHub Issues](https://github.com/anomaly-detection/issues)
- **Community**: [Discussions](https://github.com/anomaly-detection/discussions)