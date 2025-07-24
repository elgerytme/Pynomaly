# Anomaly Detection SDK Examples

This directory contains comprehensive examples for all three SDKs (Python, JavaScript/TypeScript, and Go) demonstrating various features and use cases of the Anomaly Detection service.

## Quick Start

Each SDK has its own examples directory with basic and streaming detection examples:

- **Python**: `python/`
- **JavaScript**: `javascript/`
- **Go**: `go/`

## Prerequisites

Before running the examples, ensure you have:

1. **Anomaly Detection Service Running**: The service should be available at:
   - HTTP API: `http://localhost:8000`
   - WebSocket: `ws://localhost:8000/ws/stream`

2. **SDK Dependencies Installed**:
   - Python: `pip install -r requirements.txt`
   - JavaScript: `npm install`
   - Go: `go mod tidy`

## Example Categories

### Basic Detection Examples

Demonstrate core functionality like:
- Anomaly detection with different algorithms
- Algorithm comparison
- Batch processing
- Model management
- Health checks
- Error handling

**Files:**
- `python/basic_detection.py`
- `javascript/basic_detection.js`
- `go/basic_detection.go`

### Streaming Detection Examples

Show real-time anomaly detection capabilities:
- Real-time data streaming
- Interactive data input
- Batch streaming
- Performance testing
- Connection handling

**Files:**
- `python/streaming_detection.py`
- `javascript/streaming_detection.js`
- `go/streaming_detection.go`

## Running Examples

### Python Examples

```bash
# Basic detection
cd python
python basic_detection.py

# Streaming detection - automated
python streaming_detection.py

# Streaming detection - interactive
python streaming_detection.py
# Then select option 2 for interactive mode
```

### JavaScript Examples

```bash
# Basic detection
cd javascript
node basic_detection.js

# Streaming detection - automated
node streaming_detection.js

# Streaming detection - interactive
node streaming_detection.js --interactive
# Then select option 2 for interactive mode
```

### Go Examples

```bash
# Basic detection
cd go
go run basic_detection.go

# Streaming detection - automated
go run streaming_detection.go

# Streaming detection - all examples
go run streaming_detection.go all

# Streaming detection - specific example
go run streaming_detection.go 2  # Interactive mode
```

## Example Features

### 1. Basic Anomaly Detection

```python
# Python
from anomaly_detection_sdk import AnomalyDetectionClient, AlgorithmType

client = AnomalyDetectionClient(base_url="http://localhost:8000")
result = client.detect_anomalies(data, AlgorithmType.ISOLATION_FOREST)
```

```javascript
// JavaScript
const client = new AnomalyDetectionClient({ baseUrl: 'http://localhost:8000' });
const result = await client.detectAnomalies(data, AlgorithmType.ISOLATION_FOREST);
```

```go
// Go
client := anomaly.NewClient(anomaly.ClientConfig{BaseURL: "http://localhost:8000"})
result, err := client.DetectAnomalies(ctx, data, anomaly.IsolationForest, nil, false)
```

### 2. Real-time Streaming

```python
# Python
from anomaly_detection_sdk import StreamingClient

client = StreamingClient("ws://localhost:8000/ws/stream")

@client.on_anomaly
def handle_anomaly(anomaly):
    print(f"Anomaly: {anomaly.score}")

client.start()
client.send_data([2.5, 3.1])
```

```javascript
// JavaScript
const client = new StreamingClient({ wsUrl: 'ws://localhost:8000/ws/stream' });

client.on('anomaly', (anomaly) => {
    console.log(`Anomaly: ${anomaly.score}`);
});

await client.start();
client.sendData([2.5, 3.1]);
```

```go
// Go
client := anomaly.NewStreamingClient(config)
client.SetHandlers(anomaly.StreamingHandlers{
    OnAnomaly: func(anomaly anomaly.AnomalyData) {
        fmt.Printf("Anomaly: %.4f\n", anomaly.Score)
    },
})
client.Start()
client.SendData([]float64{2.5, 3.1})
```

### 3. Model Management

All SDKs support:
- Training custom models
- Listing available models
- Getting model information
- Using specific models for detection

### 4. Utility Functions

Each SDK includes utilities for:
- Data validation and normalization
- Statistics calculation
- Sample data generation
- Performance monitoring

## Algorithm Support

All examples demonstrate these algorithms:
- **Isolation Forest** (default)
- **Local Outlier Factor**
- **One-Class SVM**
- **Elliptic Envelope**
- **Autoencoder**
- **Ensemble Methods**

## Data Formats

Examples use consistent data formats:
- **Input**: Array of arrays (2D array) where each inner array represents a data point
- **Output**: Structured results with anomaly indices, scores, and metadata

## Error Handling

All examples include comprehensive error handling:
- Connection errors
- Validation errors
- API errors
- Timeout errors
- Streaming errors

## Performance Testing

Streaming examples include performance tests demonstrating:
- High-throughput data processing
- Latency measurements
- Throughput calculations
- Memory usage monitoring

## Interactive Features

Several examples support interactive mode:
- Manual data point input
- Real-time feedback
- Dynamic parameter adjustment
- Live performance monitoring

## Sample Data

Examples generate various types of sample data:
- **Normal Data**: Gaussian distribution around [0, 0]
- **Anomalous Data**: Gaussian distribution around [5, 5]
- **Mixed Data**: Combination with configurable anomaly ratios
- **Time Series**: Sequential data with trends and seasonality

## Configuration Options

Examples demonstrate various configuration options:

### Client Configuration
- Base URL and endpoints
- Authentication (API keys)
- Timeout settings
- Retry policies
- Custom headers

### Streaming Configuration
- Buffer sizes
- Batch sizes
- Detection thresholds
- Auto-reconnection
- Algorithm selection

### Algorithm Parameters
- Contamination ratios
- Model hyperparameters
- Training/validation splits
- Performance metrics

## Troubleshooting

### Common Issues

1. **Service Not Running**
   ```
   Error: Connection refused
   Solution: Start the anomaly detection service
   ```

2. **Invalid Data Format**
   ```
   Error: Data must be 2D array
   Solution: Ensure data is [[x1,y1], [x2,y2], ...]
   ```

3. **WebSocket Connection Failed**
   ```
   Error: WebSocket connection failed
   Solution: Check WebSocket URL and service status
   ```

### Debug Mode

Enable debug logging in examples:

```python
# Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

```javascript
// JavaScript - check browser console or use debug package
process.env.DEBUG = 'anomaly-detection-sdk:*'
```

```go
// Go - examples include verbose error messages
log.SetLevel(log.DebugLevel)
```

## Contributing

When adding new examples:

1. Follow the existing structure and naming conventions
2. Include comprehensive error handling
3. Add both basic and advanced usage patterns
4. Document all configuration options
5. Include performance considerations
6. Test with various data sizes and types

## Support

For help with examples:
- Check the main SDK documentation
- Review error messages and logs
- Ensure service dependencies are running
- Verify data formats and configurations

## License

All examples are provided under the same license as the main SDK packages.