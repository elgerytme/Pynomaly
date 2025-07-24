# Anomaly Detection Go SDK

A comprehensive Go client library for the Anomaly Detection service, providing both HTTP API and WebSocket streaming capabilities.

## Installation

```bash
go get github.com/anomaly-detection/go-sdk
```

## Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
    
    anomaly "github.com/anomaly-detection/go-sdk"
)

func main() {
    // Initialize client
    config := anomaly.ClientConfig{
        BaseURL:    "http://localhost:8000",
        APIKey:     nil, // Optional: &apiKey
        Timeout:    30 * time.Second,
        MaxRetries: 3,
    }
    
    client := anomaly.NewClient(config)
    
    // Detect anomalies
    data := [][]float64{
        {1.0, 2.0},
        {1.1, 2.1},
        {10.0, 20.0}, // This point is anomalous
    }
    
    ctx := context.Background()
    result, err := client.DetectAnomalies(
        ctx, 
        data, 
        anomaly.IsolationForest, 
        nil, // parameters
        false, // return explanations
    )
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Found %d anomalies\n", result.AnomalyCount)
    for _, anomaly := range result.Anomalies {
        fmt.Printf("Anomaly at index %d with score %.4f\n", 
            anomaly.Index, anomaly.Score)
    }
}
```

### Streaming Detection

```go
package main

import (
    "fmt"
    "log"
    "time"
    
    anomaly "github.com/anomaly-detection/go-sdk"
)

func main() {
    config := anomaly.StreamingClientConfig{
        WSURL:  "ws://localhost:8000/ws/stream",
        APIKey: nil, // Optional: &apiKey
        Config: anomaly.StreamingConfig{
            Algorithm:          anomaly.IsolationForest,
            BatchSize:          10,
            DetectionThreshold: 0.5,
        },
        AutoReconnect:  true,
        ReconnectDelay: 5 * time.Second,
    }
    
    client := anomaly.NewStreamingClient(config)
    
    // Set up event handlers
    handlers := anomaly.StreamingHandlers{
        OnConnect: func() {
            fmt.Println("Connected to streaming service")
        },
        OnAnomaly: func(anomaly anomaly.AnomalyData) {
            fmt.Printf("ANOMALY DETECTED: Index=%d, Score=%.4f\n", 
                anomaly.Index, anomaly.Score)
        },
        OnError: func(err error) {
            fmt.Printf("Streaming error: %v\n", err)
        },
        OnDisconnect: func() {
            fmt.Println("Disconnected from streaming service")
        },
    }
    client.SetHandlers(handlers)
    
    // Start streaming
    if err := client.Start(); err != nil {
        log.Fatal(err)
    }
    defer client.Stop()
    
    // Send data points
    dataPoints := [][]float64{
        {2.5, 3.1},
        {2.6, 3.2},
        {15.0, 25.0}, // Anomalous point
    }
    
    for _, point := range dataPoints {
        if err := client.SendData(point); err != nil {
            log.Printf("Error sending data: %v", err)
        }
        time.Sleep(1 * time.Second)
    }
    
    // Keep running
    time.Sleep(10 * time.Second)
}
```

### Model Management

```go
// Train a new model
trainingRequest := anomaly.TrainingRequest{
    Data:            trainingData,
    Algorithm:       anomaly.IsolationForest,
    ModelName:       &modelName,
    ValidationSplit: 0.2,
}

trainingResult, err := client.TrainModel(ctx, trainingRequest)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Model trained: %s\n", trainingResult.ModelID)

// List all models
models, err := client.ListModels(ctx)
if err != nil {
    log.Fatal(err)
}

for _, model := range models {
    fmt.Printf("%s: %s (%s)\n", model.ModelID, model.Algorithm, model.Status)
}

// Use specific model for detection
explainOptions := anomaly.ExplainOptions{
    ModelID: &trainingResult.ModelID,
    Method:  "shap",
}

explanation, err := client.ExplainAnomaly(ctx, testPoint, explainOptions)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Explanation: %s\n", explanation.ExplanationText)
```

### Utility Functions

```go
// Validate data format
if err := anomaly.ValidateDataFormat(data); err != nil {
    log.Printf("Invalid data format: %v", err)
}

// Normalize data
normalizedData, params, err := anomaly.NormalizeData(rawData)
if err != nil {
    log.Fatal(err)
}

// Apply normalization to new data
newNormalizedData, err := anomaly.ApplyNormalization(newData, params)
if err != nil {
    log.Fatal(err)
}

// Generate sample data for testing
sampleData, err := anomaly.GenerateSampleData(1000, 5, 0.1)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Generated %d samples with %d anomalies\n", 
    len(sampleData.Data), 
    func() int {
        count := 0
        for _, label := range sampleData.Labels {
            if label {
                count++
            }
        }
        return count
    }())

// Calculate statistics
stats, err := anomaly.CalculateDataStatistics(data)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Dataset: %d samples, %d features\n", 
    stats.NumSamples, stats.NumFeatures)
```

## API Reference

### Client

The main HTTP client for interacting with the anomaly detection service.

#### Creating a Client

```go
config := anomaly.ClientConfig{
    BaseURL:    "http://localhost:8000",
    APIKey:     &apiKey,        // Optional
    Timeout:    30 * time.Second,
    MaxRetries: 3,
    Headers:    map[string]string{}, // Optional additional headers
}

client := anomaly.NewClient(config)
```

#### Methods

- `DetectAnomalies(ctx, data, algorithm, parameters, returnExplanations)` - Detect anomalies
- `BatchDetect(ctx, request)` - Process batch detection request
- `TrainModel(ctx, request)` - Train a new model
- `GetModel(ctx, modelID)` - Get model information
- `ListModels(ctx)` - List all models
- `DeleteModel(ctx, modelID)` - Delete a model
- `ExplainAnomaly(ctx, dataPoint, options)` - Get anomaly explanation
- `GetHealth(ctx)` - Get service health
- `GetMetrics(ctx)` - Get service metrics
- `UploadData(ctx, data, datasetName, description)` - Upload training data

### StreamingClient

WebSocket client for real-time anomaly detection.

#### Creating a Streaming Client

```go
config := anomaly.StreamingClientConfig{
    WSURL:  "ws://localhost:8000/ws/stream",
    APIKey: &apiKey, // Optional
    Config: anomaly.StreamingConfig{
        BufferSize:         100,
        DetectionThreshold: 0.5,
        BatchSize:          10,
        Algorithm:          anomaly.IsolationForest,
        AutoRetrain:        false,
    },
    AutoReconnect:  true,
    ReconnectDelay: 5 * time.Second,
}

client := anomaly.NewStreamingClient(config)
```

#### Methods

- `Start()` - Start streaming client
- `Stop()` - Stop streaming client
- `SendData(dataPoint)` - Send single data point
- `SendBatch(batch)` - Send multiple data points
- `SetHandlers(handlers)` - Set event handlers
- `IsConnected()` - Check connection status
- `BufferSize()` - Get current buffer size

#### Event Handlers

```go
handlers := anomaly.StreamingHandlers{
    OnConnect:    func() { /* handle connection */ },
    OnDisconnect: func() { /* handle disconnection */ },
    OnAnomaly:    func(anomaly anomaly.AnomalyData) { /* handle anomaly */ },
    OnError:      func(err error) { /* handle error */ },
    OnMessage:    func(data map[string]interface{}) { /* handle raw message */ },
}
```

## Types

### Core Types

- `AlgorithmType` - Enumeration of available algorithms
- `AnomalyData` - Individual anomaly detection result
- `DetectionResult` - Complete detection results
- `ModelInfo` - Model information
- `StreamingConfig` - Streaming configuration
- `ExplanationResult` - Anomaly explanation
- `HealthStatus` - Service health status

### Request/Response Types

- `BatchProcessingRequest` - Batch processing request
- `TrainingRequest` - Model training request
- `TrainingResult` - Model training result
- `ExplainOptions` - Options for anomaly explanation
- `UploadResult` - Data upload result

### Error Types

- `SDKError` - Base error type
- `APIError` - API service error
- `ValidationError` - Data validation error
- `ConnectionError` - Network connection error
- `TimeoutError` - Request timeout error
- `StreamingError` - Streaming connection error

## Error Handling

```go
result, err := client.DetectAnomalies(ctx, data, anomaly.IsolationForest, nil, false)
if err != nil {
    switch e := err.(type) {
    case *anomaly.ValidationError:
        fmt.Printf("Validation error: %s\n", e.Message)
    case *anomaly.APIError:
        fmt.Printf("API error %d: %s\n", e.StatusCode, e.Message)
    case *anomaly.ConnectionError:
        fmt.Printf("Connection error: %s\n", e.Message)
    case *anomaly.TimeoutError:
        fmt.Printf("Timeout error: %s\n", e.Message)
    default:
        fmt.Printf("Unknown error: %v\n", err)
    }
    return
}
```

## Context Support

All client methods accept a `context.Context` parameter for cancellation and timeouts:

```go
// Set timeout for individual request
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

result, err := client.DetectAnomalies(ctx, data, anomaly.IsolationForest, nil, false)
```

## Testing

```bash
# Run tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./...
```

## Examples

See the [examples](../../examples/) directory for more comprehensive usage examples.

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/anomaly-detection/go-sdk/issues)
- Documentation: [Full API documentation](../../docs/api.md)
- Go Package Documentation: [pkg.go.dev](https://pkg.go.dev/github.com/anomaly-detection/go-sdk)