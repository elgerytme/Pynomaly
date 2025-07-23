# Anomaly Detection Go SDK

[![Go Version](https://img.shields.io/badge/go-%3E%3D1.21-blue.svg)](https://golang.org/dl/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/monorepo/anomaly-detection-client-go)](https://goreportcard.com/report/github.com/monorepo/anomaly-detection-client-go)
[![GoDoc](https://godoc.org/github.com/monorepo/anomaly-detection-client-go?status.svg)](https://godoc.org/github.com/monorepo/anomaly-detection-client-go)

A comprehensive Go client library for the Anomaly Detection Platform API, providing robust anomaly detection capabilities with support for multiple algorithms, real-time streaming, and enterprise features.

## ✨ Features

- **🧮 20+ Algorithms**: Support for statistical, machine learning, and deep learning algorithms
- **⚡ Real-time Streaming**: WebSocket-based streaming for real-time anomaly detection
- **🎯 Ensemble Methods**: Combine multiple algorithms for improved accuracy
- **🔐 Enterprise Security**: JWT and API key authentication with session management
- **📊 Comprehensive Types**: Full type safety with detailed request/response models
- **🔄 Retry Logic**: Built-in exponential backoff and error handling
- **📈 Rate Limiting**: Token bucket rate limiting for API compliance
- **🌐 Context Support**: Full context.Context support for cancellation and timeouts
- **🧪 Extensive Examples**: Complete examples for all features

## 🚀 Quick Start

### Installation

```bash
go get github.com/monorepo/anomaly-detection-client-go
```

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    anomaly_detection "github.com/monorepo/anomaly-detection-client-go"
)

func main() {
    // Configure the client
    config := &anomaly_detection.ClientConfig{
        BaseURL:    "https://api.anomaly-detection.com",
        APIKey:     "your-api-key-here",
        Timeout:    30 * time.Second,
        MaxRetries: 3,
    }

    // Create client
    client := anomaly_detection.NewClient(config)
    defer client.Close()

    // Create context
    ctx := context.Background()

    // Authenticate
    if err := client.AuthenticateWithAPIKey(ctx, config.APIKey); err != nil {
        log.Fatalf("Authentication failed: %v", err)
    }

    // Prepare data for anomaly detection
    data := [][]float64{
        {1.0, 2.0, 3.0},
        {1.1, 2.1, 3.1},
        {0.9, 1.9, 2.9},
        {10.0, 20.0, 30.0}, // This should be detected as an anomaly
    }

    // Detect anomalies
    request := &anomaly_detection.DetectionRequest{
        Data:           data,
        Algorithm:      "isolation_forest",
        Contamination:  floatPtr(0.25),
        ExplainResults: true,
    }

    response, err := client.DetectAnomalies(ctx, request)
    if err != nil {
        log.Fatalf("Detection failed: %v", err)
    }

    // Print results
    fmt.Printf("Detected %d anomalies out of %d samples\n", 
        response.AnomaliesDetected, response.TotalSamples)
    
    for i, isAnomaly := range response.Anomalies {
        if isAnomaly {
            fmt.Printf("Sample %d is an anomaly (confidence: %.3f)\n", 
                i, response.ConfidenceScores[i])
        }
    }
}

func floatPtr(f float64) *float64 { return &f }
```

## 📚 API Reference

### Client Configuration

```go
type ClientConfig struct {
    BaseURL           string        // API base URL
    APIKey            string        // API key for authentication
    JWTToken          string        // JWT token (alternative to API key)
    Timeout           time.Duration // Request timeout
    MaxRetries        int           // Maximum retry attempts
    RetryBackoff      time.Duration // Retry backoff duration
    RateLimit         int           // Requests per minute
    UserAgent         string        // Custom user agent
    EnableStreaming   bool          // Enable WebSocket streaming
    StreamingURL      string        // WebSocket endpoint URL
    Debug             bool          // Enable debug logging
}
```

### Core Methods

#### Anomaly Detection

```go
// Single algorithm detection
func (c *Client) DetectAnomalies(ctx context.Context, request *DetectionRequest) (*DetectionResponse, error)

// Ensemble detection with multiple algorithms
func (c *Client) DetectAnomaliesEnsemble(ctx context.Context, request *EnsembleRequest) (*EnsembleResponse, error)
```

#### Model Management

```go
// Train a new model
func (c *Client) TrainModel(ctx context.Context, request *TrainingRequest) (*TrainingResponse, error)

// List trained models
func (c *Client) ListModels(ctx context.Context, page, perPage int) (*ModelsListResponse, error)

// Get specific model
func (c *Client) GetModel(ctx context.Context, modelID string) (*Model, error)

// Delete model
func (c *Client) DeleteModel(ctx context.Context, modelID string) error
```

#### Streaming

```go
// Start real-time streaming
func (c *Client) StartStreaming(ctx context.Context) (<-chan StreamingMessage, error)

// Send data for real-time detection
func (c *Client) SendStreamingData(ctx context.Context, data []float64) error

// Stop streaming
func (c *Client) StopStreaming() error
```

#### Utility Methods

```go
// List available algorithms
func (c *Client) ListAlgorithms(ctx context.Context) (*AlgorithmsResponse, error)

// Health check
func (c *Client) HealthCheck(ctx context.Context) (*HealthResponse, error)
```

## 🎯 Advanced Examples

### Ensemble Detection

```go
request := &anomaly_detection.EnsembleRequest{
    Data:       data,
    Algorithms: []string{"isolation_forest", "one_class_svm", "local_outlier_factor"},
    Method:     "voting",
    Parameters: map[string]interface{}{
        "contamination": 0.1,
    },
}

response, err := client.DetectAnomaliesEnsemble(ctx, request)
if err != nil {
    log.Fatalf("Ensemble detection failed: %v", err)
}

fmt.Printf("Ensemble method: %s\n", response.Method)
fmt.Printf("Individual results:\n")
for algorithm, results := range response.IndividualResults {
    anomalyCount := countAnomalies(results)
    fmt.Printf("  %s: %d anomalies\n", algorithm, anomalyCount)
}
```

### Model Training

```go
request := &anomaly_detection.TrainingRequest{
    Data:            trainingData,
    Algorithm:       "isolation_forest",
    ModelName:       "production_model",
    ValidationSplit: 0.2,
    Parameters: map[string]interface{}{
        "n_estimators":  200,
        "contamination": 0.1,
        "random_state":  42,
    },
}

response, err := client.TrainModel(ctx, request)
if err != nil {
    log.Fatalf("Training failed: %v", err)
}

fmt.Printf("Model trained successfully!\n")
fmt.Printf("Model ID: %s\n", response.ModelID)
fmt.Printf("Training time: %.2f seconds\n", response.TrainingTime)
```

### Real-time Streaming

```go
// Start streaming
msgChan, err := client.StartStreaming(ctx)
if err != nil {
    log.Fatalf("Failed to start streaming: %v", err)
}

// Handle streaming messages
go func() {
    for msg := range msgChan {
        switch msg.Type {
        case "anomaly":
            if msg.Anomaly {
                fmt.Printf("🚨 ANOMALY: %v (confidence: %.3f)\n", 
                    msg.Data, msg.Confidence)
            }
        case "error":
            fmt.Printf("❌ Error: %s\n", msg.Message)
        }
    }
}()

// Send real-time data
ticker := time.NewTicker(100 * time.Millisecond)
defer ticker.Stop()

for i := 0; i < 100; i++ {
    select {
    case <-ticker.C:
        dataPoint := generateSensorReading() // Your data generation logic
        if err := client.SendStreamingData(ctx, dataPoint); err != nil {
            log.Printf("Failed to send data: %v", err)
        }
    case <-ctx.Done():
        return
    }
}
```

## 🔧 Error Handling

The SDK provides comprehensive error types for different scenarios:

```go
import "errors"

response, err := client.DetectAnomalies(ctx, request)
if err != nil {
    switch {
    case anomaly_detection.IsAuthenticationError(err):
        fmt.Println("Authentication failed - check your API key")
    case anomaly_detection.IsRateLimitError(err):
        if rateLimitErr, ok := err.(anomaly_detection.RateLimitError); ok {
            fmt.Printf("Rate limited - retry after %d seconds\n", rateLimitErr.RetryAfter)
        }
    case anomaly_detection.IsValidationError(err):
        fmt.Printf("Validation error: %v\n", err)
    case anomaly_detection.IsNetworkError(err):
        fmt.Printf("Network error: %v\n", err)
    default:
        fmt.Printf("Unexpected error: %v\n", err)
    }
    return
}
```

## 🛠️ Available Algorithms

The SDK supports 20+ algorithms across different categories:

### Statistical Methods
- `z_score` - Z-Score anomaly detection
- `modified_z_score` - Modified Z-Score with median absolute deviation
- `iqr` - Interquartile Range method
- `seasonal_decomposition` - Seasonal decomposition of time series

### Machine Learning
- `isolation_forest` - Isolation Forest (recommended for most use cases)
- `one_class_svm` - One-Class Support Vector Machine
- `local_outlier_factor` - Local Outlier Factor
- `elliptic_envelope` - Elliptic Envelope (Mahalanobis distance)

### Deep Learning
- `autoencoder` - Neural network autoencoder
- `lstm_autoencoder` - LSTM-based autoencoder for time series
- `transformer` - Transformer-based anomaly detection
- `variational_autoencoder` - Variational autoencoder

### Ensemble Methods
- `voting` - Majority voting across algorithms
- `average` - Average confidence scores
- `weighted` - Weighted combination with custom weights

## 📊 Performance and Scaling

### Rate Limiting

The SDK includes built-in rate limiting to comply with API limits:

```go
config := &anomaly_detection.ClientConfig{
    RateLimit: 100, // 100 requests per minute
    // ... other config
}
```

### Context and Timeouts

All methods support context for cancellation and timeouts:

```go
// Set timeout for individual requests
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

response, err := client.DetectAnomalies(ctx, request)
```

### Retry Logic

Automatic retry with exponential backoff for transient failures:

```go
config := &anomaly_detection.ClientConfig{
    MaxRetries:   3,
    RetryBackoff: time.Second,
    // ... other config
}
```

## 🧪 Testing

Run the test suite:

```bash
go test ./...
```

Run tests with coverage:

```bash
go test -cover ./...
```

Run benchmarks:

```bash
go test -bench=. ./...
```

## 📁 Project Structure

```
anomaly-detection-client-go/
├── client.go              # Main client implementation
├── types.go               # Request/response types
├── errors.go              # Error types and handling
├── auth.go                # Authentication manager
├── rate_limiter.go        # Rate limiting implementation
├── examples/              # Usage examples
│   ├── basic_detection.go
│   └── streaming_detection.go
├── go.mod                 # Go module file
└── README.md             # This file
```

## 🔐 Security Best Practices

### API Key Management

- Store API keys in environment variables, not in code
- Use different API keys for different environments
- Rotate API keys regularly

```go
config := &anomaly_detection.ClientConfig{
    APIKey: os.Getenv("ANOMALY_DETECTION_API_KEY"),
    // ... other config
}
```

### JWT Token Handling

- JWT tokens are automatically refreshed when they expire
- Tokens are stored securely in memory only
- Use HTTPS in production

## 🚀 Production Deployment

### Docker Example

```dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o anomaly-detector ./cmd/detector

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/anomaly-detector .

CMD ["./anomaly-detector"]
```

### Environment Variables

```bash
# Required
ANOMALY_DETECTION_API_KEY=your-api-key-here
ANOMALY_DETECTION_BASE_URL=https://api.anomaly-detection.com

# Optional
ANOMALY_DETECTION_TIMEOUT=30s
ANOMALY_DETECTION_MAX_RETRIES=3
ANOMALY_DETECTION_RATE_LIMIT=100
ANOMALY_DETECTION_DEBUG=false
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.anomaly-detection.com](https://docs.anomaly-detection.com)
- **API Reference**: [https://api.anomaly-detection.com/docs](https://api.anomaly-detection.com/docs)
- **Issues**: [GitHub Issues](https://github.com/monorepo/anomaly-detection-client-go/issues)
- **Discussions**: [GitHub Discussions](https://github.com/monorepo/anomaly-detection-client-go/discussions)

## 🔗 Related SDKs

- [Python SDK](https://github.com/monorepo/anomaly-detection-client-python)
- [TypeScript SDK](https://github.com/monorepo/anomaly-detection-client-typescript)
- [Java SDK](https://github.com/monorepo/anomaly-detection-client-java)

---

**Happy anomaly detecting! 🎯**