# Pynomaly Go SDK

Official Go client library for the Pynomaly anomaly detection platform.

## Features

- **Complete API Coverage**: Full access to all Pynomaly platform features
- **Type Safety**: Comprehensive type definitions for better development experience
- **Context Support**: Full context.Context support for cancellation and timeouts
- **Error Handling**: Comprehensive error types with detailed information
- **Multi-tenant Support**: Built-in support for multi-tenant architectures
- **Authentication**: JWT and API key authentication support
- **Retry Logic**: Automatic retry with exponential backoff for transient failures
- **High Performance**: Optimized for high-throughput services and microservices

## Installation

```bash
go get github.com/pynomaly/pynomaly-go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/pynomaly/pynomaly-go/pkg/pynomaly"
)

func main() {
    // Initialize client
    client := pynomaly.NewClient(pynomaly.Config{
        BaseURL:  "https://api.pynomaly.ai",
        APIKey:   "your-api-key",
        TenantID: "your-tenant-id",
        Timeout:  30 * time.Second,
    })

    ctx := context.Background()

    // Create a detector
    detector, err := client.Detection.CreateDetector(ctx, &pynomaly.CreateDetectorRequest{
        Name:      "My Go Detector",
        Algorithm: "isolation_forest",
        Parameters: map[string]interface{}{
            "contamination": 0.1,
            "n_estimators":  100,
        },
    })
    if err != nil {
        log.Fatalf("Failed to create detector: %v", err)
    }

    fmt.Printf("Created detector: %s\n", detector.ID)

    // Upload and create dataset
    fileContent := []byte("your,csv,data") // Your actual CSV data
    uploadResp, err := client.UploadFile(ctx, "dataset.csv", fileContent, "text/csv")
    if err != nil {
        log.Fatalf("Failed to upload file: %v", err)
    }

    dataset, err := client.Detection.CreateDataset(ctx, uploadResp.FileID, "My Dataset", "Sample dataset", nil)
    if err != nil {
        log.Fatalf("Failed to create dataset: %v", err)
    }

    // Train detector
    trainingJob, err := client.Detection.TrainDetector(ctx, detector.ID, &pynomaly.TrainDetectorRequest{
        DatasetID:       dataset.ID,
        ValidationSplit: &[]float64{0.2}[0],
    })
    if err != nil {
        log.Fatalf("Failed to start training: %v", err)
    }

    fmt.Printf("Training started with job ID: %s\n", trainingJob.JobID)

    // Wait for training to complete (simplified polling)
    for {
        status, err := client.Detection.GetTrainingStatus(ctx, detector.ID, trainingJob.JobID)
        if err != nil {
            log.Fatalf("Failed to get training status: %v", err)
        }

        fmt.Printf("Training progress: %.2f%%\n", status.Progress*100)
        
        if status.Status == "completed" {
            break
        } else if status.Status == "failed" {
            log.Fatalf("Training failed")
        }

        time.Sleep(5 * time.Second)
    }

    // Detect anomalies
    results, err := client.Detection.DetectAnomalies(ctx, &pynomaly.DetectionRequest{
        DatasetID:  dataset.ID,
        DetectorID: detector.ID,
    })
    if err != nil {
        log.Fatalf("Failed to detect anomalies: %v", err)
    }

    fmt.Printf("Found %d anomalies out of %d samples (%.2f%%)\n", 
        results.AnomalyCount, results.TotalSamples, results.AnomalyRate*100)
}
```

## API Reference

### Detection Client

```go
// List detectors
detectors, err := client.Detection.ListDetectors(ctx, &pynomaly.ListOptions{
    Page:     &[]int{1}[0],
    PageSize: &[]int{50}[0],
})

// Get detector
detector, err := client.Detection.GetDetector(ctx, "detector-id")

// Create detector
detector, err := client.Detection.CreateDetector(ctx, &pynomaly.CreateDetectorRequest{
    Name:      "My Detector",
    Algorithm: "isolation_forest",
    Parameters: map[string]interface{}{
        "contamination": 0.1,
    },
})

// Train detector
job, err := client.Detection.TrainDetector(ctx, "detector-id", &pynomaly.TrainDetectorRequest{
    DatasetID:       "dataset-id",
    ValidationSplit: &[]float64{0.2}[0],
})

// Detect anomalies
results, err := client.Detection.DetectAnomalies(ctx, &pynomaly.DetectionRequest{
    DatasetID:  "dataset-id",
    DetectorID: "detector-id",
})
```

### Streaming Client

```go
// Create stream processor
processor, err := client.Streaming.CreateProcessor(ctx, &pynomaly.CreateProcessorRequest{
    ProcessorID: "my-processor",
    DetectorID:  "detector-id",
    WindowConfig: pynomaly.WindowConfig{
        Type:        pynomaly.WindowTypeTumbling,
        SizeSeconds: 60,
    },
})

// Start processor
status, err := client.Streaming.StartProcessor(ctx, "my-processor")

// Send data
records := []pynomaly.StreamRecord{
    {
        ID:        "record-1",
        Timestamp: time.Now(),
        Data:      map[string]interface{}{"value": 42},
        TenantID:  "tenant-id",
        Metadata:  map[string]interface{}{},
    },
}

response, err := client.Streaming.SendData(ctx, "my-processor", records)

// Get metrics
metrics, err := client.Streaming.GetProcessorMetrics(ctx, "my-processor")
```

### A/B Testing Client

```go
// Create A/B test
test, err := client.ABTesting.CreateTest(ctx, &pynomaly.CreateABTestRequest{
    Name:        "Algorithm Comparison",
    Description: "Compare two detection algorithms",
    Variants: []pynomaly.TestVariant{
        {
            Name:              "Control",
            Description:       "Isolation Forest",
            DetectorID:        "detector-1",
            TrafficPercentage: 50,
            IsControl:         true,
        },
        {
            Name:              "Treatment",
            Description:       "Local Outlier Factor",
            DetectorID:        "detector-2",
            TrafficPercentage: 50,
            IsControl:         false,
        },
    },
})

// Start test
status, err := client.ABTesting.StartTest(ctx, test.ID)

// Get results
results, err := client.ABTesting.GetTestResults(ctx, test.ID, nil)
```

### User Management Client

```go
// Create user
user, err := client.Users.CreateUser(ctx, &pynomaly.CreateUserRequest{
    Email:     "user@example.com",
    FirstName: "John",
    LastName:  "Doe",
    Password:  "secure-password",
    Roles:     []pynomaly.UserRole{pynomaly.RoleDataScientist},
    TenantID:  "tenant-id",
})

// Get current user
currentUser, err := client.Users.GetCurrentUser(ctx)

// Add role to user
updatedUser, err := client.Users.AddUserRole(ctx, user.ID, pynomaly.RoleAnalyst)

// Get current tenant
tenant, err := client.Users.GetCurrentTenant(ctx)
```

### Compliance Client

```go
// List audit events
events, err := client.Compliance.ListAuditEvents(ctx, &pynomaly.AuditEventListOptions{
    Severity:  &pynomaly.SeverityHigh,
    StartDate: &"2023-01-01",
    EndDate:   &"2023-12-31",
})

// Create audit event
event, err := client.Compliance.CreateAuditEvent(ctx, &pynomaly.CreateAuditEventRequest{
    Action:      "user_login",
    Severity:    pynomaly.SeverityMedium,
    UserID:      "user-id",
    Details:     map[string]interface{}{"ip": "192.168.1.1"},
    Outcome:     "success",
})

// Get compliance assessment
assessment, err := client.Compliance.GetComplianceAssessment(ctx, []pynomaly.ComplianceFramework{
    pynomaly.FrameworkGDPR,
    pynomaly.FrameworkHIPAA,
})

// Process right to be forgotten
deletion, err := client.Compliance.DeleteUserData(ctx, "user-id", &pynomaly.DeleteUserDataOptions{
    HardDelete:      &[]bool{true}[0],
    Anonymize:       &[]bool{false}[0],
    RetainAuditLogs: &[]bool{true}[0],
})
```

## Error Handling

```go
import (
    "errors"
    "github.com/pynomaly/pynomaly-go/pkg/pynomaly"
)

detector, err := client.Detection.CreateDetector(ctx, invalidRequest)
if err != nil {
    var authErr *pynomaly.AuthenticationError
    var validationErr *pynomaly.ValidationError
    var networkErr *pynomaly.NetworkError
    var pynomalyErr *pynomaly.PynomalyError

    switch {
    case errors.As(err, &authErr):
        fmt.Printf("Authentication failed: %s\n", authErr.Message)
    case errors.As(err, &validationErr):
        fmt.Printf("Validation error: %s\n", validationErr.Message)
        fmt.Printf("Details: %+v\n", validationErr.Details)
    case errors.As(err, &networkErr):
        fmt.Printf("Network error: %s\n", networkErr.Message)
        // Implement retry logic
    case errors.As(err, &pynomalyErr):
        fmt.Printf("Pynomaly error [%s]: %s\n", pynomalyErr.Code, pynomalyErr.Message)
    default:
        fmt.Printf("Unknown error: %v\n", err)
    }
}

// Check if error is retryable
if pynomaly.IsRetryableError(err) {
    // Implement retry logic
    fmt.Println("Error is retryable, implementing backoff...")
}
```

## Configuration

```go
client := pynomaly.NewClient(pynomaly.Config{
    BaseURL:    "https://api.pynomaly.ai",
    APIKey:     "your-api-key",
    TenantID:   "your-tenant-id", // Optional for multi-tenant
    Timeout:    30 * time.Second,
    UserAgent:  "my-service/1.0.0",
    HTTPClient: customHTTPClient, // Optional custom HTTP client
})

// Update authentication
client.SetAuth("new-api-key", "new-tenant-id")

// Clear authentication
client.ClearAuth()
```

## Context and Cancellation

```go
// Create context with timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// Create context with cancellation
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

// Use context in API calls
detector, err := client.Detection.GetDetector(ctx, "detector-id")
if err != nil {
    if errors.Is(err, context.DeadlineExceeded) {
        fmt.Println("Request timed out")
    } else if errors.Is(err, context.Canceled) {
        fmt.Println("Request was cancelled")
    }
}
```

## High-Performance Usage

```go
// Use connection pooling for high-throughput scenarios
transport := &http.Transport{
    MaxIdleConns:        100,
    MaxIdleConnsPerHost: 100,
    IdleConnTimeout:     90 * time.Second,
}

httpClient := &http.Client{
    Transport: transport,
    Timeout:   30 * time.Second,
}

client := pynomaly.NewClient(pynomaly.Config{
    BaseURL:    "https://api.pynomaly.ai",
    APIKey:     "your-api-key",
    HTTPClient: httpClient,
})

// Use goroutines for concurrent operations
var wg sync.WaitGroup
results := make(chan *pynomaly.DetectionResult, 10)

for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(datasetID string) {
        defer wg.Done()
        
        result, err := client.Detection.DetectAnomalies(ctx, &pynomaly.DetectionRequest{
            DatasetID:  datasetID,
            DetectorID: "detector-id",
        })
        if err != nil {
            log.Printf("Detection failed for dataset %s: %v", datasetID, err)
            return
        }
        
        results <- result
    }(fmt.Sprintf("dataset-%d", i))
}

go func() {
    wg.Wait()
    close(results)
}()

for result := range results {
    fmt.Printf("Processed %d samples, found %d anomalies\n", 
        result.TotalSamples, result.AnomalyCount)
}
```

## Requirements

- **Go**: 1.21 or later
- **Dependencies**: 
  - github.com/gorilla/websocket (for WebSocket support)
  - github.com/stretchr/testify (for testing)

## Development

```bash
# Clone repository
git clone https://github.com/pynomaly/pynomaly-go.git
cd pynomaly-go

# Install dependencies
go mod download

# Run tests
go test ./...

# Run tests with coverage
go test -race -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Build examples
go build ./examples/...

# Lint code (requires golangci-lint)
golangci-lint run
```

## Examples

See the `examples/` directory for complete working examples:

- **Basic Detection**: Simple anomaly detection workflow
- **Streaming Processing**: Real-time data processing
- **A/B Testing**: Algorithm comparison and testing
- **High Throughput**: Optimized for high-performance scenarios
- **Error Handling**: Comprehensive error handling patterns

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://docs.pynomaly.ai
- **GitHub Issues**: https://github.com/pynomaly/pynomaly/issues
- **Email**: support@pynomaly.ai
- **Go Package Documentation**: https://pkg.go.dev/github.com/pynomaly/pynomaly-go