# Anomaly Detection SDK Documentation

Complete documentation for the Anomaly Detection SDKs, covering Python, JavaScript/TypeScript, and Go implementations.

## Quick Navigation

- **[Getting Started](getting-started.md)** - Installation and basic setup
- **[API Reference](api-reference.md)** - Complete API documentation
- **[SDK Guides](sdk-guides/)** - Language-specific guides
- **[Examples](../examples/)** - Working code examples
- **[Migration Guide](migration-guide.md)** - Upgrading between versions

## SDK Overview

The Anomaly Detection SDK provides unified access to anomaly detection capabilities across multiple programming languages. Each SDK offers:

### Core Features
- **Synchronous & Asynchronous Clients** - Choose the right client for your use case
- **Real-time Streaming** - WebSocket-based streaming anomaly detection
- **Multiple Algorithms** - Support for various anomaly detection algorithms
- **Model Management** - Train, deploy, and manage custom models
- **Comprehensive Error Handling** - Detailed error types and messages
- **Type Safety** - Full type definitions and validation

### Supported Languages

| SDK | Version | Documentation | Package |
|-----|---------|---------------|---------|
| **Python** | 1.0.0 | [Python Guide](sdk-guides/python.md) | `anomaly-detection-sdk` |
| **JavaScript/TypeScript** | 1.0.0 | [JavaScript Guide](sdk-guides/javascript.md) | `@anomaly-detection/sdk` |
| **Go** | 1.0.0 | [Go Guide](sdk-guides/go.md) | `github.com/anomaly-detection/go-sdk` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Application                      │
├─────────────────────────────────────────────────────────────┤
│                    Anomaly Detection SDK                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Client    │  │  Streaming  │  │    Utilities &      │  │
│  │    API      │  │   Client    │  │     Models          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Network Layer                             │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │    HTTP     │  │ WebSocket    │                          │
│  │   Client    │  │   Client     │                          │
│  └─────────────┘  └─────────────┘                          │
├─────────────────────────────────────────────────────────────┤
│                Anomaly Detection Service                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Detection  │  │  Streaming  │  │     Model           │  │
│  │   Engine    │  │   Engine    │  │   Management        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Data Format
All SDKs use consistent data formats:
- **Input Data**: 2D arrays where each row represents a data point
- **Feature Vectors**: Arrays of numeric values representing features
- **Results**: Structured objects with anomaly indices, scores, and metadata

### Algorithms
Supported anomaly detection algorithms:
- **Isolation Forest**: Tree-based algorithm for detecting anomalies
- **Local Outlier Factor**: Density-based anomaly detection
- **One-Class SVM**: Support Vector Machine for novelty detection
- **Elliptic Envelope**: Gaussian distribution-based detection
- **Autoencoder**: Neural network-based anomaly detection
- **Ensemble**: Combination of multiple algorithms

### Client Types

#### Synchronous Client
- **Use Case**: Simple scripts, batch processing
- **Features**: Blocking calls, easy error handling
- **Best For**: Data processing pipelines, analysis scripts

#### Asynchronous Client
- **Use Case**: Web applications, concurrent processing
- **Features**: Non-blocking calls, parallel execution
- **Best For**: Web services, real-time applications

#### Streaming Client
- **Use Case**: Real-time anomaly detection
- **Features**: WebSocket connection, event-driven
- **Best For**: Monitoring systems, live data feeds

## Installation

### Python
```bash
pip install anomaly-detection-sdk
```

### JavaScript/TypeScript
```bash
npm install @anomaly-detection/sdk
```

### Go
```bash
go get github.com/anomaly-detection/go-sdk
```

## Quick Example

### Python
```python
from anomaly_detection_sdk import AnomalyDetectionClient, AlgorithmType

client = AnomalyDetectionClient(base_url="http://localhost:8000")
data = [[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]]  # Last point is anomalous

result = client.detect_anomalies(data, AlgorithmType.ISOLATION_FOREST)
print(f"Found {result.anomaly_count} anomalies")
```

### JavaScript/TypeScript
```javascript
import { AnomalyDetectionClient, AlgorithmType } from '@anomaly-detection/sdk';

const client = new AnomalyDetectionClient({ baseUrl: 'http://localhost:8000' });
const data = [[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]];

const result = await client.detectAnomalies(data, AlgorithmType.ISOLATION_FOREST);
console.log(`Found ${result.anomalyCount} anomalies`);
```

### Go
```go
import anomaly "github.com/anomaly-detection/go-sdk"

client := anomaly.NewClient(anomaly.ClientConfig{BaseURL: "http://localhost:8000"})
data := [][]float64{{1.0, 2.0}, {1.1, 2.1}, {10.0, 20.0}}

result, err := client.DetectAnomalies(ctx, data, anomaly.IsolationForest, nil, false)
fmt.Printf("Found %d anomalies\n", result.AnomalyCount)
```

## Documentation Structure

### Core Documentation
- **[Getting Started](getting-started.md)** - Installation, configuration, first detection
- **[API Reference](api-reference.md)** - Complete method and type documentation
- **[Configuration](configuration.md)** - Client and service configuration options

### SDK-Specific Guides
- **[Python SDK Guide](sdk-guides/python.md)** - Python-specific features and patterns
- **[JavaScript SDK Guide](sdk-guides/javascript.md)** - JavaScript/TypeScript usage patterns
- **[Go SDK Guide](sdk-guides/go.md)** - Go-specific implementation details

### Advanced Topics
- **[Streaming Guide](streaming-guide.md)** - Real-time anomaly detection
- **[Model Management](model-management.md)** - Training and managing custom models
- **[Performance Tuning](performance-tuning.md)** - Optimization best practices
- **[Error Handling](error-handling.md)** - Error types and handling strategies

### Integration Guides
- **[Web Applications](integrations/web-applications.md)** - Frontend and backend integration
- **[Data Pipelines](integrations/data-pipelines.md)** - Batch processing workflows
- **[Microservices](integrations/microservices.md)** - Service-to-service communication
- **[Monitoring Systems](integrations/monitoring-systems.md)** - Real-time monitoring integration

## Support and Community

### Getting Help
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and examples
- **Community**: Join discussions and share experiences

### Contributing
- **SDK Development**: Improve existing SDKs or add new language support
- **Documentation**: Help improve guides and examples
- **Examples**: Contribute new use cases and patterns

### Versioning
All SDKs follow semantic versioning:
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

## What's Next?

1. **[Get Started](getting-started.md)** - Install and run your first detection
2. **[Choose Your SDK](sdk-guides/)** - Language-specific guides
3. **[Explore Examples](../examples/)** - Working code examples
4. **[Advanced Features](streaming-guide.md)** - Streaming and model management

---

**Note**: This documentation covers SDK version 1.0.0. For updates and changes, see the [changelog](changelog.md) and [migration guide](migration-guide.md).