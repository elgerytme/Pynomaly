# anomaly_detection SDKs

This directory contains official client SDKs for the anomaly_detection anomaly detection API in multiple programming languages.

## Available SDKs

### 🐍 Python SDK
**Location**: `python/`  
**Installation**: `pip install anomaly_detection-client`  
**Features**: 
- Full async/await support
- Type hints with Pydantic models
- Comprehensive error handling
- Built-in retry logic and rate limiting

```python
from anomaly_detection_client import AnomalyDetectionClient

async with AnomalyDetectionClient(api_key="your-key") as client:
    result = await client.detection.detect(
        data=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
        algorithm="isolation_forest"
    )
    print(f"Anomalies: {result.anomalies}")
```

### 📘 TypeScript SDK
**Location**: `typescript/`  
**Installation**: `npm install @anomaly_detection/client`  
**Features**:
- Full TypeScript type definitions
- Node.js and browser compatibility
- Promise-based async operations
- Comprehensive error handling

```typescript
import { AnomalyDetectionClient } from '@anomaly_detection/client';

const client = new AnomalyDetectionClient({ apiKey: 'your-key' });

const result = await client.detection.detect({
  data: [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
  algorithm: 'isolation_forest'
});

console.log('Anomalies:', result.anomalies);
```

### ☕ Java SDK
**Location**: `java/`  
**Installation**: Maven/Gradle dependency  
**Features**:
- Java 11+ compatibility
- Fluent builder patterns
- CompletableFuture async support
- Jackson JSON serialization

```java
AnomalyDetectionClient client = AnomalyDetectionClient.builder()
    .apiKey("your-key")
    .build();

DetectionRequest request = DetectionRequest.builder()
    .data(Arrays.asList(1.0, 2.0, 3.0, 100.0, 4.0, 5.0))
    .algorithm("isolation_forest")
    .build();

DetectionResponse result = client.detection().detect(request);
System.out.println("Anomalies: " + result.getAnomalies());
```

## Common Features

All SDKs provide:

- ✅ **Complete API Coverage**: Full access to all anomaly detection API endpoints
- 🔐 **Authentication**: JWT and API Key authentication support
- 🔄 **Retry Logic**: Automatic retry with exponential backoff
- ⚡ **Rate Limiting**: Built-in request throttling
- 🛡️ **Error Handling**: Comprehensive exception types
- 📚 **Documentation**: Extensive inline documentation
- 🧪 **Testing**: Comprehensive test suites
- 📦 **CI/CD**: Automated building and publishing

## Getting Started

1. **Choose your SDK** based on your preferred programming language
2. **Install the SDK** using the language's package manager
3. **Get your API key** from the anomaly_detection dashboard
4. **Follow the examples** in each SDK's README file

## API Documentation

For detailed API documentation, visit: https://docs.anomaly_detection.com/api

## Support

- 📧 Email: support@anomaly_detection.com
- 📖 Documentation: https://docs.anomaly_detection.com
- 🐛 Issues: https://github.com/anomaly_detection/anomaly_detection/issues
- 💬 Community: https://discord.gg/anomaly_detection

## Contributing

We welcome contributions to our SDKs! Please see the CONTRIBUTING.md file in each SDK directory for specific guidelines.

## License

All SDKs are licensed under the MIT License. See LICENSE file for details.