# Pynomaly SDKs

This directory contains official client SDKs for the Pynomaly pattern analysis API in multiple programming languages.

## Available SDKs

### 🐍 Python SDK
**Location**: `python/`  
**Installation**: `pip install pynomaly-client`  
**Features**: 
- Full async/await support
- Type hints with Pydantic models
- Comprehensive error handling
- Built-in retry logic and rate limiting

```python
from pynomaly_client import PynomaliClient

async with PynomaliClient(api_key="your-key") as client:
    result = await client.analysis.analyze(
        data=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
        algorithm="isolation_forest"
    )
    print(f"Patterns: {result.patterns}")
```

### 📘 TypeScript SDK
**Location**: `typescript/`  
**Installation**: `npm install @pynomaly/client`  
**Features**:
- Full TypeScript type definitions
- Node.js and browser compatibility
- Promise-based async operations
- Comprehensive error handling

```typescript
import { PynomaliClient } from '@pynomaly/client';

const client = new PynomaliClient({ apiKey: 'your-key' });

const result = await client.analysis.analyze({
  data: [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
  algorithm: 'isolation_forest'
});

console.log('Patterns:', result.patterns);
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
PynomaliClient client = PynomaliClient.builder()
    .apiKey("your-key")
    .build();

AnalysisRequest request = AnalysisRequest.builder()
    .data(Arrays.asList(1.0, 2.0, 3.0, 100.0, 4.0, 5.0))
    .algorithm("isolation_forest")
    .build();

AnalysisResponse result = client.analysis().analyze(request);
System.out.println("Patterns: " + result.getPatterns());
```

## Common Features

All SDKs provide:

- ✅ **Complete API Coverage**: Full access to all Pynomaly API endpoints
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
3. **Get your API key** from the Pynomaly dashboard
4. **Follow the examples** in each SDK's README file

## API Documentation

For detailed API documentation, visit: https://docs.pynomaly.com/api

## Support

- 📧 Email: support@pynomaly.com
- 📖 Documentation: https://docs.pynomaly.com
- 🐛 Issues: https://github.com/pynomaly/pynomaly/issues
- 💬 Community: https://discord.gg/pynomaly

## Contributing

We welcome contributions to our SDKs! Please see the CONTRIBUTING.md file in each SDK directory for specific guidelines.

## License

All SDKs are licensed under the MIT License. See LICENSE file for details.