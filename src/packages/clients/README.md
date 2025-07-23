# Platform Client Libraries

This directory contains official client libraries for the platform's services in multiple programming languages.

## 🌟 Overview

Our client libraries provide easy-to-use interfaces for all platform services, including:

- **Anomaly Detection**: Detect outliers and anomalies in your data
- **MLOps**: Manage ML pipelines, models, and deployments
- **Data Analytics**: Analyze and visualize your data
- **Enterprise Features**: Authentication, governance, and scalability

## 📚 Available Clients

### Python SDK
**Location**: `anomaly_detection_client/`  
**Installation**: `pip install anomaly-detection-client`

```python
from anomaly_detection_client import AnomalyDetectionClient

async with AnomalyDetectionClient(api_key="your-key") as client:
    result = await client.detect(
        data=[[1.0, 2.0], [100.0, 200.0]],
        algorithm="isolation_forest"
    )
```

**Features**:
- ✅ Full async/await support
- ✅ Type hints with Pydantic models
- ✅ NumPy and Pandas integration
- ✅ Comprehensive error handling
- ✅ Automatic retry logic

### TypeScript/JavaScript SDK
**Location**: `platform_client_ts/`  
**Installation**: `npm install @platform/client`

```typescript
import { PlatformClient } from '@platform/client';

const client = new PlatformClient({ apiKey: 'your-key' });
const result = await client.anomalyDetection.detect({
  data: [[1.0, 2.0], [100.0, 200.0]],
  algorithm: 'isolation_forest'
});
```

**Features**:
- ✅ Full TypeScript type definitions
- ✅ Node.js and browser support
- ✅ Framework integrations (React, Vue, Angular)
- ✅ Tree-shakeable imports
- ✅ Modern Promise-based API

### Core SDK (Shared)
**Location**: `../shared/sdk_core/`

Common functionality shared across all client libraries:
- Authentication and token management
- HTTP client with retry logic
- Rate limiting and error handling
- Configuration management
- Response models and types

## 🚀 Quick Start

### 1. Choose Your Language

Pick the client library that matches your development stack:
- **Python**: Data science, ML workflows, backend services
- **TypeScript/JavaScript**: Web applications, Node.js services, React/Vue apps

### 2. Install the Client

```bash
# Python
pip install anomaly-detection-client

# JavaScript/TypeScript
npm install @platform/client
```

### 3. Get Your API Key

1. Sign up at [platform.com](https://platform.com)
2. Generate an API key from your dashboard
3. Set the environment variable: `export PLATFORM_API_KEY=your-key`

### 4. Run Your First Detection

```python
# Python
import asyncio
from anomaly_detection_client import AnomalyDetectionClient

async def main():
    async with AnomalyDetectionClient() as client:  # Uses env var
        result = await client.detect(
            data=[[1, 2], [2, 3], [100, 200]],  # Last point is anomaly
            algorithm="isolation_forest"
        )
        print(f"Anomalies found: {result.anomalies}")

asyncio.run(main())
```

```typescript
// TypeScript
import { PlatformClient } from '@platform/client';

const client = new PlatformClient({
  apiKey: process.env.PLATFORM_API_KEY
});

const result = await client.anomalyDetection.detect({
  data: [[1, 2], [2, 3], [100, 200]],
  algorithm: 'isolation_forest'
});

console.log('Anomalies found:', result.anomalies);
```

## 🎯 Examples

### Comprehensive Examples
- **Python**: [`examples/python/`](examples/python/)
- **TypeScript**: [`examples/typescript/`](examples/typescript/)

### Framework-Specific Examples
- **React**: [`examples/react/`](examples/react/)
- **Vue**: [`examples/vue/`](examples/vue/)
- **Angular**: [`examples/angular/`](examples/angular/)
- **Jupyter Notebooks**: [`examples/notebooks/`](examples/notebooks/)

## 🔧 Configuration

### Environment Variables

```bash
# Required
PLATFORM_API_KEY=your-api-key

# Optional
PLATFORM_BASE_URL=https://api.platform.com
PLATFORM_TIMEOUT=30000
PLATFORM_MAX_RETRIES=3
```

### Configuration Objects

```python
# Python
from sdk_core import ClientConfig, Environment

config = ClientConfig.for_environment(
    Environment.PRODUCTION,
    api_key="your-key",
    timeout=60.0,
    max_retries=5
)
```

```typescript
// TypeScript
import { PlatformClient, Environment } from '@platform/client';

const client = new PlatformClient({
  apiKey: 'your-key',
  environment: Environment.Production,
  timeout: 60000,
  maxRetries: 5
});
```

## 🛡️ Error Handling

All clients provide comprehensive error handling:

```python
# Python
from anomaly_detection_client import ValidationError, RateLimitError

try:
    result = await client.detect(data=invalid_data)
except ValidationError as e:
    print(f"Validation failed: {e.message}")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
```

```typescript
// TypeScript
import { ValidationError, RateLimitError } from '@platform/client';

try {
  const result = await client.anomalyDetection.detect({ data: invalidData });
} catch (error) {
  if (error instanceof ValidationError) {
    console.log('Validation failed:', error.message);
  } else if (error instanceof RateLimitError) {
    console.log('Rate limited. Retry after:', error.retryAfter);
  }
}
```

## 🔐 Authentication

All clients support multiple authentication methods:

### JWT (Recommended)
```bash
export PLATFORM_API_KEY=your-jwt-api-key
```

### Direct Token
```python
client = AnomalyDetectionClient(jwt_token="your-jwt-token")
```

### Custom Authentication
```python
from sdk_core import TokenAuth

auth = TokenAuth("custom-token", "Bearer")
client = AnomalyDetectionClient(auth=auth)
```

## 🏗️ Architecture

```
Platform Client Libraries
├── Core SDK (shared/)           # Common functionality
│   ├── Authentication          # JWT, token management
│   ├── HTTP Client             # Retry, rate limiting
│   ├── Error Handling          # Exception hierarchy
│   └── Configuration           # Settings management
├── Python SDK                  # Python-specific client
│   ├── Pydantic Models        # Type-safe data models
│   ├── Async/Sync Support     # Both async and sync APIs
│   └── NumPy/Pandas Support   # Data science integration
└── TypeScript SDK             # JavaScript/TypeScript client
    ├── Type Definitions       # Full TypeScript types
    ├── Framework Support      # React, Vue, Angular
    └── Browser Compatibility  # Modern browsers + Node.js
```

## 📊 Service Coverage

| Service | Python | TypeScript | Status |
|---------|--------|------------|--------|
| Anomaly Detection | ✅ | ✅ | Production Ready |
| MLOps Platform | ✅ | ✅ | Production Ready |
| Data Analytics | 🚧 | 🚧 | In Development |
| Enterprise Auth | ✅ | ✅ | Production Ready |
| Enterprise Governance | 🚧 | 🚧 | In Development |

## 🧪 Testing

Each client includes comprehensive test suites:

```bash
# Python
cd anomaly_detection_client/
pytest

# TypeScript
cd platform_client_ts/
npm test
```

## 📖 Documentation

- **API Reference**: [docs.platform.com/api](https://docs.platform.com/api)
- **Python SDK**: [docs.platform.com/clients/python](https://docs.platform.com/clients/python)
- **TypeScript SDK**: [docs.platform.com/clients/typescript](https://docs.platform.com/clients/typescript)
- **Examples**: [docs.platform.com/examples](https://docs.platform.com/examples)

## 🤝 Contributing

We welcome contributions! Please see:
- [CONTRIBUTING.md](../../../CONTRIBUTING.md)
- [Client Development Guide](docs/DEVELOPMENT.md)
- [Adding New Services](docs/NEW_SERVICES.md)

## 📄 License

All client libraries are licensed under the MIT License. See [LICENSE](../../../LICENSE) for details.

## 🆘 Support

- 📧 **Email**: support@platform.com
- 💬 **Discord**: [discord.gg/platform](https://discord.gg/platform)
- 🐛 **Issues**: [GitHub Issues](https://github.com/platform/platform/issues)
- 📚 **Docs**: [docs.platform.com](https://docs.platform.com)

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and breaking changes.