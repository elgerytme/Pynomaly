# Pynomaly TypeScript SDK

![Version](https://img.shields.io/npm/v/@pynomaly/client)
![License](https://img.shields.io/npm/l/@pynomaly/client)
![TypeScript](https://img.shields.io/badge/TypeScript-5.1+-blue)
![Node.js](https://img.shields.io/badge/Node.js-16+-green)
![Browser](https://img.shields.io/badge/Browser-Modern-orange)

Official TypeScript/JavaScript client library for the Pynomaly anomaly detection API. This SDK provides comprehensive functionality with TypeScript type safety, modern Promise-based async/await API, WebSocket support for real-time updates, and framework integration examples.

## Features

- ✅ **TypeScript Support**: Full TypeScript definitions with comprehensive type safety
- ✅ **Modern API**: Promise-based async/await API with ES6+ support
- ✅ **Framework Integration**: Ready-to-use React, Vue.js, and Angular components
- ✅ **WebSocket Support**: Real-time streaming and alerts
- ✅ **Authentication**: JWT tokens, session management, and MFA support
- ✅ **Cross-Platform**: Works in both browser and Node.js environments
- ✅ **Rate Limiting**: Built-in adaptive rate limiting with retry logic
- ✅ **Error Handling**: Comprehensive error handling with custom error types
- ✅ **Auto-Retry**: Configurable retry mechanism with exponential backoff
- ✅ **Debugging**: Optional debug mode with detailed logging

## Installation

```bash
npm install @pynomaly/client
```

### For Node.js environments:
```bash
npm install @pynomaly/client node-fetch
```

## Quick Start

```typescript
import { PynomaliClient } from '@pynomaly/client';

const client = new PynomaliClient({
  baseUrl: 'https://api.pynomaly.com',
  apiKey: 'your-api-key',
  debug: true,
  websocket: {
    enabled: true,
    autoReconnect: true,
  },
});

// Authenticate
await client.auth.login({
  username: 'your-username',
  password: 'your-password',
});

// Detect anomalies
const result = await client.detection.detect({
  data: [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
  algorithm: 'isolation_forest',
  parameters: {
    contamination: 0.1,
    n_estimators: 100,
  },
});

console.log(`Anomaly rate: ${result.anomalyRate * 100}%`);
```

## Configuration

### Client Options

```typescript
const client = new PynomaliClient({
  baseUrl: 'https://api.pynomaly.com',     // API base URL
  apiKey: 'your-api-key',                  // API key for authentication
  timeout: 30000,                          // Request timeout in milliseconds
  maxRetries: 3,                           // Maximum retry attempts
  userAgent: 'custom-agent/1.0',           // Custom user agent
  debug: false,                            // Enable debug logging
  
  // WebSocket configuration
  websocket: {
    enabled: true,                         // Enable WebSocket support
    autoReconnect: true,                   // Auto-reconnect on disconnect
    maxRetries: 5,                         // Max reconnection attempts
    reconnectInterval: 5000,               // Reconnection interval in ms
  },
  
  // Rate limiting
  rateLimitRequests: 100,                  // Max requests per window
  rateLimitPeriod: 60000,                  // Rate limit window in ms
});
```

## API Reference

### Authentication

```typescript
// Login with credentials
const response = await client.auth.login({
  username: 'your-username',
  password: 'your-password',
  mfaCode: 'optional-mfa-code',
});

// Logout
await client.auth.logout();

// Get current user
const user = await client.auth.getCurrentUser();

// Enable MFA
const { qrCode, backupCodes } = await client.auth.enableMFA();
```

### Anomaly Detection

```typescript
// Basic detection
const result = await client.detection.detect({
  data: [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
  algorithm: 'isolation_forest',
  parameters: {
    contamination: 0.1,
    n_estimators: 100,
  },
  includeExplanations: true,
});

// Batch detection
const results = await client.detection.batchDetect([
  [1, 2, 3, 4, 5],
  [10, 20, 30, 40, 50],
], 'local_outlier_factor');
```

### Real-time Streaming

```typescript
// Connect to WebSocket
await client.connectWebSocket({
  onConnect: () => console.log('Connected'),
  onDisconnect: () => console.log('Disconnected'),
  onData: (data) => console.log('Stream data:', data),
  onAlert: (alert) => console.log('Alert:', alert),
  onError: (error) => console.error('Error:', error),
});

// Send streaming data
const streamingManager = client.getStreamingManager();
await streamingManager.sendStreamData('stream-id', [1, 2, 3, 4, 5]);

// Disconnect
client.disconnectWebSocket();
```

### Health Monitoring

```typescript
// Get system health
const health = await client.health.getHealth();
console.log(`Status: ${health.status}`);

// Get metrics
const metrics = await client.health.getMetrics();
```

## Framework Integration

### React

```jsx
import React, { useState, useEffect } from 'react';
import { PynomaliClient } from '@pynomaly/client';

export const usePynomaliClient = (config) => {
  const [client] = useState(() => new PynomaliClient(config));
  return client;
};

export const AnomalyDetectionComponent = () => {
  const client = usePynomaliClient({ baseUrl: 'https://api.pynomaly.com' });
  const [result, setResult] = useState(null);
  
  const detectAnomalies = async (data) => {
    const response = await client.detection.detect({
      data,
      algorithm: 'isolation_forest',
    });
    setResult(response);
  };
  
  return (
    <div>
      <button onClick={() => detectAnomalies([1, 2, 3, 100, 4, 5])}>
        Detect Anomalies
      </button>
      {result && (
        <div>Anomaly Rate: {(result.anomalyRate * 100).toFixed(2)}%</div>
      )}
    </div>
  );
};
```

### Vue.js

```vue
<template>
  <div>
    <button @click=\"detectAnomalies([1, 2, 3, 100, 4, 5])\">
      Detect Anomalies
    </button>
    <div v-if=\"result\">
      Anomaly Rate: {{ (result.anomalyRate * 100).toFixed(2) }}%
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { PynomaliClient } from '@pynomaly/client';

const client = new PynomaliClient({ baseUrl: 'https://api.pynomaly.com' });
const result = ref(null);

const detectAnomalies = async (data) => {
  const response = await client.detection.detect({
    data,
    algorithm: 'isolation_forest',
  });
  result.value = response;
};
</script>
```

### Angular

```typescript
import { Injectable } from '@angular/core';
import { PynomaliClient } from '@pynomaly/client';

@Injectable({ providedIn: 'root' })
export class PynomaliService {
  private client = new PynomaliClient({ baseUrl: 'https://api.pynomaly.com' });
  
  async detectAnomalies(data: number[]) {
    return await this.client.detection.detect({
      data,
      algorithm: 'isolation_forest',
    });
  }
}
```

## Error Handling

```typescript
import { 
  PynomaliError, 
  AuthenticationError, 
  ValidationError, 
  NetworkError 
} from '@pynomaly/client';

try {
  await client.detection.detect({ data: [1, 2, 3] });
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.log('Authentication failed');
  } else if (error instanceof ValidationError) {
    console.log('Validation error:', error.details);
  } else if (error instanceof NetworkError) {
    console.log('Network error:', error.message);
  } else if (error instanceof PynomaliError) {
    console.log('Pynomaly error:', error.message);
  }
}
```

## Environment Support

### Browser

The SDK works in all modern browsers that support:
- ES6+ features
- Fetch API
- WebSocket (for real-time features)
- AbortController (for request cancellation)

### Node.js

Requires Node.js 16+ and the `node-fetch` package for HTTP requests:

```bash
npm install node-fetch
```

## TypeScript Support

The SDK includes comprehensive TypeScript definitions:

```typescript
import { 
  PynomaliClient,
  DetectionRequest,
  DetectionResponse,
  ClientConfig,
  HealthStatus,
  StreamAlert,
} from '@pynomaly/client';

const client: PynomaliClient = new PynomaliClient(config);

const request: DetectionRequest = {
  data: [1, 2, 3, 4, 5],
  algorithm: 'isolation_forest',
  parameters: {
    contamination: 0.1,
  },
};

const response: DetectionResponse = await client.detection.detect(request);
```

## Advanced Features

### Custom Error Handling

```typescript
const client = new PynomaliClient({
  baseUrl: 'https://api.pynomaly.com',
  debug: true,
});

// Custom error handler
client.on('error', (error) => {
  console.error('Client error:', error);
});
```

### Request Interception

```typescript
// Custom request options
const response = await client.request('GET', '/custom-endpoint', {
  headers: { 'Custom-Header': 'value' },
  timeout: 5000,
  skipRateLimit: true,
});
```

### Rate Limiting

```typescript
const client = new PynomaliClient({
  rateLimitRequests: 50,      // 50 requests
  rateLimitPeriod: 60000,     // per minute
});

// Check rate limit status
const status = client.getClientInfo().rateLimitStatus;
console.log(`Requests remaining: ${status.remaining}`);
```

## Examples

See the `/examples` directory for complete integration examples:

- `react-example.tsx` - Complete React application
- `vue-example.vue` - Complete Vue.js application  
- `angular-example.ts` - Complete Angular application
- `node-example.js` - Node.js usage example

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

## Development

```bash
# Install dependencies
npm install

# Build the library
npm run build

# Run type checking
npm run type-check

# Run linting
npm run lint

# Generate documentation
npm run docs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- Documentation: [https://docs.pynomaly.com](https://docs.pynomaly.com)
- Issues: [GitHub Issues](https://github.com/pynomaly/pynomaly-typescript-sdk/issues)
- Email: support@pynomaly.com

## Changelog

### v1.0.0
- Initial release
- TypeScript support with comprehensive type definitions
- Modern Promise-based async/await API
- React, Vue.js, and Angular framework integration examples
- WebSocket support for real-time updates
- Authentication and session management
- Browser and Node.js compatibility
- Rate limiting with retry logic
- Comprehensive error handling
- Debug mode and logging
- Auto-retry with exponential backoff
- Comprehensive test suite