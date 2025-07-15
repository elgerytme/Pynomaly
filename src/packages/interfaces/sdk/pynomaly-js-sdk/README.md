# Pynomaly JavaScript SDK

A comprehensive JavaScript/TypeScript SDK for integrating with the Pynomaly data science platform. This SDK provides modern JavaScript support with Promise-based APIs, WebSocket real-time updates, and comprehensive authentication management.

## Features

- 🔧 **Modern JavaScript Support**: Full TypeScript support with comprehensive type definitions
- 🔄 **Promise-based API**: Async/await support for all operations
- 🌐 **WebSocket Integration**: Real-time updates and job monitoring
- 🔐 **Authentication Management**: Complete session management with token refresh
- 🔒 **Security Features**: Built-in security utilities and token validation
- 📱 **Universal Compatibility**: Works in both browser and Node.js environments
- 🎯 **Event-driven Architecture**: EventEmitter-based for reactive programming
- 🚀 **Performance Optimized**: Built-in retry mechanisms and circuit breakers

## Installation

```bash
npm install pynomaly-js-sdk
```

## Quick Start

```javascript
import { PynomalyClient, AuthManager } from 'pynomaly-js-sdk';

// Initialize client
const client = new PynomalyClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.pynomaly.com'
});

// Initialize authentication manager
const authManager = new AuthManager({
  enablePersistence: true,
  autoRefresh: true
});

// Authenticate
const user = await authManager.login(
  { email: 'user@example.com', password: 'password' },
  client
);

// Detect anomalies
const result = await client.detectAnomalies({
  data: [[1, 2, 3], [4, 5, 6], [100, 200, 300]], // anomaly: [100, 200, 300]
  algorithm: 'isolation_forest'
});

console.log('Anomalies detected:', result.anomalies);
```

## Core Features

### 1. Anomaly Detection

```javascript
// Synchronous anomaly detection
const result = await client.detectAnomalies({
  data: [[1, 2], [2, 3], [3, 4], [100, 200]],
  algorithm: 'isolation_forest',
  parameters: { contamination: 0.1 }
});

// Asynchronous anomaly detection with job monitoring
const asyncAPI = new AsyncAPI(client);
const result = await asyncAPI.detectAnomalies({
  data: largeDataset,
  algorithm: 'auto'
});
```

### 2. Data Quality Analysis

```javascript
// Analyze data quality
const qualityResult = await client.analyzeDataQuality({
  data: [
    { name: 'John', age: 30, email: 'john@example.com' },
    { name: '', age: -5, email: 'invalid-email' }
  ],
  rules: [
    {
      id: 'name-not-empty',
      name: 'Name Required',
      type: 'completeness',
      column: 'name',
      condition: 'not_empty',
      threshold: 0.95,
      severity: 'high'
    }
  ]
});
```

### 3. Data Profiling

```javascript
// Profile dataset
const profile = await client.profileData({
  data: dataset,
  includeAdvanced: true,
  generateVisualization: true
});

console.log('Dataset statistics:', profile.datasetStats);
console.log('Column profiles:', profile.columnProfiles);
```

### 4. Real-time Updates with WebSocket

```javascript
import { PynomalyWebSocket } from 'pynomaly-js-sdk';

const ws = new PynomalyWebSocket({
  url: 'wss://api.pynomaly.com/ws',
  reconnectInterval: 5000,
  maxReconnectAttempts: 10
});

// Connect and authenticate
await ws.connect(authToken);

// Subscribe to job updates
ws.subscribeToJob(jobId);

// Listen for job completion
ws.on('job:completed', (result) => {
  console.log('Job completed:', result);
});

// Listen for job status updates
ws.on('job:status', (status) => {
  console.log('Job status:', status);
});
```

### 5. Authentication Management

```javascript
import { AuthManager, SecurityUtils } from 'pynomaly-js-sdk';

// Create auth manager with persistent sessions
const authManager = new AuthManager({
  enablePersistence: true,
  autoRefresh: true,
  refreshThreshold: 5 // refresh 5 minutes before expiry
});

// Login with credentials
const user = await authManager.login(
  { email: 'user@example.com', password: 'password' },
  client
);

// Login with API key
const user = await authManager.loginWithApiKey('your-api-key', client);

// Check authentication status
if (authManager.isAuthenticated()) {
  const authState = authManager.getAuthState();
  console.log('User:', authState.user);
  console.log('Token expires at:', authState.expiresAt);
}

// Listen for auth events
authManager.on('auth:login', ({ user, token }) => {
  console.log('User logged in:', user);
});

authManager.on('auth:expired', () => {
  console.log('Session expired, please login again');
});
```

### 6. Security Features

```javascript
import { SecurityUtils, TokenValidator } from 'pynomaly-js-sdk';

// Password strength validation
const passwordResult = SecurityUtils.validatePassword('myPassword123!');
console.log('Password strength:', passwordResult.strength);
console.log('Feedback:', passwordResult.feedback);

// Token validation
const tokenValidation = TokenValidator.validate(authToken, {
  validateFormat: true,
  validateClaims: true,
  refreshThreshold: 10
});

if (!tokenValidation.isValid) {
  console.log('Token validation errors:', tokenValidation.errors);
}

// Generate secure password
const securePassword = SecurityUtils.generateSecurePassword(16);
```

### 7. Advanced Features

#### Streaming Processing

```javascript
const asyncAPI = new AsyncAPI(client);

// Process large datasets with streaming
const stream = asyncAPI.streamProcessing(
  largeDatasetRequests,
  'anomaly',
  { bufferSize: 50, autoFlush: true }
);

for await (const message of stream) {
  switch (message.type) {
    case 'data':
      console.log('Result:', message.payload.result);
      break;
    case 'error':
      console.error('Error:', message.payload.error);
      break;
    case 'control':
      console.log('Control message:', message.payload.action);
      break;
  }
}
```

#### Parallel Processing

```javascript
// Process multiple requests in parallel with concurrency control
const results = await asyncAPI.processParallel(
  [
    () => client.detectAnomalies(dataset1),
    () => client.detectAnomalies(dataset2),
    () => client.detectAnomalies(dataset3)
  ],
  3 // max 3 concurrent requests
);

results.forEach((result, index) => {
  if (result.success) {
    console.log(`Dataset ${index + 1} result:`, result.result);
  } else {
    console.error(`Dataset ${index + 1} error:`, result.error);
  }
});
```

#### Circuit Breaker Pattern

```javascript
// Use circuit breaker for resilient API calls
const result = await asyncAPI.withCircuitBreaker(
  () => client.detectAnomalies(dataset),
  5, // failure threshold
  60000 // recovery timeout (1 minute)
);
```

## Configuration

### Client Configuration

```javascript
const client = new PynomalyClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.pynomaly.com',
  timeout: 30000,
  retryAttempts: 3,
  enableWebSocket: true,
  debug: false,
  version: 'v1'
});
```

### Authentication Configuration

```javascript
const authManager = new AuthManager({
  storageKey: 'pynomaly_auth',
  enablePersistence: true,
  autoRefresh: true,
  refreshThreshold: 5 // minutes
});
```

### WebSocket Configuration

```javascript
const ws = new PynomalyWebSocket({
  url: 'wss://api.pynomaly.com/ws',
  protocols: ['pynomaly-v1'],
  reconnectInterval: 5000,
  maxReconnectAttempts: 10,
  heartbeatInterval: 30000,
  messageQueueSize: 100,
  debug: false
});
```

### Security Configuration

```javascript
SecurityUtils.setSecurityPolicy({
  passwordMinLength: 12,
  passwordRequireUppercase: true,
  passwordRequireLowercase: true,
  passwordRequireNumbers: true,
  passwordRequireSpecialChars: true,
  sessionTimeout: 120, // 2 hours
  maxLoginAttempts: 3,
  lockoutDuration: 30, // 30 minutes
  enforceHttps: true,
  enableCsrfProtection: true
});
```

## Error Handling

```javascript
try {
  const result = await client.detectAnomalies(data);
} catch (error) {
  if (error.code === 'AUTH_TOKEN_EXPIRED') {
    // Handle token expiration
    await authManager.refreshToken(client);
    // Retry the operation
  } else if (error.code === 'RATE_LIMIT_EXCEEDED') {
    // Handle rate limiting
    const retryAfter = error.details?.retryAfter || 60;
    await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
  } else {
    console.error('API Error:', error.message);
  }
}
```

## Event Handling

```javascript
// Client events
client.on('connection:error', (error) => {
  console.error('Connection error:', error);
});

// Auth events
authManager.on('auth:login', ({ user, token }) => {
  console.log('User logged in:', user.email);
});

authManager.on('auth:logout', () => {
  console.log('User logged out');
});

authManager.on('auth:refresh', (newToken) => {
  console.log('Token refreshed, expires at:', newToken.expiresAt);
});

// WebSocket events
ws.on('connection:open', () => {
  console.log('WebSocket connected');
});

ws.on('connection:close', () => {
  console.log('WebSocket disconnected');
});

ws.on('job:status', (status) => {
  console.log('Job status update:', status);
});
```

## TypeScript Support

The SDK is written in TypeScript and includes comprehensive type definitions:

```typescript
import { 
  PynomalyClient,
  AnomalyDetectionRequest,
  AnomalyDetectionResult,
  AuthToken,
  User
} from 'pynomaly-js-sdk';

const client = new PynomalyClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.pynomaly.com'
});

const request: AnomalyDetectionRequest = {
  data: [[1, 2], [3, 4], [100, 200]],
  algorithm: 'isolation_forest',
  parameters: { contamination: 0.1 }
};

const result: AnomalyDetectionResult = await client.detectAnomalies(request);
```

## Browser vs Node.js

The SDK works in both browser and Node.js environments:

### Browser Usage

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://unpkg.com/pynomaly-js-sdk/dist/pynomaly-js-sdk.umd.js"></script>
</head>
<body>
  <script>
    const client = new PynomalySDK.PynomalyClient({
      apiKey: 'your-api-key',
      baseUrl: 'https://api.pynomaly.com'
    });
  </script>
</body>
</html>
```

### Node.js Usage

```javascript
const { PynomalyClient } = require('pynomaly-js-sdk');
// or
import { PynomalyClient } from 'pynomaly-js-sdk';

const client = new PynomalyClient({
  apiKey: process.env.PYNOMALY_API_KEY,
  baseUrl: 'https://api.pynomaly.com'
});
```

## Examples

See the `/examples` directory for complete examples:

- [Basic Anomaly Detection](./examples/basic-anomaly-detection.js)
- [Data Quality Analysis](./examples/data-quality-analysis.js)
- [Real-time WebSocket Updates](./examples/websocket-updates.js)
- [Authentication Management](./examples/authentication.js)
- [Advanced Features](./examples/advanced-features.js)

## API Reference

### PynomalyClient

#### Methods

- `authenticate(credentials)` - Authenticate with email/password
- `authenticateWithApiKey(apiKey)` - Authenticate with API key
- `detectAnomalies(request)` - Detect anomalies synchronously
- `detectAnomaliesAsync(request)` - Detect anomalies asynchronously
- `analyzeDataQuality(request)` - Analyze data quality
- `analyzeDataQualityAsync(request)` - Analyze data quality asynchronously
- `profileData(request)` - Profile dataset
- `profileDataAsync(request)` - Profile dataset asynchronously
- `getJobStatus(jobId)` - Get job status
- `getJobResult(jobId)` - Get job result
- `cancelJob(jobId)` - Cancel job
- `healthCheck()` - Check API health

### AuthManager

#### Methods

- `login(credentials, client)` - Login with credentials
- `loginWithApiKey(apiKey, client)` - Login with API key
- `logout(client)` - Logout
- `refreshToken(client)` - Refresh authentication token
- `isAuthenticated()` - Check authentication status
- `getAuthState()` - Get current authentication state
- `getUser()` - Get current user
- `getToken()` - Get current token

### PynomalyWebSocket

#### Methods

- `connect(authToken)` - Connect to WebSocket
- `disconnect()` - Disconnect from WebSocket
- `subscribeToJob(jobId)` - Subscribe to job updates
- `unsubscribeFromJob(jobId)` - Unsubscribe from job updates
- `subscribeToNotifications(userId)` - Subscribe to notifications
- `send(data)` - Send message

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please contact:
- Email: support@pynomaly.com
- GitHub Issues: [https://github.com/your-org/pynomaly-js-sdk/issues](https://github.com/your-org/pynomaly-js-sdk/issues)
- Documentation: [https://docs.pynomaly.com](https://docs.pynomaly.com)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for details about changes in each version.