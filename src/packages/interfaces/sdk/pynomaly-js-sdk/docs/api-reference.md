# Pynomaly JavaScript SDK - API Reference

Complete API reference for the Pynomaly JavaScript SDK.

## Table of Contents

- [PynomalyClient](#pynomalyclient)
- [AuthManager](#authmanager)
- [PynomalyWebSocket](#pynomalywebsocket)
- [AsyncAPI](#asyncapi)
- [SecurityUtils](#securityutils)
- [TokenValidator](#tokenvalidator)
- [Types](#types)
- [Error Handling](#error-handling)

## PynomalyClient

The main client class for interacting with the Pynomaly API.

### Constructor

```typescript
new PynomalyClient(config: PynomalyConfig)
```

#### Parameters

- `config` - Configuration object

```typescript
interface PynomalyConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  debug?: boolean;
  version?: string;
}
```

### Methods

#### Authentication

##### `authenticate(email: string, password: string): Promise<AuthResult>`

Authenticate with email and password.

```typescript
const result = await client.authenticate('user@example.com', 'password');
```

##### `authenticateWithApiKey(apiKey: string): Promise<AuthResult>`

Authenticate with API key.

```typescript
const result = await client.authenticateWithApiKey('your-api-key');
```

##### `logout(): Promise<void>`

Logout the current user.

```typescript
await client.logout();
```

##### `isAuthenticated(): boolean`

Check if user is authenticated.

```typescript
if (client.isAuthenticated()) {
  // User is authenticated
}
```

#### Anomaly Detection

##### `detectAnomalies(params: AnomalyDetectionParams): Promise<AnomalyDetectionResult>`

Detect anomalies synchronously.

```typescript
const result = await client.detectAnomalies({
  data: [[1, 2], [3, 4], [100, 200]],
  algorithm: 'isolation_forest',
  parameters: {
    contamination: 0.1,
    n_estimators: 100
  }
});
```

##### `detectAnomaliesAsync(params: AnomalyDetectionParams): Promise<JobResponse>`

Start asynchronous anomaly detection.

```typescript
const job = await client.detectAnomaliesAsync({
  data: largeDataset,
  algorithm: 'local_outlier_factor'
});
```

#### Data Quality

##### `analyzeDataQuality(params: DataQualityParams): Promise<DataQualityResult>`

Analyze data quality synchronously.

```typescript
const result = await client.analyzeDataQuality({
  data: dataset,
  rules: ['completeness', 'validity', 'consistency'],
  customRules: [{
    name: 'email_format',
    description: 'Validate email format',
    condition: 'email matches regex'
  }]
});
```

##### `analyzeDataQualityAsync(params: DataQualityParams): Promise<JobResponse>`

Start asynchronous data quality analysis.

#### Data Profiling

##### `profileData(params: DataProfilingParams): Promise<DataProfilingResult>`

Profile dataset synchronously.

```typescript
const profile = await client.profileData({
  data: dataset,
  includeAdvanced: true
});
```

##### `profileDataAsync(params: DataProfilingParams): Promise<JobResponse>`

Start asynchronous data profiling.

#### Job Management

##### `getJobStatus(jobId: string): Promise<JobStatus>`

Get the status of a job.

```typescript
const status = await client.getJobStatus('job-123');
```

##### `getJobResult(jobId: string): Promise<any>`

Get the result of a completed job.

```typescript
const result = await client.getJobResult('job-123');
```

##### `cancelJob(jobId: string): Promise<CancelJobResponse>`

Cancel a running job.

```typescript
await client.cancelJob('job-123');
```

##### `waitForJobCompletion(jobId: string, pollInterval?: number): Promise<JobResult>`

Wait for job completion with automatic polling.

```typescript
const result = await client.waitForJobCompletion('job-123', 2000); // Poll every 2 seconds
```

#### Utility Methods

##### `healthCheck(): Promise<HealthCheckResult>`

Check API health status.

```typescript
const health = await client.healthCheck();
```

##### `updateConfig(newConfig: Partial<PynomalyConfig>): void`

Update client configuration.

```typescript
client.updateConfig({ timeout: 60000 });
```

##### `disconnect(): void`

Clean up client resources.

```typescript
client.disconnect();
```

### Events

The PynomalyClient extends EventEmitter and emits the following events:

- `authenticated` - User authenticated
- `logout` - User logged out
- `error` - Error occurred
- `requestStart` - Request started
- `requestComplete` - Request completed
- `tokenRefreshed` - Token refreshed

```typescript
client.on('authenticated', (authResult) => {
  console.log('User authenticated:', authResult.user);
});

client.on('error', (error) => {
  console.error('Client error:', error);
});
```

## AuthManager

Manages authentication state and session persistence.

### Constructor

```typescript
new AuthManager(config: AuthManagerConfig)
```

### Methods

##### `login(credentials: AuthCredentials): Promise<AuthResult>`

Login with credentials.

```typescript
const result = await authManager.login({
  email: 'user@example.com',
  password: 'password'
});
```

##### `loginWithApiKey(apiKey: string): Promise<AuthResult>`

Login with API key.

##### `logout(): Promise<void>`

Logout current user.

##### `refreshToken(): Promise<AuthToken>`

Refresh authentication token.

##### `getAuthState(): AuthState`

Get current authentication state.

```typescript
const state = authManager.getAuthState();
console.log('Is authenticated:', state.isAuthenticated);
console.log('User:', state.user);
console.log('Token expires at:', state.token?.expiresAt);
```

##### `isAuthenticated(): boolean`

Check if user is authenticated.

##### `getCurrentToken(): AuthToken | null`

Get current authentication token.

##### `clearSession(): void`

Clear stored session data.

## PynomalyWebSocket

Real-time WebSocket communication.

### Constructor

```typescript
new PynomalyWebSocket(url: string, config?: WebSocketConfig)
```

### Methods

##### `connect(): Promise<void>`

Connect to WebSocket server.

```typescript
await ws.connect();
```

##### `disconnect(): void`

Disconnect from WebSocket server.

##### `subscribe(channel: string): void`

Subscribe to a channel.

```typescript
ws.subscribe('anomaly-alerts');
```

##### `unsubscribe(channel: string): void`

Unsubscribe from a channel.

##### `send(message: any): void`

Send message to server.

```typescript
ws.send({ type: 'custom', data: 'hello' });
```

##### `getSubscriptions(): string[]`

Get list of current subscriptions.

##### `isConnected(): boolean`

Check if WebSocket is connected.

### Events

- `open` - Connection opened
- `close` - Connection closed
- `error` - Error occurred
- `message` - Message received
- `reconnecting` - Attempting to reconnect
- `reconnected` - Successfully reconnected

```typescript
ws.on('message', (message) => {
  console.log('Received:', message);
});

ws.on('error', (error) => {
  console.error('WebSocket error:', error);
});
```

## AsyncAPI

Advanced asynchronous operations and job management.

### Constructor

```typescript
new AsyncAPI(client: PynomalyClient)
```

### Methods

##### `waitForJobCompletion(jobId: string, pollInterval?: number): Promise<any>`

Wait for job completion with polling.

##### `streamProcessing(requests: Request[], bufferSize?: number): AsyncGenerator`

Process requests with streaming.

```typescript
const stream = asyncAPI.streamProcessing(requests, 50);

for await (const message of stream) {
  console.log('Stream message:', message);
}
```

##### `processParallel(requests: (() => Promise<any>)[], maxConcurrency?: number): Promise<ParallelResult[]>`

Process multiple requests in parallel.

```typescript
const results = await asyncAPI.processParallel([
  () => client.detectAnomalies(dataset1),
  () => client.detectAnomalies(dataset2)
], 2);
```

##### `withCircuitBreaker(operation: () => Promise<any>, threshold?: number, timeout?: number): Promise<any>`

Execute operation with circuit breaker pattern.

## SecurityUtils

Security utility functions.

### Static Methods

##### `validatePassword(password: string): PasswordStrengthResult`

Validate password strength.

```typescript
const result = SecurityUtils.validatePassword('myPassword123!');
console.log('Strength:', result.strength);
console.log('Score:', result.score);
console.log('Feedback:', result.feedback);
```

##### `generateSecurePassword(length?: number): string`

Generate secure password.

```typescript
const password = SecurityUtils.generateSecurePassword(16);
```

##### `hashPassword(password: string): string`

Hash password securely.

##### `sanitizeInput(input: string): string`

Sanitize user input.

##### `validateUrl(url: string): boolean`

Validate URL format.

##### `generateNonce(): string`

Generate cryptographic nonce.

## TokenValidator

Token validation utilities.

### Static Methods

##### `validate(token: AuthToken, options?: TokenValidationOptions): TokenValidationResult`

Validate authentication token.

```typescript
const result = TokenValidator.validate(token, {
  validateFormat: true,
  validateClaims: true,
  refreshThreshold: 10
});

if (!result.isValid) {
  console.log('Validation errors:', result.errors);
}
```

##### `isExpired(token: AuthToken): boolean`

Check if token is expired.

##### `needsRefresh(token: AuthToken, thresholdMinutes?: number): boolean`

Check if token needs refresh.

##### `extractClaims(token: string): any`

Extract claims from JWT token.

## Types

### Core Types

#### PynomalyConfig

```typescript
interface PynomalyConfig {
  apiKey: string;
  baseUrl?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  debug?: boolean;
  version?: string;
}
```

#### AuthToken

```typescript
interface AuthToken {
  token: string;
  refreshToken?: string;
  expiresAt: Date;
  tokenType: string;
}
```

#### User

```typescript
interface User {
  id: string;
  email: string;
  name?: string;
  roles: string[];
  permissions: string[];
  createdAt?: Date;
  lastLogin?: Date;
}
```

### Anomaly Detection Types

#### AnomalyDetectionParams

```typescript
interface AnomalyDetectionParams {
  data: number[][];
  algorithm: 'isolation_forest' | 'local_outlier_factor' | 'one_class_svm' | 'auto';
  parameters?: {
    contamination?: number;
    n_estimators?: number;
    max_samples?: number;
    n_neighbors?: number;
    [key: string]: any;
  };
}
```

#### AnomalyDetectionResult

```typescript
interface AnomalyDetectionResult {
  id: string;
  anomalies: Anomaly[];
  algorithm: string;
  parameters: Record<string, any>;
  metrics: AnomalyMetrics;
  createdAt: Date;
  processingTime: number;
}
```

#### Anomaly

```typescript
interface Anomaly {
  index: number;
  score: number;
  isAnomaly: boolean;
  confidence: number;
  explanation: string;
  data: number[];
}
```

### Data Quality Types

#### DataQualityParams

```typescript
interface DataQualityParams {
  data: any[];
  rules: string[];
  customRules?: CustomRule[];
  options?: DataQualityOptions;
}
```

#### DataQualityResult

```typescript
interface DataQualityResult {
  id: string;
  overallScore: number;
  dimensionScores: Record<string, number>;
  ruleResults: RuleResult[];
  issues: DataQualityIssue[];
  recommendations: string[];
  createdAt: Date;
}
```

### Job Types

#### JobStatus

```typescript
interface JobStatus {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message?: string;
  result?: any;
  error?: string;
  createdAt: Date;
  updatedAt: Date;
}
```

### WebSocket Types

#### WebSocketMessage

```typescript
interface WebSocketMessage {
  type: string;
  channel?: string;
  data?: any;
  timestamp: Date;
}
```

#### WebSocketConfig

```typescript
interface WebSocketConfig {
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
  debug?: boolean;
}
```

## Error Handling

### Error Types

The SDK throws specific error types for different scenarios:

#### PynomalyError

Base error class for all SDK errors.

```typescript
interface PynomalyError extends Error {
  code: string;
  details?: any;
  response?: any;
}
```

#### Common Error Codes

- `AUTH_TOKEN_EXPIRED` - Authentication token expired
- `AUTH_INVALID_CREDENTIALS` - Invalid login credentials
- `AUTH_REQUIRED` - Authentication required
- `RATE_LIMIT_EXCEEDED` - API rate limit exceeded
- `NETWORK_ERROR` - Network connection error
- `TIMEOUT_ERROR` - Request timeout
- `VALIDATION_ERROR` - Input validation error
- `API_ERROR` - General API error
- `WEBSOCKET_ERROR` - WebSocket connection error

### Error Handling Patterns

```typescript
try {
  const result = await client.detectAnomalies(params);
} catch (error) {
  switch (error.code) {
    case 'AUTH_TOKEN_EXPIRED':
      // Handle token expiration
      await client.refreshToken();
      break;
    
    case 'RATE_LIMIT_EXCEEDED':
      // Handle rate limiting
      const retryAfter = error.details?.retryAfter || 60;
      await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
      break;
    
    case 'VALIDATION_ERROR':
      // Handle validation errors
      console.error('Validation errors:', error.details.errors);
      break;
    
    default:
      console.error('Unexpected error:', error.message);
  }
}
```

## Framework Integration

The SDK provides framework-specific integrations:

### React Hooks

- `usePynomalyClient()` - Client management
- `usePynomalyAuth()` - Authentication state
- `useAnomalyDetection()` - Anomaly detection operations
- `useWebSocket()` - WebSocket connection

### Vue Composables

- `usePynomalyClient()` - Client management
- `usePynomalyAuth()` - Authentication state
- `useAnomalyDetection()` - Anomaly detection operations

### Angular Services

- `PynomalyService` - Main service
- `PynomalyAuthService` - Authentication service

See the [Framework Integration Guide](./framework-integration.md) for detailed usage examples.