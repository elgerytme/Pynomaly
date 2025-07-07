# Pynomaly JavaScript/TypeScript SDK

Official JavaScript/TypeScript SDK for the Pynomaly anomaly detection platform.

## Features

- **Comprehensive API Coverage**: Full access to all Pynomaly platform features
- **TypeScript Support**: Complete type definitions for better development experience
- **Real-time Streaming**: WebSocket support for real-time anomaly detection
- **Automatic Retry Logic**: Built-in retry mechanism with exponential backoff
- **Multi-tenant Support**: Full support for multi-tenant architectures
- **Authentication**: JWT and API key authentication
- **Error Handling**: Comprehensive error types and handling
- **Browser & Node.js**: Works in both browser and Node.js environments

## Installation

```bash
npm install @pynomaly/client
```

## Quick Start

```typescript
import PynomalyClient from '@pynomaly/client';

// Initialize client
const client = new PynomalyClient({
  baseUrl: 'https://api.pynomaly.ai',
  apiKey: 'your-api-key',
  tenantId: 'your-tenant-id'
});

// Create a detector
const detector = await client.detection.createDetector({
  name: 'My Detector',
  algorithm: 'isolation_forest',
  parameters: {
    contamination: 0.1,
    n_estimators: 100
  }
});

// Upload and create dataset
const fileId = await client.uploadFile(file, 'dataset.csv');
const dataset = await client.detection.createDataset(fileId, 'My Dataset');

// Train detector
const trainingJob = await client.detection.trainDetector(detector.id, {
  dataset_id: dataset.id,
  validation_split: 0.2
});

// Detect anomalies
const results = await client.detection.detectAnomalies({
  dataset_id: dataset.id,
  detector_id: detector.id
});

console.log(`Found ${results.anomaly_count} anomalies`);
```

## API Reference

### Detection Client

```typescript
// List detectors
const detectors = await client.detection.listDetectors();

// Get detector
const detector = await client.detection.getDetector('detector-id');

// Create detector
const detector = await client.detection.createDetector({
  name: 'My Detector',
  algorithm: 'isolation_forest',
  parameters: { contamination: 0.1 }
});

// Train detector
const job = await client.detection.trainDetector('detector-id', {
  dataset_id: 'dataset-id',
  validation_split: 0.2
});

// Detect anomalies
const results = await client.detection.detectAnomalies({
  dataset_id: 'dataset-id',
  detector_id: 'detector-id'
});
```

### Streaming Client

```typescript
// Create stream processor
const processor = await client.streaming.createProcessor({
  processor_id: 'my-processor',
  detector_id: 'detector-id',
  window_config: {
    type: 'tumbling',
    size_seconds: 60
  }
});

// Start processor
await client.streaming.startProcessor('my-processor');

// Send data
await client.streaming.sendData('my-processor', [
  {
    id: 'record-1',
    timestamp: new Date().toISOString(),
    data: { value: 42 },
    tenant_id: 'tenant-id',
    metadata: {}
  }
]);

// Get metrics
const metrics = await client.streaming.getProcessorMetrics('my-processor');
```

### A/B Testing Client

```typescript
// Create A/B test
const test = await client.abTesting.createTest({
  name: 'Algorithm Comparison',
  description: 'Compare two detection algorithms',
  variants: [
    {
      name: 'Control',
      description: 'Isolation Forest',
      detector_id: 'detector-1',
      traffic_percentage: 50,
      is_control: true
    },
    {
      name: 'Treatment',
      description: 'Local Outlier Factor',
      detector_id: 'detector-2',
      traffic_percentage: 50,
      is_control: false
    }
  ]
});

// Start test
await client.abTesting.startTest(test.id);

// Get results
const results = await client.abTesting.getTestResults(test.id);
const analysis = await client.abTesting.getStatisticalAnalysis(test.id);
```

### User Management Client

```typescript
// Create user
const user = await client.users.createUser({
  email: 'user@example.com',
  first_name: 'John',
  last_name: 'Doe',
  password: 'secure-password',
  roles: ['data_scientist'],
  tenant_id: 'tenant-id'
});

// Get current user
const currentUser = await client.users.getCurrentUser();

// Update user roles
await client.users.addUserRole(user.id, 'analyst');
```

### Compliance Client

```typescript
// List audit events
const events = await client.compliance.listAuditEvents({
  severity: 'high',
  start_date: '2023-01-01',
  end_date: '2023-12-31'
});

// Create GDPR request
const gdprRequest = await client.compliance.createGDPRRequest({
  request_type: 'data_access',
  data_subject_id: 'user-id',
  data_subject_email: 'user@example.com',
  request_details: 'Request for data access'
});

// Get compliance assessment
const assessment = await client.compliance.getComplianceAssessment(['gdpr', 'hipaa']);
```

## Real-time Features

### WebSocket Connection

```typescript
import { WebSocketClient } from '@pynomaly/client';

const wsClient = new WebSocketClient({
  url: 'wss://api.pynomaly.ai/ws',
  apiKey: 'your-api-key',
  tenantId: 'your-tenant-id'
});

// Connect
await wsClient.connect();

// Listen for anomaly alerts
wsClient.on('anomaly_alert', (alert) => {
  console.log('Anomaly detected:', alert);
});

// Subscribe to alerts
wsClient.subscribeToAnomalyAlerts('processor-id');
```

## Error Handling

```typescript
import { 
  PynomalyError, 
  AuthenticationError, 
  ValidationError, 
  NetworkError 
} from '@pynomaly/client';

try {
  await client.detection.createDetector(invalidData);
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.log('Authentication failed');
  } else if (error instanceof ValidationError) {
    console.log('Validation error:', error.details);
  } else if (error instanceof NetworkError) {
    console.log('Network error, retrying...');
  } else if (error instanceof PynomalyError) {
    console.log('Pynomaly error:', error.code, error.message);
  }
}
```

## Configuration

```typescript
const client = new PynomalyClient({
  baseUrl: 'https://api.pynomaly.ai',
  apiKey: 'your-api-key',
  tenantId: 'your-tenant-id', // Optional for multi-tenant
  timeout: 30000, // Request timeout in ms
  retryAttempts: 3, // Number of retry attempts
  retryDelay: 1000 // Initial retry delay in ms
});

// Update configuration
client.updateConfig({
  timeout: 60000,
  retryAttempts: 5
});

// Update authentication
client.setAuth({
  apiKey: 'new-api-key',
  tenantId: 'new-tenant-id'
});
```

## Browser Support

The SDK works in modern browsers and Node.js environments:

- **Browsers**: Chrome 60+, Firefox 55+, Safari 12+, Edge 79+
- **Node.js**: 16.0+
- **TypeScript**: 4.0+

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Lint
npm run lint

# Type check
npm run typecheck
```

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: https://docs.pynomaly.ai
- GitHub Issues: https://github.com/pynomaly/pynomaly/issues
- Email: support@pynomaly.ai
