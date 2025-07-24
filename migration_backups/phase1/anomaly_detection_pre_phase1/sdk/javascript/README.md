# Anomaly Detection JavaScript/TypeScript SDK

A comprehensive SDK for integrating with the Anomaly Detection service in both browser and Node.js environments.

## Installation

```bash
npm install @anomaly-detection/sdk
```

## Quick Start

### Basic Usage

```typescript
import { AnomalyDetectionClient, AlgorithmType } from '@anomaly-detection/sdk';

// Initialize client
const client = new AnomalyDetectionClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key', // optional
  timeout: 30000,
});

// Detect anomalies
const data = [[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]]; // Last point is anomalous
const result = await client.detectAnomalies(data, AlgorithmType.ISOLATION_FOREST);

console.log(`Found ${result.anomalyCount} anomalies`);
result.anomalies.forEach(anomaly => {
  console.log(`Anomaly at index ${anomaly.index} with score ${anomaly.score}`);
});
```

### Streaming Detection

```typescript
import { StreamingClient, AlgorithmType } from '@anomaly-detection/sdk';

const streamingClient = new StreamingClient({
  wsUrl: 'ws://localhost:8000/ws/stream',
  algorithm: AlgorithmType.ISOLATION_FOREST,
  batchSize: 10,
  detectionThreshold: 0.5,
});

// Set up event handlers
streamingClient.on('connect', () => {
  console.log('Connected to streaming service');
});

streamingClient.on('anomaly', (anomaly) => {
  console.log('Anomaly detected:', anomaly);
});

streamingClient.on('error', (error) => {
  console.error('Streaming error:', error);
});

// Start streaming
await streamingClient.start();

// Send data points
streamingClient.sendData([2.5, 3.1]);
streamingClient.sendData([2.6, 3.2]);
```

### Model Management

```typescript
// Train a new model
const trainingResult = await client.trainModel({
  data: trainingData,
  algorithm: AlgorithmType.ISOLATION_FOREST,
  modelName: 'my-custom-model',
  validationSplit: 0.2,
});

console.log(`Model trained: ${trainingResult.modelId}`);

// List all models
const models = await client.listModels();
models.forEach(model => {
  console.log(`${model.modelId}: ${model.algorithm} (${model.status})`);
});

// Use specific model for detection
const modelResult = await client.detectAnomalies(
  testData,
  AlgorithmType.ISOLATION_FOREST,
  { model_id: trainingResult.modelId }
);
```

### Utility Functions

```typescript
import { 
  validateDataFormat, 
  normalizeData, 
  generateSampleData,
  parseCSV,
  calculateDataStatistics 
} from '@anomaly-detection/sdk';

// Validate data format
try {
  validateDataFormat(myData);
  console.log('Data format is valid');
} catch (error) {
  console.error('Invalid data format:', error.message);
}

// Normalize data
const { normalizedData, means, stds } = normalizeData(rawData);

// Generate sample data for testing
const { data, labels } = generateSampleData(1000, 5, 0.1);

// Parse CSV data
const csvData = parseCSV(csvString, true); // true = has header

// Calculate statistics
const stats = calculateDataStatistics(data);
console.log(`Dataset: ${stats.numSamples} samples, ${stats.numFeatures} features`);
```

## Browser Usage

The SDK works seamlessly in browser environments:

```html
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import { AnomalyDetectionClient, AlgorithmType } from 'https://unpkg.com/@anomaly-detection/sdk/dist/index.esm.js';
        
        const client = new AnomalyDetectionClient({
            baseUrl: 'http://localhost:8000'
        });
        
        const data = [[1, 2], [1.1, 2.1], [10, 20]];
        client.detectAnomalies(data, AlgorithmType.ISOLATION_FOREST)
            .then(result => {
                console.log('Anomalies found:', result.anomalies);
            });
    </script>
</head>
<body>
    <h1>Anomaly Detection in Browser</h1>
</body>
</html>
```

## API Reference

### AnomalyDetectionClient

Main client for interacting with the anomaly detection service.

#### Constructor

```typescript
new AnomalyDetectionClient(config: ClientConfig)
```

#### Methods

- `detectAnomalies(data, algorithm?, parameters?, returnExplanations?)` - Detect anomalies
- `batchDetect(request)` - Process batch detection request
- `trainModel(request)` - Train a new model
- `getModel(modelId)` - Get model information
- `listModels()` - List all models
- `deleteModel(modelId)` - Delete a model
- `explainAnomaly(dataPoint, options?)` - Get anomaly explanation
- `getHealth()` - Get service health
- `getMetrics()` - Get service metrics
- `uploadData(data, datasetName, description?)` - Upload training data

### StreamingClient

WebSocket client for real-time anomaly detection.

#### Constructor

```typescript
new StreamingClient(config: StreamingClientConfig)
```

#### Methods

- `start()` - Start streaming client
- `stop()` - Stop streaming client
- `sendData(dataPoint)` - Send single data point
- `sendBatch(batch)` - Send multiple data points

#### Events

- `connect` - Connected to streaming service
- `disconnect` - Disconnected from streaming service
- `anomaly` - Anomaly detected
- `error` - Error occurred
- `message` - Raw message received

## Types

All TypeScript types are exported from the main module:

```typescript
import type { 
  DetectionResult,
  AnomalyData,
  ModelInfo,
  AlgorithmType,
  ClientConfig,
  StreamingClientConfig
} from '@anomaly-detection/sdk';
```

## Error Handling

The SDK provides detailed error types:

```typescript
import { APIError, ValidationError, ConnectionError } from '@anomaly-detection/sdk';

try {
  const result = await client.detectAnomalies(invalidData);
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Data validation failed:', error.message);
  } else if (error instanceof APIError) {
    console.error(`API error ${error.statusCode}:`, error.message);
  } else if (error instanceof ConnectionError) {
    console.error('Connection failed:', error.message);
  }
}
```

## Development

```bash
# Install dependencies
npm install

# Build the SDK
npm run build

# Run tests
npm test

# Watch mode for development
npm run dev
```

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/anomaly-detection/javascript-sdk/issues)
- Documentation: [Full API documentation](../../docs/api.md)
- Examples: [Usage examples](../../examples/)