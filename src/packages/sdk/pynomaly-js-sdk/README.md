# Pynomaly JavaScript SDK

[![npm version](https://badge.fury.io/js/%40pynomaly%2Fjs-sdk.svg)](https://badge.fury.io/js/%40pynomaly%2Fjs-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive JavaScript/TypeScript SDK for the Pynomaly anomaly detection platform. Provides easy integration with web applications, React components, and Node.js backends.

## Features

- 🚀 **Modern JavaScript/TypeScript** - Full TypeScript support with comprehensive type definitions
- ⚛️ **React Integration** - Pre-built React hooks and components
- 🔒 **Authentication** - Built-in API key authentication and session management
- 📊 **Data Science APIs** - Complete anomaly detection, training, and experiment management
- 🎯 **Type Safety** - Full TypeScript support with detailed type definitions
- 🔄 **Async/Await** - Modern async patterns with Promise-based APIs
- 📱 **Web & Node.js** - Works in browsers and Node.js environments
- 🛠️ **Utilities** - Data conversion, validation, and analysis utilities

## Installation

```bash
npm install @pynomaly/js-sdk
```

Or with yarn:

```bash
yarn add @pynomaly/js-sdk
```

## Quick Start

### Basic Usage

```typescript
import { PynomalyClient } from '@pynomaly/js-sdk';

// Create client
const client = new PynomalyClient({
  baseUrl: 'https://api.pynomaly.com',
  apiKey: 'your-api-key'
});

// Initialize connection
await client.initialize();

// List detectors
const detectors = await client.dataScience.listDetectors();

// Detect anomalies
const result = await client.dataScience.detectAnomalies('detector-id', {
  name: 'my-data',
  data: [[1, 2, 3], [4, 5, 6], [100, 200, 300]] // Anomaly in last row
});

console.log(`Found ${result.nAnomalies} anomalies`);
```

### React Integration

```tsx
import React from 'react';
import { PynomalyClient, useDetection, DetectionResults } from '@pynomaly/js-sdk';

const client = new PynomalyClient({
  baseUrl: 'https://api.pynomaly.com',
  apiKey: 'your-api-key'
});

function AnomalyDetectionApp() {
  const { result, isDetecting, detectAnomalies, error } = useDetection(client);

  const handleDetect = async () => {
    const dataset = {
      name: 'sample-data',
      data: [[1, 2], [2, 3], [100, 200]] // Last row is anomaly
    };
    
    await detectAnomalies('detector-id', dataset);
  };

  return (
    <div>
      <button onClick={handleDetect} disabled={isDetecting}>
        {isDetecting ? 'Detecting...' : 'Detect Anomalies'}
      </button>
      
      {error && <div className="error">{error.message}</div>}
      
      {result && (
        <DetectionResults 
          result={result} 
          showVisualization={true}
          showMetrics={true}
        />
      )}
    </div>
  );
}
```

## API Reference

### Client

#### PynomalyClient

Main client for interacting with Pynomaly APIs.

```typescript
const client = new PynomalyClient({
  baseUrl: 'https://api.pynomaly.com',
  apiKey: 'your-api-key',
  timeout: 30000,
  maxRetries: 3,
  debug: false
});
```

**Methods:**
- `initialize()` - Initialize and verify connection
- `healthCheck()` - Check API health status
- `setApiKey(apiKey)` - Update API key
- `dispose()` - Clean up resources

### Data Science API

#### Detector Management

```typescript
// Create detector
const detector = await client.dataScience.createDetector('my-detector', {
  algorithmName: 'IsolationForest',
  contaminationRate: 0.1,
  hyperparameters: { n_estimators: 100 }
});

// List detectors
const { items, total } = await client.dataScience.listDetectors({
  page: 1,
  pageSize: 20,
  algorithmName: 'IsolationForest'
});

// Update detector
await client.dataScience.updateDetector('detector-id', {
  name: 'updated-name',
  config: { algorithmName: 'LOF' }
});

// Delete detector
await client.dataScience.deleteDetector('detector-id');
```

#### Anomaly Detection

```typescript
// Detect anomalies
const result = await client.dataScience.detectAnomalies('detector-id', {
  name: 'my-dataset',
  data: [[1, 2, 3], [4, 5, 6], [100, 200, 300]],
  featureNames: ['feature1', 'feature2', 'feature3']
});

// Batch detection
const batchJob = await client.dataScience.batchDetect('detector-id', [
  dataset1, dataset2, dataset3
], 'batch-job-name');
```

#### Training

```typescript
// Train detector
const trainingJob = await client.dataScience.trainDetector('detector-id', {
  name: 'training-data',
  data: trainingData
});

// Monitor training
const job = await client.dataScience.getTrainingJob(trainingJob.jobId);
console.log(`Status: ${job.status}`);
```

### React Hooks

#### useDetection

```tsx
const { result, isDetecting, detectAnomalies, error } = useDetection(client);

// Detect anomalies
await detectAnomalies('detector-id', dataset, {
  returnScores: true,
  threshold: 0.5
});
```

#### useDetector

```tsx
const { 
  detector, 
  isLoading, 
  createDetector, 
  updateDetector, 
  deleteDetector 
} = useDetector(client);

// Create new detector
await createDetector('my-detector', {
  algorithmName: 'IsolationForest',
  contaminationRate: 0.1
});
```

#### useTraining

```tsx
const { job, isTraining, startTraining } = useTraining(client);

// Start training
await startTraining('detector-id', dataset, 'training-job');
```

### Components

#### DetectorList

```tsx
<DetectorList
  client={client}
  onDetectorSelect={(detector) => console.log(detector)}
  filters={{ algorithmName: 'IsolationForest' }}
  pageSize={20}
/>
```

#### DetectionResults

```tsx
<DetectionResults
  result={detectionResult}
  dataset={dataset}
  showVisualization={true}
  showMetrics={true}
/>
```

### Utilities

#### DatasetConverter

```typescript
import { DatasetConverter } from '@pynomaly/js-sdk';

// From CSV
const dataset = DatasetConverter.fromCSV('my-data', csvString, {
  hasHeader: true,
  delimiter: ','
});

// From JSON
const dataset = DatasetConverter.fromJSON('my-data', jsonArray);

// From File
const dataset = await DatasetConverter.fromFile(file);
```

#### DataValidator

```typescript
import { DataValidator } from '@pynomaly/js-sdk';

// Validate dataset
const { isValid, errors } = DataValidator.validate(dataset);

// Get statistics
const stats = DataValidator.getStatistics(dataset);
```

#### ResultAnalyzer

```typescript
import { ResultAnalyzer } from '@pynomaly/js-sdk';

// Analyze results
const analysis = ResultAnalyzer.analyze(detectionResult);

// Get top anomalies
const topAnomalies = ResultAnalyzer.getAnomaliesByScore(detectionResult, true);

// Compare results
const comparison = ResultAnalyzer.compare(result1, result2);
```

## Data Formats

### Dataset Format

```typescript
interface Dataset {
  name: string;
  data: any[][] | Record<string, any>[];
  metadata?: Record<string, any>;
  featureNames?: string[];
  targetColumn?: string;
}
```

### Detection Result Format

```typescript
interface DetectionResult {
  anomalyScores: number[];
  anomalyLabels: number[];
  nAnomalies: number;
  nSamples: number;
  contaminationRate: number;
  threshold: number;
  executionTime: number;
  metadata?: Record<string, any>;
}
```

## Error Handling

The SDK provides comprehensive error handling with specific error types:

```typescript
import { isPynomalyError, getErrorMessage } from '@pynomaly/js-sdk';

try {
  await client.dataScience.detectAnomalies('detector-id', dataset);
} catch (error) {
  if (isPynomalyError(error)) {
    console.error('Pynomaly Error:', error.getUserMessage());
    console.error('Details:', error.details);
  } else {
    console.error('Unexpected Error:', getErrorMessage(error));
  }
}
```

## Configuration

### Environment Variables

```bash
PYNOMALY_API_URL=https://api.pynomaly.com
PYNOMALY_API_KEY=your-api-key
```

### TypeScript Configuration

Add to your `tsconfig.json`:

```json
{
  "compilerOptions": {
    "moduleResolution": "node",
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true
  }
}
```

## Examples

See the [examples directory](./examples) for complete examples:

- [Basic Detection](./examples/basic-detection.js)
- [React App](./examples/react-app)
- [Node.js Backend](./examples/nodejs-backend)
- [Data Processing](./examples/data-processing.js)

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Node.js Support

- Node.js 14+

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and contribution guidelines.

## License

MIT © [Pynomaly Team](https://pynomaly.com)

## Support

- 📖 [Documentation](https://docs.pynomaly.com)
- 🐛 [Issue Tracker](https://github.com/pynomaly/pynomaly/issues)
- 💬 [Community Discord](https://discord.gg/pynomaly)
- 📧 [Email Support](mailto:support@pynomaly.com)