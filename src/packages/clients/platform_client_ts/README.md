# Platform TypeScript Client

Official TypeScript/JavaScript client for monorepo services with full type safety and modern features.

## Features

- 🔍 **Anomaly Detection**: Full support for anomaly detection APIs
- 🤖 **MLOps**: Model management and deployment
- 🔐 **JWT Authentication**: Automatic token management and refresh
- 📝 **Full TypeScript**: Complete type definitions for all APIs
- 🌐 **Cross-Platform**: Works in Node.js and modern browsers
- 🔄 **Retry Logic**: Automatic retry with exponential backoff
- ⚡ **Modern**: Promise-based async/await API
- 📦 **Tree Shakeable**: Import only what you need

## Installation

```bash
npm install @platform/client
```

```bash
yarn add @platform/client
```

```bash
pnpm add @platform/client
```

## Quick Start

### Basic Usage

```typescript
import { PlatformClient } from '@platform/client';

const client = new PlatformClient({
  apiKey: 'your-api-key',
  environment: 'production' // or 'development', 'staging'
});

// Anomaly detection
const result = await client.anomalyDetection.detect({
  data: [[1.0, 2.0], [2.0, 3.0], [100.0, 200.0]],
  algorithm: 'isolation_forest'
});

console.log('Anomalies found:', result.anomalies);
```

### Service-Specific Clients

```typescript
import { AnomalyDetectionClient } from '@platform/client/anomaly-detection';

const anomalyClient = new AnomalyDetectionClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.platform.com'
});

const result = await anomalyClient.detect({
  data: [[1, 2], [2, 3], [100, 200]],
  algorithm: 'isolation_forest',
  contamination: 0.1
});
```

## API Reference

### Anomaly Detection

```typescript
// Simple detection
const detection = await client.anomalyDetection.detect({
  data: number[][],
  algorithm: string,
  contamination?: number,
  parameters?: Record<string, any>
});

// Ensemble detection
const ensemble = await client.anomalyDetection.detectEnsemble({
  data: number[][],
  algorithms: string[],
  votingStrategy: 'majority' | 'average' | 'max',
  contamination?: number
});

// Model training
const training = await client.anomalyDetection.trainModel({
  data: number[][],
  algorithm: string,
  name: string,
  description?: string,
  contamination?: number,
  parameters?: Record<string, any>
});

// Prediction with trained model
const prediction = await client.anomalyDetection.predict({
  data: number[][],
  modelId: string
});

// Model management
const models = await client.anomalyDetection.listModels();
const model = await client.anomalyDetection.getModel('model-id');
await client.anomalyDetection.deleteModel('model-id');

// Get available algorithms
const algorithms = await client.anomalyDetection.getAlgorithms();
```

### MLOps

```typescript
// Pipeline management
const pipeline = await client.mlops.createPipeline({
  name: 'my-pipeline',
  description: 'Production pipeline',
  pipelineType: 'training',
  algorithm: 'isolation_forest'
});

const execution = await client.mlops.executePipeline(pipeline.id);
const deployments = await client.mlops.listDeployments();
```

## Framework Integration

### React

```tsx
import { useEffect, useState } from 'react';
import { PlatformClient } from '@platform/client';

function AnomalyDetector() {
  const [client] = useState(() => new PlatformClient({
    apiKey: process.env.REACT_APP_API_KEY!
  }));
  
  const [result, setResult] = useState(null);
  
  const detectAnomalies = async () => {
    try {
      const detection = await client.anomalyDetection.detect({
        data: [[1, 2], [2, 3], [100, 200]],
        algorithm: 'isolation_forest'
      });
      setResult(detection);
    } catch (error) {
      console.error('Detection failed:', error);
    }
  };
  
  return (
    <div>
      <button onClick={detectAnomalies}>Detect Anomalies</button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}
```

### Angular

```typescript
import { Injectable } from '@angular/core';
import { PlatformClient } from '@platform/client';

@Injectable({
  providedIn: 'root'
})
export class AnomalyService {
  private client = new PlatformClient({
    apiKey: environment.apiKey
  });
  
  async detectAnomalies(data: number[][]) {
    return this.client.anomalyDetection.detect({
      data,
      algorithm: 'isolation_forest'
    });
  }
}
```

### Vue 3

```vue
<template>
  <div>
    <button @click="detectAnomalies">Detect Anomalies</button>
    <pre v-if="result">{{ result }}</pre>  
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { PlatformClient } from '@platform/client';

const client = new PlatformClient({
  apiKey: import.meta.env.VITE_API_KEY
});

const result = ref(null);

const detectAnomalies = async () => {
  try {
    result.value = await client.anomalyDetection.detect({
      data: [[1, 2], [2, 3], [100, 200]],
      algorithm: 'isolation_forest'
    });
  } catch (error) {
    console.error('Detection failed:', error);
  }
};
</script>
```

## Configuration

```typescript
import { PlatformClient, Environment } from '@platform/client';

const client = new PlatformClient({
  apiKey: 'your-api-key',
  environment: Environment.Production,
  baseUrl: 'https://api.platform.com', // Optional custom base URL
  timeout: 30000, // Request timeout in ms
  maxRetries: 3, // Max retry attempts  
  retryDelay: 1000, // Base retry delay in ms
  rateLimitRequests: 100, // Rate limit per period
  rateLimitPeriod: 60000, // Rate limit period in ms
  headers: { // Custom headers
    'X-Custom-Header': 'value'
  }
});
```

## Error Handling

```typescript
import { 
  PlatformError,
  AuthenticationError,
  ValidationError,
  RateLimitError,
  ServerError 
} from '@platform/client';

try {
  const result = await client.anomalyDetection.detect({
    data: invalidData
  });
} catch (error) {
  if (error instanceof ValidationError) {
    console.log('Validation failed:', error.message);
    console.log('Details:', error.details);
  } else if (error instanceof RateLimitError) {
    console.log('Rate limited. Retry after:', error.retryAfter);
  } else if (error instanceof AuthenticationError) {
    console.log('Authentication failed:', error.message);
  } else if (error instanceof ServerError) {
    console.log('Server error:', error.message);
  } else {
    console.log('Unknown error:', error);
  }
}
```

## Environment Variables

```bash
# .env file
PLATFORM_API_KEY=your-api-key
PLATFORM_BASE_URL=https://api.platform.com
PLATFORM_TIMEOUT=30000
```

## Examples

Check out the `/examples` directory for complete working examples:

- Basic anomaly detection
- Ensemble methods  
- Model training and management
- React integration
- Node.js server usage
- Error handling patterns

## TypeScript Support

This package includes full TypeScript definitions. No additional `@types` package is needed.

```typescript
import type { 
  DetectionRequest,
  DetectionResponse,
  ModelInfo,
  AlgorithmInfo 
} from '@platform/client/types';
```

## Browser Support

- Chrome >= 90
- Firefox >= 88
- Safari >= 14
- Edge >= 90

## Node.js Support

- Node.js >= 16.0.0