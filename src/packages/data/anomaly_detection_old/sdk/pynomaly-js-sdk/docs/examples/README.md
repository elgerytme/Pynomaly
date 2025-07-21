# JavaScript SDK Examples

This directory contains comprehensive examples demonstrating how to use the Pynomaly JavaScript SDK in various scenarios.

## Example Categories

### Basic Usage
- [Getting Started](./getting-started.js) - Basic SDK setup and usage
- [Authentication](./authentication.js) - User authentication and session management
- [Configuration](./configuration.js) - SDK configuration options

### Core Features
- [Anomaly Detection](./anomaly-detection.js) - Complete anomaly detection examples
- [Data Quality Analysis](./data-quality.js) - Data quality assessment
- [Data Profiling](./data-profiling.js) - Dataset profiling and statistics

### Advanced Features
- [Async Operations](./async-operations.js) - Asynchronous job processing
- [WebSocket Integration](./websocket-integration.js) - Real-time updates
- [Streaming Processing](./streaming-processing.js) - Large dataset handling
- [Error Handling](./error-handling.js) - Comprehensive error management

### Framework Integration
- [React Examples](./frameworks/react-examples.jsx) - React hooks and components
- [Vue Examples](./frameworks/vue-examples.vue) - Vue composables and components
- [Angular Examples](./frameworks/angular-examples.ts) - Angular services and components

### Production Ready
- [Production Setup](./production-setup.js) - Production environment configuration
- [Security Best Practices](./security-best-practices.js) - Security implementation
- [Performance Optimization](./performance-optimization.js) - Performance tips

## Running Examples

1. Install dependencies:
```bash
npm install @pynomaly/js-sdk
```

2. Set your API credentials:
```bash
export PYNOMALY_API_KEY="your-api-key"
export PYNOMALY_BASE_URL="https://api.pynomaly.com"
```

3. Run an example:
```bash
node examples/getting-started.js
```

## Example Data

All examples use sample datasets located in the `data/` directory:
- `sample-numeric.json` - Numeric dataset with anomalies
- `sample-categorical.json` - Categorical dataset with quality issues
- `sample-timeseries.json` - Time series data
- `sample-mixed.json` - Mixed data types

## Support

If you have questions about any examples, please:
1. Check the [main documentation](../README.md)
2. Review the [API reference](../docs/api-reference.md)
3. Open an issue on GitHub