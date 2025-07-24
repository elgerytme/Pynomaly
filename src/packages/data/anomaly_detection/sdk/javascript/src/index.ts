/**
 * Anomaly Detection JavaScript/TypeScript SDK
 * 
 * A comprehensive SDK for integrating with the Anomaly Detection service
 * in both browser and Node.js environments.
 */

// Main client classes
export { AnomalyDetectionClient } from './client';
export { StreamingClient, createStreamingClient } from './streaming-client';

// Type definitions
export {
  // Enums
  AlgorithmType,
  
  // Core data types
  AnomalyData,
  DetectionResult,
  ModelInfo,
  StreamingConfig,
  ExplanationResult,
  HealthStatus,
  
  // Request/Response types
  BatchProcessingRequest,
  TrainingRequest,
  TrainingResult,
  
  // Configuration types
  ClientConfig,
  StreamingClientConfig,
  
  // Error types
  AnomalyDetectionError,
  APIError,
  ValidationError,
  ConnectionError,
  TimeoutError,
  StreamingError,
} from './types';

// Utility functions
export * from './utils';

// Version
export const VERSION = '1.0.0';