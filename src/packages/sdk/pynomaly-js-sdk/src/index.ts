/**
 * Pynomaly JavaScript SDK
 * 
 * Main entry point for the Pynomaly JavaScript SDK.
 * Provides comprehensive anomaly detection capabilities for web applications.
 * 
 * @version 0.1.0
 * @author Pynomaly Team
 */

// Core exports
export { PynomalyClient, createClient } from './core/client';
export { DataScienceAPI } from './core/dataScience';

// Error exports
export {
  PynomalySDKError,
  AuthenticationError,
  ValidationError,
  APIError,
  ResourceNotFoundError,
  RateLimitError,
  DataError,
  ModelError,
  ConfigurationError,
  NetworkError,
  WebSocketError,
  isPynomalyError,
  getErrorMessage,
  getErrorDetails
} from './core/errors';

// Type exports
export type {
  // Configuration types
  PynomalyConfig,
  DetectorConfig,
  ExperimentConfig,
  
  // Data types
  Dataset,
  DetectionResult,
  ModelMetrics,
  TrainingJob,
  
  // API response types
  ApiResponse,
  PaginatedResponse,
  HealthStatus,
  
  // Error types
  PynomalyError,
  
  // Event types
  DetectionEvent,
  TrainingEvent,
  ExperimentEvent,
  
  // Hook types (React)
  UseDetectorState,
  UseDetectionState,
  UseTrainingState,
  UseExperimentState,
  
  // Component props types
  DetectorListProps,
  DetectionResultsProps,
  AnomalyVisualizationProps,
  TrainingJobMonitorProps,
  
  // Utility types
  HttpMethod,
  RequestConfig,
  RetryConfig,
  DataProcessor,
  ValidationRule,
  WebSocketConfig,
  WebSocketMessage
} from './types';

// React hooks exports (optional - only if React is available)
export { useDetector, useDetectorList } from './hooks/useDetector';
export { useDetection, useDetectionHistory, useDetectionComparison } from './hooks/useDetection';
export { useTraining, useTrainingMonitor, useTrainingJobList } from './hooks/useTraining';

// React components exports (optional - only if React is available)
export { DetectorList, SimpleDetectorList } from './components/DetectorList';
export { DetectionResults } from './components/DetectionResults';

// Utility exports
export {
  DatasetConverter,
  DataValidator,
  ResultAnalyzer,
  Utils
} from './utils';

// Version info
export const VERSION = '0.1.0';

// Default export - main client class
export default PynomalyClient;