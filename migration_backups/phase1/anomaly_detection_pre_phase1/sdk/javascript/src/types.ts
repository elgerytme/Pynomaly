/**
 * Type definitions for the Anomaly Detection SDK
 */

export enum AlgorithmType {
  ISOLATION_FOREST = 'isolation_forest',
  LOCAL_OUTLIER_FACTOR = 'local_outlier_factor',
  ONE_CLASS_SVM = 'one_class_svm',
  ELLIPTIC_ENVELOPE = 'elliptic_envelope',
  AUTOENCODER = 'autoencoder',
  ENSEMBLE = 'ensemble',
}

export interface AnomalyData {
  index: number;
  score: number;
  dataPoint: number[];
  confidence?: number;
  timestamp?: string;
}

export interface DetectionResult {
  anomalies: AnomalyData[];
  totalPoints: number;
  anomalyCount: number;
  algorithmUsed: AlgorithmType;
  executionTime: number;
  modelVersion?: string;
  metadata: Record<string, any>;
}

export interface ModelInfo {
  modelId: string;
  algorithm: AlgorithmType;
  createdAt: string;
  trainingDataSize: number;
  performanceMetrics: Record<string, number>;
  hyperparameters: Record<string, any>;
  version: string;
  status: string;
}

export interface StreamingConfig {
  bufferSize?: number;
  detectionThreshold?: number;
  batchSize?: number;
  algorithm?: AlgorithmType;
  autoRetrain?: boolean;
}

export interface ExplanationResult {
  anomalyIndex: number;
  featureImportance: Record<string, number>;
  shapValues?: number[];
  limeExplanation?: Record<string, any>;
  explanationText: string;
  confidence: number;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  version: string;
  uptime: number;
  components: Record<string, string>;
  metrics: Record<string, number | string>;
}

export interface BatchProcessingRequest {
  data: number[][];
  algorithm?: AlgorithmType;
  parameters?: Record<string, any>;
  returnExplanations?: boolean;
}

export interface TrainingRequest {
  data: number[][];
  algorithm: AlgorithmType;
  hyperparameters?: Record<string, any>;
  validationSplit?: number;
  modelName?: string;
}

export interface TrainingResult {
  modelId: string;
  trainingTime: number;
  performanceMetrics: Record<string, number>;
  validationMetrics: Record<string, number>;
  modelInfo: ModelInfo;
}

export interface ClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  maxRetries?: number;
  headers?: Record<string, string>;
}

export interface StreamingClientConfig extends StreamingConfig {
  wsUrl: string;
  apiKey?: string;
  autoReconnect?: boolean;
  reconnectDelay?: number;
}

export class AnomalyDetectionError extends Error {
  public readonly code?: string;
  public readonly details?: Record<string, any>;

  constructor(message: string, code?: string, details?: Record<string, any>) {
    super(message);
    this.name = 'AnomalyDetectionError';
    this.code = code;
    this.details = details;
  }
}

export class APIError extends AnomalyDetectionError {
  public readonly statusCode: number;
  public readonly responseData?: Record<string, any>;

  constructor(message: string, statusCode: number, responseData?: Record<string, any>) {
    super(message, `HTTP_${statusCode}`, responseData);
    this.name = 'APIError';
    this.statusCode = statusCode;
    this.responseData = responseData;
  }
}

export class ValidationError extends AnomalyDetectionError {
  public readonly field?: string;
  public readonly value?: any;

  constructor(message: string, field?: string, value?: any) {
    super(message, 'VALIDATION_ERROR', { field, value });
    this.name = 'ValidationError';
    this.field = field;
    this.value = value;
  }
}

export class ConnectionError extends AnomalyDetectionError {
  public readonly url?: string;

  constructor(message: string, url?: string) {
    super(message, 'CONNECTION_ERROR', { url });
    this.name = 'ConnectionError';
    this.url = url;
  }
}

export class TimeoutError extends AnomalyDetectionError {
  public readonly timeoutDuration?: number;

  constructor(message: string, timeoutDuration?: number) {
    super(message, 'TIMEOUT_ERROR', { timeoutDuration });
    this.name = 'TimeoutError';
    this.timeoutDuration = timeoutDuration;
  }
}

export class StreamingError extends AnomalyDetectionError {
  public readonly connectionStatus?: string;

  constructor(message: string, connectionStatus?: string) {
    super(message, 'STREAMING_ERROR', { connectionStatus });
    this.name = 'StreamingError';
    this.connectionStatus = connectionStatus;
  }
}