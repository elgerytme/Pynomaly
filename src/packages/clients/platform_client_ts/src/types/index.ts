/**
 * Type definitions for platform client
 */

// Base types
export interface BaseResponse {
  success: boolean;
  message?: string;
  timestamp: string;
  requestId?: string;
}

export interface ErrorResponse extends BaseResponse {
  success: false;
  errorCode: string;
  errorMessage: string;
  details?: ErrorDetail[];
}

export interface ErrorDetail {
  field?: string;
  code: string;
  message: string;
}

export interface DataResponse<T> extends BaseResponse {
  data: T;
}

export interface PaginatedResponse<T> extends BaseResponse {
  data: T[];
  pagination: PaginationInfo;
}

export interface PaginationInfo {
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
  hasNext: boolean;
  hasPrevious: boolean;
}

// Anomaly Detection types
export interface DetectionRequest {
  data: number[][];
  algorithm?: string;
  contamination?: number;
  parameters?: Record<string, any>;
}

export interface DetectionResponse extends BaseResponse {
  anomalies: number[];
  scores?: number[];
  algorithm: string;
  totalSamples: number;
  anomalyCount: number;
  processingTimeMs: number;
}

export interface EnsembleDetectionRequest {
  data: number[][];
  algorithms: string[];
  votingStrategy?: 'majority' | 'average' | 'max';
  contamination?: number;
  algorithmParameters?: Record<string, Record<string, any>>;
}

export interface EnsembleDetectionResponse extends BaseResponse {
  anomalies: number[];
  ensembleScores: number[];
  individualResults: Record<string, any>;
  votingStrategy: string;
  algorithmsUsed: string[];
  totalSamples: number;
  anomalyCount: number;
  processingTimeMs: number;
}

export interface TrainingRequest {
  data: number[][];
  algorithm: string;
  name: string;
  description?: string;
  contamination?: number;
  parameters?: Record<string, any>;
  validationSplit?: number;
}

export interface TrainingResponse extends BaseResponse {
  model: ModelInfo;
  trainingMetrics: Record<string, number>;
  validationMetrics?: Record<string, number>;
  trainingTimeMs: number;
}

export interface PredictionRequest {
  data: number[][];
  modelId: string;
}

export interface PredictionResponse extends BaseResponse {
  anomalies: number[];
  scores: number[];
  modelId: string;
  modelName: string;
  totalSamples: number;
  anomalyCount: number;
  processingTimeMs: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  algorithm: string;
  status: string;
  createdAt: string;
  updatedAt?: string;
  contamination: number;
  trainingSamples: number;
  parameters: Record<string, any>;
  performanceMetrics?: Record<string, number>;
}

export interface AlgorithmInfo {
  name: string;
  displayName: string;
  description: string;
  parameters: Record<string, ParameterInfo>;
  supportsOnlineLearning: boolean;
  supportsFeatureImportance: boolean;
  computationalComplexity: string;
}

export interface ParameterInfo {
  type: string;
  default: any;
  range?: [number, number];
  options?: any[];
  description?: string;
}

// MLOps types
export interface PipelineCreateRequest {
  name: string;
  description?: string;
  pipelineType: string;
  algorithm: string;
  hyperparameters?: Record<string, any>;
  dataSource?: Record<string, any>;
}

export interface PipelineInfo {
  id: string;
  name: string;
  description?: string;
  pipelineType: string;
  algorithm: string;
  status: string;
  createdAt: string;
  updatedAt?: string;
}

export interface ExecutionInfo {
  id: string;
  pipelineId: string;
  status: string;
  startedAt: string;
  completedAt?: string;
  metrics?: Record<string, number>;
}

// Health and metrics types
export interface HealthResponse extends BaseResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime: number;
  checks: Record<string, Record<string, any>>;
}

export interface MetricsResponse extends BaseResponse {
  metrics: Record<string, number | string>;
}

// Configuration types
export interface ClientOptions {
  apiKey: string;
  baseUrl?: string;
  environment?: Environment;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
  rateLimitRequests?: number;
  rateLimitPeriod?: number;
  headers?: Record<string, string>;
}

export enum Environment {
  Development = 'development',
  Staging = 'staging',
  Production = 'production',
  Local = 'local'
}

// HTTP types
export interface RequestConfig {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  url: string;
  headers?: Record<string, string>;
  params?: Record<string, any>;
  data?: any;
  timeout?: number;
}