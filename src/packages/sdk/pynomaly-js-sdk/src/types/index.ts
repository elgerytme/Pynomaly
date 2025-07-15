/**
 * Pynomaly JavaScript SDK Types
 * 
 * TypeScript type definitions for the Pynomaly anomaly detection platform.
 */

// Configuration Types
export interface PynomalyConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  maxRetries?: number;
  debug?: boolean;
}

export interface DetectorConfig {
  algorithmName: string;
  hyperparameters?: Record<string, any>;
  contaminationRate?: number;
  randomState?: number;
  nJobs?: number;
}

export interface ExperimentConfig {
  name: string;
  description?: string;
  algorithmConfigs: DetectorConfig[];
  evaluationMetrics?: string[];
  crossValidationFolds?: number;
  randomState?: number;
  parallelJobs?: number;
  optimizationEnabled?: boolean;
  optimizationTrials?: number;
  optimizationTimeout?: number;
}

// Data Types
export interface Dataset {
  name: string;
  data: any[][] | Record<string, any>[];
  metadata?: Record<string, any>;
  featureNames?: string[];
  targetColumn?: string;
}

export interface DetectionResult {
  anomalyScores: number[];
  anomalyLabels: number[];
  nAnomalies: number;
  nSamples: number;
  contaminationRate: number;
  threshold: number;
  executionTime: number;
  metadata?: Record<string, any>;
}

export interface ModelMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  aucRoc?: number;
  aucPr?: number;
  contaminationRate?: number;
  anomalyThreshold?: number;
  customMetrics?: Record<string, number>;
}

export interface TrainingJob {
  jobId: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  detectorConfig: DetectorConfig;
  datasetName: string;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  metrics?: ModelMetrics;
  modelPath?: string;
  logs: string[];
  errorMessage?: string;
  executionTime?: number;
  memoryUsage?: number;
  cpuUsage?: number;
}

// API Response Types
export interface ApiResponse<T = any> {
  statusCode: number;
  data: T;
  headers: Record<string, string>;
  success: boolean;
}

export interface PaginatedResponse<T = any> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
  hasPrevious: boolean;
}

export interface HealthStatus {
  status: string;
  version: string;
  timestamp: string;
  services: Record<string, string>;
  metrics: Record<string, any>;
}

// Error Types
export interface PynomalyError {
  message: string;
  details?: Record<string, any>;
  statusCode?: number;
  responseData?: Record<string, any>;
}

// Event Types
export interface DetectionEvent {
  type: 'detection:started' | 'detection:progress' | 'detection:completed' | 'detection:error';
  data: any;
  timestamp: string;
}

export interface TrainingEvent {
  type: 'training:started' | 'training:progress' | 'training:completed' | 'training:error';
  data: any;
  timestamp: string;
}

export interface ExperimentEvent {
  type: 'experiment:started' | 'experiment:progress' | 'experiment:completed' | 'experiment:error';
  data: any;
  timestamp: string;
}

// Hook Types (React)
export interface UseDetectorState {
  detector: any | null;
  isLoading: boolean;
  error: PynomalyError | null;
}

export interface UseDetectionState {
  result: DetectionResult | null;
  isDetecting: boolean;
  error: PynomalyError | null;
}

export interface UseTrainingState {
  job: TrainingJob | null;
  isTraining: boolean;
  error: PynomalyError | null;
}

export interface UseExperimentState {
  experiment: any | null;
  isRunning: boolean;
  error: PynomalyError | null;
}

// Component Props Types
export interface DetectorListProps {
  onDetectorSelect?: (detector: any) => void;
  filters?: {
    algorithmName?: string;
    tags?: string[];
  };
  pageSize?: number;
  className?: string;
}

export interface DetectionResultsProps {
  result: DetectionResult;
  dataset?: Dataset;
  showVisualization?: boolean;
  showMetrics?: boolean;
  className?: string;
}

export interface AnomalyVisualizationProps {
  data: any[][];
  anomalyLabels: number[];
  anomalyScores: number[];
  featureNames?: string[];
  width?: number;
  height?: number;
  className?: string;
}

export interface TrainingJobMonitorProps {
  jobId: string;
  onComplete?: (job: TrainingJob) => void;
  onError?: (error: PynomalyError) => void;
  refreshInterval?: number;
  showLogs?: boolean;
  className?: string;
}

// Utility Types
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

export interface RequestConfig {
  method: HttpMethod;
  url: string;
  data?: any;
  params?: Record<string, any>;
  headers?: Record<string, string>;
  timeout?: number;
}

export interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  backoffFactor: number;
}

// Data Processing Types
export interface DataProcessor {
  name: string;
  process: (data: any) => any;
}

export interface ValidationRule {
  name: string;
  validate: (data: any) => boolean;
  message: string;
}

// WebSocket Types
export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  reconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}