/**
 * Core types for Pynomaly JavaScript SDK
 */

// Base API Response
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: string;
}

// Configuration
export interface PynomalyConfig {
  apiKey: string;
  baseUrl: string;
  timeout?: number;
  retryAttempts?: number;
  enableWebSocket?: boolean;
  debug?: boolean;
  version?: string;
}

// Authentication
export interface AuthToken {
  token: string;
  refreshToken?: string;
  expiresAt: Date;
  tokenType: 'Bearer' | 'API-Key';
}

export interface User {
  id: string;
  email: string;
  name: string;
  roles: string[];
  permissions: string[];
  createdAt: Date;
  lastLogin?: Date;
}

// Anomaly Detection
export interface AnomalyDetectionRequest {
  data: number[][] | Record<string, any>[];
  algorithm?: 'isolation_forest' | 'local_outlier_factor' | 'one_class_svm' | 'auto';
  parameters?: Record<string, any>;
  metadata?: Record<string, any>;
}

export interface AnomalyDetectionResult {
  id: string;
  anomalies: AnomalyPoint[];
  algorithm: string;
  parameters: Record<string, any>;
  metrics: DetectionMetrics;
  createdAt: Date;
  processingTime: number;
}

export interface AnomalyPoint {
  index: number;
  score: number;
  isAnomaly: boolean;
  confidence: number;
  explanation?: string;
  data: Record<string, any>;
}

export interface DetectionMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  anomalyRate: number;
  totalPoints: number;
  anomalyCount: number;
}

// Data Quality
export interface DataQualityRequest {
  data: Record<string, any>[];
  rules?: QualityRule[];
  generateReport?: boolean;
  metadata?: Record<string, any>;
}

export interface QualityRule {
  id: string;
  name: string;
  type: 'completeness' | 'accuracy' | 'consistency' | 'validity' | 'uniqueness';
  column: string;
  condition: string;
  threshold: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface DataQualityResult {
  id: string;
  overallScore: number;
  dimensionScores: Record<string, number>;
  ruleResults: QualityRuleResult[];
  issues: QualityIssue[];
  recommendations: string[];
  report?: QualityReport;
  createdAt: Date;
}

export interface QualityRuleResult {
  ruleId: string;
  ruleName: string;
  passed: boolean;
  score: number;
  violationCount: number;
  details: string;
}

export interface QualityIssue {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  column: string;
  row?: number;
  message: string;
  suggestedFix?: string;
}

export interface QualityReport {
  id: string;
  summary: string;
  details: Record<string, any>;
  charts: ChartData[];
  recommendations: string[];
  createdAt: Date;
}

// Data Profiling
export interface DataProfilingRequest {
  data: Record<string, any>[];
  includeAdvanced?: boolean;
  generateVisualization?: boolean;
  metadata?: Record<string, any>;
}

export interface DataProfilingResult {
  id: string;
  columnProfiles: ColumnProfile[];
  datasetStats: DatasetStats;
  correlations: CorrelationMatrix;
  patterns: Pattern[];
  anomalies: AnomalyPoint[];
  visualizations?: ChartData[];
  createdAt: Date;
}

export interface ColumnProfile {
  name: string;
  type: 'numeric' | 'categorical' | 'datetime' | 'boolean' | 'text';
  count: number;
  uniqueCount: number;
  nullCount: number;
  nullPercentage: number;
  stats?: NumericStats | CategoricalStats | DatetimeStats;
}

export interface NumericStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  quartiles: [number, number, number];
  outliers: number[];
}

export interface CategoricalStats {
  categories: string[];
  frequencies: Record<string, number>;
  mostFrequent: string;
  leastFrequent: string;
}

export interface DatetimeStats {
  earliest: Date;
  latest: Date;
  range: number;
  frequency: Record<string, number>;
}

export interface DatasetStats {
  rowCount: number;
  columnCount: number;
  memoryUsage: number;
  dataTypes: Record<string, number>;
  completeness: number;
  duplicateRows: number;
}

export interface CorrelationMatrix {
  columns: string[];
  matrix: number[][];
  strongCorrelations: CorrelationPair[];
}

export interface CorrelationPair {
  column1: string;
  column2: string;
  correlation: number;
  type: 'positive' | 'negative';
}

export interface Pattern {
  type: 'trend' | 'seasonality' | 'anomaly' | 'distribution';
  description: string;
  confidence: number;
  details: Record<string, any>;
}

// Visualizations
export interface ChartData {
  type: 'line' | 'bar' | 'scatter' | 'histogram' | 'heatmap' | 'pie';
  title: string;
  data: any[];
  options?: Record<string, any>;
  metadata?: Record<string, any>;
}

// Real-time Updates
export interface WebSocketMessage {
  type: 'job_status' | 'result' | 'error' | 'notification';
  id: string;
  data: any;
  timestamp: Date;
}

export interface JobStatus {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  result?: any;
  error?: string;
  createdAt: Date;
  updatedAt: Date;
}

// Events
export interface EventCallback<T = any> {
  (data: T): void;
}

export interface EventMap {
  'job:status': JobStatus;
  'job:completed': any;
  'job:failed': { error: string };
  'connection:open': void;
  'connection:close': void;
  'connection:error': Error;
  'message': WebSocketMessage;
}

// Batch Operations
export interface BatchRequest {
  operations: BatchOperation[];
  id?: string;
  metadata?: Record<string, any>;
}

export interface BatchOperation {
  type: 'anomaly_detection' | 'data_quality' | 'data_profiling';
  id: string;
  data: any;
  parameters?: Record<string, any>;
}

export interface BatchResult {
  id: string;
  results: BatchOperationResult[];
  summary: BatchSummary;
  createdAt: Date;
  completedAt: Date;
}

export interface BatchOperationResult {
  operationId: string;
  success: boolean;
  result?: any;
  error?: string;
  processingTime: number;
}

export interface BatchSummary {
  totalOperations: number;
  successCount: number;
  failureCount: number;
  totalProcessingTime: number;
  averageProcessingTime: number;
}

// Streaming
export interface StreamConfig {
  bufferSize?: number;
  flushInterval?: number;
  autoFlush?: boolean;
}

export interface StreamMessage {
  type: 'data' | 'control' | 'error';
  payload: any;
  timestamp: Date;
}

// Errors
export interface PynomalyError {
  code: string;
  message: string;
  details?: Record<string, any>;
  requestId?: string;
  timestamp: Date;
}

// Utilities
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

export interface RequestOptions {
  method?: HttpMethod;
  headers?: Record<string, string>;
  timeout?: number;
  retries?: number;
  data?: any;
}

export interface PaginationOptions {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    pageSize: number;
    totalItems: number;
    totalPages: number;
    hasNext: boolean;
    hasPrevious: boolean;
  };
}