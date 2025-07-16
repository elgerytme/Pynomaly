/**
 * Comprehensive TypeScript definitions for the Pynomaly SDK
 * Supporting modern JavaScript features and full type safety
 */

// Base configuration and utility types
export interface ClientConfig {
  /** Base URL of the Pynomaly API */
  baseUrl?: string;
  /** API key for authentication */
  apiKey?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Maximum number of retries for failed requests */
  maxRetries?: number;
  /** Custom user agent string */
  userAgent?: string;
  /** Rate limiting configuration */
  rateLimitRequests?: number;
  rateLimitPeriod?: number;
  /** Enable debug logging */
  debug?: boolean;
  /** WebSocket configuration */
  websocket?: WebSocketConfig;
  /** Custom headers to include with all requests */
  defaultHeaders?: Record<string, string>;
}

export interface WebSocketConfig {
  /** Enable WebSocket connections */
  enabled?: boolean;
  /** WebSocket URL (defaults to baseUrl with ws:// or wss://) */
  url?: string;
  /** Auto-reconnect on connection loss */
  autoReconnect?: boolean;
  /** Heartbeat interval in milliseconds */
  heartbeatInterval?: number;
  /** Maximum reconnection attempts */
  maxReconnectAttempts?: number;
  /** Reconnection delay in milliseconds */
  reconnectDelay?: number;
}

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';

export interface RequestOptions {
  /** Request timeout override */
  timeout?: number;
  /** Additional headers for this request */
  headers?: Record<string, string>;
  /** Request body data */
  data?: any;
  /** Query parameters */
  params?: Record<string, any>;
  /** Skip rate limiting for this request */
  skipRateLimit?: boolean;
}

// Authentication types
export interface AuthToken {
  /** Access token for API authentication */
  accessToken: string;
  /** Refresh token for token renewal */
  refreshToken: string;
  /** Token expiration timestamp */
  expiresAt: number;
  /** Token type (usually 'Bearer') */
  tokenType: string;
  /** Granted scopes */
  scopes?: string[];
}

export interface LoginCredentials {
  username: string;
  password: string;
  /** Multi-factor authentication code */
  mfaCode?: string;
  /** Remember login session */
  rememberMe?: boolean;
}

export interface UserProfile {
  id: string;
  username: string;
  email: string;
  role: string;
  permissions: string[];
  createdAt: string;
  lastLoginAt?: string;
}

// Detection and anomaly types
export interface DetectionRequest {
  /** Input data for anomaly detection */
  data: number[] | number[][];
  /** Algorithm to use for detection */
  algorithm?: AlgorithmType;
  /** Algorithm-specific parameters */
  parameters?: Record<string, any>;
  /** Pre-trained model ID to use */
  modelId?: string;
  /** Detection threshold override */
  threshold?: number;
  /** Include explanations in response */
  includeExplanations?: boolean;
  /** Additional metadata */
  metadata?: Record<string, any>;
}

export interface DetectionResponse {
  /** Anomaly scores for each data point */
  scores: number[];
  /** Binary anomaly predictions */
  predictions: boolean[];
  /** Overall anomaly rate */
  anomalyRate: number;
  /** Confidence scores */
  confidence: number[];
  /** Model used for detection */
  modelId?: string;
  /** Algorithm used */
  algorithm: string;
  /** Processing time in milliseconds */
  processingTime: number;
  /** Explanations (if requested) */
  explanations?: ExplanationResult[];
  /** Additional metadata */
  metadata?: Record<string, any>;
}

export type AlgorithmType = 
  | 'isolation_forest'
  | 'one_class_svm'
  | 'local_outlier_factor'
  | 'elliptic_envelope'
  | 'autoencoder'
  | 'lstm_autoencoder'
  | 'deep_svdd'
  | 'custom';

// Training types
export interface TrainingRequest {
  /** Training dataset */
  data: number[][];
  /** Algorithm to train */
  algorithm: AlgorithmType;
  /** Training parameters */
  parameters: Record<string, any>;
  /** Model name/identifier */
  name?: string;
  /** Training job description */
  description?: string;
  /** Validation data for evaluation */
  validationData?: number[][];
  /** Training configuration */
  config?: TrainingConfig;
}

export interface TrainingConfig {
  /** Contamination rate (0.0 to 1.0) */
  contamination?: number;
  /** Cross-validation folds */
  cvFolds?: number;
  /** Enable early stopping */
  earlyStop?: boolean;
  /** Evaluation metrics to track */
  metrics?: string[];
  /** Save model checkpoints */
  saveCheckpoints?: boolean;
}

export interface TrainingResponse {
  /** Unique job identifier */
  jobId: string;
  /** Training status */
  status: TrainingStatus;
  /** Model identifier (when completed) */
  modelId?: string;
  /** Training progress (0.0 to 1.0) */
  progress: number;
  /** Estimated time remaining in seconds */
  estimatedTimeRemaining?: number;
  /** Training metrics */
  metrics?: TrainingMetrics;
  /** Error message (if failed) */
  error?: string;
}

export type TrainingStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface TrainingMetrics {
  /** Training accuracy */
  accuracy?: number;
  /** Precision score */
  precision?: number;
  /** Recall score */
  recall?: number;
  /** F1 score */
  f1Score?: number;
  /** ROC AUC score */
  rocAuc?: number;
  /** Training loss */
  loss?: number;
  /** Validation loss */
  validationLoss?: number;
}

// Model management types
export interface ModelInfo {
  /** Unique model identifier */
  id: string;
  /** Model name */
  name: string;
  /** Model description */
  description?: string;
  /** Algorithm used */
  algorithm: AlgorithmType;
  /** Model parameters */
  parameters: Record<string, any>;
  /** Training status */
  status: ModelStatus;
  /** Creation timestamp */
  createdAt: string;
  /** Last updated timestamp */
  updatedAt: string;
  /** Model performance metrics */
  metrics?: TrainingMetrics;
  /** Model size in bytes */
  size?: number;
  /** Version number */
  version: string;
  /** Tags for organization */
  tags?: string[];
}

export type ModelStatus = 'training' | 'ready' | 'failed' | 'archived';

// Dataset management types
export interface DatasetInfo {
  /** Unique dataset identifier */
  id: string;
  /** Dataset name */
  name: string;
  /** Dataset description */
  description?: string;
  /** Number of samples */
  sampleCount: number;
  /** Number of features */
  featureCount: number;
  /** Dataset size in bytes */
  size: number;
  /** Creation timestamp */
  createdAt: string;
  /** Last updated timestamp */
  updatedAt: string;
  /** Dataset statistics */
  statistics?: DatasetStatistics;
  /** Data quality metrics */
  quality?: DataQualityMetrics;
  /** Tags for organization */
  tags?: string[];
}

export interface DatasetStatistics {
  /** Feature statistics */
  features: FeatureStatistics[];
  /** Missing value counts */
  missingValues: number;
  /** Duplicate row count */
  duplicates: number;
  /** Data type distribution */
  dataTypes: Record<string, number>;
}

export interface FeatureStatistics {
  /** Feature name */
  name: string;
  /** Data type */
  type: string;
  /** Minimum value */
  min?: number;
  /** Maximum value */
  max?: number;
  /** Mean value */
  mean?: number;
  /** Standard deviation */
  std?: number;
  /** Missing value count */
  missingCount: number;
  /** Unique value count */
  uniqueCount: number;
}

export interface DataQualityMetrics {
  /** Overall quality score (0-100) */
  qualityScore: number;
  /** Completeness score */
  completeness: number;
  /** Consistency score */
  consistency: number;
  /** Validity score */
  validity: number;
  /** Detected issues */
  issues: DataQualityIssue[];
}

export interface DataQualityIssue {
  /** Issue type */
  type: string;
  /** Issue severity */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Issue description */
  description: string;
  /** Affected features */
  affectedFeatures?: string[];
  /** Suggested resolution */
  resolution?: string;
}

// Explainability types
export interface ExplanationRequest {
  /** Data to explain */
  data: number[] | number[][];
  /** Model to use for explanation */
  modelId?: string;
  /** Explanation method */
  method?: ExplanationMethod;
  /** Number of top features to return */
  topFeatures?: number;
  /** Include feature interactions */
  includeInteractions?: boolean;
}

export type ExplanationMethod = 'shap' | 'lime' | 'feature_importance' | 'permutation';

export interface ExplanationResult {
  /** Feature contributions */
  featureContributions: FeatureContribution[];
  /** Overall explanation score */
  explanationScore: number;
  /** Explanation method used */
  method: ExplanationMethod;
  /** Base value (model baseline) */
  baseValue: number;
  /** Feature interactions (if included) */
  interactions?: FeatureInteraction[];
}

export interface FeatureContribution {
  /** Feature index or name */
  feature: string | number;
  /** Contribution value */
  contribution: number;
  /** Feature value */
  value: number;
  /** Contribution rank */
  rank: number;
}

export interface FeatureInteraction {
  /** First feature */
  feature1: string | number;
  /** Second feature */
  feature2: string | number;
  /** Interaction strength */
  interaction: number;
}

// Streaming types
export interface StreamConfig {
  /** Stream buffer size */
  bufferSize?: number;
  /** Sliding window size */
  windowSize?: number;
  /** Detection algorithm */
  algorithm?: AlgorithmType;
  /** Algorithm parameters */
  parameters?: Record<string, any>;
  /** Real-time alerting */
  enableAlerts?: boolean;
  /** Alert threshold */
  alertThreshold?: number;
}

export interface StreamMessage {
  /** Message type */
  type: StreamMessageType;
  /** Message payload */
  data: any;
  /** Message timestamp */
  timestamp: number;
  /** Stream identifier */
  streamId?: string;
}

export type StreamMessageType = 
  | 'data'
  | 'detection_result'
  | 'alert'
  | 'status_update'
  | 'error'
  | 'heartbeat';

export interface StreamDetectionResult {
  /** Stream identifier */
  streamId: string;
  /** Detection result */
  result: DetectionResponse;
  /** Data point timestamp */
  timestamp: number;
  /** Alert triggered */
  alert?: StreamAlert;
}

export interface StreamAlert {
  /** Alert level */
  level: 'info' | 'warning' | 'critical';
  /** Alert message */
  message: string;
  /** Anomaly score that triggered alert */
  score: number;
  /** Threshold that was exceeded */
  threshold: number;
}

// Health and monitoring types
export interface HealthStatus {
  /** Overall system status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Individual service statuses */
  services: ServiceStatus[];
  /** System uptime in seconds */
  uptime: number;
  /** Last health check timestamp */
  timestamp: string;
  /** System version */
  version: string;
}

export interface ServiceStatus {
  /** Service name */
  name: string;
  /** Service status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Response time in milliseconds */
  responseTime?: number;
  /** Last check timestamp */
  lastCheck: string;
  /** Error message (if unhealthy) */
  error?: string;
}

export interface SystemMetrics {
  /** API request metrics */
  requests: RequestMetrics;
  /** System performance metrics */
  performance: PerformanceMetrics;
  /** Error metrics */
  errors: ErrorMetrics;
  /** Resource usage metrics */
  resources: ResourceMetrics;
}

export interface RequestMetrics {
  /** Total requests */
  total: number;
  /** Requests per second */
  rps: number;
  /** Average response time */
  avgResponseTime: number;
  /** 95th percentile response time */
  p95ResponseTime: number;
  /** Success rate */
  successRate: number;
}

export interface PerformanceMetrics {
  /** CPU usage percentage */
  cpuUsage: number;
  /** Memory usage percentage */
  memoryUsage: number;
  /** Disk usage percentage */
  diskUsage: number;
  /** Network I/O metrics */
  networkIO: {
    bytesIn: number;
    bytesOut: number;
  };
}

export interface ErrorMetrics {
  /** Total error count */
  totalErrors: number;
  /** Error rate */
  errorRate: number;
  /** Error breakdown by type */
  errorsByType: Record<string, number>;
  /** Recent errors */
  recentErrors: ErrorInfo[];
}

export interface ErrorInfo {
  /** Error type */
  type: string;
  /** Error message */
  message: string;
  /** Error timestamp */
  timestamp: string;
  /** Request ID */
  requestId?: string;
}

export interface ResourceMetrics {
  /** Active connections */
  activeConnections: number;
  /** Database connections */
  dbConnections: number;
  /** Memory allocation */
  memoryAllocation: number;
  /** Cache hit rate */
  cacheHitRate: number;
}

// Event and callback types
export interface EventCallback<T = any> {
  (data: T): void | Promise<void>;
}

export interface StreamEventHandlers {
  onData?: EventCallback<StreamDetectionResult>;
  onAlert?: EventCallback<StreamAlert>;
  onError?: EventCallback<Error>;
  onConnect?: EventCallback<void>;
  onDisconnect?: EventCallback<void>;
  onReconnect?: EventCallback<void>;
}

// Utility types
export interface PaginationOptions {
  /** Page number (1-based) */
  page?: number;
  /** Number of items per page */
  limit?: number;
  /** Sort field */
  sortBy?: string;
  /** Sort direction */
  sortOrder?: 'asc' | 'desc';
}

export interface PaginatedResponse<T> {
  /** Items in current page */
  items: T[];
  /** Total number of items */
  total: number;
  /** Current page number */
  page: number;
  /** Number of items per page */
  limit: number;
  /** Total number of pages */
  totalPages: number;
  /** Has next page */
  hasNext: boolean;
  /** Has previous page */
  hasPrevious: boolean;
}

export interface FilterOptions {
  /** Text search query */
  search?: string;
  /** Filter by tags */
  tags?: string[];
  /** Date range filter */
  dateRange?: {
    start: string;
    end: string;
  };
  /** Status filter */
  status?: string[];
  /** Custom filters */
  custom?: Record<string, any>;
}

// Error types
export interface ErrorResponse {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Detailed error information */
  details?: any;
  /** Request ID for tracking */
  requestId?: string;
  /** Error timestamp */
  timestamp: string;
}