/**
 * Type definitions for Pynomaly SDK
 */

// Base types
export interface BaseEntity {
  id: string;
  created_at: string;
  updated_at?: string;
}

// Authentication & Configuration
export interface PynomalyConfig {
  baseUrl: string;
  apiKey?: string;
  tenantId?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
}

export interface AuthConfig {
  apiKey: string;
  tenantId?: string;
}

// User Management Types
export type UserRole = 'super_admin' | 'tenant_admin' | 'data_scientist' | 'analyst' | 'viewer';

export interface User extends BaseEntity {
  email: string;
  first_name: string;
  last_name: string;
  is_active: boolean;
  roles: UserRole[];
  last_login?: string;
  tenant_id?: string;
}

export interface CreateUserRequest {
  email: string;
  first_name: string;
  last_name: string;
  password: string;
  roles: UserRole[];
  tenant_id?: string;
}

export interface Tenant extends BaseEntity {
  name: string;
  description: string;
  is_active: boolean;
  subscription_tier: string;
  max_users: number;
  max_detectors: number;
  max_datasets: number;
}

// Detection Types
export interface Dataset extends BaseEntity {
  name: string;
  description?: string;
  size_bytes: number;
  record_count: number;
  columns: DatasetColumn[];
  metadata: Record<string, any>;
}

export interface DatasetColumn {
  name: string;
  type: string;
  nullable: boolean;
  description?: string;
}

export interface Detector extends BaseEntity {
  name: string;
  algorithm: string;
  parameters: Record<string, any>;
  is_trained: boolean;
  training_metadata?: Record<string, any>;
  performance_metrics?: Record<string, any>;
}

export interface CreateDetectorRequest {
  name: string;
  algorithm: string;
  parameters?: Record<string, any>;
  description?: string;
}

export interface TrainDetectorRequest {
  dataset_id: string;
  validation_split?: number;
  training_parameters?: Record<string, any>;
}

export interface Anomaly {
  index: number;
  score: number;
  features: Record<string, any>;
  metadata: Record<string, any>;
  timestamp?: string;
}

export interface DetectionResult {
  anomalies: Anomaly[];
  algorithm: string;
  threshold: number;
  metadata: Record<string, any>;
  execution_time: number;
  total_samples: number;
  anomaly_count: number;
  anomaly_rate: number;
}

export interface DetectionRequest {
  dataset_id: string;
  detector_id: string;
  parameters?: Record<string, any>;
}

// Streaming Types
export type StreamState = 'stopped' | 'starting' | 'running' | 'paused' | 'error' | 'stopping';
export type WindowType = 'tumbling' | 'sliding' | 'session';

export interface StreamRecord {
  id: string;
  timestamp: string;
  data: Record<string, any>;
  tenant_id: string;
  metadata: Record<string, any>;
}

export interface StreamProcessor extends BaseEntity {
  processor_id: string;
  state: StreamState;
  detector_name: string;
  window_config: WindowConfig;
  metrics: StreamMetrics;
}

export interface WindowConfig {
  type: WindowType;
  size_seconds: number;
  slide_seconds?: number;
  gap_timeout_seconds?: number;
}

export interface StreamMetrics {
  total_processed: number;
  anomalies_detected: number;
  processing_rate: number;
  avg_latency_ms: number;
  error_count: number;
  last_processed?: string;
  backpressure_events: number;
  dropped_records: number;
}

export interface CreateProcessorRequest {
  processor_id: string;
  detector_id: string;
  window_config: WindowConfig;
  backpressure_config?: BackpressureConfig;
  buffer_size?: number;
  max_batch_size?: number;
  processing_timeout?: number;
}

export interface BackpressureConfig {
  max_queue_size: number;
  drop_strategy: 'drop_oldest' | 'drop_newest' | 'reject';
  pressure_threshold: number;
}

// A/B Testing Types
export type TestStatus = 'draft' | 'running' | 'paused' | 'completed' | 'cancelled' | 'failed';
export type SplitStrategy = 'random' | 'round_robin' | 'weighted' | 'user_id_hash' | 'temporal';
export type MetricType = 'accuracy' | 'precision' | 'recall' | 'f1_score' | 'false_positive_rate' | 
                        'true_negative_rate' | 'processing_time' | 'throughput' | 'memory_usage' | 'custom';

export interface TestVariant {
  id: string;
  name: string;
  description: string;
  detector_id: string;
  detector_name: string;
  traffic_percentage: number;
  configuration: Record<string, any>;
  is_control: boolean;
}

export interface ABTest extends BaseEntity {
  test_id: string;
  name: string;
  description: string;
  status: TestStatus;
  started_at?: string;
  ended_at?: string;
  duration_days?: number;
  variants: TestVariant[];
  split_strategy: SplitStrategy;
  metrics_collected: MetricType[];
  minimum_sample_size: number;
  confidence_level: number;
  significance_threshold: number;
  total_executions: number;
  variant_statistics: Record<string, any>;
}

export interface CreateABTestRequest {
  name: string;
  description: string;
  variants: CreateVariantRequest[];
  split_strategy?: SplitStrategy;
  metrics_to_collect?: MetricType[];
  minimum_sample_size?: number;
  confidence_level?: number;
  significance_threshold?: number;
  duration_days?: number;
}

export interface CreateVariantRequest {
  name: string;
  description: string;
  detector_id: string;
  traffic_percentage: number;
  configuration?: Record<string, any>;
  is_control?: boolean;
}

export interface TestResult {
  test_id: string;
  variant_id: string;
  dataset_id: string;
  execution_time: number;
  timestamp: string;
  metrics: TestMetric[];
  anomaly_count: number;
  metadata: Record<string, any>;
}

export interface TestMetric {
  name: string;
  type: MetricType;
  value: number;
  timestamp: string;
  variant_id: string;
  metadata: Record<string, any>;
}

export interface StatisticalAnalysis {
  metric_name: string;
  control_variant: string;
  treatment_variant: string;
  control_mean: number;
  treatment_mean: number;
  control_std: number;
  treatment_std: number;
  p_value: number;
  confidence_interval: [number, number];
  effect_size: number;
  is_significant: boolean;
  statistical_power: number;
  sample_size_control: number;
  sample_size_treatment: number;
  interpretation: string;
}

// Compliance Types
export type ComplianceFramework = 'gdpr' | 'hipaa' | 'sox' | 'pci_dss' | 'iso_27001' | 'soc2' | 'ccpa' | 'pipeda';
export type AuditSeverity = 'low' | 'medium' | 'high' | 'critical';

export interface AuditEvent extends BaseEntity {
  action: string;
  severity: AuditSeverity;
  timestamp: string;
  user_id?: string;
  resource_type?: string;
  resource_id?: string;
  details: Record<string, any>;
  ip_address?: string;
  outcome: string;
  risk_score: number;
  compliance_frameworks: ComplianceFramework[];
  is_high_risk: boolean;
}

export interface GDPRRequest extends BaseEntity {
  request_type: string;
  data_subject_id: string;
  data_subject_email: string;
  request_details: string;
  submitted_at: string;
  status: string;
  assigned_to?: string;
  completion_deadline?: string;
  processed_at?: string;
  is_overdue: boolean;
  notes: string;
}

// Business Intelligence Types
export interface BusinessMetrics {
  detection_accuracy: number;
  total_detections: number;
  false_positive_rate: number;
  average_processing_time: number;
  cost_per_detection: number;
  system_uptime: number;
  active_users: number;
  data_processed_gb: number;
}

export interface DashboardWidget {
  id: string;
  type: 'chart' | 'metric' | 'table' | 'alert';
  title: string;
  configuration: Record<string, any>;
  data: any;
  refresh_interval?: number;
}

// Integration Types
export type IntegrationType = 'slack' | 'pagerduty' | 'teams' | 'webhook' | 'email';

export interface Integration extends BaseEntity {
  name: string;
  type: IntegrationType;
  configuration: Record<string, any>;
  is_active: boolean;
  last_used?: string;
  success_count: number;
  failure_count: number;
}

export interface NotificationTemplate {
  id: string;
  name: string;
  integration_type: IntegrationType;
  template: string;
  variables: string[];
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiError;
  metadata?: Record<string, any>;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ListOptions {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  filter?: Record<string, any>;
}

// WebSocket Types
export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface AnomalyAlert {
  processor_id: string;
  anomaly_id: string;
  score: number;
  timestamp: string;
  data: Record<string, any>;
  metadata: Record<string, any>;
}

// Health Check Types
export interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  version?: string;
  uptime?: number;
  dependencies?: Record<string, HealthStatus>;
}