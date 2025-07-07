package pynomaly

import "time"

// Base types
type BaseEntity struct {
	ID        string    `json:"id"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
}

// Authentication & Configuration
type AuthConfig struct {
	APIKey   string `json:"api_key"`
	TenantID string `json:"tenant_id,omitempty"`
}

// User Management Types
type UserRole string

const (
	RoleSuperAdmin    UserRole = "super_admin"
	RoleTenantAdmin   UserRole = "tenant_admin"
	RoleDataScientist UserRole = "data_scientist"
	RoleAnalyst       UserRole = "analyst"
	RoleViewer        UserRole = "viewer"
)

type User struct {
	BaseEntity
	Email     string     `json:"email"`
	FirstName string     `json:"first_name"`
	LastName  string     `json:"last_name"`
	IsActive  bool       `json:"is_active"`
	Roles     []UserRole `json:"roles"`
	LastLogin *time.Time `json:"last_login,omitempty"`
	TenantID  string     `json:"tenant_id,omitempty"`
}

type CreateUserRequest struct {
	Email     string     `json:"email"`
	FirstName string     `json:"first_name"`
	LastName  string     `json:"last_name"`
	Password  string     `json:"password"`
	Roles     []UserRole `json:"roles"`
	TenantID  string     `json:"tenant_id,omitempty"`
}

type Tenant struct {
	BaseEntity
	Name             string `json:"name"`
	Description      string `json:"description"`
	IsActive         bool   `json:"is_active"`
	SubscriptionTier string `json:"subscription_tier"`
	MaxUsers         int    `json:"max_users"`
	MaxDetectors     int    `json:"max_detectors"`
	MaxDatasets      int    `json:"max_datasets"`
}

// Detection Types
type Dataset struct {
	BaseEntity
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	SizeBytes   int64                  `json:"size_bytes"`
	RecordCount int64                  `json:"record_count"`
	Columns     []DatasetColumn        `json:"columns"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type DatasetColumn struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Nullable    bool   `json:"nullable"`
	Description string `json:"description,omitempty"`
}

type Detector struct {
	BaseEntity
	Name               string                 `json:"name"`
	Algorithm          string                 `json:"algorithm"`
	Parameters         map[string]interface{} `json:"parameters"`
	IsTrained          bool                   `json:"is_trained"`
	TrainingMetadata   map[string]interface{} `json:"training_metadata,omitempty"`
	PerformanceMetrics map[string]interface{} `json:"performance_metrics,omitempty"`
}

type CreateDetectorRequest struct {
	Name        string                 `json:"name"`
	Algorithm   string                 `json:"algorithm"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
	Description string                 `json:"description,omitempty"`
}

type TrainDetectorRequest struct {
	DatasetID           string                 `json:"dataset_id"`
	ValidationSplit     *float64               `json:"validation_split,omitempty"`
	TrainingParameters  map[string]interface{} `json:"training_parameters,omitempty"`
}

type Anomaly struct {
	Index     int                    `json:"index"`
	Score     float64                `json:"score"`
	Features  map[string]interface{} `json:"features"`
	Metadata  map[string]interface{} `json:"metadata"`
	Timestamp *time.Time             `json:"timestamp,omitempty"`
}

type DetectionResult struct {
	Anomalies     []Anomaly              `json:"anomalies"`
	Algorithm     string                 `json:"algorithm"`
	Threshold     float64                `json:"threshold"`
	Metadata      map[string]interface{} `json:"metadata"`
	ExecutionTime float64                `json:"execution_time"`
	TotalSamples  int                    `json:"total_samples"`
	AnomalyCount  int                    `json:"anomaly_count"`
	AnomalyRate   float64                `json:"anomaly_rate"`
}

type DetectionRequest struct {
	DatasetID  string                 `json:"dataset_id"`
	DetectorID string                 `json:"detector_id"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

// Streaming Types
type StreamState string

const (
	StreamStateStopped  StreamState = "stopped"
	StreamStateStarting StreamState = "starting"
	StreamStateRunning  StreamState = "running"
	StreamStatePaused   StreamState = "paused"
	StreamStateError    StreamState = "error"
	StreamStateStopping StreamState = "stopping"
)

type WindowType string

const (
	WindowTypeTumbling WindowType = "tumbling"
	WindowTypeSliding  WindowType = "sliding"
	WindowTypeSession  WindowType = "session"
)

type StreamRecord struct {
	ID        string                 `json:"id"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
	TenantID  string                 `json:"tenant_id"`
	Metadata  map[string]interface{} `json:"metadata"`
}

type StreamProcessor struct {
	BaseEntity
	ProcessorID   string        `json:"processor_id"`
	State         StreamState   `json:"state"`
	DetectorName  string        `json:"detector_name"`
	WindowConfig  WindowConfig  `json:"window_config"`
	Metrics       StreamMetrics `json:"metrics"`
}

type WindowConfig struct {
	Type              WindowType `json:"type"`
	SizeSeconds       int        `json:"size_seconds"`
	SlideSeconds      *int       `json:"slide_seconds,omitempty"`
	GapTimeoutSeconds *int       `json:"gap_timeout_seconds,omitempty"`
}

type StreamMetrics struct {
	TotalProcessed      int64      `json:"total_processed"`
	AnomaliesDetected   int64      `json:"anomalies_detected"`
	ProcessingRate      float64    `json:"processing_rate"`
	AvgLatencyMs        float64    `json:"avg_latency_ms"`
	ErrorCount          int64      `json:"error_count"`
	LastProcessed       *time.Time `json:"last_processed,omitempty"`
	BackpressureEvents  int64      `json:"backpressure_events"`
	DroppedRecords      int64      `json:"dropped_records"`
}

type CreateProcessorRequest struct {
	ProcessorID         string             `json:"processor_id"`
	DetectorID          string             `json:"detector_id"`
	WindowConfig        WindowConfig       `json:"window_config"`
	BackpressureConfig  *BackpressureConfig `json:"backpressure_config,omitempty"`
	BufferSize          *int               `json:"buffer_size,omitempty"`
	MaxBatchSize        *int               `json:"max_batch_size,omitempty"`
	ProcessingTimeout   *int               `json:"processing_timeout,omitempty"`
}

type BackpressureConfig struct {
	MaxQueueSize       int    `json:"max_queue_size"`
	DropStrategy       string `json:"drop_strategy"` // "drop_oldest", "drop_newest", "reject"
	PressureThreshold  float64 `json:"pressure_threshold"`
}

// A/B Testing Types
type TestStatus string

const (
	TestStatusDraft     TestStatus = "draft"
	TestStatusRunning   TestStatus = "running"
	TestStatusPaused    TestStatus = "paused"
	TestStatusCompleted TestStatus = "completed"
	TestStatusCancelled TestStatus = "cancelled"
	TestStatusFailed    TestStatus = "failed"
)

type SplitStrategy string

const (
	SplitStrategyRandom     SplitStrategy = "random"
	SplitStrategyRoundRobin SplitStrategy = "round_robin"
	SplitStrategyWeighted   SplitStrategy = "weighted"
	SplitStrategyUserIDHash SplitStrategy = "user_id_hash"
	SplitStrategyTemporal   SplitStrategy = "temporal"
)

type MetricType string

const (
	MetricTypeAccuracy            MetricType = "accuracy"
	MetricTypePrecision           MetricType = "precision"
	MetricTypeRecall              MetricType = "recall"
	MetricTypeF1Score             MetricType = "f1_score"
	MetricTypeFalsePositiveRate   MetricType = "false_positive_rate"
	MetricTypeTrueNegativeRate    MetricType = "true_negative_rate"
	MetricTypeProcessingTime      MetricType = "processing_time"
	MetricTypeThroughput          MetricType = "throughput"
	MetricTypeMemoryUsage         MetricType = "memory_usage"
	MetricTypeCustom              MetricType = "custom"
)

type TestVariant struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Description       string                 `json:"description"`
	DetectorID        string                 `json:"detector_id"`
	DetectorName      string                 `json:"detector_name"`
	TrafficPercentage float64                `json:"traffic_percentage"`
	Configuration     map[string]interface{} `json:"configuration"`
	IsControl         bool                   `json:"is_control"`
}

type ABTest struct {
	BaseEntity
	TestID                 string                 `json:"test_id"`
	Name                   string                 `json:"name"`
	Description            string                 `json:"description"`
	Status                 TestStatus             `json:"status"`
	StartedAt              *time.Time             `json:"started_at,omitempty"`
	EndedAt                *time.Time             `json:"ended_at,omitempty"`
	DurationDays           *int                   `json:"duration_days,omitempty"`
	Variants               []TestVariant          `json:"variants"`
	SplitStrategy          SplitStrategy          `json:"split_strategy"`
	MetricsCollected       []MetricType           `json:"metrics_collected"`
	MinimumSampleSize      int                    `json:"minimum_sample_size"`
	ConfidenceLevel        float64                `json:"confidence_level"`
	SignificanceThreshold  float64                `json:"significance_threshold"`
	TotalExecutions        int64                  `json:"total_executions"`
	VariantStatistics      map[string]interface{} `json:"variant_statistics"`
}

// Compliance Types
type ComplianceFramework string

const (
	FrameworkGDPR     ComplianceFramework = "gdpr"
	FrameworkHIPAA    ComplianceFramework = "hipaa"
	FrameworkSOX      ComplianceFramework = "sox"
	FrameworkPCIDSS   ComplianceFramework = "pci_dss"
	FrameworkISO27001 ComplianceFramework = "iso_27001"
	FrameworkSOC2     ComplianceFramework = "soc2"
	FrameworkCCPA     ComplianceFramework = "ccpa"
	FrameworkPIPEDA   ComplianceFramework = "pipeda"
)

type AuditSeverity string

const (
	SeverityLow      AuditSeverity = "low"
	SeverityMedium   AuditSeverity = "medium"
	SeverityHigh     AuditSeverity = "high"
	SeverityCritical AuditSeverity = "critical"
)

type AuditEvent struct {
	BaseEntity
	Action               string                    `json:"action"`
	Severity             AuditSeverity             `json:"severity"`
	Timestamp            time.Time                 `json:"timestamp"`
	UserID               string                    `json:"user_id,omitempty"`
	ResourceType         string                    `json:"resource_type,omitempty"`
	ResourceID           string                    `json:"resource_id,omitempty"`
	Details              map[string]interface{}    `json:"details"`
	IPAddress            string                    `json:"ip_address,omitempty"`
	Outcome              string                    `json:"outcome"`
	RiskScore            float64                   `json:"risk_score"`
	ComplianceFrameworks []ComplianceFramework     `json:"compliance_frameworks"`
	IsHighRisk           bool                      `json:"is_high_risk"`
}

// API Response Types
type APIResponse[T any] struct {
	Success  bool                   `json:"success"`
	Data     *T                     `json:"data,omitempty"`
	Error    *APIError              `json:"error,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type APIError struct {
	Code    string                 `json:"code"`
	Message string                 `json:"message"`
	Details map[string]interface{} `json:"details,omitempty"`
}

type PaginatedResponse[T any] struct {
	Items      []T `json:"items"`
	Total      int `json:"total"`
	Page       int `json:"page"`
	PageSize   int `json:"page_size"`
	TotalPages int `json:"total_pages"`
}

type ListOptions struct {
	Page      *int                   `json:"page,omitempty"`
	PageSize  *int                   `json:"page_size,omitempty"`
	SortBy    *string                `json:"sort_by,omitempty"`
	SortOrder *string                `json:"sort_order,omitempty"` // "asc" or "desc"
	Filter    map[string]interface{} `json:"filter,omitempty"`
}

// Health Check Types
type HealthStatus struct {
	Status       string                    `json:"status"` // "healthy", "unhealthy", "degraded"
	Timestamp    time.Time                 `json:"timestamp"`
	Version      string                    `json:"version,omitempty"`
	Uptime       *int64                    `json:"uptime,omitempty"`
	Dependencies map[string]*HealthStatus  `json:"dependencies,omitempty"`
}

// Version Info
type VersionInfo struct {
	Version     string `json:"version"`
	Build       string `json:"build,omitempty"`
	Environment string `json:"environment,omitempty"`
}

// Auth Test Response
type AuthTestResponse struct {
	Authenticated bool        `json:"authenticated"`
	User          *User       `json:"user,omitempty"`
	Tenant        *Tenant     `json:"tenant,omitempty"`
}

// Algorithm Info
type AlgorithmInfo struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Category    string                 `json:"category"`
	Version     string                 `json:"version"`
}

// Upload Response
type UploadResponse struct {
	FileID string `json:"file_id"`
	URL    string `json:"url"`
}

// Error Response
type ErrorResponse struct {
	Code    string                 `json:"code"`
	Message string                 `json:"message"`
	Details map[string]interface{} `json:"details,omitempty"`
}