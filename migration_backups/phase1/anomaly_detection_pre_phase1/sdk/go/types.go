package anomalydetection

import (
	"time"
)

// AlgorithmType represents the available anomaly detection algorithms
type AlgorithmType string

const (
	IsolationForest     AlgorithmType = "isolation_forest"
	LocalOutlierFactor  AlgorithmType = "local_outlier_factor"
	OneClassSVM         AlgorithmType = "one_class_svm"
	EllipticEnvelope    AlgorithmType = "elliptic_envelope"
	Autoencoder         AlgorithmType = "autoencoder"
	Ensemble            AlgorithmType = "ensemble"
)

// AnomalyData represents an individual anomaly detection
type AnomalyData struct {
	Index      int       `json:"index"`
	Score      float64   `json:"score"`
	DataPoint  []float64 `json:"data_point"`
	Confidence *float64  `json:"confidence,omitempty"`
	Timestamp  *time.Time `json:"timestamp,omitempty"`
}

// DetectionResult represents the result of anomaly detection operation
type DetectionResult struct {
	Anomalies     []AnomalyData      `json:"anomalies"`
	TotalPoints   int                `json:"total_points"`
	AnomalyCount  int                `json:"anomaly_count"`
	AlgorithmUsed AlgorithmType      `json:"algorithm_used"`
	ExecutionTime float64            `json:"execution_time"`
	ModelVersion  *string            `json:"model_version,omitempty"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// ModelInfo represents information about a trained model
type ModelInfo struct {
	ModelID            string                 `json:"model_id"`
	Algorithm          AlgorithmType          `json:"algorithm"`
	CreatedAt          time.Time              `json:"created_at"`
	TrainingDataSize   int                    `json:"training_data_size"`
	PerformanceMetrics map[string]float64     `json:"performance_metrics"`
	Hyperparameters    map[string]interface{} `json:"hyperparameters"`
	Version            string                 `json:"version"`
	Status             string                 `json:"status"`
}

// StreamingConfig represents configuration for streaming detection
type StreamingConfig struct {
	BufferSize          int           `json:"buffer_size,omitempty"`
	DetectionThreshold  float64       `json:"detection_threshold,omitempty"`
	BatchSize           int           `json:"batch_size,omitempty"`
	Algorithm           AlgorithmType `json:"algorithm,omitempty"`
	AutoRetrain         bool          `json:"auto_retrain,omitempty"`
}

// ExplanationResult represents the result of anomaly explanation
type ExplanationResult struct {
	AnomalyIndex      int                    `json:"anomaly_index"`
	FeatureImportance map[string]float64     `json:"feature_importance"`
	ShapValues        []float64              `json:"shap_values,omitempty"`
	LimeExplanation   map[string]interface{} `json:"lime_explanation,omitempty"`
	ExplanationText   string                 `json:"explanation_text"`
	Confidence        float64                `json:"confidence"`
}

// HealthStatus represents the health status of the service
type HealthStatus struct {
	Status     string                    `json:"status"`
	Timestamp  time.Time                 `json:"timestamp"`
	Version    string                    `json:"version"`
	Uptime     float64                   `json:"uptime"`
	Components map[string]string         `json:"components"`
	Metrics    map[string]interface{}    `json:"metrics"`
}

// BatchProcessingRequest represents a request for batch processing
type BatchProcessingRequest struct {
	Data               [][]float64            `json:"data"`
	Algorithm          AlgorithmType          `json:"algorithm,omitempty"`
	Parameters         map[string]interface{} `json:"parameters,omitempty"`
	ReturnExplanations bool                   `json:"return_explanations,omitempty"`
}

// TrainingRequest represents a request for model training
type TrainingRequest struct {
	Data            [][]float64            `json:"data"`
	Algorithm       AlgorithmType          `json:"algorithm"`
	Hyperparameters map[string]interface{} `json:"hyperparameters,omitempty"`
	ValidationSplit float64                `json:"validation_split,omitempty"`
	ModelName       *string                `json:"model_name,omitempty"`
}

// TrainingResult represents the result of model training
type TrainingResult struct {
	ModelID           string             `json:"model_id"`
	TrainingTime      float64            `json:"training_time"`
	PerformanceMetrics map[string]float64 `json:"performance_metrics"`
	ValidationMetrics map[string]float64 `json:"validation_metrics"`
	ModelInfo         ModelInfo          `json:"model_info"`
}

// ClientConfig represents configuration for the API client
type ClientConfig struct {
	BaseURL    string
	APIKey     *string
	Timeout    time.Duration
	MaxRetries int
	Headers    map[string]string
}

// StreamingClientConfig represents configuration for the streaming client
type StreamingClientConfig struct {
	WSURL           string
	APIKey          *string
	Config          StreamingConfig
	AutoReconnect   bool
	ReconnectDelay  time.Duration
}

// Error types

// SDKError is the base error type for SDK errors
type SDKError struct {
	Message string
	Code    *string
	Details map[string]interface{}
}

func (e *SDKError) Error() string {
	if e.Code != nil {
		return "[" + *e.Code + "] " + e.Message
	}
	return e.Message
}

// APIError represents an error from the API service
type APIError struct {
	*SDKError
	StatusCode   int
	ResponseData map[string]interface{}
}

// ValidationError represents a data validation error
type ValidationError struct {
	*SDKError
	Field *string
	Value interface{}
}

// ConnectionError represents a network connection error
type ConnectionError struct {
	*SDKError
	URL *string
}

// TimeoutError represents a request timeout error
type TimeoutError struct {
	*SDKError
	Duration time.Duration
}

// StreamingError represents a streaming connection error
type StreamingError struct {
	*SDKError
	ConnectionStatus *string
}