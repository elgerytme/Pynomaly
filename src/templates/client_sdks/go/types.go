package anomaly_detection

import (
	"time"
)

// DetectionRequest represents a request for anomaly detection
type DetectionRequest struct {
	Data           [][]float64            `json:"data"`
	Algorithm      string                 `json:"algorithm"`
	Contamination  *float64               `json:"contamination,omitempty"`
	Parameters     map[string]interface{} `json:"parameters,omitempty"`
	ExplainResults bool                   `json:"explain_results,omitempty"`
}

// DetectionResponse represents the response from anomaly detection
type DetectionResponse struct {
	Success          bool      `json:"success"`
	Anomalies        []bool    `json:"anomalies"`
	ConfidenceScores []float64 `json:"confidence_scores,omitempty"`
	Algorithm        string    `json:"algorithm"`
	TotalSamples     int       `json:"total_samples"`
	AnomalyRate      float64   `json:"anomaly_rate"`
	AnomaliesDetected int      `json:"anomalies_detected"`
	Explanations     *Explanations `json:"explanations,omitempty"`
	ProcessingTime   float64   `json:"processing_time"`
	ModelVersion     string    `json:"model_version,omitempty"`
}

// Explanations contains model explanations for detected anomalies
type Explanations struct {
	SHAP         [][]float64            `json:"shap,omitempty"`
	LIME         [][]float64            `json:"lime,omitempty"`
	FeatureImportance []float64         `json:"feature_importance,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// EnsembleRequest represents a request for ensemble anomaly detection
type EnsembleRequest struct {
	Data        [][]float64              `json:"data"`
	Algorithms  []string                 `json:"algorithms"`
	Method      string                   `json:"method"` // "voting", "average", "weighted"
	Weights     []float64                `json:"weights,omitempty"`
	Parameters  map[string]interface{}   `json:"parameters,omitempty"`
}

// EnsembleResponse represents the response from ensemble detection
type EnsembleResponse struct {
	Success          bool                   `json:"success"`
	Anomalies        []bool                 `json:"anomalies"`
	ConfidenceScores []float64              `json:"confidence_scores"`
	IndividualResults map[string][]bool     `json:"individual_results"`
	Method           string                 `json:"method"`
	Algorithms       []string               `json:"algorithms"`
	TotalSamples     int                    `json:"total_samples"`
	AnomalyRate      float64                `json:"anomaly_rate"`
	ProcessingTime   float64                `json:"processing_time"`
}

// TrainingRequest represents a model training request
type TrainingRequest struct {
	Data           [][]float64            `json:"data"`
	Algorithm      string                 `json:"algorithm"`
	Parameters     map[string]interface{} `json:"parameters,omitempty"`
	ModelName      string                 `json:"model_name,omitempty"`
	ValidationSplit float64               `json:"validation_split,omitempty"`
}

// TrainingResponse represents the response from model training
type TrainingResponse struct {
	Success        bool                   `json:"success"`
	ModelID        string                 `json:"model_id"`
	ModelName      string                 `json:"model_name,omitempty"`
	Algorithm      string                 `json:"algorithm"`
	TrainingTime   float64                `json:"training_time"`
	ValidationScore float64               `json:"validation_score,omitempty"`
	Parameters     map[string]interface{} `json:"parameters"`
	CreatedAt      time.Time              `json:"created_at"`
}

// Model represents a trained anomaly detection model
type Model struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name,omitempty"`
	Algorithm       string                 `json:"algorithm"`
	Parameters      map[string]interface{} `json:"parameters"`
	TrainingScore   float64                `json:"training_score,omitempty"`
	ValidationScore float64                `json:"validation_score,omitempty"`
	Status          string                 `json:"status"` // "training", "ready", "failed"
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	Version         string                 `json:"version,omitempty"`
}

// ModelsListResponse represents the response from listing models
type ModelsListResponse struct {
	Success bool    `json:"success"`
	Models  []Model `json:"models"`
	Total   int     `json:"total"`
	Page    int     `json:"page,omitempty"`
	PerPage int     `json:"per_page,omitempty"`
}

// Algorithm represents an available anomaly detection algorithm
type Algorithm struct {
	Name        string                 `json:"name"`
	DisplayName string                 `json:"display_name"`
	Description string                 `json:"description"`
	Type        string                 `json:"type"` // "statistical", "ml", "deep_learning"
	Parameters  map[string]interface{} `json:"parameters"`
	Complexity  string                 `json:"complexity"` // "low", "medium", "high"
	SuitableFor []string               `json:"suitable_for"`
}

// AlgorithmsResponse represents the response from listing algorithms
type AlgorithmsResponse struct {
	Success    bool        `json:"success"`
	Algorithms []Algorithm `json:"algorithms"`
	Categories map[string][]string `json:"categories"`
}

// HealthResponse represents the health check response
type HealthResponse struct {
	Status     string    `json:"status"` // "healthy", "degraded", "unhealthy"
	Version    string    `json:"version"`
	Timestamp  time.Time `json:"timestamp"`
	Components map[string]ComponentHealth `json:"components"`
	Uptime     float64   `json:"uptime_seconds"`
}

// ComponentHealth represents health of individual system components
type ComponentHealth struct {
	Status      string                 `json:"status"`
	Message     string                 `json:"message,omitempty"`
	LastChecked time.Time              `json:"last_checked"`
	Metrics     map[string]interface{} `json:"metrics,omitempty"`
}

// DatasetUploadRequest represents a dataset upload request
type DatasetUploadRequest struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Data        [][]float64            `json:"data"`
	Features    []string               `json:"features,omitempty"`
	Labels      []bool                 `json:"labels,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// DatasetUploadResponse represents the response from dataset upload
type DatasetUploadResponse struct {
	Success   bool      `json:"success"`
	DatasetID string    `json:"dataset_id"`
	Name      string    `json:"name"`
	Size      int       `json:"size"`
	Features  int       `json:"features"`
	UploadedAt time.Time `json:"uploaded_at"`
}

// StreamingMessage represents a real-time streaming message
type StreamingMessage struct {
	Type      string                 `json:"type"` // "data", "anomaly", "status", "error"
	Timestamp time.Time              `json:"timestamp"`
	Data      []float64              `json:"data,omitempty"`
	Anomaly   bool                   `json:"anomaly,omitempty"`
	Confidence float64               `json:"confidence,omitempty"`
	Message   string                 `json:"message,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// ClientConfig represents configuration for the anomaly detection client
type ClientConfig struct {
	BaseURL           string
	APIKey            string
	JWTToken          string
	Timeout           time.Duration
	MaxRetries        int
	RetryBackoff      time.Duration
	RateLimit         int // requests per minute
	UserAgent         string
	EnableStreaming   bool
	StreamingURL      string
	Debug             bool
}

// AuthResponse represents authentication response
type AuthResponse struct {
	Success      bool      `json:"success"`
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	TokenType    string    `json:"token_type"`
	ExpiresIn    int       `json:"expires_in"`
	ExpiresAt    time.Time `json:"expires_at"`
	Scope        string    `json:"scope,omitempty"`
}

// AuthRequest represents authentication request
type AuthRequest struct {
	Username string `json:"username,omitempty"`
	Password string `json:"password,omitempty"`
	APIKey   string `json:"api_key,omitempty"`
	TokenType string `json:"token_type,omitempty"`
}

// ErrorResponse represents API error responses
type ErrorResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error"`
	Code    string `json:"code,omitempty"`
	Details map[string]interface{} `json:"details,omitempty"`
	RequestID string `json:"request_id,omitempty"`
}