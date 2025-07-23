package anomaly_detection

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	tests := []struct {
		name     string
		config   *ClientConfig
		expected *ClientConfig
	}{
		{
			name:   "nil config uses defaults",
			config: nil,
			expected: &ClientConfig{
				BaseURL:         "https://api.anomaly-detection.com",
				Timeout:         30 * time.Second,
				MaxRetries:      3,
				RetryBackoff:    time.Second,
				RateLimit:       100,
				UserAgent:       "anomaly-detection-go-sdk/1.0.0",
				EnableStreaming: false,
				StreamingURL:    "wss://api.anomaly-detection.com/ws",
				Debug:           false,
			},
		},
		{
			name: "custom config preserved",
			config: &ClientConfig{
				BaseURL:    "https://custom.api.com",
				APIKey:     "test-key",
				Timeout:    60 * time.Second,
				MaxRetries: 5,
				Debug:      true,
			},
			expected: &ClientConfig{
				BaseURL:         "https://custom.api.com",
				APIKey:          "test-key",
				Timeout:         60 * time.Second,
				MaxRetries:      5,
				RetryBackoff:    time.Second,
				RateLimit:       100,
				UserAgent:       "anomaly-detection-go-sdk/1.0.0",
				EnableStreaming: false,
				StreamingURL:    "wss://api.anomaly-detection.com/ws",
				Debug:           true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewClient(tt.config)
			
			assert.Equal(t, tt.expected.BaseURL, client.config.BaseURL)
			assert.Equal(t, tt.expected.APIKey, client.config.APIKey)
			assert.Equal(t, tt.expected.Timeout, client.config.Timeout)
			assert.Equal(t, tt.expected.MaxRetries, client.config.MaxRetries)
			assert.Equal(t, tt.expected.RateLimit, client.config.RateLimit)
			assert.Equal(t, tt.expected.Debug, client.config.Debug)
		})
	}
}

func TestClient_DetectAnomalies(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/v1/detect", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		assert.Equal(t, "anomaly-detection-go-sdk/1.0.0", r.Header.Get("User-Agent"))

		// Parse request body
		var request DetectionRequest
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		assert.Equal(t, "isolation_forest", request.Algorithm)
		assert.Equal(t, 2, len(request.Data))
		assert.Equal(t, 0.1, *request.Contamination)

		// Return mock response
		response := DetectionResponse{
			Success:           true,
			Anomalies:         []bool{false, true},
			ConfidenceScores:  []float64{0.1, 0.9},
			Algorithm:         "isolation_forest",
			TotalSamples:      2,
			AnomalyRate:       0.5,
			AnomaliesDetected: 1,
			ProcessingTime:    0.123,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	// Create client
	config := &ClientConfig{
		BaseURL: server.URL,
		APIKey:  "test-key",
	}
	client := NewClient(config)

	// Test request
	ctx := context.Background()
	request := &DetectionRequest{
		Data: [][]float64{
			{1.0, 2.0},
			{10.0, 20.0},
		},
		Algorithm:     "isolation_forest",
		Contamination: floatPtr(0.1),
	}

	response, err := client.DetectAnomalies(ctx, request)
	require.NoError(t, err)

	assert.True(t, response.Success)
	assert.Equal(t, []bool{false, true}, response.Anomalies)
	assert.Equal(t, 2, response.TotalSamples)
	assert.Equal(t, 1, response.AnomaliesDetected)
	assert.Equal(t, 0.5, response.AnomalyRate)
	assert.Equal(t, "isolation_forest", response.Algorithm)
}

func TestClient_DetectAnomalies_ValidationErrors(t *testing.T) {
	client := NewClient(nil)
	ctx := context.Background()

	tests := []struct {
		name    string
		request *DetectionRequest
		wantErr string
	}{
		{
			name:    "nil request",
			request: nil,
			wantErr: "detection request cannot be nil",
		},
		{
			name: "empty data",
			request: &DetectionRequest{
				Data:      [][]float64{},
				Algorithm: "isolation_forest",
			},
			wantErr: "data cannot be empty",
		},
		{
			name: "missing algorithm",
			request: &DetectionRequest{
				Data: [][]float64{{1.0, 2.0}},
			},
			wantErr: "algorithm must be specified",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := client.DetectAnomalies(ctx, tt.request)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
			assert.True(t, IsValidationError(err))
		})
	}
}

func TestClient_DetectAnomaliesEnsemble(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/v1/ensemble", r.URL.Path)

		var request EnsembleRequest
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		assert.Equal(t, []string{"isolation_forest", "one_class_svm"}, request.Algorithms)
		assert.Equal(t, "voting", request.Method)

		response := EnsembleResponse{
			Success:          true,
			Anomalies:        []bool{false, true},
			ConfidenceScores: []float64{0.2, 0.8},
			IndividualResults: map[string][]bool{
				"isolation_forest": {false, true},
				"one_class_svm":    {false, true},
			},
			Method:         "voting",
			Algorithms:     []string{"isolation_forest", "one_class_svm"},
			TotalSamples:   2,
			AnomalyRate:    0.5,
			ProcessingTime: 0.456,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(&ClientConfig{BaseURL: server.URL})
	ctx := context.Background()

	request := &EnsembleRequest{
		Data:       [][]float64{{1.0, 2.0}, {10.0, 20.0}},
		Algorithms: []string{"isolation_forest", "one_class_svm"},
		Method:     "voting",
	}

	response, err := client.DetectAnomaliesEnsemble(ctx, request)
	require.NoError(t, err)

	assert.True(t, response.Success)
	assert.Equal(t, []bool{false, true}, response.Anomalies)
	assert.Equal(t, "voting", response.Method)
	assert.Len(t, response.IndividualResults, 2)
	assert.Contains(t, response.IndividualResults, "isolation_forest")
	assert.Contains(t, response.IndividualResults, "one_class_svm")
}

func TestClient_TrainModel(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/v1/models/train", r.URL.Path)

		var request TrainingRequest
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		assert.Equal(t, "isolation_forest", request.Algorithm)
		assert.Equal(t, "test_model", request.ModelName)

		response := TrainingResponse{
			Success:         true,
			ModelID:         "model_123",
			ModelName:       "test_model",
			Algorithm:       "isolation_forest",
			TrainingTime:    12.34,
			ValidationScore: 0.95,
			Parameters:      map[string]interface{}{"n_estimators": 100.0},
			CreatedAt:       time.Now(),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(&ClientConfig{BaseURL: server.URL})
	ctx := context.Background()

	request := &TrainingRequest{
		Data:      [][]float64{{1.0, 2.0}, {1.1, 2.1}, {0.9, 1.9}},
		Algorithm: "isolation_forest",
		ModelName: "test_model",
		Parameters: map[string]interface{}{
			"n_estimators": 100,
		},
	}

	response, err := client.TrainModel(ctx, request)
	require.NoError(t, err)

	assert.True(t, response.Success)
	assert.Equal(t, "model_123", response.ModelID)
	assert.Equal(t, "test_model", response.ModelName)
	assert.Equal(t, "isolation_forest", response.Algorithm)
	assert.Equal(t, 12.34, response.TrainingTime)
	assert.Equal(t, 0.95, response.ValidationScore)
}

func TestClient_ListModels(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/api/v1/models", r.URL.Path)

		// Check query parameters
		assert.Equal(t, "1", r.URL.Query().Get("page"))
		assert.Equal(t, "10", r.URL.Query().Get("per_page"))

		response := ModelsListResponse{
			Success: true,
			Models: []Model{
				{
					ID:        "model_1",
					Name:      "test_model_1",
					Algorithm: "isolation_forest",
					Status:    "ready",
					CreatedAt: time.Now(),
					UpdatedAt: time.Now(),
				},
				{
					ID:        "model_2",
					Name:      "test_model_2",
					Algorithm: "one_class_svm",
					Status:    "training",
					CreatedAt: time.Now(),
					UpdatedAt: time.Now(),
				},
			},
			Total:   2,
			Page:    1,
			PerPage: 10,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(&ClientConfig{BaseURL: server.URL})
	ctx := context.Background()

	response, err := client.ListModels(ctx, 1, 10)
	require.NoError(t, err)

	assert.True(t, response.Success)
	assert.Len(t, response.Models, 2)
	assert.Equal(t, 2, response.Total)
	assert.Equal(t, 1, response.Page)
	assert.Equal(t, 10, response.PerPage)

	// Check first model
	model1 := response.Models[0]
	assert.Equal(t, "model_1", model1.ID)
	assert.Equal(t, "test_model_1", model1.Name)
	assert.Equal(t, "isolation_forest", model1.Algorithm)
	assert.Equal(t, "ready", model1.Status)
}

func TestClient_GetModel(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/api/v1/models/model_123", r.URL.Path)

		model := Model{
			ID:              "model_123",
			Name:            "test_model",
			Algorithm:       "isolation_forest",
			Parameters:      map[string]interface{}{"n_estimators": 100.0},
			TrainingScore:   0.98,
			ValidationScore: 0.95,
			Status:          "ready",
			CreatedAt:       time.Now(),
			UpdatedAt:       time.Now(),
			Version:         "1.0.0",
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(model)
	}))
	defer server.Close()

	client := NewClient(&ClientConfig{BaseURL: server.URL})
	ctx := context.Background()

	model, err := client.GetModel(ctx, "model_123")
	require.NoError(t, err)

	assert.Equal(t, "model_123", model.ID)
	assert.Equal(t, "test_model", model.Name)
	assert.Equal(t, "isolation_forest", model.Algorithm)
	assert.Equal(t, "ready", model.Status)
	assert.Equal(t, 0.98, model.TrainingScore)
	assert.Equal(t, 0.95, model.ValidationScore)
}

func TestClient_DeleteModel(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "DELETE", r.Method)
		assert.Equal(t, "/api/v1/models/model_123", r.URL.Path)

		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	client := NewClient(&ClientConfig{BaseURL: server.URL})
	ctx := context.Background()

	err := client.DeleteModel(ctx, "model_123")
	assert.NoError(t, err)
}

func TestClient_ListAlgorithms(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/api/v1/algorithms", r.URL.Path)

		response := AlgorithmsResponse{
			Success: true,
			Algorithms: []Algorithm{
				{
					Name:        "isolation_forest",
					DisplayName: "Isolation Forest",
					Description: "Tree-based anomaly detection algorithm",
					Type:        "ml",
					Complexity:  "medium",
					SuitableFor: []string{"tabular", "multivariate"},
					Parameters: map[string]interface{}{
						"n_estimators":  100,
						"contamination": 0.1,
					},
				},
				{
					Name:        "one_class_svm",
					DisplayName: "One-Class SVM",
					Description: "Support Vector Machine for outlier detection",
					Type:        "ml",
					Complexity:  "high",
					SuitableFor: []string{"tabular", "high_dimensional"},
					Parameters: map[string]interface{}{
						"nu":     0.05,
						"kernel": "rbf",
					},
				},
			},
			Categories: map[string][]string{
				"ml": {"isolation_forest", "one_class_svm"},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(&ClientConfig{BaseURL: server.URL})
	ctx := context.Background()

	response, err := client.ListAlgorithms(ctx)
	require.NoError(t, err)

	assert.True(t, response.Success)
	assert.Len(t, response.Algorithms, 2)
	assert.Contains(t, response.Categories, "ml")
	assert.Equal(t, []string{"isolation_forest", "one_class_svm"}, response.Categories["ml"])

	// Check first algorithm
	algo := response.Algorithms[0]
	assert.Equal(t, "isolation_forest", algo.Name)
	assert.Equal(t, "Isolation Forest", algo.DisplayName)
	assert.Equal(t, "ml", algo.Type)
	assert.Equal(t, "medium", algo.Complexity)
}

func TestClient_HealthCheck(t *testing.T) {
	// Mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/health", r.URL.Path)

		response := HealthResponse{
			Status:    "healthy",
			Version:   "1.0.0",
			Timestamp: time.Now(),
			Components: map[string]ComponentHealth{
				"database": {
					Status:      "healthy",
					Message:     "Connected",
					LastChecked: time.Now(),
				},
				"redis": {
					Status:      "healthy",
					Message:     "Connected",
					LastChecked: time.Now(),
				},
			},
			Uptime: 3600.0,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(&ClientConfig{BaseURL: server.URL})
	ctx := context.Background()

	response, err := client.HealthCheck(ctx)
	require.NoError(t, err)

	assert.Equal(t, "healthy", response.Status)
	assert.Equal(t, "1.0.0", response.Version)
	assert.Equal(t, 3600.0, response.Uptime)
	assert.Len(t, response.Components, 2)
	assert.Contains(t, response.Components, "database")
	assert.Contains(t, response.Components, "redis")
}

func TestClient_ErrorHandling(t *testing.T) {
	tests := []struct {
		name           string
		statusCode     int
		errorResponse  ErrorResponse
		expectedError  func(error) bool
		expectedStatus int
	}{
		{
			name:       "authentication error",
			statusCode: http.StatusUnauthorized,
			errorResponse: ErrorResponse{
				Success:   false,
				Error:     "Invalid API key",
				Code:      "INVALID_API_KEY",
				RequestID: "req_123",
			},
			expectedError: IsAuthenticationError,
		},
		{
			name:       "authorization error",
			statusCode: http.StatusForbidden,
			errorResponse: ErrorResponse{
				Success: false,
				Error:   "Insufficient permissions",
				Code:    "FORBIDDEN",
			},
			expectedError: IsAuthorizationError,
		},
		{
			name:       "validation error",
			statusCode: http.StatusBadRequest,
			errorResponse: ErrorResponse{
				Success: false,
				Error:   "Invalid request data",
				Code:    "VALIDATION_ERROR",
			},
			expectedError: IsValidationError,
		},
		{
			name:       "rate limit error",
			statusCode: http.StatusTooManyRequests,
			errorResponse: ErrorResponse{
				Success: false,
				Error:   "Rate limit exceeded",
				Code:    "RATE_LIMIT_EXCEEDED",
			},
			expectedError: IsRateLimitError,
		},
		{
			name:       "server error",
			statusCode: http.StatusInternalServerError,
			errorResponse: ErrorResponse{
				Success: false,
				Error:   "Internal server error",
				Code:    "INTERNAL_ERROR",
			},
			expectedError: IsServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Mock server that returns error
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.statusCode)
				json.NewEncoder(w).Encode(tt.errorResponse)
			}))
			defer server.Close()

			client := NewClient(&ClientConfig{BaseURL: server.URL})
			ctx := context.Background()

			request := &DetectionRequest{
				Data:      [][]float64{{1.0, 2.0}},
				Algorithm: "isolation_forest",
			}

			_, err := client.DetectAnomalies(ctx, request)
			require.Error(t, err)
			assert.True(t, tt.expectedError(err))
		})
	}
}

func TestClient_RetryLogic(t *testing.T) {
	requestCount := 0
	
	// Mock server that fails twice then succeeds
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		
		if requestCount < 3 {
			// Return server error for first two requests
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(ErrorResponse{
				Success: false,
				Error:   "Temporary server error",
				Code:    "TEMP_ERROR",
			})
			return
		}
		
		// Succeed on third request
		response := DetectionResponse{
			Success:           true,
			Anomalies:         []bool{false},
			Algorithm:         "isolation_forest",
			TotalSamples:      1,
			AnomaliesDetected: 0,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	client := NewClient(&ClientConfig{
		BaseURL:      server.URL,
		MaxRetries:   3,
		RetryBackoff: 10 * time.Millisecond, // Short backoff for testing
	})
	
	ctx := context.Background()
	request := &DetectionRequest{
		Data:      [][]float64{{1.0, 2.0}},
		Algorithm: "isolation_forest",
	}

	response, err := client.DetectAnomalies(ctx, request)
	require.NoError(t, err)
	assert.True(t, response.Success)
	assert.Equal(t, 3, requestCount) // Should have made 3 requests
}

// Helper function for tests
func floatPtr(f float64) *float64 {
	return &f
}