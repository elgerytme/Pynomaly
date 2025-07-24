package anomalydetection

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient(t *testing.T) {
	config := ClientConfig{
		BaseURL:    "http://localhost:8000",
		Timeout:    30 * time.Second,
		MaxRetries: 3,
	}

	client := NewClient(config)

	assert.NotNil(t, client)
	assert.Equal(t, "http://localhost:8000", client.config.BaseURL)
	assert.Equal(t, 30*time.Second, client.config.Timeout)
	assert.Equal(t, 3, client.config.MaxRetries)
}

func TestNewClientDefaults(t *testing.T) {
	config := ClientConfig{
		BaseURL: "http://localhost:8000",
	}

	client := NewClient(config)

	assert.Equal(t, 30*time.Second, client.config.Timeout)
	assert.Equal(t, 3, client.config.MaxRetries)
}

func setupTestServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

func TestDetectAnomalies(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/v1/detect", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var requestBody map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&requestBody)
		require.NoError(t, err)

		assert.Equal(t, "isolation_forest", requestBody["algorithm"])
		assert.Equal(t, false, requestBody["return_explanations"])

		response := DetectionResult{
			Anomalies: []AnomalyData{
				{
					Index:     2,
					Score:     0.85,
					DataPoint: []float64{10.0, 20.0},
				},
			},
			TotalPoints:   3,
			AnomalyCount:  1,
			AlgorithmUsed: IsolationForest,
			ExecutionTime: 0.15,
			Metadata:      map[string]interface{}{},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{
		BaseURL:    server.URL,
		Timeout:    10 * time.Second,
		MaxRetries: 1,
	}
	client := NewClient(config)

	data := [][]float64{
		{1.0, 2.0},
		{1.1, 2.1},
		{10.0, 20.0},
	}

	ctx := context.Background()
	result, err := client.DetectAnomalies(ctx, data, IsolationForest, nil, false)

	require.NoError(t, err)
	assert.Equal(t, 1, result.AnomalyCount)
	assert.Equal(t, 3, result.TotalPoints)
	assert.Len(t, result.Anomalies, 1)
	assert.Equal(t, 2, result.Anomalies[0].Index)
	assert.Equal(t, 0.85, result.Anomalies[0].Score)
}

func TestDetectAnomaliesEmptyData(t *testing.T) {
	config := ClientConfig{
		BaseURL: "http://localhost:8000",
	}
	client := NewClient(config)

	ctx := context.Background()
	_, err := client.DetectAnomalies(ctx, [][]float64{}, IsolationForest, nil, false)

	assert.Error(t, err)
	validationErr, ok := err.(*ValidationError)
	assert.True(t, ok)
	assert.Contains(t, validationErr.Message, "Data cannot be empty")
}

func TestDetectAnomaliesAPIError(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"detail": "Invalid algorithm",
		})
	})

	config := ClientConfig{
		BaseURL:    server.URL,
		MaxRetries: 0, // Don't retry for this test
	}
	client := NewClient(config)

	data := [][]float64{{1.0, 2.0}}
	ctx := context.Background()
	_, err := client.DetectAnomalies(ctx, data, IsolationForest, nil, false)

	assert.Error(t, err)
	apiError, ok := err.(*APIError)
	assert.True(t, ok)
	assert.Equal(t, 400, apiError.StatusCode)
	assert.Contains(t, apiError.Message, "Invalid algorithm")
}

func TestDetectAnomaliesTimeout(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond) // Simulate slow response
		w.WriteHeader(http.StatusOK)
	})

	config := ClientConfig{
		BaseURL:    server.URL,
		Timeout:    50 * time.Millisecond, // Short timeout
		MaxRetries: 0,
	}
	client := NewClient(config)

	data := [][]float64{{1.0, 2.0}}
	ctx := context.Background()
	_, err := client.DetectAnomalies(ctx, data, IsolationForest, nil, false)

	assert.Error(t, err)
	// Should be a timeout or connection error
	assert.True(t, err.(*ConnectionError) != nil || err.(*TimeoutError) != nil)
}

func TestBatchDetect(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/v1/batch/detect", r.URL.Path)

		response := DetectionResult{
			Anomalies:     []AnomalyData{},
			TotalPoints:   2,
			AnomalyCount:  0,
			AlgorithmUsed: IsolationForest,
			ExecutionTime: 0.1,
			Metadata:      map[string]interface{}{},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	request := BatchProcessingRequest{
		Data:      [][]float64{{1, 2}, {3, 4}},
		Algorithm: IsolationForest,
		Parameters: map[string]interface{}{
			"contamination": 0.1,
		},
		ReturnExplanations: true,
	}

	ctx := context.Background()
	result, err := client.BatchDetect(ctx, request)

	require.NoError(t, err)
	assert.Equal(t, 2, result.TotalPoints)
	assert.Equal(t, 0, result.AnomalyCount)
}

func TestTrainModel(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/v1/models/train", r.URL.Path)

		var requestBody TrainingRequest
		err := json.NewDecoder(r.Body).Decode(&requestBody)
		require.NoError(t, err)

		assert.Equal(t, IsolationForest, requestBody.Algorithm)
		assert.Equal(t, "test-model", *requestBody.ModelName)

		response := TrainingResult{
			ModelID:      "model-123",
			TrainingTime: 5.2,
			PerformanceMetrics: map[string]float64{
				"accuracy": 0.95,
			},
			ValidationMetrics: map[string]float64{
				"f1_score": 0.92,
			},
			ModelInfo: ModelInfo{
				ModelID:         "model-123",
				Algorithm:       IsolationForest,
				CreatedAt:       time.Now(),
				TrainingDataSize: 100,
				PerformanceMetrics: map[string]float64{
					"accuracy": 0.95,
				},
				Hyperparameters: map[string]interface{}{
					"n_estimators": 100,
				},
				Version: "1.0",
				Status:  "trained",
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	modelName := "test-model"
	request := TrainingRequest{
		Data:      [][]float64{{1, 2}, {3, 4}},
		Algorithm: IsolationForest,
		ModelName: &modelName,
	}

	ctx := context.Background()
	result, err := client.TrainModel(ctx, request)

	require.NoError(t, err)
	assert.Equal(t, "model-123", result.ModelID)
	assert.Equal(t, 5.2, result.TrainingTime)
	assert.Equal(t, 0.95, result.PerformanceMetrics["accuracy"])
}

func TestGetModel(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/api/v1/models/model-123", r.URL.Path)

		response := ModelInfo{
			ModelID:         "model-123",
			Algorithm:       IsolationForest,
			CreatedAt:       time.Now(),
			TrainingDataSize: 100,
			PerformanceMetrics: map[string]float64{
				"accuracy": 0.95,
			},
			Hyperparameters: map[string]interface{}{
				"n_estimators": 100,
			},
			Version: "1.0",
			Status:  "trained",
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	ctx := context.Background()
	result, err := client.GetModel(ctx, "model-123")

	require.NoError(t, err)
	assert.Equal(t, "model-123", result.ModelID)
	assert.Equal(t, IsolationForest, result.Algorithm)
	assert.Equal(t, 0.95, result.PerformanceMetrics["accuracy"])
}

func TestGetModelEmptyID(t *testing.T) {
	config := ClientConfig{BaseURL: "http://localhost:8000"}
	client := NewClient(config)

	ctx := context.Background()
	_, err := client.GetModel(ctx, "")

	assert.Error(t, err)
	validationErr, ok := err.(*ValidationError)
	assert.True(t, ok)
	assert.Contains(t, validationErr.Message, "Model ID is required")
}

func TestListModels(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/api/v1/models", r.URL.Path)

		response := struct {
			Models []ModelInfo `json:"models"`
		}{
			Models: []ModelInfo{
				{
					ModelID:         "model-1",
					Algorithm:       IsolationForest,
					CreatedAt:       time.Now(),
					TrainingDataSize: 100,
					PerformanceMetrics: map[string]float64{
						"accuracy": 0.95,
					},
					Hyperparameters: map[string]interface{}{
						"n_estimators": 100,
					},
					Version: "1.0",
					Status:  "trained",
				},
				{
					ModelID:         "model-2",
					Algorithm:       LocalOutlierFactor,
					CreatedAt:       time.Now(),
					TrainingDataSize: 50,
					PerformanceMetrics: map[string]float64{
						"accuracy": 0.88,
					},
					Hyperparameters: map[string]interface{}{
						"n_neighbors": 20,
					},
					Version: "1.0",
					Status:  "trained",
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	ctx := context.Background()
	result, err := client.ListModels(ctx)

	require.NoError(t, err)
	assert.Len(t, result, 2)
	assert.Equal(t, "model-1", result[0].ModelID)
	assert.Equal(t, "model-2", result[1].ModelID)
	assert.Equal(t, IsolationForest, result[0].Algorithm)
	assert.Equal(t, LocalOutlierFactor, result[1].Algorithm)
}

func TestDeleteModel(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "DELETE", r.Method)
		assert.Equal(t, "/api/v1/models/model-123", r.URL.Path)

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"message": "Model deleted successfully",
		})
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	ctx := context.Background()
	err := client.DeleteModel(ctx, "model-123")

	require.NoError(t, err)
}

func TestExplainAnomaly(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/v1/explain", r.URL.Path)

		var requestBody map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&requestBody)
		require.NoError(t, err)

		assert.Equal(t, []interface{}{10.0, 20.0}, requestBody["data_point"])
		assert.Equal(t, "shap", requestBody["method"])

		response := ExplanationResult{
			AnomalyIndex: 0,
			FeatureImportance: map[string]float64{
				"feature_0": 0.8,
				"feature_1": 0.2,
			},
			ShapValues:      []float64{0.3, 0.1},
			ExplanationText: "High values in feature 0",
			Confidence:      0.9,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	options := ExplainOptions{
		Method: "shap",
	}

	ctx := context.Background()
	result, err := client.ExplainAnomaly(ctx, []float64{10.0, 20.0}, options)

	require.NoError(t, err)
	assert.Equal(t, 0, result.AnomalyIndex)
	assert.Equal(t, 0.8, result.FeatureImportance["feature_0"])
	assert.Contains(t, result.ExplanationText, "feature 0")
	assert.Equal(t, 0.9, result.Confidence)
}

func TestGetHealth(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/api/v1/health", r.URL.Path)

		response := HealthStatus{
			Status:    "healthy",
			Timestamp: time.Now(),
			Version:   "1.0.0",
			Uptime:    3600.5,
			Components: map[string]string{
				"database": "healthy",
				"cache":    "healthy",
			},
			Metrics: map[string]interface{}{
				"requests_per_second": 100,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	ctx := context.Background()
	result, err := client.GetHealth(ctx)

	require.NoError(t, err)
	assert.Equal(t, "healthy", result.Status)
	assert.Equal(t, "1.0.0", result.Version)
	assert.Equal(t, 3600.5, result.Uptime)
	assert.Equal(t, "healthy", result.Components["database"])
}

func TestGetMetrics(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, "/api/v1/metrics", r.URL.Path)

		response := map[string]interface{}{
			"requests_per_second":   100,
			"average_response_time": 0.05,
			"active_connections":    25,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	ctx := context.Background()
	result, err := client.GetMetrics(ctx)

	require.NoError(t, err)
	assert.Equal(t, float64(100), result["requests_per_second"])
	assert.Equal(t, 0.05, result["average_response_time"])
}

func TestUploadData(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/api/v1/data/upload", r.URL.Path)

		var requestBody map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&requestBody)
		require.NoError(t, err)

		assert.Equal(t, "test-dataset", requestBody["name"])
		assert.Equal(t, "Test dataset", requestBody["description"])

		response := UploadResult{
			DatasetID: "dataset-123",
			Message:   "Data uploaded successfully",
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	data := [][]float64{{1, 2}, {3, 4}}
	description := "Test dataset"

	ctx := context.Background()
	result, err := client.UploadData(ctx, data, "test-dataset", &description)

	require.NoError(t, err)
	assert.Equal(t, "dataset-123", result.DatasetID)
	assert.Equal(t, "Data uploaded successfully", result.Message)
}

func TestUploadDataEmptyData(t *testing.T) {
	config := ClientConfig{BaseURL: "http://localhost:8000"}
	client := NewClient(config)

	ctx := context.Background()
	_, err := client.UploadData(ctx, [][]float64{}, "test", nil)

	assert.Error(t, err)
	validationErr, ok := err.(*ValidationError)
	assert.True(t, ok)
	assert.Contains(t, validationErr.Message, "Data cannot be empty")
}

func TestUploadDataEmptyName(t *testing.T) {
	config := ClientConfig{BaseURL: "http://localhost:8000"}
	client := NewClient(config)

	data := [][]float64{{1, 2}}
	ctx := context.Background()
	_, err := client.UploadData(ctx, data, "", nil)

	assert.Error(t, err)
	validationErr, ok := err.(*ValidationError)
	assert.True(t, ok)
	assert.Contains(t, validationErr.Message, "Dataset name is required")
}

func TestRetryLogic(t *testing.T) {
	attemptCount := 0
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		attemptCount++
		if attemptCount < 3 {
			// Fail first two attempts
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		// Succeed on third attempt
		response := DetectionResult{
			Anomalies:     []AnomalyData{},
			TotalPoints:   1,
			AnomalyCount:  0,
			AlgorithmUsed: IsolationForest,
			ExecutionTime: 0.1,
			Metadata:      map[string]interface{}{},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	config := ClientConfig{
		BaseURL:    server.URL,
		MaxRetries: 3,
	}
	client := NewClient(config)

	data := [][]float64{{1.0, 2.0}}
	ctx := context.Background()
	result, err := client.DetectAnomalies(ctx, data, IsolationForest, nil, false)

	require.NoError(t, err)
	assert.Equal(t, 3, attemptCount) // Should have tried 3 times
	assert.Equal(t, 1, result.TotalPoints)
}

func TestMaxRetriesExceeded(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		// Always fail
		w.WriteHeader(http.StatusInternalServerError)
	})

	config := ClientConfig{
		BaseURL:    server.URL,
		MaxRetries: 2,
	}
	client := NewClient(config)

	data := [][]float64{{1.0, 2.0}}
	ctx := context.Background()
	_, err := client.DetectAnomalies(ctx, data, IsolationForest, nil, false)

	assert.Error(t, err)
	apiError, ok := err.(*APIError)
	assert.True(t, ok)
	assert.Equal(t, 500, apiError.StatusCode)
}

func TestClientWithAPIKey(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		assert.Equal(t, "Bearer test-api-key", authHeader)

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "healthy",
		})
	})

	apiKey := "test-api-key"
	config := ClientConfig{
		BaseURL: server.URL,
		APIKey:  &apiKey,
	}
	client := NewClient(config)

	ctx := context.Background()
	_, err := client.GetHealth(ctx)

	require.NoError(t, err)
}

func TestClientWithCustomHeaders(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		customHeader := r.Header.Get("X-Custom-Header")
		assert.Equal(t, "custom-value", customHeader)

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "healthy",
		})
	})

	config := ClientConfig{
		BaseURL: server.URL,
		Headers: map[string]string{
			"X-Custom-Header": "custom-value",
		},
	}
	client := NewClient(config)

	ctx := context.Background()
	_, err := client.GetHealth(ctx)

	require.NoError(t, err)
}

func TestContextCancellation(t *testing.T) {
	server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		// Simulate slow response
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	})

	config := ClientConfig{BaseURL: server.URL}
	client := NewClient(config)

	// Cancel context after 50ms
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	data := [][]float64{{1.0, 2.0}}
	_, err := client.DetectAnomalies(ctx, data, IsolationForest, nil, false)

	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "context deadline exceeded") ||
		strings.Contains(err.Error(), "timeout"))
}

func TestClientErrorTypes(t *testing.T) {
	testCases := []struct {
		name         string
		statusCode   int
		expectedType interface{}
	}{
		{"BadRequest", 400, &APIError{}},
		{"Unauthorized", 401, &APIError{}},
		{"NotFound", 404, &APIError{}},
		{"TooManyRequests", 429, &APIError{}},
		{"InternalServerError", 500, &APIError{}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server := setupTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.statusCode)
				json.NewEncoder(w).Encode(map[string]string{
					"detail": "Error message",
				})
			})

			config := ClientConfig{
				BaseURL:    server.URL,
				MaxRetries: 0, // Don't retry
			}
			client := NewClient(config)

			data := [][]float64{{1.0, 2.0}}
			ctx := context.Background()
			_, err := client.DetectAnomalies(ctx, data, IsolationForest, nil, false)

			require.Error(t, err)
			assert.IsType(t, tc.expectedType, err)

			if apiErr, ok := err.(*APIError); ok {
				assert.Equal(t, tc.statusCode, apiErr.StatusCode)
			}
		})
	}
}