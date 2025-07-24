package anomalydetection

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client represents the anomaly detection API client
type Client struct {
	config     ClientConfig
	httpClient *http.Client
}

// NewClient creates a new anomaly detection client
func NewClient(config ClientConfig) *Client {
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}

	client := &Client{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}

	return client
}

// makeRequest makes HTTP request with error handling and retries
func (c *Client) makeRequest(ctx context.Context, method, endpoint string, data interface{}, params map[string]string) ([]byte, error) {
	var body io.Reader
	if data != nil {
		jsonData, err := json.Marshal(data)
		if err != nil {
			return nil, &ValidationError{
				SDKError: &SDKError{
					Message: fmt.Sprintf("Failed to marshal request data: %v", err),
					Code:    stringPtr("MARSHAL_ERROR"),
				},
			}
		}
		body = bytes.NewReader(jsonData)
	}

	url := strings.TrimSuffix(c.config.BaseURL, "/") + endpoint
	
	var lastErr error
	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			backoff := time.Second * time.Duration(1<<attempt)
			if backoff > 10*time.Second {
				backoff = 10 * time.Second
			}
			
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		req, err := http.NewRequestWithContext(ctx, method, url, body)
		if err != nil {
			lastErr = &ValidationError{
				SDKError: &SDKError{
					Message: fmt.Sprintf("Failed to create request: %v", err),
					Code:    stringPtr("REQUEST_ERROR"),
				},
			}
			continue
		}

		// Set headers
		req.Header.Set("Content-Type", "application/json")
		if c.config.APIKey != nil {
			req.Header.Set("Authorization", "Bearer "+*c.config.APIKey)
		}
		for key, value := range c.config.Headers {
			req.Header.Set(key, value)
		}

		// Add query parameters
		if params != nil {
			q := req.URL.Query()
			for key, value := range params {
				q.Add(key, value)
			}
			req.URL.RawQuery = q.Encode()
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = &ConnectionError{
				SDKError: &SDKError{
					Message: fmt.Sprintf("Request failed: %v", err),
					Code:    stringPtr("CONNECTION_ERROR"),
				},
				URL: &url,
			}
			continue
		}

		responseBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		
		if err != nil {
			lastErr = &ConnectionError{
				SDKError: &SDKError{
					Message: fmt.Sprintf("Failed to read response body: %v", err),
					Code:    stringPtr("READ_ERROR"),
				},
			}
			continue
		}

		if resp.StatusCode >= 400 {
			var errorData map[string]interface{}
			if err := json.Unmarshal(responseBody, &errorData); err != nil {
				errorData = map[string]interface{}{"detail": string(responseBody)}
			}

			message := "Unknown error"
			if detail, ok := errorData["detail"].(string); ok {
				message = detail
			}

			lastErr = &APIError{
				SDKError: &SDKError{
					Message: message,
					Code:    stringPtr(fmt.Sprintf("HTTP_%d", resp.StatusCode)),
					Details: errorData,
				},
				StatusCode:   resp.StatusCode,
				ResponseData: errorData,
			}

			// Don't retry client errors (4xx)
			if resp.StatusCode >= 400 && resp.StatusCode < 500 {
				break
			}
			continue
		}

		return responseBody, nil
	}

	return nil, lastErr
}

// DetectAnomalies detects anomalies in the provided data
func (c *Client) DetectAnomalies(ctx context.Context, data [][]float64, algorithm AlgorithmType, parameters map[string]interface{}, returnExplanations bool) (*DetectionResult, error) {
	if len(data) == 0 {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Data cannot be empty",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("data"),
			Value: data,
		}
	}

	requestData := map[string]interface{}{
		"data":                data,
		"algorithm":           algorithm,
		"parameters":          parameters,
		"return_explanations": returnExplanations,
	}

	if parameters == nil {
		requestData["parameters"] = map[string]interface{}{}
	}

	responseBody, err := c.makeRequest(ctx, "POST", "/api/v1/detect", requestData, nil)
	if err != nil {
		return nil, err
	}

	var result DetectionResult
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return &result, nil
}

// BatchDetect processes a batch detection request
func (c *Client) BatchDetect(ctx context.Context, request BatchProcessingRequest) (*DetectionResult, error) {
	responseBody, err := c.makeRequest(ctx, "POST", "/api/v1/batch/detect", request, nil)
	if err != nil {
		return nil, err
	}

	var result DetectionResult
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return &result, nil
}

// TrainModel trains a new anomaly detection model
func (c *Client) TrainModel(ctx context.Context, request TrainingRequest) (*TrainingResult, error) {
	responseBody, err := c.makeRequest(ctx, "POST", "/api/v1/models/train", request, nil)
	if err != nil {
		return nil, err
	}

	var result TrainingResult
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return &result, nil
}

// GetModel gets information about a specific model
func (c *Client) GetModel(ctx context.Context, modelID string) (*ModelInfo, error) {
	if modelID == "" {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Model ID is required",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("modelId"),
			Value: modelID,
		}
	}

	responseBody, err := c.makeRequest(ctx, "GET", fmt.Sprintf("/api/v1/models/%s", modelID), nil, nil)
	if err != nil {
		return nil, err
	}

	var result ModelInfo
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return &result, nil
}

// ListModels lists all available models
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	responseBody, err := c.makeRequest(ctx, "GET", "/api/v1/models", nil, nil)
	if err != nil {
		return nil, err
	}

	var response struct {
		Models []ModelInfo `json:"models"`
	}
	if err := json.Unmarshal(responseBody, &response); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return response.Models, nil
}

// DeleteModel deletes a model
func (c *Client) DeleteModel(ctx context.Context, modelID string) error {
	if modelID == "" {
		return &ValidationError{
			SDKError: &SDKError{
				Message: "Model ID is required",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("modelId"),
			Value: modelID,
		}
	}

	_, err := c.makeRequest(ctx, "DELETE", fmt.Sprintf("/api/v1/models/%s", modelID), nil, nil)
	return err
}

// ExplainAnomaly gets explanation for why a data point is anomalous
func (c *Client) ExplainAnomaly(ctx context.Context, dataPoint []float64, options ExplainOptions) (*ExplanationResult, error) {
	if len(dataPoint) == 0 {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Data point cannot be empty",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("dataPoint"),
			Value: dataPoint,
		}
	}

	requestData := map[string]interface{}{
		"data_point": dataPoint,
		"method":     options.Method,
	}

	if options.Method == "" {
		requestData["method"] = "shap"
	}

	if options.ModelID != nil {
		requestData["model_id"] = *options.ModelID
	}
	if options.Algorithm != nil {
		requestData["algorithm"] = *options.Algorithm
	}

	responseBody, err := c.makeRequest(ctx, "POST", "/api/v1/explain", requestData, nil)
	if err != nil {
		return nil, err
	}

	var result ExplanationResult
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return &result, nil
}

// GetHealth gets service health status
func (c *Client) GetHealth(ctx context.Context) (*HealthStatus, error) {
	responseBody, err := c.makeRequest(ctx, "GET", "/api/v1/health", nil, nil)
	if err != nil {
		return nil, err
	}

	var result HealthStatus
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return &result, nil
}

// GetMetrics gets service metrics
func (c *Client) GetMetrics(ctx context.Context) (map[string]interface{}, error) {
	responseBody, err := c.makeRequest(ctx, "GET", "/api/v1/metrics", nil, nil)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return result, nil
}

// UploadData uploads training data to the service
func (c *Client) UploadData(ctx context.Context, data [][]float64, datasetName string, description *string) (*UploadResult, error) {
	if len(data) == 0 {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Data cannot be empty",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("data"),
			Value: data,
		}
	}

	if datasetName == "" {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: "Dataset name is required",
				Code:    stringPtr("VALIDATION_ERROR"),
			},
			Field: stringPtr("datasetName"),
			Value: datasetName,
		}
	}

	requestData := map[string]interface{}{
		"data": data,
		"name": datasetName,
	}
	if description != nil {
		requestData["description"] = *description
	}

	responseBody, err := c.makeRequest(ctx, "POST", "/api/v1/data/upload", requestData, nil)
	if err != nil {
		return nil, err
	}

	var result UploadResult
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, &ValidationError{
			SDKError: &SDKError{
				Message: fmt.Sprintf("Failed to unmarshal response: %v", err),
				Code:    stringPtr("UNMARSHAL_ERROR"),
			},
		}
	}

	return &result, nil
}

// ExplainOptions represents options for anomaly explanation
type ExplainOptions struct {
	ModelID   *string
	Algorithm *AlgorithmType
	Method    string
}

// UploadResult represents the result of data upload
type UploadResult struct {
	DatasetID string `json:"dataset_id"`
	Message   string `json:"message"`
}

// Helper function to create string pointer
func stringPtr(s string) *string {
	return &s
}