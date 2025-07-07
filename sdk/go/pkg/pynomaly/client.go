// Package pynomaly provides a Go client for the Pynomaly anomaly detection platform.
package pynomaly

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Client represents the main Pynomaly API client
type Client struct {
	baseURL    string
	apiKey     string
	tenantID   string
	httpClient *http.Client
	userAgent  string

	// Sub-clients for different API areas
	Detection    *DetectionClient
	Streaming    *StreamingClient
	ABTesting    *ABTestingClient
	Users        *UserManagementClient
	Compliance   *ComplianceClient
}

// Config holds configuration for the Pynomaly client
type Config struct {
	BaseURL     string
	APIKey      string
	TenantID    string
	Timeout     time.Duration
	UserAgent   string
	HTTPClient  *http.Client
}

// NewClient creates a new Pynomaly client with the given configuration
func NewClient(config Config) *Client {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.pynomaly.ai"
	}
	
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	
	if config.UserAgent == "" {
		config.UserAgent = "pynomaly-go/1.0.0"
	}
	
	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: config.Timeout,
		}
	}

	client := &Client{
		baseURL:    strings.TrimSuffix(config.BaseURL, "/"),
		apiKey:     config.APIKey,
		tenantID:   config.TenantID,
		httpClient: httpClient,
		userAgent:  config.UserAgent,
	}

	// Initialize sub-clients
	client.Detection = &DetectionClient{client: client}
	client.Streaming = &StreamingClient{client: client}
	client.ABTesting = &ABTestingClient{client: client}
	client.Users = &UserManagementClient{client: client}
	client.Compliance = &ComplianceClient{client: client}

	return client
}

// SetAuth updates the authentication credentials
func (c *Client) SetAuth(apiKey, tenantID string) {
	c.apiKey = apiKey
	c.tenantID = tenantID
}

// ClearAuth clears authentication credentials
func (c *Client) ClearAuth() {
	c.apiKey = ""
	c.tenantID = ""
}

// makeRequest performs an HTTP request with authentication and error handling
func (c *Client) makeRequest(ctx context.Context, method, path string, body interface{}, result interface{}) error {
	url := c.baseURL + path
	
	var bodyReader io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	
	if c.tenantID != "" {
		req.Header.Set("X-Tenant-ID", c.tenantID)
	}

	// Add request ID for tracing
	req.Header.Set("X-Request-ID", generateRequestID())
	req.Header.Set("X-Request-Timestamp", time.Now().UTC().Format(time.RFC3339))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return &NetworkError{Message: fmt.Sprintf("request failed: %v", err)}
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return c.handleErrorResponse(resp)
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}

// makeRequestWithQuery performs an HTTP request with query parameters
func (c *Client) makeRequestWithQuery(ctx context.Context, method, path string, params map[string]string, result interface{}) error {
	if len(params) > 0 {
		u, err := url.Parse(c.baseURL + path)
		if err != nil {
			return fmt.Errorf("failed to parse URL: %w", err)
		}
		
		q := u.Query()
		for key, value := range params {
			q.Add(key, value)
		}
		u.RawQuery = q.Encode()
		path = u.RequestURI()
	}
	
	return c.makeRequest(ctx, method, path, nil, result)
}

// uploadFile uploads a file and returns the file ID
func (c *Client) UploadFile(ctx context.Context, filename string, content []byte, contentType string) (*UploadResponse, error) {
	// Implementation would depend on the actual upload API
	// This is a placeholder for the file upload functionality
	
	var result UploadResponse
	// For now, return a mock response
	result = UploadResponse{
		FileID: "mock-file-id",
		URL:    c.baseURL + "/files/mock-file-id",
	}
	
	return &result, nil
}

// downloadFile downloads a file by ID
func (c *Client) DownloadFile(ctx context.Context, fileID string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/download/"+fileID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	
	if c.tenantID != "" {
		req.Header.Set("X-Tenant-ID", c.tenantID)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, &NetworkError{Message: fmt.Sprintf("download failed: %v", err)}
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, c.handleErrorResponse(resp)
	}

	return io.ReadAll(resp.Body)
}

// HealthCheck performs a health check against the API
func (c *Client) HealthCheck(ctx context.Context) (*HealthStatus, error) {
	var result HealthStatus
	err := c.makeRequest(ctx, "GET", "/health", nil, &result)
	return &result, err
}

// GetVersion returns API version information
func (c *Client) GetVersion(ctx context.Context) (*VersionInfo, error) {
	var result VersionInfo
	err := c.makeRequest(ctx, "GET", "/version", nil, &result)
	return &result, err
}

// TestAuth tests authentication credentials
func (c *Client) TestAuth(ctx context.Context) (*AuthTestResponse, error) {
	var result AuthTestResponse
	err := c.makeRequest(ctx, "GET", "/auth/test", nil, &result)
	return &result, err
}

// GetAvailableAlgorithms returns list of available algorithms
func (c *Client) GetAvailableAlgorithms(ctx context.Context) ([]string, error) {
	var result struct {
		Algorithms []string `json:"algorithms"`
	}
	err := c.makeRequest(ctx, "GET", "/algorithms", nil, &result)
	return result.Algorithms, err
}

// GetAlgorithmInfo returns information about a specific algorithm
func (c *Client) GetAlgorithmInfo(ctx context.Context, algorithm string) (*AlgorithmInfo, error) {
	var result AlgorithmInfo
	err := c.makeRequest(ctx, "GET", "/algorithms/"+algorithm, nil, &result)
	return &result, err
}

// handleErrorResponse converts HTTP error responses to appropriate error types
func (c *Client) handleErrorResponse(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)
	
	var errorResp ErrorResponse
	if err := json.Unmarshal(body, &errorResp); err != nil {
		// Fallback to basic error if we can't parse the response
		return &PynomalyError{
			Code:       "HTTP_ERROR",
			Message:    fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(body)),
			StatusCode: resp.StatusCode,
		}
	}

	switch resp.StatusCode {
	case 400:
		return &ValidationError{
			PynomalyError: PynomalyError{
				Code:       errorResp.Code,
				Message:    errorResp.Message,
				StatusCode: resp.StatusCode,
				Details:    errorResp.Details,
			},
		}
	case 401:
		return &AuthenticationError{
			PynomalyError: PynomalyError{
				Code:       errorResp.Code,
				Message:    errorResp.Message,
				StatusCode: resp.StatusCode,
				Details:    errorResp.Details,
			},
		}
	case 403:
		return &AuthorizationError{
			PynomalyError: PynomalyError{
				Code:       errorResp.Code,
				Message:    errorResp.Message,
				StatusCode: resp.StatusCode,
				Details:    errorResp.Details,
			},
		}
	case 404:
		return &NotFoundError{
			PynomalyError: PynomalyError{
				Code:       errorResp.Code,
				Message:    errorResp.Message,
				StatusCode: resp.StatusCode,
				Details:    errorResp.Details,
			},
		}
	case 409:
		return &ConflictError{
			PynomalyError: PynomalyError{
				Code:       errorResp.Code,
				Message:    errorResp.Message,
				StatusCode: resp.StatusCode,
				Details:    errorResp.Details,
			},
		}
	case 429:
		return &RateLimitError{
			PynomalyError: PynomalyError{
				Code:       errorResp.Code,
				Message:    errorResp.Message,
				StatusCode: resp.StatusCode,
				Details:    errorResp.Details,
			},
		}
	case 500, 502, 503, 504:
		return &ServerError{
			PynomalyError: PynomalyError{
				Code:       errorResp.Code,
				Message:    errorResp.Message,
				StatusCode: resp.StatusCode,
				Details:    errorResp.Details,
			},
		}
	default:
		return &PynomalyError{
			Code:       errorResp.Code,
			Message:    errorResp.Message,
			StatusCode: resp.StatusCode,
			Details:    errorResp.Details,
		}
	}
}

// generateRequestID generates a unique request ID for tracing
func generateRequestID() string {
	return fmt.Sprintf("req_%d_%d", time.Now().UnixNano(), time.Now().UnixNano()%1000000)
}