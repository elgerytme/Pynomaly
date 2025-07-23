package anomaly_detection

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// Client represents the anomaly detection API client
type Client struct {
	config      *ClientConfig
	httpClient  *http.Client
	authManager *AuthManager
	rateLimiter *RateLimiter
	wsConn      *websocket.Conn
	wsMutex     sync.Mutex
}

// NewClient creates a new anomaly detection client with the provided configuration
func NewClient(config *ClientConfig) *Client {
	if config == nil {
		config = DefaultConfig()
	}

	// Apply defaults if not provided
	if config.BaseURL == "" {
		config.BaseURL = "https://api.anomaly-detection.com"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	if config.RetryBackoff == 0 {
		config.RetryBackoff = time.Second
	}
	if config.RateLimit == 0 {
		config.RateLimit = 100 // requests per minute
	}
	if config.UserAgent == "" {
		config.UserAgent = "anomaly-detection-go-sdk/1.0.0"
	}
	if config.StreamingURL == "" {
		config.StreamingURL = "wss://api.anomaly-detection.com/ws"
	}

	httpClient := &http.Client{
		Timeout: config.Timeout,
	}

	authManager := NewAuthManager(config.BaseURL, config.APIKey, httpClient)
	if config.JWTToken != "" {
		authManager.SetTokens(config.JWTToken, "", time.Now().Add(24*time.Hour))
	}

	rateLimiter := NewRateLimiter(config.RateLimit, time.Minute)

	return &Client{
		config:      config,
		httpClient:  httpClient,
		authManager: authManager,
		rateLimiter: rateLimiter,
	}
}

// DefaultConfig returns a default client configuration
func DefaultConfig() *ClientConfig {
	return &ClientConfig{
		BaseURL:         "https://api.anomaly-detection.com",
		Timeout:         30 * time.Second,
		MaxRetries:      3,
		RetryBackoff:    time.Second,
		RateLimit:       100,
		UserAgent:       "anomaly-detection-go-sdk/1.0.0",
		EnableStreaming: false,
		StreamingURL:    "wss://api.anomaly-detection.com/ws",
		Debug:           false,
	}
}

// AuthenticateWithAPIKey authenticates the client using an API key
func (c *Client) AuthenticateWithAPIKey(ctx context.Context, apiKey string) error {
	return c.authManager.AuthenticateWithAPIKey(ctx, apiKey)
}

// AuthenticateWithCredentials authenticates the client using username and password
func (c *Client) AuthenticateWithCredentials(ctx context.Context, username, password string) error {
	return c.authManager.AuthenticateWithCredentials(ctx, username, password)
}

// Logout logs out the client and clears authentication state
func (c *Client) Logout(ctx context.Context) error {
	return c.authManager.Logout(ctx)
}

// DetectAnomalies performs anomaly detection on the provided data
func (c *Client) DetectAnomalies(ctx context.Context, request *DetectionRequest) (*DetectionResponse, error) {
	if request == nil {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "detection request cannot be nil",
				Code:    "INVALID_REQUEST",
			},
		}
	}

	if len(request.Data) == 0 {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "data cannot be empty",
				Code:    "EMPTY_DATA",
			},
			Field: "data",
		}
	}

	if request.Algorithm == "" {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "algorithm must be specified",
				Code:    "MISSING_ALGORITHM",
			},
			Field: "algorithm",
		}
	}

	url := fmt.Sprintf("%s/api/v1/detect", c.config.BaseURL)
	
	var response DetectionResponse
	err := c.makeRequest(ctx, "POST", url, request, &response)
	if err != nil {
		return nil, err
	}

	return &response, nil
}

// DetectAnomaliesEnsemble performs ensemble anomaly detection
func (c *Client) DetectAnomaliesEnsemble(ctx context.Context, request *EnsembleRequest) (*EnsembleResponse, error) {
	if request == nil {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "ensemble request cannot be nil",
				Code:    "INVALID_REQUEST",
			},
		}
	}

	if len(request.Data) == 0 {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "data cannot be empty",
				Code:    "EMPTY_DATA",
			},
			Field: "data",
		}
	}

	if len(request.Algorithms) == 0 {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "algorithms cannot be empty",
				Code:    "EMPTY_ALGORITHMS",
			},
			Field: "algorithms",
		}
	}

	url := fmt.Sprintf("%s/api/v1/ensemble", c.config.BaseURL)
	
	var response EnsembleResponse
	err := c.makeRequest(ctx, "POST", url, request, &response)
	if err != nil {
		return nil, err
	}

	return &response, nil
}

// TrainModel trains a new anomaly detection model
func (c *Client) TrainModel(ctx context.Context, request *TrainingRequest) (*TrainingResponse, error) {
	if request == nil {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "training request cannot be nil",
				Code:    "INVALID_REQUEST",
			},
		}
	}

	if len(request.Data) == 0 {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "training data cannot be empty",
				Code:    "EMPTY_DATA",
			},
			Field: "data",
		}
	}

	if request.Algorithm == "" {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "algorithm must be specified",
				Code:    "MISSING_ALGORITHM",
			},
			Field: "algorithm",
		}
	}

	url := fmt.Sprintf("%s/api/v1/models/train", c.config.BaseURL)
	
	var response TrainingResponse
	err := c.makeRequest(ctx, "POST", url, request, &response)
	if err != nil {
		return nil, err
	}

	return &response, nil
}

// ListModels retrieves a list of trained models
func (c *Client) ListModels(ctx context.Context, page, perPage int) (*ModelsListResponse, error) {
	url := fmt.Sprintf("%s/api/v1/models", c.config.BaseURL)
	
	// Add query parameters
	params := make(url.Values)
	if page > 0 {
		params.Set("page", strconv.Itoa(page))
	}
	if perPage > 0 {
		params.Set("per_page", strconv.Itoa(perPage))
	}
	
	if len(params) > 0 {
		url += "?" + params.Encode()
	}

	var response ModelsListResponse
	err := c.makeRequest(ctx, "GET", url, nil, &response)
	if err != nil {
		return nil, err
	}

	return &response, nil
}

// GetModel retrieves a specific model by ID
func (c *Client) GetModel(ctx context.Context, modelID string) (*Model, error) {
	if modelID == "" {
		return nil, ValidationError{
			ClientError: ClientError{
				Message: "model ID cannot be empty",
				Code:    "MISSING_MODEL_ID",
			},
			Field: "model_id",
		}
	}

	url := fmt.Sprintf("%s/api/v1/models/%s", c.config.BaseURL, modelID)
	
	var model Model
	err := c.makeRequest(ctx, "GET", url, nil, &model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}

// DeleteModel deletes a specific model by ID
func (c *Client) DeleteModel(ctx context.Context, modelID string) error {
	if modelID == "" {
		return ValidationError{
			ClientError: ClientError{
				Message: "model ID cannot be empty",
				Code:    "MISSING_MODEL_ID",
			},
			Field: "model_id",
		}
	}

	url := fmt.Sprintf("%s/api/v1/models/%s", c.config.BaseURL, modelID)
	
	return c.makeRequest(ctx, "DELETE", url, nil, nil)
}

// ListAlgorithms retrieves available anomaly detection algorithms
func (c *Client) ListAlgorithms(ctx context.Context) (*AlgorithmsResponse, error) {
	url := fmt.Sprintf("%s/api/v1/algorithms", c.config.BaseURL)
	
	var response AlgorithmsResponse
	err := c.makeRequest(ctx, "GET", url, nil, &response)
	if err != nil {
		return nil, err
	}

	return &response, nil
}

// HealthCheck performs a health check on the API
func (c *Client) HealthCheck(ctx context.Context) (*HealthResponse, error) {
	url := fmt.Sprintf("%s/health", c.config.BaseURL)
	
	var response HealthResponse
	err := c.makeRequest(ctx, "GET", url, nil, &response)
	if err != nil {
		return nil, err
	}

	return &response, nil
}

// StartStreaming starts a WebSocket connection for real-time anomaly detection
func (c *Client) StartStreaming(ctx context.Context) (<-chan StreamingMessage, error) {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	if c.wsConn != nil {
		return nil, ClientError{
			Message: "streaming connection already active",
			Code:    "STREAMING_ALREADY_ACTIVE",
		}
	}

	// Prepare headers for WebSocket connection
	headers := http.Header{}
	headers.Set("User-Agent", c.config.UserAgent)
	
	authHeader := c.authManager.GetAuthorizationHeader()
	if authHeader != "" {
		headers.Set("Authorization", authHeader)
	}

	// Establish WebSocket connection
	conn, _, err := websocket.DefaultDialer.Dial(c.config.StreamingURL, headers)
	if err != nil {
		return nil, NetworkError{
			ClientError: ClientError{
				Message: "failed to establish WebSocket connection",
				Code:    "WEBSOCKET_CONNECTION_FAILED",
			},
			Cause: err,
		}
	}

	c.wsConn = conn

	// Create channel for streaming messages
	msgChan := make(chan StreamingMessage, 100)

	// Start goroutine to handle incoming messages
	go c.handleStreamingMessages(ctx, msgChan)

	return msgChan, nil
}

// SendStreamingData sends data for real-time anomaly detection
func (c *Client) SendStreamingData(ctx context.Context, data []float64) error {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	if c.wsConn == nil {
		return ClientError{
			Message: "no active streaming connection",
			Code:    "NO_STREAMING_CONNECTION",
		}
	}

	message := StreamingMessage{
		Type:      "data",
		Timestamp: time.Now(),
		Data:      data,
	}

	return c.wsConn.WriteJSON(message)
}

// StopStreaming stops the WebSocket connection
func (c *Client) StopStreaming() error {
	c.wsMutex.Lock()
	defer c.wsMutex.Unlock()

	if c.wsConn == nil {
		return nil
	}

	err := c.wsConn.Close()
	c.wsConn = nil
	return err
}

// handleStreamingMessages handles incoming WebSocket messages
func (c *Client) handleStreamingMessages(ctx context.Context, msgChan chan<- StreamingMessage) {
	defer close(msgChan)

	for {
		select {
		case <-ctx.Done():
			return
		default:
			var message StreamingMessage
			err := c.wsConn.ReadJSON(&message)
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					// Send error message
					msgChan <- StreamingMessage{
						Type:      "error",
						Timestamp: time.Now(),
						Message:   fmt.Sprintf("WebSocket error: %v", err),
					}
				}
				return
			}

			select {
			case msgChan <- message:
			case <-ctx.Done():
				return
			}
		}
	}
}

// makeRequest makes an HTTP request with authentication, rate limiting, and retry logic
func (c *Client) makeRequest(ctx context.Context, method, url string, requestBody interface{}, responseBody interface{}) error {
	// Apply rate limiting
	if err := c.rateLimiter.Wait(ctx); err != nil {
		return TimeoutError{
			ClientError: ClientError{
				Message: "rate limiter timeout",
				Code:    "RATE_LIMIT_TIMEOUT",
			},
			Timeout: "rate_limiter",
		}
	}

	// Ensure we have a valid authentication token
	if err := c.authManager.EnsureValidToken(ctx); err != nil {
		return err
	}

	var reqBody io.Reader
	if requestBody != nil {
		jsonData, err := json.Marshal(requestBody)
		if err != nil {
			return NetworkError{
				ClientError: ClientError{
					Message: "failed to marshal request body",
					Code:    "MARSHAL_ERROR",
				},
				Cause: err,
			}
		}
		reqBody = bytes.NewReader(jsonData)
	}

	var lastErr error
	for attempt := 0; attempt <= c.config.MaxRetries; attempt++ {
		if attempt > 0 {
			// Apply exponential backoff
			backoff := time.Duration(attempt) * c.config.RetryBackoff
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return TimeoutError{
					ClientError: ClientError{
						Message: "context cancelled during retry backoff",
						Code:    "CONTEXT_CANCELLED",
					},
					Timeout: "retry_backoff",
				}
			}
		}

		req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
		if err != nil {
			lastErr = NetworkError{
				ClientError: ClientError{
					Message: "failed to create HTTP request",
					Code:    "REQUEST_CREATE_ERROR",
				},
				Cause: err,
			}
			continue
		}

		// Set headers
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("User-Agent", c.config.UserAgent)

		// Add authentication header
		authHeader := c.authManager.GetAuthorizationHeader()
		if authHeader != "" {
			req.Header.Set("Authorization", authHeader)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = NetworkError{
				ClientError: ClientError{
					Message: "HTTP request failed",
					Code:    "REQUEST_FAILED",
				},
				Cause: err,
			}
			continue
		}

		defer resp.Body.Close()

		// Handle successful responses
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			if responseBody != nil {
				if err := json.NewDecoder(resp.Body).Decode(responseBody); err != nil {
					return NetworkError{
						ClientError: ClientError{
							Message: "failed to decode response body",
							Code:    "DECODE_ERROR",
						},
						Cause: err,
					}
				}
			}
			return nil
		}

		// Handle error responses
		var errorResp ErrorResponse
		if err := json.NewDecoder(resp.Body).Decode(&errorResp); err != nil {
			lastErr = ServerError{
				ClientError: ClientError{
					Message: "failed to decode error response",
					Code:    "ERROR_DECODE_FAILED",
				},
				StatusCode: resp.StatusCode,
			}
			continue
		}

		lastErr = mapHTTPStatusToError(resp.StatusCode, &errorResp, resp.Header.Get("X-Request-ID"))

		// Don't retry on certain error types
		if IsAuthenticationError(lastErr) || IsAuthorizationError(lastErr) || IsValidationError(lastErr) {
			break
		}
	}

	return lastErr
}

// Close closes the client and cleans up resources
func (c *Client) Close() error {
	return c.StopStreaming()
}