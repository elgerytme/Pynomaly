package anomaly_detection

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// AuthManager handles authentication for the client
type AuthManager struct {
	client       *http.Client
	baseURL      string
	apiKey       string
	accessToken  string
	refreshToken string
	tokenExpiry  time.Time
	mutex        sync.RWMutex
}

// NewAuthManager creates a new authentication manager
func NewAuthManager(baseURL, apiKey string, client *http.Client) *AuthManager {
	return &AuthManager{
		client:  client,
		baseURL: baseURL,
		apiKey:  apiKey,
	}
}

// AuthenticateWithAPIKey authenticates using an API key
func (am *AuthManager) AuthenticateWithAPIKey(ctx context.Context, apiKey string) error {
	am.mutex.Lock()
	defer am.mutex.Unlock()

	req := AuthRequest{
		APIKey:    apiKey,
		TokenType: "api_key",
	}

	return am.authenticate(ctx, req)
}

// AuthenticateWithCredentials authenticates using username and password
func (am *AuthManager) AuthenticateWithCredentials(ctx context.Context, username, password string) error {
	am.mutex.Lock()
	defer am.mutex.Unlock()

	req := AuthRequest{
		Username:  username,
		Password:  password,
		TokenType: "credentials",
	}

	return am.authenticate(ctx, req)
}

// RefreshAccessToken refreshes the access token using the refresh token
func (am *AuthManager) RefreshAccessToken(ctx context.Context) error {
	am.mutex.Lock()
	defer am.mutex.Unlock()

	if am.refreshToken == "" {
		return AuthenticationError{
			ClientError: ClientError{
				Message: "no refresh token available",
				Code:    "NO_REFRESH_TOKEN",
			},
		}
	}

	url := fmt.Sprintf("%s/api/v1/auth/refresh", am.baseURL)
	
	reqBody := map[string]string{
		"refresh_token": am.refreshToken,
	}

	resp, err := am.makeAuthRequest(ctx, "POST", url, reqBody)
	if err != nil {
		return err
	}

	return am.processAuthResponse(resp)
}

// GetAuthorizationHeader returns the authorization header value
func (am *AuthManager) GetAuthorizationHeader() string {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	if am.accessToken != "" {
		return fmt.Sprintf("Bearer %s", am.accessToken)
	}
	
	if am.apiKey != "" {
		return fmt.Sprintf("ApiKey %s", am.apiKey)
	}

	return ""
}

// IsTokenExpired checks if the current token is expired
func (am *AuthManager) IsTokenExpired() bool {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	if am.tokenExpiry.IsZero() {
		return false
	}

	// Consider token expired if it expires within the next 5 minutes
	return time.Now().Add(5 * time.Minute).After(am.tokenExpiry)
}

// EnsureValidToken ensures we have a valid authentication token
func (am *AuthManager) EnsureValidToken(ctx context.Context) error {
	if am.IsTokenExpired() && am.refreshToken != "" {
		return am.RefreshAccessToken(ctx)
	}
	return nil
}

// authenticate performs the actual authentication request
func (am *AuthManager) authenticate(ctx context.Context, authReq AuthRequest) error {
	url := fmt.Sprintf("%s/api/v1/auth/login", am.baseURL)
	
	resp, err := am.makeAuthRequest(ctx, "POST", url, authReq)
	if err != nil {
		return err
	}

	return am.processAuthResponse(resp)
}

// makeAuthRequest makes an authenticated HTTP request
func (am *AuthManager) makeAuthRequest(ctx context.Context, method, url string, body interface{}) (*http.Response, error) {
	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, NetworkError{
			ClientError: ClientError{
				Message: "failed to marshal request body",
				Code:    "MARSHAL_ERROR",
			},
			Cause: err,
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, url, strings.NewReader(string(reqBody)))
	if err != nil {
		return nil, NetworkError{
			ClientError: ClientError{
				Message: "failed to create HTTP request",
				Code:    "REQUEST_CREATE_ERROR",
			},
			Cause: err,
		}
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "anomaly-detection-go-sdk/1.0.0")

	resp, err := am.client.Do(req)
	if err != nil {
		return nil, NetworkError{
			ClientError: ClientError{
				Message: "HTTP request failed",
				Code:    "REQUEST_FAILED",
			},
			Cause: err,
		}
	}

	return resp, nil
}

// processAuthResponse processes the authentication response
func (am *AuthManager) processAuthResponse(resp *http.Response) error {
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errorResp ErrorResponse
		if err := json.NewDecoder(resp.Body).Decode(&errorResp); err != nil {
			return ServerError{
				ClientError: ClientError{
					Message: "authentication failed with unknown error",
					Code:    "AUTH_UNKNOWN_ERROR",
				},
				StatusCode: resp.StatusCode,
			}
		}

		return mapHTTPStatusToError(resp.StatusCode, &errorResp, resp.Header.Get("X-Request-ID"))
	}

	var authResp AuthResponse
	if err := json.NewDecoder(resp.Body).Decode(&authResp); err != nil {
		return NetworkError{
			ClientError: ClientError{
				Message: "failed to decode authentication response",
				Code:    "DECODE_ERROR",
			},
			Cause: err,
		}
	}

	if !authResp.Success {
		return AuthenticationError{
			ClientError: ClientError{
				Message: "authentication failed",
				Code:    "AUTH_FAILED",
			},
		}
	}

	// Store authentication tokens
	am.accessToken = authResp.AccessToken
	am.refreshToken = authResp.RefreshToken
	am.tokenExpiry = authResp.ExpiresAt

	return nil
}

// Logout clears the authentication state
func (am *AuthManager) Logout(ctx context.Context) error {
	am.mutex.Lock()
	defer am.mutex.Unlock()

	// Optionally make a logout request to the server
	if am.accessToken != "" {
		url := fmt.Sprintf("%s/api/v1/auth/logout", am.baseURL)
		req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
		if err == nil {
			req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", am.accessToken))
			req.Header.Set("Content-Type", "application/json")
			// Make the request but don't fail if it errors
			am.client.Do(req)
		}
	}

	// Clear authentication state
	am.accessToken = ""
	am.refreshToken = ""
	am.tokenExpiry = time.Time{}

	return nil
}

// SetTokens manually sets authentication tokens (for testing or custom flows)
func (am *AuthManager) SetTokens(accessToken, refreshToken string, expiresAt time.Time) {
	am.mutex.Lock()
	defer am.mutex.Unlock()

	am.accessToken = accessToken
	am.refreshToken = refreshToken
	am.tokenExpiry = expiresAt
}

// GetTokenInfo returns current token information (for debugging)
func (am *AuthManager) GetTokenInfo() (accessToken string, expiresAt time.Time, isExpired bool) {
	am.mutex.RLock()
	defer am.mutex.RUnlock()

	return am.accessToken, am.tokenExpiry, am.IsTokenExpired()
}