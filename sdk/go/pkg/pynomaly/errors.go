package pynomaly

import (
	"fmt"
	"net/http"
)

// PynomalyError is the base error type for all Pynomaly SDK errors
type PynomalyError struct {
	Code       string                 `json:"code"`
	Message    string                 `json:"message"`
	StatusCode int                    `json:"status_code,omitempty"`
	Details    map[string]interface{} `json:"details,omitempty"`
}

func (e *PynomalyError) Error() string {
	if e.StatusCode > 0 {
		return fmt.Sprintf("pynomaly error [%s] (%d): %s", e.Code, e.StatusCode, e.Message)
	}
	return fmt.Sprintf("pynomaly error [%s]: %s", e.Code, e.Message)
}

// AuthenticationError represents authentication failures (401)
type AuthenticationError struct {
	PynomalyError
}

func NewAuthenticationError(message string, details map[string]interface{}) *AuthenticationError {
	return &AuthenticationError{
		PynomalyError: PynomalyError{
			Code:       "AUTHENTICATION_ERROR",
			Message:    message,
			StatusCode: http.StatusUnauthorized,
			Details:    details,
		},
	}
}

// AuthorizationError represents authorization failures (403)
type AuthorizationError struct {
	PynomalyError
}

func NewAuthorizationError(message string, details map[string]interface{}) *AuthorizationError {
	return &AuthorizationError{
		PynomalyError: PynomalyError{
			Code:       "AUTHORIZATION_ERROR",
			Message:    message,
			StatusCode: http.StatusForbidden,
			Details:    details,
		},
	}
}

// ValidationError represents validation failures (400)
type ValidationError struct {
	PynomalyError
}

func NewValidationError(message string, details map[string]interface{}) *ValidationError {
	return &ValidationError{
		PynomalyError: PynomalyError{
			Code:       "VALIDATION_ERROR",
			Message:    message,
			StatusCode: http.StatusBadRequest,
			Details:    details,
		},
	}
}

// NotFoundError represents resource not found errors (404)
type NotFoundError struct {
	PynomalyError
}

func NewNotFoundError(resource, id string) *NotFoundError {
	message := fmt.Sprintf("%s not found", resource)
	if id != "" {
		message = fmt.Sprintf("%s with id '%s' not found", resource, id)
	}
	
	return &NotFoundError{
		PynomalyError: PynomalyError{
			Code:       "NOT_FOUND_ERROR",
			Message:    message,
			StatusCode: http.StatusNotFound,
		},
	}
}

// ConflictError represents resource conflict errors (409)
type ConflictError struct {
	PynomalyError
}

func NewConflictError(message string, details map[string]interface{}) *ConflictError {
	return &ConflictError{
		PynomalyError: PynomalyError{
			Code:       "CONFLICT_ERROR",
			Message:    message,
			StatusCode: http.StatusConflict,
			Details:    details,
		},
	}
}

// RateLimitError represents rate limit exceeded errors (429)
type RateLimitError struct {
	PynomalyError
	RetryAfter int `json:"retry_after,omitempty"`
}

func NewRateLimitError(message string, retryAfter int) *RateLimitError {
	if message == "" {
		message = "Rate limit exceeded"
	}
	
	details := make(map[string]interface{})
	if retryAfter > 0 {
		details["retry_after"] = retryAfter
	}
	
	return &RateLimitError{
		PynomalyError: PynomalyError{
			Code:       "RATE_LIMIT_ERROR",
			Message:    message,
			StatusCode: http.StatusTooManyRequests,
			Details:    details,
		},
		RetryAfter: retryAfter,
	}
}

// ServerError represents server errors (5xx)
type ServerError struct {
	PynomalyError
}

func NewServerError(message string, statusCode int, details map[string]interface{}) *ServerError {
	if message == "" {
		message = "Internal server error"
	}
	
	return &ServerError{
		PynomalyError: PynomalyError{
			Code:       "SERVER_ERROR",
			Message:    message,
			StatusCode: statusCode,
			Details:    details,
		},
	}
}

// NetworkError represents network-related errors
type NetworkError struct {
	Message string
}

func (e *NetworkError) Error() string {
	return fmt.Sprintf("network error: %s", e.Message)
}

// TimeoutError represents timeout errors
type TimeoutError struct {
	PynomalyError
	Timeout int `json:"timeout,omitempty"`
}

func NewTimeoutError(message string, timeout int) *TimeoutError {
	if message == "" {
		message = "Request timeout"
	}
	
	details := make(map[string]interface{})
	if timeout > 0 {
		details["timeout"] = timeout
	}
	
	return &TimeoutError{
		PynomalyError: PynomalyError{
			Code:       "TIMEOUT_ERROR",
			Message:    message,
			StatusCode: http.StatusRequestTimeout,
			Details:    details,
		},
		Timeout: timeout,
	}
}

// StreamingError represents streaming-related errors
type StreamingError struct {
	PynomalyError
}

func NewStreamingError(message string, details map[string]interface{}) *StreamingError {
	return &StreamingError{
		PynomalyError: PynomalyError{
			Code:    "STREAMING_ERROR",
			Message: message,
			Details: details,
		},
	}
}

// TestingError represents A/B testing-related errors
type TestingError struct {
	PynomalyError
}

func NewTestingError(message string, details map[string]interface{}) *TestingError {
	return &TestingError{
		PynomalyError: PynomalyError{
			Code:    "TESTING_ERROR",
			Message: message,
			Details: details,
		},
	}
}

// ComplianceError represents compliance-related errors
type ComplianceError struct {
	PynomalyError
}

func NewComplianceError(message string, details map[string]interface{}) *ComplianceError {
	return &ComplianceError{
		PynomalyError: PynomalyError{
			Code:    "COMPLIANCE_ERROR",
			Message: message,
			Details: details,
		},
	}
}

// IsRetryableError checks if an error is retryable
func IsRetryableError(err error) bool {
	switch e := err.(type) {
	case *NetworkError:
		return true
	case *TimeoutError:
		return true
	case *RateLimitError:
		return true
	case *ServerError:
		return e.StatusCode >= 500
	case *PynomalyError:
		return e.StatusCode >= 500 || e.Code == "NETWORK_ERROR" || e.Code == "TIMEOUT_ERROR" || e.Code == "RATE_LIMIT_ERROR"
	default:
		return false
	}
}