package anomaly_detection

import (
	"fmt"
	"net/http"
)

// Error types for the anomaly detection client
type (
	// ClientError represents a general client error
	ClientError struct {
		Message   string
		Code      string
		RequestID string
		Details   map[string]interface{}
	}

	// AuthenticationError represents authentication failures
	AuthenticationError struct {
		ClientError
	}

	// AuthorizationError represents authorization failures
	AuthorizationError struct {
		ClientError
	}

	// ValidationError represents request validation failures
	ValidationError struct {
		ClientError
		Field string
	}

	// RateLimitError represents rate limiting errors
	RateLimitError struct {
		ClientError
		RetryAfter int
	}

	// ServerError represents server-side errors
	ServerError struct {
		ClientError
		StatusCode int
	}

	// NetworkError represents network-related errors
	NetworkError struct {
		ClientError
		Cause error
	}

	// TimeoutError represents timeout errors
	TimeoutError struct {
		ClientError
		Timeout string
	}

	// ModelNotFoundError represents model not found errors
	ModelNotFoundError struct {
		ClientError
		ModelID string
	}

	// AlgorithmNotSupportedError represents unsupported algorithm errors
	AlgorithmNotSupportedError struct {
		ClientError
		Algorithm string
	}
)

// Error implements the error interface for ClientError
func (e ClientError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("[%s] %s", e.Code, e.Message)
	}
	return e.Message
}

// Error implements the error interface for AuthenticationError
func (e AuthenticationError) Error() string {
	return fmt.Sprintf("authentication failed: %s", e.ClientError.Error())
}

// Error implements the error interface for AuthorizationError
func (e AuthorizationError) Error() string {
	return fmt.Sprintf("authorization failed: %s", e.ClientError.Error())
}

// Error implements the error interface for ValidationError
func (e ValidationError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("validation error for field '%s': %s", e.Field, e.ClientError.Error())
	}
	return fmt.Sprintf("validation error: %s", e.ClientError.Error())
}

// Error implements the error interface for RateLimitError
func (e RateLimitError) Error() string {
	if e.RetryAfter > 0 {
		return fmt.Sprintf("rate limit exceeded, retry after %d seconds: %s", e.RetryAfter, e.ClientError.Error())
	}
	return fmt.Sprintf("rate limit exceeded: %s", e.ClientError.Error())
}

// Error implements the error interface for ServerError
func (e ServerError) Error() string {
	return fmt.Sprintf("server error (%d): %s", e.StatusCode, e.ClientError.Error())
}

// Error implements the error interface for NetworkError
func (e NetworkError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("network error: %s (cause: %v)", e.ClientError.Error(), e.Cause)
	}
	return fmt.Sprintf("network error: %s", e.ClientError.Error())
}

// Error implements the error interface for TimeoutError
func (e TimeoutError) Error() string {
	return fmt.Sprintf("timeout error (%s): %s", e.Timeout, e.ClientError.Error())
}

// Error implements the error interface for ModelNotFoundError
func (e ModelNotFoundError) Error() string {
	return fmt.Sprintf("model not found (ID: %s): %s", e.ModelID, e.ClientError.Error())
}

// Error implements the error interface for AlgorithmNotSupportedError
func (e AlgorithmNotSupportedError) Error() string {
	return fmt.Sprintf("algorithm '%s' not supported: %s", e.Algorithm, e.ClientError.Error())
}

// Unwrap returns the underlying error for NetworkError
func (e NetworkError) Unwrap() error {
	return e.Cause
}

// IsAuthenticationError checks if the error is an authentication error
func IsAuthenticationError(err error) bool {
	_, ok := err.(AuthenticationError)
	return ok
}

// IsAuthorizationError checks if the error is an authorization error
func IsAuthorizationError(err error) bool {
	_, ok := err.(AuthorizationError)
	return ok
}

// IsValidationError checks if the error is a validation error
func IsValidationError(err error) bool {
	_, ok := err.(ValidationError)
	return ok
}

// IsRateLimitError checks if the error is a rate limit error
func IsRateLimitError(err error) bool {
	_, ok := err.(RateLimitError)
	return ok
}

// IsServerError checks if the error is a server error
func IsServerError(err error) bool {
	_, ok := err.(ServerError)
	return ok
}

// IsNetworkError checks if the error is a network error
func IsNetworkError(err error) bool {
	_, ok := err.(NetworkError)
	return ok
}

// IsTimeoutError checks if the error is a timeout error
func IsTimeoutError(err error) bool {
	_, ok := err.(TimeoutError)
	return ok
}

// IsModelNotFoundError checks if the error is a model not found error
func IsModelNotFoundError(err error) bool {
	_, ok := err.(ModelNotFoundError)
	return ok
}

// IsAlgorithmNotSupportedError checks if the error is an algorithm not supported error
func IsAlgorithmNotSupportedError(err error) bool {
	_, ok := err.(AlgorithmNotSupportedError)
	return ok
}

// mapHTTPStatusToError maps HTTP status codes to appropriate error types
func mapHTTPStatusToError(statusCode int, response *ErrorResponse, requestID string) error {
	baseError := ClientError{
		Message:   response.Error,
		Code:      response.Code,
		RequestID: requestID,
		Details:   response.Details,
	}

	switch statusCode {
	case http.StatusUnauthorized:
		return AuthenticationError{ClientError: baseError}
	case http.StatusForbidden:
		return AuthorizationError{ClientError: baseError}
	case http.StatusBadRequest:
		return ValidationError{ClientError: baseError}
	case http.StatusTooManyRequests:
		return RateLimitError{ClientError: baseError}
	case http.StatusNotFound:
		if response.Code == "MODEL_NOT_FOUND" {
			modelID := ""
			if details := response.Details; details != nil {
				if id, ok := details["model_id"].(string); ok {
					modelID = id
				}
			}
			return ModelNotFoundError{
				ClientError: baseError,
				ModelID:     modelID,
			}
		}
		return ClientError{
			Message:   "resource not found",
			Code:      "NOT_FOUND",
			RequestID: requestID,
		}
	case http.StatusUnprocessableEntity:
		if response.Code == "ALGORITHM_NOT_SUPPORTED" {
			algorithm := ""
			if details := response.Details; details != nil {
				if algo, ok := details["algorithm"].(string); ok {
					algorithm = algo
				}
			}
			return AlgorithmNotSupportedError{
				ClientError: baseError,
				Algorithm:   algorithm,
			}
		}
		return ValidationError{ClientError: baseError}
	default:
		if statusCode >= 500 {
			return ServerError{
				ClientError: baseError,
				StatusCode:  statusCode,
			}
		}
		return baseError
	}
}