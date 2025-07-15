# RFC 7807 Problem Details Error Handling

This document describes the implementation of RFC 7807 Problem Details for HTTP APIs in the Pynomaly FastAPI application.

## Overview

RFC 7807 defines a standard format for HTTP API error responses that provides a consistent structure for error information. Our implementation provides:

1. **RFC 7807 Problem Details format** for all error responses
2. **Strict Pydantic validation** with `extra="forbid"` on all DTOs
3. **Correlation ID middleware** for request tracking
4. **Comprehensive exception handling** for all error types

## Features

### 1. Problem Details Response Format

All error responses follow the RFC 7807 standard structure:

```json
{
  "type": "about:blank",
  "title": "Validation Error",
  "status": 422,
  "detail": "Input validation failed: age -> ensure this value is greater than or equal to 0",
  "instance": "/api/v1/users",
  "validation_errors": [
    {
      "loc": ["age"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

### 2. Correlation ID Tracking

Every request and response includes a correlation ID for tracking:

- **Request header**: `X-Correlation-ID` (auto-generated if not provided)
- **Response header**: `X-Correlation-ID` (echoed back)
- **Error responses**: Include correlation ID in header and logs

### 3. Strict Validation

All DTOs now use `extra="forbid"` to reject unexpected fields:

```python
class DetectionRequestDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    detector_id: UUID
    threshold: float = Field(ge=0, le=1)
    # ... other fields
```

### 4. Exception Handling

The system handles various exception types:

- **HTTP Exceptions**: 400-500 status codes
- **Validation Errors**: Pydantic and FastAPI validation
- **General Exceptions**: Unexpected errors (500)

## Implementation Details

### Error Handler Registration

Exception handlers are registered in the FastAPI app:

```python
from pynomaly.infrastructure.error_handling.problem_details_handler import add_exception_handlers

app = FastAPI()
add_exception_handlers(app)
```

### Middleware Integration

Correlation ID middleware is added to track requests:

```python
from pynomaly.infrastructure.middleware.correlation_id_middleware import add_correlation_id

app.middleware("http")(add_correlation_id)
```

### DTO Validation

All DTOs are updated with strict validation:

```python
class MyDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    required_field: str
    optional_field: Optional[int] = None
```

## Error Response Examples

### Validation Error

**Request:**
```json
POST /api/v1/detection/predict
{
  "detector_id": "invalid-uuid",
  "threshold": 1.5,
  "extra_field": "not_allowed"
}
```

**Response:**
```json
HTTP/1.1 422 Unprocessable Entity
X-Correlation-ID: 123e4567-e89b-12d3-a456-426614174000
Content-Type: application/json

{
  "type": "https://tools.ietf.org/html/rfc7231#section-6.5.1",
  "title": "Validation Error",
  "status": 422,
  "detail": "Input validation failed: detector_id -> field required; threshold -> ensure this value is less than or equal to 1; extra_field -> extra fields not permitted",
  "instance": "/api/v1/detection/predict",
  "validation_errors": [
    {
      "loc": ["detector_id"],
      "msg": "field required",
      "type": "value_error.missing"
    },
    {
      "loc": ["threshold"],
      "msg": "ensure this value is less than or equal to 1",
      "type": "value_error.number.not_le"
    },
    {
      "loc": ["extra_field"],
      "msg": "extra fields not permitted",
      "type": "value_error.extra"
    }
  ]
}
```

### HTTP Exception

**Request:**
```json
GET /api/v1/detectors/nonexistent-id
```

**Response:**
```json
HTTP/1.1 404 Not Found
X-Correlation-ID: 123e4567-e89b-12d3-a456-426614174000
Content-Type: application/json

{
  "type": "about:blank",
  "title": "Not Found",
  "status": 404,
  "detail": "Detector not found",
  "instance": "/api/v1/detectors/nonexistent-id"
}
```

### Internal Server Error

**Response:**
```json
HTTP/1.1 500 Internal Server Error
X-Correlation-ID: 123e4567-e89b-12d3-a456-426614174000
Content-Type: application/json

{
  "type": "about:blank",
  "title": "Internal Server Error",
  "status": 500,
  "detail": "An unexpected error occurred. Please try again later.",
  "instance": "/api/v1/some/endpoint"
}
```

## Testing

The implementation includes comprehensive tests covering:

- Problem Details response structure
- Validation error handling
- Strict validation enforcement
- HTTP exception handling
- Correlation ID middleware
- General exception handling

Run tests with:

```bash
python -m pytest tests/test_rfc7807_error_handling.py -v
```

## Benefits

1. **Consistent Error Format**: All errors follow the same structure
2. **Better Debugging**: Correlation IDs enable request tracking
3. **Stricter Validation**: Extra fields are rejected, preventing issues
4. **Standards Compliance**: Follows RFC 7807 specification
5. **Improved Client Experience**: Predictable error handling

## Files Modified

- `src/pynomaly/infrastructure/error_handling/problem_details_handler.py`
- `src/pynomaly/infrastructure/middleware/correlation_id_middleware.py`
- `src/pynomaly/application/dto/*.py` (all DTOs updated)
- `src/pynomaly/presentation/api/app.py`
- `tests/test_rfc7807_error_handling.py`

## Configuration

The error handling system is automatically configured when creating the FastAPI app. No additional configuration is required.

## Future Enhancements

- Custom error types with specific URIs
- Localization support for error messages
- Enhanced logging integration
- Metrics collection for error patterns
- Rate limiting for error responses
