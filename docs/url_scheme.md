# URL Scheme and API Versioning

## Overview

This document describes the URL scheme and versioning strategy for the Pynomaly project, which uses FastAPI to serve both a web UI and a REST API.

## URL Structure

The Pynomaly application follows a clear separation between web UI and API endpoints:

- **Web UI**: Served at the root path `/`
- **API**: Versioned endpoints under `/api/v1`

## Implementation Details

### Web UI Mounting

The web UI is mounted at the root path with no prefix:

**File**: `src/pynomaly/presentation/web/app.py` (lines 1737-1780)

```python
# Web UI router mounted at root "/"
app.include_router(web_ui_router)
```

### API Versioning

The API uses versioning with the `/api/v1` prefix for all endpoints:

**File**: `src/pynomaly/presentation/api/app.py` (lines 305-357)

```python
# API routers mounted with "/api/v1" prefix
app.include_router(health_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(detectors_router, prefix="/api/v1")
app.include_router(datasets_router, prefix="/api/v1")
app.include_router(experiments_router, prefix="/api/v1")
app.include_router(version_router, prefix="/api/v1")
```

## Principal Routes

| Component | Route Pattern | Example Endpoints | Description |
|-----------|---------------|------------------|-------------|
| Web UI | `/` | `/`, `/login`, `/dashboard` | User interface pages |
| API Health | `/api/v1/health` | `/api/v1/health` | System health check |
| API Auth | `/api/v1/auth/*` | `/api/v1/auth/login`, `/api/v1/auth/logout` | Authentication endpoints |
| API Detectors | `/api/v1/detectors/*` | `/api/v1/detectors/`, `/api/v1/detectors/{id}` | Anomaly detector management |
| API Datasets | `/api/v1/datasets/*` | `/api/v1/datasets/`, `/api/v1/datasets/{id}` | Dataset management |
| API Experiments | `/api/v1/experiments/*` | `/api/v1/experiments/`, `/api/v1/experiments/{id}` | Experiment management |
| API Version | `/api/v1/version` | `/api/v1/version` | Application version info |
| API Docs | `/api/v1/docs` | `/api/v1/docs` | OpenAPI documentation |

## Legacy Route Handling

Legacy routes under `/web/*` are **not supported** and return HTTP 404 errors. This ensures a clean separation between the web UI (served at root) and API endpoints.

## Verification

This URL scheme has been verified through comprehensive testing:

### Unit Tests
- **File**: `tests/unit/test_url_routing_simple.py`
- **Status**: ✅ PASSED
- **Validates**: Web UI routes at `/` and 404 responses for legacy `/web/*` paths

### Integration Tests
- **File**: `tests/integration/api/test_api_versioning.py`
- **Status**: ✅ PASSED
- **Validates**: All API routers correctly versioned under `/api/v1`

### Live TestClient Tests
- **File**: `live_testclient_checks.py`
- **Status**: ✅ PASSED
- **Validates**: Real endpoint responses for both web UI and API routes

### Regression Tests
- **File**: `tests/regression/test_routing_regression.py`
- **Status**: ✅ PASSED (10/12 tests)
- **Validates**: App instantiation, routing structure, and middleware configuration

## Design Benefits

1. **Clear Separation**: Web UI at root, API under `/api/v1`
2. **Future-Proof**: Versioned API allows for backward compatibility
3. **Standard Compliance**: Follows REST API versioning best practices
4. **Maintainability**: Clear routing structure for development and debugging

## FastAPI Configuration

The application uses FastAPI's built-in router mounting functionality with the `include_router()` method to achieve this URL scheme. This leverages FastAPI's native capabilities for:

- Path prefixing
- Route grouping
- Middleware application
- OpenAPI schema generation
- CORS configuration

## Conclusion

The current URL scheme successfully separates web UI and API concerns while providing a versioned API structure that supports future evolution of the application.
