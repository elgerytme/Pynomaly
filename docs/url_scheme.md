# Pynomaly URL Scheme Documentation

## Overview

This document describes the URL scheme and routing strategy for the Pynomaly anomaly detection platform. The system employs a clear separation between web UI and API endpoints, with versioned API routes following REST conventions.

## 1. Web UI Served at Root "/"

### Base URL Structure
The web UI is served directly at the root path `/`, providing a clean user experience without additional path prefixes.

**Implementation Reference:**
- **Code**: `src/pynomaly/presentation/web/app.py:62-94`
- **Mount Function**: `src/pynomaly/presentation/api/app.py:356`

### Web UI Routes

| Route | Description | Handler | Code Reference |
|-------|-------------|---------|----------------|
| `/` | Main dashboard page | `index()` | `src/pynomaly/presentation/web/app.py:62-94` |
| `/login` | Login page | `login_page()` | `src/pynomaly/presentation/web/app.py:97-100` |
| `/detectors` | Detector management | `detectors_page()` | `src/pynomaly/presentation/web/app.py:157-169` |
| `/datasets` | Dataset management | `datasets_page()` | `src/pynomaly/presentation/web/app.py:200+` |
| `/detection` | Detection interface | `detection_page()` | Web UI endpoints |
| `/monitoring` | System monitoring | `monitoring_page()` | Web UI endpoints |

**Key Features:**
- **Authentication Integration**: Routes check authentication status when enabled
- **HTMX Support**: Modern interactive web interface using HTMX
- **Template Engine**: Jinja2 templates for server-side rendering
- **Static Assets**: CSS, JS, and image assets served from `/static/`

**Test Evidence:**
- **File**: `tests/unit/test_url_routing_simple.py:72-89`
- **Web UI Test**: Validates root-level web endpoints return 200 status
- **Old Route Test**: `tests/unit/test_url_routing_simple.py:107-124` confirms `/web/*` routes are disabled

## 2. API Versioning Strategy Using FastAPI with "/api/v1"

### Versioning Philosophy
The API follows semantic versioning with a clear path-based strategy:
- Current version: `v1` (stable)
- Base path: `/api/v1`
- Future versions: `/api/v2`, `/api/v3`, etc.

**Implementation Reference:**
- **Main App**: `src/pynomaly/presentation/api/app.py:304-351`
- **Version Router**: `src/pynomaly/presentation/api/endpoints/version.py`

### API Documentation URLs

| URL | Description | Status |
|-----|-------------|--------|
| `/api/v1/docs` | Interactive API documentation (Swagger UI) | Active |
| `/api/v1/redoc` | Alternative API documentation (ReDoc) | Active |
| `/api/v1/openapi.json` | OpenAPI specification | Active |

**Configuration Reference:**
```python
# src/pynomaly/presentation/api/app.py:193-195
docs_url="/api/v1/docs" if settings.docs_enabled else None,
redoc_url="/api/v1/redoc" if settings.docs_enabled else None,
openapi_url="/api/v1/openapi.json" if settings.docs_enabled else None,
```

### Version Information Endpoint
**Endpoint**: `GET /api/v1/version`
**Response**: Application version, build info, and API compatibility

## 3. Principal Routes Table

### Core System Routes

| Category | Route | Method | Description | Security | Code Reference |
|----------|-------|--------|-------------|----------|----------------|
| **Health** | `/api/v1/health/` | GET | Comprehensive health check | None | `src/pynomaly/presentation/api/endpoints/health.py:125-228` |
| **Health** | `/api/v1/health/ready` | GET | Kubernetes readiness probe | None | `src/pynomaly/presentation/api/endpoints/health.py:281-312` |
| **Health** | `/api/v1/health/live` | GET | Kubernetes liveness probe | None | `src/pynomaly/presentation/api/endpoints/health.py:315-350` |
| **Health** | `/api/v1/health/metrics` | GET | System resource metrics | None | `src/pynomaly/presentation/api/endpoints/health.py:231-246` |

### Authentication Routes

| Route | Method | Description | Security | Code Reference |
|-------|--------|-------------|----------|----------------|
| `/api/v1/auth/login` | POST | User authentication | None | `src/pynomaly/presentation/api/endpoints/auth.py:67-100` |
| `/api/v1/auth/register` | POST | User registration | None | `src/pynomaly/presentation/api/endpoints/auth.py:132-176` |
| `/api/v1/auth/refresh` | POST | Token refresh | None | `src/pynomaly/presentation/api/endpoints/auth.py:102-130` |
| `/api/v1/auth/me` | GET | Current user profile | JWT | `src/pynomaly/presentation/api/endpoints/auth.py:179-200` |
| `/api/v1/auth/logout` | POST | User logout | JWT | Auth endpoint |

### Data Management Routes

| Route | Method | Description | Security | Code Reference |
|-------|--------|-------------|----------|----------------|
| `/api/v1/datasets/` | GET | List datasets | Viewer | `src/pynomaly/presentation/api/endpoints/datasets.py` |
| `/api/v1/datasets/{id}` | GET | Get dataset details | Viewer | Dataset endpoints |
| `/api/v1/datasets/upload` | POST | Upload new dataset | Data Scientist | Dataset endpoints |
| `/api/v1/datasets/{id}/quality` | GET | Data quality report | Viewer | Dataset endpoints |
| `/api/v1/detectors/` | GET | List detectors | Viewer | `src/pynomaly/presentation/api/endpoints/detectors.py` |
| `/api/v1/detectors/create` | POST | Create detector | Data Scientist | Detector endpoints |

### ML Operations Routes

| Route | Method | Description | Security | Code Reference |
|-------|--------|-------------|----------|----------------|
| `/api/v1/detection/train` | POST | Train detector model | Data Scientist | `src/pynomaly/presentation/api/endpoints/detection.py` |
| `/api/v1/detection/predict` | POST | Run anomaly detection | Data Scientist | Detection endpoints |
| `/api/v1/automl/optimize` | POST | AutoML optimization | Data Scientist | `src/pynomaly/presentation/api/endpoints/automl.py` |
| `/api/v1/ensemble/create` | POST | Create ensemble model | Data Scientist | `src/pynomaly/presentation/api/endpoints/ensemble.py` |
| `/api/v1/explainability/analyze` | POST | Generate explanations | Viewer | `src/pynomaly/presentation/api/endpoints/explainability.py` |

### Advanced Features Routes

| Route | Method | Description | Security | Code Reference |
|-------|--------|-------------|----------|----------------|
| `/api/v1/streaming/create` | POST | Create streaming pipeline | Data Scientist | `src/pynomaly/presentation/api/endpoints/streaming.py` |
| `/api/v1/experiments/create` | POST | Create ML experiment | Data Scientist | `src/pynomaly/presentation/api/endpoints/experiments.py` |
| `/api/v1/performance/benchmark` | GET | Performance benchmarks | Viewer | `src/pynomaly/presentation/api/endpoints/performance.py` |
| `/api/v1/admin/users` | GET | User management | Admin | `src/pynomaly/presentation/api/endpoints/admin.py` |

### System Monitoring Routes

| Route | Method | Description | Security | Code Reference |
|-------|--------|-------------|----------|----------------|
| `/metrics` | GET | Prometheus metrics | None | `src/pynomaly/presentation/api/app.py:252-258` |
| `/api/v1/monitoring/alerts` | GET | System alerts | Admin | Monitoring endpoints |
| `/api/v1/events/` | GET | Event stream | Viewer | `src/pynomaly/presentation/api/endpoints/events.py` |

## Security Implementation

### Role-Based Access Control (RBAC)
- **Middleware**: `src/pynomaly/infrastructure/security/rbac_middleware.py`
- **Decorator**: `require_auth()` for JWT validation
- **Roles**: Admin, Data Scientist, Viewer

### Authentication Flow
1. **Login**: `POST /api/v1/auth/login` → JWT token
2. **Headers**: `Authorization: Bearer <token>`
3. **Validation**: Middleware validates token on protected routes

## Router Configuration

### App Structure
**Main Application**: `src/pynomaly/presentation/api/app.py:135-370`

```python
# Router inclusion pattern
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
# ... additional routers
```

### Web UI Mounting
**Implementation**: `src/pynomaly/presentation/api/app.py:356`
```python
# Mount web UI with lazy import to avoid circular dependencies
_mount_web_ui_lazy(app)
```

## Testing Evidence

### URL Routing Tests
- **File**: `tests/unit/test_url_routing_simple.py`
- **Coverage**: Root-level web UI, API versioning, deprecated route handling
- **Results**: Validates `/web/*` → `/` migration

### API Integration Tests
- **File**: `tests/integration/test_api_workflows.py`
- **Coverage**: End-to-end API workflows with proper routing
- **Authentication**: JWT token validation across routes

### Health Check Tests
- **File**: `tests/presentation/api/test_health_endpoints.py`
- **Coverage**: Health endpoint functionality and responses

## Error Handling

### HTTP Status Codes
- **200**: Success
- **400**: Bad Request (validation errors)
- **401**: Unauthorized (authentication required)
- **403**: Forbidden (insufficient permissions)
- **404**: Not Found
- **500**: Internal Server Error
- **503**: Service Unavailable

### Error Response Format
```json
{
  "detail": "Error message",
  "status_code": 400,
  "timestamp": "2024-12-25T10:30:00Z"
}
```

## Performance Considerations

### Caching Strategy
- **Redis**: Response caching for expensive operations
- **Headers**: Appropriate cache-control headers
- **Static Assets**: Efficient serving of web UI resources

### Rate Limiting
- **Health Endpoints**: 60 requests/minute
- **API Endpoints**: User-based rate limiting
- **Implementation**: `src/pynomaly/infrastructure/auth/rate_limiting.py`

## Deployment Notes

### Docker/Kubernetes
- **Readiness**: `/api/v1/health/ready`
- **Liveness**: `/api/v1/health/live`
- **Metrics**: `/metrics` for Prometheus scraping

### Load Balancer Configuration
- **Static Assets**: Direct serving bypassing application
- **API Routes**: Load balanced across instances
- **WebSocket**: Sticky sessions for real-time features

## Future Considerations

### API Versioning Strategy
- **v2 Planning**: Breaking changes will introduce `/api/v2`
- **Deprecation**: Gradual sunset of older versions
- **Migration**: Clear upgrade paths for clients

### Progressive Web App (PWA)
- **Service Worker**: Offline capability for web UI
- **Push Notifications**: Real-time alert system
- **Installation**: Native app-like experience

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Maintainer**: Pynomaly Development Team
