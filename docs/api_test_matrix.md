# API Test Coverage Matrix

This matrix tracks test coverage for all API endpoints in the Pynomaly application.

**Legend:**
- **C**: Covered (has comprehensive tests)
- **M**: Missing (no tests found)
- **P**: Partial (some tests exist, but incomplete)

**Auth Levels:**
- **None**: No authentication required
- **Basic**: Basic authentication required
- **Viewer**: Viewer role required
- **Analyst**: Analyst role required
- **Data Scientist**: Data Scientist role required
- **Admin**: Admin role required
- **Super Admin**: Super Admin role required

## Core API Endpoints

### Authentication (`/api/v1/auth`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/login` | POST | 200, 401, 503 | None | TokenResponse | C | OAuth2 form data |
| `/refresh` | POST | 200, 401, 503 | None | TokenResponse | C | Refresh token required |
| `/register` | POST | 201, 400, 503 | None | UserResponse | M | User registration |
| `/me` | GET | 200, 401 | Basic | UserResponse | C | Current user profile |
| `/api-keys` | POST | 201, 401, 503 | Basic | APIKeyResponse | P | Create API key |
| `/api-keys/{api_key}` | DELETE | 200, 401, 403, 404 | Basic | dict | M | Revoke API key |
| `/logout` | POST | 200, 401 | Basic | dict | M | Logout user |

### User Management (`/api/v1`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/users/auth/login` | POST | 200, 401 | None | LoginResponse | M | Alternative login |
| `/users/auth/logout` | POST | 200 | Basic | dict | M | Logout with session |
| `/users/auth/me` | GET | 200, 401 | Basic | UserResponse | M | Current user info |
| `/users/` | POST | 201, 400 | None | UserResponse | M | Create user |
| `/users/` | GET | 200, 500 | Basic | List[UserResponse] | M | List users |
| `/users/{user_id}` | GET | 200, 403, 404 | Basic | UserResponse | M | Get user by ID |
| `/users/{user_id}/status` | PUT | 200, 404 | Basic | UserResponse | M | Toggle user status |
| `/users/{user_id}` | PUT | 200, 404 | Basic | UserResponse | M | Update user |
| `/users/{user_id}/reset-password` | PUT | 200, 404, 400 | Basic | dict | M | Reset password |
| `/users/{user_id}` | DELETE | 200, 404 | Basic | dict | M | Delete user |
| `/users/tenants` | POST | 201, 400, 403 | Super Admin | TenantResponse | M | Create tenant |
| `/users/tenants/{tenant_id}` | GET | 200, 403, 404 | Admin | TenantResponse | M | Get tenant |
| `/users/tenants/{tenant_id}/usage` | GET | 200, 404 | Admin | TenantUsageResponse | M | Get tenant usage |
| `/users/tenants/{tenant_id}/users` | GET | 200, 500 | Admin | List[UserResponse] | M | Get tenant users |
| `/users/tenants/{tenant_id}/invite` | POST | 200, 400 | Admin | UserResponse | M | Invite user |
| `/users/tenants/{tenant_id}/users/{user_id}/role` | PUT | 200, 400 | Admin | dict | M | Update user role |
| `/users/tenants/{tenant_id}/users/{user_id}` | DELETE | 200, 404, 400 | Admin | dict | M | Remove user from tenant |

### Health Check (`/api/v1/health`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/` | GET | 200, 503 | None | HealthResponse | P | Comprehensive health check |
| `/metrics` | GET | 200 | None | SystemMetricsResponse | M | System metrics |
| `/history` | GET | 200 | None | List[dict] | M | Health check history |
| `/summary` | GET | 200 | None | dict | M | Health summary |
| `/ready` | GET | 200, 503 | None | dict | M | Readiness probe |
| `/live` | GET | 200, 503 | None | dict | M | Liveness probe |

### Admin (`/api/v1/admin`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/users` | GET | 200, 503 | Super Admin | List[UserResponse] | M | List all users |
| `/users/{user_id}` | GET | 200, 404, 503 | Admin | UserResponse | M | Get user |
| `/users` | POST | 200, 400, 503 | Admin | UserResponse | M | Create user |
| `/users/{user_id}` | PATCH | 200, 404, 503 | Admin | UserResponse | M | Update user |

### Datasets (`/api/v1/datasets`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/` | GET | 200 | Viewer | List[DatasetDTO] | P | List datasets |
| `/{dataset_id}` | GET | 200, 404 | Viewer | DatasetDTO | P | Get dataset |
| `/upload` | POST | 200, 400, 413 | Data Scientist | DatasetDTO | C | Upload dataset |
| `/{dataset_id}/quality` | GET | 200, 404 | Viewer | DataQualityReportDTO | M | Check quality |
| `/{dataset_id}/sample` | GET | 200, 404 | Viewer | dict | M | Get sample |
| `/{dataset_id}/split` | POST | 200, 400, 404 | Data Scientist | dict | M | Split dataset |
| `/{dataset_id}` | DELETE | 200, 404 | Data Scientist | dict | M | Delete dataset |

### Detectors (`/api/v1/detectors`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/` | GET | 200 | Viewer | List[DetectorDTO] | P | List detectors |
| `/algorithms` | GET | 200 | None | dict | M | List algorithms |
| `/{detector_id}` | GET | 200, 404 | Viewer | DetectorDTO | P | Get detector |
| `/` | POST | 200, 400 | Data Scientist | DetectorDTO | M | Create detector |
| `/{detector_id}` | PATCH | 200, 404 | Analyst | DetectorDTO | M | Update detector |

### Detection (`/api/v1/detection`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/train` | POST | 200, 400, 404 | Data Scientist | dict | M | Train detector |
| `/detect` | POST | 200, 400, 404 | Analyst | DetectionResultDTO | C | Detect anomalies |
| `/detect/batch` | POST | 200, 400, 404 | Analyst | dict | M | Batch detection |
| `/evaluate` | POST | 200, 400, 404 | Analyst | dict | M | Evaluate detector |

### AutoML (`/api/v1/automl`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/search` | POST | 200, 400 | Data Scientist | dict | M | Algorithm search |
| `/optimize` | POST | 200, 400 | Data Scientist | dict | M | Hyperparameter optimization |

### Autonomous (`/api/v1/autonomous`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/pipelines` | GET | 200 | Viewer | List[dict] | M | List pipelines |
| `/pipelines` | POST | 201, 400 | Data Scientist | dict | M | Create pipeline |
| `/pipelines/{pipeline_id}` | GET | 200, 404 | Viewer | dict | M | Get pipeline |
| `/pipelines/{pipeline_id}` | PUT | 200, 404 | Data Scientist | dict | M | Update pipeline |
| `/pipelines/{pipeline_id}` | DELETE | 200, 404 | Data Scientist | dict | M | Delete pipeline |
| `/pipelines/{pipeline_id}/start` | POST | 200, 404 | Data Scientist | dict | M | Start pipeline |
| `/pipelines/{pipeline_id}/stop` | POST | 200, 404 | Data Scientist | dict | M | Stop pipeline |

### Ensemble (`/api/v1/ensemble`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/models` | GET | 200 | Viewer | List[dict] | M | List ensemble models |
| `/models` | POST | 201, 400 | Data Scientist | dict | M | Create ensemble |
| `/models/{model_id}` | GET | 200, 404 | Viewer | dict | M | Get ensemble model |
| `/models/{model_id}/predict` | POST | 200, 404 | Analyst | dict | M | Ensemble prediction |

### Explainability (`/api/v1/explainability`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/explain` | POST | 200, 400 | Analyst | dict | M | Explain prediction |
| `/feature-importance` | GET | 200, 404 | Analyst | dict | M | Feature importance |
| `/shap` | POST | 200, 400 | Analyst | dict | M | SHAP explanations |

### Experiments (`/api/v1/experiments`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/` | GET | 200 | Viewer | List[dict] | M | List experiments |
| `/` | POST | 201, 400 | Data Scientist | dict | M | Create experiment |
| `/{experiment_id}` | GET | 200, 404 | Viewer | dict | M | Get experiment |
| `/{experiment_id}/results` | GET | 200, 404 | Viewer | dict | M | Get results |

### Version (`/api/v1/version`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/` | GET | 200 | None | dict | M | Version information |

### Performance (`/api/v1/performance`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/benchmarks` | GET | 200 | Viewer | List[dict] | M | Performance benchmarks |
| `/metrics` | GET | 200 | Viewer | dict | M | Performance metrics |

### Export (`/api/v1/export`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/models/{model_id}` | GET | 200, 404 | Analyst | bytes | M | Export model |
| `/results/{result_id}` | GET | 200, 404 | Viewer | bytes | M | Export results |

### Model Lineage (`/api/v1/model-lineage`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/` | GET | 200 | Viewer | List[dict] | M | Model lineage |
| `/{model_id}` | GET | 200, 404 | Viewer | dict | M | Model lineage details |

### Streaming (`/api/v1/streaming`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/start` | POST | 200, 400 | Data Scientist | dict | M | Start streaming |
| `/stop` | POST | 200, 400 | Data Scientist | dict | M | Stop streaming |
| `/status` | GET | 200 | Viewer | dict | M | Streaming status |

### Events (`/api/v1/events`)

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/` | GET | 200 | Viewer | List[dict] | M | List events |
| `/subscribe` | POST | 200, 400 | Viewer | dict | M | Subscribe to events |

### Special Endpoints

| Endpoint | Method | Expected Status Codes | Auth Required | Response Model | Coverage | Notes |
|----------|--------|-----------------------|---------------|----------------|----------|-------|
| `/` | GET | 200 | None | dict | M | API root |
| `/metrics` | GET | 200 | None | text/plain | M | Prometheus metrics |
| `/docs` | GET | 200 | None | HTML | M | API documentation |
| `/redoc` | GET | 200 | None | HTML | M | ReDoc documentation |
| `/openapi.json` | GET | 200 | None | JSON | M | OpenAPI schema |

## Coverage Summary

**Total Endpoints:** ~90
**Covered (C):** ~15 (17%)
**Partial (P):** ~20 (22%)
**Missing (M):** ~55 (61%)

### Analysis Based on Existing Test Files:

**Well-Tested Areas:**
- Authentication endpoints (login, refresh, logout)
- Dataset management (upload, listing, quality checks)
- Detector endpoints (creation, listing, configuration)
- Health check endpoints (basic and advanced)
- Detection endpoints (anomaly detection, batch processing)

**Partially Tested Areas:**
- User management (some CRUD operations)
- Admin functions (basic user management)
- Validation and error handling
- Integration tests for workflows

**Missing Test Coverage:**
- AutoML endpoints
- Streaming and real-time processing
- Model lineage tracking
- Export functionality
- Advanced explainability features
- Performance benchmarking
- Multi-tenancy features

## Priority Testing Areas

1. **High Priority (Critical Business Logic)**
   - Authentication endpoints (login, refresh, register)
   - Dataset upload and management
   - Detector creation and training
   - Anomaly detection
   - Health checks

2. **Medium Priority (Core Features)**
   - User management
   - Admin functions
   - AutoML features
   - Explainability
   - Performance monitoring

3. **Low Priority (Advanced Features)**
   - Streaming
   - Events
   - Model lineage
   - Export functions
   - Ensemble methods

## Testing Guidelines

### For Each Endpoint Test:
1. **Happy Path**: Test successful execution
2. **Error Cases**: Test all expected error status codes
3. **Authentication**: Test auth requirements and permissions
4. **Validation**: Test input validation and edge cases
5. **Response Format**: Verify response models match specification

### Coverage Checklist:
- [ ] Request/Response validation
- [ ] Authentication and authorization
- [ ] Error handling and status codes
- [ ] Edge cases and boundary conditions
- [ ] Performance and timeout handling
- [ ] Database transactions and rollbacks
- [ ] Concurrent request handling

---

*Last updated: 2024-12-25*  
*Generated from: `src/pynomaly/presentation/api/app.py::create_app()`*

