# API Test Suite

This directory contains comprehensive tests for the Pynomaly API endpoints, covering all major functionality implemented as part of Issue #141: Infrastructure and API Foundation.

## Test Files

### Core API Endpoints
- `test_auth_endpoints.py` - Authentication and authorization endpoints
- `test_data_quality_endpoints.py` - Data quality validation endpoints
- `test_data_profiling_endpoints.py` - Data profiling endpoints
- `test_ml_pipeline_endpoints.py` - ML pipeline management endpoints
- `test_health_endpoints.py` - Health check endpoints

### Test Structure

Each test file follows a consistent structure:

1. **Unit Tests** - Individual endpoint functionality
2. **Integration Tests** - Cross-service interactions
3. **Security Tests** - Authentication, authorization, input validation
4. **Performance Tests** - Load testing and response times

## Running Tests

### All API Tests
```bash
pytest tests/api/ -v
```

### Specific Test Categories
```bash
# Unit tests only
pytest tests/api/ -m unit

# Integration tests
pytest tests/api/ -m integration

# Security tests
pytest tests/api/ -m security

# Performance tests
pytest tests/api/ -m performance
```

### Individual Test Files
```bash
# Authentication tests
pytest tests/api/test_auth_endpoints.py -v

# Data quality tests
pytest tests/api/test_data_quality_endpoints.py -v

# Data profiling tests
pytest tests/api/test_data_profiling_endpoints.py -v

# ML pipeline tests
pytest tests/api/test_ml_pipeline_endpoints.py -v
```

## Test Coverage

The test suite covers:

### Authentication & Authorization
- User registration and login
- JWT token management
- Multi-factor authentication (MFA)
- API key management
- Password reset functionality
- Role-based access control (RBAC)

### Data Quality
- Dataset validation
- Quality rule management
- Quality monitoring
- Alert configuration
- Engine metrics

### Data Profiling
- Dataset profiling
- Profile comparison
- Profile management
- Cache operations
- Engine performance

### ML Pipelines
- Pipeline creation and management
- Pipeline execution
- Model deployment
- Execution monitoring
- Resource metrics

### Infrastructure
- Health checks
- System metrics
- Configuration management
- Error handling

## Test Configuration

Tests are configured via `pytest.ini` with:
- Coverage reporting
- Test timeouts
- Async support
- Test markers
- Environment variables

## Mock Services

Tests use comprehensive mocking for:
- Authentication services
- Database connections
- External API calls
- ML model operations
- Cache operations

## Security Testing

Security tests include:
- SQL injection prevention
- XSS prevention
- Rate limiting
- Input validation
- Password strength validation
- Authentication bypass attempts

## Performance Testing

Performance tests validate:
- Response times
- Concurrent request handling
- Large dataset processing
- Memory usage
- Database query optimization

## Continuous Integration

Tests are designed to run in CI/CD pipelines with:
- Parallel execution
- Detailed reporting
- Failure analysis
- Coverage tracking

## Issue #141 Compliance

This test suite ensures complete coverage of Issue #141 requirements:
- ✅ FastAPI application structure
- ✅ JWT authentication system
- ✅ Database integration (PostgreSQL, Redis, InfluxDB)
- ✅ RESTful API endpoints
- ✅ Security middleware
- ✅ Error handling
- ✅ Logging and monitoring
- ✅ Docker/Kubernetes configuration
- ✅ Comprehensive test coverage