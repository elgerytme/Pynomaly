# Test Infrastructure Documentation

This document explains the comprehensive test infrastructure setup in `conftest.py` and how to use the configured fixtures for testing FastAPI applications with proper authentication, database isolation, and async support.

## Overview

The test infrastructure provides:

1. **FastAPI Application Fixtures**: Pre-configured app and TestClient instances
2. **Authentication Fixtures**: JWT tokens and auth headers for API testing
3. **Database Fixtures**: In-memory SQLite sessions with proper isolation
4. **Async Support**: pytest-asyncio configuration for async route testing
5. **Cleanup Hooks**: Automatic global state reset between tests

## Key Fixtures

### Core Application Fixtures

```python
def test_api_endpoint(app, client):
    """Test using app and client fixtures."""
    assert app is not None
    response = client.get("/api/v1/health")
    assert response.status_code == 200
```

**Available Fixtures:**
- `app`: FastAPI application instance
- `client`: TestClient for making HTTP requests
- `container`: Dependency injection container
- `test_settings`: Test configuration settings

### Authentication Fixtures

```python
def test_authenticated_endpoint(client, auth_header):
    """Test using auth_header fixture for authenticated requests."""
    response = client.get("/api/v1/protected", headers=auth_header)
    assert response.status_code == 200

def test_user_permissions(client, user_auth_header):
    """Test using user-level auth header."""
    response = client.get("/api/v1/user-data", headers=user_auth_header)
    assert response.status_code == 200
```

**Available Authentication Fixtures:**
- `auth_service`: JWT authentication service
- `admin_token`: JWT token for admin user
- `user_token`: JWT token for regular user
- `auth_header`: Authorization header with admin token
- `user_auth_header`: Authorization header with user token
- `api_key_header`: API key header for authentication

### Database Fixtures

```python
def test_database_operations(db_session):
    """Test using synchronous database session."""
    result = db_session.execute("SELECT 1")
    assert result is not None

@pytest.mark.asyncio
async def test_async_database_operations(async_db_session):
    """Test using asynchronous database session."""
    result = await async_db_session.execute("SELECT 1")
    assert result is not None
```

**Available Database Fixtures:**
- `db_engine`: SQLite in-memory database engine
- `db_session`: Synchronous database session with auto-rollback
- `async_db_engine`: Async SQLite database engine
- `async_db_session`: Async database session with auto-rollback
- `session_factory`: Session factory for repository testing

### Async Testing with pytest-asyncio

The test infrastructure is configured with `asyncio_mode = auto` in `pytest.ini`, enabling automatic async test detection.

```python
@pytest.mark.asyncio
async def test_async_endpoint(client, auth_header):
    """Test async endpoints with proper markers."""
    # Use async client for true async testing if needed
    response = client.get("/api/v1/async-endpoint", headers=auth_header)
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_async_service(async_db_session):
    """Test async services and database operations."""
    # Your async service calls here
    result = await some_async_service.process_data(async_db_session)
    assert result is not None
```

## Test Isolation and Cleanup

### Automatic Global State Reset

The `reset_global_state` fixture automatically runs before and after each test to ensure isolation:

```python
def test_isolation_example_1(auth_service):
    """First test that modifies global state."""
    auth_service.create_user("testuser1", "test1@example.com", "password")
    assert len(auth_service._users) >= 2

def test_isolation_example_2(auth_service):
    """Second test that should not see previous changes."""
    # Should only see default admin user due to automatic reset
    users = list(auth_service._users.keys())
    assert len(users) == 1  # Only admin user
```

### Manual Isolation

For tests requiring stronger isolation:

```python
def test_with_isolated_environment(isolated_test_environment):
    """Test with completely isolated environment."""
    temp_dir = isolated_test_environment["temp_dir"]
    # Test operations in isolated temp directory
    assert os.path.exists(temp_dir)
```

## Configuration

### pytest.ini Configuration

```ini
[tool:pytest]
# Async support
asyncio_mode = auto

# Test discovery
testpaths = tests
python_files = test_*.py *_test.py

# Coverage and reporting
addopts = 
    --strict-markers
    --cov=src/pynomaly
    --cov-report=html:reports/coverage
```

### Environment Variables

The test fixtures automatically set these environment variables:

```python
os.environ.update({
    "PYNOMALY_APP_ENVIRONMENT": "test",
    "PYNOMALY_DATABASE_URL": "sqlite:///:memory:",
    "PYNOMALY_SECRET_KEY": "test-secret-key",
    "PYNOMALY_AUTH_ENABLED": "true",
    "PYNOMALY_CACHE_ENABLED": "false",
})
```

## Best Practices

### 1. Use Appropriate Fixtures

```python
# Good: Use specific fixtures for what you need
def test_authenticated_api(client, auth_header):
    response = client.post("/api/v1/data", json=data, headers=auth_header)

# Avoid: Using more fixtures than necessary
def test_simple_logic(complex_fixture_not_needed):
    result = simple_function()
    assert result == expected
```

### 2. Async Test Patterns

```python
# Good: Proper async test structure
@pytest.mark.asyncio
async def test_async_workflow(async_db_session, auth_service):
    # Setup
    user = auth_service.create_user(...)
    
    # Test async operations
    result = await async_service.process(async_db_session, user)
    
    # Assertions
    assert result is not None

# Avoid: Mixing sync and async inappropriately
def test_mixed_operations(db_session, async_db_session):
    # This pattern can cause issues
    pass
```

### 3. Database Testing

```python
# Good: Use appropriate session type
def test_sync_repository(db_session):
    repo = SyncRepository(db_session)
    result = repo.create(data)
    assert result.id is not None

@pytest.mark.asyncio
async def test_async_repository(async_db_session):
    repo = AsyncRepository(async_db_session)
    result = await repo.create(data)
    assert result.id is not None
```

### 4. Authentication Testing

```python
# Good: Test different permission levels
def test_admin_access(client, auth_header):
    response = client.delete("/api/v1/admin/users/1", headers=auth_header)
    assert response.status_code == 200

def test_user_access(client, user_auth_header):
    response = client.delete("/api/v1/admin/users/1", headers=user_auth_header)
    assert response.status_code == 403  # Forbidden

def test_unauthenticated_access(client):
    response = client.delete("/api/v1/admin/users/1")
    assert response.status_code == 401  # Unauthorized
```

## Common Patterns

### API Endpoint Testing

```python
class TestAPIEndpoints:
    """Comprehensive API endpoint testing."""
    
    def test_health_endpoint(self, client):
        """Test public health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_protected_endpoint(self, client, auth_header):
        """Test protected endpoint with authentication."""
        response = client.get("/api/v1/protected", headers=auth_header)
        assert response.status_code == 200
    
    def test_create_resource(self, client, auth_header):
        """Test resource creation."""
        data = {"name": "test", "value": 123}
        response = client.post("/api/v1/resources", 
                             json=data, headers=auth_header)
        assert response.status_code == 201
        assert response.json()["id"] is not None
```

### Database Integration Testing

```python
class TestDatabaseIntegration:
    """Database integration testing patterns."""
    
    def test_repository_crud(self, db_session):
        """Test CRUD operations."""
        repo = Repository(db_session)
        
        # Create
        entity = repo.create({"name": "test"})
        assert entity.id is not None
        
        # Read
        found = repo.get(entity.id)
        assert found.name == "test"
        
        # Update
        updated = repo.update(entity.id, {"name": "updated"})
        assert updated.name == "updated"
        
        # Delete
        repo.delete(entity.id)
        assert repo.get(entity.id) is None
```

### Async Service Testing

```python
class TestAsyncServices:
    """Async service testing patterns."""
    
    @pytest.mark.asyncio
    async def test_async_processing(self, async_db_session):
        """Test async data processing."""
        service = AsyncDataService(async_db_session)
        
        result = await service.process_large_dataset(test_data)
        assert result.status == "completed"
        assert result.processed_count > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, async_db_session):
        """Test concurrent async operations."""
        service = AsyncService(async_db_session)
        
        tasks = [
            service.process_item(item) 
            for item in test_items
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == len(test_items)
        assert all(r.success for r in results)
```

## Troubleshooting

### Common Issues

1. **Auth Service Not Available**: Ensure FastAPI app and auth dependencies are properly installed
2. **Database Errors**: Check that SQLAlchemy models are importable
3. **Async Test Failures**: Ensure `@pytest.mark.asyncio` is used for async tests
4. **Token Errors**: Verify that auth service is properly initialized in fixtures

### Debug Tips

```python
# Enable SQL logging for database debugging
@pytest.fixture
def debug_db_session(db_engine):
    """Database session with SQL logging enabled."""
    engine = create_engine(
        "sqlite:///:memory:",
        echo=True  # Enable SQL logging
    )
    # ... rest of setup

# Enable auth debugging
def test_auth_debug(auth_service, admin_token):
    """Debug authentication issues."""
    print(f"Token: {admin_token}")
    print(f"Users: {auth_service._users}")
    # ... test logic
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_infrastructure_example.py

# Run with coverage
pytest --cov=src/pynomaly

# Run async tests only
pytest -m asyncio

# Run with verbose output
pytest -v
```

### Advanced Test Execution

```bash
# Run integration tests
pytest --integration

# Run slow tests
pytest --runslow

# Run with parallel execution
pytest -n auto

# Run specific test pattern
pytest -k "test_auth"
```

This test infrastructure provides a solid foundation for comprehensive testing of FastAPI applications with proper isolation, authentication, and async support.
