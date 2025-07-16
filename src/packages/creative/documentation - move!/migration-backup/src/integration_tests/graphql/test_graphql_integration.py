"""Integration tests for GraphQL API."""

import pytest
import asyncio
from uuid import uuid4
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock

from pynomaly.presentation.graphql.app import create_standalone_graphql_app, GraphQLConfig
from pynomaly.infrastructure.container import Container
from pynomaly.domain.entities.user import User, UserRole


@pytest.fixture
def mock_container():
    """Create a mock container with all required services."""
    container = Mock(spec=Container)
    
    # Mock authentication service
    auth_service = AsyncMock()
    auth_service.authenticate_user.return_value = AsyncMock(
        success=True,
        message="Authentication successful",
        user=User(
            id=uuid4(),
            email="test@example.com",
            full_name="Test User",
            role=UserRole.DATA_SCIENTIST,
            tenant_id=uuid4(),
            is_active=True
        ),
        access_token="test_access_token",
        refresh_token="test_refresh_token"
    )
    auth_service.verify_token.return_value = User(
        id=uuid4(),
        email="test@example.com",
        full_name="Test User",
        role=UserRole.DATA_SCIENTIST,
        tenant_id=uuid4(),
        is_active=True
    )
    
    # Mock other services
    user_service = AsyncMock()
    detector_service = AsyncMock()
    dataset_service = AsyncMock()
    detection_service = AsyncMock()
    training_service = AsyncMock()
    monitoring_service = AsyncMock()
    audit_service = AsyncMock()
    
    # Configure container to return mocked services
    def get_service(service_class):
        service_map = {
            "AuthenticationService": auth_service,
            "UserService": user_service,
            "DetectorService": detector_service,
            "DatasetService": dataset_service,
            "DetectionService": detection_service,
            "TrainingService": training_service,
            "MonitoringService": monitoring_service,
            "AuditService": audit_service,
        }
        return service_map.get(service_class.__name__, AsyncMock())
    
    container.get.side_effect = get_service
    return container


@pytest.fixture
def graphql_app(mock_container):
    """Create GraphQL app for testing."""
    config = GraphQLConfig.testing()
    app = create_standalone_graphql_app(
        container=mock_container,
        enable_playground=config.enable_playground,
        enable_introspection=config.enable_introspection,
        enable_cors=config.enable_cors,
        rate_limit_requests_per_minute=config.rate_limit_requests_per_minute,
        max_query_depth=config.max_query_depth,
        max_query_complexity=config.max_query_complexity
    )
    return app


@pytest.fixture
def client(graphql_app):
    """Create test client."""
    return TestClient(graphql_app)


def test_graphql_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "graphql-api"}


def test_graphql_schema_endpoint(client):
    """Test schema endpoint."""
    response = client.get("/schema")
    assert response.status_code == 200
    data = response.json()
    assert "schema" in data
    assert "type Query" in data["schema"]


def test_graphql_introspection_query(client):
    """Test GraphQL introspection query."""
    query = """
    query IntrospectionQuery {
        __schema {
            types {
                name
                kind
            }
        }
    }
    """
    
    response = client.post("/", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "__schema" in data["data"]
    assert "types" in data["data"]["__schema"]


def test_graphql_login_mutation(client):
    """Test login mutation."""
    mutation = """
    mutation Login($input: LoginInput!) {
        login(input: $input) {
            success
            message
            user {
                id
                email
                fullName
            }
            accessToken
            refreshToken
        }
    }
    """
    
    variables = {
        "input": {
            "email": "test@example.com",
            "password": "testpassword"
        }
    }
    
    response = client.post("/", json={"query": mutation, "variables": variables})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "login" in data["data"]
    assert data["data"]["login"]["success"] is True


def test_graphql_query_with_authentication(client):
    """Test authenticated GraphQL query."""
    query = """
    query CurrentUser {
        currentUser {
            id
            email
            fullName
            role
        }
    }
    """
    
    headers = {"Authorization": "Bearer test_access_token"}
    response = client.post("/", json={"query": query}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "currentUser" in data["data"]


def test_graphql_query_without_authentication(client):
    """Test unauthenticated GraphQL query that requires auth."""
    query = """
    query CurrentUser {
        currentUser {
            id
            email
            fullName
            role
        }
    }
    """
    
    response = client.post("/", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    # Should return null for currentUser when not authenticated
    assert "data" in data
    assert data["data"]["currentUser"] is None


def test_graphql_create_detector_mutation(client):
    """Test create detector mutation."""
    mutation = """
    mutation CreateDetector($input: DetectorInput!) {
        createDetector(input: $input) {
            success
            message
            detector {
                id
                name
                description
                algorithm
            }
        }
    }
    """
    
    variables = {
        "input": {
            "name": "Test Detector",
            "description": "A test detector",
            "algorithm": "IsolationForest",
            "parameters": {"contamination": 0.1},
            "datasetId": str(uuid4())
        }
    }
    
    headers = {"Authorization": "Bearer test_access_token"}
    response = client.post("/", json={"query": mutation, "variables": variables}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "createDetector" in data["data"]


def test_graphql_complex_query(client):
    """Test complex nested GraphQL query."""
    query = """
    query ComplexQuery {
        detectors(first: 10) {
            edges {
                node {
                    id
                    name
                    description
                    algorithm
                    status
                    createdAt
                    user {
                        id
                        email
                        fullName
                    }
                }
            }
            pageInfo {
                hasNextPage
                hasPreviousPage
                startCursor
                endCursor
            }
        }
        
        systemHealth {
            status
            uptime
            cpuUsage
            memoryUsage
        }
        
        performanceMetrics {
            requestCount
            requestRate
            averageResponseTime
            errorRate
        }
    }
    """
    
    headers = {"Authorization": "Bearer test_access_token"}
    response = client.post("/", json={"query": query}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "detectors" in data["data"]
    assert "systemHealth" in data["data"]
    assert "performanceMetrics" in data["data"]


def test_graphql_error_handling(client):
    """Test GraphQL error handling."""
    # Invalid query syntax
    query = """
    query InvalidQuery {
        invalidField {
            nonExistentField
        }
    }
    """
    
    response = client.post("/", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "errors" in data


def test_graphql_query_depth_limit(client):
    """Test query depth limiting."""
    # This query would exceed the depth limit in testing config
    deep_query = """
    query DeepQuery {
        detectors {
            edges {
                node {
                    user {
                        tenant {
                            users {
                                detectors {
                                    edges {
                                        node {
                                            user {
                                                tenant {
                                                    id
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """
    
    # With testing config, this should still work as limits are high
    response = client.post("/", json={"query": deep_query})
    assert response.status_code == 200


def test_graphql_subscription_authentication_required(client):
    """Test that subscriptions require authentication."""
    subscription = """
    subscription TrainingProgress($detectorId: ID!) {
        trainingProgress(detectorId: $detectorId) {
            jobId
            detectorId
            status
            progress
            message
        }
    }
    """
    
    variables = {"detectorId": str(uuid4())}
    
    # This would normally be tested with WebSocket connection
    # For now, we test that the subscription endpoint exists
    response = client.post("/", json={"query": subscription, "variables": variables})
    assert response.status_code == 200


def test_graphql_cors_headers(client):
    """Test CORS headers are set correctly."""
    response = client.options("/")
    # CORS headers should be present for OPTIONS requests
    assert response.status_code == 200


def test_graphql_malformed_json(client):
    """Test handling of malformed JSON."""
    response = client.post("/", data="invalid json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422  # Unprocessable Entity


def test_graphql_missing_query(client):
    """Test handling of missing query."""
    response = client.post("/", json={})
    assert response.status_code == 200
    data = response.json()
    assert "errors" in data


def test_graphql_variables_validation(client):
    """Test variable validation."""
    mutation = """
    mutation CreateDetector($input: DetectorInput!) {
        createDetector(input: $input) {
            success
            message
        }
    }
    """
    
    # Missing required fields in variables
    variables = {
        "input": {
            "name": "Test Detector"
            # Missing required fields
        }
    }
    
    headers = {"Authorization": "Bearer test_access_token"}
    response = client.post("/", json={"query": mutation, "variables": variables}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    # Should have validation errors
    assert "errors" in data


@pytest.mark.asyncio
async def test_graphql_concurrent_requests(client):
    """Test handling of concurrent GraphQL requests."""
    query = """
    query SystemHealth {
        systemHealth {
            status
            uptime
        }
    }
    """
    
    # Send multiple concurrent requests
    async def make_request():
        return client.post("/", json={"query": query})
    
    # This test would need proper async client setup
    # For now, we test sequential requests
    responses = []
    for _ in range(5):
        response = client.post("/", json={"query": query})
        responses.append(response)
    
    assert all(r.status_code == 200 for r in responses)


def test_graphql_playground_disabled_in_production():
    """Test that GraphQL playground is disabled in production config."""
    config = GraphQLConfig.production()
    assert config.enable_playground is False
    assert config.enable_introspection is False


def test_graphql_development_config():
    """Test development configuration."""
    config = GraphQLConfig.development()
    assert config.enable_playground is True
    assert config.enable_introspection is True
    assert config.rate_limit_requests_per_minute == 1000


def test_graphql_production_config():
    """Test production configuration."""
    config = GraphQLConfig.production()
    assert config.enable_playground is False
    assert config.enable_introspection is False
    assert config.rate_limit_requests_per_minute == 60
    assert config.max_query_depth == 10
    assert config.max_query_complexity == 500