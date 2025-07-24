"""Service contracts and interfaces for microservices preparation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid


class ServiceType(str, Enum):
    """Service types in the microservices architecture."""
    DATA_PROCESSING = "data_processing"
    AI_ML = "ai_ml"
    MLOPS = "mlops"
    MONITORING = "monitoring"
    API_GATEWAY = "api_gateway"


class ServiceHealth(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# =============================================================================
# Base Service Contracts
# =============================================================================

class ServiceRequest(BaseModel):
    """Base request model for all services."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    source_service: Optional[str] = None
    user_id: Optional[str] = None


class ServiceResponse(BaseModel):
    """Base response model for all services."""
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    service_name: str
    status: ServiceHealth
    timestamp: datetime
    version: str
    uptime_seconds: int
    dependencies: Dict[str, ServiceHealth]
    metrics: Dict[str, Any]


# =============================================================================
# Data Processing Service Contracts
# =============================================================================

class DataValidationRequest(ServiceRequest):
    """Request for data validation."""
    data: List[List[float]]
    schema: Optional[Dict[str, Any]] = None
    validation_rules: Optional[List[str]] = None


class DataValidationResponse(ServiceResponse):
    """Response from data validation."""
    is_valid: bool
    validation_errors: List[str]
    validated_data: Optional[List[List[float]]] = None
    schema_info: Optional[Dict[str, Any]] = None


class DataPreprocessingRequest(ServiceRequest):
    """Request for data preprocessing."""
    data: List[List[float]]
    preprocessing_steps: List[str] = Field(default=["normalize", "outlier_removal"])
    feature_names: Optional[List[str]] = None


class DataPreprocessingResponse(ServiceResponse):
    """Response from data preprocessing."""
    processed_data: List[List[float]]
    preprocessing_metadata: Dict[str, Any]
    feature_info: Optional[Dict[str, Any]] = None


class IDataProcessingService(ABC):
    """Interface for data processing service."""
    
    @abstractmethod
    async def validate_data(self, request: DataValidationRequest) -> DataValidationResponse:
        """Validate input data against schema and rules."""
        pass
    
    @abstractmethod
    async def preprocess_data(self, request: DataPreprocessingRequest) -> DataPreprocessingResponse:
        """Preprocess data for machine learning."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResponse:
        """Check service health."""
        pass


# =============================================================================
# AI/ML Service Contracts
# =============================================================================

class ModelTrainingRequest(ServiceRequest):
    """Request for model training."""
    data: List[List[float]]
    algorithm: str
    hyperparameters: Dict[str, Any]
    model_name: str
    training_config: Optional[Dict[str, Any]] = None


class ModelTrainingResponse(ServiceResponse):
    """Response from model training."""
    model_id: str
    training_metrics: Dict[str, float]
    model_metadata: Dict[str, Any]
    training_duration_seconds: float


class AnomalyDetectionRequest(ServiceRequest):
    """Request for anomaly detection."""
    data: List[List[float]]
    model_id: Optional[str] = None
    algorithm: str = "isolation_forest"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AnomalyDetectionResponse(ServiceResponse):
    """Response from anomaly detection."""
    predictions: List[int]
    anomaly_scores: List[float]
    anomaly_indices: List[int]
    anomaly_count: int
    algorithm_used: str
    confidence: Optional[float] = None


class IMLService(ABC):
    """Interface for AI/ML service."""
    
    @abstractmethod
    async def train_model(self, request: ModelTrainingRequest) -> ModelTrainingResponse:
        """Train a new anomaly detection model."""
        pass
    
    @abstractmethod
    async def detect_anomalies(self, request: AnomalyDetectionRequest) -> AnomalyDetectionResponse:
        """Detect anomalies in provided data."""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a trained model."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResponse:
        """Check service health."""
        pass


# =============================================================================
# MLOps Service Contracts
# =============================================================================

class ExperimentRequest(ServiceRequest):
    """Request to create/update experiment."""
    experiment_name: str
    model_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    tags: Optional[Dict[str, str]] = None


class ExperimentResponse(ServiceResponse):
    """Response from experiment operations."""
    experiment_id: str
    experiment_name: str
    status: str
    created_at: datetime
    updated_at: datetime


class ModelDeploymentRequest(ServiceRequest):
    """Request for model deployment."""
    model_id: str
    deployment_target: str = "production"
    deployment_config: Dict[str, Any]
    rollback_config: Optional[Dict[str, Any]] = None


class ModelDeploymentResponse(ServiceResponse):
    """Response from model deployment."""
    deployment_id: str
    model_id: str
    deployment_status: str
    endpoint_url: Optional[str] = None
    rollback_available: bool


class IMLOpsService(ABC):
    """Interface for MLOps service."""
    
    @abstractmethod
    async def create_experiment(self, request: ExperimentRequest) -> ExperimentResponse:
        """Create a new experiment."""
        pass
    
    @abstractmethod
    async def deploy_model(self, request: ModelDeploymentRequest) -> ModelDeploymentResponse:
        """Deploy model to specified environment."""
        pass
    
    @abstractmethod
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results and metrics."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResponse:
        """Check service health."""
        pass


# =============================================================================
# Monitoring Service Contracts
# =============================================================================

class MetricData(BaseModel):
    """Metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = Field(default_factory=dict)
    unit: Optional[str] = None


class MetricsRequest(ServiceRequest):
    """Request to record metrics."""
    metrics: List[MetricData]
    service_name: str


class MetricsResponse(ServiceResponse):
    """Response from metrics recording."""
    metrics_recorded: int
    storage_location: str


class AlertRequest(ServiceRequest):
    """Request to create alert."""
    alert_name: str
    condition: str
    severity: str
    notification_channels: List[str]
    metadata: Optional[Dict[str, Any]] = None


class AlertResponse(ServiceResponse):
    """Response from alert creation."""
    alert_id: str
    alert_name: str
    status: str
    created_at: datetime


class IMonitoringService(ABC):
    """Interface for monitoring service."""
    
    @abstractmethod
    async def record_metrics(self, request: MetricsRequest) -> MetricsResponse:
        """Record metrics data."""
        pass
    
    @abstractmethod
    async def create_alert(self, request: AlertRequest) -> AlertResponse:
        """Create monitoring alert."""
        pass
    
    @abstractmethod
    async def get_service_health(self, service_name: str) -> HealthCheckResponse:
        """Get health status of a service."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResponse:
        """Check service health."""
        pass


# =============================================================================
# Service Communication Patterns
# =============================================================================

class EventMessage(BaseModel):
    """Base event message for async communication."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str
    correlation_id: Optional[str] = None
    payload: Dict[str, Any]


class DataProcessedEvent(EventMessage):
    """Event emitted when data processing is completed."""
    event_type: str = "data.processed"
    data_id: str
    processing_results: Dict[str, Any]


class ModelTrainedEvent(EventMessage):
    """Event emitted when model training is completed."""
    event_type: str = "model.trained"
    model_id: str
    training_metrics: Dict[str, float]


class AnomalyDetectedEvent(EventMessage):
    """Event emitted when anomalies are detected."""
    event_type: str = "anomaly.detected"
    detection_id: str
    anomaly_count: int
    severity: str


# =============================================================================
# Service Registry and Discovery
# =============================================================================

class ServiceRegistration(BaseModel):
    """Service registration information."""
    service_name: str
    service_type: ServiceType
    version: str
    host: str
    port: int
    health_check_endpoint: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=datetime.utcnow)


class ServiceDiscoveryResponse(BaseModel):
    """Response from service discovery."""
    service_name: str
    instances: List[ServiceRegistration]
    load_balancing_strategy: str = "round_robin"


class IServiceRegistry(ABC):
    """Interface for service registry."""
    
    @abstractmethod
    async def register_service(self, registration: ServiceRegistration) -> bool:
        """Register a service instance."""
        pass
    
    @abstractmethod
    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance."""
        pass
    
    @abstractmethod
    async def discover_service(self, service_name: str) -> ServiceDiscoveryResponse:
        """Discover available service instances."""
        pass
    
    @abstractmethod
    async def health_check_service(self, service_name: str, instance_id: str) -> ServiceHealth:
        """Check health of a specific service instance."""
        pass


# =============================================================================
# API Gateway Contracts
# =============================================================================

class RouteRule(BaseModel):
    """API Gateway routing rule."""
    path_pattern: str
    method: str
    target_service: str
    target_path: str
    auth_required: bool = True
    rate_limit: Optional[int] = None
    timeout_seconds: int = 30


class GatewayRequest(ServiceRequest):
    """Request processed by API Gateway."""
    path: str
    method: str
    headers: Dict[str, str]
    query_params: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None


class GatewayResponse(ServiceResponse):
    """Response from API Gateway."""
    status_code: int
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    route_info: Dict[str, str]


class IAPIGateway(ABC):
    """Interface for API Gateway."""
    
    @abstractmethod
    async def route_request(self, request: GatewayRequest) -> GatewayResponse:
        """Route request to appropriate service."""
        pass
    
    @abstractmethod
    async def add_route(self, route: RouteRule) -> bool:
        """Add new routing rule."""
        pass
    
    @abstractmethod
    async def remove_route(self, path_pattern: str, method: str) -> bool:
        """Remove routing rule."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheckResponse:
        """Check gateway health."""
        pass


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    success_threshold: int = 3
    timeout_seconds: int = 30


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status."""
    service_name: str
    state: CircuitBreakerState
    failure_count: int
    last_failure_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None


# =============================================================================
# Service Contract Registry
# =============================================================================

class ServiceContractRegistry:
    """Registry for service contracts and interfaces."""
    
    def __init__(self):
        self.contracts = {
            ServiceType.DATA_PROCESSING: {
                "interface": IDataProcessingService,
                "requests": [DataValidationRequest, DataPreprocessingRequest],
                "responses": [DataValidationResponse, DataPreprocessingResponse],
                "events": [DataProcessedEvent]
            },
            ServiceType.AI_ML: {
                "interface": IMLService,
                "requests": [ModelTrainingRequest, AnomalyDetectionRequest],
                "responses": [ModelTrainingResponse, AnomalyDetectionResponse],
                "events": [ModelTrainedEvent, AnomalyDetectedEvent]
            },
            ServiceType.MLOPS: {
                "interface": IMLOpsService,
                "requests": [ExperimentRequest, ModelDeploymentRequest],
                "responses": [ExperimentResponse, ModelDeploymentResponse],
                "events": []
            },
            ServiceType.MONITORING: {
                "interface": IMonitoringService,
                "requests": [MetricsRequest, AlertRequest],
                "responses": [MetricsResponse, AlertResponse],
                "events": []
            },
            ServiceType.API_GATEWAY: {
                "interface": IAPIGateway,
                "requests": [GatewayRequest],
                "responses": [GatewayResponse],
                "events": []
            }
        }
    
    def get_contract(self, service_type: ServiceType) -> Dict[str, Any]:
        """Get contract information for a service type."""
        return self.contracts.get(service_type, {})
    
    def get_interface(self, service_type: ServiceType) -> type:
        """Get interface class for a service type."""
        contract = self.get_contract(service_type)
        return contract.get("interface")
    
    def validate_request(self, service_type: ServiceType, request: Dict[str, Any]) -> bool:
        """Validate request against service contract."""
        contract = self.get_contract(service_type)
        request_types = contract.get("requests", [])
        
        # This is a simplified validation - in practice, you'd want more sophisticated validation
        return len(request_types) > 0


# Global registry instance
service_contract_registry = ServiceContractRegistry()


# =============================================================================
# Contract Testing Support
# =============================================================================

class ContractTest(BaseModel):
    """Contract test definition."""
    service_type: ServiceType
    operation: str
    request_schema: Dict[str, Any]
    response_schema: Dict[str, Any]
    test_cases: List[Dict[str, Any]]


def generate_contract_tests(service_type: ServiceType) -> List[ContractTest]:
    """Generate contract tests for a service type."""
    contract = service_contract_registry.get_contract(service_type)
    tests = []
    
    # Generate tests based on contract definition
    for request_type in contract.get("requests", []):
        test = ContractTest(
            service_type=service_type,
            operation=request_type.__name__,
            request_schema=request_type.schema(),
            response_schema={},  # Would be filled from corresponding response type
            test_cases=[]  # Would be generated based on schema
        )
        tests.append(test)
    
    return tests


if __name__ == "__main__":
    # Example usage and validation
    print("Service Contract Registry initialized")
    
    # Test contract registration
    for service_type in ServiceType:
        contract = service_contract_registry.get_contract(service_type)
        print(f"{service_type.value}: {len(contract.get('requests', []))} request types")
    
    # Test request creation
    sample_request = DataValidationRequest(
        data=[[1.0, 2.0], [3.0, 4.0]],
        validation_rules=["range_check", "null_check"]
    )
    print(f"Sample request: {sample_request.request_id}")