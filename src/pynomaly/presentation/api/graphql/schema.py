"""GraphQL schema definition for Pynomaly API."""

from __future__ import annotations

import strawberry
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from pynomaly.presentation.api.graphql.types import (
    AnomalyDetectionResult,
    Dataset,
    Detector,
    DetectorStatus,
    Model,
    TrainingResult,
    User,
)


@strawberry.type
class Query:
    """GraphQL Query root type."""
    
    @strawberry.field
    async def hello(self) -> str:
        """Simple health check query."""
        return "Hello from Pynomaly GraphQL API!"
    
    @strawberry.field
    async def version(self) -> str:
        """Get API version."""
        return "1.0.0"
    
    # User queries
    @strawberry.field
    async def me(self, info) -> User:
        """Get current authenticated user."""
        from pynomaly.presentation.api.graphql.resolvers.user_resolvers import resolve_current_user
        return await resolve_current_user(info)
    
    @strawberry.field
    async def user(self, info, user_id: str) -> User:
        """Get user by ID."""
        from pynomaly.presentation.api.graphql.resolvers.user_resolvers import resolve_user_by_id
        return await resolve_user_by_id(info, user_id)
    
    @strawberry.field
    async def users(self, info, limit: int = 10, offset: int = 0) -> list[User]:
        """Get list of users."""
        from pynomaly.presentation.api.graphql.resolvers.user_resolvers import resolve_users
        return await resolve_users(info, limit, offset)
    
    # Dataset queries
    @strawberry.field
    async def dataset(self, info, dataset_id: str) -> Dataset:
        """Get dataset by ID."""
        from pynomaly.presentation.api.graphql.resolvers.dataset_resolvers import resolve_dataset_by_id
        return await resolve_dataset_by_id(info, dataset_id)
    
    @strawberry.field
    async def datasets(self, info, limit: int = 10, offset: int = 0) -> list[Dataset]:
        """Get list of datasets."""
        from pynomaly.presentation.api.graphql.resolvers.dataset_resolvers import resolve_datasets
        return await resolve_datasets(info, limit, offset)
    
    # Detector queries
    @strawberry.field
    async def detector(self, info, detector_id: str) -> Detector:
        """Get detector by ID."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_detector_by_id
        return await resolve_detector_by_id(info, detector_id)
    
    @strawberry.field
    async def detectors(self, info, limit: int = 10, offset: int = 0) -> list[Detector]:
        """Get list of detectors."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_detectors
        return await resolve_detectors(info, limit, offset)
    
    # Model queries
    @strawberry.field
    async def model(self, info, model_id: str) -> Model:
        """Get model by ID."""
        from pynomaly.presentation.api.graphql.resolvers.model_resolvers import resolve_model_by_id
        return await resolve_model_by_id(info, model_id)
    
    @strawberry.field
    async def models(self, info, limit: int = 10, offset: int = 0) -> list[Model]:
        """Get list of models."""
        from pynomaly.presentation.api.graphql.resolvers.model_resolvers import resolve_models
        return await resolve_models(info, limit, offset)


@strawberry.type
class Mutation:
    """GraphQL Mutation root type."""
    
    # Authentication mutations
    @strawberry.field
    async def login(self, info, username: str, password: str) -> str:
        """Authenticate user and return JWT token."""
        from pynomaly.presentation.api.graphql.resolvers.auth_resolvers import resolve_login
        return await resolve_login(info, username, password)
    
    @strawberry.field
    async def logout(self, info) -> bool:
        """Logout current user."""
        from pynomaly.presentation.api.graphql.resolvers.auth_resolvers import resolve_logout
        return await resolve_logout(info)
    
    # Dataset mutations
    @strawberry.field
    async def create_dataset(
        self,
        info,
        name: str,
        description: str,
        data_url: str | None = None
    ) -> Dataset:
        """Create a new dataset."""
        from pynomaly.presentation.api.graphql.resolvers.dataset_resolvers import resolve_create_dataset
        return await resolve_create_dataset(info, name, description, data_url)
    
    @strawberry.field
    async def update_dataset(
        self,
        info,
        dataset_id: str,
        name: str | None = None,
        description: str | None = None
    ) -> Dataset:
        """Update an existing dataset."""
        from pynomaly.presentation.api.graphql.resolvers.dataset_resolvers import resolve_update_dataset
        return await resolve_update_dataset(info, dataset_id, name, description)
    
    @strawberry.field
    async def delete_dataset(self, info, dataset_id: str) -> bool:
        """Delete a dataset."""
        from pynomaly.presentation.api.graphql.resolvers.dataset_resolvers import resolve_delete_dataset
        return await resolve_delete_dataset(info, dataset_id)
    
    # Detector mutations
    @strawberry.field
    async def create_detector(
        self,
        info,
        name: str,
        algorithm: str,
        dataset_id: str,
        parameters: str | None = None
    ) -> Detector:
        """Create a new anomaly detector."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_create_detector
        return await resolve_create_detector(info, name, algorithm, dataset_id, parameters)
    
    @strawberry.field
    async def update_detector(
        self,
        info,
        detector_id: str,
        name: str | None = None,
        parameters: str | None = None
    ) -> Detector:
        """Update an existing detector."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_update_detector
        return await resolve_update_detector(info, detector_id, name, parameters)
    
    @strawberry.field
    async def delete_detector(self, info, detector_id: str) -> bool:
        """Delete a detector."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_delete_detector
        return await resolve_delete_detector(info, detector_id)
    
    # Training and detection mutations
    @strawberry.field
    async def train_detector(self, info, detector_id: str) -> TrainingResult:
        """Train an anomaly detector."""
        from pynomaly.presentation.api.graphql.resolvers.training_resolvers import resolve_train_detector
        return await resolve_train_detector(info, detector_id)
    
    @strawberry.field
    async def detect_anomalies(
        self,
        info,
        detector_id: str,
        data: str | None = None,
        dataset_id: str | None = None
    ) -> AnomalyDetectionResult:
        """Run anomaly detection."""
        from pynomaly.presentation.api.graphql.resolvers.detection_resolvers import resolve_detect_anomalies
        return await resolve_detect_anomalies(info, detector_id, data, dataset_id)


@strawberry.type
class Subscription:
    """GraphQL Subscription root type for real-time updates."""
    
    @strawberry.subscription
    async def detector_status_updates(self, info, detector_id: str) -> DetectorStatus:
        """Subscribe to detector status updates."""
        from pynomaly.presentation.api.graphql.resolvers.subscription_resolvers import resolve_detector_status_updates
        async for status in resolve_detector_status_updates(info, detector_id):
            yield status
    
    @strawberry.subscription
    async def training_progress(self, info, detector_id: str) -> TrainingResult:
        """Subscribe to training progress updates."""
        from pynomaly.presentation.api.graphql.resolvers.subscription_resolvers import resolve_training_progress
        async for progress in resolve_training_progress(info, detector_id):
            yield progress
    
    @strawberry.subscription
    async def anomaly_alerts(self, info, detector_id: str | None = None) -> AnomalyDetectionResult:
        """Subscribe to real-time anomaly alerts."""
        from pynomaly.presentation.api.graphql.resolvers.subscription_resolvers import resolve_anomaly_alerts
        async for alert in resolve_anomaly_alerts(info, detector_id):
            yield alert


# Create the schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    subscription_protocols=[
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
        GRAPHQL_WS_PROTOCOL,
    ]
)