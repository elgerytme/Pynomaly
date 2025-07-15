"""GraphQL mutations for the Pynomaly API."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

import strawberry
from fastapi import HTTPException

from pynomaly.application.services.auth_service import AuthenticationService
from pynomaly.application.services.detector_service import DetectorService
from pynomaly.application.services.dataset_service import DatasetService
from pynomaly.application.services.detection_service import DetectionService
from pynomaly.application.services.training_service import TrainingService
from pynomaly.application.services.user_service import UserService
from pynomaly.domain.entities.user import User
from pynomaly.presentation.graphql.types import (
    AuthResponse,
    DetectorResponse,
    DatasetResponse,
    DetectionResultResponse,
    TrainingResponse,
    UserResponse,
    DetectorInput,
    DatasetInput,
    LoginInput,
    UserInput,
    DetectionInput,
    TrainingInput,
    UpdateUserInput,
    UpdateDetectorInput,
    UpdateDatasetInput,
)


@strawberry.type
class Mutation:
    """GraphQL mutations for the Pynomaly API."""

    # Authentication Mutations
    @strawberry.mutation
    async def login(
        self,
        info: strawberry.Info,
        input: LoginInput
    ) -> AuthResponse:
        """Authenticate a user and return access tokens."""
        try:
            auth_service: AuthenticationService = info.context["container"].get(AuthenticationService)
            
            result = await auth_service.authenticate_user(
                email=input.email,
                password=input.password
            )
            
            if not result.success:
                return AuthResponse(
                    success=False,
                    message=result.message,
                    user=None,
                    access_token=None,
                    refresh_token=None
                )
            
            return AuthResponse(
                success=True,
                message="Authentication successful",
                user=result.user,
                access_token=result.access_token,
                refresh_token=result.refresh_token
            )
            
        except Exception as e:
            return AuthResponse(
                success=False,
                message=f"Authentication failed: {str(e)}",
                user=None,
                access_token=None,
                refresh_token=None
            )

    @strawberry.mutation
    async def logout(
        self,
        info: strawberry.Info,
        refresh_token: str
    ) -> AuthResponse:
        """Logout a user by invalidating their tokens."""
        try:
            auth_service: AuthenticationService = info.context["container"].get(AuthenticationService)
            
            result = await auth_service.logout_user(refresh_token)
            
            return AuthResponse(
                success=result.success,
                message=result.message,
                user=None,
                access_token=None,
                refresh_token=None
            )
            
        except Exception as e:
            return AuthResponse(
                success=False,
                message=f"Logout failed: {str(e)}",
                user=None,
                access_token=None,
                refresh_token=None
            )

    @strawberry.mutation
    async def refresh_token(
        self,
        info: strawberry.Info,
        refresh_token: str
    ) -> AuthResponse:
        """Refresh access token using refresh token."""
        try:
            auth_service: AuthenticationService = info.context["container"].get(AuthenticationService)
            
            result = await auth_service.refresh_access_token(refresh_token)
            
            if not result.success:
                return AuthResponse(
                    success=False,
                    message=result.message,
                    user=None,
                    access_token=None,
                    refresh_token=None
                )
            
            return AuthResponse(
                success=True,
                message="Token refreshed successfully",
                user=result.user,
                access_token=result.access_token,
                refresh_token=result.refresh_token
            )
            
        except Exception as e:
            return AuthResponse(
                success=False,
                message=f"Token refresh failed: {str(e)}",
                user=None,
                access_token=None,
                refresh_token=None
            )

    # User Management Mutations
    @strawberry.mutation
    async def create_user(
        self,
        info: strawberry.Info,
        input: UserInput
    ) -> UserResponse:
        """Create a new user."""
        try:
            user_service: UserService = info.context["container"].get(UserService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("user.create"):
                return UserResponse(
                    success=False,
                    message="Insufficient permissions to create users",
                    user=None
                )
            
            user = await user_service.create_user(
                email=input.email,
                password=input.password,
                full_name=input.full_name,
                role=input.role,
                tenant_id=current_user.tenant_id
            )
            
            return UserResponse(
                success=True,
                message="User created successfully",
                user=user
            )
            
        except Exception as e:
            return UserResponse(
                success=False,
                message=f"Failed to create user: {str(e)}",
                user=None
            )

    @strawberry.mutation
    async def update_user(
        self,
        info: strawberry.Info,
        user_id: strawberry.ID,
        input: UpdateUserInput
    ) -> UserResponse:
        """Update an existing user."""
        try:
            user_service: UserService = info.context["container"].get(UserService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("user.update"):
                return UserResponse(
                    success=False,
                    message="Insufficient permissions to update users",
                    user=None
                )
            
            user = await user_service.update_user(
                user_id=UUID(str(user_id)),
                full_name=input.full_name,
                role=input.role,
                is_active=input.is_active
            )
            
            return UserResponse(
                success=True,
                message="User updated successfully",
                user=user
            )
            
        except Exception as e:
            return UserResponse(
                success=False,
                message=f"Failed to update user: {str(e)}",
                user=None
            )

    @strawberry.mutation
    async def delete_user(
        self,
        info: strawberry.Info,
        user_id: strawberry.ID
    ) -> UserResponse:
        """Delete a user."""
        try:
            user_service: UserService = info.context["container"].get(UserService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("user.delete"):
                return UserResponse(
                    success=False,
                    message="Insufficient permissions to delete users",
                    user=None
                )
            
            success = await user_service.delete_user(UUID(str(user_id)))
            
            return UserResponse(
                success=success,
                message="User deleted successfully" if success else "Failed to delete user",
                user=None
            )
            
        except Exception as e:
            return UserResponse(
                success=False,
                message=f"Failed to delete user: {str(e)}",
                user=None
            )

    # Detector Management Mutations
    @strawberry.mutation
    async def create_detector(
        self,
        info: strawberry.Info,
        input: DetectorInput
    ) -> DetectorResponse:
        """Create a new anomaly detector."""
        try:
            detector_service: DetectorService = info.context["container"].get(DetectorService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("detector.create"):
                return DetectorResponse(
                    success=False,
                    message="Insufficient permissions to create detectors",
                    detector=None
                )
            
            detector = await detector_service.create_detector(
                name=input.name,
                description=input.description,
                algorithm=input.algorithm,
                parameters=input.parameters,
                dataset_id=UUID(str(input.dataset_id)),
                user_id=current_user.id,
                tenant_id=current_user.tenant_id
            )
            
            return DetectorResponse(
                success=True,
                message="Detector created successfully",
                detector=detector
            )
            
        except Exception as e:
            return DetectorResponse(
                success=False,
                message=f"Failed to create detector: {str(e)}",
                detector=None
            )

    @strawberry.mutation
    async def update_detector(
        self,
        info: strawberry.Info,
        detector_id: strawberry.ID,
        input: UpdateDetectorInput
    ) -> DetectorResponse:
        """Update an existing detector."""
        try:
            detector_service: DetectorService = info.context["container"].get(DetectorService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("detector.update"):
                return DetectorResponse(
                    success=False,
                    message="Insufficient permissions to update detectors",
                    detector=None
                )
            
            detector = await detector_service.update_detector(
                detector_id=UUID(str(detector_id)),
                name=input.name,
                description=input.description,
                parameters=input.parameters,
                is_active=input.is_active
            )
            
            return DetectorResponse(
                success=True,
                message="Detector updated successfully",
                detector=detector
            )
            
        except Exception as e:
            return DetectorResponse(
                success=False,
                message=f"Failed to update detector: {str(e)}",
                detector=None
            )

    @strawberry.mutation
    async def delete_detector(
        self,
        info: strawberry.Info,
        detector_id: strawberry.ID
    ) -> DetectorResponse:
        """Delete a detector."""
        try:
            detector_service: DetectorService = info.context["container"].get(DetectorService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("detector.delete"):
                return DetectorResponse(
                    success=False,
                    message="Insufficient permissions to delete detectors",
                    detector=None
                )
            
            success = await detector_service.delete_detector(UUID(str(detector_id)))
            
            return DetectorResponse(
                success=success,
                message="Detector deleted successfully" if success else "Failed to delete detector",
                detector=None
            )
            
        except Exception as e:
            return DetectorResponse(
                success=False,
                message=f"Failed to delete detector: {str(e)}",
                detector=None
            )

    @strawberry.mutation
    async def train_detector(
        self,
        info: strawberry.Info,
        input: TrainingInput
    ) -> TrainingResponse:
        """Start training a detector."""
        try:
            training_service: TrainingService = info.context["container"].get(TrainingService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("detector.train"):
                return TrainingResponse(
                    success=False,
                    message="Insufficient permissions to train detectors",
                    training_job=None
                )
            
            training_job = await training_service.start_training(
                detector_id=UUID(str(input.detector_id)),
                dataset_id=UUID(str(input.dataset_id)),
                parameters=input.parameters,
                user_id=current_user.id
            )
            
            return TrainingResponse(
                success=True,
                message="Training started successfully",
                training_job=training_job
            )
            
        except Exception as e:
            return TrainingResponse(
                success=False,
                message=f"Failed to start training: {str(e)}",
                training_job=None
            )

    # Dataset Management Mutations
    @strawberry.mutation
    async def create_dataset(
        self,
        info: strawberry.Info,
        input: DatasetInput
    ) -> DatasetResponse:
        """Create a new dataset."""
        try:
            dataset_service: DatasetService = info.context["container"].get(DatasetService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("dataset.create"):
                return DatasetResponse(
                    success=False,
                    message="Insufficient permissions to create datasets",
                    dataset=None
                )
            
            dataset = await dataset_service.create_dataset(
                name=input.name,
                description=input.description,
                file_path=input.file_path,
                format=input.format,
                user_id=current_user.id,
                tenant_id=current_user.tenant_id
            )
            
            return DatasetResponse(
                success=True,
                message="Dataset created successfully",
                dataset=dataset
            )
            
        except Exception as e:
            return DatasetResponse(
                success=False,
                message=f"Failed to create dataset: {str(e)}",
                dataset=None
            )

    @strawberry.mutation
    async def update_dataset(
        self,
        info: strawberry.Info,
        dataset_id: strawberry.ID,
        input: UpdateDatasetInput
    ) -> DatasetResponse:
        """Update an existing dataset."""
        try:
            dataset_service: DatasetService = info.context["container"].get(DatasetService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("dataset.update"):
                return DatasetResponse(
                    success=False,
                    message="Insufficient permissions to update datasets",
                    dataset=None
                )
            
            dataset = await dataset_service.update_dataset(
                dataset_id=UUID(str(dataset_id)),
                name=input.name,
                description=input.description,
                is_active=input.is_active
            )
            
            return DatasetResponse(
                success=True,
                message="Dataset updated successfully",
                dataset=dataset
            )
            
        except Exception as e:
            return DatasetResponse(
                success=False,
                message=f"Failed to update dataset: {str(e)}",
                dataset=None
            )

    @strawberry.mutation
    async def delete_dataset(
        self,
        info: strawberry.Info,
        dataset_id: strawberry.ID
    ) -> DatasetResponse:
        """Delete a dataset."""
        try:
            dataset_service: DatasetService = info.context["container"].get(DatasetService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("dataset.delete"):
                return DatasetResponse(
                    success=False,
                    message="Insufficient permissions to delete datasets",
                    dataset=None
                )
            
            success = await dataset_service.delete_dataset(UUID(str(dataset_id)))
            
            return DatasetResponse(
                success=success,
                message="Dataset deleted successfully" if success else "Failed to delete dataset",
                dataset=None
            )
            
        except Exception as e:
            return DatasetResponse(
                success=False,
                message=f"Failed to delete dataset: {str(e)}",
                dataset=None
            )

    # Detection Mutations
    @strawberry.mutation
    async def run_detection(
        self,
        info: strawberry.Info,
        input: DetectionInput
    ) -> DetectionResultResponse:
        """Run anomaly detection on data."""
        try:
            detection_service: DetectionService = info.context["container"].get(DetectionService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("detection.run"):
                return DetectionResultResponse(
                    success=False,
                    message="Insufficient permissions to run detection",
                    detection_result=None
                )
            
            detection_result = await detection_service.run_detection(
                detector_id=UUID(str(input.detector_id)),
                data=input.data,
                user_id=current_user.id
            )
            
            return DetectionResultResponse(
                success=True,
                message="Detection completed successfully",
                detection_result=detection_result
            )
            
        except Exception as e:
            return DetectionResultResponse(
                success=False,
                message=f"Detection failed: {str(e)}",
                detection_result=None
            )

    @strawberry.mutation
    async def batch_detection(
        self,
        info: strawberry.Info,
        detector_id: strawberry.ID,
        dataset_id: strawberry.ID
    ) -> DetectionResultResponse:
        """Run batch detection on a dataset."""
        try:
            detection_service: DetectionService = info.context["container"].get(DetectionService)
            current_user: User = info.context.get("user")
            
            if not current_user or not current_user.has_permission("detection.batch"):
                return DetectionResultResponse(
                    success=False,
                    message="Insufficient permissions to run batch detection",
                    detection_result=None
                )
            
            detection_result = await detection_service.run_batch_detection(
                detector_id=UUID(str(detector_id)),
                dataset_id=UUID(str(dataset_id)),
                user_id=current_user.id
            )
            
            return DetectionResultResponse(
                success=True,
                message="Batch detection completed successfully",
                detection_result=detection_result
            )
            
        except Exception as e:
            return DetectionResultResponse(
                success=False,
                message=f"Batch detection failed: {str(e)}",
                detection_result=None
            )