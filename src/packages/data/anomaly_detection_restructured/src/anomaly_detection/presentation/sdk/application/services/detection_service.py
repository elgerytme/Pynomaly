"""
Detection Application Service

Orchestrates anomaly detection operations and coordinates between domain and infrastructure layers.
"""

import asyncio
from typing import List, Optional
from uuid import UUID
import logging

from ...domain.entities.detection_request import DetectionRequest
from ...domain.value_objects.algorithm_config import AlgorithmConfig
from ...domain.value_objects.detection_metadata import DetectionMetadata
from ...domain.repositories.detection_repository import DetectionRepository
from ...domain.services.detection_validator import DetectionValidator
from ...domain.exceptions.validation_exceptions import ValidationError, DetectionRequestError
from .dto.detection_dto import DetectionRequestDTO, DetectionResponseDTO
from ...infrastructure.adapters.algorithm_adapter import AlgorithmAdapter


class DetectionService:
    """
    Application service for orchestrating anomaly detection operations.
    
    This service coordinates between the domain layer (business logic),
    infrastructure layer (external systems), and presentation layer (APIs/CLI).
    """
    
    def __init__(
        self,
        detection_repository: DetectionRepository,
        algorithm_adapter: AlgorithmAdapter,
        logger: Optional[logging.Logger] = None
    ):
        self._detection_repository = detection_repository
        self._algorithm_adapter = algorithm_adapter
        self._logger = logger or logging.getLogger(__name__)
    
    async def submit_detection_request(
        self, 
        request_dto: DetectionRequestDTO
    ) -> DetectionResponseDTO:
        """
        Submit a new anomaly detection request.
        
        Args:
            request_dto: Data transfer object containing request details.
            
        Returns:
            DetectionResponseDTO: Response containing request ID and initial status.
            
        Raises:
            ValidationError: If the request is invalid.
            DetectionRequestError: If the request cannot be processed.
        """
        try:
            # Convert DTO to domain entity
            algorithm_config = AlgorithmConfig.from_dict(request_dto.algorithm_config)
            metadata = DetectionMetadata.from_dict(request_dto.metadata or {})
            
            detection_request = DetectionRequest(
                data=request_dto.data,
                algorithm_config=algorithm_config,
                metadata=metadata
            )
            
            # Validate the request
            DetectionValidator.validate_and_raise(detection_request)
            
            # Save the request
            await self._detection_repository.save_request(detection_request)
            
            self._logger.info(f"Detection request {detection_request.id} submitted successfully")
            
            # Start async processing
            asyncio.create_task(self._process_detection_request(detection_request))
            
            return DetectionResponseDTO(
                request_id=str(detection_request.id),
                status="submitted",
                message="Detection request submitted for processing"
            )
            
        except ValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to submit detection request: {str(e)}")
            raise DetectionRequestError("unknown", f"Failed to submit request: {str(e)}")
    
    async def get_detection_status(self, request_id: UUID) -> DetectionResponseDTO:
        """
        Get the status of a detection request.
        
        Args:
            request_id: Unique identifier of the detection request.
            
        Returns:
            DetectionResponseDTO: Current status and details of the request.
            
        Raises:
            DetectionRequestError: If the request cannot be found or accessed.
        """
        try:
            request = await self._detection_repository.get_request_by_id(request_id)
            
            if not request:
                raise DetectionRequestError(
                    str(request_id), 
                    "Detection request not found"
                )
            
            return DetectionResponseDTO(
                request_id=str(request.id),
                status=request.status,
                message=f"Request is {request.status}",
                created_at=request.created_at
            )
            
        except DetectionRequestError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get detection status for {request_id}: {str(e)}")
            raise DetectionRequestError(
                str(request_id), 
                f"Failed to retrieve status: {str(e)}"
            )
    
    async def list_detection_requests(
        self, 
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DetectionResponseDTO]:
        """
        List detection requests with optional filtering.
        
        Args:
            user_id: Optional user ID filter.
            limit: Maximum number of requests to return.
            offset: Number of requests to skip.
            
        Returns:
            List[DetectionResponseDTO]: List of detection request summaries.
        """
        try:
            if user_id:
                requests = await self._detection_repository.get_requests_by_user(user_id)
            else:
                requests = await self._detection_repository.list_requests(limit, offset)
            
            return [
                DetectionResponseDTO(
                    request_id=str(req.id),
                    status=req.status,
                    message=f"Request {req.status}",
                    created_at=req.created_at
                )
                for req in requests
            ]
            
        except Exception as e:
            self._logger.error(f"Failed to list detection requests: {str(e)}")
            raise DetectionRequestError("list", f"Failed to list requests: {str(e)}")
    
    async def cancel_detection_request(self, request_id: UUID) -> bool:
        """
        Cancel a pending detection request.
        
        Args:
            request_id: Unique identifier of the detection request.
            
        Returns:
            bool: True if the request was cancelled successfully.
            
        Raises:
            DetectionRequestError: If the request cannot be cancelled.
        """
        try:
            request = await self._detection_repository.get_request_by_id(request_id)
            
            if not request:
                raise DetectionRequestError(
                    str(request_id), 
                    "Detection request not found"
                )
            
            if request.status not in ["pending", "submitted"]:
                raise DetectionRequestError(
                    str(request_id), 
                    f"Cannot cancel request with status: {request.status}"
                )
            
            await self._detection_repository.update_request_status(request_id, "cancelled")
            
            self._logger.info(f"Detection request {request_id} cancelled")
            return True
            
        except DetectionRequestError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to cancel detection request {request_id}: {str(e)}")
            raise DetectionRequestError(
                str(request_id), 
                f"Failed to cancel request: {str(e)}"
            )
    
    async def _process_detection_request(self, request: DetectionRequest) -> None:
        """
        Process a detection request asynchronously.
        
        Args:
            request: The detection request to process.
        """
        try:
            # Update status to processing
            request.mark_as_processing()
            await self._detection_repository.update_request_status(
                request.id, 
                "processing"
            )
            
            self._logger.info(f"Starting processing for detection request {request.id}")
            
            # Execute the detection algorithm
            result = await self._algorithm_adapter.detect_anomalies(
                data=request.data,
                algorithm_config=request.algorithm_config
            )
            
            # Mark as completed
            request.mark_as_completed()
            await self._detection_repository.update_request_status(
                request.id, 
                "completed"
            )
            
            self._logger.info(f"Detection request {request.id} completed successfully")
            
        except Exception as e:
            # Mark as failed
            error_message = str(e)
            request.mark_as_failed(error_message)
            await self._detection_repository.update_request_status(
                request.id, 
                "failed"
            )
            
            self._logger.error(f"Detection request {request.id} failed: {error_message}")
    
    async def get_algorithm_recommendations(
        self, 
        data: List[float]
    ) -> List[str]:
        """
        Get algorithm recommendations based on data characteristics.
        
        Args:
            data: Input data to analyze.
            
        Returns:
            List[str]: Recommended algorithm types.
        """
        try:
            # Simple heuristic-based recommendations
            data_size = len(data)
            recommendations = []
            
            if data_size < 1000:
                recommendations.extend(["local_outlier_factor", "elliptic_envelope"])
            elif data_size < 10000:
                recommendations.extend(["isolation_forest", "local_outlier_factor"])
            else:
                recommendations.extend(["isolation_forest", "autoencoder"])
            
            # Add ensemble for larger datasets
            if data_size > 5000:
                recommendations.append("ensemble")
            
            return recommendations[:3]  # Return top 3 recommendations
            
        except Exception as e:
            self._logger.error(f"Failed to get algorithm recommendations: {str(e)}")
            return ["isolation_forest"]  # Default fallback