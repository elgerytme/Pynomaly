"""
Detection Application Service

Orchestrates anomaly detection operations and coordinates between domain and infrastructure layers.
"""

import asyncio
from typing import List, Optional
from uuid import UUID
import logging

from ...domain.entities.pattern_analysis_request import PatternAnalysisRequest
from ...domain.value_objects.algorithm_config import AlgorithmConfig
from ...domain.value_objects.pattern_analysis_metadata import PatternAnalysisMetadata
from ...domain.repositories.pattern_analysis_repository import PatternAnalysisRepository
from ...domain.services.pattern_analysis_validator import PatternAnalysisValidator
from ...domain.exceptions.validation_exceptions import ValidationError, PatternAnalysisRequestError
from ..dto.pattern_analysis_dto import PatternAnalysisRequestDTO, PatternAnalysisResponseDTO
from ...infrastructure.adapters.algorithm_adapter import AlgorithmAdapter


class PatternAnalysisService:
    """
    Application service for orchestrating anomaly detection operations.
    
    This service coordinates between the domain layer (business logic),
    infrastructure layer (external systems), and presentation layer (APIs/CLI).
    """
    
    def __init__(
        self,
        pattern_analysis_repository: PatternAnalysisRepository,
        algorithm_adapter: AlgorithmAdapter,
        logger: Optional[logging.Logger] = None
    ):
        self._pattern_analysis_repository = pattern_analysis_repository
        self._algorithm_adapter = algorithm_adapter
        self._logger = logger or logging.getLogger(__name__)
    
    async def submit_pattern_analysis_request(
        self, 
        request_dto: PatternAnalysisRequestDTO
    ) -> PatternAnalysisResponseDTO:
        """
        Submit a new anomaly pattern analysis request.
        
        Args:
            request_dto: Data transfer object containing request details.
            
        Returns:
            PatternAnalysisResponseDTO: Response containing request ID and initial status.
            
        Raises:
            ValidationError: If the request is invalid.
            PatternAnalysisRequestError: If the request cannot be processed.
        """
        try:
            # Convert DTO to domain entity
            algorithm_config = AlgorithmConfig.from_dict(request_dto.algorithm_config)
            metadata = PatternAnalysisMetadata.from_dict(request_dto.metadata or {})
            
            pattern_analysis_request = PatternAnalysisRequest(
                data=request_dto.data,
                algorithm_config=algorithm_config,
                metadata=metadata
            )
            
            # Validate the request
            PatternAnalysisValidator.validate_and_raise(pattern_analysis_request)
            
            # Save the request
            await self._pattern_analysis_repository.save_request(pattern_analysis_request)
            
            self._logger.info(f"Pattern analysis request {pattern_analysis_request.id} submitted successfully")
            
            # Start async processing
            asyncio.create_task(self._process_pattern_analysis_request(pattern_analysis_request))
            
            return PatternAnalysisResponseDTO(
                request_id=str(pattern_analysis_request.id),
                status="submitted",
                message="Pattern analysis request submitted for processing"
            )
            
        except ValidationError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to submit pattern analysis request: {str(e)}")
            raise PatternAnalysisRequestError("unknown", f"Failed to submit request: {str(e)}")
    
    async def get_pattern_analysis_status(self, request_id: UUID) -> PatternAnalysisResponseDTO:
        """
        Get the status of a pattern analysis request.
        
        Args:
            request_id: Unique identifier of the pattern analysis request.
            
        Returns:
            PatternAnalysisResponseDTO: Current status and details of the request.
            
        Raises:
            PatternAnalysisRequestError: If the request cannot be found or accessed.
        """
        try:
            request = await self._pattern_analysis_repository.get_request_by_id(request_id)
            
            if not request:
                raise PatternAnalysisRequestError(
                    str(request_id), 
                    "Pattern analysis request not found"
                )
            
            return PatternAnalysisResponseDTO(
                request_id=str(request.id),
                status=request.status,
                message=f"Request is {request.status}",
                created_at=request.created_at
            )
            
        except PatternAnalysisRequestError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to get detection status for {request_id}: {str(e)}")
            raise PatternAnalysisRequestError(
                str(request_id), 
                f"Failed to retrieve status: {str(e)}"
            )
    
    async def list_pattern_analysis_requests(
        self, 
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[PatternAnalysisResponseDTO]:
        """
        List pattern analysis requests with optional filtering.
        
        Args:
            user_id: Optional user ID filter.
            limit: Maximum number of requests to return.
            offset: Number of requests to skip.
            
        Returns:
            List[PatternAnalysisResponseDTO]: List of pattern analysis request summaries.
        """
        try:
            if user_id:
                requests = await self._pattern_analysis_repository.get_requests_by_user(user_id)
            else:
                requests = await self._pattern_analysis_repository.list_requests(limit, offset)
            
            return [
                PatternAnalysisResponseDTO(
                    request_id=str(req.id),
                    status=req.status,
                    message=f"Request {req.status}",
                    created_at=req.created_at
                )
                for req in requests
            ]
            
        except Exception as e:
            self._logger.error(f"Failed to list pattern analysis requests: {str(e)}")
            raise PatternAnalysisRequestError("list", f"Failed to list requests: {str(e)}")
    
    async def cancel_pattern_analysis_request(self, request_id: UUID) -> bool:
        """
        Cancel a pending pattern analysis request.
        
        Args:
            request_id: Unique identifier of the pattern analysis request.
            
        Returns:
            bool: True if the request was cancelled successfully.
            
        Raises:
            PatternAnalysisRequestError: If the request cannot be cancelled.
        """
        try:
            request = await self._pattern_analysis_repository.get_request_by_id(request_id)
            
            if not request:
                raise PatternAnalysisRequestError(
                    str(request_id), 
                    "Pattern analysis request not found"
                )
            
            if request.status not in ["pending", "submitted"]:
                raise PatternAnalysisRequestError(
                    str(request_id), 
                    f"Cannot cancel request with status: {request.status}"
                )
            
            await self._pattern_analysis_repository.update_request_status(request_id, "cancelled")
            
            self._logger.info(f"Pattern analysis request {request_id} cancelled")
            return True
            
        except PatternAnalysisRequestError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to cancel pattern analysis request {request_id}: {str(e)}")
            raise PatternAnalysisRequestError(
                str(request_id), 
                f"Failed to cancel request: {str(e)}"
            )
    
    async def _process_pattern_analysis_request(self, request: PatternAnalysisRequest) -> None:
        """
        Process a pattern analysis request asynchronously.
        
        Args:
            request: The pattern analysis request to process.
        """
        try:
            # Update status to processing
            request.mark_as_processing()
            await self._pattern_analysis_repository.update_request_status(
                request.id, 
                "processing"
            )
            
            self._logger.info(f"Starting processing for pattern analysis request {request.id}")
            
            # Execute the detection algorithm
            result = await self._algorithm_adapter.analyze_patterns(
                data=request.data,
                algorithm_config=request.algorithm_config
            )
            
            # Mark as completed
            request.mark_as_completed()
            await self._pattern_analysis_repository.update_request_status(
                request.id, 
                "completed"
            )
            
            self._logger.info(f"Pattern analysis request {request.id} completed successfully")
            
        except Exception as e:
            # Mark as failed
            error_message = str(e)
            request.mark_as_failed(error_message)
            await self._pattern_analysis_repository.update_request_status(
                request.id, 
                "failed"
            )
            
            self._logger.error(f"Pattern analysis request {request.id} failed: {error_message}")
    
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