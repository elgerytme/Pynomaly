"""UC-004: Process Streaming Data for Real-time Detection use case implementation."""

from typing import Dict, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio

from ...domain.entities.dataset import Dataset
from ...domain.entities.detection_result import DetectionResult
from ...domain.services.detection_service import DetectionService
from ...infrastructure.repositories.model_repository import ModelRepository


@dataclass
class ProcessStreamingRequest:
    """Request for streaming processing."""
    model_id: str
    stream_config: Dict[str, Any]
    alert_threshold: float = 0.7
    batch_size: int = 1


@dataclass
class ProcessStreamingResponse:
    """Response from streaming processing."""
    prediction: DetectionResult = None
    alert_triggered: bool = False
    success: bool = False
    error_message: str = None


class ProcessStreamingUseCase:
    """Use case for processing streaming data."""
    
    def __init__(
        self,
        detection_service: DetectionService,
        model_repository: ModelRepository
    ):
        self._detection_service = detection_service
        self._model_repository = model_repository
        self._loaded_model = None
        self._current_model_id = None
    
    async def execute_stream(
        self,
        request: ProcessStreamingRequest,
        data_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[ProcessStreamingResponse, None]:
        """Execute streaming anomaly detection.
        
        Args:
            request: Streaming request
            data_stream: Async generator of data points
            
        Yields:
            Streaming responses
        """
        try:
            # Load model if needed
            if self._current_model_id != request.model_id:
                self._loaded_model = self._model_repository.load(request.model_id)
                self._current_model_id = request.model_id
                
                if not self._loaded_model:
                    yield ProcessStreamingResponse(
                        success=False,
                        error_message=f"Model {request.model_id} not found"
                    )
                    return
            
            # Process stream
            batch = []
            async for data_point in data_stream:
                try:
                    # Add to batch
                    batch.append(data_point)
                    
                    # Process when batch is full or single item mode
                    if len(batch) >= request.batch_size:
                        # Create dataset from batch
                        dataset = Dataset.from_dict_list(batch)
                        
                        # Validate data quality
                        if not dataset.is_valid():
                            yield ProcessStreamingResponse(
                                success=False,
                                error_message="Invalid data point format or quality"
                            )
                            batch = []
                            continue
                        
                        # Perform detection
                        result = self._detection_service.predict(
                            dataset, self._loaded_model
                        )
                        
                        # Check for alerts
                        alert_triggered = any(
                            score >= request.alert_threshold
                            for score in result.anomaly_scores
                        )
                        
                        yield ProcessStreamingResponse(
                            prediction=result,
                            alert_triggered=alert_triggered,
                            success=True
                        )
                        
                        # Clear batch
                        batch = []
                        
                except Exception as e:
                    yield ProcessStreamingResponse(
                        success=False,
                        error_message=f"Error processing data point: {str(e)}"
                    )
                    batch = []  # Clear batch on error
                    
        except Exception as e:
            yield ProcessStreamingResponse(
                success=False,
                error_message=f"Stream processing error: {str(e)}"
            )
    
    def execute_single(self, request: ProcessStreamingRequest, data_point: Dict[str, Any]) -> ProcessStreamingResponse:
        """Execute detection on a single data point.
        
        Args:
            request: Processing request
            data_point: Single data point
            
        Returns:
            Processing response
        """
        try:
            # Load model if needed
            if self._current_model_id != request.model_id:
                self._loaded_model = self._model_repository.load(request.model_id)
                self._current_model_id = request.model_id
                
                if not self._loaded_model:
                    return ProcessStreamingResponse(
                        success=False,
                        error_message=f"Model {request.model_id} not found"
                    )
            
            # Create dataset from single point
            dataset = Dataset.from_dict_list([data_point])
            
            # Validate data quality
            if not dataset.is_valid():
                return ProcessStreamingResponse(
                    success=False,
                    error_message="Invalid data point format or quality"
                )
            
            # Perform detection
            result = self._detection_service.predict(dataset, self._loaded_model)
            
            # Check for alert
            alert_triggered = result.anomaly_scores[0] >= request.alert_threshold
            
            return ProcessStreamingResponse(
                prediction=result,
                alert_triggered=alert_triggered,
                success=True
            )
            
        except Exception as e:
            return ProcessStreamingResponse(
                success=False,
                error_message=str(e)
            )