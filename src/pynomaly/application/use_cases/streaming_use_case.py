"""Use case for streaming anomaly detection operations."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass

from pynomaly.application.services.streaming_service import (
    ApplicationStreamingService,
    StreamingRequest,
    StreamingResponse,
    StreamingConfiguration
)
from pynomaly.domain.services.streaming_service import (
    StreamRecord,
    StreamingResult,
    WindowConfiguration,
    WindowType,
    StreamingMode
)

logger = logging.getLogger(__name__)


@dataclass
class StartStreamingRequest:
    """Request to start streaming anomaly detection."""
    detector_ids: List[str]
    feature_names: List[str]
    window_type: str = "count_based"
    window_size: int = 100
    streaming_mode: str = "real_time"
    anomaly_threshold: float = 0.5
    enable_online_learning: bool = False
    output_stream: Optional[str] = None


@dataclass
class ProcessStreamRequest:
    """Request to process streaming data."""
    session_id: str
    records: List[Dict[str, Any]]


@dataclass
class StreamingSessionInfo:
    """Information about a streaming session."""
    session_id: str
    status: str
    detector_count: int
    feature_names: List[str]
    configuration: Dict[str, Any]
    statistics: Optional[Dict[str, Any]] = None


@dataclass
class StreamingUseCaseResponse:
    """Response from streaming use case operations."""
    success: bool
    session_id: Optional[str] = None
    session_info: Optional[StreamingSessionInfo] = None
    results: Optional[List[Dict[str, Any]]] = None
    statistics: Optional[Dict[str, Any]] = None
    message: str = ""
    error: Optional[str] = None


class StreamingUseCase:
    """Use case for managing streaming anomaly detection workflows."""
    
    def __init__(self, streaming_service: ApplicationStreamingService):
        """Initialize streaming use case.
        
        Args:
            streaming_service: Application streaming service
        """
        self.streaming_service = streaming_service
    
    async def start_streaming_session(
        self,
        request: StartStreamingRequest
    ) -> StreamingUseCaseResponse:
        """Start a new streaming session.
        
        Args:
            request: Request to start streaming
            
        Returns:
            Response with session information
        """
        try:
            logger.info(f"Starting streaming session for {len(request.detector_ids)} detectors")
            
            # Map window type string to enum
            window_type_map = {
                "count_based": WindowType.COUNT_BASED,
                "time_based": WindowType.TIME_BASED,
                "session_based": WindowType.SESSION_BASED,
                "adaptive_size": WindowType.ADAPTIVE_SIZE
            }
            
            window_type = window_type_map.get(
                request.window_type.lower(),
                WindowType.COUNT_BASED
            )
            
            # Map streaming mode string to enum
            mode_map = {
                "real_time": StreamingMode.REAL_TIME,
                "batch": StreamingMode.BATCH,
                "micro_batch": StreamingMode.MICRO_BATCH,
                "adaptive": StreamingMode.ADAPTIVE
            }
            
            streaming_mode = mode_map.get(
                request.streaming_mode.lower(),
                StreamingMode.REAL_TIME
            )
            
            # Create window configuration
            window_config = WindowConfiguration(
                window_type=window_type,
                size=request.window_size,
                step=request.window_size // 2,  # 50% overlap
                overlap=0.5
            )
            
            # Create streaming configuration
            streaming_config = StreamingConfiguration(
                window_config=window_config,
                mode=streaming_mode,
                anomaly_threshold=request.anomaly_threshold,
                enable_online_learning=request.enable_online_learning
            )
            
            # Create streaming request
            streaming_request = StreamingRequest(
                detector_ids=request.detector_ids,
                config=streaming_config,
                feature_names=request.feature_names,
                output_stream=request.output_stream
            )
            
            # Start session
            response = await self.streaming_service.start_streaming_session(streaming_request)
            
            if response.success:
                # Get session info
                statistics = await self.streaming_service.get_session_statistics(response.session_id)
                
                session_info = StreamingSessionInfo(
                    session_id=response.session_id,
                    status="active",
                    detector_count=len(request.detector_ids),
                    feature_names=request.feature_names,
                    configuration={
                        "window_type": request.window_type,
                        "window_size": request.window_size,
                        "streaming_mode": request.streaming_mode,
                        "anomaly_threshold": request.anomaly_threshold,
                        "online_learning": request.enable_online_learning
                    },
                    statistics=statistics
                )
                
                return StreamingUseCaseResponse(
                    success=True,
                    session_id=response.session_id,
                    session_info=session_info,
                    message=f"Streaming session started successfully"
                )
            else:
                return StreamingUseCaseResponse(
                    success=False,
                    error=response.error,
                    message=response.message
                )
            
        except Exception as e:
            logger.error(f"Failed to start streaming session: {e}")
            return StreamingUseCaseResponse(
                success=False,
                error=str(e),
                message="Failed to start streaming session"
            )
    
    async def process_streaming_batch(
        self,
        request: ProcessStreamRequest
    ) -> StreamingUseCaseResponse:
        """Process a batch of streaming records.
        
        Args:
            request: Request with batch data to process
            
        Returns:
            Response with processing results
        """
        try:
            logger.info(f"Processing batch of {len(request.records)} records for session {request.session_id}")
            
            # Convert records to StreamRecord objects
            stream_records = []
            for i, record_data in enumerate(request.records):
                stream_record = StreamRecord(
                    id=f"batch_{request.session_id}_{i}",
                    timestamp=record_data.get("timestamp", ""),
                    data=record_data.get("data", {}),
                    metadata=record_data.get("metadata", {})
                )
                stream_records.append(stream_record)
            
            # Process batch
            results = await self.streaming_service.process_batch_data(
                request.session_id,
                stream_records
            )
            
            # Convert results to dictionaries
            result_dicts = []
            for result in results:
                result_dict = {
                    "record_id": result.record_id,
                    "timestamp": result.timestamp.isoformat(),
                    "anomaly_score": result.anomaly_score,
                    "is_anomaly": result.is_anomaly,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                    "metadata": result.metadata
                }
                result_dicts.append(result_dict)
            
            # Get updated statistics
            statistics = await self.streaming_service.get_session_statistics(request.session_id)
            
            return StreamingUseCaseResponse(
                success=True,
                session_id=request.session_id,
                results=result_dicts,
                statistics=statistics,
                message=f"Processed {len(results)} records successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to process streaming batch: {e}")
            return StreamingUseCaseResponse(
                success=False,
                error=str(e),
                message="Failed to process streaming batch"
            )
    
    async def stop_streaming_session(self, session_id: str) -> StreamingUseCaseResponse:
        """Stop a streaming session.
        
        Args:
            session_id: ID of session to stop
            
        Returns:
            Response with operation status
        """
        try:
            logger.info(f"Stopping streaming session: {session_id}")
            
            response = await self.streaming_service.stop_streaming_session(session_id)
            
            if response.success:
                return StreamingUseCaseResponse(
                    success=True,
                    session_id=session_id,
                    statistics=response.statistics,
                    message="Streaming session stopped successfully"
                )
            else:
                return StreamingUseCaseResponse(
                    success=False,
                    error=response.error,
                    message=response.message
                )
                
        except Exception as e:
            logger.error(f"Failed to stop streaming session: {e}")
            return StreamingUseCaseResponse(
                success=False,
                error=str(e),
                message="Failed to stop streaming session"
            )
    
    async def get_session_info(self, session_id: str) -> StreamingUseCaseResponse:
        """Get information about a streaming session.
        
        Args:
            session_id: ID of session
            
        Returns:
            Response with session information
        """
        try:
            # Get session statistics
            statistics = await self.streaming_service.get_session_statistics(session_id)
            session_info_data = statistics.get("session_info", {})
            
            session_info = StreamingSessionInfo(
                session_id=session_id,
                status=session_info_data.get("status", "unknown"),
                detector_count=1,  # Will be updated with real info
                feature_names=[],  # Will be updated with real info
                configuration=session_info_data.get("config", {}),
                statistics=statistics
            )
            
            return StreamingUseCaseResponse(
                success=True,
                session_id=session_id,
                session_info=session_info,
                statistics=statistics,
                message="Session information retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return StreamingUseCaseResponse(
                success=False,
                error=str(e),
                message="Failed to get session information"
            )
    
    def list_active_sessions(self) -> StreamingUseCaseResponse:
        """List all active streaming sessions.
        
        Returns:
            Response with list of active sessions
        """
        try:
            sessions = self.streaming_service.list_active_sessions()
            
            session_infos = []
            for session in sessions:
                session_info = StreamingSessionInfo(
                    session_id=session["session_id"],
                    status=session["status"],
                    detector_count=1,  # Placeholder
                    feature_names=[],  # Placeholder
                    configuration={
                        "mode": session["mode"],
                        "feature_count": session["feature_count"]
                    }
                )
                session_infos.append(session_info)
            
            return StreamingUseCaseResponse(
                success=True,
                results=[
                    {
                        "session_id": info.session_id,
                        "status": info.status,
                        "configuration": info.configuration
                    }
                    for info in session_infos
                ],
                message=f"Found {len(sessions)} active sessions"
            )
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return StreamingUseCaseResponse(
                success=False,
                error=str(e),
                message="Failed to list active sessions"
            )
    
    async def create_test_stream(
        self,
        session_id: str,
        count: int = 100,
        delay: float = 0.1,
        anomaly_rate: float = 0.05
    ) -> AsyncIterator[Dict[str, Any]]:
        """Create a test stream for demonstration.
        
        Args:
            session_id: ID of session
            count: Number of records to generate
            delay: Delay between records
            anomaly_rate: Rate of anomalies
            
        Yields:
            Dictionary representations of StreamingResults
        """
        try:
            # Create test stream
            test_stream = self.streaming_service.create_test_stream(
                session_id=session_id,
                count=count,
                delay=delay,
                anomaly_rate=anomaly_rate
            )
            
            # Process the test stream
            async for result in self.streaming_service.process_stream_data(session_id, test_stream):
                yield {
                    "record_id": result.record_id,
                    "timestamp": result.timestamp.isoformat(),
                    "anomaly_score": result.anomaly_score,
                    "is_anomaly": result.is_anomaly,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                    "metadata": result.metadata
                }
                
        except Exception as e:
            logger.error(f"Test stream creation failed: {e}")
            yield {
                "error": str(e),
                "message": "Test stream creation failed"
            }
    
    def get_service_statistics(self) -> StreamingUseCaseResponse:
        """Get overall streaming service statistics.
        
        Returns:
            Response with service statistics
        """
        try:
            statistics = self.streaming_service.get_service_statistics()
            
            return StreamingUseCaseResponse(
                success=True,
                statistics=statistics,
                message="Service statistics retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to get service statistics: {e}")
            return StreamingUseCaseResponse(
                success=False,
                error=str(e),
                message="Failed to get service statistics"
            )