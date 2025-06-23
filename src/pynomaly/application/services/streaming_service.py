"""Application service for streaming anomaly detection operations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from dataclasses import dataclass

from pynomaly.domain.services.streaming_service import (
    StreamingDetectionService,
    WindowConfiguration,
    StreamingMode,
    StreamRecord,
    StreamingResult,
    StreamDataGenerator
)
from pynomaly.infrastructure.repositories.detector_repository import DetectorRepository
from pynomaly.infrastructure.streaming import (
    ModelBasedStreamProcessor,
    StatisticalStreamProcessor,
    EnsembleStreamProcessor
)

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfiguration:
    """Configuration for streaming operations."""
    window_config: WindowConfiguration
    mode: StreamingMode = StreamingMode.REAL_TIME
    buffer_size: int = 1000
    anomaly_threshold: float = 0.5
    enable_online_learning: bool = False
    enable_callbacks: bool = True
    max_processing_time: float = 1.0  # Maximum processing time per record/batch (seconds)


@dataclass
class StreamingRequest:
    """Request for streaming anomaly detection."""
    detector_ids: List[str]
    config: StreamingConfiguration
    feature_names: List[str]
    output_stream: Optional[str] = None
    callback_url: Optional[str] = None


@dataclass
class StreamingResponse:
    """Response from streaming operations."""
    success: bool
    session_id: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None


class ApplicationStreamingService:
    """Application service for managing streaming anomaly detection."""
    
    def __init__(
        self,
        detector_repository: DetectorRepository,
        max_concurrent_sessions: int = 10
    ):
        """Initialize application streaming service.
        
        Args:
            detector_repository: Repository for detector management
            max_concurrent_sessions: Maximum concurrent streaming sessions
        """
        self.detector_repository = detector_repository
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # Active streaming sessions
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._session_counter = 0
    
    async def start_streaming_session(
        self,
        request: StreamingRequest
    ) -> StreamingResponse:
        """Start a new streaming session.
        
        Args:
            request: Streaming configuration request
            
        Returns:
            Response with session information
        """
        try:
            # Check session limit
            if len(self._active_sessions) >= self.max_concurrent_sessions:
                return StreamingResponse(
                    success=False,
                    error="Maximum concurrent sessions reached",
                    message="Too many active streaming sessions"
                )
            
            # Validate detectors
            processors = []
            for detector_id in request.detector_ids:
                detector = await self.detector_repository.get_by_id(detector_id)
                if not detector:
                    return StreamingResponse(
                        success=False,
                        error=f"Detector not found: {detector_id}",
                        message="Invalid detector ID"
                    )
                
                if not detector.is_trained:
                    return StreamingResponse(
                        success=False,
                        error=f"Detector not trained: {detector_id}",
                        message="Detector must be trained before streaming"
                    )
                
                # Create processor for this detector
                processor = ModelBasedStreamProcessor(
                    model=detector.model,
                    feature_names=request.feature_names,
                    anomaly_threshold=request.config.anomaly_threshold,
                    enable_adaptation=request.config.enable_online_learning
                )
                processors.append(processor)
            
            # Create ensemble processor if multiple detectors
            if len(processors) > 1:
                main_processor = EnsembleStreamProcessor(
                    processors=processors,
                    voting_strategy="average"
                )
            else:
                main_processor = processors[0]
            
            # Create streaming service
            streaming_service = StreamingDetectionService(
                window_config=request.config.window_config,
                mode=request.config.mode,
                buffer_size=request.config.buffer_size,
                anomaly_threshold=request.config.anomaly_threshold,
                enable_online_learning=request.config.enable_online_learning
            )
            
            # Register processor
            streaming_service.register_processor(main_processor)
            
            # Generate session ID
            self._session_counter += 1
            session_id = f"session_{self._session_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store session
            self._active_sessions[session_id] = {
                "service": streaming_service,
                "processor": main_processor,
                "config": request.config,
                "feature_names": request.feature_names,
                "created_at": datetime.now(),
                "status": "active"
            }
            
            logger.info(f"Started streaming session: {session_id}")
            
            return StreamingResponse(
                success=True,
                session_id=session_id,
                message=f"Streaming session started with {len(request.detector_ids)} detector(s)"
            )
            
        except Exception as e:
            logger.error(f"Failed to start streaming session: {e}")
            return StreamingResponse(
                success=False,
                error=str(e),
                message="Failed to start streaming session"
            )
    
    async def process_stream_data(
        self,
        session_id: str,
        stream: AsyncIterator[StreamRecord]
    ) -> AsyncIterator[StreamingResult]:
        """Process streaming data for a session.
        
        Args:
            session_id: ID of streaming session
            stream: Stream of records to process
            
        Yields:
            StreamingResult instances
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self._active_sessions[session_id]
        streaming_service = session["service"]
        
        try:
            # Process the stream
            async for result in streaming_service.process_stream(stream):
                yield result
                
        except Exception as e:
            logger.error(f"Stream processing failed for session {session_id}: {e}")
            session["status"] = "error"
            raise
    
    async def process_batch_data(
        self,
        session_id: str,
        records: List[StreamRecord]
    ) -> List[StreamingResult]:
        """Process batch of records for a session.
        
        Args:
            session_id: ID of streaming session
            records: List of records to process
            
        Returns:
            List of StreamingResults
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self._active_sessions[session_id]
        processor = session["processor"]
        
        try:
            # Process records individually
            results = []
            for record in records:
                if hasattr(processor, 'process_record'):
                    result = await processor.process_record(record)
                    if result:
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed for session {session_id}: {e}")
            session["status"] = "error"
            raise
    
    async def stop_streaming_session(self, session_id: str) -> StreamingResponse:
        """Stop a streaming session.
        
        Args:
            session_id: ID of session to stop
            
        Returns:
            Response with operation status
        """
        if session_id not in self._active_sessions:
            return StreamingResponse(
                success=False,
                error="Session not found",
                message=f"No active session with ID: {session_id}"
            )
        
        try:
            session = self._active_sessions[session_id]
            streaming_service = session["service"]
            
            # Stop the streaming service
            streaming_service.stop()
            
            # Get final statistics
            statistics = streaming_service.get_statistics()
            
            # Update session status
            session["status"] = "stopped"
            session["stopped_at"] = datetime.now()
            
            # Remove from active sessions after brief delay
            asyncio.create_task(self._cleanup_session(session_id, delay=5.0))
            
            logger.info(f"Stopped streaming session: {session_id}")
            
            return StreamingResponse(
                success=True,
                session_id=session_id,
                message="Streaming session stopped",
                statistics=statistics
            )
            
        except Exception as e:
            logger.error(f"Failed to stop streaming session {session_id}: {e}")
            return StreamingResponse(
                success=False,
                error=str(e),
                message="Failed to stop streaming session"
            )
    
    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a streaming session.
        
        Args:
            session_id: ID of session
            
        Returns:
            Statistics dictionary
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self._active_sessions[session_id]
        streaming_service = session["service"]
        processor = session["processor"]
        
        # Combine statistics from service and processor
        stats = streaming_service.get_statistics()
        
        if hasattr(processor, 'get_statistics'):
            processor_stats = processor.get_statistics()
            stats["processor_statistics"] = processor_stats
        
        # Add session metadata
        stats["session_info"] = {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "status": session["status"],
            "feature_count": len(session["feature_names"]),
            "config": {
                "mode": session["config"].mode.value,
                "window_type": session["config"].window_config.window_type.value,
                "anomaly_threshold": session["config"].anomaly_threshold
            }
        }
        
        return stats
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active streaming sessions.
        
        Returns:
            List of session information
        """
        sessions = []
        
        for session_id, session in self._active_sessions.items():
            session_info = {
                "session_id": session_id,
                "created_at": session["created_at"].isoformat(),
                "status": session["status"],
                "feature_count": len(session["feature_names"]),
                "mode": session["config"].mode.value
            }
            
            # Add stopped time if available
            if "stopped_at" in session:
                session_info["stopped_at"] = session["stopped_at"].isoformat()
            
            sessions.append(session_info)
        
        return sessions
    
    async def _cleanup_session(self, session_id: str, delay: float = 0.0) -> None:
        """Clean up a session after delay.
        
        Args:
            session_id: ID of session to clean up
            delay: Delay before cleanup in seconds
        """
        if delay > 0:
            await asyncio.sleep(delay)
        
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")
    
    async def create_test_stream(
        self,
        session_id: str,
        count: int = 100,
        delay: float = 0.1,
        anomaly_rate: float = 0.05
    ) -> AsyncIterator[StreamRecord]:
        """Create a test stream for demonstration purposes.
        
        Args:
            session_id: ID of session (for feature names)
            count: Number of records to generate
            delay: Delay between records
            anomaly_rate: Rate of anomalies to inject
            
        Yields:
            StreamRecord instances
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self._active_sessions[session_id]
        feature_names = session["feature_names"]
        
        # Create data generator
        generator = StreamDataGenerator(
            base_data=None,  # Will generate random data
            anomaly_rate=anomaly_rate
        )
        
        # Generate stream with proper feature names
        record_count = 0
        async for record in generator.generate_stream(count=count, delay=delay):
            # Ensure record has correct feature names
            filtered_data = {}
            for i, feature_name in enumerate(feature_names):
                if i < len(record.data):
                    # Get value by index or feature name
                    if feature_name in record.data:
                        filtered_data[feature_name] = record.data[feature_name]
                    else:
                        # Use indexed features if available
                        data_values = list(record.data.values())
                        if i < len(data_values):
                            filtered_data[feature_name] = data_values[i]
                        else:
                            filtered_data[feature_name] = 0.0
                else:
                    filtered_data[feature_name] = 0.0
            
            # Create new record with correct features
            yield StreamRecord(
                id=f"test_{record_count}",
                timestamp=datetime.now(),
                data=filtered_data,
                metadata={"test_data": True, "session_id": session_id}
            )
            
            record_count += 1
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get overall service statistics.
        
        Returns:
            Service statistics
        """
        active_count = len([s for s in self._active_sessions.values() if s["status"] == "active"])
        stopped_count = len([s for s in self._active_sessions.values() if s["status"] == "stopped"])
        error_count = len([s for s in self._active_sessions.values() if s["status"] == "error"])
        
        return {
            "total_sessions": len(self._active_sessions),
            "active_sessions": active_count,
            "stopped_sessions": stopped_count,
            "error_sessions": error_count,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "session_counter": self._session_counter
        }