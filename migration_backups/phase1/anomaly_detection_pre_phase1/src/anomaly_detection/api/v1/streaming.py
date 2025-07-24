"""Streaming detection endpoints."""

import asyncio
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
import uuid

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...domain.services.streaming_service import StreamingService
from ...domain.services.detection_service import DetectionService
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class StreamingSample(BaseModel):
    """Single streaming sample."""
    data: List[float] = Field(..., description="Feature vector")
    timestamp: Optional[str] = Field(None, description="Sample timestamp")


class StreamingBatch(BaseModel):
    """Batch of streaming samples."""
    samples: List[List[float]] = Field(..., description="List of feature vectors")
    algorithm: str = Field("isolation_forest", description="Detection algorithm")
    timestamp: Optional[str] = Field(None, description="Batch timestamp")


class StreamingResult(BaseModel):
    """Streaming detection result."""
    success: bool = Field(..., description="Whether detection completed successfully")
    sample_id: str = Field(..., description="Unique sample identifier")
    is_anomaly: bool = Field(..., description="Whether sample is anomalous")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    algorithm: str = Field(..., description="Algorithm used")
    timestamp: str = Field(..., description="Processing timestamp")
    buffer_size: int = Field(..., description="Current buffer size")
    model_fitted: bool = Field(..., description="Whether model is fitted")


class StreamingStats(BaseModel):
    """Streaming service statistics."""
    total_samples: int = Field(..., description="Total samples processed")
    buffer_size: int = Field(..., description="Current buffer size")
    buffer_capacity: int = Field(..., description="Buffer capacity")
    model_fitted: bool = Field(..., description="Whether model is fitted")
    current_algorithm: str = Field(..., description="Current algorithm")
    samples_since_update: int = Field(..., description="Samples since last model update")
    update_frequency: int = Field(..., description="Model update frequency")
    last_update_at: int = Field(..., description="Sample count at last update")


class DriftDetectionResult(BaseModel):
    """Concept drift detection result."""
    drift_detected: bool = Field(..., description="Whether drift was detected")
    max_relative_change: Optional[float] = Field(None, description="Maximum relative change")
    drift_threshold: Optional[float] = Field(None, description="Drift detection threshold")
    reason: Optional[str] = Field(None, description="Reason for drift detection result")
    buffer_size: int = Field(..., description="Current buffer size")
    recent_samples: Optional[int] = Field(None, description="Recent samples analyzed")


# Global streaming service instances (in production, use dependency injection)
_streaming_services: Dict[str, StreamingService] = {}


def get_streaming_service(
    session_id: str = "default",
    window_size: int = 1000,
    update_frequency: int = 100
) -> StreamingService:
    """Get or create streaming service instance."""
    if session_id not in _streaming_services:
        detection_service = DetectionService()
        _streaming_services[session_id] = StreamingService(
            detection_service=detection_service,
            window_size=window_size,
            update_frequency=update_frequency
        )
        logger.info("Created new streaming service", session_id=session_id)
    
    return _streaming_services[session_id]


@router.post("/sample", response_model=StreamingResult)
async def process_sample(
    sample: StreamingSample,
    algorithm: str = "isolation_forest",
    session_id: str = "default",
    window_size: int = 1000,
    update_frequency: int = 100
) -> StreamingResult:
    """Process a single streaming sample."""
    try:
        streaming_service = get_streaming_service(session_id, window_size, update_frequency)
        
        # Convert data to numpy array
        sample_array = np.array(sample.data, dtype=np.float64)
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        # Process sample
        result = streaming_service.process_sample(sample_array, mapped_algorithm)
        
        # Get streaming stats
        stats = streaming_service.get_streaming_stats()
        
        # Generate unique sample ID
        sample_id = str(uuid.uuid4())
        
        return StreamingResult(
            success=result.success,
            sample_id=sample_id,
            is_anomaly=bool(result.predictions[0] == -1),
            confidence_score=float(result.confidence_scores[0]) if result.confidence_scores is not None else None,
            algorithm=algorithm,
            timestamp=datetime.utcnow().isoformat(),
            buffer_size=stats['buffer_size'],
            model_fitted=stats['model_fitted']
        )
        
    except Exception as e:
        logger.error("Streaming sample processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample processing failed: {str(e)}"
        )


@router.post("/batch", response_model=List[StreamingResult])
async def process_batch(
    batch: StreamingBatch,
    session_id: str = "default",
    window_size: int = 1000,
    update_frequency: int = 100
) -> List[StreamingResult]:
    """Process a batch of streaming samples."""
    try:
        streaming_service = get_streaming_service(session_id, window_size, update_frequency)
        
        # Convert data to numpy array
        batch_array = np.array(batch.samples, dtype=np.float64)
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(batch.algorithm, batch.algorithm)
        
        # Process batch
        result = streaming_service.process_batch(batch_array, mapped_algorithm)
        
        # Get streaming stats
        stats = streaming_service.get_streaming_stats()
        
        # Create results for each sample
        results = []
        for i, (prediction, score) in enumerate(zip(result.predictions, result.confidence_scores or [None] * len(result.predictions))):
            sample_id = str(uuid.uuid4())
            results.append(StreamingResult(
                success=result.success,
                sample_id=sample_id,
                is_anomaly=bool(prediction == -1),
                confidence_score=float(score) if score is not None else None,
                algorithm=batch.algorithm,
                timestamp=datetime.utcnow().isoformat(),
                buffer_size=stats['buffer_size'],
                model_fitted=stats['model_fitted']
            ))
        
        return results
        
    except Exception as e:
        logger.error("Streaming batch processing error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/stats", response_model=StreamingStats)
async def get_streaming_stats(
    session_id: str = "default",
    window_size: int = 1000,
    update_frequency: int = 100
) -> StreamingStats:
    """Get streaming service statistics."""
    try:
        streaming_service = get_streaming_service(session_id, window_size, update_frequency)
        stats = streaming_service.get_streaming_stats()
        
        return StreamingStats(**stats)
        
    except Exception as e:
        logger.error("Error getting streaming stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get streaming stats: {str(e)}"
        )


@router.post("/drift", response_model=DriftDetectionResult)
async def detect_drift(
    session_id: str = "default",
    window_size: int = 200,
) -> DriftDetectionResult:
    """Detect concept drift in the data stream."""
    try:
        if session_id not in _streaming_services:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Streaming session '{session_id}' not found"
            )
        
        streaming_service = _streaming_services[session_id]
        drift_result = streaming_service.detect_concept_drift(window_size)
        
        return DriftDetectionResult(**drift_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Drift detection error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}"
        )


@router.post("/reset")
async def reset_stream(session_id: str = "default") -> Dict[str, str]:
    """Reset streaming service state."""
    try:
        if session_id in _streaming_services:
            _streaming_services[session_id].reset_stream()
            return {"message": f"Streaming session '{session_id}' reset successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Streaming session '{session_id}' not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Stream reset error", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset stream: {str(e)}"
        )


@router.websocket("/ws/{session_id}")
async def websocket_streaming(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time streaming detection."""
    await websocket.accept()
    logger.info("WebSocket connection established", session_id=session_id)
    
    try:
        streaming_service = get_streaming_service(session_id)
        
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "sample":
                # Process single sample
                sample_data = np.array(message["data"], dtype=np.float64)
                algorithm = message.get("algorithm", "isolation_forest")
                
                # Algorithm mapping
                algorithm_map = {
                    'isolation_forest': 'iforest',
                    'one_class_svm': 'ocsvm',
                    'lof': 'lof'
                }
                mapped_algorithm = algorithm_map.get(algorithm, algorithm)
                
                # Process sample
                result = streaming_service.process_sample(sample_data, mapped_algorithm)
                stats = streaming_service.get_streaming_stats()
                
                # Send result back to client
                response = {
                    "type": "result",
                    "sample_id": str(uuid.uuid4()),
                    "is_anomaly": bool(result.predictions[0] == -1),
                    "confidence_score": float(result.confidence_scores[0]) if result.confidence_scores is not None else None,
                    "algorithm": algorithm,
                    "timestamp": datetime.utcnow().isoformat(),
                    "buffer_size": stats['buffer_size'],
                    "model_fitted": stats['model_fitted']
                }
                
                await websocket.send_text(json.dumps(response))
                
            elif message.get("type") == "stats":
                # Send current stats
                stats = streaming_service.get_streaming_stats()
                response = {
                    "type": "stats",
                    **stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(response))
                
            elif message.get("type") == "drift":
                # Check for concept drift
                window_size = message.get("window_size", 200)
                drift_result = streaming_service.detect_concept_drift(window_size)
                response = {
                    "type": "drift",
                    **drift_result,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed", session_id=session_id)
    except Exception as e:
        logger.error("WebSocket error", session_id=session_id, error=str(e))
        await websocket.close(code=1000)