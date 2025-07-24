"""Streaming service for real-time anomaly detection."""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import structlog

from ..entities.dataset import Dataset
from ..entities.detection_result import DetectionResult
from .detection_service import DetectionService
from ...infrastructure.repositories.model_repository import ModelRepository

logger = structlog.get_logger()


@dataclass
class StreamingConfig:
    """Streaming configuration."""
    buffer_size: int = 100
    batch_size: int = 10
    timeout_seconds: float = 5.0
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1
    alert_threshold: float = 0.7


@dataclass
class StreamingStats:
    """Streaming statistics."""
    session_id: str
    total_processed: int = 0
    anomalies_detected: int = 0
    average_processing_time: float = 0.0
    last_processed: Optional[datetime] = None
    active: bool = True
    drift_detected: bool = False


class StreamingSession:
    """Individual streaming session."""
    
    def __init__(self, session_id: str, model_id: str, config: StreamingConfig):
        self.session_id = session_id
        self.model_id = model_id
        self.config = config
        self.buffer: List[Dict[str, Any]] = []
        self.stats = StreamingStats(session_id=session_id)
        self.model = None
        self.reference_data: Optional[Dataset] = None
        self.processing_times: List[float] = []
        
    async def initialize(self, model_repository: ModelRepository):
        """Initialize the streaming session."""
        self.model = model_repository.load(self.model_id)
        if not self.model:
            raise ValueError(f"Model {self.model_id} not found")
    
    def add_sample(self, sample: Dict[str, Any]) -> bool:
        """Add a sample to the buffer."""
        if len(self.buffer) >= self.config.buffer_size:
            # Remove oldest sample
            self.buffer.pop(0)
        
        self.buffer.append(sample)
        return len(self.buffer) >= self.config.batch_size
    
    def get_batch(self) -> List[Dict[str, Any]]:
        """Get current batch for processing."""
        if len(self.buffer) >= self.config.batch_size:
            batch = self.buffer[:self.config.batch_size]
            self.buffer = self.buffer[self.config.batch_size:]
            return batch
        return []
    
    def update_stats(self, processing_time: float, anomalies_count: int):
        """Update session statistics."""
        self.stats.total_processed += 1
        self.stats.anomalies_detected += anomalies_count
        self.stats.last_processed = datetime.now()
        
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:  # Keep last 100 measurements
            self.processing_times.pop(0)
        
        self.stats.average_processing_time = sum(self.processing_times) / len(self.processing_times)


class StreamingService:
    """Service for managing streaming anomaly detection."""
    
    def __init__(self):
        self.sessions: Dict[str, StreamingSession] = {}
        self.detection_service = DetectionService()
        self.model_repository = ModelRepository()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start_session(self, model_id: str, config: StreamingConfig = None) -> str:
        """Start a new streaming session."""
        session_id = str(uuid.uuid4())
        config = config or StreamingConfig()
        
        session = StreamingSession(session_id, model_id, config)
        await session.initialize(self.model_repository)
        
        self.sessions[session_id] = session
        
        logger.info("Streaming session started", 
                   session_id=session_id, 
                   model_id=model_id)
        
        # Start cleanup task if not running
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())
        
        return session_id
    
    async def stop_session(self, session_id: str) -> bool:
        """Stop a streaming session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.stats.active = False
        
        logger.info("Streaming session stopped", session_id=session_id)
        
        # Remove from active sessions after a delay to allow final stats retrieval
        asyncio.create_task(self._delayed_session_removal(session_id))
        
        return True
    
    async def process_sample(self, session_id: str, sample: Dict[str, Any]) -> Optional[DetectionResult]:
        """Process a single sample."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        if not session.stats.active:
            raise ValueError(f"Session {session_id} is not active")
        
        start_time = datetime.now()
        
        # Add sample to buffer
        should_process = session.add_sample(sample)
        
        if should_process:
            # Process batch
            batch = session.get_batch()
            if batch:
                try:
                    # Convert to dataset
                    dataset = Dataset.from_dict_list(batch)
                    
                    # Perform detection
                    result = self.detection_service.predict(dataset, session.model)
                    
                    # Update statistics
                    processing_time = (datetime.now() - start_time).total_seconds()
                    anomalies_count = sum(1 for pred in result.predictions if pred == -1)
                    session.update_stats(processing_time, anomalies_count)
                    
                    # Check for drift if enabled
                    if session.config.enable_drift_detection:
                        await self._check_drift(session, dataset)
                    
                    logger.debug("Batch processed", 
                               session_id=session_id,
                               batch_size=len(batch),
                               anomalies=anomalies_count,
                               processing_time=processing_time)
                    
                    return result
                    
                except Exception as e:
                    logger.error("Failed to process batch", 
                               session_id=session_id, 
                               error=str(e))
                    raise
        
        return None
    
    async def process_batch(self, session_id: str, samples: List[Dict[str, Any]]) -> DetectionResult:
        """Process a batch of samples."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        if not session.stats.active:
            raise ValueError(f"Session {session_id} is not active")
        
        start_time = datetime.now()
        
        try:
            # Convert to dataset
            dataset = Dataset.from_dict_list(samples)
            
            # Perform detection
            result = self.detection_service.predict(dataset, session.model)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            anomalies_count = sum(1 for pred in result.predictions if pred == -1)
            session.update_stats(processing_time, anomalies_count)
            
            # Check for drift if enabled
            if session.config.enable_drift_detection:
                await self._check_drift(session, dataset)
            
            logger.info("Batch processed", 
                       session_id=session_id,
                       batch_size=len(samples),
                       anomalies=anomalies_count,
                       processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Failed to process batch", 
                       session_id=session_id, 
                       error=str(e))
            raise
    
    def get_session_stats(self, session_id: str) -> Optional[StreamingStats]:
        """Get statistics for a streaming session."""
        session = self.sessions.get(session_id)
        return session.stats if session else None
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs."""
        return [sid for sid, session in self.sessions.items() if session.stats.active]
    
    async def get_all_stats(self) -> Dict[str, StreamingStats]:
        """Get statistics for all sessions."""
        return {sid: session.stats for sid, session in self.sessions.items()}
    
    async def detect_drift(self, session_id: str, current_data: Dataset) -> Dict[str, Any]:
        """Detect concept drift for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if not session.reference_data:
            # Use current data as reference if none exists
            session.reference_data = current_data
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "message": "Reference data established"
            }
        
        # Perform drift detection
        drift_score = await self._calculate_drift_score(session.reference_data, current_data)
        drift_detected = drift_score > session.config.drift_threshold
        
        if drift_detected:
            session.stats.drift_detected = True
            logger.warning("Concept drift detected", 
                         session_id=session_id,
                         drift_score=drift_score)
        
        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "threshold": session.config.drift_threshold,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _check_drift(self, session: StreamingSession, current_data: Dataset):
        """Internal drift checking."""
        if not session.config.enable_drift_detection:
            return
        
        try:
            drift_result = await self.detect_drift(session.session_id, current_data)
            if drift_result["drift_detected"]:
                # Update reference data with current data
                session.reference_data = current_data
                
        except Exception as e:
            logger.error("Drift detection failed", 
                       session_id=session.session_id, 
                       error=str(e))
    
    async def _calculate_drift_score(self, reference: Dataset, current: Dataset) -> float:
        """Calculate drift score between two datasets."""
        try:
            import numpy as np
            from scipy import stats
            
            ref_data = reference.to_numpy()
            curr_data = current.to_numpy()
            
            # Ensure same number of features
            min_features = min(ref_data.shape[1], curr_data.shape[1])
            ref_data = ref_data[:, :min_features]
            curr_data = curr_data[:, :min_features]
            
            # Calculate KS test statistic for each feature
            drift_scores = []
            for i in range(min_features):
                try:
                    ks_stat, _ = stats.ks_2samp(ref_data[:, i], curr_data[:, i])
                    drift_scores.append(ks_stat)
                except Exception:
                    # Skip problematic features
                    continue
            
            # Return average drift score
            return float(np.mean(drift_scores)) if drift_scores else 0.0
            
        except ImportError:
            # Fallback if scipy not available
            logger.warning("Scipy not available, using simplified drift calculation")
            return 0.0
        except Exception as e:
            logger.error("Drift calculation failed", error=str(e))
            return 0.0
    
    async def _cleanup_inactive_sessions(self):
        """Background task to clean up inactive sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now()
                sessions_to_remove = []
                
                for session_id, session in self.sessions.items():
                    if (not session.stats.active and 
                        session.stats.last_processed and
                        (current_time - session.stats.last_processed).total_seconds() > 3600):  # 1 hour
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    del self.sessions[session_id]
                    logger.info("Cleaned up inactive session", session_id=session_id)
                
            except Exception as e:
                logger.error("Session cleanup failed", error=str(e))
    
    async def _delayed_session_removal(self, session_id: str):
        """Remove session after delay."""
        await asyncio.sleep(300)  # 5 minutes delay
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.debug("Session removed from memory", session_id=session_id)
    
    async def shutdown(self):
        """Shutdown the streaming service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Stop all active sessions
        for session_id in list(self.sessions.keys()):
            await self.stop_session(session_id)
        
        logger.info("Streaming service shutdown complete")