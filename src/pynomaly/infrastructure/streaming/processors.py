"""Infrastructure implementations for stream processors."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

import numpy as np
import pandas as pd

from pynomaly.domain.services.streaming_service import (
    StreamProcessorProtocol,
    StreamRecord,
    StreamBatch,
    StreamingResult
)

logger = logging.getLogger(__name__)


class ModelBasedStreamProcessor(StreamProcessorProtocol):
    """Stream processor using trained anomaly detection models."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        anomaly_threshold: float = 0.5,
        confidence_threshold: float = 0.6,
        enable_adaptation: bool = False
    ):
        """Initialize model-based processor.
        
        Args:
            model: Trained anomaly detection model
            feature_names: List of feature names expected by model
            anomaly_threshold: Threshold for anomaly detection
            confidence_threshold: Minimum confidence for results
            enable_adaptation: Whether to enable model adaptation
        """
        self.model = model
        self.feature_names = feature_names
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_adaptation = enable_adaptation
        self._adaptation_buffer = []
        self._stats = {
            "processed_records": 0,
            "anomalies_detected": 0,
            "adaptations_performed": 0
        }
    
    async def process_record(self, record: StreamRecord) -> Optional[StreamingResult]:
        """Process a single record."""
        try:
            # Extract features
            features = self._extract_features(record.data)
            if features is None:
                return None
            
            # Predict anomaly score
            anomaly_score = await self._predict_async(features)
            is_anomaly = anomaly_score > self.anomaly_threshold
            confidence = abs(anomaly_score - self.anomaly_threshold)
            
            # Skip low confidence results
            if confidence < self.confidence_threshold:
                return None
            
            # Update statistics
            self._stats["processed_records"] += 1
            if is_anomaly:
                self._stats["anomalies_detected"] += 1
            
            # Add to adaptation buffer if enabled
            if self.enable_adaptation:
                self._adaptation_buffer.append((features, anomaly_score, is_anomaly))
                if len(self._adaptation_buffer) > 100:  # Keep buffer size manageable
                    self._adaptation_buffer.pop(0)
            
            return StreamingResult(
                record_id=record.id,
                timestamp=record.timestamp,
                anomaly_score=float(anomaly_score),
                is_anomaly=is_anomaly,
                confidence=float(confidence),
                metadata={
                    "processor": self.__class__.__name__,
                    "model_type": self.model.__class__.__name__,
                    "feature_count": len(features)
                }
            )
            
        except Exception as e:
            logger.error(f"Record processing failed: {e}")
            return None
    
    async def process_batch(self, batch: StreamBatch) -> List[StreamingResult]:
        """Process a batch of records."""
        results = []
        
        try:
            # Extract features for all records
            batch_features = []
            valid_records = []
            
            for record in batch.records:
                features = self._extract_features(record.data)
                if features is not None:
                    batch_features.append(features)
                    valid_records.append(record)
            
            if not batch_features:
                return results
            
            # Batch prediction
            batch_features_array = np.array(batch_features)
            anomaly_scores = await self._predict_batch_async(batch_features_array)
            
            # Create results
            for i, (record, score) in enumerate(zip(valid_records, anomaly_scores)):
                is_anomaly = score > self.anomaly_threshold
                confidence = abs(score - self.anomaly_threshold)
                
                if confidence >= self.confidence_threshold:
                    self._stats["processed_records"] += 1
                    if is_anomaly:
                        self._stats["anomalies_detected"] += 1
                    
                    results.append(StreamingResult(
                        record_id=record.id,
                        timestamp=record.timestamp,
                        anomaly_score=float(score),
                        is_anomaly=is_anomaly,
                        confidence=float(confidence),
                        metadata={
                            "processor": self.__class__.__name__,
                            "batch_id": batch.batch_id,
                            "batch_size": len(batch.records)
                        }
                    ))
            
            # Add to adaptation buffer
            if self.enable_adaptation:
                for features, score in zip(batch_features, anomaly_scores):
                    is_anomaly = score > self.anomaly_threshold
                    self._adaptation_buffer.append((features, score, is_anomaly))
                
                # Keep buffer size manageable
                if len(self._adaptation_buffer) > 1000:
                    self._adaptation_buffer = self._adaptation_buffer[-500:]
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
        
        return results
    
    async def update_model(self, records: List[StreamRecord]) -> None:
        """Update model with new data (online learning)."""
        if not self.enable_adaptation or not self._adaptation_buffer:
            return
        
        try:
            # Simple adaptation: retrain on recent data if enough samples
            if len(self._adaptation_buffer) >= 50:
                # Extract features and labels from buffer
                features = np.array([item[0] for item in self._adaptation_buffer])
                
                # Check if model supports incremental learning
                if hasattr(self.model, 'partial_fit'):
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.model.partial_fit, features
                    )
                    self._stats["adaptations_performed"] += 1
                    logger.info("Model updated with incremental learning")
                
                elif hasattr(self.model, 'fit'):
                    # Full retraining (more expensive)
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.model.fit, features
                    )
                    self._stats["adaptations_performed"] += 1
                    logger.info("Model retrained with recent data")
                
                # Clear adaptation buffer after update
                self._adaptation_buffer.clear()
        
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def _extract_features(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract feature vector from record data."""
        try:
            features = []
            for feature_name in self.feature_names:
                if feature_name in data:
                    value = data[feature_name]
                    if isinstance(value, (int, float)):
                        features.append(float(value))
                    else:
                        # Simple encoding for non-numeric values
                        features.append(hash(str(value)) % 1000 / 1000.0)
                else:
                    features.append(0.0)  # Missing value
            
            return np.array(features)
        
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    async def _predict_async(self, features: np.ndarray) -> float:
        """Async wrapper for model prediction."""
        try:
            # Run prediction in thread pool to avoid blocking
            if hasattr(self.model, 'decision_function'):
                score = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.decision_function, features.reshape(1, -1)
                )
                return float(score[0])
            elif hasattr(self.model, 'predict_proba'):
                proba = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.predict_proba, features.reshape(1, -1)
                )
                return float(proba[0, 1])  # Probability of anomaly
            elif hasattr(self.model, 'predict'):
                prediction = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.predict, features.reshape(1, -1)
                )
                return float(prediction[0])
            else:
                return 0.0
        
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 0.0
    
    async def _predict_batch_async(self, features: np.ndarray) -> np.ndarray:
        """Async wrapper for batch model prediction."""
        try:
            if hasattr(self.model, 'decision_function'):
                scores = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.decision_function, features
                )
                return scores
            elif hasattr(self.model, 'predict_proba'):
                proba = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.predict_proba, features
                )
                return proba[:, 1]  # Probability of anomaly
            elif hasattr(self.model, 'predict'):
                predictions = await asyncio.get_event_loop().run_in_executor(
                    None, self.model.predict, features
                )
                return predictions
            else:
                return np.zeros(len(features))
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return np.zeros(len(features))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        stats = self._stats.copy()
        if stats["processed_records"] > 0:
            stats["anomaly_rate"] = stats["anomalies_detected"] / stats["processed_records"]
        else:
            stats["anomaly_rate"] = 0.0
        
        stats["buffer_size"] = len(self._adaptation_buffer)
        return stats


class StatisticalStreamProcessor(StreamProcessorProtocol):
    """Stream processor using statistical methods."""
    
    def __init__(
        self,
        feature_names: List[str],
        window_size: int = 100,
        z_threshold: float = 3.0,
        adaptation_rate: float = 0.1
    ):
        """Initialize statistical processor.
        
        Args:
            feature_names: List of feature names
            window_size: Size of sliding window for statistics
            z_threshold: Z-score threshold for anomaly detection
            adaptation_rate: Rate of adaptation for online statistics
        """
        self.feature_names = feature_names
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.adaptation_rate = adaptation_rate
        
        # Online statistics for each feature
        self._feature_stats = {
            name: {"mean": 0.0, "var": 1.0, "count": 0}
            for name in feature_names
        }
        self._recent_data = []
        self._stats = {"processed_records": 0, "anomalies_detected": 0}
    
    async def process_record(self, record: StreamRecord) -> Optional[StreamingResult]:
        """Process a single record using statistical methods."""
        try:
            features = self._extract_features(record.data)
            if features is None:
                return None
            
            # Calculate anomaly score using Z-scores
            anomaly_score = self._calculate_statistical_score(features)
            is_anomaly = anomaly_score > self.z_threshold
            confidence = min(anomaly_score / self.z_threshold, 1.0)
            
            # Update online statistics
            self._update_statistics(features)
            
            # Update global statistics
            self._stats["processed_records"] += 1
            if is_anomaly:
                self._stats["anomalies_detected"] += 1
            
            return StreamingResult(
                record_id=record.id,
                timestamp=record.timestamp,
                anomaly_score=float(anomaly_score),
                is_anomaly=is_anomaly,
                confidence=float(confidence),
                metadata={
                    "processor": self.__class__.__name__,
                    "method": "z_score",
                    "feature_count": len(features)
                }
            )
        
        except Exception as e:
            logger.error(f"Statistical processing failed: {e}")
            return None
    
    async def process_batch(self, batch: StreamBatch) -> List[StreamingResult]:
        """Process batch using statistical methods."""
        results = []
        
        try:
            for record in batch.records:
                result = await self.process_record(record)
                if result:
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Statistical batch processing failed: {e}")
        
        return results
    
    async def update_model(self, records: List[StreamRecord]) -> None:
        """Update statistical model with new records."""
        # Statistics are updated automatically in process_record
        pass
    
    def _extract_features(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract numeric features from record data."""
        try:
            features = []
            for feature_name in self.feature_names:
                if feature_name in data:
                    value = data[feature_name]
                    if isinstance(value, (int, float)):
                        features.append(float(value))
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)
            
            return np.array(features)
        
        except Exception:
            return None
    
    def _calculate_statistical_score(self, features: np.ndarray) -> float:
        """Calculate anomaly score using Z-scores."""
        z_scores = []
        
        for i, (feature_name, value) in enumerate(zip(self.feature_names, features)):
            stats = self._feature_stats[feature_name]
            if stats["count"] > 0 and stats["var"] > 0:
                z_score = abs(value - stats["mean"]) / np.sqrt(stats["var"])
                z_scores.append(z_score)
        
        # Return maximum Z-score across features
        return float(max(z_scores)) if z_scores else 0.0
    
    def _update_statistics(self, features: np.ndarray) -> None:
        """Update online statistics with new features."""
        for i, (feature_name, value) in enumerate(zip(self.feature_names, features)):
            stats = self._feature_stats[feature_name]
            
            # Online mean and variance update (Welford's algorithm)
            stats["count"] += 1
            delta = value - stats["mean"]
            stats["mean"] += delta / stats["count"]
            delta2 = value - stats["mean"]
            
            if stats["count"] > 1:
                stats["var"] = (
                    (stats["count"] - 2) * stats["var"] + delta * delta2
                ) / (stats["count"] - 1)
        
        # Keep recent data for windowed statistics
        self._recent_data.append(features)
        if len(self._recent_data) > self.window_size:
            self._recent_data.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        stats = self._stats.copy()
        if stats["processed_records"] > 0:
            stats["anomaly_rate"] = stats["anomalies_detected"] / stats["processed_records"]
        else:
            stats["anomaly_rate"] = 0.0
        
        stats["feature_statistics"] = self._feature_stats.copy()
        stats["window_data_size"] = len(self._recent_data)
        return stats


class EnsembleStreamProcessor(StreamProcessorProtocol):
    """Stream processor using ensemble of multiple processors."""
    
    def __init__(
        self,
        processors: List[StreamProcessorProtocol],
        voting_strategy: str = "average",
        weights: Optional[List[float]] = None
    ):
        """Initialize ensemble processor.
        
        Args:
            processors: List of stream processors
            voting_strategy: Strategy for combining results ('average', 'majority', 'weighted')
            weights: Weights for processors (used in weighted voting)
        """
        self.processors = processors
        self.voting_strategy = voting_strategy
        self.weights = weights or [1.0] * len(processors)
        
        if len(self.weights) != len(self.processors):
            self.weights = [1.0] * len(processors)
        
        self._stats = {"processed_records": 0, "anomalies_detected": 0}
    
    async def process_record(self, record: StreamRecord) -> Optional[StreamingResult]:
        """Process record using ensemble of processors."""
        try:
            # Get results from all processors
            results = []
            for processor in self.processors:
                result = await processor.process_record(record)
                if result:
                    results.append(result)
            
            if not results:
                return None
            
            # Combine results using voting strategy
            combined_result = self._combine_results(results, record)
            
            # Update statistics
            self._stats["processed_records"] += 1
            if combined_result.is_anomaly:
                self._stats["anomalies_detected"] += 1
            
            return combined_result
        
        except Exception as e:
            logger.error(f"Ensemble processing failed: {e}")
            return None
    
    async def process_batch(self, batch: StreamBatch) -> List[StreamingResult]:
        """Process batch using ensemble."""
        results = []
        
        try:
            for record in batch.records:
                result = await self.process_record(record)
                if result:
                    results.append(result)
        
        except Exception as e:
            logger.error(f"Ensemble batch processing failed: {e}")
        
        return results
    
    async def update_model(self, records: List[StreamRecord]) -> None:
        """Update all processors in ensemble."""
        for processor in self.processors:
            try:
                await processor.update_model(records)
            except Exception as e:
                logger.error(f"Processor update failed: {e}")
    
    def _combine_results(
        self,
        results: List[StreamingResult],
        record: StreamRecord
    ) -> StreamingResult:
        """Combine results from multiple processors."""
        if self.voting_strategy == "average":
            avg_score = np.mean([r.anomaly_score for r in results])
            avg_confidence = np.mean([r.confidence for r in results])
            is_anomaly = avg_score > 0.5
        
        elif self.voting_strategy == "majority":
            anomaly_votes = sum(1 for r in results if r.is_anomaly)
            is_anomaly = anomaly_votes > len(results) / 2
            avg_score = np.mean([r.anomaly_score for r in results])
            avg_confidence = anomaly_votes / len(results)
        
        elif self.voting_strategy == "weighted":
            # Weight by processor weights
            valid_results = results[:len(self.weights)]
            weighted_scores = [
                r.anomaly_score * w
                for r, w in zip(valid_results, self.weights[:len(valid_results)])
            ]
            avg_score = sum(weighted_scores) / sum(self.weights[:len(valid_results)])
            avg_confidence = np.mean([r.confidence for r in valid_results])
            is_anomaly = avg_score > 0.5
        
        else:
            # Default to average
            avg_score = np.mean([r.anomaly_score for r in results])
            avg_confidence = np.mean([r.confidence for r in results])
            is_anomaly = avg_score > 0.5
        
        return StreamingResult(
            record_id=record.id,
            timestamp=record.timestamp,
            anomaly_score=float(avg_score),
            is_anomaly=is_anomaly,
            confidence=float(avg_confidence),
            metadata={
                "processor": self.__class__.__name__,
                "ensemble_size": len(results),
                "voting_strategy": self.voting_strategy,
                "individual_results": [
                    {"score": r.anomaly_score, "is_anomaly": r.is_anomaly}
                    for r in results
                ]
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        stats = self._stats.copy()
        if stats["processed_records"] > 0:
            stats["anomaly_rate"] = stats["anomalies_detected"] / stats["processed_records"]
        else:
            stats["anomaly_rate"] = 0.0
        
        # Get statistics from individual processors
        stats["processor_statistics"] = []
        for i, processor in enumerate(self.processors):
            if hasattr(processor, 'get_statistics'):
                proc_stats = processor.get_statistics()
                proc_stats["processor_index"] = i
                stats["processor_statistics"].append(proc_stats)
        
        return stats