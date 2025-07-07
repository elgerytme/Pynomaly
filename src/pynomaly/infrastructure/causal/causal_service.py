"""Causal anomaly detection service for coordinating causal analysis operations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.models.causal import (
    AnomalyType,
    CausalAnalysisConfig,
    CausalAnomalyEvent,
    CausalDataset,
    CausalDetector,
    CausalGraph,
    CausalMethod,
    InterventionSpec,
)
from pynomaly.domain.value_objects import PerformanceMetrics
from pynomaly.infrastructure.causal.structure_learning import StructureLearningService

# Type alias for backward compatibility
ModelMetrics = PerformanceMetrics


class CausalAnomalyDetectionService:
    """Main service for causal anomaly detection operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Service components
        self.structure_learner = StructureLearningService()
        
        # Detector registry
        self.detectors: Dict[UUID, CausalDetector] = {}
        self.graphs: Dict[UUID, CausalGraph] = {}
        self.datasets: Dict[UUID, CausalDataset] = {}
        
        # Performance tracking
        self.detection_stats: Dict[str, Any] = {
            "total_detections": 0,
            "total_anomalies_found": 0,
            "average_detection_time": 0.0,
            "causal_score_distribution": [],
        }
        
        self.logger.info("Causal anomaly detection service initialized")
    
    async def create_detector(
        self,
        name: str,
        config: CausalAnalysisConfig,
    ) -> CausalDetector:
        """Create new causal anomaly detector."""
        
        detector_id = uuid4()
        
        detector = CausalDetector(
            detector_id=detector_id,
            name=name,
            config=config,
        )
        
        # Store detector
        self.detectors[detector_id] = detector
        
        self.logger.info(
            f"Created causal detector '{name}' using method {config.method.value}"
        )
        
        return detector
    
    async def register_dataset(
        self,
        name: str,
        data: np.ndarray,
        variable_names: List[str],
        is_time_series: bool = False,
        **metadata
    ) -> CausalDataset:
        """Register a dataset for causal analysis."""
        
        dataset_id = uuid4()
        
        dataset = CausalDataset(
            dataset_id=dataset_id,
            name=name,
            data=data,
            variable_names=variable_names,
            is_time_series=is_time_series,
            metadata=metadata,
        )
        
        self.datasets[dataset_id] = dataset
        
        self.logger.info(
            f"Registered dataset '{name}' with {data.shape[0]} samples, "
            f"{len(variable_names)} variables"
        )
        
        return dataset
    
    async def train_detector(
        self,
        detector_id: UUID,
        dataset_id: UUID,
        validation_split: float = 0.2,
    ) -> CausalDetector:
        """Train causal anomaly detector on dataset."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        detector = self.detectors[detector_id]
        dataset = self.datasets[dataset_id]
        
        self.logger.info(
            f"Training detector '{detector.name}' on dataset '{dataset.name}'"
        )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Split data for training and validation
            n_samples = dataset.data.shape[0]
            split_idx = int(n_samples * (1 - validation_split))
            
            train_data = dataset.data[:split_idx]
            val_data = dataset.data[split_idx:]
            
            # Learn baseline causal structure
            baseline_graph = await self.structure_learner.learn_causal_structure(
                data=train_data,
                variable_names=dataset.variable_names,
                method=detector.config.method,
                alpha=detector.config.alpha,
                max_lag=detector.config.max_lag,
            )
            
            detector.baseline_graph = baseline_graph
            self.graphs[baseline_graph.graph_id] = baseline_graph
            
            # Train anomaly detection mechanisms
            await self._train_anomaly_mechanisms(detector, train_data, dataset.variable_names)
            
            # Validate detector
            if val_data.shape[0] > 10:
                validation_metrics = await self._validate_detector(
                    detector, val_data, dataset.variable_names
                )
                detector.validation_metrics = validation_metrics
            
            # Finalize training
            detector.is_trained = True
            detector.training_samples = train_data.shape[0]
            detector.last_updated = datetime.utcnow()
            
            training_time = asyncio.get_event_loop().time() - start_time
            
            self.logger.info(
                f"Training completed for detector '{detector.name}' in {training_time:.2f}s. "
                f"Baseline graph: {len(baseline_graph.edges)} edges"
            )
            
            return detector
            
        except Exception as e:
            self.logger.error(f"Training failed for detector '{detector.name}': {e}")
            raise
    
    async def detect_causal_anomalies(
        self,
        detector_id: UUID,
        data: np.ndarray,
        variable_names: List[str],
        window_size: Optional[int] = None,
    ) -> List[CausalAnomalyEvent]:
        """Detect causal anomalies in new data."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        detector = self.detectors[detector_id]
        
        if not detector.is_trained:
            raise ValueError("Detector must be trained before detection")
        
        self.logger.info(f"Detecting causal anomalies with detector '{detector.name}'")
        
        start_time = asyncio.get_event_loop().time()
        
        anomalies = []
        
        if window_size is None:
            window_size = min(100, data.shape[0] // 2)
        
        # Sliding window analysis
        for i in range(0, data.shape[0] - window_size + 1, window_size // 2):
            window_data = data[i:i + window_size]
            
            # Detect anomalies in this window
            window_anomalies = await self._detect_window_anomalies(
                detector, window_data, variable_names, i
            )
            
            anomalies.extend(window_anomalies)
        
        detection_time = asyncio.get_event_loop().time() - start_time
        
        # Update statistics
        self._update_detection_stats(detection_time, len(anomalies))
        
        self.logger.info(
            f"Detection completed: {len(anomalies)} anomalies found in {detection_time:.2f}s"
        )
        
        return anomalies
    
    async def _detect_window_anomalies(
        self,
        detector: CausalDetector,
        window_data: np.ndarray,
        variable_names: List[str],
        window_offset: int,
    ) -> List[CausalAnomalyEvent]:
        """Detect anomalies in a data window."""
        
        anomalies = []
        
        # Learn current causal structure
        current_graph = await self.structure_learner.learn_causal_structure(
            data=window_data,
            variable_names=variable_names,
            method=detector.config.method,
            alpha=detector.config.alpha,
            max_lag=detector.config.max_lag,
        )
        
        # Compare with baseline graph
        structural_anomalies = await self._detect_structural_anomalies(
            detector, current_graph, window_offset
        )
        anomalies.extend(structural_anomalies)
        
        # Detect mechanism failures
        mechanism_anomalies = await self._detect_mechanism_anomalies(
            detector, window_data, variable_names, window_offset
        )
        anomalies.extend(mechanism_anomalies)
        
        # Detect intervention effects
        intervention_anomalies = await self._detect_intervention_anomalies(
            detector, window_data, variable_names, window_offset
        )
        anomalies.extend(intervention_anomalies)
        
        return anomalies
    
    async def _detect_structural_anomalies(
        self,
        detector: CausalDetector,
        current_graph: CausalGraph,
        window_offset: int,
    ) -> List[CausalAnomalyEvent]:
        """Detect structural changes in causal graph."""
        
        if not detector.baseline_graph:
            return []
        
        anomalies = []
        baseline = detector.baseline_graph
        
        # Compare edge sets
        baseline_edges = set((e.source, e.target) for e in baseline.edges)
        current_edges = set((e.source, e.target) for e in current_graph.edges)
        
        # Detect added edges
        added_edges = current_edges - baseline_edges
        if added_edges:
            anomaly = CausalAnomalyEvent(
                event_id=uuid4(),
                anomaly_type=AnomalyType.STRUCTURAL_BREAK,
                affected_variables=list(set(var for edge in added_edges for var in edge)),
                causal_score=min(1.0, len(added_edges) / len(baseline_edges) if baseline_edges else 1.0),
                timestamp=datetime.utcnow(),
                evidence={
                    "added_edges": list(added_edges),
                    "window_offset": window_offset,
                },
                confidence=0.8,
            )
            anomalies.append(anomaly)
        
        # Detect removed edges
        removed_edges = baseline_edges - current_edges
        if removed_edges:
            anomaly = CausalAnomalyEvent(
                event_id=uuid4(),
                anomaly_type=AnomalyType.MECHANISM_FAILURE,
                affected_variables=list(set(var for edge in removed_edges for var in edge)),
                causal_score=min(1.0, len(removed_edges) / len(baseline_edges) if baseline_edges else 1.0),
                timestamp=datetime.utcnow(),
                evidence={
                    "removed_edges": list(removed_edges),
                    "window_offset": window_offset,
                },
                confidence=0.8,
            )
            anomalies.append(anomaly)
        
        # Detect strength changes in existing edges
        for baseline_edge in baseline.edges:
            current_edge = current_graph.get_edge(baseline_edge.source, baseline_edge.target)
            if current_edge:
                strength_diff = abs(current_edge.strength - baseline_edge.strength)
                if strength_diff > 0.3:  # Significant strength change
                    anomaly = CausalAnomalyEvent(
                        event_id=uuid4(),
                        anomaly_type=AnomalyType.MECHANISM_FAILURE,
                        affected_variables=[baseline_edge.source, baseline_edge.target],
                        causal_score=strength_diff,
                        timestamp=datetime.utcnow(),
                        evidence={
                            "edge": (baseline_edge.source, baseline_edge.target),
                            "baseline_strength": baseline_edge.strength,
                            "current_strength": current_edge.strength,
                            "strength_change": strength_diff,
                            "window_offset": window_offset,
                        },
                        confidence=0.7,
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_mechanism_anomalies(
        self,
        detector: CausalDetector,
        window_data: np.ndarray,
        variable_names: List[str],
        window_offset: int,
    ) -> List[CausalAnomalyEvent]:
        """Detect failures in learned causal mechanisms."""
        
        anomalies = []
        
        if not detector.baseline_graph:
            return anomalies
        
        # Check for unexpected patterns in the data
        for i, var_name in enumerate(variable_names):
            var_data = window_data[:, i]
            
            # Simple anomaly detection: outliers
            mean_val = np.mean(var_data)
            std_val = np.std(var_data)
            outliers = np.abs(var_data - mean_val) > 3 * std_val
            
            if np.any(outliers):
                outlier_ratio = np.sum(outliers) / len(var_data)
                
                if outlier_ratio > 0.1:  # More than 10% outliers
                    anomaly = CausalAnomalyEvent(
                        event_id=uuid4(),
                        anomaly_type=AnomalyType.MECHANISM_FAILURE,
                        affected_variables=[var_name],
                        causal_score=min(1.0, outlier_ratio * 5),
                        timestamp=datetime.utcnow(),
                        evidence={
                            "variable": var_name,
                            "outlier_ratio": outlier_ratio,
                            "outlier_indices": np.where(outliers)[0].tolist(),
                            "window_offset": window_offset,
                        },
                        confidence=0.6,
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_intervention_anomalies(
        self,
        detector: CausalDetector,
        window_data: np.ndarray,
        variable_names: List[str],
        window_offset: int,
    ) -> List[CausalAnomalyEvent]:
        """Detect unexpected intervention effects."""
        
        anomalies = []
        
        # Look for sudden changes that might indicate interventions
        if window_data.shape[0] < 10:
            return anomalies
        
        for i, var_name in enumerate(variable_names):
            var_data = window_data[:, i]
            
            # Detect changepoints using simple method
            mid_point = len(var_data) // 2
            first_half_mean = np.mean(var_data[:mid_point])
            second_half_mean = np.mean(var_data[mid_point:])
            
            change_magnitude = abs(second_half_mean - first_half_mean)
            var_std = np.std(var_data)
            
            if var_std > 0 and change_magnitude > 2 * var_std:
                anomaly = CausalAnomalyEvent(
                    event_id=uuid4(),
                    anomaly_type=AnomalyType.INTERVENTION_EFFECT,
                    affected_variables=[var_name],
                    causal_score=min(1.0, change_magnitude / var_std / 5),
                    timestamp=datetime.utcnow(),
                    evidence={
                        "variable": var_name,
                        "change_magnitude": change_magnitude,
                        "normalized_change": change_magnitude / var_std,
                        "changepoint": mid_point + window_offset,
                        "window_offset": window_offset,
                    },
                    confidence=0.7,
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _train_anomaly_mechanisms(
        self,
        detector: CausalDetector,
        training_data: np.ndarray,
        variable_names: List[str],
    ) -> None:
        """Train anomaly detection mechanisms."""
        
        # Store basic statistics for each variable
        for i, var_name in enumerate(variable_names):
            var_data = training_data[:, i]
            
            detector.learned_mechanisms[var_name] = {
                "mean": float(np.mean(var_data)),
                "std": float(np.std(var_data)),
                "min": float(np.min(var_data)),
                "max": float(np.max(var_data)),
                "percentiles": {
                    "5": float(np.percentile(var_data, 5)),
                    "95": float(np.percentile(var_data, 95)),
                },
            }
        
        # Store pairwise correlations
        if len(variable_names) > 1:
            correlation_matrix = np.corrcoef(training_data.T)
            detector.learned_mechanisms["correlations"] = correlation_matrix.tolist()
    
    async def _validate_detector(
        self,
        detector: CausalDetector,
        validation_data: np.ndarray,
        variable_names: List[str],
    ) -> ModelMetrics:
        """Validate trained detector."""
        
        # Simple validation: check if detector can distinguish anomalies
        # This is a placeholder - real validation would need labeled data
        
        # Inject synthetic anomalies for validation
        anomalous_data = validation_data.copy()
        n_samples = anomalous_data.shape[0]
        
        # Add outliers to 10% of samples
        anomaly_indices = np.random.choice(n_samples, size=n_samples // 10, replace=False)
        for idx in anomaly_indices:
            var_idx = np.random.choice(len(variable_names))
            anomalous_data[idx, var_idx] += np.random.normal(0, 5)  # Large outlier
        
        # Detect anomalies
        detected_anomalies = await self.detect_causal_anomalies(
            detector.detector_id, anomalous_data, variable_names
        )
        
        # Calculate basic metrics
        true_positive_rate = len(detected_anomalies) / len(anomaly_indices) if anomaly_indices.size > 0 else 0
        precision = min(1.0, true_positive_rate)  # Simplified
        recall = true_positive_rate
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return ModelMetrics(
            accuracy=precision,  # Simplified metric
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            training_time=0.0,  # Placeholder
            inference_time=1.0,  # Placeholder
            model_size=1024,  # Placeholder
        )
    
    async def simulate_intervention(
        self,
        detector_id: UUID,
        intervention: InterventionSpec,
        simulation_steps: int = 100,
    ) -> Dict[str, Any]:
        """Simulate the effect of a causal intervention."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        detector = self.detectors[detector_id]
        
        if not detector.is_trained or not detector.baseline_graph:
            raise ValueError("Detector must be trained with baseline graph")
        
        self.logger.info(
            f"Simulating intervention on '{intervention.target_variable}' "
            f"for {simulation_steps} steps"
        )
        
        # Use detector's simulation method
        simulation_result = detector.simulate_intervention(
            intervention, np.array([]), simulation_steps
        )
        
        # Store intervention in history
        detector.add_intervention(intervention)
        
        return simulation_result
    
    async def explain_anomaly(
        self,
        detector_id: UUID,
        anomaly: CausalAnomalyEvent,
    ) -> Dict[str, Any]:
        """Provide detailed causal explanation for an anomaly."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        detector = self.detectors[detector_id]
        
        # Get explanation from detector
        explanation = detector.explain_anomaly(anomaly)
        
        # Add service-level context
        explanation["service_context"] = {
            "detector_name": detector.name,
            "detection_method": detector.config.method.value,
            "baseline_graph_available": detector.baseline_graph is not None,
            "total_anomalies_detected": len(detector.detected_anomalies),
        }
        
        return explanation
    
    async def get_detector_summary(self, detector_id: UUID) -> Dict[str, Any]:
        """Get comprehensive detector summary."""
        
        if detector_id not in self.detectors:
            raise ValueError(f"Detector {detector_id} not found")
        
        detector = self.detectors[detector_id]
        summary = detector.get_detector_summary()
        
        # Add service-level information
        summary["service_statistics"] = {
            "total_detectors": len(self.detectors),
            "total_graphs": len(self.graphs),
            "total_datasets": len(self.datasets),
        }
        
        return summary
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        
        detector_stats = {
            "total_detectors": len(self.detectors),
            "trained_detectors": sum(1 for d in self.detectors.values() if d.is_trained),
            "methods_used": list(set(d.config.method.value for d in self.detectors.values())),
        }
        
        graph_stats = {
            "total_graphs": len(self.graphs),
            "graph_types": list(set(g.graph_type.value for g in self.graphs.values())),
            "average_edges": np.mean([len(g.edges) for g in self.graphs.values()]) if self.graphs else 0,
        }
        
        dataset_stats = {
            "total_datasets": len(self.datasets),
            "time_series_datasets": sum(1 for d in self.datasets.values() if d.is_time_series),
            "experimental_datasets": sum(1 for d in self.datasets.values() if d.is_experimental),
        }
        
        return {
            "detector_statistics": detector_stats,
            "graph_statistics": graph_stats,
            "dataset_statistics": dataset_stats,
            "detection_statistics": self.detection_stats,
            "supported_methods": [method.value for method in self.structure_learner.get_supported_methods()],
        }
    
    def _update_detection_stats(self, detection_time: float, num_anomalies: int) -> None:
        """Update detection statistics."""
        
        self.detection_stats["total_detections"] += 1
        self.detection_stats["total_anomalies_found"] += num_anomalies
        
        # Update average detection time
        total_time = (
            self.detection_stats["average_detection_time"] * 
            (self.detection_stats["total_detections"] - 1) + 
            detection_time
        )
        self.detection_stats["average_detection_time"] = total_time / self.detection_stats["total_detections"]
    
    async def cleanup_resources(self, keep_trained: bool = True) -> int:
        """Clean up resources to free memory."""
        
        cleanup_count = 0
        
        # Clean up detectors
        detectors_to_remove = []
        for detector_id, detector in self.detectors.items():
            if not keep_trained or not detector.is_trained:
                detectors_to_remove.append(detector_id)
        
        for detector_id in detectors_to_remove:
            del self.detectors[detector_id]
            cleanup_count += 1
        
        # Clean up orphaned graphs and datasets
        # (Simple cleanup - in production would be more sophisticated)
        
        self.logger.info(f"Cleaned up {cleanup_count} resources")
        
        return cleanup_count