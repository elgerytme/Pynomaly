"""Optimized PyOD adapter with performance enhancements and memory management."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import AnomalyScore, PerformanceMetrics
from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter


class OptimizedPyODAdapter(PyODAdapter):
    """Performance-optimized PyOD adapter with intelligent feature selection and memory management."""
    
    def __init__(
        self,
        algorithm: str,
        parameters: Optional[Dict[str, Any]] = None,
        feature_importance_threshold: float = 0.01,
        enable_feature_selection: bool = True,
        enable_batch_processing: bool = True,
        batch_size: int = 10000,
        enable_prediction_cache: bool = True,
        memory_optimization: bool = True,
        max_workers: int = 4,
    ):
        """Initialize optimized PyOD adapter.
        
        Args:
            algorithm: PyOD algorithm name
            parameters: Algorithm parameters
            feature_importance_threshold: Threshold for feature selection
            enable_feature_selection: Enable automatic feature selection
            enable_batch_processing: Enable batch processing for large datasets
            batch_size: Batch size for processing
            enable_prediction_cache: Cache prediction results
            memory_optimization: Enable memory optimizations
            max_workers: Maximum worker threads for parallel processing
        """
        super().__init__(algorithm, parameters)
        
        self.feature_importance_threshold = feature_importance_threshold
        self.enable_feature_selection = enable_feature_selection
        self.enable_batch_processing = enable_batch_processing
        self.batch_size = batch_size
        self.enable_prediction_cache = enable_prediction_cache
        self.memory_optimization = memory_optimization
        self.max_workers = max_workers
        
        # Feature selection state
        self._selected_features: Optional[np.ndarray] = None
        self._feature_selector: Optional[VarianceThreshold] = None
        self._scaler: Optional[StandardScaler] = None
        
        # Performance tracking
        self._prediction_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._performance_metrics: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def train(self, dataset: Dataset) -> Detector:
        """Train with performance optimizations."""
        start_time = time.time()
        self.logger.info(f"Training {self.algorithm} with optimizations")
        
        try:
            # Get features with optimization
            X = await self._prepare_features_optimized(dataset, training=True)
            
            # Train model
            if self.enable_batch_processing and len(X) > self.batch_size:
                detector = await self._train_batch_optimized(X, dataset)
            else:
                detector = await self._train_standard_optimized(X, dataset)
            
            # Record performance metrics
            training_time = time.time() - start_time
            detector.metadata["training_time_seconds"] = training_time
            detector.metadata["optimizations_enabled"] = {
                "feature_selection": self.enable_feature_selection,
                "batch_processing": self.enable_batch_processing,
                "memory_optimization": self.memory_optimization,
                "prediction_cache": self.enable_prediction_cache
            }
            
            if self._selected_features is not None:
                detector.metadata["selected_features_count"] = int(np.sum(self._selected_features))
                detector.metadata["feature_reduction_ratio"] = float(
                    np.sum(self._selected_features) / len(self._selected_features)
                )
            
            self.logger.info(f"Training completed in {training_time:.2f}s")
            return detector
            
        except Exception as e:
            self.logger.error(f"Optimized training failed: {e}")
            raise
    
    async def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect with performance optimizations."""
        start_time = time.time()
        
        try:
            # Check prediction cache
            if self.enable_prediction_cache:
                cache_key = self._get_dataset_hash(dataset)
                if cache_key in self._prediction_cache:
                    self.logger.info("Using cached prediction results")
                    anomaly_indices, scores = self._prediction_cache[cache_key]
                    return self._create_detection_result_optimized(
                        anomaly_indices, scores, dataset, use_cache=True
                    )
            
            # Prepare features
            X = await self._prepare_features_optimized(dataset, training=False)
            
            # Detect anomalies
            if self.enable_batch_processing and len(X) > self.batch_size:
                result = await self._detect_batch_optimized(X, dataset)
            else:
                result = await self._detect_standard_optimized(X, dataset)
            
            # Cache results
            if self.enable_prediction_cache:
                cache_key = self._get_dataset_hash(dataset)
                self._prediction_cache[cache_key] = (
                    np.array([a.index for a in result.anomalies]),
                    np.array([a.score.value for a in result.anomalies])
                )
            
            detection_time = time.time() - start_time
            result.metadata["detection_time_seconds"] = detection_time
            
            self.logger.info(f"Detection completed in {detection_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Optimized detection failed: {e}")
            raise
    
    async def _prepare_features_optimized(self, dataset: Dataset, training: bool = False) -> np.ndarray:
        """Prepare features with optimizations."""
        # Get numeric features
        numeric_features = dataset.get_numeric_features()
        X = dataset.data[numeric_features].values
        
        # Apply feature selection
        if self.enable_feature_selection:
            X = self._apply_feature_selection(X, training=training)
        
        # Apply scaling
        if training:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        elif self._scaler is not None:
            X = self._scaler.transform(X)
        
        # Memory optimization
        if self.memory_optimization:
            X = X.astype(np.float32)  # Use float32 instead of float64
            
            # Force garbage collection
            gc.collect()
        
        return X
    
    def _apply_feature_selection(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply intelligent feature selection."""
        if training:
            # Create feature selector
            self._feature_selector = VarianceThreshold(threshold=self.feature_importance_threshold)
            X_selected = self._feature_selector.fit_transform(X)
            self._selected_features = self._feature_selector.get_support()
            
            # Log feature selection results
            n_original = X.shape[1]
            n_selected = X_selected.shape[1]
            reduction_ratio = n_selected / n_original
            
            self.logger.info(
                f"Feature selection: {n_selected}/{n_original} features selected "
                f"({reduction_ratio:.2%} retention)"
            )
            
            return X_selected
        else:
            # Apply existing feature selection
            if self._feature_selector is not None:
                return self._feature_selector.transform(X)
            return X
    
    async def _train_batch_optimized(self, X: np.ndarray, dataset: Dataset) -> Detector:
        """Train using batch processing for large datasets."""
        self.logger.info(f"Using batch training with batch size: {self.batch_size}")
        
        # For unsupervised algorithms, we can train on a representative sample
        if len(X) > self.batch_size * 5:  # Only sample if dataset is significantly larger
            # Use stratified sampling to maintain data distribution
            sample_indices = self._get_representative_sample(X, self.batch_size * 2)
            X_sample = X[sample_indices]
            
            self.logger.info(f"Training on representative sample: {len(X_sample)} samples")
        else:
            X_sample = X
        
        # Train model
        self._model.fit(X_sample)
        
        return self._create_detector_optimized(dataset)
    
    async def _train_standard_optimized(self, X: np.ndarray, dataset: Dataset) -> Detector:
        """Standard training with optimizations."""
        # Run training in thread pool for CPU-intensive operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, self._model.fit, X)
        
        return self._create_detector_optimized(dataset)
    
    async def _detect_batch_optimized(self, X: np.ndarray, dataset: Dataset) -> DetectionResult:
        """Detect using batch processing for large datasets."""
        self.logger.info(f"Using batch detection with batch size: {self.batch_size}")
        
        all_anomaly_indices = []
        all_scores = []
        
        # Process in batches
        num_batches = (len(X) + self.batch_size - 1) // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(X))
            
            X_batch = X[start_idx:end_idx]
            
            # Process batch
            batch_labels = self._model.predict(X_batch)
            batch_scores = self._model.decision_function(X_batch)
            
            # Find anomalies in batch
            batch_anomaly_indices = np.where(batch_labels == 1)[0] + start_idx
            batch_anomaly_scores = batch_scores[batch_labels == 1]
            
            all_anomaly_indices.extend(batch_anomaly_indices)
            all_scores.extend(batch_anomaly_scores)
            
            # Log progress
            if (i + 1) % 10 == 0 or i == num_batches - 1:
                self.logger.info(f"Processed batch {i + 1}/{num_batches}")
                
                # Memory cleanup for large datasets
                if self.memory_optimization:
                    gc.collect()
        
        return self._create_detection_result_optimized(
            np.array(all_anomaly_indices), np.array(all_scores), dataset
        )
    
    async def _detect_standard_optimized(self, X: np.ndarray, dataset: Dataset) -> DetectionResult:
        """Standard detection with optimizations."""
        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            labels_future = loop.run_in_executor(executor, self._model.predict, X)
            scores_future = loop.run_in_executor(executor, self._model.decision_function, X)
            
            labels, scores = await asyncio.gather(labels_future, scores_future)
        
        # Find anomalies
        anomaly_indices = np.where(labels == 1)[0]
        anomaly_scores = scores[labels == 1]
        
        return self._create_detection_result_optimized(anomaly_indices, anomaly_scores, dataset)
    
    def _get_representative_sample(self, X: np.ndarray, sample_size: int) -> np.ndarray:
        """Get representative sample using intelligent sampling."""
        if len(X) <= sample_size:
            return np.arange(len(X))
        
        # Use k-means based sampling for better representation
        try:
            from sklearn.cluster import MiniBatchKMeans
            
            # Create clusters
            n_clusters = min(sample_size // 10, 50)  # Reasonable number of clusters
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
            cluster_labels = kmeans.fit_predict(X)
            
            # Sample from each cluster proportionally
            unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
            samples_per_cluster = (sample_size * cluster_counts / len(X)).astype(int)
            
            sample_indices = []
            for cluster_id, n_samples in zip(unique_clusters, samples_per_cluster):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    selected = np.random.choice(
                        cluster_indices, 
                        size=min(n_samples, len(cluster_indices)), 
                        replace=False
                    )
                    sample_indices.extend(selected)
            
            return np.array(sample_indices)
            
        except ImportError:
            # Fallback to random sampling
            return np.random.choice(len(X), size=sample_size, replace=False)
    
    def _get_dataset_hash(self, dataset: Dataset) -> str:
        """Generate hash for dataset to use as cache key."""
        # Create hash based on data content and shape
        data_str = f"{dataset.data.shape}_{dataset.data.dtypes.to_string()}"
        if len(dataset.data) < 10000:  # For small datasets, include actual data
            data_str += dataset.data.to_string()
        else:  # For large datasets, use sample
            sample = dataset.data.sample(n=100, random_state=42)
            data_str += sample.to_string()
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _create_detector_optimized(self, dataset: Dataset) -> Detector:
        """Create detector with optimization metadata."""
        detector = super()._create_detector(dataset)
        
        # Add optimization metadata
        detector.metadata.update({
            "feature_selection_enabled": self.enable_feature_selection,
            "batch_processing_enabled": self.enable_batch_processing,
            "memory_optimization_enabled": self.memory_optimization,
            "prediction_cache_enabled": self.enable_prediction_cache,
        })
        
        if self._selected_features is not None:
            detector.metadata["selected_features_mask"] = self._selected_features.tolist()
        
        return detector
    
    def _create_detection_result_optimized(
        self, 
        anomaly_indices: np.ndarray, 
        scores: np.ndarray, 
        dataset: Dataset,
        use_cache: bool = False
    ) -> DetectionResult:
        """Create detection result with optimization metadata."""
        result = super()._create_detection_result(anomaly_indices, scores, dataset)
        
        # Add optimization metadata
        result.metadata.update({
            "used_cache": use_cache,
            "feature_selection_applied": self._selected_features is not None,
            "batch_processing_used": self.enable_batch_processing and len(dataset.data) > self.batch_size,
            "memory_optimized": self.memory_optimization,
        })
        
        if hasattr(self, '_performance_metrics'):
            result.metadata["performance_metrics"] = self._performance_metrics
        
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "algorithm": self.algorithm,
            "optimizations": {
                "feature_selection": self.enable_feature_selection,
                "batch_processing": self.enable_batch_processing,
                "memory_optimization": self.memory_optimization,
                "prediction_cache": self.enable_prediction_cache,
            },
            "configuration": {
                "feature_importance_threshold": self.feature_importance_threshold,
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
            },
            "feature_selection_stats": {
                "enabled": self.enable_feature_selection,
                "features_selected": int(np.sum(self._selected_features)) if self._selected_features is not None else None,
                "total_features": len(self._selected_features) if self._selected_features is not None else None,
                "reduction_ratio": float(np.sum(self._selected_features) / len(self._selected_features)) if self._selected_features is not None else None,
            },
            "cache_stats": {
                "enabled": self.enable_prediction_cache,
                "cached_predictions": len(self._prediction_cache),
            },
            "performance_metrics": self._performance_metrics,
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache to free memory."""
        self._prediction_cache.clear()
        gc.collect()
        self.logger.info("Prediction cache cleared")


class AsyncAlgorithmExecutor:
    """Asynchronous executor for running multiple algorithms in parallel."""
    
    def __init__(self, max_concurrent: int = 4, timeout: Optional[float] = None):
        """Initialize async algorithm executor.
        
        Args:
            max_concurrent: Maximum number of concurrent algorithm executions
            timeout: Timeout for individual algorithm execution
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.logger = logging.getLogger(__name__)
    
    async def execute_multiple_algorithms(
        self, 
        algorithms: List[str], 
        dataset: Dataset,
        algorithm_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Tuple[str, Optional[DetectionResult]]]:
        """Execute multiple algorithms in parallel.
        
        Args:
            algorithms: List of algorithm names to execute
            dataset: Dataset to process
            algorithm_params: Optional parameters for each algorithm
            
        Returns:
            List of (algorithm_name, result) tuples
        """
        self.logger.info(f"Executing {len(algorithms)} algorithms in parallel")
        
        # Create adapters
        adapters = []
        for algo in algorithms:
            params = algorithm_params.get(algo, {}) if algorithm_params else {}
            adapter = OptimizedPyODAdapter(algorithm=algo, parameters=params)
            adapters.append((algo, adapter))
        
        # Execute training in parallel
        training_tasks = [
            self._execute_algorithm_training(algo_name, adapter, dataset)
            for algo_name, adapter in adapters
        ]
        
        trained_detectors = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # Execute detection in parallel
        detection_tasks = []
        for i, (algo_name, adapter) in enumerate(adapters):
            if not isinstance(trained_detectors[i], Exception):
                detection_tasks.append(
                    self._execute_algorithm_detection(algo_name, adapter, dataset)
                )
            else:
                detection_tasks.append(
                    asyncio.create_task(self._handle_exception(algo_name, trained_detectors[i]))
                )
        
        results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Combine results
        final_results = []
        for i, algo_name in enumerate(algorithms):
            if isinstance(results[i], Exception):
                self.logger.error(f"Algorithm {algo_name} failed: {results[i]}")
                final_results.append((algo_name, None))
            else:
                final_results.append((algo_name, results[i]))
        
        successful_algos = sum(1 for _, result in final_results if result is not None)
        self.logger.info(f"Completed: {successful_algos}/{len(algorithms)} algorithms successful")
        
        return final_results
    
    async def _execute_algorithm_training(
        self, 
        algo_name: str, 
        adapter: OptimizedPyODAdapter, 
        dataset: Dataset
    ) -> Detector:
        """Execute algorithm training with timeout and semaphore."""
        async with self.semaphore:
            try:
                if self.timeout:
                    detector = await asyncio.wait_for(
                        adapter.train(dataset), 
                        timeout=self.timeout
                    )
                else:
                    detector = await adapter.train(dataset)
                
                self.logger.info(f"Training completed for {algo_name}")
                return detector
                
            except asyncio.TimeoutError:
                self.logger.error(f"Training timeout for {algo_name}")
                raise
            except Exception as e:
                self.logger.error(f"Training failed for {algo_name}: {e}")
                raise
    
    async def _execute_algorithm_detection(
        self, 
        algo_name: str, 
        adapter: OptimizedPyODAdapter, 
        dataset: Dataset
    ) -> DetectionResult:
        """Execute algorithm detection with timeout and semaphore."""
        async with self.semaphore:
            try:
                if self.timeout:
                    result = await asyncio.wait_for(
                        adapter.detect(dataset), 
                        timeout=self.timeout
                    )
                else:
                    result = await adapter.detect(dataset)
                
                self.logger.info(f"Detection completed for {algo_name}")
                return result
                
            except asyncio.TimeoutError:
                self.logger.error(f"Detection timeout for {algo_name}")
                raise
            except Exception as e:
                self.logger.error(f"Detection failed for {algo_name}: {e}")
                raise
    
    async def _handle_exception(self, algo_name: str, exception: Exception) -> None:
        """Handle algorithm execution exception."""
        self.logger.error(f"Algorithm {algo_name} failed: {exception}")
        return None
    
    async def close(self) -> None:
        """Close the executor."""
        self.executor.shutdown(wait=True)