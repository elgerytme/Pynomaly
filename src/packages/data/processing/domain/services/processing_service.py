"""Consolidated processing service for both batch and streaming anomaly detection."""

import asyncio
import json
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog

from .detection_service import DetectionService, DetectionResult
from ..entities.dataset import Dataset, DatasetType, DatasetMetadata
from ...infrastructure.repositories.model_repository import ModelRepository

logger = structlog.get_logger(__name__)


class ProcessingService:
    """
    Consolidated service for both batch and streaming processing of anomaly detection tasks.
    
    Combines:
    - Batch processing for large-scale data processing
    - Streaming processing for real-time anomaly detection
    - Unified configuration and monitoring
    """
    
    def __init__(
        self,
        detection_service: Optional[DetectionService] = None,
        model_repository: Optional[ModelRepository] = None,
        # Batch processing parameters
        parallel_jobs: int = 4,
        chunk_size: int = 1000,
        # Streaming processing parameters
        window_size: int = 1000,
        update_frequency: int = 100
    ):
        self.detection_service = detection_service or DetectionService()
        self.model_repository = model_repository
        
        # Batch processing configuration
        self.parallel_jobs = parallel_jobs
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=parallel_jobs)
        
        # Streaming processing configuration
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        # Streaming state
        self._data_buffer: deque = deque(maxlen=window_size)
        self._sample_count = 0
        self._last_update = 0
        self._model_fitted = False
        self._current_algorithm = "iforest"
        self._processing_times: deque = deque(maxlen=1000)
        
        # Memory management
        self._initial_memory_mb = self._get_memory_usage()
        
        # Processing mode
        self._batch_mode = True
        self._stream_lock = threading.Lock()

    # ==============================================================================
    # BATCH PROCESSING METHODS
    # ==============================================================================
    
    async def process_file(
        self,
        input_file: Path,
        output_dir: Path,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Process a single file for anomaly detection in batch mode."""
        start_time = time.time()
        
        try:
            logger.info("Starting batch file processing", file=str(input_file))
            
            # Load data
            data = await self._load_file_data(input_file)
            if data is None or data.empty:
                raise ValueError(f"Could not load data from {input_file}")
            
            # Process data in chunks
            results = []
            total_chunks = len(data) // self.chunk_size + (1 if len(data) % self.chunk_size else 0)
            
            for i, chunk_start in enumerate(range(0, len(data), self.chunk_size)):
                chunk_end = min(chunk_start + self.chunk_size, len(data))
                chunk = data.iloc[chunk_start:chunk_end]
                
                # Process chunk
                chunk_result = await self._process_chunk(chunk, config)
                results.append(chunk_result)
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / total_chunks
                    progress_callback(progress)
                
                logger.debug("Processed chunk", 
                           chunk_number=i+1, 
                           total_chunks=total_chunks,
                           chunk_size=len(chunk))
            
            # Combine results
            combined_result = await self._combine_batch_results(results, data)
            
            # Save results
            output_file = await self._save_batch_results(combined_result, output_dir, input_file)
            
            processing_time = time.time() - start_time
            logger.info("Batch file processing completed",
                       file=str(input_file),
                       processing_time=processing_time,
                       anomalies_found=len(combined_result.get('anomaly_indices', [])))
            
            return {
                'input_file': str(input_file),
                'output_file': str(output_file),
                'processing_time': processing_time,
                'total_samples': len(data),
                'anomalies_found': len(combined_result.get('anomaly_indices', [])),
                'result': combined_result
            }
            
        except Exception as e:
            logger.error("Batch file processing failed", file=str(input_file), error=str(e))
            raise

    async def process_batch(
        self,
        input_files: List[Path],
        output_dir: Path,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """Process multiple files concurrently in batch mode."""
        
        start_time = time.time()
        results = {
            'total_files': len(input_files),
            'successful': 0,
            'failed': 0,
            'results': [],
            'errors': []
        }
        
        logger.info("Starting batch processing", total_files=len(input_files))
        
        # Process files concurrently
        semaphore = asyncio.Semaphore(self.parallel_jobs)
        
        async def process_single_file(file_path: Path):
            async with semaphore:
                try:
                    file_progress_callback = None
                    if progress_callback:
                        file_progress_callback = lambda p: progress_callback(str(file_path), p)
                    
                    result = await self.process_file(file_path, output_dir, config, file_progress_callback)
                    return {'success': True, 'result': result}
                except Exception as e:
                    return {'success': False, 'file': str(file_path), 'error': str(e)}
        
        # Execute batch processing
        tasks = [process_single_file(file_path) for file_path in input_files]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in batch_results:
            if isinstance(result, dict):
                results['results'].append(result)
                if result['success']:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(result)
            else:
                results['failed'] += 1
                results['errors'].append({'error': str(result)})
        
        total_time = time.time() - start_time
        logger.info("Batch processing completed",
                   total_files=results['total_files'],
                   successful=results['successful'],
                   failed=results['failed'],
                   total_time=total_time)
        
        results['total_processing_time'] = total_time
        return results

    # ==============================================================================
    # STREAMING PROCESSING METHODS
    # ==============================================================================
    
    def start_streaming(self, algorithm: str = "iforest") -> None:
        """Start streaming mode for real-time processing."""
        with self._stream_lock:
            self._batch_mode = False
            self._current_algorithm = algorithm
            self._sample_count = 0
            self._last_update = 0
            self._model_fitted = False
            self._data_buffer.clear()
            self._processing_times.clear()
            
        logger.info("Streaming mode started", algorithm=algorithm)
    
    def stop_streaming(self) -> Dict[str, Any]:
        """Stop streaming mode and return statistics."""
        with self._stream_lock:
            stats = self.get_streaming_stats()
            self._batch_mode = True
            
        logger.info("Streaming mode stopped")
        return stats
    
    def process_stream_sample(self, sample: npt.ArrayLike) -> DetectionResult:
        """Process a single sample in streaming mode."""
        if self._batch_mode:
            raise RuntimeError("Service is in batch mode. Call start_streaming() first.")
        
        start_time = time.time()
        
        with self._stream_lock:
            # Convert to numpy array
            sample_array = np.array(sample).reshape(1, -1)
            
            # Add to buffer
            self._data_buffer.append(sample_array[0])
            self._sample_count += 1
            
            # Check if we need to update the model
            should_update = (
                not self._model_fitted or 
                (self._sample_count - self._last_update) >= self.update_frequency
            )
            
            if should_update and len(self._data_buffer) >= 10:  # Minimum samples needed
                self._update_streaming_model()
            
            # Perform detection
            if self._model_fitted:
                result = self.detection_service.detect(
                    sample_array,
                    algorithm=self._current_algorithm
                )
            else:
                # No model yet, assume normal
                result = DetectionResult(
                    is_anomaly=[False],
                    anomaly_score=[0.0],
                    algorithm=self._current_algorithm,
                    model_info={'status': 'training'}
                )
            
            # Track processing time
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)
            
            return result
    
    def process_stream_batch(self, samples: npt.ArrayLike) -> List[DetectionResult]:
        """Process multiple samples in streaming mode."""
        if self._batch_mode:
            raise RuntimeError("Service is in batch mode. Call start_streaming() first.")
        
        samples_array = np.array(samples)
        if samples_array.ndim == 1:
            samples_array = samples_array.reshape(1, -1)
        
        results = []
        for sample in samples_array:
            result = self.process_stream_sample(sample)
            results.append(result)
        
        return results
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming processing statistics."""
        with self._stream_lock:
            if self._processing_times:
                avg_processing_time = np.mean(self._processing_times)
                max_processing_time = np.max(self._processing_times)
                min_processing_time = np.min(self._processing_times)
            else:
                avg_processing_time = max_processing_time = min_processing_time = 0.0
            
            current_memory = self._get_memory_usage()
            memory_increase = current_memory - self._initial_memory_mb
            
            return {
                'total_samples_processed': self._sample_count,
                'buffer_size': len(self._data_buffer),
                'buffer_capacity': self.window_size,
                'model_fitted': self._model_fitted,
                'last_model_update': self._last_update,
                'current_algorithm': self._current_algorithm,
                'performance': {
                    'avg_processing_time_ms': avg_processing_time * 1000,
                    'max_processing_time_ms': max_processing_time * 1000,
                    'min_processing_time_ms': min_processing_time * 1000,
                    'samples_per_second': 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                },
                'memory': {
                    'current_mb': current_memory,
                    'initial_mb': self._initial_memory_mb,
                    'increase_mb': memory_increase
                }
            }
    
    def create_stream_generator(
        self,
        data_source: Any,
        batch_size: int = 1
    ) -> Generator[List[DetectionResult], None, None]:
        """Create a generator for processing streaming data."""
        if self._batch_mode:
            raise RuntimeError("Service is in batch mode. Call start_streaming() first.")
        
        # This is a template method - actual implementation depends on data source type
        # Could be Kafka, file stream, API endpoint, etc.
        
        if isinstance(data_source, pd.DataFrame):
            # Process DataFrame as streaming data
            for i in range(0, len(data_source), batch_size):
                batch = data_source.iloc[i:i+batch_size].values
                results = self.process_stream_batch(batch)
                yield results
        else:
            raise NotImplementedError("Data source type not supported yet")

    # ==============================================================================
    # UNIFIED PROCESSING METHODS
    # ==============================================================================
    
    async def auto_process(
        self,
        data_source: Any,
        config: Dict[str, Any],
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """Automatically choose and execute the best processing mode."""
        
        if mode == "auto":
            # Decide based on data characteristics
            if isinstance(data_source, (list, tuple)) and len(data_source) > 1:
                # Multiple files - use batch mode
                mode = "batch"
            elif isinstance(data_source, pd.DataFrame) and len(data_source) > 10000:
                # Large dataset - use batch mode
                mode = "batch"
            else:
                # Default to batch for now
                mode = "batch"
        
        if mode == "batch":
            if isinstance(data_source, list):
                return await self.process_batch(data_source, Path(config.get('output_dir', '.')), config)
            else:
                # Handle single file or DataFrame
                raise NotImplementedError("Single data source batch processing not implemented yet")
        
        elif mode == "streaming":
            self.start_streaming(config.get('algorithm', 'iforest'))
            # Streaming processing would be handled by the caller using process_stream_sample
            return {'status': 'streaming_started', 'stats': self.get_streaming_stats()}
        
        else:
            raise ValueError(f"Unsupported processing mode: {mode}")

    # ==============================================================================
    # UTILITY METHODS
    # ==============================================================================
    
    async def _load_file_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load data from file."""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.csv':
                return pd.read_csv(file_path)
            elif suffix == '.json':
                return pd.read_json(file_path)
            elif suffix == '.parquet':
                return pd.read_parquet(file_path)
            elif suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                # Try CSV as default
                return pd.read_csv(file_path)
                
        except Exception as e:
            logger.error("Failed to load file data", file=str(file_path), error=str(e))
            return None
    
    async def _process_chunk(self, chunk: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single chunk of data."""
        algorithm = config.get('algorithm', 'iforest')
        
        # Convert to numpy array for detection
        data_array = chunk.select_dtypes(include=[np.number]).values
        
        if data_array.shape[0] == 0:
            return {
                'anomaly_indices': [],
                'anomaly_scores': [],
                'chunk_size': len(chunk)
            }
        
        # Perform detection
        result = self.detection_service.detect(data_array, algorithm=algorithm)
        
        return {
            'anomaly_indices': [i for i, is_anom in enumerate(result.is_anomaly) if is_anom],
            'anomaly_scores': result.anomaly_score,
            'chunk_size': len(chunk),
            'algorithm': result.algorithm
        }
    
    async def _combine_batch_results(self, results: List[Dict[str, Any]], original_data: pd.DataFrame) -> Dict[str, Any]:
        """Combine results from multiple chunks."""
        all_anomaly_indices = []
        all_anomaly_scores = []
        offset = 0
        
        for result in results:
            # Adjust indices based on chunk offset
            chunk_indices = [idx + offset for idx in result['anomaly_indices']]
            all_anomaly_indices.extend(chunk_indices)
            all_anomaly_scores.extend(result['anomaly_scores'])
            offset += result['chunk_size']
        
        return {
            'anomaly_indices': all_anomaly_indices,
            'anomaly_scores': all_anomaly_scores,
            'total_samples': len(original_data),
            'total_anomalies': len(all_anomaly_indices),
            'algorithm': results[0]['algorithm'] if results else 'unknown'
        }
    
    async def _save_batch_results(self, result: Dict[str, Any], output_dir: Path, input_file: Path) -> Path:
        """Save batch processing results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{input_file.stem}_anomalies.json"
        
        # Add metadata
        result['metadata'] = {
            'input_file': str(input_file),
            'processing_timestamp': datetime.utcnow().isoformat(),
            'service_version': '1.0.0'
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return output_file
    
    def _update_streaming_model(self) -> None:
        """Update the streaming model with current buffer data."""
        try:
            if len(self._data_buffer) < 10:
                return
            
            # Convert buffer to array
            data_array = np.array(list(self._data_buffer))
            
            # Update/retrain the model
            # Note: This would need to be implemented based on the specific algorithm
            # For now, we'll just mark as fitted
            self._model_fitted = True
            self._last_update = self._sample_count
            
            logger.debug("Streaming model updated",
                        samples_used=len(self._data_buffer),
                        total_samples=self._sample_count)
            
        except Exception as e:
            logger.error("Failed to update streaming model", error=str(e))
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # If psutil is not available, return 0
            return 0.0
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)