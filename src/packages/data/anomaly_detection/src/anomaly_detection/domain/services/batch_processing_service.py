"""Batch processing service for large-scale anomaly detection."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog

from .detection_service import DetectionService
from ..entities.dataset import Dataset, DatasetType, DatasetMetadata
from ...infrastructure.repositories.model_repository import ModelRepository

logger = structlog.get_logger(__name__)


class BatchProcessingService:
    """Service for batch processing of anomaly detection tasks."""
    
    def __init__(
        self,
        detection_service: DetectionService,
        model_repository: Optional[ModelRepository] = None,
        parallel_jobs: int = 4,
        chunk_size: int = 1000
    ):
        self.detection_service = detection_service
        self.model_repository = model_repository
        self.parallel_jobs = parallel_jobs
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=parallel_jobs)
        
    async def process_file(
        self,
        input_file: Path,
        output_dir: Path,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Process a single file for anomaly detection."""
        start_time = time.time()
        
        try:
            logger.info("Starting file processing", file=str(input_file))
            
            # Load data
            if progress_callback:
                progress_callback(10)
            
            data = await self._load_data(input_file)
            total_samples = len(data)
            
            logger.info("Data loaded", samples=total_samples, columns=data.shape[1])
            
            # Process in chunks if file is large
            if total_samples > self.chunk_size:
                result = await self._process_large_file(
                    data, input_file, output_dir, config, progress_callback
                )
            else:
                result = await self._process_small_file(
                    data, input_file, output_dir, config, progress_callback
                )
            
            processing_time = time.time() - start_time
            
            # Add metadata to result
            result.update({
                'input_file': str(input_file),
                'processing_time': processing_time,
                'total_samples': total_samples,
                'timestamp': datetime.utcnow().isoformat(),
                'config': config
            })
            
            logger.info("File processing completed", 
                       file=str(input_file),
                       processing_time=processing_time,
                       anomalies=result.get('anomaly_count', 0))
            
            return result
            
        except Exception as e:
            logger.error("File processing failed", file=str(input_file), error=str(e))
            raise
    
    async def _load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from file."""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        elif file_extension == '.parquet':
            return pd.read_parquet(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    async def _process_small_file(
        self,
        data: pd.DataFrame,
        input_file: Path,
        output_dir: Path,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Process a small file that fits in memory."""
        
        if progress_callback:
            progress_callback(30)
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found for anomaly detection")
        
        algorithms = config.get('algorithms', ['isolation_forest'])
        contamination = config.get('contamination', 0.1)
        
        results = {}
        anomaly_counts = {}
        
        # Run detection for each algorithm
        for i, algorithm in enumerate(algorithms):
            if progress_callback:
                progress_callback(30 + (50 * (i + 1) / len(algorithms)))
            
            try:
                detection_result = self.detection_service.detect_anomalies(
                    data=numeric_data.values,
                    algorithm=algorithm,
                    contamination=contamination
                )
                
                results[algorithm] = {
                    'predictions': detection_result.predictions.tolist(),
                    'confidence_scores': detection_result.confidence_scores.tolist() if detection_result.confidence_scores is not None else None,
                    'anomaly_count': detection_result.anomaly_count,
                    'anomaly_rate': detection_result.anomaly_rate,
                    'success': detection_result.success
                }
                
                anomaly_counts[algorithm] = detection_result.anomaly_count
                
                # Save model if requested
                if config.get('save_models', False) and self.model_repository:
                    await self._save_model(algorithm, input_file.stem)
                
            except Exception as e:
                logger.error("Algorithm failed", algorithm=algorithm, error=str(e))
                results[algorithm] = {
                    'error': str(e),
                    'success': False
                }
        
        # Save results
        if progress_callback:
            progress_callback(90)
        
        await self._save_results(results, data, input_file, output_dir, config)
        
        if progress_callback:
            progress_callback(100)
        
        return {
            'results': results,
            'anomaly_count': sum(anomaly_counts.values()),
            'algorithms_used': algorithms,
            'success': True
        }
    
    async def _process_large_file(
        self,
        data: pd.DataFrame,
        input_file: Path,
        output_dir: Path,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Process a large file using chunking."""
        
        total_samples = len(data)
        num_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        logger.info("Processing large file in chunks", 
                   total_samples=total_samples,
                   chunk_size=self.chunk_size,
                   num_chunks=num_chunks)
        
        algorithms = config.get('algorithms', ['isolation_forest'])
        contamination = config.get('contamination', 0.1)
        
        all_results = {algorithm: [] for algorithm in algorithms}
        total_anomalies = 0
        
        # Process chunks
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, total_samples)
            
            chunk_data = data.iloc[start_idx:end_idx]
            numeric_chunk = chunk_data.select_dtypes(include=[np.number])
            
            if numeric_chunk.empty:
                continue
            
            # Process chunk with each algorithm
            for algorithm in algorithms:
                try:
                    detection_result = self.detection_service.detect_anomalies(
                        data=numeric_chunk.values,
                        algorithm=algorithm,
                        contamination=contamination
                    )
                    
                    all_results[algorithm].extend(detection_result.predictions.tolist())
                    total_anomalies += detection_result.anomaly_count
                    
                except Exception as e:
                    logger.error("Chunk processing failed", 
                               chunk=chunk_idx,
                               algorithm=algorithm,
                               error=str(e))
                    # Fill with normal predictions for failed chunks
                    chunk_size_actual = len(numeric_chunk)
                    all_results[algorithm].extend([1] * chunk_size_actual)
            
            # Update progress
            if progress_callback:
                progress = 30 + (60 * (chunk_idx + 1) / num_chunks)
                progress_callback(progress)
        
        # Compile final results
        results = {}
        for algorithm in algorithms:
            if all_results[algorithm]:
                predictions = all_results[algorithm]
                anomaly_count = sum(1 for p in predictions if p == -1)
                
                results[algorithm] = {
                    'predictions': predictions,
                    'confidence_scores': None,  # Not available for chunked processing
                    'anomaly_count': anomaly_count,
                    'anomaly_rate': anomaly_count / len(predictions) if predictions else 0,
                    'success': True
                }
            else:
                results[algorithm] = {
                    'error': 'No valid chunks processed',
                    'success': False
                }
        
        # Save results
        if progress_callback:
            progress_callback(90)
        
        await self._save_results(results, data, input_file, output_dir, config)
        
        if progress_callback:
            progress_callback(100)
        
        return {
            'results': results,
            'anomaly_count': total_anomalies,
            'algorithms_used': algorithms,
            'chunks_processed': num_chunks,
            'success': True
        }
    
    async def _save_model(self, algorithm: str, file_stem: str) -> None:
        """Save trained model to repository."""
        if not self.model_repository:
            return
        
        try:
            from ..entities.model import Model, ModelMetadata, ModelStatus
            
            # Get fitted model from detection service
            fitted_model = self.detection_service._fitted_models.get(algorithm)
            if not fitted_model:
                logger.warning("No fitted model found for algorithm", algorithm=algorithm)
                return
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=f"batch_{file_stem}_{algorithm}_{int(time.time())}",
                name=f"Batch Model - {file_stem} ({algorithm})",
                algorithm=algorithm,
                status=ModelStatus.TRAINED,
                description=f"Batch trained model for {file_stem} using {algorithm}",
                created_at=datetime.utcnow()
            )
            
            # Create and save model
            model = Model(metadata=metadata, model_object=fitted_model)
            model_id = self.model_repository.save(model)
            
            logger.info("Model saved", model_id=model_id, algorithm=algorithm)
            
        except Exception as e:
            logger.error("Failed to save model", algorithm=algorithm, error=str(e))
    
    async def _save_results(
        self,
        results: Dict[str, Any],
        original_data: pd.DataFrame,
        input_file: Path,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> None:
        """Save detection results to files."""
        
        output_format = config.get('output_format', 'json')
        base_name = input_file.stem
        
        if output_format == 'json':
            # Save JSON results
            output_file = output_dir / f"{base_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif output_format == 'csv':
            # Save CSV with anomaly flags
            output_data = original_data.copy()
            
            for algorithm, result in results.items():
                if result.get('success', False):
                    predictions = result.get('predictions', [])
                    if len(predictions) == len(output_data):
                        output_data[f'{algorithm}_anomaly'] = [p == -1 for p in predictions]
                        
                        confidence_scores = result.get('confidence_scores')
                        if confidence_scores:
                            output_data[f'{algorithm}_confidence'] = confidence_scores
            
            output_file = output_dir / f"{base_name}_results.csv"
            output_data.to_csv(output_file, index=False)
        
        elif output_format == 'parquet':
            # Save Parquet with anomaly flags
            output_data = original_data.copy()
            
            for algorithm, result in results.items():
                if result.get('success', False):
                    predictions = result.get('predictions', [])
                    if len(predictions) == len(output_data):
                        output_data[f'{algorithm}_anomaly'] = [p == -1 for p in predictions]
                        
                        confidence_scores = result.get('confidence_scores')
                        if confidence_scores:
                            output_data[f'{algorithm}_confidence'] = confidence_scores
            
            output_file = output_dir / f"{base_name}_results.parquet"
            output_data.to_parquet(output_file, index=False)
        
        logger.info("Results saved", output_file=str(output_file), format=output_format)
    
    async def generate_summary_report(
        self,
        results: List[Dict[str, Any]],
        failed_files: List[tuple],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate a summary report for batch processing."""
        
        report_data = {
            'batch_summary': {
                'total_files': len(results) + len(failed_files),
                'successful_files': len(results),
                'failed_files': len(failed_files),
                'total_samples': sum(r.get('total_samples', 0) for r in results),
                'total_anomalies': sum(r.get('anomaly_count', 0) for r in results),
                'total_processing_time': sum(r.get('processing_time', 0) for r in results),
                'avg_processing_time': sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0,
                'timestamp': datetime.utcnow().isoformat()
            },
            'file_results': results,
            'failed_files': [{'file': str(f[0]), 'error': f[1]} for f in failed_files]
        }
        
        # Algorithm statistics
        algorithm_stats = {}
        for result in results:
            for algorithm in result.get('algorithms_used', []):
                if algorithm not in algorithm_stats:
                    algorithm_stats[algorithm] = {
                        'files_processed': 0,
                        'total_anomalies': 0,
                        'avg_anomaly_rate': 0
                    }
                
                algorithm_stats[algorithm]['files_processed'] += 1
                
                # Get anomaly count for this algorithm
                algo_result = result.get('results', {}).get(algorithm, {})
                if algo_result.get('success', False):
                    algorithm_stats[algorithm]['total_anomalies'] += algo_result.get('anomaly_count', 0)
        
        # Calculate average anomaly rates
        for algorithm, stats in algorithm_stats.items():
            if stats['files_processed'] > 0:
                total_samples = sum(
                    r.get('total_samples', 0) for r in results 
                    if algorithm in r.get('algorithms_used', [])
                )
                if total_samples > 0:
                    stats['avg_anomaly_rate'] = stats['total_anomalies'] / total_samples
        
        report_data['algorithm_statistics'] = algorithm_stats
        
        # Save report
        report_file = output_dir / f"batch_summary_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info("Summary report generated", report_file=str(report_file))
        
        return {
            'total_anomalies': report_data['batch_summary']['total_anomalies'],
            'avg_processing_time': report_data['batch_summary']['avg_processing_time'],
            'report_path': str(report_file),
            'algorithm_stats': algorithm_stats
        }
    
    async def process_parallel(
        self,
        input_files: List[Path],
        output_dir: Path,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process multiple files in parallel."""
        
        logger.info("Starting parallel batch processing", 
                   files=len(input_files),
                   parallel_jobs=self.parallel_jobs)
        
        # Create tasks for parallel processing
        tasks = []
        for file_path in input_files:
            task = asyncio.create_task(
                self.process_file(file_path, output_dir, config)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error("Task failed", error=str(e))
                results.append({
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    async def batch_detect_anomalies(
        self,
        file_paths: List[Path],
        output_dir: Path,
        algorithms: List[str] = None,
        output_format: str = "json",
        save_models: bool = False,
        model_name_prefix: Optional[str] = None,
        chunk_size: int = 1000,
        parallel_jobs: int = 4,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Run batch anomaly detection on multiple files."""
        
        if algorithms is None:
            algorithms = ["isolation_forest"]
        
        # Map algorithm names to service names
        algorithm_mapping = {
            "isolation_forest": "iforest",
            "local_outlier_factor": "lof",
            "one_class_svm": "one_class_svm",
            "elliptic_envelope": "elliptic_envelope"
        }
        
        # Create config for processing
        config = {
            'algorithms': [algorithm_mapping.get(alg, alg) for alg in algorithms],
            'contamination': contamination,
            'output_format': output_format,
            'save_models': save_models,
            'model_name_prefix': model_name_prefix
        }
        
        logger.info("Starting batch anomaly detection",
                   files=len(file_paths),
                   algorithms=config['algorithms'],
                   output_format=output_format)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files
        results = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                result = await self.process_file(file_path, output_dir, config)
                results.append(result)
            except Exception as e:
                logger.error("File processing failed", file=str(file_path), error=str(e))
                failed_files.append((file_path, str(e)))
        
        # Generate summary report
        summary = await self.generate_summary_report(results, failed_files, output_dir)
        
        return {
            'summary': summary,
            'successful_files': len(results),
            'failed_files': len(failed_files),
            'total_files': len(file_paths),
            'output_directory': str(output_dir),
            'algorithms_used': algorithms,
            'results': results,
            'failed_files': [{'file': str(f[0]), 'error': f[1]} for f in failed_files]
        }
    
    def __del__(self):
        """Cleanup executor on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)