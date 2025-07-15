"""Batch Configuration Manager.

This module provides intelligent batch configuration management with auto-optimization
based on data characteristics, system resources, and processing patterns.
"""

from __future__ import annotations

import logging
import math
import psutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field

from ...infrastructure.config.settings import Settings
from .batch_processing_service import BatchConfig

logger = logging.getLogger(__name__)


class SystemResources(BaseModel):
    """Current system resource information."""
    
    cpu_count: int
    cpu_percent: float
    memory_total_gb: float
    memory_available_gb: float
    memory_percent: float
    
    @classmethod
    def get_current(cls) -> SystemResources:
        """Get current system resources."""
        memory = psutil.virtual_memory()
        return cls(
            cpu_count=psutil.cpu_count(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_total_gb=memory.total / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            memory_percent=memory.percent
        )


class DataCharacteristics(BaseModel):
    """Characteristics of the data to be processed."""
    
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    data_types: Dict[str, str] = Field(default_factory=dict)
    has_numeric_data: bool = False
    has_text_data: bool = False
    has_categorical_data: bool = False
    average_row_size_bytes: float = 0.0
    complexity_score: float = 0.0
    
    @classmethod
    def analyze_dataframe(cls, df: pd.DataFrame) -> DataCharacteristics:
        """Analyze a pandas DataFrame to extract characteristics."""
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        
        # Analyze data types
        data_types = {}
        has_numeric = False
        has_text = False
        has_categorical = False
        
        for col, dtype in df.dtypes.items():
            dtype_str = str(dtype)
            data_types[col] = dtype_str
            
            if dtype_str in ['int64', 'int32', 'float64', 'float32']:
                has_numeric = True
            elif dtype_str in ['object', 'string']:
                has_text = True
            elif dtype_str == 'category':
                has_categorical = True
        
        # Calculate complexity score
        complexity_score = cls._calculate_complexity_score(df)
        
        return cls(
            total_rows=len(df),
            total_columns=len(df.columns),
            memory_usage_mb=memory_usage,
            data_types=data_types,
            has_numeric_data=has_numeric,
            has_text_data=has_text,
            has_categorical_data=has_categorical,
            average_row_size_bytes=memory_usage * 1024**2 / len(df) if len(df) > 0 else 0,
            complexity_score=complexity_score
        )
    
    @staticmethod
    def _calculate_complexity_score(df: pd.DataFrame) -> float:
        """Calculate a complexity score for the data (0-1, higher = more complex)."""
        factors = []
        
        # Column count factor
        factors.append(min(len(df.columns) / 100, 1.0))
        
        # Data type diversity factor
        unique_types = len(set(str(dtype) for dtype in df.dtypes))
        factors.append(min(unique_types / 10, 1.0))
        
        # Text data factor (more complex)
        text_columns = sum(1 for dtype in df.dtypes if str(dtype) in ['object', 'string'])
        factors.append(min(text_columns / len(df.columns), 1.0))
        
        # Missing data factor
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        factors.append(missing_ratio)
        
        return sum(factors) / len(factors)


class ProcessingProfile(BaseModel):
    """Profile of processing characteristics and requirements."""
    
    processor_name: str
    cpu_intensive: bool = False
    memory_intensive: bool = False
    io_intensive: bool = False
    network_intensive: bool = False
    requires_order: bool = False
    supports_parallel: bool = True
    estimated_processing_time_per_row_ms: float = 1.0
    memory_overhead_factor: float = 1.5
    
    @classmethod
    def get_default_profiles(cls) -> Dict[str, ProcessingProfile]:
        """Get default processing profiles for common operations."""
        return {
            'anomaly_detection': cls(
                processor_name='anomaly_detection',
                cpu_intensive=True,
                memory_intensive=True,
                estimated_processing_time_per_row_ms=2.0,
                memory_overhead_factor=2.0
            ),
            'data_quality': cls(
                processor_name='data_quality',
                cpu_intensive=False,
                memory_intensive=False,
                estimated_processing_time_per_row_ms=0.5,
                memory_overhead_factor=1.2
            ),
            'data_profiling': cls(
                processor_name='data_profiling',
                cpu_intensive=True,
                memory_intensive=False,
                estimated_processing_time_per_row_ms=1.5,
                memory_overhead_factor=1.3
            ),
            'feature_engineering': cls(
                processor_name='feature_engineering',
                cpu_intensive=True,
                memory_intensive=True,
                estimated_processing_time_per_row_ms=3.0,
                memory_overhead_factor=2.5
            ),
            'model_training': cls(
                processor_name='model_training',
                cpu_intensive=True,
                memory_intensive=True,
                requires_order=True,
                estimated_processing_time_per_row_ms=10.0,
                memory_overhead_factor=3.0
            ),
            'data_export': cls(
                processor_name='data_export',
                io_intensive=True,
                network_intensive=True,
                estimated_processing_time_per_row_ms=0.8,
                memory_overhead_factor=1.1
            )
        }


class BatchOptimizationResult(BaseModel):
    """Result of batch optimization calculation."""
    
    recommended_batch_size: int
    recommended_concurrency: int
    estimated_memory_usage_mb: float
    estimated_processing_time_seconds: float
    optimization_factors: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0


class BatchConfigurationManager:
    """Manages and optimizes batch processing configuration."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the configuration manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        
        # Load default processing profiles
        self.processing_profiles = ProcessingProfile.get_default_profiles()
        
        # Configuration limits
        self.min_batch_size = 10
        self.max_batch_size = 100000
        self.min_concurrency = 1
        self.max_concurrency = min(32, psutil.cpu_count() * 2)
        
        # Memory safety margins
        self.memory_safety_margin = 0.8  # Use max 80% of available memory
        self.memory_overhead_buffer = 0.2  # 20% buffer for overhead
    
    def register_processing_profile(self, profile: ProcessingProfile) -> None:
        """Register a custom processing profile.
        
        Args:
            profile: Processing profile to register
        """
        self.processing_profiles[profile.processor_name] = profile
        self.logger.info(f"Registered processing profile: {profile.processor_name}")
    
    def calculate_optimal_batch_config(self,
                                     data: Any,
                                     processor_name: str,
                                     target_memory_usage_mb: Optional[float] = None,
                                     target_processing_time_seconds: Optional[float] = None,
                                     max_concurrency: Optional[int] = None) -> BatchOptimizationResult:
        """Calculate optimal batch configuration for given data and processing requirements.
        
        Args:
            data: Input data to analyze
            processor_name: Name of the processor
            target_memory_usage_mb: Target memory usage limit
            target_processing_time_seconds: Target processing time limit
            max_concurrency: Maximum concurrency override
            
        Returns:
            Optimization result with recommendations
        """
        self.logger.info(f"Calculating optimal batch config for processor: {processor_name}")
        
        # Get system resources
        system_resources = SystemResources.get_current()
        
        # Analyze data characteristics
        data_characteristics = self._analyze_data(data)
        
        # Get processing profile
        processing_profile = self.processing_profiles.get(processor_name)
        if not processing_profile:
            self.logger.warning(f"No profile found for processor {processor_name}, using default")
            processing_profile = ProcessingProfile(processor_name=processor_name)
        
        # Calculate optimal configuration
        result = self._optimize_batch_configuration(
            data_characteristics=data_characteristics,
            processing_profile=processing_profile,
            system_resources=system_resources,
            target_memory_usage_mb=target_memory_usage_mb,
            target_processing_time_seconds=target_processing_time_seconds,
            max_concurrency=max_concurrency
        )
        
        self.logger.info(
            f"Recommended config: batch_size={result.recommended_batch_size}, "
            f"concurrency={result.recommended_concurrency}, "
            f"estimated_memory={result.estimated_memory_usage_mb:.1f}MB"
        )
        
        return result
    
    def _analyze_data(self, data: Any) -> DataCharacteristics:
        """Analyze data to extract characteristics."""
        if isinstance(data, pd.DataFrame):
            return DataCharacteristics.analyze_dataframe(data)
        elif isinstance(data, list):
            # Convert to DataFrame for analysis if possible
            try:
                df = pd.DataFrame(data)
                return DataCharacteristics.analyze_dataframe(df)
            except Exception:
                # Fallback for non-tabular data
                return DataCharacteristics(
                    total_rows=len(data),
                    total_columns=1,
                    memory_usage_mb=len(data) * 8 / (1024**2),  # Rough estimate
                    average_row_size_bytes=8,  # Rough estimate
                    complexity_score=0.3
                )
        elif hasattr(data, '__len__'):
            return DataCharacteristics(
                total_rows=len(data),
                total_columns=1,
                memory_usage_mb=len(data) * 8 / (1024**2),
                average_row_size_bytes=8,
                complexity_score=0.3
            )
        else:
            # Single item or unknown data type
            return DataCharacteristics(
                total_rows=1,
                total_columns=1,
                memory_usage_mb=0.001,
                average_row_size_bytes=8,
                complexity_score=0.1
            )
    
    def _optimize_batch_configuration(self,
                                    data_characteristics: DataCharacteristics,
                                    processing_profile: ProcessingProfile,
                                    system_resources: SystemResources,
                                    target_memory_usage_mb: Optional[float],
                                    target_processing_time_seconds: Optional[float],
                                    max_concurrency: Optional[int]) -> BatchOptimizationResult:
        """Optimize batch configuration based on all factors."""
        
        optimization_factors = {}
        warnings = []
        
        # 1. Memory-based batch size calculation
        available_memory_mb = system_resources.memory_available_gb * 1024 * self.memory_safety_margin
        target_memory = target_memory_usage_mb or available_memory_mb
        
        # Account for processing overhead
        effective_memory = target_memory / processing_profile.memory_overhead_factor
        
        # Calculate batch size based on memory
        if data_characteristics.average_row_size_bytes > 0:
            memory_based_batch_size = int(
                effective_memory * 1024**2 / data_characteristics.average_row_size_bytes
            )
        else:
            memory_based_batch_size = 1000  # Default fallback
        
        optimization_factors['memory_based_batch_size'] = memory_based_batch_size
        
        # 2. Performance-based batch size calculation
        if target_processing_time_seconds:
            # Calculate batch size that would complete within target time
            time_per_row_ms = processing_profile.estimated_processing_time_per_row_ms
            performance_based_batch_size = int(
                target_processing_time_seconds * 1000 / time_per_row_ms
            )
        else:
            # Aim for reasonable batch processing time (30 seconds per batch)
            performance_based_batch_size = int(30000 / processing_profile.estimated_processing_time_per_row_ms)
        
        optimization_factors['performance_based_batch_size'] = performance_based_batch_size
        
        # 3. System resource-based batch size
        cpu_factor = 1.0
        if processing_profile.cpu_intensive:
            cpu_factor = max(0.5, 1.0 - system_resources.cpu_percent / 100)
        
        memory_factor = 1.0
        if processing_profile.memory_intensive:
            memory_factor = max(0.3, 1.0 - system_resources.memory_percent / 100)
        
        resource_adjustment = cpu_factor * memory_factor
        resource_based_batch_size = int(memory_based_batch_size * resource_adjustment)
        
        optimization_factors['resource_adjustment'] = resource_adjustment
        optimization_factors['resource_based_batch_size'] = resource_based_batch_size
        
        # 4. Data complexity adjustment
        complexity_factor = max(0.3, 1.0 - data_characteristics.complexity_score * 0.5)
        complexity_adjusted_batch_size = int(resource_based_batch_size * complexity_factor)
        
        optimization_factors['complexity_factor'] = complexity_factor
        optimization_factors['complexity_adjusted_batch_size'] = complexity_adjusted_batch_size
        
        # 5. Final batch size selection
        batch_size_candidates = [
            memory_based_batch_size,
            performance_based_batch_size,
            resource_based_batch_size,
            complexity_adjusted_batch_size
        ]
        
        # Use the most conservative (smallest) batch size for safety
        recommended_batch_size = min(batch_size_candidates)
        
        # Apply limits
        recommended_batch_size = max(self.min_batch_size, 
                                   min(self.max_batch_size, recommended_batch_size))
        
        # Ensure we don't exceed total data size
        if data_characteristics.total_rows > 0:
            recommended_batch_size = min(recommended_batch_size, data_characteristics.total_rows)
        
        # 6. Concurrency calculation
        if processing_profile.requires_order:
            recommended_concurrency = 1
            warnings.append("Sequential processing required - concurrency limited to 1")
        else:
            # Base concurrency on CPU cores and processing characteristics
            base_concurrency = system_resources.cpu_count
            
            if processing_profile.cpu_intensive:
                recommended_concurrency = max(1, base_concurrency)
            elif processing_profile.io_intensive:
                recommended_concurrency = min(base_concurrency * 2, 16)
            else:
                recommended_concurrency = max(2, base_concurrency // 2)
            
            # Apply limits
            max_conc = max_concurrency or self.max_concurrency
            recommended_concurrency = max(self.min_concurrency, 
                                        min(max_conc, recommended_concurrency))
        
        # 7. Estimate final resource usage
        estimated_memory_usage = (
            recommended_batch_size * data_characteristics.average_row_size_bytes * 
            recommended_concurrency * processing_profile.memory_overhead_factor / (1024**2)
        )
        
        estimated_processing_time = (
            data_characteristics.total_rows * processing_profile.estimated_processing_time_per_row_ms / 
            (1000 * recommended_concurrency)
        )
        
        # 8. Generate warnings
        if estimated_memory_usage > available_memory_mb:
            warnings.append(f"Estimated memory usage ({estimated_memory_usage:.1f}MB) exceeds available memory")
        
        if recommended_batch_size == self.min_batch_size:
            warnings.append("Batch size limited to minimum value - consider reducing concurrency")
        
        if recommended_batch_size == self.max_batch_size:
            warnings.append("Batch size at maximum limit - performance may not be optimal")
        
        # 9. Calculate confidence score
        confidence_factors = []
        
        # Higher confidence if we have good data characteristics
        if data_characteristics.memory_usage_mb > 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Higher confidence if system has available resources
        if system_resources.memory_percent < 70 and system_resources.cpu_percent < 70:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Higher confidence if processing profile is well-defined
        if processor_name in self.processing_profiles:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)
        
        confidence_score = sum(confidence_factors) / len(confidence_factors)
        
        return BatchOptimizationResult(
            recommended_batch_size=recommended_batch_size,
            recommended_concurrency=recommended_concurrency,
            estimated_memory_usage_mb=estimated_memory_usage,
            estimated_processing_time_seconds=estimated_processing_time,
            optimization_factors=optimization_factors,
            warnings=warnings,
            confidence_score=confidence_score
        )
    
    def create_optimized_config(self,
                              data: Any,
                              processor_name: str,
                              **kwargs) -> BatchConfig:
        """Create an optimized BatchConfig for the given data and processor.
        
        Args:
            data: Input data
            processor_name: Processor name
            **kwargs: Additional configuration overrides
            
        Returns:
            Optimized BatchConfig
        """
        optimization_result = self.calculate_optimal_batch_config(data, processor_name, **kwargs)
        
        config = BatchConfig(
            batch_size=optimization_result.recommended_batch_size,
            max_concurrent_batches=optimization_result.recommended_concurrency,
            memory_limit_mb=optimization_result.estimated_memory_usage_mb * 1.2,  # Add buffer
            **kwargs
        )
        
        if optimization_result.warnings:
            self.logger.warning(f"Configuration warnings: {optimization_result.warnings}")
        
        return config
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Get system-wide batch processing recommendations.
        
        Returns:
            Dictionary of recommendations and system status
        """
        system_resources = SystemResources.get_current()
        
        recommendations = {
            'system_status': {
                'cpu_usage': system_resources.cpu_percent,
                'memory_usage': system_resources.memory_percent,
                'available_memory_gb': system_resources.memory_available_gb,
                'cpu_cores': system_resources.cpu_count
            },
            'recommendations': []
        }
        
        # CPU recommendations
        if system_resources.cpu_percent > 80:
            recommendations['recommendations'].append({
                'type': 'warning',
                'message': 'High CPU usage detected - reduce batch concurrency for better performance'
            })
        elif system_resources.cpu_percent < 30:
            recommendations['recommendations'].append({
                'type': 'info',
                'message': 'Low CPU usage - consider increasing batch concurrency'
            })
        
        # Memory recommendations
        if system_resources.memory_percent > 85:
            recommendations['recommendations'].append({
                'type': 'warning',
                'message': 'High memory usage detected - reduce batch sizes to prevent OOM errors'
            })
        elif system_resources.memory_percent < 40:
            recommendations['recommendations'].append({
                'type': 'info',
                'message': 'Ample memory available - can use larger batch sizes for better efficiency'
            })
        
        # General recommendations
        if system_resources.memory_available_gb < 1:
            recommendations['recommendations'].append({
                'type': 'critical',
                'message': 'Very low available memory - batch processing may fail or be very slow'
            })
        
        return recommendations