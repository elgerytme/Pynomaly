"""Real-time streaming profiling service for continuous data analysis."""

import asyncio
import time
import threading
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, AsyncIterator, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from queue import Queue, Empty
import logging
import json
from collections import defaultdict, deque

from ...domain.entities.data_profile import (
    DataProfile, ProfileId, DatasetId, ProfilingStatus,
    ColumnProfile, SchemaProfile, QualityAssessment, ProfilingMetadata,
    StatisticalSummary, Pattern, QualityIssue
)
from ...infrastructure.exceptions import DataProfilingError, ResourceLimitError
from ...infrastructure.logging.profiling_logger import get_logger, ProfilingContext
from .schema_analysis_service import SchemaAnalysisService
from .statistical_profiling_service import StatisticalProfilingService
from .pattern_discovery_service import PatternDiscoveryService
from .quality_assessment_service import QualityAssessmentService

logger = get_logger(__name__)


class StreamingMode(str, Enum):
    """Streaming processing modes."""
    REAL_TIME = "real_time"
    MICRO_BATCH = "micro_batch"
    SLIDING_WINDOW = "sliding_window"
    TUMBLING_WINDOW = "tumbling_window"


class UpdateStrategy(str, Enum):
    """Profile update strategies."""
    INCREMENTAL = "incremental"
    ROLLING_AVERAGE = "rolling_average"
    EXPONENTIAL_DECAY = "exponential_decay"
    FULL_RECALCULATION = "full_recalculation"


@dataclass
class StreamingConfig:
    """Configuration for streaming profiling."""
    mode: StreamingMode = StreamingMode.MICRO_BATCH
    batch_size: int = 1000
    window_size: int = 10000
    update_interval_seconds: float = 30.0
    update_strategy: UpdateStrategy = UpdateStrategy.INCREMENTAL
    memory_limit_mb: float = 500.0
    max_age_hours: float = 24.0
    quality_threshold: float = 0.8
    enable_anomaly_detection: bool = True
    persist_intermediate_results: bool = True


@dataclass
class StreamingMetrics:
    """Metrics for streaming profiling performance."""
    records_processed: int = 0
    batches_processed: int = 0
    processing_rate_per_second: float = 0.0
    average_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    last_update_time: Optional[datetime] = None
    errors_count: int = 0
    quality_alerts: int = 0


@dataclass
class WindowedStatistics:
    """Windowed statistics for streaming data."""
    window_id: str
    start_time: datetime
    end_time: datetime
    record_count: int
    column_stats: Dict[str, Any] = field(default_factory=dict)
    patterns: Dict[str, List[Pattern]] = field(default_factory=dict)
    quality_issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class IncrementalStatisticsAccumulator:
    """Accumulator for incremental statistical calculations."""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'sum_sq': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'null_count': 0,
            'unique_values': set(),
            'pattern_matches': defaultdict(int),
            'quality_issues': []
        })
        self.total_records = 0
        self.first_seen = datetime.utcnow()
        self.last_updated = datetime.utcnow()
    
    def update(self, data_batch: pd.DataFrame) -> None:
        """Update statistics with new data batch."""
        self.total_records += len(data_batch)
        self.last_updated = datetime.utcnow()
        
        for column in data_batch.columns:
            series = data_batch[column]
            column_stats = self.stats[column]
            
            # Update basic counts
            non_null_series = series.dropna()
            column_stats['count'] += len(non_null_series)
            column_stats['null_count'] += series.isnull().sum()
            
            if len(non_null_series) > 0:
                # Update unique values (with memory limit)
                new_unique = set(non_null_series.unique())
                column_stats['unique_values'].update(new_unique)
                
                # Limit unique values to prevent memory explosion
                if len(column_stats['unique_values']) > 10000:
                    # Keep a sample of unique values
                    column_stats['unique_values'] = set(
                        list(column_stats['unique_values'])[:5000]
                    )
                
                # Update numeric statistics
                if pd.api.types.is_numeric_dtype(series):
                    column_stats['sum'] += non_null_series.sum()
                    column_stats['sum_sq'] += (non_null_series ** 2).sum()
                    column_stats['min'] = min(column_stats['min'], non_null_series.min())
                    column_stats['max'] = max(column_stats['max'], non_null_series.max())
    
    def get_current_statistics(self) -> Dict[str, Any]:
        """Get current accumulated statistics."""
        result = {}
        
        for column, stats in self.stats.items():
            if stats['count'] > 0:
                # Calculate derived statistics
                mean = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
                variance = (stats['sum_sq'] / stats['count'] - mean ** 2) if stats['count'] > 0 else 0
                std_dev = np.sqrt(max(0, variance))
                
                result[column] = {
                    'count': stats['count'],
                    'mean': mean,
                    'std_dev': std_dev,
                    'min': stats['min'] if stats['min'] != float('inf') else None,
                    'max': stats['max'] if stats['max'] != float('-inf') else None,
                    'null_count': stats['null_count'],
                    'unique_count': len(stats['unique_values']),
                    'completeness_ratio': stats['count'] / self.total_records if self.total_records > 0 else 0,
                    'sample_unique_values': list(stats['unique_values'])[:100]  # Sample for display
                }
        
        return result
    
    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self.stats.clear()
        self.stats = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'sum_sq': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'null_count': 0,
            'unique_values': set(),
            'pattern_matches': defaultdict(int),
            'quality_issues': []
        })
        self.total_records = 0
        self.first_seen = datetime.utcnow()


class StreamingPatternDetector:
    """Real-time pattern detection for streaming data."""
    
    def __init__(self, pattern_service: PatternDiscoveryService):
        self.pattern_service = pattern_service
        self.known_patterns = defaultdict(dict)  # column -> {pattern_regex: count}
        self.pattern_confidence_threshold = 0.7
    
    def analyze_batch_patterns(self, data_batch: pd.DataFrame) -> Dict[str, List[Pattern]]:
        """Analyze patterns in a data batch."""
        batch_patterns = {}
        
        for column in data_batch.select_dtypes(include=['object']).columns:
            series = data_batch[column].dropna()
            if len(series) == 0:
                continue
            
            # Quick pattern detection for streaming
            column_patterns = self._detect_streaming_patterns(series, column)
            if column_patterns:
                batch_patterns[column] = column_patterns
        
        return batch_patterns
    
    def _detect_streaming_patterns(self, series: pd.Series, column_name: str) -> List[Pattern]:
        """Detect patterns optimized for streaming processing."""
        patterns = []
        unique_values = series.unique()
        
        if len(unique_values) > 100:  # Too many unique values, sample
            unique_values = np.random.choice(unique_values, 100, replace=False)
        
        # Quick email detection
        email_matches = sum(1 for val in unique_values 
                          if re.match(r'^[^@]+@[^@]+\.[^@]+$', str(val)))
        if email_matches > len(unique_values) * 0.3:
            patterns.append(Pattern(
                pattern_type='email',
                regex=r'^[^@]+@[^@]+\.[^@]+$',
                frequency=email_matches,
                percentage=(email_matches / len(unique_values)) * 100,
                examples=unique_values[:3].tolist(),
                confidence=email_matches / len(unique_values)
            ))
        
        # Quick phone detection
        phone_matches = sum(1 for val in unique_values 
                          if re.match(r'^[\+]?[\d\s\-\(\)]{10,}$', str(val)))
        if phone_matches > len(unique_values) * 0.3:
            patterns.append(Pattern(
                pattern_type='phone',
                regex=r'^[\+]?[\d\s\-\(\)]{10,}$',
                frequency=phone_matches,
                percentage=(phone_matches / len(unique_values)) * 100,
                examples=unique_values[:3].tolist(),
                confidence=phone_matches / len(unique_values)
            ))
        
        return patterns


class StreamingQualityMonitor:
    """Real-time quality monitoring for streaming data."""
    
    def __init__(self, quality_service: QualityAssessmentService, config: StreamingConfig):
        self.quality_service = quality_service
        self.config = config
        self.quality_history = deque(maxlen=100)  # Keep last 100 quality measurements
        self.alert_callbacks: List[Callable] = []
    
    def assess_batch_quality(self, data_batch: pd.DataFrame, 
                           column_profiles: List[ColumnProfile]) -> Dict[str, Any]:
        """Assess quality of a data batch."""
        quality_metrics = {}
        alerts = []
        
        for column in data_batch.columns:
            series = data_batch[column]
            
            # Quick quality checks
            null_percentage = (series.isnull().sum() / len(series)) * 100
            
            quality_metrics[column] = {
                'completeness': 100 - null_percentage,
                'null_percentage': null_percentage,
                'record_count': len(series),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Check for quality alerts
            if null_percentage > 50:  # High missing rate
                alerts.append({
                    'type': 'high_missing_data',
                    'column': column,
                    'severity': 'high',
                    'value': null_percentage,
                    'message': f'High missing data rate: {null_percentage:.1f}%'
                })
        
        # Calculate overall quality score
        overall_completeness = np.mean([m['completeness'] for m in quality_metrics.values()])
        
        quality_result = {
            'overall_score': overall_completeness / 100,
            'column_metrics': quality_metrics,
            'alerts': alerts,
            'timestamp': datetime.utcnow(),
            'batch_size': len(data_batch)
        }
        
        # Store in history
        self.quality_history.append(quality_result)
        
        # Trigger alerts if needed
        if alerts:
            self._trigger_quality_alerts(alerts)
        
        return quality_result
    
    def add_alert_callback(self, callback: Callable[[List[Dict]], None]) -> None:
        """Add callback for quality alerts."""
        self.alert_callbacks.append(callback)
    
    def _trigger_quality_alerts(self, alerts: List[Dict]) -> None:
        """Trigger quality alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alerts)
            except Exception as e:
                logger.error(f"Error in quality alert callback: {e}")
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Get quality trends from history."""
        if not self.quality_history:
            return {}
        
        # Calculate trends
        scores = [q['overall_score'] for q in self.quality_history]
        timestamps = [q['timestamp'] for q in self.quality_history]
        
        return {
            'current_score': scores[-1] if scores else 0,
            'average_score': np.mean(scores) if scores else 0,
            'trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'stable',
            'measurements_count': len(scores),
            'last_update': timestamps[-1] if timestamps else None
        }


class StreamingProfilingService:
    """Main service for real-time streaming data profiling."""
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self.is_running = False
        self.current_profile: Optional[DataProfile] = None
        
        # Core services
        self.schema_service = SchemaAnalysisService()
        self.stats_service = StatisticalProfilingService()
        self.pattern_service = PatternDiscoveryService()
        self.quality_service = QualityAssessmentService()
        
        # Streaming-specific components
        self.stats_accumulator = IncrementalStatisticsAccumulator()
        self.pattern_detector = StreamingPatternDetector(self.pattern_service)
        self.quality_monitor = StreamingQualityMonitor(self.quality_service, self.config)
        
        # Processing state
        self.data_queue: Queue = Queue(maxsize=1000)
        self.metrics = StreamingMetrics()
        self.windows: Dict[str, WindowedStatistics] = {}
        self.worker_thread: Optional[threading.Thread] = None
        
        # Event handlers
        self.update_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
    
    def start_streaming(self, dataset_id: str, source_config: Dict[str, Any]) -> DataProfile:
        """Start streaming profiling process."""
        if self.is_running:
            raise DataProfilingError("Streaming profiling is already running")
        
        with ProfilingContext("start_streaming_profiling"):
            # Initialize profile
            self.current_profile = DataProfile(
                profile_id=ProfileId(),
                dataset_id=DatasetId(value=dataset_id),
                status=ProfilingStatus.RUNNING,
                source_type="stream",
                source_connection=source_config,
                created_at=datetime.utcnow()
            )
            
            # Start processing
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.worker_thread.start()
            
            logger.info(f"Started streaming profiling for dataset {dataset_id}")
            return self.current_profile
    
    def stop_streaming(self) -> Optional[DataProfile]:
        """Stop streaming profiling and return final profile."""
        if not self.is_running:
            return self.current_profile
        
        with ProfilingContext("stop_streaming_profiling"):
            self.is_running = False
            
            # Wait for processing to complete
            if self.worker_thread:
                self.worker_thread.join(timeout=30)
            
            # Generate final profile
            if self.current_profile:
                final_profile = self._generate_final_profile()
                logger.info("Stopped streaming profiling")
                return final_profile
            
            return None
    
    def add_data_batch(self, data_batch: pd.DataFrame) -> None:
        """Add a new data batch for processing."""
        if not self.is_running:
            raise DataProfilingError("Streaming profiling is not running")
        
        try:
            batch_info = {
                'data': data_batch,
                'timestamp': datetime.utcnow(),
                'batch_id': f"batch_{int(time.time() * 1000)}"
            }
            self.data_queue.put(batch_info, timeout=5)
        except Exception as e:
            logger.error(f"Failed to add data batch: {e}")
            raise DataProfilingError(f"Failed to add data batch: {e}")
    
    def get_current_profile(self) -> Optional[DataProfile]:
        """Get current streaming profile snapshot."""
        if not self.current_profile or not self.is_running:
            return self.current_profile
        
        # Generate current snapshot
        return self._generate_current_snapshot()
    
    def get_streaming_metrics(self) -> StreamingMetrics:
        """Get current streaming metrics."""
        return self.metrics
    
    def add_update_callback(self, callback: Callable[[DataProfile], None]) -> None:
        """Add callback for profile updates."""
        self.update_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add callback for processing errors."""
        self.error_callbacks.append(callback)
    
    def _processing_loop(self) -> None:
        """Main processing loop for streaming data."""
        last_update_time = time.time()
        
        while self.is_running:
            try:
                # Get batch from queue
                try:
                    batch_info = self.data_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                start_time = time.time()
                
                # Process batch
                self._process_data_batch(batch_info)
                
                # Update metrics
                processing_time = time.time() - start_time
                self.metrics.records_processed += len(batch_info['data'])
                self.metrics.batches_processed += 1
                self.metrics.average_latency_ms = processing_time * 1000
                self.metrics.processing_rate_per_second = len(batch_info['data']) / processing_time if processing_time > 0 else 0
                self.metrics.last_update_time = datetime.utcnow()
                
                # Check if update is needed
                current_time = time.time()
                if current_time - last_update_time >= self.config.update_interval_seconds:
                    self._trigger_profile_update()
                    last_update_time = current_time
                
                self.data_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.metrics.errors_count += 1
                self._trigger_error_callbacks(e)
    
    def _process_data_batch(self, batch_info: Dict[str, Any]) -> None:
        """Process a single data batch."""
        data_batch = batch_info['data']
        timestamp = batch_info['timestamp']
        
        # Update statistics
        self.stats_accumulator.update(data_batch)
        
        # Detect patterns
        batch_patterns = self.pattern_detector.analyze_batch_patterns(data_batch)
        
        # Assess quality
        quality_metrics = self.quality_monitor.assess_batch_quality(data_batch, [])
        
        # Handle windowing if configured
        if self.config.mode in [StreamingMode.SLIDING_WINDOW, StreamingMode.TUMBLING_WINDOW]:
            self._update_windows(data_batch, timestamp, batch_patterns, quality_metrics)
    
    def _update_windows(self, data_batch: pd.DataFrame, timestamp: datetime,
                       patterns: Dict[str, List[Pattern]], quality_metrics: Dict[str, Any]) -> None:
        """Update windowed statistics."""
        window_id = self._get_window_id(timestamp)
        
        if window_id not in self.windows:
            self.windows[window_id] = WindowedStatistics(
                window_id=window_id,
                start_time=timestamp,
                end_time=timestamp,
                record_count=0
            )
        
        window = self.windows[window_id]
        window.record_count += len(data_batch)
        window.end_time = timestamp
        window.patterns.update(patterns)
        
        # Cleanup old windows
        self._cleanup_old_windows()
    
    def _get_window_id(self, timestamp: datetime) -> str:
        """Generate window ID based on timestamp and window size."""
        if self.config.mode == StreamingMode.TUMBLING_WINDOW:
            # Fixed time windows
            window_start = timestamp.replace(second=0, microsecond=0)
            return f"window_{window_start.isoformat()}"
        else:
            # Sliding windows
            return f"window_{int(timestamp.timestamp())}"
    
    def _cleanup_old_windows(self) -> None:
        """Remove windows older than max_age_hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.max_age_hours)
        
        old_windows = [
            window_id for window_id, window in self.windows.items()
            if window.end_time < cutoff_time
        ]
        
        for window_id in old_windows:
            del self.windows[window_id]
    
    def _generate_current_snapshot(self) -> DataProfile:
        """Generate current profile snapshot."""
        current_stats = self.stats_accumulator.get_current_statistics()
        
        # Create column profiles
        columns = []
        for column_name, stats in current_stats.items():
            # Create basic column profile
            column_profile = ColumnProfile(
                column_name=column_name,
                data_type=self._infer_column_type(stats),
                inferred_type=None,
                nullable=stats['null_count'] > 0,
                distribution=self._create_value_distribution(stats),
                cardinality=self._determine_cardinality(stats),
                statistical_summary=self._create_statistical_summary(stats),
                patterns=[],
                quality_score=self._calculate_quality_score(stats),
                quality_issues=[],
                semantic_type=None,
                business_meaning=None
            )
            columns.append(column_profile)
        
        # Create schema profile
        schema_profile = SchemaProfile(
            table_name="streaming_data",
            total_columns=len(columns),
            total_rows=self.stats_accumulator.total_records,
            columns=columns,
            primary_keys=[],
            foreign_keys={},
            unique_constraints=[],
            check_constraints=[],
            estimated_size_bytes=None,
            compression_ratio=None
        )
        
        # Create quality assessment
        quality_trends = self.quality_monitor.get_quality_trends()
        quality_assessment = QualityAssessment(
            overall_score=quality_trends.get('current_score', 0.0),
            completeness_score=quality_trends.get('average_score', 0.0),
            consistency_score=0.8,  # Placeholder
            accuracy_score=0.8,     # Placeholder
            validity_score=0.8,     # Placeholder
            uniqueness_score=0.8,   # Placeholder
            dimension_weights={},
            critical_issues=0,
            high_issues=0,
            medium_issues=0,
            low_issues=0,
            recommendations=[]
        )
        
        # Create metadata
        metadata = ProfilingMetadata(
            profiling_strategy="streaming",
            sample_size=None,
            sample_percentage=None,
            execution_time_seconds=(datetime.utcnow() - self.stats_accumulator.first_seen).total_seconds(),
            memory_usage_mb=self.metrics.memory_usage_mb,
            include_patterns=True,
            include_statistical_analysis=True,
            include_quality_assessment=True
        )
        
        # Update current profile
        if self.current_profile:
            self.current_profile.schema_profile = schema_profile
            self.current_profile.quality_assessment = quality_assessment
            self.current_profile.profiling_metadata = metadata
            self.current_profile.updated_at = datetime.utcnow()
        
        return self.current_profile
    
    def _generate_final_profile(self) -> DataProfile:
        """Generate final profile when streaming stops."""
        final_profile = self._generate_current_snapshot()
        
        if final_profile:
            final_profile.status = ProfilingStatus.COMPLETED
            final_profile.completed_at = datetime.utcnow()
        
        return final_profile
    
    def _trigger_profile_update(self) -> None:
        """Trigger profile update callbacks."""
        try:
            current_profile = self._generate_current_snapshot()
            for callback in self.update_callbacks:
                callback(current_profile)
        except Exception as e:
            logger.error(f"Error in profile update callback: {e}")
    
    def _trigger_error_callbacks(self, error: Exception) -> None:
        """Trigger error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    # Helper methods for profile generation
    def _infer_column_type(self, stats: Dict[str, Any]) -> str:
        """Infer column data type from statistics."""
        if 'mean' in stats and stats['mean'] is not None:
            return 'numeric'
        return 'string'
    
    def _create_value_distribution(self, stats: Dict[str, Any]):
        """Create value distribution from statistics."""
        from ...domain.entities.data_profile import ValueDistribution
        return ValueDistribution(
            unique_count=stats.get('unique_count', 0),
            null_count=stats.get('null_count', 0),
            total_count=stats.get('count', 0) + stats.get('null_count', 0),
            completeness_ratio=stats.get('completeness_ratio', 0.0),
            top_values={}
        )
    
    def _determine_cardinality(self, stats: Dict[str, Any]):
        """Determine cardinality level."""
        from ...domain.entities.data_profile import CardinalityLevel
        unique_count = stats.get('unique_count', 0)
        if unique_count < 10:
            return CardinalityLevel.LOW
        elif unique_count < 100:
            return CardinalityLevel.MEDIUM
        elif unique_count < 1000:
            return CardinalityLevel.HIGH
        else:
            return CardinalityLevel.VERY_HIGH
    
    def _create_statistical_summary(self, stats: Dict[str, Any]):
        """Create statistical summary if numeric data."""
        if 'mean' not in stats:
            return None
        
        return StatisticalSummary(
            min_value=stats.get('min'),
            max_value=stats.get('max'),
            mean=stats.get('mean'),
            median=None,  # Not calculated in streaming
            std_dev=stats.get('std_dev'),
            quartiles=None  # Not calculated in streaming
        )
    
    def _calculate_quality_score(self, stats: Dict[str, Any]) -> float:
        """Calculate basic quality score."""
        completeness = stats.get('completeness_ratio', 0.0)
        return completeness  # Simplified for streaming