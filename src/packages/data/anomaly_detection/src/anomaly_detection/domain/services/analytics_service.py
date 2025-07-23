"""Analytics service for comprehensive dashboard insights."""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import structlog

from ..entities.detection_result import DetectionResult
from ...infrastructure.monitoring.metrics_collector import get_metrics_collector, MetricsCollector

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for the system."""
    total_detections: int = 0
    total_anomalies: int = 0
    average_detection_time: float = 0.0
    success_rate: float = 0.0
    throughput: float = 0.0  # detections per second
    error_rate: float = 0.0


@dataclass
class AlgorithmStats:
    """Statistics for a specific algorithm."""
    algorithm: str
    detections_count: int = 0
    anomalies_found: int = 0
    average_score: float = 0.0
    average_time: float = 0.0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    total_samples: int = 0
    missing_values: int = 0
    duplicate_samples: int = 0
    outliers_count: int = 0
    data_drift_events: int = 0
    quality_score: float = 0.0


class AnalyticsService:
    """Service providing comprehensive analytics for the dashboard."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or get_metrics_collector()
        
        # In-memory storage for demo (in production, use persistent storage)
        self.detection_history: deque = deque(maxlen=10000)
        self.algorithm_stats: Dict[str, AlgorithmStats] = defaultdict(AlgorithmStats)
        self.performance_history: deque = deque(maxlen=1000)
        self.data_quality_history: deque = deque(maxlen=1000)
        
        # System status tracking
        self.system_health = {
            'status': 'healthy',
            'uptime': datetime.now(),
            'last_check': datetime.now(),
            'issues': []
        }
        
        logger.info("Analytics service initialized")
    
    def record_detection(
        self,
        algorithm: str,
        result: DetectionResult,
        processing_time: float,
        data_quality: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a detection result for analytics."""
        timestamp = datetime.now()
        
        # Record detection history
        detection_record = {
            'timestamp': timestamp,
            'algorithm': algorithm,
            'total_samples': result.total_samples,
            'anomalies_found': result.anomaly_count,
            'anomaly_rate': result.anomaly_rate,
            'processing_time': processing_time,
            'success': result.success,
            'data_quality': data_quality or {}
        }
        
        self.detection_history.append(detection_record)
        
        # Update algorithm statistics
        if algorithm not in self.algorithm_stats:
            self.algorithm_stats[algorithm] = AlgorithmStats(algorithm=algorithm)
        
        algo_stats = self.algorithm_stats[algorithm]
        algo_stats.detections_count += 1
        algo_stats.anomalies_found += result.anomaly_count
        algo_stats.last_used = timestamp
        
        # Update averages
        if result.anomaly_scores is not None:
            current_avg = algo_stats.average_score
            new_score = np.mean(np.abs(result.anomaly_scores))
            algo_stats.average_score = (current_avg * (algo_stats.detections_count - 1) + new_score) / algo_stats.detections_count
        
        # Update timing
        current_avg_time = algo_stats.average_time
        algo_stats.average_time = (current_avg_time * (algo_stats.detections_count - 1) + processing_time) / algo_stats.detections_count
        
        # Update success rate
        if result.success:
            algo_stats.success_rate = (algo_stats.success_rate * (algo_stats.detections_count - 1) + 1.0) / algo_stats.detections_count
        else:
            algo_stats.success_rate = (algo_stats.success_rate * (algo_stats.detections_count - 1)) / algo_stats.detections_count
        
        logger.debug("Detection recorded for analytics", 
                    algorithm=algorithm,
                    anomalies=result.anomaly_count,
                    processing_time=processing_time)
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get comprehensive dashboard statistics."""
        current_time = datetime.now()
        
        # Calculate time-based metrics
        last_hour = current_time - timedelta(hours=1)
        last_day = current_time - timedelta(days=1)
        
        recent_detections = [d for d in self.detection_history if d['timestamp'] > last_hour]
        daily_detections = [d for d in self.detection_history if d['timestamp'] > last_day]
        
        # Performance metrics
        performance = PerformanceMetrics()
        performance.total_detections = len(self.detection_history)
        performance.total_anomalies = sum(d['anomalies_found'] for d in self.detection_history)
        
        if self.detection_history:
            performance.average_detection_time = np.mean([d['processing_time'] for d in self.detection_history])
            performance.success_rate = np.mean([d['success'] for d in self.detection_history])
            
            # Throughput (detections per hour)
            if len(recent_detections) > 0:
                performance.throughput = len(recent_detections)
            
            # Error rate
            failed_detections = sum(1 for d in self.detection_history if not d['success'])
            performance.error_rate = failed_detections / len(self.detection_history)
        
        # Algorithm performance
        algorithm_performance = {}
        for algo, stats in self.algorithm_stats.items():
            algorithm_performance[algo] = {
                'name': algo,
                'detections': stats.detections_count,
                'anomalies': stats.anomalies_found,
                'average_score': round(stats.average_score, 3),
                'average_time': round(stats.average_time, 3),
                'success_rate': round(stats.success_rate, 3),
                'last_used': stats.last_used.isoformat() if stats.last_used else None
            }
        
        # Time series data for charts
        chart_data = self._generate_chart_data()
        
        return {
            'performance': asdict(performance),
            'algorithms': algorithm_performance,
            'charts': chart_data,
            'system_health': self.system_health,
            'recent_activity': {
                'last_hour': len(recent_detections),
                'last_24h': len(daily_detections),
                'latest_detections': [
                    {
                        'algorithm': d['algorithm'],
                        'anomalies': d['anomalies_found'],
                        'timestamp': d['timestamp'].strftime('%H:%M:%S'),
                        'processing_time': round(d['processing_time'], 2)
                    }
                    for d in list(self.detection_history)[-10:]
                ]
            },
            'timestamp': current_time.isoformat()
        }
    
    def _generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for dashboard charts."""
        current_time = datetime.now()
        
        # Time series for last 24 hours (hourly buckets)
        hourly_data = defaultdict(lambda: {'detections': 0, 'anomalies': 0, 'avg_time': 0})
        
        for detection in self.detection_history:
            hour_key = detection['timestamp'].strftime('%H:00')
            hourly_data[hour_key]['detections'] += 1
            hourly_data[hour_key]['anomalies'] += detection['anomalies_found']
            hourly_data[hour_key]['avg_time'] += detection['processing_time']
        
        # Average the processing times
        for hour_data in hourly_data.values():
            if hour_data['detections'] > 0:
                hour_data['avg_time'] /= hour_data['detections']
        
        # Generate 24-hour timeline
        timeline_labels = []
        timeline_detections = []
        timeline_anomalies = []
        timeline_processing_times = []
        
        for i in range(24):
            hour = (current_time - timedelta(hours=23-i)).strftime('%H:00')
            timeline_labels.append(hour)
            
            data = hourly_data.get(hour, {'detections': 0, 'anomalies': 0, 'avg_time': 0})
            timeline_detections.append(data['detections'])
            timeline_anomalies.append(data['anomalies'])
            timeline_processing_times.append(round(data['avg_time'], 2))
        
        # Algorithm distribution
        algorithm_labels = []
        algorithm_counts = []
        algorithm_colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']
        
        for i, (algo, stats) in enumerate(self.algorithm_stats.items()):
            algorithm_labels.append(algo.replace('_', ' ').title())
            algorithm_counts.append(stats.detections_count)
        
        # Anomaly rate distribution
        if self.detection_history:
            anomaly_rates = [d['anomaly_rate'] for d in self.detection_history if d['anomaly_rate'] > 0]
            anomaly_rate_bins = np.histogram(anomaly_rates, bins=10) if anomaly_rates else ([], [])
            
            anomaly_distribution = {
                'bins': [f"{edge:.2f}" for edge in anomaly_rate_bins[1][:-1]],
                'counts': anomaly_rate_bins[0].tolist() if len(anomaly_rate_bins[0]) > 0 else []
            }
        else:
            anomaly_distribution = {'bins': [], 'counts': []}
        
        # Performance trend (last 50 detections)
        recent_performance = list(self.detection_history)[-50:] if len(self.detection_history) >= 50 else list(self.detection_history)
        performance_timeline = {
            'labels': [f"T-{len(recent_performance)-i}" for i in range(len(recent_performance))],
            'processing_times': [round(d['processing_time'], 2) for d in recent_performance],
            'anomaly_rates': [round(d['anomaly_rate'] * 100, 1) for d in recent_performance]
        }
        
        return {
            'timeline': {
                'labels': timeline_labels,
                'detections': timeline_detections,
                'anomalies': timeline_anomalies,
                'processing_times': timeline_processing_times
            },
            'algorithms': {
                'labels': algorithm_labels,
                'counts': algorithm_counts,
                'colors': algorithm_colors[:len(algorithm_labels)]
            },
            'anomaly_distribution': anomaly_distribution,
            'performance_trend': performance_timeline
        }
    
    def get_algorithm_comparison(self) -> Dict[str, Any]:
        """Get detailed algorithm comparison metrics."""
        comparison_data = []
        
        for algo, stats in self.algorithm_stats.items():
            if stats.detections_count > 0:
                comparison_data.append({
                    'algorithm': algo.replace('_', ' ').title(),
                    'detections': stats.detections_count,
                    'anomalies_found': stats.anomalies_found,
                    'average_anomaly_rate': round(stats.anomalies_found / stats.detections_count, 3) if stats.detections_count > 0 else 0,
                    'average_score': round(stats.average_score, 3),
                    'average_time': round(stats.average_time, 3),
                    'success_rate': round(stats.success_rate * 100, 1),
                    'efficiency_score': self._calculate_efficiency_score(stats),
                    'last_used': stats.last_used.strftime('%Y-%m-%d %H:%M') if stats.last_used else 'Never'
                })
        
        # Sort by efficiency score
        comparison_data.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        return {
            'algorithms': comparison_data,
            'total_algorithms': len(comparison_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_efficiency_score(self, stats: AlgorithmStats) -> float:
        """Calculate efficiency score for an algorithm."""
        if stats.detections_count == 0:
            return 0.0
        
        # Factors: success rate (40%), speed (30%), detection capability (30%)
        success_factor = stats.success_rate * 0.4
        
        # Speed factor (inverse of processing time, normalized)
        max_time = max(s.average_time for s in self.algorithm_stats.values()) if self.algorithm_stats else 1
        speed_factor = (1 - (stats.average_time / max_time)) * 0.3 if max_time > 0 else 0
        
        # Detection capability (anomaly finding rate)
        detection_factor = (stats.anomalies_found / stats.detections_count) * 0.3
        
        efficiency = success_factor + speed_factor + detection_factor
        return round(efficiency, 3)
    
    def get_data_insights(self) -> Dict[str, Any]:
        """Get data quality and processing insights."""
        if not self.detection_history:
            return {
                'total_samples_processed': 0,
                'data_quality_score': 0,
                'insights': [],
                'recommendations': []
            }
        
        # Calculate data insights
        total_samples = sum(d.get('total_samples', 0) for d in self.detection_history)
        total_anomalies = sum(d.get('anomalies_found', 0) for d in self.detection_history)
        
        # Quality insights
        insights = []
        recommendations = []
        
        # Anomaly rate analysis
        recent_detections = list(self.detection_history)[-20:] if len(self.detection_history) >= 20 else list(self.detection_history)
        if recent_detections:
            avg_anomaly_rate = np.mean([d['anomaly_rate'] for d in recent_detections])
            
            if avg_anomaly_rate > 0.2:
                insights.append(f"High anomaly rate detected: {avg_anomaly_rate:.1%}")
                recommendations.append("Consider reviewing data quality or adjusting contamination parameters")
            elif avg_anomaly_rate < 0.01:
                insights.append(f"Very low anomaly rate: {avg_anomaly_rate:.1%}")
                recommendations.append("Data might be too clean or detection parameters too strict")
        
        # Performance insights
        avg_processing_time = np.mean([d['processing_time'] for d in recent_detections])
        if avg_processing_time > 1.0:
            insights.append(f"Processing time is high: {avg_processing_time:.2f}s average")
            recommendations.append("Consider optimizing algorithms or reducing data dimensionality")
        
        # Algorithm usage insights
        algorithm_usage = defaultdict(int)
        for d in recent_detections:
            algorithm_usage[d['algorithm']] += 1
        
        if len(algorithm_usage) == 1:
            insights.append("Only one algorithm is being used")
            recommendations.append("Consider using ensemble methods for better detection accuracy")
        
        # Data quality score (simple heuristic)
        success_rate = np.mean([d['success'] for d in self.detection_history])
        processing_efficiency = min(1.0, 1.0 / max(avg_processing_time, 0.1))
        anomaly_consistency = 1.0 - np.std([d['anomaly_rate'] for d in recent_detections]) if len(recent_detections) > 1 else 1.0
        
        data_quality_score = (success_rate * 0.4 + processing_efficiency * 0.3 + anomaly_consistency * 0.3)
        
        return {
            'total_samples_processed': total_samples,
            'total_anomalies_found': total_anomalies,
            'overall_anomaly_rate': round(total_anomalies / max(total_samples, 1), 4),
            'data_quality_score': round(data_quality_score * 100, 1),
            'average_processing_time': round(avg_processing_time, 3),
            'success_rate': round(success_rate * 100, 1),
            'insights': insights,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for live dashboard updates."""
        current_time = datetime.now()
        last_minute = current_time - timedelta(minutes=1)
        last_5_minutes = current_time - timedelta(minutes=5)
        
        # Recent activity
        recent_1min = [d for d in self.detection_history if d['timestamp'] > last_minute]
        recent_5min = [d for d in self.detection_history if d['timestamp'] > last_5_minutes]
        
        # System metrics from metrics collector
        system_stats = self.metrics_collector.get_summary_stats()
        
        return {
            'current_time': current_time.isoformat(),
            'activity_1min': len(recent_1min),
            'activity_5min': len(recent_5min),
            'anomalies_1min': sum(d['anomalies_found'] for d in recent_1min),
            'anomalies_5min': sum(d['anomalies_found'] for d in recent_5min),
            'avg_processing_time_5min': round(np.mean([d['processing_time'] for d in recent_5min]), 3) if recent_5min else 0,
            'system_metrics': {
                'total_operations': system_stats.get('total_performance_metrics', 0),
                'success_rate': system_stats.get('recent_success_rate', 0),
                'active_operations': system_stats.get('active_operations', 0)
            },
            'system_health': self.system_health['status']
        }
    
    def update_system_health(self, status: str, issues: List[str] = None) -> None:
        """Update system health status.""" 
        self.system_health.update({
            'status': status,
            'last_check': datetime.now(),
            'issues': issues or []
        })
        
        logger.info("System health updated", status=status, issues=len(issues or []))
    
    def clear_history(self, older_than_hours: int = 24) -> int:
        """Clear old detection history."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        original_count = len(self.detection_history)
        self.detection_history = deque(
            (d for d in self.detection_history if d['timestamp'] > cutoff_time),
            maxlen=self.detection_history.maxlen
        )
        
        cleared_count = original_count - len(self.detection_history)
        logger.info("Analytics history cleared", cleared_entries=cleared_count)
        
        return cleared_count