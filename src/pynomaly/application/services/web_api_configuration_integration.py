"""Web API Configuration Integration Service.

This service provides integration between web API endpoints and the configuration
capture system, enabling automatic tracking of API-based anomaly detection workflows.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from pynomaly.application.dto.configuration_dto import (
    ConfigurationSource, ConfigurationSearchRequestDTO,
    WebAPIContextDTO, RequestConfigurationDTO, ResponseConfigurationDTO
)
from pynomaly.application.services.configuration_capture_service import ConfigurationCaptureService
from pynomaly.infrastructure.persistence.configuration_repository import ConfigurationRepository
from pynomaly.infrastructure.config.feature_flags import require_feature

logger = logging.getLogger(__name__)


class WebAPIConfigurationIntegration:
    """Service for integrating web API requests with configuration management."""
    
    def __init__(
        self,
        configuration_service: ConfigurationCaptureService,
        auto_analyze_patterns: bool = True,
        pattern_analysis_interval_hours: int = 24,
        performance_threshold_ms: float = 1000.0
    ):
        """Initialize web API configuration integration.
        
        Args:
            configuration_service: Configuration capture service
            auto_analyze_patterns: Automatically analyze API usage patterns
            pattern_analysis_interval_hours: How often to analyze patterns
            performance_threshold_ms: Threshold for flagging slow requests
        """
        self.configuration_service = configuration_service
        self.auto_analyze_patterns = auto_analyze_patterns
        self.pattern_analysis_interval_hours = pattern_analysis_interval_hours
        self.performance_threshold_ms = performance_threshold_ms
        
        # Integration statistics
        self.integration_stats = {
            "total_api_requests": 0,
            "configurations_captured": 0,
            "unique_endpoints": set(),
            "unique_clients": set(),
            "total_processing_time_ms": 0,
            "last_analysis_time": None,
            "error_count": 0
        }
        
        # Pattern analysis cache
        self._pattern_cache = {
            "endpoint_usage": defaultdict(int),
            "client_patterns": defaultdict(list),
            "performance_data": [],
            "error_patterns": defaultdict(list)
        }
        
        # Performance tracking
        self.performance_tracker = {
            "slow_requests": [],
            "fast_requests": [],
            "average_response_time": 0.0,
            "p95_response_time": 0.0,
            "p99_response_time": 0.0
        }
    
    @require_feature("advanced_automl")
    async def analyze_api_usage_patterns(
        self,
        time_period_days: int = 7
    ) -> Dict[str, Any]:
        """Analyze API usage patterns from captured configurations.
        
        Args:
            time_period_days: Time period for analysis
            
        Returns:
            Comprehensive API usage analysis
        """
        logger.info(f"Analyzing API usage patterns over {time_period_days} days")
        
        # Search for web API configurations
        search_request = ConfigurationSearchRequestDTO(
            source=ConfigurationSource.WEB_API,
            limit=10000,
            sort_by="created_at",
            sort_order="desc"
        )
        
        configurations = await self.configuration_service.repository.search_configurations(search_request)
        
        # Filter by time period
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_configs = [
            config for config in configurations
            if config.metadata.created_at >= cutoff_date
        ]
        
        logger.info(f"Found {len(recent_configs)} API configurations for analysis")
        
        # Analyze patterns
        analysis = {
            "time_period_days": time_period_days,
            "total_configurations": len(recent_configs),
            "endpoint_analysis": self._analyze_endpoints(recent_configs),
            "client_analysis": self._analyze_clients(recent_configs),
            "performance_analysis": self._analyze_performance(recent_configs),
            "temporal_analysis": self._analyze_temporal_patterns(recent_configs),
            "error_analysis": self._analyze_errors(recent_configs),
            "algorithm_usage": self._analyze_algorithm_usage(recent_configs),
            "recommendations": self._generate_api_recommendations(recent_configs)
        }
        
        # Update statistics
        self.integration_stats["last_analysis_time"] = datetime.now().isoformat()
        
        return analysis
    
    async def get_endpoint_performance_metrics(
        self,
        endpoint: str,
        time_period_days: int = 7
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific endpoint.
        
        Args:
            endpoint: API endpoint to analyze
            time_period_days: Time period for analysis
            
        Returns:
            Endpoint performance metrics
        """
        # Search for configurations from specific endpoint
        configurations = await self._get_endpoint_configurations(endpoint, time_period_days)
        
        if not configurations:
            return {
                "endpoint": endpoint,
                "error": "No configurations found for endpoint",
                "time_period_days": time_period_days
            }
        
        # Extract performance data
        response_times = []
        success_count = 0
        error_count = 0
        
        for config in configurations:
            context = config.source_context.get("web_api_context", {})
            if isinstance(context, dict) and "response_config" in context:
                response_config = context["response_config"]
                
                # Extract response time
                if "processing_time_ms" in response_config:
                    response_times.append(response_config["processing_time_ms"])
                
                # Count success/errors
                if response_config.get("status_code", 500) < 400:
                    success_count += 1
                else:
                    error_count += 1
        
        # Calculate metrics
        metrics = {
            "endpoint": endpoint,
            "time_period_days": time_period_days,
            "total_requests": len(configurations),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / len(configurations) if configurations else 0,
            "performance_metrics": self._calculate_performance_metrics(response_times)
        }
        
        return metrics
    
    async def track_api_configuration_quality(
        self,
        config_id: UUID
    ) -> Dict[str, Any]:
        """Track the quality and effectiveness of an API configuration.
        
        Args:
            config_id: Configuration ID to track
            
        Returns:
            Configuration quality metrics
        """
        # Load configuration
        config = await self.configuration_service.repository.load_configuration(config_id)
        
        if not config or config.metadata.source != ConfigurationSource.WEB_API:
            return {"error": "Configuration not found or not from web API"}
        
        # Extract web API context
        context = config.source_context.get("web_api_context", {})
        if not isinstance(context, dict):
            return {"error": "Invalid web API context"}
        
        # Analyze configuration quality
        quality_metrics = {
            "configuration_id": str(config_id),
            "endpoint": context.get("endpoint", "unknown"),
            "request_analysis": self._analyze_request_quality(context.get("request_config", {})),
            "response_analysis": self._analyze_response_quality(context.get("response_config", {})),
            "parameter_completeness": self._analyze_parameter_completeness(config.raw_parameters),
            "performance_score": self._calculate_performance_score(context.get("response_config", {})),
            "overall_quality_score": 0.0
        }
        
        # Calculate overall quality score
        quality_metrics["overall_quality_score"] = self._calculate_overall_quality(quality_metrics)
        
        return quality_metrics
    
    async def generate_api_configuration_report(
        self,
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive API configuration report.
        
        Args:
            time_period_days: Time period for report
            
        Returns:
            Comprehensive API configuration report
        """
        logger.info(f"Generating API configuration report for {time_period_days} days")
        
        # Get usage patterns
        usage_patterns = await self.analyze_api_usage_patterns(time_period_days)
        
        # Get top endpoints performance
        top_endpoints = usage_patterns["endpoint_analysis"]["top_endpoints"][:10]
        endpoint_performance = {}
        
        for endpoint_data in top_endpoints:
            endpoint = endpoint_data["endpoint"]
            performance = await self.get_endpoint_performance_metrics(endpoint, time_period_days)
            endpoint_performance[endpoint] = performance
        
        # Generate report
        report = {
            "report_generated": datetime.now().isoformat(),
            "time_period_days": time_period_days,
            "executive_summary": self._generate_executive_summary(usage_patterns),
            "usage_patterns": usage_patterns,
            "endpoint_performance": endpoint_performance,
            "recommendations": self._generate_comprehensive_recommendations(
                usage_patterns, endpoint_performance
            ),
            "configuration_trends": self._analyze_configuration_trends(usage_patterns),
            "quality_metrics": self._calculate_overall_api_quality(usage_patterns)
        }
        
        return report
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration service statistics.
        
        Returns:
            Integration statistics
        """
        # Convert sets to counts for JSON serialization
        stats = self.integration_stats.copy()
        stats["unique_endpoints"] = len(stats["unique_endpoints"])
        stats["unique_clients"] = len(stats["unique_clients"])
        
        return {
            "integration_stats": stats,
            "performance_tracker": self.performance_tracker,
            "configuration": {
                "auto_analyze_patterns": self.auto_analyze_patterns,
                "pattern_analysis_interval_hours": self.pattern_analysis_interval_hours,
                "performance_threshold_ms": self.performance_threshold_ms
            }
        }
    
    # Private methods
    
    async def _get_endpoint_configurations(
        self, 
        endpoint: str, 
        time_period_days: int
    ) -> List:
        """Get configurations for a specific endpoint."""
        search_request = ConfigurationSearchRequestDTO(
            source=ConfigurationSource.WEB_API,
            limit=1000,
            sort_by="created_at",
            sort_order="desc"
        )
        
        all_configs = await self.configuration_service.repository.search_configurations(search_request)
        
        # Filter by endpoint and time period
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        endpoint_configs = []
        
        for config in all_configs:
            if config.metadata.created_at < cutoff_date:
                continue
            
            context = config.source_context.get("web_api_context", {})
            if isinstance(context, dict) and context.get("endpoint") == endpoint:
                endpoint_configs.append(config)
        
        return endpoint_configs
    
    def _analyze_endpoints(self, configurations: List) -> Dict[str, Any]:
        """Analyze endpoint usage patterns."""
        endpoint_counts = Counter()
        endpoint_performance = defaultdict(list)
        endpoint_errors = defaultdict(int)
        
        for config in configurations:
            context = config.source_context.get("web_api_context", {})
            if isinstance(context, dict):
                endpoint = context.get("endpoint", "unknown")
                endpoint_counts[endpoint] += 1
                
                # Track performance
                response_config = context.get("response_config", {})
                if "processing_time_ms" in response_config:
                    endpoint_performance[endpoint].append(response_config["processing_time_ms"])
                
                # Track errors
                if response_config.get("status_code", 200) >= 400:
                    endpoint_errors[endpoint] += 1
        
        # Calculate endpoint metrics
        top_endpoints = []
        for endpoint, count in endpoint_counts.most_common(20):
            perf_times = endpoint_performance[endpoint]
            avg_time = sum(perf_times) / len(perf_times) if perf_times else 0
            error_rate = endpoint_errors[endpoint] / count if count > 0 else 0
            
            top_endpoints.append({
                "endpoint": endpoint,
                "request_count": count,
                "average_response_time_ms": avg_time,
                "error_rate": error_rate,
                "error_count": endpoint_errors[endpoint]
            })
        
        return {
            "total_unique_endpoints": len(endpoint_counts),
            "top_endpoints": top_endpoints,
            "endpoint_distribution": dict(endpoint_counts.most_common(10))
        }
    
    def _analyze_clients(self, configurations: List) -> Dict[str, Any]:
        """Analyze client usage patterns."""
        client_ips = Counter()
        user_agents = Counter()
        client_types = Counter()
        
        for config in configurations:
            context = config.source_context.get("web_api_context", {})
            if isinstance(context, dict):
                client_info = context.get("client_info", {})
                
                # Count client IPs
                if "ip" in client_info:
                    client_ips[client_info["ip"]] += 1
                
                # Count user agents
                request_config = context.get("request_config", {})
                if "user_agent" in request_config:
                    user_agents[request_config["user_agent"]] += 1
                
                # Count client types
                if "client_type" in client_info:
                    client_types[client_info["client_type"]] += 1
        
        return {
            "unique_client_ips": len(client_ips),
            "top_client_ips": dict(client_ips.most_common(10)),
            "client_type_distribution": dict(client_types),
            "top_user_agents": dict(user_agents.most_common(5))
        }
    
    def _analyze_performance(self, configurations: List) -> Dict[str, Any]:
        """Analyze API performance patterns."""
        response_times = []
        status_codes = Counter()
        
        for config in configurations:
            context = config.source_context.get("web_api_context", {})
            if isinstance(context, dict):
                response_config = context.get("response_config", {})
                
                # Collect response times
                if "processing_time_ms" in response_config:
                    response_times.append(response_config["processing_time_ms"])
                
                # Count status codes
                if "status_code" in response_config:
                    status_codes[response_config["status_code"]] += 1
        
        return {
            "response_time_metrics": self._calculate_performance_metrics(response_times),
            "status_code_distribution": dict(status_codes),
            "total_requests": len(configurations),
            "success_rate": sum(1 for code in status_codes if code < 400) / len(configurations) if configurations else 0
        }
    
    def _analyze_temporal_patterns(self, configurations: List) -> Dict[str, Any]:
        """Analyze temporal usage patterns."""
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for config in configurations:
            created_at = config.metadata.created_at
            hour = created_at.hour
            date = created_at.date()
            
            hourly_counts[hour] += 1
            daily_counts[str(date)] += 1
        
        # Find peak hours
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1]) if hourly_counts else (0, 0)
        
        return {
            "hourly_distribution": dict(hourly_counts),
            "daily_distribution": dict(sorted(daily_counts.items())),
            "peak_hour": {"hour": peak_hour[0], "requests": peak_hour[1]},
            "total_days": len(daily_counts)
        }
    
    def _analyze_errors(self, configurations: List) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_configs = []
        error_types = Counter()
        error_endpoints = Counter()
        
        for config in configurations:
            context = config.source_context.get("web_api_context", {})
            if isinstance(context, dict):
                response_config = context.get("response_config", {})
                status_code = response_config.get("status_code", 200)
                
                if status_code >= 400:
                    error_configs.append(config)
                    error_types[status_code] += 1
                    error_endpoints[context.get("endpoint", "unknown")] += 1
        
        return {
            "total_errors": len(error_configs),
            "error_rate": len(error_configs) / len(configurations) if configurations else 0,
            "error_type_distribution": dict(error_types),
            "top_error_endpoints": dict(error_endpoints.most_common(5))
        }
    
    def _analyze_algorithm_usage(self, configurations: List) -> Dict[str, Any]:
        """Analyze algorithm usage from API requests."""
        algorithm_counts = Counter()
        
        for config in configurations:
            # Extract algorithm from raw parameters
            raw_params = config.raw_parameters
            if isinstance(raw_params, dict):
                algorithm = raw_params.get("algorithm")
                if algorithm:
                    algorithm_counts[algorithm] += 1
        
        return {
            "total_algorithm_requests": sum(algorithm_counts.values()),
            "algorithm_distribution": dict(algorithm_counts),
            "top_algorithms": dict(algorithm_counts.most_common(10))
        }
    
    def _generate_api_recommendations(self, configurations: List) -> List[Dict[str, str]]:
        """Generate recommendations based on API usage analysis."""
        recommendations = []
        
        # Analyze response times
        response_times = []
        for config in configurations:
            context = config.source_context.get("web_api_context", {})
            if isinstance(context, dict):
                response_config = context.get("response_config", {})
                if "processing_time_ms" in response_config:
                    response_times.append(response_config["processing_time_ms"])
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            slow_requests = [t for t in response_times if t > self.performance_threshold_ms]
            
            if slow_requests:
                recommendations.append({
                    "type": "performance",
                    "title": "Optimize Slow Endpoints",
                    "description": f"{len(slow_requests)} requests exceeded {self.performance_threshold_ms}ms threshold"
                })
            
            if avg_time > 500:
                recommendations.append({
                    "type": "performance",
                    "title": "Overall Performance Improvement Needed",
                    "description": f"Average response time is {avg_time:.1f}ms, consider optimization"
                })
        
        # Error rate analysis
        error_count = 0
        for config in configurations:
            context = config.source_context.get("web_api_context", {})
            if isinstance(context, dict):
                response_config = context.get("response_config", {})
                if response_config.get("status_code", 200) >= 400:
                    error_count += 1
        
        if configurations and error_count / len(configurations) > 0.05:  # 5% error rate
            recommendations.append({
                "type": "reliability",
                "title": "High Error Rate Detected",
                "description": f"Error rate is {error_count / len(configurations):.1%}, investigate failing endpoints"
            })
        
        return recommendations
    
    def _calculate_performance_metrics(self, response_times: List[float]) -> Dict[str, float]:
        """Calculate performance metrics from response times."""
        if not response_times:
            return {}
        
        response_times.sort()
        n = len(response_times)
        
        return {
            "count": n,
            "min_ms": min(response_times),
            "max_ms": max(response_times),
            "mean_ms": sum(response_times) / n,
            "median_ms": response_times[n // 2],
            "p95_ms": response_times[int(n * 0.95)] if n > 0 else 0,
            "p99_ms": response_times[int(n * 0.99)] if n > 0 else 0
        }
    
    def _analyze_request_quality(self, request_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of request configuration."""
        quality_score = 0.0
        issues = []
        
        # Check for required fields
        if request_config.get("method"):
            quality_score += 20
        else:
            issues.append("Missing HTTP method")
        
        if request_config.get("path"):
            quality_score += 20
        else:
            issues.append("Missing request path")
        
        # Check for parameters
        query_params = request_config.get("query_parameters", {})
        if query_params:
            quality_score += 20
        
        body = request_config.get("body")
        if body:
            quality_score += 20
        
        # Check for headers
        headers = request_config.get("headers", {})
        if headers:
            quality_score += 20
        
        return {
            "quality_score": quality_score,
            "issues": issues,
            "completeness": quality_score / 100.0
        }
    
    def _analyze_response_quality(self, response_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of response configuration."""
        quality_score = 0.0
        issues = []
        
        # Check status code
        status_code = response_config.get("status_code")
        if status_code:
            quality_score += 30
            if status_code < 400:
                quality_score += 20
        else:
            issues.append("Missing status code")
        
        # Check processing time
        if "processing_time_ms" in response_config:
            quality_score += 25
            processing_time = response_config["processing_time_ms"]
            if processing_time < self.performance_threshold_ms:
                quality_score += 15
        else:
            issues.append("Missing processing time")
        
        # Check headers
        if response_config.get("headers"):
            quality_score += 10
        
        return {
            "quality_score": quality_score,
            "issues": issues,
            "completeness": quality_score / 100.0
        }
    
    def _analyze_parameter_completeness(self, raw_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze completeness of captured parameters."""
        completeness_score = 0.0
        
        if not raw_parameters:
            return {"completeness_score": 0.0, "parameter_count": 0}
        
        # Check for algorithm parameters
        if "algorithm" in raw_parameters:
            completeness_score += 40
        
        # Check for anomaly detection specific parameters
        anomaly_params = ["contamination", "threshold", "outlier", "anomaly", "score"]
        for param in anomaly_params:
            if any(param in str(key).lower() for key in raw_parameters.keys()):
                completeness_score += 15
                break
        
        # Check for data parameters
        data_params = ["dataset", "features", "data", "samples"]
        for param in data_params:
            if any(param in str(key).lower() for key in raw_parameters.keys()):
                completeness_score += 15
                break
        
        # Check for configuration parameters
        config_params = ["model", "training", "cross_validation", "cv"]
        for param in config_params:
            if any(param in str(key).lower() for key in raw_parameters.keys()):
                completeness_score += 15
                break
        
        return {
            "completeness_score": min(completeness_score, 100.0),
            "parameter_count": len(raw_parameters)
        }
    
    def _calculate_performance_score(self, response_config: Dict[str, Any]) -> float:
        """Calculate performance score from response data."""
        processing_time = response_config.get("processing_time_ms", float('inf'))
        status_code = response_config.get("status_code", 500)
        
        # Base score for successful response
        if status_code < 400:
            base_score = 50.0
        else:
            base_score = 0.0
        
        # Performance bonus
        if processing_time < 100:
            perf_score = 50.0
        elif processing_time < 500:
            perf_score = 30.0
        elif processing_time < 1000:
            perf_score = 15.0
        else:
            perf_score = 0.0
        
        return base_score + perf_score
    
    def _calculate_overall_quality(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        request_quality = quality_metrics.get("request_analysis", {}).get("quality_score", 0)
        response_quality = quality_metrics.get("response_analysis", {}).get("quality_score", 0)
        param_completeness = quality_metrics.get("parameter_completeness", {}).get("completeness_score", 0)
        performance_score = quality_metrics.get("performance_score", 0)
        
        # Weighted average
        overall_score = (
            request_quality * 0.25 +
            response_quality * 0.25 +
            param_completeness * 0.25 +
            performance_score * 0.25
        )
        
        return overall_score
    
    def _generate_executive_summary(self, usage_patterns: Dict[str, Any]) -> Dict[str, str]:
        """Generate executive summary of API usage."""
        total_requests = usage_patterns.get("total_configurations", 0)
        unique_endpoints = usage_patterns.get("endpoint_analysis", {}).get("total_unique_endpoints", 0)
        avg_response_time = usage_patterns.get("performance_analysis", {}).get("response_time_metrics", {}).get("mean_ms", 0)
        success_rate = usage_patterns.get("performance_analysis", {}).get("success_rate", 0)
        
        return {
            "total_api_requests": f"{total_requests:,}",
            "unique_endpoints": str(unique_endpoints),
            "average_response_time": f"{avg_response_time:.1f}ms",
            "success_rate": f"{success_rate:.1%}",
            "status": "healthy" if success_rate > 0.95 else "needs_attention"
        }
    
    def _generate_comprehensive_recommendations(
        self, 
        usage_patterns: Dict[str, Any], 
        endpoint_performance: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Add usage pattern recommendations
        pattern_recs = usage_patterns.get("recommendations", [])
        recommendations.extend(pattern_recs)
        
        # Add endpoint-specific recommendations
        for endpoint, perf_data in endpoint_performance.items():
            metrics = perf_data.get("performance_metrics", {})
            if metrics.get("p95_ms", 0) > 1000:
                recommendations.append({
                    "type": "endpoint_optimization",
                    "title": f"Optimize {endpoint}",
                    "description": f"95th percentile response time is {metrics['p95_ms']:.1f}ms"
                })
        
        return recommendations
    
    def _analyze_configuration_trends(self, usage_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze configuration trends."""
        daily_distribution = usage_patterns.get("temporal_analysis", {}).get("daily_distribution", {})
        
        # Calculate trend direction
        if len(daily_distribution) >= 2:
            dates = sorted(daily_distribution.keys())
            first_half = dates[:len(dates)//2]
            second_half = dates[len(dates)//2:]
            
            first_half_avg = sum(daily_distribution[date] for date in first_half) / len(first_half)
            second_half_avg = sum(daily_distribution[date] for date in second_half) / len(second_half)
            
            if second_half_avg > first_half_avg * 1.1:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "usage_trend": trend,
            "daily_distribution": daily_distribution
        }
    
    def _calculate_overall_api_quality(self, usage_patterns: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall API quality metrics."""
        performance_metrics = usage_patterns.get("performance_analysis", {})
        success_rate = performance_metrics.get("success_rate", 0)
        avg_response_time = performance_metrics.get("response_time_metrics", {}).get("mean_ms", 0)
        
        # Quality score calculation
        success_score = success_rate * 50  # 0-50 points
        
        # Performance score (0-50 points)
        if avg_response_time < 100:
            perf_score = 50
        elif avg_response_time < 500:
            perf_score = 30
        elif avg_response_time < 1000:
            perf_score = 15
        else:
            perf_score = 0
        
        overall_quality = success_score + perf_score
        
        return {
            "success_score": success_score,
            "performance_score": perf_score,
            "overall_quality": overall_quality,
            "quality_grade": self._get_quality_grade(overall_quality)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"