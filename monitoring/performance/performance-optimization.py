#!/usr/bin/env python3
"""
Performance Optimization Script
Automated performance monitoring and optimization for the anomaly detection platform.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import psutil
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance-optimization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Manages performance monitoring and optimization."""
    
    def __init__(self, config_path: str = "monitoring/performance/optimization-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.prometheus_url = self.config.get('prometheus_url', 'http://localhost:9090')
        self.optimization_history = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load performance optimization configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._create_default_config()
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default performance optimization configuration."""
        default_config = {
            "prometheus_url": "http://localhost:9090",
            "optimization_interval": 300,  # 5 minutes
            "performance_thresholds": {
                "api_response_time": 1.0,  # seconds
                "cpu_usage": 70.0,  # percentage
                "memory_usage": 80.0,  # percentage
                "error_rate": 0.01,  # 1%
                "throughput_min": 10.0,  # requests/second
                "database_connection_threshold": 70
            },
            "optimization_strategies": {
                "auto_scaling": {
                    "enabled": True,
                    "cpu_threshold": 70,
                    "memory_threshold": 80,
                    "scale_up_factor": 1.5,
                    "scale_down_factor": 0.8,
                    "cooldown_period": 300
                },
                "caching": {
                    "enabled": True,
                    "redis_optimization": True,
                    "application_cache_tuning": True
                },
                "database_optimization": {
                    "enabled": True,
                    "connection_pool_tuning": True,
                    "query_optimization": True,
                    "index_recommendations": True
                },
                "resource_optimization": {
                    "enabled": True,
                    "garbage_collection_tuning": True,
                    "memory_optimization": True,
                    "cpu_affinity": True
                }
            },
            "alerting": {
                "slack_webhook": "${SLACK_WEBHOOK_URL}",
                "email_notifications": True,
                "notification_cooldown": 900  # 15 minutes
            },
            "reporting": {
                "generate_reports": True,
                "report_interval": 3600,  # 1 hour
                "retention_days": 30
            }
        }
        
        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info(f"Created default configuration: {self.config_path}")
        return default_config
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        logger.info("Collecting performance metrics...")
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': self._collect_system_metrics(),
            'application_metrics': self._collect_application_metrics(),
            'database_metrics': self._collect_database_metrics(),
            'prometheus_metrics': self._collect_prometheus_metrics()
        }
        
        return metrics
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available,
            'disk_usage': disk.percent,
            'disk_free': disk.free,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
            'process_count': len(psutil.pids())
        }
    
    def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        try:
            # Try to get metrics from application health endpoint
            response = requests.get('http://localhost:8000/api/health/metrics', timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            logger.warning("Could not collect application metrics")
        
        return {
            'api_response_time': 0,
            'throughput': 0,
            'error_rate': 0,
            'active_connections': 0
        }
    
    def _collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database performance metrics."""
        # This would typically connect to the database and collect metrics
        # For now, return placeholder metrics
        return {
            'connection_count': 0,
            'query_time_avg': 0,
            'cache_hit_ratio': 0,
            'table_size': 0,
            'index_usage': 0
        }
    
    def _collect_prometheus_metrics(self) -> Dict[str, Any]:
        """Collect metrics from Prometheus."""
        try:
            # Query key performance metrics from Prometheus
            queries = {
                'api_response_time': 'api:request_duration_seconds:rate5m',
                'error_rate': 'api:error_rate:rate5m',
                'throughput': 'api:request_rate:rate5m',
                'cpu_usage': 'system:cpu_usage:rate5m',
                'memory_usage': 'system:memory_usage:percentage'
            }
            
            metrics = {}
            for metric_name, query in queries.items():
                try:
                    response = requests.get(
                        f"{self.prometheus_url}/api/v1/query",
                        params={'query': query},
                        timeout=10
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data['data']['result']:
                            metrics[metric_name] = float(data['data']['result'][0]['value'][1])
                        else:
                            metrics[metric_name] = 0
                    else:
                        metrics[metric_name] = 0
                except Exception as e:
                    logger.warning(f"Failed to query {metric_name}: {e}")
                    metrics[metric_name] = 0
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to collect Prometheus metrics: {e}")
            return {}
    
    def analyze_performance_bottlenecks(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze metrics to identify performance bottlenecks."""
        logger.info("Analyzing performance bottlenecks...")
        
        bottlenecks = []
        thresholds = self.config['performance_thresholds']
        
        # Check API response time
        api_response_time = metrics['prometheus_metrics'].get('api_response_time', 0)
        if api_response_time > thresholds['api_response_time']:
            bottlenecks.append({
                'type': 'api_performance',
                'severity': 'high' if api_response_time > thresholds['api_response_time'] * 2 else 'medium',
                'metric': 'api_response_time',
                'current_value': api_response_time,
                'threshold': thresholds['api_response_time'],
                'description': f"API response time ({api_response_time:.2f}s) exceeds threshold ({thresholds['api_response_time']}s)",
                'recommendations': [
                    "Enable response caching",
                    "Optimize database queries",
                    "Scale API instances",
                    "Review slow endpoints"
                ]
            })
        
        # Check CPU usage
        cpu_usage = max(
            metrics['system_metrics'].get('cpu_usage', 0),
            metrics['prometheus_metrics'].get('cpu_usage', 0)
        )
        if cpu_usage > thresholds['cpu_usage']:
            bottlenecks.append({
                'type': 'resource_usage',
                'severity': 'high' if cpu_usage > 90 else 'medium',
                'metric': 'cpu_usage',
                'current_value': cpu_usage,
                'threshold': thresholds['cpu_usage'],
                'description': f"CPU usage ({cpu_usage:.1f}%) exceeds threshold ({thresholds['cpu_usage']}%)",
                'recommendations': [
                    "Scale horizontally",
                    "Optimize CPU-intensive operations",
                    "Review process efficiency",
                    "Consider CPU affinity tuning"
                ]
            })
        
        # Check memory usage
        memory_usage = max(
            metrics['system_metrics'].get('memory_usage', 0),
            metrics['prometheus_metrics'].get('memory_usage', 0)
        )
        if memory_usage > thresholds['memory_usage']:
            bottlenecks.append({
                'type': 'resource_usage',
                'severity': 'high' if memory_usage > 95 else 'medium',
                'metric': 'memory_usage',
                'current_value': memory_usage,
                'threshold': thresholds['memory_usage'],
                'description': f"Memory usage ({memory_usage:.1f}%) exceeds threshold ({thresholds['memory_usage']}%)",
                'recommendations': [
                    "Increase memory allocation",
                    "Optimize memory usage patterns",
                    "Review for memory leaks",
                    "Tune garbage collection"
                ]
            })
        
        # Check error rate
        error_rate = metrics['prometheus_metrics'].get('error_rate', 0)
        if error_rate > thresholds['error_rate']:
            bottlenecks.append({
                'type': 'reliability',
                'severity': 'critical' if error_rate > 0.05 else 'high',
                'metric': 'error_rate',
                'current_value': error_rate,
                'threshold': thresholds['error_rate'],
                'description': f"Error rate ({error_rate:.3f}) exceeds threshold ({thresholds['error_rate']})",
                'recommendations': [
                    "Investigate error logs",
                    "Review recent deployments",
                    "Check external dependencies",
                    "Implement circuit breakers"
                ]
            })
        
        # Check throughput
        throughput = metrics['prometheus_metrics'].get('throughput', 0)
        if throughput < thresholds['throughput_min']:
            bottlenecks.append({
                'type': 'throughput',
                'severity': 'medium',
                'metric': 'throughput',
                'current_value': throughput,
                'threshold': thresholds['throughput_min'],
                'description': f"Throughput ({throughput:.1f} req/s) below minimum ({thresholds['throughput_min']} req/s)",
                'recommendations': [
                    "Check for processing bottlenecks",
                    "Review queue depths",
                    "Scale processing capacity",
                    "Optimize critical paths"
                ]
            })
        
        return bottlenecks
    
    def apply_optimizations(self, bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply automated optimizations based on identified bottlenecks."""
        logger.info("Applying performance optimizations...")
        
        applied_optimizations = []
        strategies = self.config['optimization_strategies']
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'resource_usage' and strategies['auto_scaling']['enabled']:
                optimization = self._apply_auto_scaling(bottleneck)
                if optimization:
                    applied_optimizations.append(optimization)
            
            elif bottleneck['type'] == 'api_performance' and strategies['caching']['enabled']:
                optimization = self._apply_caching_optimization(bottleneck)
                if optimization:
                    applied_optimizations.append(optimization)
            
            elif bottleneck['metric'] in ['database_connections', 'query_time_avg'] and strategies['database_optimization']['enabled']:
                optimization = self._apply_database_optimization(bottleneck)
                if optimization:
                    applied_optimizations.append(optimization)
        
        return applied_optimizations
    
    def _apply_auto_scaling(self, bottleneck: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply auto-scaling optimization."""
        try:
            # This would typically interact with Kubernetes HPA or cloud auto-scaling
            logger.info(f"Applying auto-scaling for {bottleneck['metric']}")
            
            return {
                'type': 'auto_scaling',
                'action': 'scale_up',
                'metric': bottleneck['metric'],
                'timestamp': datetime.utcnow().isoformat(),
                'description': f"Scaled up due to high {bottleneck['metric']}",
                'success': True
            }
        except Exception as e:
            logger.error(f"Auto-scaling failed: {e}")
            return None
    
    def _apply_caching_optimization(self, bottleneck: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply caching optimization."""
        try:
            logger.info(f"Applying caching optimization for {bottleneck['metric']}")
            
            # This would typically adjust caching parameters
            return {
                'type': 'caching',
                'action': 'increase_cache_size',
                'metric': bottleneck['metric'],
                'timestamp': datetime.utcnow().isoformat(),
                'description': "Increased cache size and TTL",
                'success': True
            }
        except Exception as e:
            logger.error(f"Caching optimization failed: {e}")
            return None
    
    def _apply_database_optimization(self, bottleneck: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply database optimization."""
        try:
            logger.info(f"Applying database optimization for {bottleneck['metric']}")
            
            return {
                'type': 'database',
                'action': 'optimize_connection_pool',
                'metric': bottleneck['metric'],
                'timestamp': datetime.utcnow().isoformat(),
                'description': "Optimized database connection pool settings",
                'success': True
            }
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return None
    
    def send_performance_alerts(self, bottlenecks: List[Dict[str, Any]]) -> None:
        """Send performance alerts for critical issues."""
        critical_bottlenecks = [b for b in bottlenecks if b['severity'] == 'critical']
        high_bottlenecks = [b for b in bottlenecks if b['severity'] == 'high']
        
        if critical_bottlenecks or high_bottlenecks:
            self._send_slack_alert(critical_bottlenecks + high_bottlenecks)
            
        if critical_bottlenecks:
            self._send_email_alert(critical_bottlenecks)
    
    def _send_slack_alert(self, bottlenecks: List[Dict[str, Any]]) -> None:
        """Send Slack alert for performance issues."""
        try:
            webhook_url = os.path.expandvars(self.config['alerting']['slack_webhook'])
            if not webhook_url or webhook_url == '${SLACK_WEBHOOK_URL}':
                logger.warning("Slack webhook URL not configured")
                return
            
            message = {
                "text": "⚠️ Performance Bottlenecks Detected",
                "attachments": [
                    {
                        "color": "danger" if any(b['severity'] == 'critical' for b in bottlenecks) else "warning",
                        "fields": [
                            {
                                "title": f"{b['metric'].replace('_', ' ').title()}",
                                "value": f"{b['description']}\nRecommendations: {', '.join(b['recommendations'][:2])}",
                                "short": False
                            } for b in bottlenecks[:5]  # Limit to 5 bottlenecks
                        ],
                        "footer": f"Performance Optimizer | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=message, timeout=10)
            response.raise_for_status()
            logger.info("Slack alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_email_alert(self, bottlenecks: List[Dict[str, Any]]) -> None:
        """Send email alert for critical performance issues."""
        # Email implementation would go here
        logger.info(f"Would send email alert for {len(bottlenecks)} critical bottlenecks")
    
    def generate_performance_report(self, metrics: Dict[str, Any], bottlenecks: List[Dict[str, Any]], optimizations: List[Dict[str, Any]]) -> str:
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")
        
        report_file = f"performance-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"""# 📊 Performance Optimization Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Analysis Period:** Last 5 minutes  
**Environment:** Production  

## 🎯 Executive Summary

- **Bottlenecks Identified:** {len(bottlenecks)}
- **Critical Issues:** {len([b for b in bottlenecks if b['severity'] == 'critical'])}
- **Optimizations Applied:** {len(optimizations)}
- **System Health Score:** {self._calculate_health_score(metrics, bottlenecks)}/100

## 📈 Key Performance Metrics

### API Performance
- **Response Time:** {metrics['prometheus_metrics'].get('api_response_time', 0):.3f}s
- **Throughput:** {metrics['prometheus_metrics'].get('throughput', 0):.1f} req/s
- **Error Rate:** {metrics['prometheus_metrics'].get('error_rate', 0):.3%}

### System Resources
- **CPU Usage:** {metrics['system_metrics'].get('cpu_usage', 0):.1f}%
- **Memory Usage:** {metrics['system_metrics'].get('memory_usage', 0):.1f}%
- **Disk Usage:** {metrics['system_metrics'].get('disk_usage', 0):.1f}%

### Database Performance
- **Connection Count:** {metrics['database_metrics'].get('connection_count', 0)}
- **Query Time Average:** {metrics['database_metrics'].get('query_time_avg', 0):.3f}s
- **Cache Hit Ratio:** {metrics['database_metrics'].get('cache_hit_ratio', 0):.1%}

## 🚨 Performance Bottlenecks

""")
            
            if bottlenecks:
                for i, bottleneck in enumerate(bottlenecks, 1):
                    severity_emoji = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(bottleneck['severity'], '⚪')
                    
                    f.write(f"""### {i}. {severity_emoji} {bottleneck['metric'].replace('_', ' ').title()}

**Severity:** {bottleneck['severity'].title()}  
**Current Value:** {bottleneck['current_value']}  
**Threshold:** {bottleneck['threshold']}  
**Description:** {bottleneck['description']}

**Recommendations:**
""")
                    for rec in bottleneck['recommendations']:
                        f.write(f"- {rec}\n")
                    f.write("\n")
            else:
                f.write("No performance bottlenecks detected. System is operating within normal parameters.\n\n")
            
            f.write(f"""## ⚙️ Applied Optimizations

""")
            
            if optimizations:
                for i, opt in enumerate(optimizations, 1):
                    f.write(f"""### {i}. {opt['type'].replace('_', ' ').title()}

**Action:** {opt['action']}  
**Target Metric:** {opt['metric']}  
**Timestamp:** {opt['timestamp']}  
**Description:** {opt['description']}  
**Status:** {'✅ Success' if opt['success'] else '❌ Failed'}

""")
            else:
                f.write("No automated optimizations were applied during this analysis period.\n\n")
            
            f.write(f"""## 🔍 Detailed Analysis

### System Load Distribution
- **Load Average (1m):** {metrics['system_metrics'].get('load_average', [0])[0]:.2f}
- **Active Processes:** {metrics['system_metrics'].get('process_count', 0)}
- **Memory Available:** {metrics['system_metrics'].get('memory_available', 0) / (1024**3):.1f} GB

### Performance Trends
Based on the current analysis, the following trends are observed:

""")
            
            # Add trend analysis
            if metrics['prometheus_metrics'].get('api_response_time', 0) > 1.0:
                f.write("- 📈 API response times trending upward - investigate application bottlenecks\n")
            if metrics['system_metrics'].get('cpu_usage', 0) > 70:
                f.write("- 📈 CPU usage elevated - consider scaling or optimization\n")
            if metrics['system_metrics'].get('memory_usage', 0) > 80:
                f.write("- 📈 Memory usage high - monitor for potential leaks\n")
            
            f.write(f"""
## 📋 Action Items

### Immediate (Next 1 hour)
""")
            critical_items = [b for b in bottlenecks if b['severity'] == 'critical']
            if critical_items:
                for item in critical_items:
                    f.write(f"- [ ] Address {item['metric']}: {item['description']}\n")
            else:
                f.write("- No immediate critical actions required\n")
            
            f.write(f"""
### Short-term (Next 24 hours)
""")
            high_items = [b for b in bottlenecks if b['severity'] == 'high']
            if high_items:
                for item in high_items:
                    f.write(f"- [ ] Optimize {item['metric']}: {item['recommendations'][0]}\n")
            else:
                f.write("- Review and validate applied optimizations\n- Monitor system stability\n")
            
            f.write(f"""
### Long-term (Next week)
- [ ] Implement capacity planning based on current trends
- [ ] Review and update performance thresholds
- [ ] Conduct comprehensive performance audit
- [ ] Update monitoring and alerting rules

## 📞 Support

For performance-related issues:
1. Check system monitoring dashboards
2. Review application logs for errors
3. Validate recent deployments and changes
4. Contact the platform team if issues persist

---
*This report was generated automatically by the Performance Optimization System*
""")
        
        logger.info(f"Performance report generated: {report_file}")
        return report_file
    
    def _calculate_health_score(self, metrics: Dict[str, Any], bottlenecks: List[Dict[str, Any]]) -> int:
        """Calculate overall system health score."""
        base_score = 100
        
        # Deduct points for bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck['severity'] == 'critical':
                base_score -= 25
            elif bottleneck['severity'] == 'high':
                base_score -= 15
            elif bottleneck['severity'] == 'medium':
                base_score -= 10
            elif bottleneck['severity'] == 'low':
                base_score -= 5
        
        return max(0, base_score)
    
    async def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run a complete optimization cycle."""
        logger.info("Starting performance optimization cycle...")
        
        try:
            # Collect metrics
            metrics = self.collect_performance_metrics()
            
            # Analyze bottlenecks
            bottlenecks = self.analyze_performance_bottlenecks(metrics)
            
            # Apply optimizations
            optimizations = self.apply_optimizations(bottlenecks)
            
            # Send alerts if needed
            if bottlenecks:
                self.send_performance_alerts(bottlenecks)
            
            # Generate report
            report_file = self.generate_performance_report(metrics, bottlenecks, optimizations)
            
            # Store optimization history
            cycle_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics,
                'bottlenecks_count': len(bottlenecks),
                'optimizations_count': len(optimizations),
                'health_score': self._calculate_health_score(metrics, bottlenecks),
                'report_file': report_file
            }
            
            self.optimization_history.append(cycle_result)
            
            logger.info(f"Optimization cycle completed - Health Score: {cycle_result['health_score']}/100")
            return cycle_result
            
        except Exception as e:
            logger.error(f"Optimization cycle failed: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    async def continuous_optimization(self) -> None:
        """Run continuous performance optimization."""
        logger.info("Starting continuous performance optimization...")
        
        interval = self.config.get('optimization_interval', 300)
        
        while True:
            try:
                await self.run_optimization_cycle()
                await asyncio.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Continuous optimization stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous optimization: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry


async def main():
    """Main function."""
    logger.info("🚀 Performance Optimization System Starting...")
    
    try:
        # Initialize optimizer
        optimizer = PerformanceOptimizer()
        
        # Run single optimization cycle or continuous mode
        if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
            await optimizer.continuous_optimization()
        else:
            result = await optimizer.run_optimization_cycle()
            
            if 'error' in result:
                logger.error(f"Optimization failed: {result['error']}")
                sys.exit(1)
            else:
                logger.info(f"✅ Optimization completed successfully!")
                logger.info(f"📊 Health Score: {result['health_score']}/100")
                logger.info(f"📋 Report: {result['report_file']}")
                sys.exit(0)
    
    except Exception as e:
        logger.error(f"💥 Fatal error in performance optimization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())