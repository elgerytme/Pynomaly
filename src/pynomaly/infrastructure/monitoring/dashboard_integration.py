#!/usr/bin/env python3
"""Integration layer for real-time monitoring dashboard with error tracking and health monitoring."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from pynomaly.infrastructure.monitoring.error_tracking_integration import (
    get_error_tracker,
    get_health_monitor
)

logger = logging.getLogger(__name__)


class MonitoringDashboardIntegration:
    """Enhanced monitoring dashboard integration with error tracking and health monitoring."""
    
    def __init__(
        self,
        metrics_collector=None,
        websocket_manager=None,
        dashboard_service=None
    ):
        self.metrics_collector = metrics_collector
        self.websocket_manager = websocket_manager
        self.dashboard_service = dashboard_service
        
        # Get global error tracking and health monitoring instances
        self.error_tracker = get_error_tracker()
        self.health_monitor = get_health_monitor()
        
        # Subscribe to error events for real-time notifications
        self.error_tracker.subscribers.append(self._handle_error_event)
        
        # Setup standard health checks
        self._setup_standard_health_checks()
        
        logger.info("Monitoring dashboard integration initialized with error tracking and health monitoring")
    
    def _setup_standard_health_checks(self):
        """Setup standard health checks for the system."""
        
        # System resources health check
        def check_system_resources():
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                return cpu_percent < 90 and memory_percent < 90
            except Exception:
                return False
        
        # Application health check
        def check_application():
            return True  # Basic check - can be enhanced
        
        # Register health checks
        self.health_monitor.register_health_check("system_resources", check_system_resources, critical=True)
        self.health_monitor.register_health_check("application", check_application, critical=True)
    
    async def _handle_error_event(self, error_event):
        """Handle new error events by broadcasting to WebSocket subscribers."""
        try:
            if self.websocket_manager:
                await self.websocket_manager.broadcast_to_topic('errors', {
                    'type': 'error_event',
                    'data': error_event.to_dict(),
                    'timestamp': datetime.utcnow().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to broadcast error event: {e}")
    
    async def get_comprehensive_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data including errors and health."""
        try:
            # Get base metrics from metrics collector
            base_metrics = {}
            if self.metrics_collector:
                base_metrics = self.metrics_collector.get_metrics_summary()
            
            # Get error tracking data
            error_summary = self.error_tracker.get_error_summary()
            recent_errors = self.error_tracker.get_recent_errors(limit=10)
            
            # Get health monitoring data
            health_checks = await self.health_monitor.run_health_checks()
            health_summary = self.health_monitor.get_system_health_summary()
            
            return {
                'system_overview': {
                    'metrics': base_metrics,
                    'health': health_summary,
                    'status': health_checks['overall_status'],
                    'last_updated': datetime.utcnow().isoformat(),
                },
                'error_tracking': {
                    'summary': error_summary,
                    'recent_errors': recent_errors,
                },
                'health_monitoring': {
                    'checks': health_checks,
                    'summary': health_summary,
                },
                'dashboard_metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'data_sources': ['metrics_collector', 'error_tracker', 'health_monitor'],
                    'features_enabled': {
                        'real_time_metrics': self.metrics_collector is not None,
                        'error_tracking': True,
                        'health_monitoring': True,
                        'websocket_updates': self.websocket_manager is not None,
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting comprehensive dashboard data: {e}")
            return {
                'system_overview': {
                    'status': 'error',
                    'error': str(e),
                    'last_updated': datetime.utcnow().isoformat(),
                },
                'dashboard_metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'error': 'Failed to generate dashboard data',
                }
            }
    
    def get_dashboard_health_status(self) -> Dict[str, Any]:
        """Get current dashboard health status."""
        try:
            health_summary = self.health_monitor.get_system_health_summary()
            error_summary = self.error_tracker.get_error_summary()
            
            # Calculate overall dashboard health
            health_score = health_summary.get('health_score', 0)
            recent_error_rate = error_summary.get('error_rate_per_minute', 0)
            
            # Determine dashboard status
            if health_score >= 80 and recent_error_rate < 1:
                status = "healthy"
                status_color = "green"
            elif health_score >= 60 and recent_error_rate < 5:
                status = "warning"
                status_color = "yellow"
            else:
                status = "critical"
                status_color = "red"
            
            return {
                'status': status,
                'status_color': status_color,
                'health_score': health_score,
                'error_rate': recent_error_rate,
                'system_status': health_summary.get('system_status', 'unknown'),
                'components_status': {
                    'error_tracking': 'healthy',
                    'health_monitoring': 'healthy',
                    'metrics_collection': 'healthy' if self.metrics_collector else 'disabled',
                    'websocket_streaming': 'healthy' if self.websocket_manager else 'disabled',
                },
                'last_updated': datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard health status: {e}")
            return {
                'status': 'error',
                'status_color': 'red',
                'health_score': 0,
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat(),
            }


# Global dashboard integration instance
_dashboard_integration = None


def get_dashboard_integration(
    metrics_collector=None,
    websocket_manager=None,
    dashboard_service=None
) -> MonitoringDashboardIntegration:
    """Get the global dashboard integration instance."""
    global _dashboard_integration
    if _dashboard_integration is None:
        _dashboard_integration = MonitoringDashboardIntegration(
            metrics_collector=metrics_collector,
            websocket_manager=websocket_manager,
            dashboard_service=dashboard_service
        )
    return _dashboard_integration