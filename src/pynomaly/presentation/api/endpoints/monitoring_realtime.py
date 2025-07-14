#!/usr/bin/env python3
"""Real-time monitoring dashboard API endpoints."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from aiohttp import web, WSMsgType
from aiohttp.web_response import Response
from dependency_injector.wiring import Provide, inject

from pynomaly.infrastructure.config.container import Container
from pynomaly.infrastructure.monitoring.enhanced_metrics_collector import EnhancedMetricsCollector
from pynomaly.infrastructure.monitoring.realtime_websocket_manager import RealtimeWebSocketManager
from pynomaly.infrastructure.monitoring.dashboard_service import DashboardService
from pynomaly.infrastructure.monitoring.dashboard_integration import get_dashboard_integration
from pynomaly.infrastructure.monitoring.error_tracking_integration import get_error_tracker, get_health_monitor
logger = logging.getLogger(__name__)


class RealtimeMonitoringAPI:
    """Real-time monitoring dashboard API endpoints."""
    
    @inject
    def __init__(
        self,
        metrics_collector: EnhancedMetricsCollector = Provide[Container.enhanced_metrics_collector],
        websocket_manager: RealtimeWebSocketManager = Provide[Container.websocket_manager],
        dashboard_service: DashboardService = Provide[Container.dashboard_service],
    ):
        self.metrics_collector = metrics_collector
        self.websocket_manager = websocket_manager
        self.dashboard_service = dashboard_service
        
        # Initialize enhanced monitoring integration
        self.dashboard_integration = get_dashboard_integration(
            metrics_collector=metrics_collector,
            websocket_manager=websocket_manager,
            dashboard_service=dashboard_service
        )
        self.error_tracker = get_error_tracker()
        self.health_monitor = get_health_monitor()
    
    async def get_metrics_summary(self, request: web.Request) -> web.Response:
        """Get real-time metrics summary."""
        try:
            summary = self.metrics_collector.get_metrics_summary()
            return web.json_response({
                'success': True,
                'data': summary,
                'timestamp': datetime.utcnow().isoformat(),
            })
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def get_time_series_metrics(self, request: web.Request) -> web.Response:
        """Get time series data for specific metrics."""
        try:
            # Parse query parameters
            metric_names = request.query.get('metrics', '').split(',')
            time_range = request.query.get('time_range', '1h')
            
            if not metric_names or metric_names == ['']:
                return web.json_response({
                    'success': False,
                    'error': 'No metrics specified',
                }, status=400)
            
            # Parse time range
            time_delta = self._parse_time_range(time_range)
            
            # Get time series data for each metric
            metrics_data = {}
            for metric_name in metric_names:
                metric_name = metric_name.strip()
                if metric_name:
                    time_series = self.metrics_collector.get_time_series_data(metric_name, time_delta)
                    metrics_data[metric_name] = time_series
            
            return web.json_response({
                'success': True,
                'data': metrics_data,
                'time_range': time_range,
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error getting time series metrics: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def get_system_health(self, request: web.Request) -> web.Response:
        """Get comprehensive system health status."""
        try:
            summary = self.metrics_collector.get_metrics_summary()
            
            # Calculate health status
            health_score = summary['system'].get('health_score', 0)
            
            if health_score >= 80:
                status = 'healthy'
                status_color = 'green'
            elif health_score >= 60:
                status = 'warning'
                status_color = 'yellow'
            else:
                status = 'critical'
                status_color = 'red'
            
            health_data = {
                'status': status,
                'status_color': status_color,
                'health_score': health_score,
                'uptime': summary['system'].get('uptime', 0),
                'system_metrics': summary['system'],
                'application_metrics': summary['application'],
                'timestamp': datetime.utcnow().isoformat(),
                'checks': {
                    'cpu_usage': {
                        'status': 'ok' if summary['system'].get('cpu_usage', 0) < 80 else 'warning',
                        'value': summary['system'].get('cpu_usage', 0),
                        'unit': '%',
                    },
                    'memory_usage': {
                        'status': 'ok' if summary['system'].get('memory_usage', 0) < 85 else 'warning',
                        'value': summary['system'].get('memory_usage', 0),
                        'unit': '%',
                    },
                    'disk_usage': {
                        'status': 'ok' if summary['system'].get('disk_usage', 0) < 90 else 'warning',
                        'value': summary['system'].get('disk_usage', 0),
                        'unit': '%',
                    },
                    'websocket_connections': {
                        'status': 'ok',
                        'value': summary['application'].get('websocket_connections', 0),
                        'unit': 'connections',
                    },
                }
            }
            
            return web.json_response({
                'success': True,
                'data': health_data,
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def get_dashboard_list(self, request: web.Request) -> web.Response:
        """Get list of available dashboards."""
        try:
            # For now, return predefined dashboard types
            dashboards = [
                {
                    'id': 'system_overview',
                    'name': 'System Overview',
                    'description': 'High-level system health and performance metrics',
                    'type': 'system_overview',
                    'is_default': True,
                },
                {
                    'id': 'application_performance',
                    'name': 'Application Performance',
                    'description': 'Application-specific performance metrics',
                    'type': 'application_performance',
                    'is_default': False,
                },
                {
                    'id': 'ml_model_performance',
                    'name': 'ML Model Performance',
                    'description': 'Machine learning model metrics and performance',
                    'type': 'ml_model_performance',
                    'is_default': False,
                },
                {
                    'id': 'business_metrics',
                    'name': 'Business Metrics',
                    'description': 'Key business performance indicators',
                    'type': 'business_metrics',
                    'is_default': False,
                },
            ]
            
            return web.json_response({
                'success': True,
                'data': dashboards,
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error getting dashboard list: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def create_dashboard(self, request: web.Request) -> web.Response:
        """Create a new custom dashboard."""
        try:
            data = await request.json()
            
            name = data.get('name')
            description = data.get('description', '')
            dashboard_type = data.get('type', 'custom')
            
            if not name:
                return web.json_response({
                    'success': False,
                    'error': 'Dashboard name is required',
                }, status=400)
            
            # Create dashboard using dashboard service
            from pynomaly.domain.models.monitoring import DashboardType
            
            dashboard_type_enum = getattr(DashboardType, dashboard_type.upper(), DashboardType.CUSTOM)
            
            dashboard = await self.dashboard_service.create_dashboard(
                name=name,
                description=description,
                dashboard_type=dashboard_type_enum,
            )
            
            return web.json_response({
                'success': True,
                'data': {
                    'id': str(dashboard.dashboard_id),
                    'name': dashboard.name,
                    'description': dashboard.description,
                    'type': dashboard.dashboard_type.value,
                    'created_at': dashboard.created_at.isoformat(),
                },
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def get_dashboard_data(self, request: web.Request) -> web.Response:
        """Get dashboard data with real-time metrics."""
        try:
            dashboard_id = request.match_info.get('dashboard_id')
            time_range_override = request.query.get('time_range')
            
            if not dashboard_id:
                return web.json_response({
                    'success': False,
                    'error': 'Dashboard ID is required',
                }, status=400)
            
            # For predefined dashboards, use dashboard type
            if dashboard_id in ['system_overview', 'application_performance', 'ml_model_performance', 'business_metrics']:
                dashboard_data = await self._get_predefined_dashboard_data(dashboard_id, time_range_override)
            else:
                # Try to get custom dashboard
                from uuid import UUID
                try:
                    dashboard_uuid = UUID(dashboard_id)
                    dashboard_data = await self.dashboard_service.get_dashboard_data(
                        dashboard_uuid, 
                        user_id=UUID('00000000-0000-0000-0000-000000000000'),  # Default user for now
                        time_range_override=time_range_override
                    )
                except ValueError:
                    return web.json_response({
                        'success': False,
                        'error': 'Invalid dashboard ID format',
                    }, status=400)
            
            if not dashboard_data:
                return web.json_response({
                    'success': False,
                    'error': 'Dashboard not found',
                }, status=404)
            
            return web.json_response({
                'success': True,
                'data': dashboard_data,
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates."""
        try:
            return await self.websocket_manager.handle_websocket(request)
        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.close(code=1011, message=b'Internal server error')
            return ws
    
    async def get_websocket_stats(self, request: web.Request) -> web.Response:
        """Get WebSocket connection statistics."""
        try:
            stats = self.websocket_manager.get_connection_stats()
            return web.json_response({
                'success': True,
                'data': stats,
                'timestamp': datetime.utcnow().isoformat(),
            })
        except Exception as e:
            logger.error(f"Error getting WebSocket stats: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def record_custom_metric(self, request: web.Request) -> web.Response:
        """Record a custom metric value."""
        try:
            data = await request.json()
            
            metric_name = data.get('name')
            metric_value = data.get('value')
            tags = data.get('tags', {})
            
            if not metric_name or metric_value is None:
                return web.json_response({
                    'success': False,
                    'error': 'Metric name and value are required',
                }, status=400)
            
            # Record the metric
            from pynomaly.infrastructure.monitoring.enhanced_metrics_collector import MetricPoint
            
            metric_point = MetricPoint(
                name=metric_name,
                value=float(metric_value),
                timestamp=datetime.utcnow(),
                tags=tags
            )
            
            self.metrics_collector.aggregator.add_metric(metric_point)
            
            # Broadcast to WebSocket subscribers if relevant
            await self.websocket_manager.broadcast_to_topic('custom_metrics', {
                'type': 'custom_metric',
                'data': {
                    'name': metric_name,
                    'value': metric_value,
                    'tags': tags,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            })
            
            return web.json_response({
                'success': True,
                'message': 'Metric recorded successfully',
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error recording custom metric: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def get_error_tracking_summary(self, request: web.Request) -> web.Response:
        """Get error tracking summary."""
        try:
            time_window_param = request.query.get('time_window', '5m')
            
            # Parse time window
            time_delta = self._parse_time_range(time_window_param)
            error_summary = self.error_tracker.get_error_summary(time_delta)
            
            return web.json_response({
                'success': True,
                'data': error_summary,
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error getting error tracking summary: {e}")
            self.error_tracker.track_error(e, component="api_monitoring", metadata={'endpoint': 'get_error_tracking_summary'})
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def get_recent_errors(self, request: web.Request) -> web.Response:
        """Get recent error events."""
        try:
            limit = int(request.query.get('limit', '50'))
            severity = request.query.get('severity')
            
            recent_errors = self.error_tracker.get_recent_errors(limit=limit, severity=severity)
            
            return web.json_response({
                'success': True,
                'data': recent_errors,
                'count': len(recent_errors),
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error getting recent errors: {e}")
            self.error_tracker.track_error(e, component="api_monitoring", metadata={'endpoint': 'get_recent_errors'})
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def get_enhanced_system_health(self, request: web.Request) -> web.Response:
        """Get enhanced system health with error tracking integration."""
        try:
            # Run health checks
            health_checks = await self.health_monitor.run_health_checks()
            
            # Get comprehensive health summary
            health_summary = self.health_monitor.get_system_health_summary()
            
            # Get dashboard health status
            dashboard_health = self.dashboard_integration.get_dashboard_health_status()
            
            return web.json_response({
                'success': True,
                'data': {
                    'health_checks': health_checks,
                    'health_summary': health_summary,
                    'dashboard_health': dashboard_health,
                },
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error getting enhanced system health: {e}")
            self.error_tracker.track_error(e, component="api_monitoring", metadata={'endpoint': 'get_enhanced_system_health'})
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def get_comprehensive_dashboard_data(self, request: web.Request) -> web.Response:
        """Get comprehensive dashboard data with error tracking and health monitoring."""
        try:
            dashboard_data = await self.dashboard_integration.get_comprehensive_dashboard_data()
            
            return web.json_response({
                'success': True,
                'data': dashboard_data,
                'timestamp': datetime.utcnow().isoformat(),
            })
            
        except Exception as e:
            logger.error(f"Error getting comprehensive dashboard data: {e}")
            self.error_tracker.track_error(e, component="api_monitoring", metadata={'endpoint': 'get_comprehensive_dashboard_data'})
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
            }, status=500)
    
    async def _get_predefined_dashboard_data(self, dashboard_id: str, time_range_override: Optional[str]) -> Dict[str, Any]:
        """Get data for predefined dashboard types."""
        summary = self.metrics_collector.get_metrics_summary()
        time_range = time_range_override or '1h'
        
        base_data = {
            'dashboard': {
                'id': dashboard_id,
                'name': dashboard_id.replace('_', ' ').title(),
                'type': dashboard_id,
                'last_updated': datetime.utcnow().isoformat(),
            },
            'widgets': [],
        }
        
        if dashboard_id == 'system_overview':
            base_data['widgets'] = [
                {
                    'id': 'system_health',
                    'title': 'System Health',
                    'type': 'gauge',
                    'data': {
                        'value': summary['system'].get('health_score', 0),
                        'status': 'healthy' if summary['system'].get('health_score', 0) > 80 else 'warning',
                        'unit': '%',
                    }
                },
                {
                    'id': 'cpu_usage',
                    'title': 'CPU Usage',
                    'type': 'chart',
                    'data': self.metrics_collector.get_time_series_data('cpu_usage_percent', self._parse_time_range(time_range))
                },
                {
                    'id': 'memory_usage',
                    'title': 'Memory Usage',
                    'type': 'chart',
                    'data': self.metrics_collector.get_time_series_data('memory_usage_percent', self._parse_time_range(time_range))
                },
            ]
        
        elif dashboard_id == 'application_performance':
            base_data['widgets'] = [
                {
                    'id': 'active_sessions',
                    'title': 'Active Sessions',
                    'type': 'text',
                    'data': {
                        'value': summary['application'].get('active_sessions', 0),
                        'unit': 'sessions',
                    }
                },
                {
                    'id': 'websocket_connections',
                    'title': 'WebSocket Connections',
                    'type': 'text',
                    'data': {
                        'value': summary['application'].get('websocket_connections', 0),
                        'unit': 'connections',
                    }
                },
            ]
        
        return base_data
    
    def _parse_time_range(self, time_range: str) -> timedelta:
        """Parse time range string to timedelta."""
        time_map = {
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '6h': timedelta(hours=6),
            '12h': timedelta(hours=12),
            '24h': timedelta(hours=24),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30),
        }
        return time_map.get(time_range, timedelta(hours=1))


def setup_monitoring_routes(app: web.Application, api: RealtimeMonitoringAPI):
    """Setup monitoring API routes."""
    
    # Real-time metrics endpoints
    app.router.add_get('/api/monitoring/metrics/summary', api.get_metrics_summary)
    app.router.add_get('/api/monitoring/metrics/timeseries', api.get_time_series_metrics)
    app.router.add_post('/api/monitoring/metrics/custom', api.record_custom_metric)
    
    # System health endpoints
    app.router.add_get('/api/monitoring/health', api.get_system_health)
    app.router.add_get('/api/monitoring/health/enhanced', api.get_enhanced_system_health)
    
    # Error tracking endpoints
    app.router.add_get('/api/monitoring/errors/summary', api.get_error_tracking_summary)
    app.router.add_get('/api/monitoring/errors/recent', api.get_recent_errors)
    
    # Enhanced dashboard endpoints
    app.router.add_get('/api/monitoring/dashboard/comprehensive', api.get_comprehensive_dashboard_data)
    
    # Dashboard endpoints
    app.router.add_get('/api/monitoring/dashboards', api.get_dashboard_list)
    app.router.add_post('/api/monitoring/dashboards', api.create_dashboard)
    app.router.add_get('/api/monitoring/dashboards/{dashboard_id}', api.get_dashboard_data)
    
    # WebSocket endpoints
    app.router.add_get('/api/monitoring/ws', api.websocket_handler)
    app.router.add_get('/api/monitoring/ws/stats', api.get_websocket_stats)
    
    logger.info("Enhanced real-time monitoring API routes registered")


# For backward compatibility
MonitoringRealtimeAPI = RealtimeMonitoringAPI