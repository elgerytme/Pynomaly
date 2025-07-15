"""Web UI monitoring and logging utilities."""

import logging
import time
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class WebUIMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring web UI performance and errors."""

    def __init__(self, app, logger: logging.Logger | None = None):
        super().__init__(app)
        self.logger = logger or logging.getLogger(__name__)
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.start_time = time.time()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and collect monitoring metrics."""
        start_time = time.time()
        self.request_count += 1

        try:
            # Log request
            self.logger.info(
                f"Request: {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'}"
            )

            # Process request
            response = await call_next(request)

            # Calculate response time
            response_time = time.time() - start_time
            self.response_times.append(response_time)

            # Log response
            self.logger.info(
                f"Response: {response.status_code} in {response_time:.3f}s"
            )

            # Track slow requests
            if response_time > 1.0:  # Requests taking more than 1 second
                self.logger.warning(
                    f"Slow request: {request.method} {request.url.path} "
                    f"took {response_time:.3f}s"
                )

            # Add monitoring headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Request-ID"] = str(self.request_count)

            return response

        except Exception as e:
            self.error_count += 1
            self.logger.error(
                f"Error processing request {request.method} {request.url.path}: {str(e)}"
            )
            raise

    def get_metrics(self) -> dict:
        """Get current monitoring metrics."""
        uptime = time.time() - self.start_time
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times
            else 0
        )

        return {
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count
            if self.request_count > 0
            else 0,
            "average_response_time": avg_response_time,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
        }


class WebUILogger:
    """Centralized logging for web UI components."""

    def __init__(self, name: str = "pynomaly.web"):
        self.logger = logging.getLogger(name)
        self.setup_logger()

    def setup_logger(self):
        """Configure logger with appropriate handlers and formatters."""
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # File handler (if log file path is configured)
            try:
                import os

                log_file = os.getenv("LOG_FILE_PATH", "/tmp/pynomaly_web.log")
                file_handler = logging.FileHandler(log_file)
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not setup file logging: {e}")

            # Set log level
            import os

            log_level = os.getenv("LOG_LEVEL", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level))

    def log_user_action(self, user_id: str, action: str, details: dict = None):
        """Log user actions for audit trail."""
        self.logger.info(
            f"User action: {user_id} performed {action}",
            extra={
                "user_id": user_id,
                "action": action,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def log_security_event(self, event_type: str, details: dict):
        """Log security events."""
        self.logger.warning(
            f"Security event: {event_type}",
            extra={
                "event_type": event_type,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
                "security_event": True,
            },
        )

    def log_performance_metric(
        self, metric_name: str, value: float, context: dict = None
    ):
        """Log performance metrics."""
        self.logger.info(
            f"Performance metric: {metric_name} = {value}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat(),
                "performance_metric": True,
            },
        )

    def log_error(self, error: Exception, context: dict = None):
        """Log errors with context."""
        self.logger.error(
            f"Error: {str(error)}",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
            exc_info=True,
        )


class PerformanceMonitor:
    """Monitor web UI performance metrics."""

    def __init__(self):
        self.metrics = {
            "page_load_times": [],
            "api_response_times": [],
            "memory_usage": [],
            "core_web_vitals": {
                "LCP": [],  # Largest Contentful Paint
                "FID": [],  # First Input Delay
                "CLS": [],  # Cumulative Layout Shift
            },
        }
        self.logger = WebUILogger("pynomaly.web.performance")

    def record_page_load_time(self, page: str, load_time: float):
        """Record page load time."""
        self.metrics["page_load_times"].append(
            {
                "page": page,
                "load_time": load_time,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Log slow page loads
        if load_time > 3.0:  # Pages taking more than 3 seconds
            self.logger.logger.warning(f"Slow page load: {page} took {load_time:.3f}s")

    def record_api_response_time(self, endpoint: str, response_time: float):
        """Record API response time."""
        self.metrics["api_response_times"].append(
            {
                "endpoint": endpoint,
                "response_time": response_time,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Log slow API responses
        if response_time > 1.0:  # API calls taking more than 1 second
            self.logger.logger.warning(
                f"Slow API response: {endpoint} took {response_time:.3f}s"
            )

    def record_core_web_vital(self, metric: str, value: float):
        """Record Core Web Vitals metric."""
        if metric in self.metrics["core_web_vitals"]:
            self.metrics["core_web_vitals"][metric].append(
                {"value": value, "timestamp": datetime.utcnow().isoformat()}
            )

            # Check thresholds
            thresholds = {
                "LCP": 2.5,  # 2.5 seconds
                "FID": 100,  # 100 milliseconds
                "CLS": 0.1,  # 0.1 score
            }

            if value > thresholds.get(metric, float("inf")):
                self.logger.logger.warning(
                    f"Poor {metric}: {value} exceeds threshold {thresholds[metric]}"
                )

    def get_performance_summary(self) -> dict:
        """Get performance metrics summary."""
        summary = {}

        # Page load times
        if self.metrics["page_load_times"]:
            load_times = [m["load_time"] for m in self.metrics["page_load_times"]]
            summary["page_load_times"] = {
                "average": sum(load_times) / len(load_times),
                "min": min(load_times),
                "max": max(load_times),
                "count": len(load_times),
            }

        # API response times
        if self.metrics["api_response_times"]:
            response_times = [
                m["response_time"] for m in self.metrics["api_response_times"]
            ]
            summary["api_response_times"] = {
                "average": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "count": len(response_times),
            }

        # Core Web Vitals
        summary["core_web_vitals"] = {}
        for metric, values in self.metrics["core_web_vitals"].items():
            if values:
                metric_values = [v["value"] for v in values]
                summary["core_web_vitals"][metric] = {
                    "average": sum(metric_values) / len(metric_values),
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "count": len(metric_values),
                }

        return summary


class SecurityMonitor:
    """Monitor security events and threats."""

    def __init__(self):
        self.security_events = []
        self.threat_patterns = {
            "xss_patterns": [
                "<script",
                "</script>",
                "javascript:",
                "onerror=",
                "onload=",
                "alert(",
                "confirm(",
                "prompt(",
                "eval(",
                "setTimeout(",
            ],
            "sql_injection_patterns": [
                "' OR '1'='1",
                "' OR 1=1",
                "UNION SELECT",
                "DROP TABLE",
                "INSERT INTO",
                "UPDATE SET",
                "DELETE FROM",
                "; --",
                "/*",
            ],
            "path_traversal_patterns": [
                "../",
                "..\\",
                "..\\/",
                "..%2F",
                "..%5C",
                "%2E%2E%2F",
                "%2E%2E%5C",
            ],
        }
        self.logger = WebUILogger("pynomaly.web.security")

    def check_request_security(self, request: Request) -> list[dict]:
        """Check request for security threats."""
        threats = []

        # Check URL path
        path = str(request.url.path)
        for pattern in self.threat_patterns["path_traversal_patterns"]:
            if pattern in path:
                threats.append(
                    {
                        "type": "path_traversal",
                        "pattern": pattern,
                        "location": "url_path",
                        "value": path,
                    }
                )

        # Check query parameters
        for key, value in request.query_params.items():
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in str(value).lower():
                        threats.append(
                            {
                                "type": threat_type,
                                "pattern": pattern,
                                "location": f"query_param_{key}",
                                "value": str(value),
                            }
                        )

        # Log threats
        for threat in threats:
            self.logger.log_security_event(
                f"Request threat detected: {threat['type']}",
                {
                    "threat": threat,
                    "request_path": path,
                    "user_agent": request.headers.get("user-agent", "unknown"),
                    "client_ip": request.client.host if request.client else "unknown",
                },
            )

        return threats

    def record_security_event(self, event_type: str, details: dict):
        """Record security event."""
        event = {
            "type": event_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.security_events.append(event)
        self.logger.log_security_event(event_type, details)

    def get_security_summary(self) -> dict:
        """Get security events summary."""
        if not self.security_events:
            return {"total_events": 0, "event_types": {}}

        event_types = {}
        for event in self.security_events:
            event_type = event["type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            "total_events": len(self.security_events),
            "event_types": event_types,
            "recent_events": self.security_events[-10:],  # Last 10 events
        }


# Global instances
web_ui_logger = WebUILogger()
performance_monitor = PerformanceMonitor()
security_monitor = SecurityMonitor()
