"""Resource monitoring utilities for AutoML pipeline execution."""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor resource usage during pipeline execution"""

    def __init__(self):
        """Initialize resource monitor."""
        self._psutil_available = self._check_psutil_available()

    def _check_psutil_available(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil
            return True
        except ImportError:
            logger.warning("psutil not available - resource monitoring disabled")
            return False

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if not self._psutil_available:
            return 0.0

        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        if not self._psutil_available:
            return 0.0

        try:
            import psutil

            return psutil.cpu_percent(interval=1)
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {e}")
            return 0.0

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        if not self._psutil_available:
            return {
                "memory_mb": 0.0,
                "cpu_percent": 0.0,
                "disk_usage_percent": 0.0,
                "available": False
            }

        try:
            import psutil

            # Memory information
            memory = psutil.virtual_memory()
            
            # Disk information
            disk = psutil.disk_usage('/')
            
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=0.1)

            return {
                "memory_mb": memory.used / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024),
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                "available": True
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {
                "memory_mb": 0.0,
                "cpu_percent": 0.0,
                "disk_usage_percent": 0.0,
                "available": False,
                "error": str(e)
            }

    def check_resource_limits(self, max_memory_gb: float = 8.0, max_cpu_percent: float = 90.0) -> Dict[str, Any]:
        """Check if current resource usage exceeds limits."""
        system_info = self.get_system_info()
        
        if not system_info["available"]:
            return {
                "within_limits": True,
                "warnings": ["Resource monitoring not available"],
                "violations": []
            }

        warnings = []
        violations = []
        
        # Check memory limit
        memory_gb = system_info["memory_mb"] / 1024
        if memory_gb > max_memory_gb:
            violations.append(f"Memory usage ({memory_gb:.1f}GB) exceeds limit ({max_memory_gb}GB)")
        elif memory_gb > max_memory_gb * 0.8:
            warnings.append(f"Memory usage ({memory_gb:.1f}GB) approaching limit ({max_memory_gb}GB)")

        # Check CPU limit
        cpu_percent = system_info["cpu_percent"]
        if cpu_percent > max_cpu_percent:
            violations.append(f"CPU usage ({cpu_percent:.1f}%) exceeds limit ({max_cpu_percent}%)")
        elif cpu_percent > max_cpu_percent * 0.8:
            warnings.append(f"CPU usage ({cpu_percent:.1f}%) approaching limit ({max_cpu_percent}%)")

        # Check disk space
        disk_percent = system_info["disk_usage_percent"]
        if disk_percent > 95:
            violations.append(f"Disk usage ({disk_percent:.1f}%) critically high")
        elif disk_percent > 85:
            warnings.append(f"Disk usage ({disk_percent:.1f}%) getting high")

        return {
            "within_limits": len(violations) == 0,
            "warnings": warnings,
            "violations": violations,
            "system_info": system_info
        }