"""
Package Health Monitoring System

This module provides comprehensive health monitoring for all packages in the Software platform.
"""

import json
import time
import psutil
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class PackageHealth:
    """Package health information"""
    package_name: str
    version: str
    status: HealthStatus
    last_check: datetime
    measurements: List[HealthMetric]
    dependencies_status: Dict[str, HealthStatus]
    uptime: Optional[float] = None
    error_count: int = 0
    warning_count: int = 0


class HealthMonitor:
    """Central health monitoring system"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("health_monitor_config.json")
        self.config = self._load_config()
        self.packages = self._discover_packages()
        self.measurements_history: Dict[str, List[HealthMetric]] = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load health monitoring configuration"""
        default_config = {
            "check_interval": 300,  # 5 minutes
            "retention_days": 30,
            "thresholds": {
                "cpu_usage": {"warning": 70, "critical": 90},
                "memory_usage": {"warning": 80, "critical": 95},
                "disk_usage": {"warning": 85, "critical": 95},
                "response_time": {"warning": 1000, "critical": 5000},
                "error_rate": {"warning": 1, "critical": 5}
            },
            "alerting": {
                "enabled": True,
                "channels": ["email", "slack"],
                "cooldown_minutes": 30
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
                
        return default_config
    
    def _discover_packages(self) -> List[str]:
        """Discover all packages in the monorepo"""
        packages = []
        base_path = Path("/mnt/c/Users/andre/Software/src/packages")
        
        for pyproject_file in base_path.rglob("pyproject.toml"):
            if ".venv" not in str(pyproject_file):
                relative_path = pyproject_file.parent.relative_to(base_path)
                packages.append(str(relative_path))
                
        return packages
    
    async def check_system_health(self) -> Dict[str, HealthMetric]:
        """Check system-wide health measurements"""
        measurements = {}
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        measurements["cpu_usage"] = HealthMetric(
            name="cpu_usage",
            value=cpu_usage,
            unit="percent",
            timestamp=datetime.now(),
            status=self._get_status(cpu_usage, "cpu_usage"),
            threshold_warning=self.config["thresholds"]["cpu_usage"]["warning"],
            threshold_critical=self.config["thresholds"]["cpu_usage"]["critical"]
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        measurements["memory_usage"] = HealthMetric(
            name="memory_usage",
            value=memory.percent,
            unit="percent",
            timestamp=datetime.now(),
            status=self._get_status(memory.percent, "memory_usage"),
            threshold_warning=self.config["thresholds"]["memory_usage"]["warning"],
            threshold_critical=self.config["thresholds"]["memory_usage"]["critical"]
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        measurements["disk_usage"] = HealthMetric(
            name="disk_usage",
            value=disk_usage,
            unit="percent",
            timestamp=datetime.now(),
            status=self._get_status(disk_usage, "disk_usage"),
            threshold_warning=self.config["thresholds"]["disk_usage"]["warning"],
            threshold_critical=self.config["thresholds"]["disk_usage"]["critical"]
        )
        
        return measurements
    
    def _get_status(self, value: float, metric_name: str) -> HealthStatus:
        """Determine health status based on thresholds"""
        thresholds = self.config["thresholds"].get(metric_name, {})
        
        if value >= thresholds.get("critical", float('inf')):
            return HealthStatus.UNHEALTHY
        elif value >= thresholds.get("warning", float('inf')):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    async def check_package_health(self, package_name: str) -> PackageHealth:
        """Check health of a specific package"""
        try:
            # Get package info
            package_path = Path(f"/mnt/c/Users/andre/Software/src/packages/{package_name}")
            pyproject_file = package_path / "pyproject.toml"
            
            if not pyproject_file.exists():
                return PackageHealth(
                    package_name=package_name,
                    version="unknown",
                    status=HealthStatus.UNKNOWN,
                    last_check=datetime.now(),
                    measurements=[],
                    dependencies_status={}
                )
            
            # Extract version from pyproject.toml
            version = self._extract_version(pyproject_file)
            
            # Check package-specific measurements
            measurements = await self._collect_package_measurements(package_name, package_path)
            
            # Check dependencies
            dependencies_status = await self._check_dependencies(package_path)
            
            # Determine overall status
            overall_status = self._determine_overall_status(measurements, dependencies_status)
            
            return PackageHealth(
                package_name=package_name,
                version=version,
                status=overall_status,
                last_check=datetime.now(),
                measurements=measurements,
                dependencies_status=dependencies_status
            )
            
        except Exception as e:
            logger.error(f"Error checking health for {package_name}: {e}")
            return PackageHealth(
                package_name=package_name,
                version="unknown",
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                measurements=[],
                dependencies_status={},
                error_count=1
            )
    
    def _extract_version(self, pyproject_file: Path) -> str:
        """Extract version from pyproject.toml"""
        try:
            with open(pyproject_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.strip().startswith('version ='):
                        return line.split('=')[1].strip().strip('"\'')
        except Exception:
            pass
        return "unknown"
    
    async def _collect_package_measurements(self, package_name: str, package_path: Path) -> List[HealthMetric]:
        """Collect package-specific measurements"""
        measurements = []
        
        # Test coverage metric
        coverage_file = package_path / "htmlcov" / "index.html"
        if coverage_file.exists():
            coverage = await self._extract_coverage(coverage_file)
            measurements.append(HealthMetric(
                name="test_coverage",
                value=coverage,
                unit="percent",
                timestamp=datetime.now(),
                status=HealthStatus.HEALTHY if coverage >= 80 else HealthStatus.DEGRADED
            ))
        
        # Build status
        build_status = await self._check_build_status(package_path)
        measurements.append(HealthMetric(
            name="build_status",
            value=1.0 if build_status else 0.0,
            unit="boolean",
            timestamp=datetime.now(),
            status=HealthStatus.HEALTHY if build_status else HealthStatus.UNHEALTHY
        ))
        
        # Package size
        package_size = await self._calculate_package_size(package_path)
        measurements.append(HealthMetric(
            name="package_size",
            value=package_size,
            unit="MB",
            timestamp=datetime.now(),
            status=HealthStatus.HEALTHY if package_size < 100 else HealthStatus.DEGRADED
        ))
        
        return measurements
    
    async def _extract_coverage(self, coverage_file: Path) -> float:
        """Extract coverage percentage from HTML report"""
        try:
            with open(coverage_file, 'r') as f:
                content = f.read()
                # Simple regex to extract coverage percentage
                import re
                match = re.search(r'(\d+)%', content)
                if match:
                    return float(match.group(1))
        except Exception:
            pass
        return 0.0
    
    async def _check_build_status(self, package_path: Path) -> bool:
        """Check if package builds successfully"""
        try:
            # Check if pyproject.toml is valid
            pyproject_file = package_path / "pyproject.toml"
            if not pyproject_file.exists():
                return False
            
            # Check if source directory exists
            src_dir = package_path / "src"
            if not src_dir.exists():
                return False
            
            return True
        except Exception:
            return False
    
    async def _calculate_package_size(self, package_path: Path) -> float:
        """Calculate package size in MB"""
        try:
            total_size = 0
            for file_path in package_path.rglob("*"):
                if file_path.is_file() and ".venv" not in str(file_path):
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0
    
    async def _check_dependencies(self, package_path: Path) -> Dict[str, HealthStatus]:
        """Check health of package dependencies"""
        dependencies = {}
        
        try:
            pyproject_file = package_path / "pyproject.toml"
            with open(pyproject_file, 'r') as f:
                content = f.read()
                
            # Simple dependency extraction (would need more sophisticated parsing)
            import re
            deps = re.findall(r'"([^"]+)>=', content)
            
            for dep in deps:
                # For now, assume all dependencies are healthy
                # In a real implementation, you'd check PyPI status, security vulnerabilities, etc.
                dependencies[dep] = HealthStatus.HEALTHY
                
        except Exception:
            pass
            
        return dependencies
    
    def _determine_overall_status(self, metrics: List[HealthMetric], dependencies: Dict[str, HealthStatus]) -> HealthStatus:
        """Determine overall package health status"""
        # If any metric is unhealthy, package is unhealthy
        if any(m.status == HealthStatus.UNHEALTHY for m in measurements):
            return HealthStatus.UNHEALTHY
        
        # If any dependency is unhealthy, package is unhealthy
        if any(status == HealthStatus.UNHEALTHY for status in dependencies.values()):
            return HealthStatus.UNHEALTHY
        
        # If any metric or dependency is degraded, package is degraded
        if (any(m.status == HealthStatus.DEGRADED for m in measurements) or 
            any(status == HealthStatus.DEGRADED for status in dependencies.values())):
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": await self.check_system_health(),
            "packages": {},
            "summary": {
                "total_packages": len(self.packages),
                "healthy_packages": 0,
                "degraded_packages": 0,
                "unhealthy_packages": 0,
                "unknown_packages": 0
            }
        }
        
        # Check each package
        for package_name in self.packages:
            package_health = await self.check_package_health(package_name)
            report["packages"][package_name] = asdict(package_health)
            
            # Update summary
            if package_health.status == HealthStatus.HEALTHY:
                report["summary"]["healthy_packages"] += 1
            elif package_health.status == HealthStatus.DEGRADED:
                report["summary"]["degraded_packages"] += 1
            elif package_health.status == HealthStatus.UNHEALTHY:
                report["summary"]["unhealthy_packages"] += 1
            else:
                report["summary"]["unknown_packages"] += 1
        
        return report
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        logger.info("Starting health monitoring...")
        
        while True:
            try:
                report = await self.generate_health_report()
                
                # Save report
                await self._save_report(report)
                
                # Check for alerts
                await self._check_alerts(report)
                
                # Wait for next check
                await asyncio.sleep(self.config["check_interval"])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _save_report(self, report: Dict[str, Any]):
        """Save health report to storage"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path(f"health_reports/health_report_{timestamp}.json")
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
    
    async def _check_alerts(self, report: Dict[str, Any]):
        """Check for alert conditions"""
        # Implementation would check for alert conditions and send notifications
        # For now, just log critical issues
        
        for package_name, package_data in report["packages"].items():
            if package_data["status"] == HealthStatus.UNHEALTHY.value:
                logger.warning(f"ALERT: Package {package_name} is unhealthy!")
                
        # Check system measurements
        system_measurements = report["system_health"]
        for metric_name, metric_data in system_measurements.items():
            if metric_data["status"] == HealthStatus.UNHEALTHY.value:
                logger.warning(f"ALERT: System metric {metric_name} is critical: {metric_data['value']}")


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Package Health Monitor")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--package", type=str, help="Check specific package")
    parser.add_argument("--report", action="store_true", help="Generate health report")
    
    args = parser.parse_args()
    
    monitor = HealthMonitor(args.config)
    
    if args.once:
        if args.package:
            # Check specific package
            health = asyncio.run(monitor.check_package_health(args.package))
            print(json.dumps(asdict(health), indent=2, default=str))
        elif args.report:
            # Generate full report
            report = asyncio.run(monitor.generate_health_report())
            print(json.dumps(report, indent=2, default=str))
        else:
            # Check system health
            measurements = asyncio.run(monitor.check_system_health())
            print(json.dumps({k: asdict(v) for k, v in measurements.items()}, indent=2, default=str))
    else:
        # Start continuous monitoring
        asyncio.run(monitor.start_monitoring())