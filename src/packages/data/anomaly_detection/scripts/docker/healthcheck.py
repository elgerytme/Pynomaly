#!/usr/bin/env python3
"""Health check script for anomaly detection service."""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional
import logging

try:
    import httpx
    import psutil
    HTTP_CLIENT_AVAILABLE = True
except ImportError:
    HTTP_CLIENT_AVAILABLE = False

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('healthcheck')


class HealthChecker:
    """Comprehensive health checker for anomaly detection service."""
    
    def __init__(
        self,
        api_host: str = "localhost",
        api_port: int = 8000,
        web_port: int = 8080,
        timeout: float = 10.0,
        gpu_check: bool = False,
        dev_mode: bool = False
    ):
        """Initialize health checker.
        
        Args:
            api_host: API server host
            api_port: API server port
            web_port: Web interface port
            timeout: Request timeout in seconds
            gpu_check: Whether to check GPU health
            dev_mode: Whether running in development mode
        """
        self.api_host = api_host
        self.api_port = api_port
        self.web_port = web_port
        self.timeout = timeout
        self.gpu_check = gpu_check
        self.dev_mode = dev_mode
        
        self.checks = {
            "system": self._check_system_health,
            "api": self._check_api_health,
            "web": self._check_web_health,
            "database": self._check_database_health,
            "redis": self._check_redis_health,
            "models": self._check_models_health,
        }
        
        if self.gpu_check:
            self.checks["gpu"] = self._check_gpu_health
        
        if self.dev_mode:
            self.checks["development"] = self._check_development_health
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Dictionary with check results
        """
        results = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "checks": {},
            "summary": {
                "total_checks": len(self.checks),
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        
        for check_name, check_func in self.checks.items():
            try:
                logger.info(f"Running {check_name} health check...")
                check_result = await check_func()
                results["checks"][check_name] = check_result
                
                if check_result["status"] == "healthy":
                    results["summary"]["passed"] += 1
                elif check_result["status"] == "warning":
                    results["summary"]["warnings"] += 1
                else:
                    results["summary"]["failed"] += 1
                    results["overall_status"] = "unhealthy"
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed with exception: {e}")
                results["checks"][check_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                }
                results["summary"]["failed"] += 1
                results["overall_status"] = "unhealthy"
        
        # Set overall status based on results
        if results["summary"]["failed"] > 0:
            results["overall_status"] = "unhealthy"
        elif results["summary"]["warnings"] > 0:
            results["overall_status"] = "warning"
        
        return results
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Process count
            process_count = len(psutil.pids())
            
            # Determine status
            status = "healthy"
            warnings = []
            
            if cpu_percent > 90:
                status = "warning"
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory_percent > 90:
                status = "warning"
                warnings.append(f"High memory usage: {memory_percent}%")
            
            if disk_percent > 90:
                status = "warning"
                warnings.append(f"High disk usage: {disk_percent}%")
            
            return {
                "status": status,
                "metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "process_count": process_count
                },
                "warnings": warnings,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API server health."""
        if not HTTP_CLIENT_AVAILABLE:
            return {
                "status": "warning",
                "message": "HTTP client not available for API health check",
                "timestamp": time.time()
            }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Check health endpoint
                health_url = f"http://{self.api_host}:{self.api_port}/health"
                response = await client.get(health_url)
                
                if response.status_code == 200:
                    health_data = response.json()
                    
                    # Check API-specific metrics
                    metrics_url = f"http://{self.api_host}:{self.api_port}/metrics"
                    try:
                        metrics_response = await client.get(metrics_url)
                        metrics_available = metrics_response.status_code == 200
                    except:
                        metrics_available = False
                    
                    return {
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "health_data": health_data,
                        "metrics_available": metrics_available,
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"API returned status code {response.status_code}",
                        "response_time": response.elapsed.total_seconds(),
                        "timestamp": time.time()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _check_web_health(self) -> Dict[str, Any]:
        """Check web interface health."""
        if not HTTP_CLIENT_AVAILABLE:
            return {
                "status": "warning",
                "message": "HTTP client not available for web health check",
                "timestamp": time.time()
            }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                web_url = f"http://{self.api_host}:{self.web_port}/"
                response = await client.get(web_url)
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "content_length": len(response.content),
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"Web interface returned status code {response.status_code}",
                        "timestamp": time.time()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity."""
        import os
        
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return {
                "status": "warning",
                "message": "No database configured",
                "timestamp": time.time()
            }
        
        if not POSTGRES_AVAILABLE:
            return {
                "status": "warning",
                "message": "PostgreSQL client not available",
                "timestamp": time.time()
            }
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path[1:] if parsed.path else 'postgres'
            )
            
            # Test query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                "status": "healthy",
                "message": "Database connection successful",
                "test_query_result": result[0] if result else None,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        import os
        
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            return {
                "status": "warning",
                "message": "No Redis configured",
                "timestamp": time.time()
            }
        
        if not REDIS_AVAILABLE:
            return {
                "status": "warning",
                "message": "Redis client not available",
                "timestamp": time.time()
            }
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(redis_url)
            
            r = redis.Redis(
                host=parsed.hostname,
                port=parsed.port or 6379,
                password=parsed.password,
                decode_responses=True
            )
            
            # Test ping
            ping_result = r.ping()
            
            # Get info
            info = r.info()
            
            return {
                "status": "healthy",
                "ping_result": ping_result,
                "connected_clients": info.get('connected_clients', 0),
                "used_memory": info.get('used_memory_human', 'unknown'),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _check_models_health(self) -> Dict[str, Any]:
        """Check model availability and health."""
        try:
            from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
            
            repo = ModelRepository()
            
            # List available models
            models = repo.list_models()
            
            # Check if default models are available
            required_models = ['default_isolation_forest']
            missing_models = []
            
            available_model_names = [model.get('name', '') for model in models]
            
            for required_model in required_models:
                if required_model not in available_model_names:
                    missing_models.append(required_model)
            
            status = "healthy" if not missing_models else "warning"
            
            return {
                "status": status,
                "total_models": len(models),
                "available_models": available_model_names,
                "missing_required_models": missing_models,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health and availability."""
        try:
            import subprocess
            import json
            
            # Try nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            gpu_info = {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "temperature": int(parts[2]) if parts[2] != '[Not Supported]' else None,
                                "utilization": int(parts[3]) if parts[3] != '[Not Supported]' else None,
                                "memory_used": int(parts[4]) if parts[4] != '[Not Supported]' else None,
                                "memory_total": int(parts[5]) if parts[5] != '[Not Supported]' else None
                            }
                            gpus.append(gpu_info)
                
                # Check for warnings
                warnings = []
                for gpu in gpus:
                    if gpu["temperature"] and gpu["temperature"] > 85:
                        warnings.append(f"GPU {gpu['index']} temperature high: {gpu['temperature']}°C")
                    if gpu["utilization"] and gpu["utilization"] > 95:
                        warnings.append(f"GPU {gpu['index']} utilization high: {gpu['utilization']}%")
                
                status = "warning" if warnings else "healthy"
                
                return {
                    "status": status,
                    "gpu_count": len(gpus),
                    "gpus": gpus,
                    "warnings": warnings,
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "warning",
                    "message": "nvidia-smi not available or no GPUs detected",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            return {
                "status": "warning",
                "error": str(e),
                "message": "GPU health check failed",
                "timestamp": time.time()
            }
    
    async def _check_development_health(self) -> Dict[str, Any]:
        """Check development-specific health."""
        try:
            import os
            
            # Check development directories
            dev_dirs = ['logs', 'data', 'models', 'temp', '.cache', 'notebooks']
            missing_dirs = []
            
            for dir_name in dev_dirs:
                if not os.path.exists(dir_name):
                    missing_dirs.append(dir_name)
            
            # Check if Jupyter is running (port 8888)
            jupyter_running = False
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 8888))
                jupyter_running = result == 0
                sock.close()
            except:
                pass
            
            # Check if TensorBoard is running (port 6006)
            tensorboard_running = False
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 6006))
                tensorboard_running = result == 0
                sock.close()
            except:
                pass
            
            status = "warning" if missing_dirs else "healthy"
            
            return {
                "status": status,
                "missing_directories": missing_dirs,
                "jupyter_running": jupyter_running,
                "tensorboard_running": tensorboard_running,
                "environment": os.getenv('ENVIRONMENT', 'unknown'),
                "debug_mode": os.getenv('DEBUG', 'false').lower() == 'true',
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


async def main():
    """Main health check function."""
    parser = argparse.ArgumentParser(description="Anomaly Detection Service Health Check")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument("--web-port", type=int, default=8080, help="Web port")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout")
    parser.add_argument("--gpu", action="store_true", help="Include GPU health checks")
    parser.add_argument("--dev", action="store_true", help="Development mode checks")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    checker = HealthChecker(
        api_host=args.host,
        api_port=args.api_port,
        web_port=args.web_port,
        timeout=args.timeout,
        gpu_check=args.gpu,
        dev_mode=args.dev
    )
    
    results = await checker.run_all_checks()
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Checks: {results['summary']['passed']} passed, {results['summary']['failed']} failed, {results['summary']['warnings']} warnings")
        print()
        
        for check_name, check_result in results['checks'].items():
            status_emoji = "✅" if check_result['status'] == 'healthy' else "⚠️" if check_result['status'] == 'warning' else "❌"
            print(f"{status_emoji} {check_name}: {check_result['status']}")
            
            if 'error' in check_result:
                print(f"   Error: {check_result['error']}")
            if 'warnings' in check_result and check_result['warnings']:
                for warning in check_result['warnings']:
                    print(f"   Warning: {warning}")
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] == 'healthy' else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())