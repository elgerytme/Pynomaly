#!/usr/bin/env python3
"""
Production deployment validation script for anomaly_detection.
This script deploys the system and validates all components are working correctly.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceCheck:
    """Service health check configuration."""

    name: str
    url: str
    expected_status: int = 200
    timeout: int = 30
    required_fields: list[str] = None

    def __post_init__(self):
        if self.required_fields is None:
            self.required_fields = []


@dataclass
class ValidationResult:
    """Validation result data structure."""

    service: str
    status: str
    message: str
    timestamp: datetime
    details: dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class DeploymentValidator:
    """Main deployment validation orchestrator."""

    def __init__(self):
        """Initialize deployment validator."""
        self.base_url = "http://localhost"
        self.session = self._create_session()
        self.results: list[ValidationResult] = []

        # Define service checks
        self.service_checks = [
            ServiceCheck(
                name="API Health Check",
                url=f"{self.base_url}/health",
                required_fields=["status", "timestamp", "version"],
            ),
            ServiceCheck(
                name="API Documentation",
                url=f"{self.base_url}/docs",
                expected_status=200,
            ),
            ServiceCheck(
                name="API OpenAPI Spec",
                url=f"{self.base_url}/openapi.json",
                required_fields=["info", "paths"],
            ),
            ServiceCheck(
                name="Grafana Dashboard",
                url=f"{self.base_url}:3000/api/health",
                required_fields=["database", "version"],
            ),
            ServiceCheck(
                name="Prometheus Metrics",
                url=f"{self.base_url}:9090/api/v1/targets",
                required_fields=["data"],
            ),
        ]

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry configuration."""
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    async def deploy_system(self) -> bool:
        """Deploy the production system."""
        logger.info("🚀 Starting production deployment...")

        try:
            # Check if environment file exists
            env_file = ".env"
            if not os.path.exists(env_file):
                logger.warning(
                    f"Environment file {env_file} not found, creating default..."
                )
                self._create_default_env()

            # Stop any existing containers
            logger.info("Stopping existing containers...")
            subprocess.run(
                ["docker-compose", "-f", "docker-compose.simple.yml", "down", "-v"],
                check=False,
                capture_output=True,
            )

            # Pull latest images
            logger.info("Pulling latest Docker images...")
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.simple.yml", "pull"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Image pull had issues: {result.stderr}")

            # Build and start services
            logger.info("Building and starting services...")
            result = subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "docker-compose.simple.yml",
                    "up",
                    "-d",
                    "--build",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Deployment failed: {result.stderr}")
                return False

            logger.info("✅ Services started successfully")

            # Wait for services to initialize
            logger.info("Waiting for services to initialize...")
            await asyncio.sleep(30)

            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False

    def _create_default_env(self):
        """Create default environment file."""
        env_content = """# Production Environment Variables
POSTGRES_PASSWORD=secure_prod_password_123
DATABASE_URL=postgresql://anomaly_detection:secure_prod_password_123@postgres:5432/anomaly_detection_prod
REDIS_URL=redis://redis-cluster:6379
SECRET_KEY=your_very_secure_secret_key_change_this_in_production
GRAFANA_PASSWORD=admin_password_123
ENVIRONMENT=production

# Optional: External Services
# ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
# SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
"""

        with open(".env", "w") as f:
            f.write(env_content)

        logger.info("Created default .env file")

    async def validate_service_health(self) -> bool:
        """Validate health of all services."""
        logger.info("🔍 Validating service health...")

        all_healthy = True

        for check in self.service_checks:
            try:
                logger.info(f"Checking {check.name}...")

                response = self.session.get(check.url, timeout=check.timeout)

                if response.status_code != check.expected_status:
                    self._record_result(
                        service=check.name,
                        status="FAILED",
                        message=f"Expected status {check.expected_status}, got {response.status_code}",
                        details={"response": response.text[:500]},
                    )
                    all_healthy = False
                    continue

                # Check JSON response structure
                if check.required_fields and response.headers.get(
                    "content-type", ""
                ).startswith("application/json"):
                    try:
                        data = response.json()
                        missing_fields = [
                            field
                            for field in check.required_fields
                            if field not in data
                        ]

                        if missing_fields:
                            self._record_result(
                                service=check.name,
                                status="FAILED",
                                message=f"Missing required fields: {missing_fields}",
                                details={"response": data},
                            )
                            all_healthy = False
                            continue
                    except json.JSONDecodeError:
                        self._record_result(
                            service=check.name,
                            status="FAILED",
                            message="Invalid JSON response",
                            details={"response": response.text[:500]},
                        )
                        all_healthy = False
                        continue

                self._record_result(
                    service=check.name,
                    status="HEALTHY",
                    message="Service is healthy",
                    details={"response_time": response.elapsed.total_seconds()},
                )

            except requests.exceptions.RequestException as e:
                self._record_result(
                    service=check.name,
                    status="FAILED",
                    message=f"Connection error: {str(e)}",
                    details={"error_type": type(e).__name__},
                )
                all_healthy = False

            except Exception as e:
                self._record_result(
                    service=check.name,
                    status="ERROR",
                    message=f"Unexpected error: {str(e)}",
                    details={"error_type": type(e).__name__},
                )
                all_healthy = False

        return all_healthy

    async def validate_docker_services(self) -> bool:
        """Validate Docker services are running correctly."""
        logger.info("🐳 Validating Docker services...")

        try:
            # Check service status
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.simple.yml", "ps"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self._record_result(
                    service="Docker Services",
                    status="FAILED",
                    message="Could not get service status",
                    details={"error": result.stderr},
                )
                return False

            # Parse service status
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            services_status = {}

            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:
                        service_name = parts[0]
                        status = parts[5] if len(parts) > 5 else "unknown"
                        services_status[service_name] = status

            # Check expected services
            expected_services = [
                "anomaly_detection-api",
                "anomaly_detection-streaming",
                "postgres",
                "redis-cluster",
                "prometheus",
                "grafana",
                "nginx",
            ]

            all_running = True
            for service in expected_services:
                service_found = any(service in name for name in services_status.keys())
                if not service_found:
                    self._record_result(
                        service=f"Docker Service: {service}",
                        status="FAILED",
                        message="Service not found in docker-compose output",
                        details={"available_services": list(services_status.keys())},
                    )
                    all_running = False
                else:
                    # Find the actual service name
                    actual_service = next(
                        name for name in services_status.keys() if service in name
                    )
                    status = services_status[actual_service]

                    if "Up" not in status:
                        self._record_result(
                            service=f"Docker Service: {service}",
                            status="FAILED",
                            message=f"Service status: {status}",
                            details={"full_status": status},
                        )
                        all_running = False
                    else:
                        self._record_result(
                            service=f"Docker Service: {service}",
                            status="HEALTHY",
                            message="Service is running",
                            details={"status": status},
                        )

            return all_running

        except Exception as e:
            self._record_result(
                service="Docker Services",
                status="ERROR",
                message=f"Error checking Docker services: {str(e)}",
                details={"error_type": type(e).__name__},
            )
            return False

    async def validate_database_connectivity(self) -> bool:
        """Validate database connectivity and basic operations."""
        logger.info("🗄️ Validating database connectivity...")

        try:
            # Test database connection
            result = subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "docker-compose.simple.yml",
                    "exec",
                    "-T",
                    "postgres",
                    "pg_isready",
                    "-U",
                    "anomaly_detection",
                    "-d",
                    "anomaly_detection_prod",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self._record_result(
                    service="Database Connectivity",
                    status="FAILED",
                    message="Database not ready",
                    details={"error": result.stderr},
                )
                return False

            # Test basic query
            result = subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "docker-compose.simple.yml",
                    "exec",
                    "-T",
                    "postgres",
                    "psql",
                    "-U",
                    "anomaly_detection",
                    "-d",
                    "anomaly_detection_prod",
                    "-c",
                    "SELECT version();",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self._record_result(
                    service="Database Query",
                    status="FAILED",
                    message="Could not execute test query",
                    details={"error": result.stderr},
                )
                return False

            self._record_result(
                service="Database",
                status="HEALTHY",
                message="Database is connected and responsive",
                details={"version_info": result.stdout.strip()},
            )

            return True

        except Exception as e:
            self._record_result(
                service="Database",
                status="ERROR",
                message=f"Database validation error: {str(e)}",
                details={"error_type": type(e).__name__},
            )
            return False

    async def validate_redis_connectivity(self) -> bool:
        """Validate Redis connectivity."""
        logger.info("🔴 Validating Redis connectivity...")

        try:
            result = subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    "docker-compose.simple.yml",
                    "exec",
                    "-T",
                    "redis-cluster",
                    "redis-cli",
                    "ping",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0 or "PONG" not in result.stdout:
                self._record_result(
                    service="Redis",
                    status="FAILED",
                    message="Redis not responding to ping",
                    details={"output": result.stdout, "error": result.stderr},
                )
                return False

            self._record_result(
                service="Redis",
                status="HEALTHY",
                message="Redis is connected and responsive",
                details={"response": result.stdout.strip()},
            )

            return True

        except Exception as e:
            self._record_result(
                service="Redis",
                status="ERROR",
                message=f"Redis validation error: {str(e)}",
                details={"error_type": type(e).__name__},
            )
            return False

    async def validate_api_functionality(self) -> bool:
        """Validate API functionality with sample requests."""
        logger.info("🔌 Validating API functionality...")

        try:
            # Test anomaly detection endpoint
            test_data = {
                "data": [
                    {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
                    {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0},
                    {
                        "feature1": 100.0,
                        "feature2": 200.0,
                        "feature3": 300.0,
                    },  # Potential anomaly
                ]
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/detect", json=test_data, timeout=30
            )

            if response.status_code != 200:
                self._record_result(
                    service="API Detection Endpoint",
                    status="FAILED",
                    message=f"Detection endpoint returned {response.status_code}",
                    details={"response": response.text},
                )
                return False

            try:
                result = response.json()
                required_fields = ["detection_id", "anomalies_detected", "timestamp"]
                missing_fields = [
                    field for field in required_fields if field not in result
                ]

                if missing_fields:
                    self._record_result(
                        service="API Detection Endpoint",
                        status="FAILED",
                        message=f"Missing required fields: {missing_fields}",
                        details={"response": result},
                    )
                    return False

                self._record_result(
                    service="API Detection Endpoint",
                    status="HEALTHY",
                    message="Detection endpoint is functional",
                    details={"detection_result": result},
                )

                return True

            except json.JSONDecodeError:
                self._record_result(
                    service="API Detection Endpoint",
                    status="FAILED",
                    message="Invalid JSON response from detection endpoint",
                    details={"response": response.text},
                )
                return False

        except Exception as e:
            self._record_result(
                service="API Detection Endpoint",
                status="ERROR",
                message=f"API validation error: {str(e)}",
                details={"error_type": type(e).__name__},
            )
            return False

    def _record_result(
        self, service: str, status: str, message: str, details: dict[str, Any] = None
    ):
        """Record validation result."""
        result = ValidationResult(
            service=service,
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details or {},
        )
        self.results.append(result)

        # Log result
        if status == "HEALTHY":
            logger.info(f"✅ {service}: {message}")
        elif status == "FAILED":
            logger.error(f"❌ {service}: {message}")
        elif status == "ERROR":
            logger.error(f"🔥 {service}: {message}")

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        healthy_count = sum(1 for r in self.results if r.status == "HEALTHY")
        failed_count = sum(1 for r in self.results if r.status == "FAILED")
        error_count = sum(1 for r in self.results if r.status == "ERROR")

        report = {
            "deployment_validation": {
                "timestamp": datetime.now().isoformat(),
                "total_checks": len(self.results),
                "healthy": healthy_count,
                "failed": failed_count,
                "errors": error_count,
                "success_rate": healthy_count / len(self.results) * 100
                if self.results
                else 0,
                "overall_status": "HEALTHY"
                if failed_count == 0 and error_count == 0
                else "FAILED",
            },
            "service_results": [
                {
                    "service": r.service,
                    "status": r.status,
                    "message": r.message,
                    "timestamp": r.timestamp.isoformat(),
                    "details": r.details,
                }
                for r in self.results
            ],
        }

        return report

    def save_report(self, report: dict[str, Any]):
        """Save validation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deployment_validation_report_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Validation report saved to {filename}")

    def print_summary(self, report: dict[str, Any]):
        """Print validation summary."""
        validation = report["deployment_validation"]

        print("\n" + "=" * 60)
        print("🚀 anomaly_detection DEPLOYMENT VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {validation['timestamp']}")
        print(f"Total Checks: {validation['total_checks']}")
        print(f"✅ Healthy: {validation['healthy']}")
        print(f"❌ Failed: {validation['failed']}")
        print(f"🔥 Errors: {validation['errors']}")
        print(f"Success Rate: {validation['success_rate']:.1f}%")
        print(f"Overall Status: {validation['overall_status']}")

        if validation["failed"] > 0 or validation["errors"] > 0:
            print("\n⚠️  ISSUES FOUND:")
            for result in report["service_results"]:
                if result["status"] in ["FAILED", "ERROR"]:
                    print(f"  - {result['service']}: {result['message']}")

        print("\n" + "=" * 60)

        if validation["overall_status"] == "HEALTHY":
            print("🎉 DEPLOYMENT SUCCESSFUL! All services are healthy.")
        else:
            print("❌ DEPLOYMENT ISSUES DETECTED! Please review the report.")

        print("=" * 60)


async def main():
    """Main deployment validation workflow."""
    validator = DeploymentValidator()

    try:
        # Step 1: Deploy the system
        deployment_success = await validator.deploy_system()
        if not deployment_success:
            logger.error("❌ Deployment failed, skipping validation")
            return False

        # Step 2: Validate Docker services
        docker_healthy = await validator.validate_docker_services()

        # Step 3: Validate database connectivity
        db_healthy = await validator.validate_database_connectivity()

        # Step 4: Validate Redis connectivity
        redis_healthy = await validator.validate_redis_connectivity()

        # Step 5: Validate service health endpoints
        services_healthy = await validator.validate_service_health()

        # Step 6: Validate API functionality
        api_healthy = await validator.validate_api_functionality()

        # Generate and save report
        report = validator.generate_report()
        validator.save_report(report)
        validator.print_summary(report)

        # Overall success
        overall_success = all(
            [
                deployment_success,
                docker_healthy,
                db_healthy,
                redis_healthy,
                services_healthy,
                api_healthy,
            ]
        )

        return overall_success

    except Exception as e:
        logger.error(f"Validation process failed: {e}")
        return False


if __name__ == "__main__":
    # Run the validation
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
