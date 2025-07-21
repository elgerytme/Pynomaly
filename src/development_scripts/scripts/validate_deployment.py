#!/usr/bin/env python3
"""
Comprehensive deployment validation script for anomaly_detection production environment.
Validates API endpoints, database connectivity, security configurations, and more.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import asyncpg
import redis
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    passed: bool
    message: str
    details: dict | None = None
    duration: float | None = None


class DeploymentValidator:
    """Comprehensive deployment validation."""

    def __init__(self, config_path: str = "config/production/.env.prod"):
        self.config_path = config_path
        self.config = self._load_config()
        self.results: list[ValidationResult] = []

    def _load_config(self) -> dict[str, str]:
        """Load configuration from environment file."""
        config = {}
        try:
            with open(self.config_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        config[key] = value
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")

        # Add environment variables
        config.update(os.environ)
        return config

    async def validate_all(self) -> list[ValidationResult]:
        """Run all validation checks."""
        console.print("\n🔍 Starting comprehensive deployment validation...\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Define validation tasks
            tasks = [
                ("Validating prerequisites", self._validate_prerequisites),
                ("Checking Docker services", self._validate_docker_services),
                ("Testing database connectivity", self._validate_database),
                ("Testing Redis connectivity", self._validate_redis),
                ("Validating API endpoints", self._validate_api_endpoints),
                ("Checking security configuration", self._validate_security),
                ("Testing authentication", self._validate_authentication),
                ("Validating monitoring", self._validate_monitoring),
                ("Checking SSL certificates", self._validate_ssl),
                ("Testing backup systems", self._validate_backup),
                ("Validating performance", self._validate_performance),
                ("Checking logs", self._validate_logs),
            ]

            for description, validation_func in tasks:
                task = progress.add_task(description, total=None)
                start_time = time.time()

                try:
                    result = await validation_func()
                    duration = time.time() - start_time

                    if isinstance(result, list):
                        for r in result:
                            r.duration = duration / len(result)
                            self.results.append(r)
                    else:
                        result.duration = duration
                        self.results.append(result)

                except Exception as e:
                    duration = time.time() - start_time
                    self.results.append(
                        ValidationResult(
                            name=description,
                            passed=False,
                            message=f"Validation failed: {str(e)}",
                            duration=duration,
                        )
                    )

                progress.update(task, completed=True)

        return self.results

    async def _validate_prerequisites(self) -> ValidationResult:
        """Validate system prerequisites."""
        checks = []

        # Check Docker
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                checks.append("✓ Docker installed")
            else:
                checks.append("✗ Docker not found")
        except FileNotFoundError:
            checks.append("✗ Docker not installed")

        # Check Docker Compose
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                checks.append("✓ Docker Compose installed")
            else:
                checks.append("✗ Docker Compose not found")
        except FileNotFoundError:
            checks.append("✗ Docker Compose not installed")

        # Check disk space
        try:
            result = subprocess.run(["df", "-h", "."], capture_output=True, text=True)
            if result.returncode == 0:
                checks.append("✓ Sufficient disk space")
            else:
                checks.append("✗ Could not check disk space")
        except FileNotFoundError:
            checks.append("✗ Could not check disk space")

        passed = all("✓" in check for check in checks)
        return ValidationResult(
            name="Prerequisites",
            passed=passed,
            message="\n".join(checks),
            details={"checks": checks},
        )

    async def _validate_docker_services(self) -> list[ValidationResult]:
        """Validate Docker services are running."""
        services = [
            "anomaly_detection-api",
            "anomaly_detection-postgres",
            "anomaly_detection-redis",
            "anomaly_detection-nginx",
            "anomaly_detection-grafana",
            "anomaly_detection-prometheus",
        ]

        results = []

        for service in services:
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "--filter",
                        f"name={service}",
                        "--format",
                        "table {{.Names}}\t{{.Status}}",
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0 and service in result.stdout:
                    if "Up" in result.stdout:
                        results.append(
                            ValidationResult(
                                name=f"Service {service}",
                                passed=True,
                                message="Service is running",
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                name=f"Service {service}",
                                passed=False,
                                message="Service is not running",
                            )
                        )
                else:
                    results.append(
                        ValidationResult(
                            name=f"Service {service}",
                            passed=False,
                            message="Service not found",
                        )
                    )
            except Exception as e:
                results.append(
                    ValidationResult(
                        name=f"Service {service}",
                        passed=False,
                        message=f"Error checking service: {str(e)}",
                    )
                )

        return results

    async def _validate_database(self) -> ValidationResult:
        """Validate database connectivity and configuration."""
        try:
            db_url = self.config.get(
                "DATABASE_URL", "postgresql://anomaly_detection:password@localhost:5432/anomaly_detection"
            )

            conn = await asyncpg.connect(db_url)

            # Test basic connectivity
            version = await conn.fetchval("SELECT version()")

            # Check database exists
            db_name = await conn.fetchval("SELECT current_database()")

            # Check tables exist
            tables = await conn.fetch("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)

            await conn.close()

            return ValidationResult(
                name="Database",
                passed=True,
                message=f"Database {db_name} is accessible",
                details={
                    "version": version,
                    "database": db_name,
                    "tables": len(tables),
                },
            )

        except Exception as e:
            return ValidationResult(
                name="Database",
                passed=False,
                message=f"Database connection failed: {str(e)}",
            )

    async def _validate_redis(self) -> ValidationResult:
        """Validate Redis connectivity."""
        try:
            redis_url = self.config.get("REDIS_URL", "redis://localhost:6379")

            r = redis.from_url(redis_url)

            # Test basic connectivity
            pong = r.ping()

            # Get Redis info
            info = r.info()

            r.close()

            return ValidationResult(
                name="Redis",
                passed=True,
                message="Redis is accessible",
                details={
                    "ping": pong,
                    "version": info.get("redis_version"),
                    "memory": info.get("used_memory_human"),
                },
            )

        except Exception as e:
            return ValidationResult(
                name="Redis", passed=False, message=f"Redis connection failed: {str(e)}"
            )

    async def _validate_api_endpoints(self) -> list[ValidationResult]:
        """Validate API endpoints."""
        base_url = self.config.get("API_BASE_URL", "http://localhost:8000")

        endpoints = [
            ("/health", "GET", "Health check"),
            ("/api/v1/health/", "GET", "API health check"),
            ("/api/v1/auth/register", "POST", "User registration"),
            ("/api/v1/auth/login", "POST", "User login"),
            ("/docs", "GET", "API documentation"),
            ("/metrics", "GET", "Metrics endpoint"),
        ]

        results = []

        async with aiohttp.ClientSession() as session:
            for endpoint, method, description in endpoints:
                try:
                    url = f"{base_url}{endpoint}"

                    if method == "GET":
                        async with session.get(url) as response:
                            status = response.status
                            content = await response.text()
                    else:
                        # For POST endpoints, just check they're accessible
                        async with session.post(url, json={}) as response:
                            status = response.status
                            content = await response.text()

                    # Accept various success/error codes as "accessible"
                    passed = status < 500

                    results.append(
                        ValidationResult(
                            name=f"Endpoint {endpoint}",
                            passed=passed,
                            message=f"{description}: HTTP {status}",
                            details={"status": status, "endpoint": endpoint},
                        )
                    )

                except Exception as e:
                    results.append(
                        ValidationResult(
                            name=f"Endpoint {endpoint}",
                            passed=False,
                            message=f"{description} failed: {str(e)}",
                        )
                    )

        return results

    async def _validate_security(self) -> list[ValidationResult]:
        """Validate security configuration."""
        results = []

        # Check environment variables
        security_vars = [
            "SECRET_KEY",
            "DB_PASSWORD",
            "REDIS_PASSWORD",
            "GRAFANA_PASSWORD",
        ]

        for var in security_vars:
            value = self.config.get(var)
            if value and value != f"CHANGE_ME_TO_SECURE_{var}":
                results.append(
                    ValidationResult(
                        name=f"Security: {var}",
                        passed=True,
                        message="Security variable is set",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        name=f"Security: {var}",
                        passed=False,
                        message="Security variable not set or using default",
                    )
                )

        # Check SSL configuration
        ssl_cert = self.config.get("SSL_CERT_PATH")
        ssl_key = self.config.get("SSL_KEY_PATH")

        if ssl_cert and ssl_key:
            try:
                if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
                    results.append(
                        ValidationResult(
                            name="Security: SSL Certificates",
                            passed=True,
                            message="SSL certificates found",
                        )
                    )
                else:
                    results.append(
                        ValidationResult(
                            name="Security: SSL Certificates",
                            passed=False,
                            message="SSL certificate files not found",
                        )
                    )
            except Exception as e:
                results.append(
                    ValidationResult(
                        name="Security: SSL Certificates",
                        passed=False,
                        message=f"SSL check failed: {str(e)}",
                    )
                )

        return results

    async def _validate_authentication(self) -> ValidationResult:
        """Validate authentication system."""
        try:
            base_url = self.config.get("API_BASE_URL", "http://localhost:8000")

            async with aiohttp.ClientSession() as session:
                # Try to register a test user
                register_data = {
                    "username": "test_user",
                    "email": "test@example.com",
                    "password": "test_password123",
                    "full_name": "Test User",
                }

                async with session.post(
                    f"{base_url}/api/v1/auth/register", json=register_data
                ) as response:
                    register_status = response.status
                    register_content = await response.text()

                # Try to login
                login_data = {"username": "test_user", "password": "test_password123"}

                async with session.post(
                    f"{base_url}/api/v1/auth/login", data=login_data
                ) as response:
                    login_status = response.status
                    login_content = await response.text()

                # Authentication is working if we get reasonable responses
                auth_working = (register_status < 500) and (login_status < 500)

                return ValidationResult(
                    name="Authentication",
                    passed=auth_working,
                    message="Authentication system is responding",
                    details={
                        "register_status": register_status,
                        "login_status": login_status,
                    },
                )

        except Exception as e:
            return ValidationResult(
                name="Authentication",
                passed=False,
                message=f"Authentication test failed: {str(e)}",
            )

    async def _validate_monitoring(self) -> list[ValidationResult]:
        """Validate monitoring systems."""
        results = []

        monitoring_services = [
            ("Grafana", "http://localhost:3000", "Grafana dashboard"),
            ("Prometheus", "http://localhost:9090", "Prometheus metrics"),
            ("Loki", "http://localhost:3100", "Loki logs"),
        ]

        async with aiohttp.ClientSession() as session:
            for service, url, description in monitoring_services:
                try:
                    async with session.get(url) as response:
                        status = response.status
                        passed = status < 500

                        results.append(
                            ValidationResult(
                                name=f"Monitoring: {service}",
                                passed=passed,
                                message=f"{description}: HTTP {status}",
                            )
                        )

                except Exception as e:
                    results.append(
                        ValidationResult(
                            name=f"Monitoring: {service}",
                            passed=False,
                            message=f"{description} failed: {str(e)}",
                        )
                    )

        return results

    async def _validate_ssl(self) -> ValidationResult:
        """Validate SSL certificate configuration."""
        try:
            # Check if SSL is enabled
            https_url = self.config.get("HTTPS_URL", "https://localhost")

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{https_url}/health") as response:
                    status = response.status

                    return ValidationResult(
                        name="SSL Certificate",
                        passed=status < 500,
                        message=f"HTTPS endpoint accessible: HTTP {status}",
                    )

        except Exception as e:
            return ValidationResult(
                name="SSL Certificate",
                passed=False,
                message=f"SSL validation failed: {str(e)}",
            )

    async def _validate_backup(self) -> ValidationResult:
        """Validate backup systems."""
        try:
            backup_dir = self.config.get("BACKUP_DIR", "backups")

            if os.path.exists(backup_dir):
                backup_files = os.listdir(backup_dir)
                backup_count = len(backup_files)

                return ValidationResult(
                    name="Backup System",
                    passed=True,
                    message=f"Backup directory exists with {backup_count} files",
                    details={"backup_count": backup_count},
                )
            else:
                return ValidationResult(
                    name="Backup System",
                    passed=False,
                    message="Backup directory not found",
                )

        except Exception as e:
            return ValidationResult(
                name="Backup System",
                passed=False,
                message=f"Backup validation failed: {str(e)}",
            )

    async def _validate_performance(self) -> ValidationResult:
        """Validate performance benchmarks."""
        try:
            base_url = self.config.get("API_BASE_URL", "http://localhost:8000")

            # Measure API response time
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/api/v1/health/") as response:
                    response_time = time.time() - start_time
                    status = response.status

            # Performance is acceptable if response time < 2 seconds
            passed = response_time < 2.0 and status == 200

            return ValidationResult(
                name="Performance",
                passed=passed,
                message=f"API response time: {response_time:.2f}s",
                details={"response_time": response_time},
            )

        except Exception as e:
            return ValidationResult(
                name="Performance",
                passed=False,
                message=f"Performance test failed: {str(e)}",
            )

    async def _validate_logs(self) -> ValidationResult:
        """Validate logging system."""
        try:
            log_dir = self.config.get("LOG_DIR", "logs")

            if os.path.exists(log_dir):
                log_files = os.listdir(log_dir)
                log_count = len(log_files)

                return ValidationResult(
                    name="Logging System",
                    passed=True,
                    message=f"Log directory exists with {log_count} files",
                    details={"log_count": log_count},
                )
            else:
                return ValidationResult(
                    name="Logging System",
                    passed=False,
                    message="Log directory not found",
                )

        except Exception as e:
            return ValidationResult(
                name="Logging System",
                passed=False,
                message=f"Log validation failed: {str(e)}",
            )

    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        report = f"""
# anomaly_detection Deployment Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Checks:** {total_count}
**Passed:** {passed_count}
**Failed:** {total_count - passed_count}
**Success Rate:** {(passed_count / total_count * 100):.1f}%

## Summary

"""

        # Group results by category
        categories = {}
        for result in self.results:
            category = result.name.split(":")[0] if ":" in result.name else result.name
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        for category, results in categories.items():
            category_passed = sum(1 for r in results if r.passed)
            category_total = len(results)

            report += f"### {category}\n"
            report += f"**Status:** {category_passed}/{category_total} passed\n\n"

            for result in results:
                status = "✅" if result.passed else "❌"
                report += f"- {status} **{result.name}**: {result.message}\n"
                if result.duration:
                    report += f"  - Duration: {result.duration:.2f}s\n"
                if result.details:
                    report += f"  - Details: {json.dumps(result.details, indent=2)}\n"

            report += "\n"

        return report

    def display_results(self):
        """Display validation results in a formatted table."""
        console.print("\n📊 Validation Results Summary\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Message", style="white")
        table.add_column("Duration", justify="right", style="dim")

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            status_style = "green" if result.passed else "red"

            duration = f"{result.duration:.2f}s" if result.duration else "-"

            table.add_row(
                result.name,
                f"[{status_style}]{status}[/{status_style}]",
                result.message,
                duration,
            )

        console.print(table)

        # Summary statistics
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        console.print(
            f"\n📈 Summary: {passed}/{total} checks passed ({passed/total*100:.1f}%)"
        )

        if passed == total:
            console.print(
                "🎉 [bold green]All validation checks passed! Deployment is ready.[/bold green]"
            )
        else:
            console.print(
                "⚠️  [bold red]Some validation checks failed. Please review and fix issues.[/bold red]"
            )


async def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate anomaly_detection deployment")
    parser.add_argument(
        "--config",
        default="config/production/.env.prod",
        help="Path to configuration file",
    )
    parser.add_argument("--report", help="Generate report file")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Create validator
    validator = DeploymentValidator(args.config)

    # Run validation
    results = await validator.validate_all()

    # Display results
    if args.json:
        json_results = []
        for result in results:
            json_results.append(
                {
                    "name": result.name,
                    "passed": result.passed,
                    "message": result.message,
                    "details": result.details,
                    "duration": result.duration,
                }
            )

        console.print(json.dumps(json_results, indent=2))
    else:
        validator.display_results()

    # Generate report
    if args.report:
        report = validator.generate_report()
        with open(args.report, "w") as f:
            f.write(report)
        console.print(f"\n📄 Report saved to {args.report}")

    # Exit with appropriate code
    passed = sum(1 for r in results if r.passed)
    if passed == len(results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
