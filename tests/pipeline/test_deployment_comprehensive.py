"""Comprehensive deployment pipeline tests.

This module contains comprehensive tests for deployment processes,
infrastructure provisioning, monitoring setup, and deployment validation.
"""

import time
import uuid
from typing import Any

import pytest


class TestDeploymentPipeline:
    """Test deployment pipeline functionality."""

    @pytest.fixture
    def mock_deployment_environment(self):
        """Create mock deployment environment."""

        class MockDeploymentEnvironment:
            def __init__(self):
                self.environments = {
                    "development": {"url": "dev.pynomaly.com", "replicas": 1},
                    "staging": {"url": "staging.pynomaly.com", "replicas": 2},
                    "production": {"url": "pynomaly.com", "replicas": 3},
                }
                self.deployments = {}
                self.infrastructure = {}
                self.monitoring_configs = {}
                self.rollback_history = {}
                self.deployment_logs = []

            def provision_infrastructure(
                self, environment: str, config: dict[str, Any]
            ) -> dict[str, Any]:
                """Provision infrastructure for deployment."""
                infrastructure_id = str(uuid.uuid4())

                # Mock infrastructure provisioning
                infrastructure_config = {
                    "compute": {
                        "type": config.get("compute_type", "container"),
                        "cpu": config.get("cpu", "1000m"),
                        "memory": config.get("memory", "2Gi"),
                        "replicas": config.get("replicas", 2),
                    },
                    "networking": {
                        "load_balancer": config.get("load_balancer", True),
                        "ssl_termination": config.get("ssl", True),
                        "domain": self.environments[environment]["url"],
                    },
                    "storage": {
                        "persistent_volumes": config.get("storage", []),
                        "database": config.get("database", {}),
                    },
                    "security": {
                        "firewall_rules": config.get("firewall_rules", []),
                        "network_policies": config.get("network_policies", []),
                        "secrets_management": config.get("secrets", {}),
                    },
                }

                # Simulate provisioning process
                provisioning_result = {
                    "infrastructure_id": infrastructure_id,
                    "environment": environment,
                    "config": infrastructure_config,
                    "status": "provisioned",
                    "provisioning_time": 180.0,  # 3 minutes mock time
                    "endpoints": {
                        "api": f"https://{infrastructure_config['networking']['domain']}/api",
                        "health": f"https://{infrastructure_config['networking']['domain']}/health",
                        "metrics": f"https://{infrastructure_config['networking']['domain']}/metrics",
                    },
                    "resources": {
                        "load_balancer_ip": "203.0.113.10",
                        "cluster_nodes": [
                            f"node-{i}"
                            for i in range(infrastructure_config["compute"]["replicas"])
                        ],
                        "database_endpoint": (
                            "db.internal.com:5432"
                            if infrastructure_config["storage"]["database"]
                            else None
                        ),
                    },
                }

                # Store infrastructure
                self.infrastructure[infrastructure_id] = provisioning_result

                self._log_deployment_event(
                    f"Infrastructure provisioned for {environment}",
                    {
                        "infrastructure_id": infrastructure_id,
                        "environment": environment,
                    },
                )

                return provisioning_result

            def deploy_application(
                self,
                environment: str,
                artifacts: list[str],
                config: dict[str, Any] = None,
            ) -> dict[str, Any]:
                """Deploy application to environment."""
                deployment_id = str(uuid.uuid4())
                config = config or {}

                # Find infrastructure for environment
                infrastructure = None
                for infra in self.infrastructure.values():
                    if infra["environment"] == environment:
                        infrastructure = infra
                        break

                if not infrastructure:
                    return {
                        "deployment_id": deployment_id,
                        "success": False,
                        "error": f"No infrastructure found for environment {environment}",
                    }

                # Simulate deployment process
                deployment_steps = [
                    {"name": "validate_artifacts", "duration": 5.0},
                    {"name": "backup_current_version", "duration": 15.0},
                    {"name": "update_configuration", "duration": 3.0},
                    {"name": "deploy_containers", "duration": 30.0},
                    {"name": "run_migrations", "duration": 10.0},
                    {"name": "health_check", "duration": 20.0},
                    {"name": "update_load_balancer", "duration": 8.0},
                    {"name": "smoke_tests", "duration": 25.0},
                ]

                step_results = []
                total_duration = 0
                deployment_success = True

                for step in deployment_steps:
                    step_result = self._simulate_deployment_step(
                        step["name"], environment, step["duration"]
                    )
                    step_results.append(step_result)
                    total_duration += step_result["duration"]

                    if not step_result["success"]:
                        deployment_success = False
                        break

                deployment_result = {
                    "deployment_id": deployment_id,
                    "environment": environment,
                    "infrastructure_id": infrastructure["infrastructure_id"],
                    "artifacts": artifacts,
                    "config": config,
                    "success": deployment_success,
                    "total_duration": total_duration,
                    "step_results": step_results,
                    "endpoints": (
                        infrastructure["endpoints"] if deployment_success else {}
                    ),
                    "deployment_time": time.time(),
                    "version": config.get("version", "latest"),
                    "rollback_available": True,
                }

                # Store deployment
                self.deployments[deployment_id] = deployment_result

                # Update rollback history
                if environment not in self.rollback_history:
                    self.rollback_history[environment] = []

                self.rollback_history[environment].append(
                    {
                        "deployment_id": deployment_id,
                        "version": deployment_result["version"],
                        "timestamp": deployment_result["deployment_time"],
                        "success": deployment_success,
                    }
                )

                # Keep only last 10 deployments for rollback
                self.rollback_history[environment] = self.rollback_history[environment][
                    -10:
                ]

                self._log_deployment_event(
                    f"Application {'deployed' if deployment_success else 'deployment failed'} to {environment}",
                    {
                        "deployment_id": deployment_id,
                        "environment": environment,
                        "success": deployment_success,
                    },
                )

                return deployment_result

            def _simulate_deployment_step(
                self, step_name: str, environment: str, base_duration: float
            ) -> dict[str, Any]:
                """Simulate individual deployment step."""
                # Different environments have different success rates
                success_rates = {
                    "development": 0.95,
                    "staging": 0.90,
                    "production": 0.85,
                }

                success_rate = success_rates.get(environment, 0.90)

                # Some steps are more likely to fail than others
                step_risks = {
                    "validate_artifacts": 0.05,
                    "backup_current_version": 0.02,
                    "update_configuration": 0.08,
                    "deploy_containers": 0.15,
                    "run_migrations": 0.20,
                    "health_check": 0.25,
                    "update_load_balancer": 0.10,
                    "smoke_tests": 0.30,
                }

                risk_factor = step_risks.get(step_name, 0.10)
                success_rate * (1 - risk_factor)

                # For testing, assume success (can be made probabilistic)
                success = True

                # Simulate some variation in duration
                actual_duration = base_duration * (
                    0.8 + 0.4 * 0.5
                )  # 80-120% of base duration

                step_result = {
                    "step_name": step_name,
                    "success": success,
                    "duration": actual_duration,
                    "message": f"Step '{step_name}' {'completed successfully' if success else 'failed'}",
                    "environment": environment,
                }

                if not success:
                    step_result["error"] = f"Simulated failure in {step_name}"

                return step_result

            def setup_monitoring(
                self, deployment_id: str, monitoring_config: dict[str, Any]
            ) -> dict[str, Any]:
                """Setup monitoring for deployment."""
                if deployment_id not in self.deployments:
                    return {"error": "Deployment not found", "success": False}

                deployment = self.deployments[deployment_id]
                environment = deployment["environment"]

                # Setup monitoring configuration
                monitoring_setup = {
                    "deployment_id": deployment_id,
                    "environment": environment,
                    "metrics": {
                        "enabled": monitoring_config.get("metrics_enabled", True),
                        "endpoint": f"{deployment['endpoints'].get('metrics', '')}/metrics",
                        "scrape_interval": monitoring_config.get(
                            "scrape_interval", "30s"
                        ),
                        "retention": monitoring_config.get("retention", "7d"),
                    },
                    "logging": {
                        "enabled": monitoring_config.get("logging_enabled", True),
                        "level": monitoring_config.get("log_level", "INFO"),
                        "aggregation": monitoring_config.get("log_aggregation", True),
                        "retention": monitoring_config.get("log_retention", "30d"),
                    },
                    "alerting": {
                        "enabled": monitoring_config.get("alerting_enabled", True),
                        "channels": monitoring_config.get(
                            "alert_channels", ["email", "slack"]
                        ),
                        "rules": self._create_default_alert_rules(environment),
                    },
                    "dashboards": {
                        "application_dashboard": f"dashboard-{deployment_id}-app",
                        "infrastructure_dashboard": f"dashboard-{deployment_id}-infra",
                        "custom_dashboards": monitoring_config.get(
                            "custom_dashboards", []
                        ),
                    },
                }

                # Store monitoring configuration
                self.monitoring_configs[deployment_id] = monitoring_setup

                self._log_deployment_event(
                    f"Monitoring setup for deployment {deployment_id}",
                    {"deployment_id": deployment_id, "environment": environment},
                )

                return {
                    "success": True,
                    "monitoring_id": f"monitoring-{deployment_id}",
                    "config": monitoring_setup,
                }

            def _create_default_alert_rules(
                self, environment: str
            ) -> list[dict[str, Any]]:
                """Create default alert rules for environment."""
                # Environment-specific alert thresholds
                thresholds = {
                    "development": {
                        "error_rate": 0.10,
                        "response_time": 2000,
                        "cpu": 80,
                        "memory": 85,
                    },
                    "staging": {
                        "error_rate": 0.05,
                        "response_time": 1000,
                        "cpu": 70,
                        "memory": 80,
                    },
                    "production": {
                        "error_rate": 0.02,
                        "response_time": 500,
                        "cpu": 60,
                        "memory": 75,
                    },
                }

                env_thresholds = thresholds.get(environment, thresholds["production"])

                return [
                    {
                        "name": "high_error_rate",
                        "condition": f"error_rate > {env_thresholds['error_rate']}",
                        "severity": "critical",
                        "duration": "5m",
                    },
                    {
                        "name": "high_response_time",
                        "condition": f"response_time_ms > {env_thresholds['response_time']}",
                        "severity": "warning",
                        "duration": "10m",
                    },
                    {
                        "name": "high_cpu_usage",
                        "condition": f"cpu_usage_percent > {env_thresholds['cpu']}",
                        "severity": "warning",
                        "duration": "15m",
                    },
                    {
                        "name": "high_memory_usage",
                        "condition": f"memory_usage_percent > {env_thresholds['memory']}",
                        "severity": "critical",
                        "duration": "5m",
                    },
                    {
                        "name": "service_down",
                        "condition": "up == 0",
                        "severity": "critical",
                        "duration": "1m",
                    },
                ]

            def validate_deployment(self, deployment_id: str) -> dict[str, Any]:
                """Validate deployment health and functionality."""
                if deployment_id not in self.deployments:
                    return {"error": "Deployment not found", "valid": False}

                deployment = self.deployments[deployment_id]

                if not deployment["success"]:
                    return {"error": "Deployment failed", "valid": False}

                # Perform validation checks
                validation_checks = {
                    "endpoint_health": self._check_endpoint_health(
                        deployment["endpoints"]
                    ),
                    "service_availability": self._check_service_availability(
                        deployment
                    ),
                    "database_connectivity": self._check_database_connectivity(
                        deployment
                    ),
                    "external_dependencies": self._check_external_dependencies(
                        deployment
                    ),
                    "performance_baseline": self._check_performance_baseline(
                        deployment
                    ),
                    "security_configuration": self._check_security_configuration(
                        deployment
                    ),
                }

                all_checks_passed = all(
                    check["passed"] for check in validation_checks.values()
                )

                return {
                    "deployment_id": deployment_id,
                    "valid": all_checks_passed,
                    "checks": validation_checks,
                    "passed_checks": sum(
                        1 for check in validation_checks.values() if check["passed"]
                    ),
                    "total_checks": len(validation_checks),
                    "validation_time": time.time(),
                }

            def _check_endpoint_health(
                self, endpoints: dict[str, str]
            ) -> dict[str, Any]:
                """Check endpoint health."""
                # Mock endpoint health checks
                health_results = {}

                for endpoint_name, endpoint_url in endpoints.items():
                    # Simulate HTTP health check
                    health_results[endpoint_name] = {
                        "url": endpoint_url,
                        "status_code": 200,
                        "response_time_ms": 150,
                        "healthy": True,
                    }

                all_healthy = all(
                    result["healthy"] for result in health_results.values()
                )

                return {
                    "passed": all_healthy,
                    "details": health_results,
                    "message": (
                        "All endpoints healthy"
                        if all_healthy
                        else "Some endpoints unhealthy"
                    ),
                }

            def _check_service_availability(
                self, deployment: dict[str, Any]
            ) -> dict[str, Any]:
                """Check service availability."""
                # Mock service availability check
                expected_replicas = deployment.get("config", {}).get("replicas", 2)
                available_replicas = (
                    expected_replicas  # Assume all replicas are available
                )

                availability_percent = (available_replicas / expected_replicas) * 100
                passed = availability_percent >= 80  # 80% minimum availability

                return {
                    "passed": passed,
                    "expected_replicas": expected_replicas,
                    "available_replicas": available_replicas,
                    "availability_percent": availability_percent,
                    "message": f"{available_replicas}/{expected_replicas} replicas available",
                }

            def _check_database_connectivity(
                self, deployment: dict[str, Any]
            ) -> dict[str, Any]:
                """Check database connectivity."""
                # Mock database connectivity check
                has_database = deployment.get("config", {}).get("database", False)

                if not has_database:
                    return {
                        "passed": True,
                        "message": "No database configured",
                        "skipped": True,
                    }

                # Simulate database connection test
                connection_successful = True  # Mock successful connection
                response_time = 25  # Mock response time in ms

                return {
                    "passed": connection_successful,
                    "response_time_ms": response_time,
                    "message": (
                        "Database connection successful"
                        if connection_successful
                        else "Database connection failed"
                    ),
                }

            def _check_external_dependencies(
                self, deployment: dict[str, Any]
            ) -> dict[str, Any]:
                """Check external dependencies."""
                # Mock external dependency checks
                dependencies = deployment.get("config", {}).get(
                    "external_dependencies", []
                )

                if not dependencies:
                    return {
                        "passed": True,
                        "message": "No external dependencies configured",
                        "skipped": True,
                    }

                # Simulate dependency checks
                dependency_results = {}
                for dep in dependencies:
                    dependency_results[dep] = {
                        "status": "healthy",
                        "response_time_ms": 100,
                        "available": True,
                    }

                all_available = all(
                    result["available"] for result in dependency_results.values()
                )

                return {
                    "passed": all_available,
                    "dependencies": dependency_results,
                    "message": (
                        "All dependencies available"
                        if all_available
                        else "Some dependencies unavailable"
                    ),
                }

            def _check_performance_baseline(
                self, deployment: dict[str, Any]
            ) -> dict[str, Any]:
                """Check performance against baseline."""
                # Mock performance metrics
                current_metrics = {
                    "response_time_p95": 200,  # ms
                    "throughput_rps": 500,
                    "error_rate": 0.01,  # 1%
                    "cpu_usage": 45,  # %
                    "memory_usage": 60,  # %
                }

                baseline_metrics = {
                    "response_time_p95": 300,  # Max allowed
                    "throughput_rps": 100,  # Min required
                    "error_rate": 0.05,  # Max allowed
                    "cpu_usage": 70,  # Max allowed
                    "memory_usage": 80,  # Max allowed
                }

                performance_passed = (
                    current_metrics["response_time_p95"]
                    <= baseline_metrics["response_time_p95"]
                    and current_metrics["throughput_rps"]
                    >= baseline_metrics["throughput_rps"]
                    and current_metrics["error_rate"] <= baseline_metrics["error_rate"]
                    and current_metrics["cpu_usage"] <= baseline_metrics["cpu_usage"]
                    and current_metrics["memory_usage"]
                    <= baseline_metrics["memory_usage"]
                )

                return {
                    "passed": performance_passed,
                    "current_metrics": current_metrics,
                    "baseline_metrics": baseline_metrics,
                    "message": (
                        "Performance within baseline"
                        if performance_passed
                        else "Performance below baseline"
                    ),
                }

            def _check_security_configuration(
                self, deployment: dict[str, Any]
            ) -> dict[str, Any]:
                """Check security configuration."""
                # Mock security configuration checks
                security_checks = {
                    "https_enabled": True,
                    "authentication_configured": True,
                    "authorization_enabled": True,
                    "secrets_encrypted": True,
                    "network_policies_applied": True,
                    "vulnerability_scan_passed": True,
                }

                all_secure = all(security_checks.values())

                return {
                    "passed": all_secure,
                    "security_checks": security_checks,
                    "message": (
                        "Security configuration valid"
                        if all_secure
                        else "Security issues found"
                    ),
                }

            def rollback_deployment(
                self, environment: str, target_version: str = None
            ) -> dict[str, Any]:
                """Rollback deployment to previous version."""
                if environment not in self.rollback_history:
                    return {"error": "No deployment history found", "success": False}

                history = self.rollback_history[environment]
                successful_deployments = [d for d in history if d["success"]]

                if len(successful_deployments) < 2:
                    return {
                        "error": "No previous successful deployment found",
                        "success": False,
                    }

                # Find target deployment
                if target_version:
                    target_deployment = next(
                        (
                            d
                            for d in reversed(successful_deployments)
                            if d["version"] == target_version
                        ),
                        None,
                    )
                else:
                    # Get previous successful deployment
                    target_deployment = (
                        successful_deployments[-2]
                        if len(successful_deployments) >= 2
                        else None
                    )

                if not target_deployment:
                    return {
                        "error": f"Target version {target_version} not found",
                        "success": False,
                    }

                # Simulate rollback process
                rollback_id = str(uuid.uuid4())

                rollback_result = {
                    "rollback_id": rollback_id,
                    "environment": environment,
                    "target_version": target_deployment["version"],
                    "target_deployment_id": target_deployment["deployment_id"],
                    "rollback_time": time.time(),
                    "success": True,  # Assume rollback succeeds
                    "duration": 45.0,  # Mock rollback duration
                    "steps": [
                        "validate_target_version",
                        "backup_current_state",
                        "switch_to_previous_version",
                        "update_load_balancer",
                        "verify_rollback",
                        "cleanup_failed_deployment",
                    ],
                }

                self._log_deployment_event(
                    f"Rollback completed for {environment}",
                    {
                        "rollback_id": rollback_id,
                        "environment": environment,
                        "target_version": target_deployment["version"],
                    },
                )

                return rollback_result

            def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
                """Get deployment status and metrics."""
                if deployment_id not in self.deployments:
                    return {"error": "Deployment not found"}

                deployment = self.deployments[deployment_id]

                # Mock current metrics
                current_metrics = {
                    "uptime": time.time() - deployment["deployment_time"],
                    "requests_per_second": 150,
                    "error_rate": 0.008,
                    "average_response_time": 180,
                    "active_connections": 45,
                    "memory_usage_mb": 512,
                    "cpu_usage_percent": 35,
                }

                return {
                    "deployment_id": deployment_id,
                    "environment": deployment["environment"],
                    "version": deployment["version"],
                    "status": "healthy" if deployment["success"] else "failed",
                    "deployed_at": deployment["deployment_time"],
                    "metrics": current_metrics,
                    "endpoints": deployment["endpoints"],
                }

            def _log_deployment_event(
                self, message: str, context: dict[str, Any] = None
            ):
                """Log deployment event."""
                log_entry = {
                    "timestamp": time.time(),
                    "message": message,
                    "context": context or {},
                }
                self.deployment_logs.append(log_entry)

            def get_deployment_logs(
                self, environment: str = None, limit: int = 50
            ) -> list[dict[str, Any]]:
                """Get deployment logs."""
                logs = self.deployment_logs

                if environment:
                    logs = [
                        log
                        for log in logs
                        if log.get("context", {}).get("environment") == environment
                    ]

                return logs[-limit:] if limit else logs

        return MockDeploymentEnvironment()

    def test_infrastructure_provisioning(self, mock_deployment_environment):
        """Test infrastructure provisioning."""
        deploy_env = mock_deployment_environment

        # Test provisioning for different environments
        environments = ["development", "staging", "production"]

        for environment in environments:
            # Environment-specific configuration
            if environment == "development":
                config = {
                    "compute_type": "container",
                    "cpu": "500m",
                    "memory": "1Gi",
                    "replicas": 1,
                    "load_balancer": False,
                    "ssl": False,
                }
            elif environment == "staging":
                config = {
                    "compute_type": "container",
                    "cpu": "1000m",
                    "memory": "2Gi",
                    "replicas": 2,
                    "load_balancer": True,
                    "ssl": True,
                    "database": {"type": "postgres", "size": "small"},
                }
            else:  # production
                config = {
                    "compute_type": "container",
                    "cpu": "2000m",
                    "memory": "4Gi",
                    "replicas": 3,
                    "load_balancer": True,
                    "ssl": True,
                    "database": {"type": "postgres", "size": "large"},
                    "firewall_rules": ["allow-https", "allow-ssh-admin"],
                    "secrets": {"encryption": "enabled"},
                }

            # Provision infrastructure
            provisioning_result = deploy_env.provision_infrastructure(
                environment, config
            )

            assert provisioning_result["status"] == "provisioned"
            assert provisioning_result["environment"] == environment
            assert "infrastructure_id" in provisioning_result
            assert "config" in provisioning_result
            assert "endpoints" in provisioning_result
            assert "resources" in provisioning_result

            # Verify environment-specific configuration
            infra_config = provisioning_result["config"]

            assert infra_config["compute"]["replicas"] == config["replicas"]
            assert infra_config["compute"]["cpu"] == config["cpu"]
            assert infra_config["compute"]["memory"] == config["memory"]
            assert (
                infra_config["networking"]["load_balancer"] == config["load_balancer"]
            )
            assert infra_config["networking"]["ssl_termination"] == config["ssl"]

            # Verify endpoints
            endpoints = provisioning_result["endpoints"]
            expected_domain = deploy_env.environments[environment]["url"]

            assert expected_domain in endpoints["api"]
            assert expected_domain in endpoints["health"]
            assert expected_domain in endpoints["metrics"]

            # Verify resources
            resources = provisioning_result["resources"]
            assert len(resources["cluster_nodes"]) == config["replicas"]

            if config.get("database"):
                assert resources["database_endpoint"] is not None
            else:
                assert resources["database_endpoint"] is None

    def test_application_deployment(self, mock_deployment_environment):
        """Test application deployment process."""
        deploy_env = mock_deployment_environment

        # Provision infrastructure first
        environment = "staging"
        infra_config = {
            "compute_type": "container",
            "cpu": "1000m",
            "memory": "2Gi",
            "replicas": 2,
            "load_balancer": True,
            "ssl": True,
        }

        provisioning_result = deploy_env.provision_infrastructure(
            environment, infra_config
        )
        assert provisioning_result["status"] == "provisioned"

        # Deploy application
        artifacts = ["pynomaly-1.0.0-py3-none-any.whl", "pynomaly-1.0.0.tar.gz"]

        app_config = {
            "version": "1.0.0",
            "replicas": 2,
            "environment_variables": {
                "LOG_LEVEL": "INFO",
                "DATABASE_URL": "postgresql://user:pass@db:5432/pynomaly",
            },
            "health_check_path": "/health",
            "readiness_probe": "/ready",
        }

        deployment_result = deploy_env.deploy_application(
            environment, artifacts, app_config
        )

        assert deployment_result["success"]
        assert deployment_result["environment"] == environment
        assert "deployment_id" in deployment_result
        assert deployment_result["artifacts"] == artifacts
        assert deployment_result["config"] == app_config
        assert deployment_result["version"] == "1.0.0"
        assert deployment_result["rollback_available"]

        # Verify deployment steps
        step_results = deployment_result["step_results"]
        assert len(step_results) > 0

        expected_steps = [
            "validate_artifacts",
            "backup_current_version",
            "update_configuration",
            "deploy_containers",
            "run_migrations",
            "health_check",
            "update_load_balancer",
            "smoke_tests",
        ]

        executed_steps = [step["step_name"] for step in step_results]
        for expected_step in expected_steps:
            assert expected_step in executed_steps

        # All steps should succeed for this test
        for step in step_results:
            assert step["success"]
            assert step["duration"] > 0
            assert step["environment"] == environment

        # Verify endpoints are available
        assert "endpoints" in deployment_result
        if deployment_result["success"]:
            assert len(deployment_result["endpoints"]) > 0

    def test_deployment_monitoring_setup(self, mock_deployment_environment):
        """Test deployment monitoring setup."""
        deploy_env = mock_deployment_environment

        # Deploy application first
        environment = "production"

        # Provision and deploy
        deploy_env.provision_infrastructure(
            environment, {"replicas": 3, "load_balancer": True, "ssl": True}
        )

        deployment_result = deploy_env.deploy_application(
            environment, ["pynomaly-1.0.0.whl"], {"version": "1.0.0"}
        )

        deployment_id = deployment_result["deployment_id"]

        # Setup monitoring
        monitoring_config = {
            "metrics_enabled": True,
            "scrape_interval": "15s",
            "retention": "30d",
            "logging_enabled": True,
            "log_level": "INFO",
            "log_retention": "60d",
            "alerting_enabled": True,
            "alert_channels": ["email", "slack", "pagerduty"],
            "custom_dashboards": ["business_metrics", "sla_dashboard"],
        }

        monitoring_result = deploy_env.setup_monitoring(
            deployment_id, monitoring_config
        )

        assert monitoring_result["success"]
        assert "monitoring_id" in monitoring_result
        assert "config" in monitoring_result

        # Verify monitoring configuration
        monitoring_setup = monitoring_result["config"]

        # Metrics configuration
        metrics_config = monitoring_setup["metrics"]
        assert metrics_config["enabled"]
        assert metrics_config["scrape_interval"] == "15s"
        assert metrics_config["retention"] == "30d"
        assert "endpoint" in metrics_config

        # Logging configuration
        logging_config = monitoring_setup["logging"]
        assert logging_config["enabled"]
        assert logging_config["level"] == "INFO"
        assert logging_config["retention"] == "60d"
        assert logging_config["aggregation"]

        # Alerting configuration
        alerting_config = monitoring_setup["alerting"]
        assert alerting_config["enabled"]
        assert alerting_config["channels"] == ["email", "slack", "pagerduty"]
        assert "rules" in alerting_config

        # Verify alert rules
        alert_rules = alerting_config["rules"]
        assert len(alert_rules) > 0

        expected_alerts = [
            "high_error_rate",
            "high_response_time",
            "high_cpu_usage",
            "high_memory_usage",
            "service_down",
        ]
        rule_names = [rule["name"] for rule in alert_rules]

        for expected_alert in expected_alerts:
            assert expected_alert in rule_names

        # Verify environment-specific thresholds (production should be stricter)
        error_rate_rule = next(
            rule for rule in alert_rules if rule["name"] == "high_error_rate"
        )
        assert "0.02" in error_rate_rule["condition"]  # 2% for production

        # Dashboards configuration
        dashboards_config = monitoring_setup["dashboards"]
        assert "application_dashboard" in dashboards_config
        assert "infrastructure_dashboard" in dashboards_config
        assert dashboards_config["custom_dashboards"] == [
            "business_metrics",
            "sla_dashboard",
        ]

    def test_deployment_validation(self, mock_deployment_environment):
        """Test deployment validation process."""
        deploy_env = mock_deployment_environment

        # Deploy application
        environment = "staging"

        deploy_env.provision_infrastructure(environment, {"replicas": 2})
        deployment_result = deploy_env.deploy_application(
            environment, ["pynomaly-1.0.0.whl"], {"version": "1.0.0", "replicas": 2}
        )

        deployment_id = deployment_result["deployment_id"]

        # Validate deployment
        validation_result = deploy_env.validate_deployment(deployment_id)

        assert "valid" in validation_result
        assert validation_result["deployment_id"] == deployment_id
        assert "checks" in validation_result
        assert "passed_checks" in validation_result
        assert "total_checks" in validation_result

        # Verify validation checks
        checks = validation_result["checks"]
        expected_checks = [
            "endpoint_health",
            "service_availability",
            "database_connectivity",
            "external_dependencies",
            "performance_baseline",
            "security_configuration",
        ]

        for check_name in expected_checks:
            assert check_name in checks
            check_result = checks[check_name]
            assert "passed" in check_result
            assert "message" in check_result

        # Endpoint health check
        endpoint_check = checks["endpoint_health"]
        if "details" in endpoint_check:
            for _endpoint_name, endpoint_result in endpoint_check["details"].items():
                assert "status_code" in endpoint_result
                assert "response_time_ms" in endpoint_result
                assert "healthy" in endpoint_result

        # Service availability check
        availability_check = checks["service_availability"]
        assert "expected_replicas" in availability_check
        assert "available_replicas" in availability_check
        assert "availability_percent" in availability_check

        # Performance baseline check
        performance_check = checks["performance_baseline"]
        assert "current_metrics" in performance_check
        assert "baseline_metrics" in performance_check

        # Security configuration check
        security_check = checks["security_configuration"]
        assert "security_checks" in security_check

        # Overall validation should pass for successful deployment
        if deployment_result["success"]:
            assert validation_result["valid"]

    def test_deployment_rollback(self, mock_deployment_environment):
        """Test deployment rollback functionality."""
        deploy_env = mock_deployment_environment

        environment = "production"

        # Provision infrastructure
        deploy_env.provision_infrastructure(environment, {"replicas": 3})

        # Deploy version 1.0.0
        deployment_v1 = deploy_env.deploy_application(
            environment, ["pynomaly-1.0.0.whl"], {"version": "1.0.0"}
        )
        assert deployment_v1["success"]

        # Deploy version 1.1.0
        deployment_v1_1 = deploy_env.deploy_application(
            environment, ["pynomaly-1.1.0.whl"], {"version": "1.1.0"}
        )
        assert deployment_v1_1["success"]

        # Deploy version 1.2.0 (this will be the "failed" deployment we rollback from)
        deploy_env.deploy_application(
            environment, ["pynomaly-1.2.0.whl"], {"version": "1.2.0"}
        )

        # Test rollback to previous version
        rollback_result = deploy_env.rollback_deployment(environment)

        assert rollback_result["success"]
        assert rollback_result["environment"] == environment
        assert rollback_result["target_version"] == "1.1.0"  # Previous version
        assert "rollback_id" in rollback_result
        assert "duration" in rollback_result
        assert "steps" in rollback_result

        # Verify rollback steps
        expected_steps = [
            "validate_target_version",
            "backup_current_state",
            "switch_to_previous_version",
            "update_load_balancer",
            "verify_rollback",
            "cleanup_failed_deployment",
        ]

        for step in expected_steps:
            assert step in rollback_result["steps"]

        # Test rollback to specific version
        specific_rollback = deploy_env.rollback_deployment(
            environment, target_version="1.0.0"
        )

        assert specific_rollback["success"]
        assert specific_rollback["target_version"] == "1.0.0"

        # Test rollback with no deployment history
        rollback_no_history = deploy_env.rollback_deployment("nonexistent-env")
        assert not rollback_no_history["success"]
        assert "error" in rollback_no_history

        # Test rollback to non-existent version
        rollback_bad_version = deploy_env.rollback_deployment(
            environment, target_version="999.0.0"
        )
        assert not rollback_bad_version["success"]
        assert "not found" in rollback_bad_version["error"]

    def test_deployment_status_monitoring(self, mock_deployment_environment):
        """Test deployment status and metrics monitoring."""
        deploy_env = mock_deployment_environment

        # Deploy application
        environment = "staging"

        deploy_env.provision_infrastructure(environment, {"replicas": 2})
        deployment_result = deploy_env.deploy_application(
            environment, ["pynomaly-1.0.0.whl"], {"version": "1.0.0"}
        )

        deployment_id = deployment_result["deployment_id"]

        # Get deployment status
        status_result = deploy_env.get_deployment_status(deployment_id)

        assert status_result["deployment_id"] == deployment_id
        assert status_result["environment"] == environment
        assert status_result["version"] == "1.0.0"
        assert "status" in status_result
        assert "deployed_at" in status_result
        assert "metrics" in status_result
        assert "endpoints" in status_result

        # Verify metrics
        metrics = status_result["metrics"]
        expected_metrics = [
            "uptime",
            "requests_per_second",
            "error_rate",
            "average_response_time",
            "active_connections",
            "memory_usage_mb",
            "cpu_usage_percent",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], int | float)

        # Verify reasonable metric values
        assert metrics["uptime"] >= 0
        assert 0 <= metrics["error_rate"] <= 1
        assert metrics["average_response_time"] > 0
        assert metrics["memory_usage_mb"] > 0
        assert 0 <= metrics["cpu_usage_percent"] <= 100

        # Test status for non-existent deployment
        status_error = deploy_env.get_deployment_status("non-existent-deployment")
        assert "error" in status_error

    def test_deployment_logging(self, mock_deployment_environment):
        """Test deployment logging and audit trail."""
        deploy_env = mock_deployment_environment

        environment = "development"

        # Perform multiple deployment operations
        deploy_env.provision_infrastructure(environment, {"replicas": 1})

        deployment_result = deploy_env.deploy_application(
            environment, ["pynomaly-1.0.0.whl"], {"version": "1.0.0"}
        )

        deploy_env.setup_monitoring(
            deployment_result["deployment_id"], {"metrics_enabled": True}
        )

        # Get deployment logs
        all_logs = deploy_env.get_deployment_logs()

        assert len(all_logs) > 0

        # Verify log structure
        for log_entry in all_logs:
            assert "timestamp" in log_entry
            assert "message" in log_entry
            assert "context" in log_entry
            assert isinstance(log_entry["timestamp"], int | float)
            assert isinstance(log_entry["message"], str)
            assert isinstance(log_entry["context"], dict)

        # Get environment-specific logs
        env_logs = deploy_env.get_deployment_logs(environment=environment)

        # Should have logs for this environment
        assert len(env_logs) > 0

        # All logs should be for the specified environment
        for log_entry in env_logs:
            if "environment" in log_entry["context"]:
                assert log_entry["context"]["environment"] == environment

        # Test log filtering with limit
        limited_logs = deploy_env.get_deployment_logs(limit=2)
        assert len(limited_logs) <= 2

        # Should be the most recent logs
        if len(all_logs) >= 2:
            assert limited_logs == all_logs[-2:]

    def test_multi_environment_deployment_workflow(self, mock_deployment_environment):
        """Test complete multi-environment deployment workflow."""
        deploy_env = mock_deployment_environment

        environments = ["development", "staging", "production"]
        version = "1.0.0"
        artifacts = ["pynomaly-1.0.0.whl"]

        deployment_results = {}

        # Deploy to each environment in sequence
        for environment in environments:
            # Environment-specific configuration
            if environment == "development":
                config = {"replicas": 1, "resources": "minimal"}
            elif environment == "staging":
                config = {"replicas": 2, "resources": "moderate", "database": True}
            else:  # production
                config = {
                    "replicas": 3,
                    "resources": "high",
                    "database": True,
                    "monitoring": "full",
                }

            # Provision infrastructure
            infra_result = deploy_env.provision_infrastructure(environment, config)
            assert infra_result["status"] == "provisioned"

            # Deploy application
            app_config = {"version": version, **config}
            deployment_result = deploy_env.deploy_application(
                environment, artifacts, app_config
            )

            assert deployment_result["success"]
            deployment_results[environment] = deployment_result

            # Setup monitoring (more comprehensive for production)
            monitoring_config = {
                "metrics_enabled": True,
                "alerting_enabled": environment in ["staging", "production"],
                "log_level": "DEBUG" if environment == "development" else "INFO",
            }

            monitoring_result = deploy_env.setup_monitoring(
                deployment_result["deployment_id"], monitoring_config
            )
            assert monitoring_result["success"]

            # Validate deployment
            validation_result = deploy_env.validate_deployment(
                deployment_result["deployment_id"]
            )
            assert validation_result["valid"]

        # Verify all environments were deployed
        assert len(deployment_results) == 3

        # Verify environment-specific configurations
        dev_deployment = deployment_results["development"]
        staging_deployment = deployment_results["staging"]
        prod_deployment = deployment_results["production"]

        # Development should have minimal resources
        assert dev_deployment["config"]["replicas"] == 1

        # Staging should have moderate resources
        assert staging_deployment["config"]["replicas"] == 2

        # Production should have high availability
        assert prod_deployment["config"]["replicas"] == 3

        # All deployments should have same version
        for deployment in deployment_results.values():
            assert deployment["version"] == version

        # Get comprehensive deployment logs
        all_logs = deploy_env.get_deployment_logs()

        # Should have logs for all environments
        log_environments = set()
        for log_entry in all_logs:
            if "environment" in log_entry["context"]:
                log_environments.add(log_entry["context"]["environment"])

        assert log_environments == set(environments)
