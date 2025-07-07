"""Configuration templates for different environments and use cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ConfigTemplate:
    """Configuration template with metadata."""
    
    name: str
    description: str
    environment: str
    config: dict[str, Any]
    required_env_vars: list[str]
    optional_env_vars: list[str]


class ConfigTemplateRegistry:
    """Registry of configuration templates."""

    def __init__(self):
        """Initialize template registry."""
        self._templates: dict[str, ConfigTemplate] = {}
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """Register default configuration templates."""
        
        # Development template
        self.register_template(ConfigTemplate(
            name="development",
            description="Development environment configuration",
            environment="development",
            config={
                "app": {
                    "environment": "development",
                    "debug": True,
                },
                "api_host": "127.0.0.1",
                "api_port": 8000,
                "storage_path": "./storage",
                "model_storage_path": "./storage/models",
                "experiment_storage_path": "./storage/experiments",
                "cache_enabled": True,
                "docs_enabled": True,
                "auth_enabled": False,
                "monitoring": {
                    "metrics_enabled": True,
                    "tracing_enabled": False,
                    "prometheus_enabled": True,
                    "log_level": "DEBUG",
                    "log_format": "text",
                }
            },
            required_env_vars=[],
            optional_env_vars=["PYNOMALY_SECRET_KEY", "PYNOMALY_DATABASE_URL"],
        ))

        # Testing template
        self.register_template(ConfigTemplate(
            name="testing",
            description="Testing environment configuration",
            environment="test",
            config={
                "app": {
                    "environment": "test",
                    "debug": False,
                },
                "api_host": "127.0.0.1",
                "api_port": 8001,
                "storage_path": "./test_storage",
                "model_storage_path": "./test_storage/models",
                "experiment_storage_path": "./test_storage/experiments",
                "cache_enabled": False,
                "docs_enabled": False,
                "auth_enabled": False,
                "database_url": "sqlite:///./test.db",
                "monitoring": {
                    "metrics_enabled": False,
                    "tracing_enabled": False,
                    "prometheus_enabled": False,
                    "log_level": "WARNING",
                    "log_format": "text",
                }
            },
            required_env_vars=[],
            optional_env_vars=[],
        ))

        # Production template
        self.register_template(ConfigTemplate(
            name="production",
            description="Production environment configuration",
            environment="production",
            config={
                "app": {
                    "environment": "production",
                    "debug": False,
                },
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "api_workers": 4,
                "api_cors_origins": [],  # Should be explicitly set
                "storage_path": "/app/storage",
                "model_storage_path": "/app/storage/models",
                "experiment_storage_path": "/app/storage/experiments",
                "cache_enabled": True,
                "docs_enabled": False,
                "auth_enabled": True,
                "use_database_repositories": True,
                "monitoring": {
                    "metrics_enabled": True,
                    "tracing_enabled": True,
                    "prometheus_enabled": True,
                    "log_level": "INFO",
                    "log_format": "json",
                    "otlp_endpoint": "http://jaeger:14268/api/traces",
                },
                "security": {
                    "sanitization_level": "strict",
                    "enable_audit_logging": True,
                    "enable_security_monitoring": True,
                    "security_headers_enabled": True,
                    "csp_enabled": True,
                    "hsts_enabled": True,
                }
            },
            required_env_vars=[
                "PYNOMALY_SECRET_KEY",
                "PYNOMALY_DATABASE_URL",
            ],
            optional_env_vars=[
                "PYNOMALY_REDIS_URL",
                "PYNOMALY_MONITORING_OTLP_ENDPOINT",
            ],
        ))

        # Docker template
        self.register_template(ConfigTemplate(
            name="docker",
            description="Docker container configuration",
            environment="production",
            config={
                "app": {
                    "environment": "production",
                    "debug": False,
                },
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "api_workers": 2,
                "storage_path": "/app/data",
                "model_storage_path": "/app/data/models",
                "experiment_storage_path": "/app/data/experiments",
                "cache_enabled": True,
                "docs_enabled": True,
                "auth_enabled": True,
                "use_database_repositories": True,
                "monitoring": {
                    "metrics_enabled": True,
                    "prometheus_enabled": True,
                    "log_level": "INFO",
                    "log_format": "json",
                }
            },
            required_env_vars=[
                "PYNOMALY_SECRET_KEY",
                "PYNOMALY_DATABASE_URL",
            ],
            optional_env_vars=[
                "PYNOMALY_REDIS_URL",
                "PYNOMALY_API_WORKERS",
            ],
        ))

        # High-performance template
        self.register_template(ConfigTemplate(
            name="high_performance",
            description="High-performance configuration with optimizations",
            environment="production",
            config={
                "app": {
                    "environment": "production",
                    "debug": False,
                },
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "api_workers": 8,
                "api_rate_limit": 1000,
                "storage_path": "/fast_storage",
                "model_storage_path": "/fast_storage/models",
                "experiment_storage_path": "/fast_storage/experiments",
                "cache_enabled": True,
                "cache_ttl": 7200,
                "docs_enabled": False,
                "auth_enabled": True,
                "use_database_repositories": True,
                "database_pool_size": 20,
                "database_max_overflow": 40,
                "monitoring": {
                    "metrics_enabled": True,
                    "tracing_enabled": False,  # Disabled for performance
                    "prometheus_enabled": True,
                    "log_level": "WARNING",  # Reduced logging
                    "log_format": "json",
                }
            },
            required_env_vars=[
                "PYNOMALY_SECRET_KEY",
                "PYNOMALY_DATABASE_URL",
                "PYNOMALY_REDIS_URL",
            ],
            optional_env_vars=[],
        ))

        # Minimal template
        self.register_template(ConfigTemplate(
            name="minimal",
            description="Minimal configuration for simple deployments",
            environment="production",
            config={
                "app": {
                    "environment": "production",
                    "debug": False,
                },
                "api_host": "127.0.0.1",
                "api_port": 8000,
                "storage_path": "./data",
                "model_storage_path": "./data/models",
                "experiment_storage_path": "./data/experiments",
                "cache_enabled": False,
                "docs_enabled": True,
                "auth_enabled": False,
                "use_database_repositories": False,
                "monitoring": {
                    "metrics_enabled": False,
                    "tracing_enabled": False,
                    "prometheus_enabled": False,
                    "log_level": "INFO",
                    "log_format": "text",
                }
            },
            required_env_vars=[],
            optional_env_vars=["PYNOMALY_SECRET_KEY"],
        ))

    def register_template(self, template: ConfigTemplate) -> None:
        """Register a configuration template."""
        self._templates[template.name] = template

    def get_template(self, name: str) -> ConfigTemplate | None:
        """Get a configuration template by name."""
        return self._templates.get(name)

    def list_templates(self) -> list[str]:
        """List available template names."""
        return list(self._templates.keys())

    def get_template_info(self, name: str) -> dict[str, Any] | None:
        """Get template information without the full config."""
        template = self._templates.get(name)
        if not template:
            return None
        
        return {
            "name": template.name,
            "description": template.description,
            "environment": template.environment,
            "required_env_vars": template.required_env_vars,
            "optional_env_vars": template.optional_env_vars,
        }

    def generate_config_file(
        self,
        template_name: str,
        output_path: str | Path,
        format: str = "yaml",
    ) -> None:
        """Generate configuration file from template."""
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "yaml":
            import yaml
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(template.config, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(template.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def generate_env_file(
        self,
        template_name: str,
        output_path: str | Path,
        include_optional: bool = True,
    ) -> None:
        """Generate .env file template."""
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Environment variables for {template.name} configuration",
            f"# {template.description}",
            "",
            "# Required environment variables:",
        ]

        for var in template.required_env_vars:
            lines.append(f"{var}=")

        if include_optional and template.optional_env_vars:
            lines.extend([
                "",
                "# Optional environment variables:",
            ])
            for var in template.optional_env_vars:
                lines.append(f"# {var}=")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# Global template registry instance
template_registry = ConfigTemplateRegistry()


def get_template_registry() -> ConfigTemplateRegistry:
    """Get the global template registry."""
    return template_registry