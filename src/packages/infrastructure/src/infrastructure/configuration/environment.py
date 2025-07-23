"""Environment detection and management."""

from __future__ import annotations

import os
from enum import Enum
from typing import Dict, Any, Optional


class Environment(str, Enum):
    """Supported environment types."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    
    @classmethod
    def from_string(cls, env_str: str) -> Environment:
        """Create environment from string with validation."""
        env_str = env_str.lower().strip()
        
        # Handle common aliases
        aliases = {
            "dev": cls.DEVELOPMENT,
            "develop": cls.DEVELOPMENT,
            "test": cls.TESTING,
            "tests": cls.TESTING,
            "stage": cls.STAGING,
            "prod": cls.PRODUCTION,
            "production": cls.PRODUCTION,
        }
        
        if env_str in aliases:
            return aliases[env_str]
        
        try:
            return cls(env_str)
        except ValueError:
            raise ValueError(
                f"Invalid environment '{env_str}'. "
                f"Valid options: {', '.join([e.value for e in cls])}"
            )
    
    def is_production(self) -> bool:
        """Check if this is production environment."""
        return self == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if this is development environment."""
        return self == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if this is testing environment."""
        return self == Environment.TESTING
    
    def is_staging(self) -> bool:
        """Check if this is staging environment."""
        return self == Environment.STAGING
    
    def allows_debug(self) -> bool:
        """Check if debug mode is allowed in this environment."""
        return self in [Environment.DEVELOPMENT, Environment.TESTING]
    
    def requires_https(self) -> bool:
        """Check if HTTPS is required in this environment."""
        return self in [Environment.STAGING, Environment.PRODUCTION]
    
    def allows_profiling(self) -> bool:
        """Check if profiling is allowed in this environment."""
        return self != Environment.PRODUCTION
    
    def get_log_level(self) -> str:
        """Get recommended log level for this environment."""
        log_levels = {
            Environment.DEVELOPMENT: "DEBUG",
            Environment.TESTING: "WARNING",
            Environment.STAGING: "INFO",
            Environment.PRODUCTION: "INFO",
        }
        return log_levels[self]


class EnvironmentManager:
    """Manages environment detection and configuration."""
    
    def __init__(self):
        self._environment: Optional[Environment] = None
        self._is_containerized: Optional[bool] = None
        self._is_kubernetes: Optional[bool] = None
        self._cloud_provider: Optional[str] = None
    
    def get_environment(self) -> Environment:
        """Get the current environment."""
        if self._environment is None:
            env_str = os.getenv("ENVIRONMENT", os.getenv("ENV", "development"))
            self._environment = Environment.from_string(env_str)
        return self._environment
    
    def set_environment(self, environment: Environment) -> None:
        """Set the current environment (mainly for testing)."""
        self._environment = environment
    
    def is_containerized(self) -> bool:
        """Check if running in a containerized environment."""
        if self._is_containerized is None:
            # Check for common container indicators
            indicators = [
                os.path.exists("/.dockerenv"),
                os.path.exists("/proc/1/cgroup") and "docker" in open("/proc/1/cgroup").read(),
                os.getenv("CONTAINER") == "true",
                os.getenv("DOCKER_CONTAINER") == "true",
            ]
            self._is_containerized = any(indicators)
        return self._is_containerized
    
    def is_kubernetes(self) -> bool:
        """Check if running in Kubernetes."""
        if self._is_kubernetes is None:
            indicators = [
                os.getenv("KUBERNETES_SERVICE_HOST") is not None,
                os.path.exists("/var/run/secrets/kubernetes.io"),
                os.getenv("K8S_NODE_NAME") is not None,
            ]
            self._is_kubernetes = any(indicators)
        return self._is_kubernetes
    
    def get_cloud_provider(self) -> Optional[str]:
        """Detect cloud provider if running in cloud."""
        if self._cloud_provider is None:
            # AWS detection
            if any([
                os.getenv("AWS_REGION"),
                os.getenv("AWS_DEFAULT_REGION"),
                os.path.exists("/sys/hypervisor/uuid") and 
                open("/sys/hypervisor/uuid").read().startswith("ec2"),
            ]):
                self._cloud_provider = "aws"
            
            # GCP detection
            elif any([
                os.getenv("GOOGLE_CLOUD_PROJECT"),
                os.getenv("GCLOUD_PROJECT"),
                os.path.exists("/sys/class/dmi/id/product_name") and
                "Google" in open("/sys/class/dmi/id/product_name").read(),
            ]):
                self._cloud_provider = "gcp"
            
            # Azure detection
            elif any([
                os.getenv("AZURE_RESOURCE_GROUP"),
                os.getenv("WEBSITE_SITE_NAME"),  # Azure App Service
                os.path.exists("/sys/class/dmi/id/sys_vendor") and
                "Microsoft Corporation" in open("/sys/class/dmi/id/sys_vendor").read(),
            ]):
                self._cloud_provider = "azure"
        
        return self._cloud_provider
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get comprehensive runtime environment information."""
        return {
            "environment": self.get_environment().value,
            "containerized": self.is_containerized(),
            "kubernetes": self.is_kubernetes(),
            "cloud_provider": self.get_cloud_provider(),
            "python_version": os.sys.version,
            "platform": os.sys.platform,
            "hostname": os.getenv("HOSTNAME", "unknown"),
            "pid": os.getpid(),
            "uid": os.getuid() if hasattr(os, 'getuid') else None,
            "working_directory": os.getcwd(),
            "environment_variables": {
                key: value for key, value in os.environ.items()
                if not any(secret in key.upper() for secret in [
                    "PASSWORD", "SECRET", "KEY", "TOKEN", "CREDENTIAL"
                ])
            }
        }
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate current environment configuration."""
        env = self.get_environment()
        issues = []
        warnings = []
        
        # Production environment checks
        if env.is_production():
            if os.getenv("DEBUG", "false").lower() == "true":
                issues.append("Debug mode enabled in production")
            
            if os.getenv("SECRET_KEY") == "change-me-in-production":
                issues.append("Default secret key used in production")
            
            if not env.requires_https() and not os.getenv("FORCE_HTTPS"):
                warnings.append("HTTPS not enforced in production")
        
        # Development environment checks
        if env.is_development():
            if not os.getenv("DEBUG"):
                warnings.append("Debug mode not explicitly set in development")
        
        # General checks
        required_vars = ["ENVIRONMENT"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            warnings.extend([f"Missing environment variable: {var}" for var in missing_vars])
        
        return {
            "environment": env.value,
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": self._get_recommendations(env)
        }
    
    def _get_recommendations(self, env: Environment) -> list[str]:
        """Get environment-specific recommendations."""
        recommendations = []
        
        if env.is_production():
            recommendations.extend([
                "Use encrypted secrets management",
                "Enable comprehensive logging",
                "Configure health checks",
                "Set up monitoring and alerting",
                "Use connection pooling for databases",
            ])
        
        if env.is_development():
            recommendations.extend([
                "Enable debug mode for easier development",
                "Use local development databases",
                "Enable code reloading",
            ])
        
        if self.is_containerized():
            recommendations.extend([
                "Set resource limits for containers",
                "Use health check endpoints",
                "Configure proper logging drivers",
            ])
        
        if self.is_kubernetes():
            recommendations.extend([
                "Use ConfigMaps for configuration",
                "Use Secrets for sensitive data",
                "Configure readiness and liveness probes",
                "Set up proper RBAC",
            ])
        
        return recommendations


# Global environment manager instance
_env_manager: Optional[EnvironmentManager] = None


def get_environment_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager


def get_environment() -> Environment:
    """Get the current environment."""
    return get_environment_manager().get_environment()


def is_production() -> bool:
    """Check if running in production."""
    return get_environment().is_production()


def is_development() -> bool:
    """Check if running in development."""
    return get_environment().is_development()


def is_testing() -> bool:
    """Check if running in testing."""
    return get_environment().is_testing()