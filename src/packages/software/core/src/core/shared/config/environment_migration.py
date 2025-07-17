"""Environment variable migration mapping from anomaly-specific to generic names."""

from typing import Dict

# Mapping from old PYNOMALY_* variables to new generic PLATFORM_* variables
ENVIRONMENT_VARIABLE_MIGRATION: Dict[str, str] = {
    # Core application settings
    "PYNOMALY_ENV": "PLATFORM_ENV",
    "PYNOMALY_ENVIRONMENT": "PLATFORM_ENVIRONMENT", 
    "PYNOMALY_DEBUG": "PLATFORM_DEBUG",
    "PYNOMALY_LOG_LEVEL": "PLATFORM_LOG_LEVEL",
    "PYNOMALY_VERSION": "PLATFORM_VERSION",
    
    # Security and authentication
    "PYNOMALY_SECRET_KEY": "PLATFORM_SECRET_KEY",
    "PYNOMALY_JWT_SECRET": "PLATFORM_JWT_SECRET", 
    "PYNOMALY_AUTH_ENABLED": "PLATFORM_AUTH_ENABLED",
    "PYNOMALY_SECURITY_SMTP_SERVER": "PLATFORM_SECURITY_SMTP_SERVER",
    "PYNOMALY_SECURITY_SMTP_USERNAME": "PLATFORM_SECURITY_SMTP_USERNAME",
    "PYNOMALY_SECURITY_SMTP_PASSWORD": "PLATFORM_SECURITY_SMTP_PASSWORD",
    "PYNOMALY_SECURITY_SENDER_EMAIL": "PLATFORM_SECURITY_SENDER_EMAIL",
    
    # Database and storage
    "PYNOMALY_DATABASE_URL": "PLATFORM_DATABASE_URL",
    "PYNOMALY_REDIS_URL": "PLATFORM_REDIS_URL",
    "PYNOMALY_STORAGE_BUCKET": "PLATFORM_STORAGE_BUCKET",
    
    # API and networking
    "PYNOMALY_API_HOST": "PLATFORM_API_HOST",
    "PYNOMALY_API_PORT": "PLATFORM_API_PORT",
    "PYNOMALY_API_PREFIX": "PLATFORM_API_PREFIX",
    "PYNOMALY_CORS_ORIGINS": "PLATFORM_CORS_ORIGINS",
    
    # Feature flags
    "PYNOMALY_CACHE_ENABLED": "PLATFORM_CACHE_ENABLED",
    "PYNOMALY_MONITORING_ENABLED": "PLATFORM_MONITORING_ENABLED", 
    "PYNOMALY_MONITORING_METRICS_ENABLED": "PLATFORM_MONITORING_METRICS_ENABLED",
    "PYNOMALY_TELEMETRY_ENABLED": "PLATFORM_TELEMETRY_ENABLED",
    
    # Performance and scaling
    "PYNOMALY_MAX_WORKERS": "PLATFORM_MAX_WORKERS",
    "PYNOMALY_TIMEOUT": "PLATFORM_TIMEOUT",
    "PYNOMALY_RATE_LIMIT": "PLATFORM_RATE_LIMIT",
    
    # ML/AI specific (keep these generic for any algorithm type)
    "PYNOMALY_MODEL_CACHE_DIR": "PLATFORM_MODEL_CACHE_DIR",
    "PYNOMALY_ALGORITHM_TIMEOUT": "PLATFORM_ALGORITHM_TIMEOUT",
    "PYNOMALY_ENABLE_GPU": "PLATFORM_ENABLE_GPU",
    "PYNOMALY_MEMORY_LIMIT": "PLATFORM_MEMORY_LIMIT",
}

# Reverse mapping for backward compatibility
REVERSE_MIGRATION_MAP = {v: k for k, v in ENVIRONMENT_VARIABLE_MIGRATION.items()}


def migrate_environment_variable(old_var: str) -> str:
    """Migrate an old PYNOMALY_* variable name to new PLATFORM_* name.
    
    Args:
        old_var: Old environment variable name
        
    Returns:
        New environment variable name
    """
    return ENVIRONMENT_VARIABLE_MIGRATION.get(old_var, old_var)


def get_legacy_variable_name(new_var: str) -> str:
    """Get the legacy PYNOMALY_* name for a PLATFORM_* variable.
    
    Args:
        new_var: New environment variable name
        
    Returns:
        Legacy environment variable name
    """
    return REVERSE_MIGRATION_MAP.get(new_var, new_var)


def get_all_pynomaly_variables() -> list[str]:
    """Get list of all PYNOMALY_* variables that should be migrated.
    
    Returns:
        List of old variable names
    """
    return list(ENVIRONMENT_VARIABLE_MIGRATION.keys())


def get_all_platform_variables() -> list[str]:
    """Get list of all new PLATFORM_* variable names.
    
    Returns:
        List of new variable names
    """
    return list(ENVIRONMENT_VARIABLE_MIGRATION.values())