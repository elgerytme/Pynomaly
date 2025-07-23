"""Configuration validation utilities."""

from __future__ import annotations

import re
import ipaddress
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse


class ConfigValidator:
    """Validates infrastructure configuration values."""
    
    @staticmethod
    def validate_url(url: str, schemes: Optional[List[str]] = None) -> bool:
        """Validate URL format and scheme."""
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            
            # Check if scheme is present
            if not parsed.scheme:
                return False
            
            # Check if netloc (domain/host) is present
            if not parsed.netloc:
                return False
            
            # Check allowed schemes
            if schemes and parsed.scheme not in schemes:
                return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_database_url(url: str) -> bool:
        """Validate database URL format."""
        valid_schemes = [
            "postgresql", "postgres", "mysql", "sqlite", 
            "oracle", "mssql", "mongodb", "redis"
        ]
        return ConfigValidator.validate_url(url, valid_schemes)
    
    @staticmethod
    def validate_redis_url(url: str) -> bool:
        """Validate Redis URL format."""
        valid_schemes = ["redis", "rediss"]
        return ConfigValidator.validate_url(url, valid_schemes)
    
    @staticmethod
    def validate_message_broker_url(url: str) -> bool:
        """Validate message broker URL format."""
        valid_schemes = [
            "redis", "rediss", "amqp", "amqps", 
            "kafka", "sqs", "mongodb", "filesystem"
        ]
        return ConfigValidator.validate_url(url, valid_schemes)
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address format (IPv4 or IPv6)."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_port(port: Union[int, str]) -> bool:
        """Validate port number (1-65535)."""
        try:
            port_num = int(port)
            return 1 <= port_num <= 65535
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_log_level(level: str) -> bool:
        """Validate log level."""
        valid_levels = [
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
            "NOTSET"
        ]
        return level.upper() in valid_levels
    
    @staticmethod
    def validate_environment(env: str) -> bool:
        """Validate environment name."""
        valid_environments = [
            "development", "testing", "staging", "production",
            "dev", "test", "stage", "prod"
        ]
        return env.lower() in valid_environments
    
    @staticmethod
    def validate_secret_key(key: str, min_length: int = 32) -> bool:
        """Validate secret key strength."""
        if not key or len(key) < min_length:
            return False
        
        # Check for common weak keys
        weak_keys = [
            "change-me", "secret-key", "my-secret", "password",
            "123456", "admin", "default", "test"
        ]
        
        return key.lower() not in weak_keys
    
    @staticmethod
    def validate_jwt_algorithm(algorithm: str) -> bool:
        """Validate JWT algorithm."""
        valid_algorithms = [
            "HS256", "HS384", "HS512",
            "RS256", "RS384", "RS512",
            "ES256", "ES384", "ES512",
            "PS256", "PS384", "PS512",
        ]
        return algorithm in valid_algorithms
    
    @staticmethod
    def validate_cron_expression(expression: str) -> bool:
        """Validate cron expression format."""
        if not expression:
            return False
        
        # Basic cron validation (5 or 6 fields)
        parts = expression.split()
        if len(parts) not in [5, 6]:
            return False
        
        # Validate each field using regex patterns
        patterns = [
            r'^(\*|[0-5]?\d)$',  # minute (0-59)
            r'^(\*|[01]?\d|2[0-3])$',  # hour (0-23)
            r'^(\*|[12]?\d|3[01])$',  # day of month (1-31)
            r'^(\*|[01]?\d)$',  # month (1-12)
            r'^(\*|[0-6])$',  # day of week (0-6)
        ]
        
        # If 6 fields, first is seconds
        if len(parts) == 6:
            patterns.insert(0, r'^(\*|[0-5]?\d)$')  # second (0-59)
        
        for i, (part, pattern) in enumerate(zip(parts, patterns)):
            # Handle ranges (e.g., 1-5), lists (e.g., 1,3,5), and steps (e.g., */5)
            if not re.match(pattern, part) and not self._validate_cron_field(part):
                return False
        
        return True
    
    @staticmethod
    def _validate_cron_field(field: str) -> bool:
        """Validate complex cron field expressions."""
        # Handle lists (comma-separated)
        if ',' in field:
            return all(ConfigValidator._validate_cron_field(f.strip()) 
                      for f in field.split(','))
        
        # Handle ranges (dash-separated)
        if '-' in field:
            parts = field.split('-')
            if len(parts) == 2:
                try:
                    start, end = int(parts[0]), int(parts[1])
                    return start <= end
                except ValueError:
                    return False
        
        # Handle steps (slash-separated)
        if '/' in field:
            parts = field.split('/')
            if len(parts) == 2:
                base, step = parts[0], parts[1]
                try:
                    int(step)
                    return base == '*' or ConfigValidator._validate_cron_field(base)
                except ValueError:
                    return False
        
        return False
    
    @staticmethod
    def validate_configuration(config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate complete configuration and return errors."""
        errors = {}
        
        # Database validation
        if 'database' in config:
            db_config = config['database'] 
            db_errors = []
            
            if 'url' in db_config and not ConfigValidator.validate_database_url(db_config['url']):
                db_errors.append("Invalid database URL format")
            
            if 'pool_size' in db_config and not isinstance(db_config['pool_size'], int):
                db_errors.append("Pool size must be an integer")
            
            if db_errors:
                errors['database'] = db_errors
        
        # Redis validation
        if 'redis' in config:
            redis_config = config['redis']
            redis_errors = []
            
            if 'url' in redis_config and not ConfigValidator.validate_redis_url(redis_config['url']):
                redis_errors.append("Invalid Redis URL format")
            
            if 'port' in redis_config and not ConfigValidator.validate_port(redis_config['port']):
                redis_errors.append("Invalid Redis port number")
            
            if redis_errors:
                errors['redis'] = redis_errors
        
        # Security validation
        if 'security' in config:
            security_config = config['security']
            security_errors = []
            
            if 'secret_key' in security_config and not ConfigValidator.validate_secret_key(security_config['secret_key']):
                security_errors.append("Weak or invalid secret key")
            
            if 'jwt_algorithm' in security_config and not ConfigValidator.validate_jwt_algorithm(security_config['jwt_algorithm']):
                security_errors.append("Invalid JWT algorithm")
            
            if security_errors:
                errors['security'] = security_errors
        
        # Monitoring validation
        if 'monitoring' in config:
            monitoring_config = config['monitoring']
            monitoring_errors = []
            
            if 'log_level' in monitoring_config and not ConfigValidator.validate_log_level(monitoring_config['log_level']):
                monitoring_errors.append("Invalid log level")
            
            if 'metrics_port' in monitoring_config and not ConfigValidator.validate_port(monitoring_config['metrics_port']):
                monitoring_errors.append("Invalid metrics port number")
            
            if monitoring_errors:
                errors['monitoring'] = monitoring_errors
        
        # Environment validation
        if 'environment' in config and not ConfigValidator.validate_environment(config['environment']):
            errors['general'] = errors.get('general', [])
            errors['general'].append("Invalid environment name")
        
        return errors
    
    @staticmethod
    def get_validation_summary(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        errors = ConfigValidator.validate_configuration(config)
        
        total_errors = sum(len(error_list) for error_list in errors.values())
        
        return {
            "valid": total_errors == 0,
            "total_errors": total_errors,
            "errors_by_section": errors,
            "all_errors": [
                f"{section}: {error}" 
                for section, error_list in errors.items() 
                for error in error_list
            ],
            "validation_timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        }