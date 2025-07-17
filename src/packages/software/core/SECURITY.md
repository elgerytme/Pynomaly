# Security Policy - Core Package

## Overview

The Core package contains the fundamental business logic and domain models that drive the Pynomaly platform. As the foundation layer following Clean Architecture principles, security is critical to ensure the integrity of business rules, data validation, and system behavior.

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | End of Life    |
| ------- | ------------------ | -------------- |
| 2.x.x   | :white_check_mark: | -              |
| 1.9.x   | :white_check_mark: | 2025-06-01     |
| 1.8.x   | :warning:          | 2024-12-31     |
| < 1.8   | :x:                | Ended          |

## Security Model

### Core Security Domains

Our security model addresses these key areas for the foundational business logic layer:

**1. Domain Logic Security**
- Business rule integrity and validation
- Domain invariant enforcement
- Input validation and sanitization
- Type safety and data integrity

**2. Data Model Security**
- Entity state consistency
- Value object immutability
- Aggregate boundary protection
- Data validation and constraints

**3. Application Layer Security**
- Use case authorization patterns
- Input validation at application boundaries
- Error handling without information leakage
- Service interface protection

**4. Architectural Security**
- Clean Architecture compliance
- Dependency inversion maintenance
- Layer boundary enforcement
- Abstraction integrity

## Threat Model

### High-Risk Scenarios

**Domain Logic Attacks**
- Business rule bypass attempts
- Domain invariant violations
- Data consistency corruption
- State manipulation attacks

**Input Validation Attacks**
- Injection attacks through domain inputs
- Type confusion vulnerabilities
- Boundary condition exploits
- Serialization/deserialization attacks

**Application Layer Attacks**
- Use case privilege escalation
- Business workflow manipulation
- Unauthorized state transitions
- Service interface abuse

**Architectural Violations**
- Clean Architecture boundary violations
- Dependency injection attacks
- Layer separation bypasses
- Abstraction leakage exploits

## Security Features

### Domain Security

**Entity Validation and Invariants**
```python
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from core.domain.value_objects import AnomalyScore, EntityId
from core.domain.exceptions import DomainSecurityError
from core.shared.validation import validate_not_empty, validate_range

@dataclass
class Detector:
    """Secure domain entity with comprehensive validation."""
    
    id: EntityId = field()
    name: str = field()
    algorithm: str = field()
    contamination_rate: float = field()
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate all domain invariants on creation."""
        self._validate_security_constraints()
    
    def _validate_security_constraints(self) -> None:
        """Enforce security-critical domain constraints."""
        # Input validation with security focus
        validate_not_empty(self.name, "Detector name")
        validate_not_empty(self.algorithm, "Algorithm name")
        
        # Prevent injection attacks through names
        if not self._is_safe_identifier(self.name):
            raise DomainSecurityError(f"Invalid detector name: {self.name}")
        
        if not self._is_safe_identifier(self.algorithm):
            raise DomainSecurityError(f"Invalid algorithm name: {self.algorithm}")
        
        # Validate numeric ranges to prevent overflow/underflow
        validate_range(
            self.contamination_rate, 
            min_val=0.0, 
            max_val=1.0,
            field_name="contamination_rate"
        )
    
    def _is_safe_identifier(self, identifier: str) -> bool:
        """Validate identifier is safe from injection attacks."""
        import re
        # Only allow alphanumeric, underscore, hyphen
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, identifier)) and len(identifier) <= 100
    
    def update_configuration(
        self, 
        name: Optional[str] = None,
        contamination_rate: Optional[float] = None
    ) -> None:
        """Securely update detector configuration."""
        # Validate before any changes
        if name is not None:
            validate_not_empty(name, "Detector name")
            if not self._is_safe_identifier(name):
                raise DomainSecurityError(f"Invalid detector name: {name}")
            self.name = name
        
        if contamination_rate is not None:
            validate_range(
                contamination_rate, 
                min_val=0.0, 
                max_val=1.0,
                field_name="contamination_rate"
            )
            self.contamination_rate = contamination_rate
```

**Value Object Security**
```python
from typing import Union
from dataclasses import dataclass

@dataclass(frozen=True)
class AnomalyScore:
    """Immutable and secure anomaly score value object."""
    
    value: float
    
    def __post_init__(self) -> None:
        """Validate value with security considerations."""
        self._validate_security()
    
    def _validate_security(self) -> None:
        """Comprehensive security validation."""
        # Type safety check
        if not isinstance(self.value, (int, float)):
            raise DomainSecurityError(
                f"Score must be numeric, got {type(self.value)}"
            )
        
        # Range validation to prevent exploitation
        if not (0.0 <= self.value <= 1.0):
            raise DomainSecurityError(
                f"Score must be between 0.0 and 1.0, got {self.value}"
            )
        
        # Check for special float values that could cause issues
        import math
        if math.isnan(self.value) or math.isinf(self.value):
            raise DomainSecurityError(
                f"Score cannot be NaN or infinite, got {self.value}"
            )
    
    def is_anomalous(self, threshold: float = 0.5) -> bool:
        """Safely determine anomaly status with validation."""
        # Validate threshold to prevent manipulation
        if not isinstance(threshold, (int, float)):
            raise DomainSecurityError("Threshold must be numeric")
        
        if not (0.0 <= threshold <= 1.0):
            raise DomainSecurityError("Threshold must be between 0.0 and 1.0")
        
        return self.value >= threshold
```

**Secure Input Validation**
```python
from typing import Any, Optional, Union, Type
import re
from decimal import Decimal

class SecureValidator:
    """Secure validation utilities for domain inputs."""
    
    @staticmethod
    def validate_string_input(
        value: Any,
        field_name: str,
        max_length: int = 255,
        allow_empty: bool = False,
        pattern: Optional[str] = None
    ) -> str:
        """Validate string input with security checks."""
        # Type validation
        if not isinstance(value, str):
            raise DomainSecurityError(
                f"{field_name} must be string, got {type(value)}"
            )
        
        # Length validation to prevent DoS
        if len(value) > max_length:
            raise DomainSecurityError(
                f"{field_name} exceeds maximum length {max_length}"
            )
        
        # Empty check
        if not allow_empty and not value.strip():
            raise DomainSecurityError(f"{field_name} cannot be empty")
        
        # Pattern validation for injection prevention
        if pattern and not re.match(pattern, value):
            raise DomainSecurityError(
                f"{field_name} contains invalid characters"
            )
        
        # Check for control characters and null bytes
        if any(ord(char) < 32 for char in value if char not in '\t\n\r'):
            raise DomainSecurityError(
                f"{field_name} contains invalid control characters"
            )
        
        return value
    
    @staticmethod
    def validate_numeric_input(
        value: Any,
        field_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        precision: Optional[int] = None
    ) -> Union[int, float]:
        """Validate numeric input with overflow protection."""
        # Type validation
        if not isinstance(value, (int, float, Decimal)):
            raise DomainSecurityError(
                f"{field_name} must be numeric, got {type(value)}"
            )
        
        # Convert to appropriate type
        if isinstance(value, Decimal):
            value = float(value)
        
        # Check for special values
        import math
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise DomainSecurityError(
                f"{field_name} cannot be NaN or infinite"
            )
        
        # Range validation
        if min_val is not None and value < min_val:
            raise DomainSecurityError(
                f"{field_name} must be >= {min_val}, got {value}"
            )
        
        if max_val is not None and value > max_val:
            raise DomainSecurityError(
                f"{field_name} must be <= {max_val}, got {value}"
            )
        
        # Precision validation for floats
        if precision is not None and isinstance(value, float):
            decimal_places = len(str(value).split('.')[-1])
            if decimal_places > precision:
                raise DomainSecurityError(
                    f"{field_name} precision exceeds {precision} decimal places"
                )
        
        return value
```

### Application Layer Security

**Secure Use Cases**
```python
from typing import Protocol
from abc import abstractmethod

from core.domain.entities import Dataset, Detector, DetectionResult
from core.domain.repositories import DatasetRepository, DetectorRepository
from core.application.security import AuthorizationService, AuditLogger
from core.shared.exceptions import SecurityError, AuthorizationError

class SecureDetectAnomaliesUseCase:
    """Secure implementation of anomaly detection use case."""
    
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        detector_repository: DetectorRepository,
        authorization_service: AuthorizationService,
        audit_logger: AuditLogger
    ) -> None:
        self._dataset_repository = dataset_repository
        self._detector_repository = detector_repository
        self._authorization_service = authorization_service
        self._audit_logger = audit_logger
    
    async def execute(
        self,
        dataset_id: str,
        detector_id: str,
        user_context: UserContext
    ) -> DetectionResult:
        """Execute anomaly detection with comprehensive security."""
        operation_id = self._generate_operation_id()
        
        try:
            # Input validation with security focus
            self._validate_inputs(dataset_id, detector_id, user_context)
            
            # Authorization check
            await self._authorize_operation(
                user_context, dataset_id, detector_id, operation_id
            )
            
            # Load and validate domain objects
            dataset = await self._load_and_validate_dataset(
                dataset_id, user_context
            )
            detector = await self._load_and_validate_detector(
                detector_id, user_context
            )
            
            # Execute business logic with monitoring
            result = await self._execute_detection_with_monitoring(
                dataset, detector, user_context, operation_id
            )
            
            # Audit successful operation
            await self._audit_logger.log_success(
                operation="anomaly_detection",
                user=user_context.user_id,
                resources=[dataset_id, detector_id],
                operation_id=operation_id
            )
            
            return result
            
        except Exception as e:
            # Audit failed operation
            await self._audit_logger.log_failure(
                operation="anomaly_detection",
                user=user_context.user_id,
                error=str(e),
                operation_id=operation_id
            )
            raise
    
    def _validate_inputs(
        self, 
        dataset_id: str, 
        detector_id: str, 
        user_context: UserContext
    ) -> None:
        """Validate all inputs with security checks."""
        # Validate IDs to prevent injection
        self._validate_entity_id(dataset_id, "dataset_id")
        self._validate_entity_id(detector_id, "detector_id")
        
        # Validate user context
        if not user_context or not user_context.user_id:
            raise SecurityError("Invalid user context")
    
    def _validate_entity_id(self, entity_id: str, field_name: str) -> None:
        """Validate entity ID format to prevent injection."""
        import re
        
        if not isinstance(entity_id, str):
            raise SecurityError(f"{field_name} must be string")
        
        if len(entity_id) > 100:
            raise SecurityError(f"{field_name} too long")
        
        # UUID or similar safe format
        if not re.match(r'^[a-zA-Z0-9_-]+$', entity_id):
            raise SecurityError(f"Invalid {field_name} format")
    
    async def _authorize_operation(
        self,
        user_context: UserContext,
        dataset_id: str,
        detector_id: str,
        operation_id: str
    ) -> None:
        """Authorize the detection operation."""
        # Check user permissions
        if not await self._authorization_service.can_access_dataset(
            user_context.user_id, dataset_id
        ):
            raise AuthorizationError(f"Access denied to dataset {dataset_id}")
        
        if not await self._authorization_service.can_use_detector(
            user_context.user_id, detector_id
        ):
            raise AuthorizationError(f"Access denied to detector {detector_id}")
        
        # Check rate limits
        if not await self._authorization_service.check_rate_limit(
            user_context.user_id, "anomaly_detection"
        ):
            raise SecurityError("Rate limit exceeded")
```

### Secure Error Handling

**Information Leakage Prevention**
```python
from typing import Optional, Dict, Any
import logging
from datetime import datetime

class SecureErrorHandler:
    """Handle errors without leaking sensitive information."""
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def handle_domain_error(
        self, 
        error: Exception, 
        context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle domain errors securely."""
        # Generate safe error ID for tracking
        error_id = self._generate_error_id()
        
        # Log full error details securely
        self._logger.error(
            "Domain error occurred",
            extra={
                "error_id": error_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": self._sanitize_context(context),
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Return sanitized error response
        return {
            "error": True,
            "error_id": error_id,
            "message": self._get_safe_error_message(error),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_safe_error_message(self, error: Exception) -> str:
        """Get user-safe error message without sensitive details."""
        if isinstance(error, DomainSecurityError):
            return "Invalid input provided"
        elif isinstance(error, AuthorizationError):
            return "Access denied"
        elif isinstance(error, ValidationError):
            return "Validation failed"
        else:
            return "An error occurred while processing your request"
    
    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from context."""
        sensitive_keys = {
            'password', 'token', 'key', 'secret', 'credential',
            'api_key', 'auth', 'session', 'cookie'
        }
        
        sanitized = {}
        for key, value in context.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:100] + "...[TRUNCATED]"
            else:
                sanitized[key] = value
        
        return sanitized
```

## Security Best Practices

### Development

**Secure Coding Standards**
- Input validation at all domain boundaries
- Comprehensive type checking with mypy strict mode
- Immutable value objects to prevent state corruption
- Defensive programming with assertion checks
- Proper error handling without information leakage

**Domain Security Patterns**
- Factory methods for secure object creation
- Builder patterns with validation steps
- Command pattern for auditable operations
- Specification pattern for secure business rules
- Repository pattern with access control

**Testing Security**
- Security-focused unit tests for all validation logic
- Property-based testing for domain invariants
- Fuzz testing for input validation
- Boundary condition testing with malicious inputs
- Integration tests for security controls

### Architecture Security

**Clean Architecture Compliance**
- Strict dependency inversion enforcement
- Layer boundary validation in CI/CD
- Interface segregation for security
- Single responsibility for each component
- Open/closed principle with secure extensions

**Dependency Management**
- Minimal external dependencies in domain layer
- Regular security audits of dependencies
- Pin dependency versions in production
- Automated vulnerability scanning
- Dependency injection container security

### Production Security

**Runtime Protection**
- Input sanitization at application boundaries
- Business rule enforcement at all times
- Resource consumption monitoring
- Performance-based DoS protection
- Comprehensive audit logging

**Monitoring and Alerting**
- Domain security violation detection
- Unusual pattern recognition
- Performance anomaly detection
- Error rate monitoring
- Security event correlation

## Vulnerability Reporting

### Reporting Process

Domain-level security vulnerabilities can have widespread impact across the platform.

**1. Critical Domain Vulnerabilities**
- Business rule bypass vulnerabilities
- Data integrity violations
- Authentication/authorization bypasses
- Input validation failures leading to corruption

**2. Contact Security Team**
- Email: core-security@yourorg.com
- PGP Key: [Provide core security PGP key]
- Include "Core Package Security Vulnerability" in the subject line

**3. Provide Detailed Information**
```
Subject: Core Package Security Vulnerability - [Brief Description]

Vulnerability Details:
- Component affected: [e.g., domain entity, value object, use case]
- Vulnerability type: [e.g., input validation, business rule bypass]
- Severity level: [Critical/High/Medium/Low]
- Attack vector: [How the vulnerability can be exploited]
- Business impact: [What business rules could be violated]
- Reproduction steps: [Detailed steps to reproduce]
- Proof of concept: [If available, but avoid causing data corruption]
- Suggested fix: [If you have recommendations]

Environment Information:
- Core package version: [Version number]
- Python version: [Version]
- Related dependencies: [Pydantic, etc.]
- Operating system: [OS and version]
```

### Response Timeline

**Critical Domain Vulnerabilities**
- **Acknowledgment**: Within 2 hours
- **Initial Assessment**: Within 6 hours
- **Emergency Response**: Within 12 hours if actively exploited
- **Resolution Timeline**: 24-48 hours depending on complexity

**High/Medium Severity**
- **Acknowledgment**: Within 8 hours
- **Initial Assessment**: Within 24 hours
- **Detailed Analysis**: Within 72 hours
- **Resolution Timeline**: 1-2 weeks depending on impact

## Security Configuration

### Secure Development Configuration

**Type Safety Configuration**
```ini
# mypy.ini
[mypy]
python_version = 3.11
strict = True
warn_unused_configs = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_return_any = True
warn_unreachable = True
check_untyped_defs = True
disallow_untyped_defs = True
disallow_any_generics = True
disallow_subclassing_any = True
```

**Validation Configuration**
```python
# Security-focused validation settings
from pydantic import BaseSettings

class CoreSecuritySettings(BaseSettings):
    """Security configuration for core package."""
    
    # Input validation
    max_string_length: int = 1000
    max_collection_size: int = 10000
    validate_assignment: bool = True
    
    # Business rule enforcement
    strict_mode: bool = True
    validate_invariants: bool = True
    
    # Logging
    log_security_events: bool = True
    log_level: str = "INFO"
    
    # Performance limits
    max_processing_time_seconds: int = 30
    max_memory_usage_mb: int = 500
```

**Test Security Configuration**
```python
# pytest.ini
[tool:pytest]
addopts = 
    --strict-markers
    --strict-config
    --cov=core
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=95
    -p no:warnings
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## Compliance and Auditing

### Security Standards

**Code Security Standards**
- OWASP secure coding practices
- SANS secure development guidelines
- Input validation best practices
- Error handling security patterns

**Architecture Security Standards**
- Clean Architecture security patterns
- Domain-Driven Design security principles
- SOLID principles with security considerations
- Dependency injection security patterns

### Audit Procedures

**Regular Security Reviews**
- Monthly code security audits
- Quarterly architecture security assessments
- Annual penetration testing simulation
- Continuous static analysis scanning

**Compliance Documentation**
- Security control implementation records
- Code review security checklists
- Security test coverage reports
- Vulnerability assessment records

## Contact Information

**Core Security Team**
- Email: core-security@yourorg.com
- Emergency Phone: [Emergency contact for critical domain vulnerabilities]
- PGP Key: [Core security PGP key fingerprint]

**Escalation Contacts**
- Core Package Maintainer: [Contact information]
- Security Architect: [Contact information]
- Domain Expert: [Contact information]

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Next Review**: March 2025