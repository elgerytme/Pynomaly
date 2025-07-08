"""
Secure Adapter Base Class

Provides security-enhanced base functionality for all adapters including:
- Client-side encryption support via adapter parameters
- Secure model serialization and deserialization
- Integrity validation for model artifacts
- Security audit logging for adapter operations
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import pickle
import json

from ..security.security_hardening import (
    SecurityHardeningService,
    SecurityHardeningConfig,
    UploadSecurityMetadata,
    ChecksumValidationResult
)
from ..security.encryption import get_encryption_service

logger = logging.getLogger(__name__)


class SecureAdapterMixin:
    """Mixin class that adds security features to adapters."""
    
    def __init__(self, *args, **kwargs):
        # Extract security parameters from adapter kwargs
        self.encryption_key_id = kwargs.pop('encryption_key_id', None)
        self.enable_client_encryption = kwargs.pop('enable_client_encryption', True)
        self.security_audit_enabled = kwargs.pop('security_audit_enabled', True)
        
        # Initialize security services
        self.security_service = SecurityHardeningService()
        self.encryption_service = get_encryption_service()
        
        # Track security events
        self.security_events: List[Dict[str, Any]] = []
        
        super().__init__(*args, **kwargs)
        
        if self.security_audit_enabled:
            self._log_security_event(
                event_type="adapter_initialization",
                adapter_class=self.__class__.__name__,
                encryption_enabled=self.enable_client_encryption,
                encryption_key_id=self.encryption_key_id
            )
    
    def secure_serialize_model(self, model: Any, model_metadata: Dict[str, Any]) -> Tuple[bytes, UploadSecurityMetadata]:
        """Securely serialize model with encryption and integrity protection."""
        
        try:
            # Serialize the model
            model_data = pickle.dumps(model)
            
            # Add metadata
            serialized_data = {
                'model_data': model_data,
                'metadata': model_metadata,
                'serialization_timestamp': datetime.utcnow().isoformat(),
                'adapter_class': self.__class__.__name__,
                'adapter_version': getattr(self, 'version', '1.0.0')
            }
            
            # Convert to bytes
            json_data = json.dumps(serialized_data, default=str).encode()
            
            # Apply security hardening
            client_info = {
                'adapter_name': self.__class__.__name__,
                'user_agent': f'Pynomaly-Adapter/{getattr(self, "version", "1.0.0")}',
                'encryption_key_id': self.encryption_key_id
            }
            
            security_metadata = self.security_service.secure_upload(
                file_data=json_data,
                file_name=f"model_{model_metadata.get('id', 'unknown')}.pkl",
                content_type="application/octet-stream",
                client_info=client_info,
                encryption_key=self.encryption_key_id
            )
            
            self._log_security_event(
                event_type="model_serialization",
                model_id=model_metadata.get('id'),
                encrypted=security_metadata.client_side_encrypted,
                checksum=security_metadata.checksum,
                file_size=security_metadata.file_size
            )
            
            return json_data, security_metadata
            
        except Exception as e:
            self._log_security_event(
                event_type="model_serialization_error",
                error=str(e),
                model_id=model_metadata.get('id')
            )
            raise
    
    def secure_deserialize_model(
        self, 
        model_data: bytes, 
        expected_metadata: UploadSecurityMetadata
    ) -> Tuple[Any, Dict[str, Any]]:
        """Securely deserialize model with integrity validation."""
        
        try:
            # Validate integrity
            validation_result = self.security_service.validate_upload_integrity(
                file_data=model_data,
                expected_metadata=expected_metadata
            )
            
            if not validation_result.is_valid:
                raise ValueError(f"Model integrity validation failed: {validation_result.error_message}")
            
            # Decrypt if necessary
            if expected_metadata.client_side_encrypted and expected_metadata.encryption_key_id:
                decrypted_data = self.security_service.client_encryption.decrypt_data(
                    model_data, expected_metadata.encryption_key_id
                )
            else:
                decrypted_data = model_data
            
            # Deserialize
            serialized_data = json.loads(decrypted_data.decode())
            model = pickle.loads(serialized_data['model_data'])
            metadata = serialized_data['metadata']
            
            # Validate adapter compatibility
            if serialized_data.get('adapter_class') != self.__class__.__name__:
                logger.warning(
                    f"Model was serialized with different adapter: {serialized_data.get('adapter_class')} "
                    f"vs current: {self.__class__.__name__}"
                )
            
            self._log_security_event(
                event_type="model_deserialization",
                model_id=metadata.get('id'),
                validation_successful=validation_result.is_valid,
                adapter_class=serialized_data.get('adapter_class')
            )
            
            return model, metadata
            
        except Exception as e:
            self._log_security_event(
                event_type="model_deserialization_error",
                error=str(e),
                expected_checksum=expected_metadata.checksum
            )
            raise
    
    def secure_parameter_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize adapter parameters for security."""
        
        sanitized_params = {}
        security_issues = []
        
        for key, value in parameters.items():
            # Check for potentially dangerous parameters
            if key.lower() in ['eval', 'exec', 'import', '__import__', 'open', 'file']:
                security_issues.append(f"Potentially dangerous parameter: {key}")
                continue
            
            # Sanitize string values
            if isinstance(value, str):
                # Remove potentially dangerous characters
                if any(char in value for char in ['<', '>', '&', '"', "'"]):
                    sanitized_value = value.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
                    sanitized_params[key] = sanitized_value
                    if value != sanitized_value:
                        security_issues.append(f"Sanitized parameter {key}: removed dangerous characters")
                else:
                    sanitized_params[key] = value
            else:
                sanitized_params[key] = value
        
        if security_issues:
            self._log_security_event(
                event_type="parameter_sanitization",
                security_issues=security_issues,
                original_param_count=len(parameters),
                sanitized_param_count=len(sanitized_params)
            )
        
        return sanitized_params
    
    def _log_security_event(self, event_type: str, **kwargs) -> None:
        """Log security event for audit trail."""
        
        if not self.security_audit_enabled:
            return
        
        event = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'adapter_class': self.__class__.__name__,
            **kwargs
        }
        
        self.security_events.append(event)
        logger.info(f"Adapter security event: {event}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security configuration and event summary for this adapter."""
        
        return {
            'adapter_class': self.__class__.__name__,
            'encryption_enabled': self.enable_client_encryption,
            'encryption_key_id': self.encryption_key_id,
            'security_audit_enabled': self.security_audit_enabled,
            'security_events_count': len(self.security_events),
            'recent_events': self.security_events[-5:] if self.security_events else [],
            'security_service_summary': self.security_service.get_security_summary()
        }


class SecureDetectorAdapterBase(SecureAdapterMixin, ABC):
    """Base class for secure detector adapters."""
    
    def __init__(self, algorithm: str, **kwargs):
        self.algorithm = algorithm
        super().__init__(**kwargs)
    
    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **kwargs) -> 'SecureDetectorAdapterBase':
        """Fit the detector with security enhancements."""
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """Make predictions with security validation."""
        pass
    
    def secure_fit(self, X: Any, y: Optional[Any] = None, **kwargs) -> 'SecureDetectorAdapterBase':
        """Secure wrapper for fit method with parameter validation."""
        
        # Validate and sanitize parameters
        safe_kwargs = self.secure_parameter_validation(kwargs)
        
        # Log training attempt
        self._log_security_event(
            event_type="model_training_start",
            algorithm=self.algorithm,
            data_shape=getattr(X, 'shape', None),
            has_labels=y is not None,
            parameter_count=len(safe_kwargs)
        )
        
        try:
            # Call the actual fit method
            result = self.fit(X, y, **safe_kwargs)
            
            self._log_security_event(
                event_type="model_training_complete",
                algorithm=self.algorithm,
                success=True
            )
            
            return result
            
        except Exception as e:
            self._log_security_event(
                event_type="model_training_error",
                algorithm=self.algorithm,
                error=str(e)
            )
            raise
    
    def secure_predict(self, X: Any, **kwargs) -> Any:
        """Secure wrapper for predict method with validation."""
        
        # Validate and sanitize parameters
        safe_kwargs = self.secure_parameter_validation(kwargs)
        
        # Log prediction attempt
        self._log_security_event(
            event_type="prediction_start",
            algorithm=self.algorithm,
            data_shape=getattr(X, 'shape', None),
            parameter_count=len(safe_kwargs)
        )
        
        try:
            # Call the actual predict method
            result = self.predict(X, **safe_kwargs)
            
            self._log_security_event(
                event_type="prediction_complete",
                algorithm=self.algorithm,
                success=True,
                result_shape=getattr(result, 'shape', None)
            )
            
            return result
            
        except Exception as e:
            self._log_security_event(
                event_type="prediction_error",
                algorithm=self.algorithm,
                error=str(e)
            )
            raise


# Security-enhanced configuration for adapters
class SecureAdapterConfig:
    """Configuration class for secure adapter settings."""
    
    def __init__(
        self,
        enable_client_encryption: bool = True,
        encryption_key_id: Optional[str] = None,
        security_audit_enabled: bool = True,
        parameter_validation_enabled: bool = True,
        integrity_check_enabled: bool = True
    ):
        self.enable_client_encryption = enable_client_encryption
        self.encryption_key_id = encryption_key_id
        self.security_audit_enabled = security_audit_enabled
        self.parameter_validation_enabled = parameter_validation_enabled
        self.integrity_check_enabled = integrity_check_enabled
    
    def to_adapter_kwargs(self) -> Dict[str, Any]:
        """Convert configuration to kwargs for adapter initialization."""
        return {
            'enable_client_encryption': self.enable_client_encryption,
            'encryption_key_id': self.encryption_key_id,
            'security_audit_enabled': self.security_audit_enabled,
            'parameter_validation_enabled': self.parameter_validation_enabled,
            'integrity_check_enabled': self.integrity_check_enabled
        }


def create_secure_adapter(
    adapter_class: type, 
    config: Optional[SecureAdapterConfig] = None,
    **kwargs
) -> Any:
    """Factory function to create security-enhanced adapters."""
    
    if config is None:
        config = SecureAdapterConfig()
    
    # Combine config and additional kwargs
    adapter_kwargs = {**config.to_adapter_kwargs(), **kwargs}
    
    # Create the adapter with security enhancements
    return adapter_class(**adapter_kwargs)
