"""
Security Hardening Service

Provides comprehensive security hardening features including:
- TLS enforcement and minimum SDK version requirements
- Checksum validation for uploads using ETag/MD5
- Client-side encryption key support via adapter parameters
- Security configuration validation
"""

from __future__ import annotations

import hashlib
import logging
import ssl
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from packaging import version

from .encryption import DataEncryption, EncryptionConfig, EncryptionService

logger = logging.getLogger(__name__)


class TLSVersion(Enum):
    """Supported TLS versions."""
    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"


class ChecksumAlgorithm(Enum):
    """Supported checksum algorithms."""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"


@dataclass
class SecurityHardeningConfig:
    """Configuration for security hardening."""
    
    # TLS Configuration
    enforce_tls: bool = True
    minimum_tls_version: TLSVersion = TLSVersion.TLS_1_2
    verify_ssl_certificates: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # SDK Version Requirements
    enforce_minimum_sdk_version: bool = True
    minimum_sdk_version: str = "1.0.0"
    minimum_python_version: str = "3.8"
    
    # Checksum Validation
    enable_checksum_validation: bool = True
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.SHA256
    enable_etag_validation: bool = True
    enable_md5_validation: bool = True
    
    # Client-side Encryption
    enable_client_side_encryption: bool = True
    encryption_config: Optional[EncryptionConfig] = None
    require_encrypted_uploads: bool = False
    
    # Security Headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    })
    
    # Rate Limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    max_upload_size_mb: int = 100
    
    # Audit Settings
    enable_security_audit: bool = True
    audit_failed_attempts: bool = True
    audit_checksum_failures: bool = True


@dataclass
class ChecksumValidationResult:
    """Result of checksum validation."""
    is_valid: bool
    expected_checksum: str
    actual_checksum: str
    algorithm: ChecksumAlgorithm
    validation_time: datetime
    error_message: Optional[str] = None


@dataclass
class UploadSecurityMetadata:
    """Security metadata for file uploads."""
    checksum: str
    checksum_algorithm: ChecksumAlgorithm
    file_size: int
    content_type: str
    encryption_key_id: Optional[str] = None
    client_side_encrypted: bool = False
    upload_timestamp: datetime = field(default_factory=datetime.utcnow)
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None


class TLSEnforcer:
    """Enforces TLS requirements for connections."""
    
    def __init__(self, config: SecurityHardeningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with security requirements."""
        
        # Create context with required TLS version
        if self.config.minimum_tls_version == TLSVersion.TLS_1_3:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_3
        else:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            
        # Disable insecure protocols
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        
        # Set secure cipher suites
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS")
        
        # Configure certificate verification
        if self.config.verify_ssl_certificates:
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            self.logger.warning("SSL certificate verification disabled")
            
        # Load custom certificates if provided
        if self.config.ssl_cert_path and self.config.ssl_key_path:
            try:
                context.load_cert_chain(self.config.ssl_cert_path, self.config.ssl_key_path)
                self.logger.info("Loaded custom SSL certificates")
            except Exception as e:
                self.logger.error(f"Failed to load SSL certificates: {e}")
                raise
                
        return context
        
    def validate_tls_connection(self, url: str) -> bool:
        """Validate that a URL uses proper TLS."""
        
        if not self.config.enforce_tls:
            return True
            
        parsed_url = urlparse(url)
        
        # Check if HTTPS is used
        if parsed_url.scheme != "https":
            self.logger.error(f"Non-HTTPS URL rejected: {url}")
            return False
            
        # Test TLS connection
        try:
            context = self.create_ssl_context()
            response = requests.get(url, timeout=10, verify=context)
            response.raise_for_status()
            
            # Check TLS version in response headers
            if hasattr(response.raw, 'version'):
                tls_version = response.raw.version
                self.logger.info(f"TLS version: {tls_version}")
                
            return True
            
        except requests.exceptions.SSLError as e:
            self.logger.error(f"TLS validation failed for {url}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Connection validation failed for {url}: {e}")
            return False


class ChecksumValidator:
    """Validates file checksums for uploads."""
    
    def __init__(self, config: SecurityHardeningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_checksum(self, data: bytes, algorithm: ChecksumAlgorithm) -> str:
        """Calculate checksum for data."""
        
        if algorithm == ChecksumAlgorithm.MD5:
            hasher = hashlib.md5()
        elif algorithm == ChecksumAlgorithm.SHA1:
            hasher = hashlib.sha1()
        elif algorithm == ChecksumAlgorithm.SHA256:
            hasher = hashlib.sha256()
        elif algorithm == ChecksumAlgorithm.SHA512:
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
            
        hasher.update(data)
        return hasher.hexdigest()
        
    def calculate_file_checksum(self, file_path: Path, algorithm: ChecksumAlgorithm) -> str:
        """Calculate checksum for a file."""
        
        if algorithm == ChecksumAlgorithm.MD5:
            hasher = hashlib.md5()
        elif algorithm == ChecksumAlgorithm.SHA1:
            hasher = hashlib.sha1()
        elif algorithm == ChecksumAlgorithm.SHA256:
            hasher = hashlib.sha256()
        elif algorithm == ChecksumAlgorithm.SHA512:
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
            
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
                
        return hasher.hexdigest()
        
    def validate_checksum(
        self, 
        data: bytes, 
        expected_checksum: str, 
        algorithm: ChecksumAlgorithm
    ) -> ChecksumValidationResult:
        """Validate data checksum."""
        
        actual_checksum = self.calculate_checksum(data, algorithm)
        is_valid = actual_checksum.lower() == expected_checksum.lower()
        
        result = ChecksumValidationResult(
            is_valid=is_valid,
            expected_checksum=expected_checksum,
            actual_checksum=actual_checksum,
            algorithm=algorithm,
            validation_time=datetime.utcnow()
        )
        
        if not is_valid:
            result.error_message = f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
            self.logger.error(result.error_message)
            
        return result
        
    def validate_etag(self, response: requests.Response, expected_etag: str) -> bool:
        """Validate ETag from HTTP response."""
        
        if not self.config.enable_etag_validation:
            return True
            
        actual_etag = response.headers.get('ETag', '').strip('"')
        
        if not actual_etag:
            self.logger.warning("No ETag found in response")
            return False
            
        is_valid = actual_etag == expected_etag.strip('"')
        
        if not is_valid:
            self.logger.error(f"ETag mismatch: expected {expected_etag}, got {actual_etag}")
            
        return is_valid
        
    def validate_md5_header(self, response: requests.Response, expected_md5: str) -> bool:
        """Validate MD5 from HTTP headers."""
        
        if not self.config.enable_md5_validation:
            return True
            
        actual_md5 = response.headers.get('Content-MD5', '')
        
        if not actual_md5:
            self.logger.warning("No Content-MD5 found in response")
            return False
            
        is_valid = actual_md5 == expected_md5
        
        if not is_valid:
            self.logger.error(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
            
        return is_valid


class ClientSideEncryption:
    """Handles client-side encryption for uploads."""
    
    def __init__(self, config: SecurityHardeningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption service
        encryption_config = config.encryption_config or EncryptionConfig()
        self.encryption_service = EncryptionService(encryption_config)
        
    def encrypt_data(self, data: bytes, encryption_key: Optional[str] = None) -> Tuple[bytes, str]:
        """Encrypt data using client-side encryption."""
        
        if not self.config.enable_client_side_encryption:
            return data, ""
            
        try:
            # Create or use provided encryption key
            if encryption_key:
                key_id = encryption_key
            else:
                key_id = self.encryption_service.create_key()
                
            # Encrypt the data
            encrypted_data = self.encryption_service.encrypt(data, key_id)
            
            # Return encrypted bytes and key ID
            import json
            encrypted_bytes = json.dumps(encrypted_data).encode()
            
            self.logger.info(f"Data encrypted with key: {key_id}")
            return encrypted_bytes, key_id
            
        except Exception as e:
            self.logger.error(f"Client-side encryption failed: {e}")
            raise
            
    def decrypt_data(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using client-side encryption."""
        
        try:
            # Parse encrypted data
            import json
            encrypted_dict = json.loads(encrypted_data.decode())
            
            # Decrypt the data
            decrypted_bytes = self.encryption_service.decrypt(encrypted_dict)
            
            self.logger.info(f"Data decrypted with key: {key_id}")
            return decrypted_bytes
            
        except Exception as e:
            self.logger.error(f"Client-side decryption failed: {e}")
            raise


class SDKVersionValidator:
    """Validates SDK and Python version requirements."""
    
    def __init__(self, config: SecurityHardeningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def validate_sdk_version(self, client_version: str) -> bool:
        """Validate client SDK version."""
        
        if not self.config.enforce_minimum_sdk_version:
            return True
            
        try:
            client_ver = version.parse(client_version)
            min_ver = version.parse(self.config.minimum_sdk_version)
            
            if client_ver < min_ver:
                self.logger.error(
                    f"SDK version {client_version} is below minimum required version {self.config.minimum_sdk_version}"
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"SDK version validation failed: {e}")
            return False
            
    def validate_python_version(self, python_version: str) -> bool:
        """Validate Python version."""
        
        try:
            python_ver = version.parse(python_version)
            min_ver = version.parse(self.config.minimum_python_version)
            
            if python_ver < min_ver:
                self.logger.error(
                    f"Python version {python_version} is below minimum required version {self.config.minimum_python_version}"
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Python version validation failed: {e}")
            return False


class SecurityHardeningService:
    """Main security hardening service."""
    
    def __init__(self, config: Optional[SecurityHardeningConfig] = None):
        self.config = config or SecurityHardeningConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tls_enforcer = TLSEnforcer(self.config)
        self.checksum_validator = ChecksumValidator(self.config)
        self.client_encryption = ClientSideEncryption(self.config)
        self.version_validator = SDKVersionValidator(self.config)
        
        # Track security events
        self.security_events: List[Dict[str, Any]] = []
        
    def validate_client_connection(
        self, 
        client_info: Dict[str, Any], 
        connection_url: str
    ) -> Tuple[bool, List[str]]:
        """Validate client connection with security checks."""
        
        errors = []
        
        # Validate TLS connection
        if not self.tls_enforcer.validate_tls_connection(connection_url):
            errors.append("TLS validation failed")
            
        # Validate SDK version
        sdk_version = client_info.get('sdk_version', '')
        if not self.version_validator.validate_sdk_version(sdk_version):
            errors.append(f"SDK version {sdk_version} is not supported")
            
        # Validate Python version
        python_version = client_info.get('python_version', '')
        if not self.version_validator.validate_python_version(python_version):
            errors.append(f"Python version {python_version} is not supported")
            
        is_valid = len(errors) == 0
        
        # Log security event
        self._log_security_event(
            event_type="client_connection_validation",
            client_info=client_info,
            connection_url=connection_url,
            is_valid=is_valid,
            errors=errors
        )
        
        return is_valid, errors
        
    def secure_upload(
        self,
        file_data: bytes,
        file_name: str,
        content_type: str,
        client_info: Dict[str, Any],
        encryption_key: Optional[str] = None
    ) -> UploadSecurityMetadata:
        """Secure file upload with validation and encryption."""
        
        # Check upload size
        file_size = len(file_data)
        max_size = self.config.max_upload_size_mb * 1024 * 1024
        
        if file_size > max_size:
            raise ValueError(f"File size {file_size} exceeds maximum allowed size {max_size}")
            
        # Encrypt data if enabled
        encrypted_data = file_data
        encryption_key_id = None
        client_side_encrypted = False
        
        if self.config.enable_client_side_encryption:
            if self.config.require_encrypted_uploads or encryption_key:
                encrypted_data, encryption_key_id = self.client_encryption.encrypt_data(
                    file_data, encryption_key
                )
                client_side_encrypted = True
                
        # Calculate checksum
        checksum = self.checksum_validator.calculate_checksum(
            encrypted_data, self.config.checksum_algorithm
        )
        
        # Create security metadata
        metadata = UploadSecurityMetadata(
            checksum=checksum,
            checksum_algorithm=self.config.checksum_algorithm,
            file_size=len(encrypted_data),
            content_type=content_type,
            encryption_key_id=encryption_key_id,
            client_side_encrypted=client_side_encrypted,
            client_ip=client_info.get('client_ip'),
            user_agent=client_info.get('user_agent')
        )
        
        # Log security event
        self._log_security_event(
            event_type="secure_upload",
            file_name=file_name,
            file_size=file_size,
            encrypted=client_side_encrypted,
            checksum=checksum,
            client_info=client_info
        )
        
        return metadata
        
    def validate_upload_integrity(
        self,
        file_data: bytes,
        expected_metadata: UploadSecurityMetadata,
        response_headers: Optional[Dict[str, str]] = None
    ) -> ChecksumValidationResult:
        """Validate upload integrity using checksums."""
        
        # Validate checksum
        result = self.checksum_validator.validate_checksum(
            file_data, 
            expected_metadata.checksum, 
            expected_metadata.checksum_algorithm
        )
        
        # Validate ETag if available
        if response_headers and self.config.enable_etag_validation:
            etag = response_headers.get('ETag')
            if etag:
                etag_valid = self.checksum_validator.validate_etag(
                    type('Response', (), {'headers': response_headers})(),
                    expected_metadata.checksum
                )
                if not etag_valid:
                    result.is_valid = False
                    result.error_message = f"ETag validation failed: {etag}"
                    
        # Validate MD5 if available
        if response_headers and self.config.enable_md5_validation:
            md5_header = response_headers.get('Content-MD5')
            if md5_header:
                md5_valid = self.checksum_validator.validate_md5_header(
                    type('Response', (), {'headers': response_headers})(),
                    expected_metadata.checksum
                )
                if not md5_valid:
                    result.is_valid = False
                    result.error_message = f"MD5 validation failed: {md5_header}"
                    
        # Log validation result
        self._log_security_event(
            event_type="upload_integrity_validation",
            is_valid=result.is_valid,
            expected_checksum=result.expected_checksum,
            actual_checksum=result.actual_checksum,
            algorithm=result.algorithm.value,
            error_message=result.error_message
        )
        
        return result
        
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        return self.config.security_headers.copy()
        
    def _log_security_event(self, event_type: str, **kwargs) -> None:
        """Log security event for audit trail."""
        
        event = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        self.security_events.append(event)
        
        if self.config.enable_security_audit:
            self.logger.info(f"Security event: {event}")
            
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security configuration summary."""
        
        return {
            'tls_enforced': self.config.enforce_tls,
            'minimum_tls_version': self.config.minimum_tls_version.value,
            'checksum_validation_enabled': self.config.enable_checksum_validation,
            'client_side_encryption_enabled': self.config.enable_client_side_encryption,
            'minimum_sdk_version': self.config.minimum_sdk_version,
            'minimum_python_version': self.config.minimum_python_version,
            'security_events_count': len(self.security_events),
            'max_upload_size_mb': self.config.max_upload_size_mb,
            'rate_limiting_enabled': self.config.enable_rate_limiting
        }


# Global security hardening service instance
_security_hardening_service: Optional[SecurityHardeningService] = None


def get_security_hardening_service() -> SecurityHardeningService:
    """Get global security hardening service instance."""
    global _security_hardening_service
    if _security_hardening_service is None:
        _security_hardening_service = SecurityHardeningService()
    return _security_hardening_service


def init_security_hardening_service(
    config: Optional[SecurityHardeningConfig] = None
) -> SecurityHardeningService:
    """Initialize global security hardening service."""
    global _security_hardening_service
    _security_hardening_service = SecurityHardeningService(config)
    return _security_hardening_service
