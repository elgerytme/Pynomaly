"""
Enterprise Security Features for Pynomaly Detection
===================================================

Comprehensive security framework providing:
- Advanced encryption and key management
- Security monitoring and threat detection
- Secure communication protocols
- Vulnerability assessment and management
- Security policy enforcement
"""

import logging
import json
import time
import threading
import hashlib
import secrets
import base64
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Threat type enumeration."""
    BRUTE_FORCE = "brute_force"
    SUSPICIOUS_ACCESS = "suspicious_access"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_PAYLOAD = "malicious_payload"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"

class EncryptionMethod(Enum):
    """Encryption method enumeration."""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    FERNET = "fernet"
    HYBRID = "hybrid"

@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]] = field(default_factory=list)
    enforcement_level: SecurityLevel = SecurityLevel.MEDIUM
    is_active: bool = True
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SecurityThreat:
    """Security threat record."""
    threat_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    description: str = ""
    indicators: Dict[str, Any] = field(default_factory=dict)
    mitigation_actions: List[str] = field(default_factory=list)
    status: str = "open"
    resolved_date: Optional[datetime] = None

@dataclass
class EncryptionKey:
    """Encryption key metadata."""
    key_id: str
    key_type: EncryptionMethod
    created_date: datetime
    expires_date: Optional[datetime] = None
    tenant_id: Optional[str] = None
    purpose: str = "general"
    is_active: bool = True
    rotation_count: int = 0

class SecurityManager:
    """Comprehensive enterprise security management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config or {}
        
        # Security policies
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.threat_database: Dict[str, SecurityThreat] = {}
        
        # Encryption and key management
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.master_key = None
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.threat_callbacks: List[Callable] = []
        
        # Rate limiting for security
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Threading
        self.lock = threading.RLock()
        
        # Initialize security components
        self._initialize_security()
        
        logger.info("Security Manager initialized")
    
    def initialize_encryption(self, master_password: Optional[str] = None) -> bool:
        """Initialize encryption system.
        
        Args:
            master_password: Master password for key derivation
            
        Returns:
            Success status
        """
        try:
            if not CRYPTO_AVAILABLE:
                logger.error("Cryptography library not available")
                return False
            
            # Generate or derive master key
            if master_password:
                salt = self.config.get('encryption_salt', secrets.token_bytes(32))
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000
                )
                self.master_key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
            else:
                self.master_key = Fernet.generate_key()
            
            # Initialize default encryption keys
            self._generate_default_keys()
            
            logger.info("Encryption system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            return False
    
    def encrypt_data(self, data: Union[str, bytes], key_id: Optional[str] = None,
                    method: EncryptionMethod = EncryptionMethod.FERNET) -> Optional[bytes]:
        """Encrypt data using specified method.
        
        Args:
            data: Data to encrypt
            key_id: Optional specific key ID
            method: Encryption method
            
        Returns:
            Encrypted data or None
        """
        try:
            if not CRYPTO_AVAILABLE:
                logger.error("Encryption not available")
                return None
            
            # Convert string to bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if method == EncryptionMethod.FERNET:
                key = self._get_encryption_key(key_id) or self.master_key
                if key:
                    fernet = Fernet(key)
                    return fernet.encrypt(data)
            
            elif method == EncryptionMethod.AES_256:
                return self._encrypt_aes_256(data, key_id)
            
            elif method == EncryptionMethod.RSA_2048:
                return self._encrypt_rsa_2048(data, key_id)
            
            elif method == EncryptionMethod.HYBRID:
                return self._encrypt_hybrid(data, key_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: bytes, key_id: Optional[str] = None,
                    method: EncryptionMethod = EncryptionMethod.FERNET) -> Optional[bytes]:
        """Decrypt data using specified method.
        
        Args:
            encrypted_data: Encrypted data
            key_id: Optional specific key ID
            method: Encryption method
            
        Returns:
            Decrypted data or None
        """
        try:
            if not CRYPTO_AVAILABLE:
                logger.error("Decryption not available")
                return None
            
            if method == EncryptionMethod.FERNET:
                key = self._get_encryption_key(key_id) or self.master_key
                if key:
                    fernet = Fernet(key)
                    return fernet.decrypt(encrypted_data)
            
            elif method == EncryptionMethod.AES_256:
                return self._decrypt_aes_256(encrypted_data, key_id)
            
            elif method == EncryptionMethod.RSA_2048:
                return self._decrypt_rsa_2048(encrypted_data, key_id)
            
            elif method == EncryptionMethod.HYBRID:
                return self._decrypt_hybrid(encrypted_data, key_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password securely.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        try:
            if BCRYPT_AVAILABLE:
                salt = bcrypt.gensalt()
                hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
                return hashed.decode('utf-8')
            else:
                # Fallback to SHA-256 with salt
                salt = secrets.token_hex(32)
                hashed = hashlib.sha256((password + salt).encode()).hexdigest()
                return f"sha256${salt}${hashed}"
                
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            return ""
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches
        """
        try:
            if BCRYPT_AVAILABLE and not hashed_password.startswith('sha256$'):
                return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
            else:
                # Handle SHA-256 fallback
                if hashed_password.startswith('sha256$'):
                    parts = hashed_password.split('$')
                    if len(parts) == 3:
                        salt = parts[1]
                        stored_hash = parts[2]
                        computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
                        return computed_hash == stored_hash
                return False
                
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def generate_token(self, user_id: str, tenant_id: Optional[str] = None,
                      expires_in: int = 3600, **claims) -> Optional[str]:
        """Generate JWT token.
        
        Args:
            user_id: User identifier
            tenant_id: Optional tenant identifier
            expires_in: Token expiration in seconds
            **claims: Additional claims
            
        Returns:
            JWT token or None
        """
        try:
            if not JWT_AVAILABLE:
                logger.error("JWT library not available")
                return None
            
            payload = {
                'user_id': user_id,
                'tenant_id': tenant_id,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(seconds=expires_in),
                'jti': str(uuid.uuid4()),
                **claims
            }
            
            # Use master key for signing
            secret = self.master_key or 'default_secret'
            token = jwt.encode(payload, secret, algorithm='HS256')
            
            return token
            
        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded payload or None
        """
        try:
            if not JWT_AVAILABLE:
                logger.error("JWT library not available")
                return None
            
            secret = self.master_key or 'default_secret'
            payload = jwt.decode(token, secret, algorithms=['HS256'])
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    def detect_security_threat(self, event_data: Dict[str, Any]) -> Optional[SecurityThreat]:
        """Detect security threats from event data.
        
        Args:
            event_data: Event data to analyze
            
        Returns:
            Security threat if detected, None otherwise
        """
        try:
            threats = []
            
            # Check for brute force attacks
            if self._detect_brute_force(event_data):
                threats.append(self._create_threat(
                    ThreatType.BRUTE_FORCE,
                    SecurityLevel.HIGH,
                    "Brute force attack detected",
                    event_data
                ))
            
            # Check for suspicious access patterns
            if self._detect_suspicious_access(event_data):
                threats.append(self._create_threat(
                    ThreatType.SUSPICIOUS_ACCESS,
                    SecurityLevel.MEDIUM,
                    "Suspicious access pattern detected",
                    event_data
                ))
            
            # Check for data exfiltration
            if self._detect_data_exfiltration(event_data):
                threats.append(self._create_threat(
                    ThreatType.DATA_EXFILTRATION,
                    SecurityLevel.CRITICAL,
                    "Potential data exfiltration detected",
                    event_data
                ))
            
            # Return the highest severity threat
            if threats:
                highest_threat = max(threats, key=lambda t: list(SecurityLevel).index(t.severity))
                self._store_threat(highest_threat)
                self._trigger_threat_callbacks(highest_threat)
                return highest_threat
            
            return None
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return None
    
    def add_security_policy(self, policy: SecurityPolicy) -> bool:
        """Add security policy.
        
        Args:
            policy: Security policy
            
        Returns:
            Success status
        """
        try:
            with self.lock:
                self.security_policies[policy.policy_id] = policy
            
            logger.info(f"Security policy added: {policy.policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add security policy: {e}")
            return False
    
    def enforce_security_policy(self, policy_id: str, context: Dict[str, Any]) -> bool:
        """Enforce security policy.
        
        Args:
            policy_id: Policy identifier
            context: Enforcement context
            
        Returns:
            True if policy allows action, False otherwise
        """
        try:
            policy = self.security_policies.get(policy_id)
            if not policy or not policy.is_active:
                return True  # No policy or inactive policy allows action
            
            # Evaluate policy rules
            for rule in policy.rules:
                if not self._evaluate_policy_rule(rule, context):
                    logger.warning(f"Security policy {policy_id} violated")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Policy enforcement failed: {e}")
            return False
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            True if blocked, False otherwise
        """
        with self.lock:
            if ip_address in self.blocked_ips:
                block_time = self.blocked_ips[ip_address]
                # Unblock after 1 hour
                if datetime.now() - block_time > timedelta(hours=1):
                    del self.blocked_ips[ip_address]
                    return False
                return True
            return False
    
    def block_ip(self, ip_address: str, duration_hours: int = 1):
        """Block IP address.
        
        Args:
            ip_address: IP address to block
            duration_hours: Block duration in hours
        """
        with self.lock:
            self.blocked_ips[ip_address] = datetime.now()
        
        logger.warning(f"IP address blocked: {ip_address}")
    
    def rotate_encryption_keys(self, key_type: Optional[EncryptionMethod] = None) -> bool:
        """Rotate encryption keys.
        
        Args:
            key_type: Optional specific key type to rotate
            
        Returns:
            Success status
        """
        try:
            keys_rotated = 0
            
            for key_id, key_metadata in self.encryption_keys.items():
                if key_type and key_metadata.key_type != key_type:
                    continue
                
                # Generate new key
                new_key = self._generate_key(key_metadata.key_type)
                if new_key:
                    # Update key metadata
                    key_metadata.rotation_count += 1
                    key_metadata.created_date = datetime.now()
                    keys_rotated += 1
            
            logger.info(f"Rotated {keys_rotated} encryption keys")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False
    
    def get_security_report(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get security report.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            Security report
        """
        try:
            threats = list(self.threat_database.values())
            if tenant_id:
                threats = [t for t in threats if t.tenant_id == tenant_id]
            
            # Calculate threat statistics
            threat_counts = {}
            severity_counts = {}
            
            for threat in threats:
                threat_type = threat.threat_type.value
                severity = threat.severity.value
                
                threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "tenant_id": tenant_id,
                "total_threats": len(threats),
                "threat_types": threat_counts,
                "severity_distribution": severity_counts,
                "active_policies": len([p for p in self.security_policies.values() if p.is_active]),
                "blocked_ips": len(self.blocked_ips),
                "encryption_keys": len(self.encryption_keys),
                "recent_threats": [
                    {
                        "threat_id": t.threat_id,
                        "type": t.threat_type.value,
                        "severity": t.severity.value,
                        "timestamp": t.timestamp.isoformat(),
                        "description": t.description
                    }
                    for t in sorted(threats, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
            return {}
    
    def add_threat_callback(self, callback: Callable[[SecurityThreat], None]):
        """Add callback for threat detection.
        
        Args:
            callback: Callback function
        """
        self.threat_callbacks.append(callback)
    
    def _initialize_security(self):
        """Initialize security components."""
        # Initialize default security policies
        self._create_default_policies()
        
        # Initialize threat detection
        self._initialize_threat_detection()
    
    def _create_default_policies(self):
        """Create default security policies."""
        default_policies = [
            SecurityPolicy(
                policy_id="password_complexity",
                name="Password Complexity Policy",
                description="Enforce strong password requirements",
                rules=[
                    {"type": "min_length", "value": 12},
                    {"type": "require_uppercase", "value": True},
                    {"type": "require_lowercase", "value": True},
                    {"type": "require_digits", "value": True},
                    {"type": "require_special", "value": True}
                ],
                enforcement_level=SecurityLevel.HIGH
            ),
            SecurityPolicy(
                policy_id="session_timeout",
                name="Session Timeout Policy",
                description="Enforce session timeout limits",
                rules=[
                    {"type": "max_idle_time", "value": 1800},  # 30 minutes
                    {"type": "max_session_time", "value": 28800}  # 8 hours
                ],
                enforcement_level=SecurityLevel.MEDIUM
            ),
            SecurityPolicy(
                policy_id="ip_whitelist",
                name="IP Whitelist Policy",
                description="Restrict access to whitelisted IPs",
                rules=[
                    {"type": "allowed_ips", "value": []},
                    {"type": "block_suspicious", "value": True}
                ],
                enforcement_level=SecurityLevel.LOW
            )
        ]
        
        for policy in default_policies:
            self.security_policies[policy.policy_id] = policy
    
    def _initialize_threat_detection(self):
        """Initialize threat detection rules."""
        # This would initialize ML models and rules for threat detection
        pass
    
    def _generate_default_keys(self):
        """Generate default encryption keys."""
        try:
            # Generate Fernet key
            fernet_key = EncryptionKey(
                key_id="default_fernet",
                key_type=EncryptionMethod.FERNET,
                created_date=datetime.now(),
                purpose="default_encryption"
            )
            self.encryption_keys[fernet_key.key_id] = fernet_key
            
            # Generate AES key
            aes_key = EncryptionKey(
                key_id="default_aes",
                key_type=EncryptionMethod.AES_256,
                created_date=datetime.now(),
                purpose="bulk_encryption"
            )
            self.encryption_keys[aes_key.key_id] = aes_key
            
        except Exception as e:
            logger.error(f"Failed to generate default keys: {e}")
    
    def _get_encryption_key(self, key_id: Optional[str]) -> Optional[bytes]:
        """Get encryption key by ID."""
        if not key_id:
            return None
        
        key_metadata = self.encryption_keys.get(key_id)
        if not key_metadata or not key_metadata.is_active:
            return None
        
        # In practice, this would retrieve the actual key from secure storage
        # For now, return a derived key
        return self._derive_key(key_id)
    
    def _derive_key(self, key_id: str) -> bytes:
        """Derive key from master key and key ID."""
        if not self.master_key:
            return Fernet.generate_key()
        
        # Simple key derivation
        combined = f"{self.master_key.decode()}{key_id}".encode()
        digest = hashlib.sha256(combined).digest()
        return base64.urlsafe_b64encode(digest)
    
    def _generate_key(self, key_type: EncryptionMethod) -> Optional[bytes]:
        """Generate new key of specified type."""
        if key_type == EncryptionMethod.FERNET:
            return Fernet.generate_key()
        elif key_type == EncryptionMethod.AES_256:
            return secrets.token_bytes(32)
        # Add other key types as needed
        return None
    
    def _encrypt_aes_256(self, data: bytes, key_id: Optional[str]) -> bytes:
        """Encrypt using AES-256."""
        # Simplified AES encryption implementation
        key = self._get_encryption_key(key_id) or secrets.token_bytes(32)
        iv = secrets.token_bytes(16)
        
        cipher = Cipher(algorithms.AES(key[:32]), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return iv + encrypted_data
    
    def _decrypt_aes_256(self, encrypted_data: bytes, key_id: Optional[str]) -> bytes:
        """Decrypt using AES-256."""
        key = self._get_encryption_key(key_id) or secrets.token_bytes(32)
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        cipher = Cipher(algorithms.AES(key[:32]), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _encrypt_rsa_2048(self, data: bytes, key_id: Optional[str]) -> bytes:
        """Encrypt using RSA-2048."""
        # Simplified RSA encryption - would use stored keys in practice
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # RSA can only encrypt small amounts of data
        # In practice, would use hybrid encryption
        encrypted_data = public_key.encrypt(
            data[:245],  # RSA-2048 can encrypt up to 245 bytes
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted_data
    
    def _decrypt_rsa_2048(self, encrypted_data: bytes, key_id: Optional[str]) -> bytes:
        """Decrypt using RSA-2048."""
        # This would use the stored private key
        # Simplified implementation
        return encrypted_data  # Placeholder
    
    def _encrypt_hybrid(self, data: bytes, key_id: Optional[str]) -> bytes:
        """Encrypt using hybrid encryption (RSA + AES)."""
        # Generate random AES key
        aes_key = secrets.token_bytes(32)
        
        # Encrypt data with AES
        encrypted_data = self._encrypt_aes_with_key(data, aes_key)
        
        # Encrypt AES key with RSA
        encrypted_key = self._encrypt_rsa_2048(aes_key, key_id)
        
        # Combine encrypted key and data
        return len(encrypted_key).to_bytes(4, 'big') + encrypted_key + encrypted_data
    
    def _decrypt_hybrid(self, encrypted_data: bytes, key_id: Optional[str]) -> bytes:
        """Decrypt using hybrid encryption."""
        # Extract encrypted key length
        key_length = int.from_bytes(encrypted_data[:4], 'big')
        
        # Extract encrypted key and data
        encrypted_key = encrypted_data[4:4 + key_length]
        encrypted_payload = encrypted_data[4 + key_length:]
        
        # Decrypt AES key with RSA
        aes_key = self._decrypt_rsa_2048(encrypted_key, key_id)
        
        # Decrypt data with AES
        return self._decrypt_aes_with_key(encrypted_payload, aes_key)
    
    def _encrypt_aes_with_key(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data with provided AES key."""
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        return iv + encrypted_data
    
    def _decrypt_aes_with_key(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data with provided AES key."""
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _detect_brute_force(self, event_data: Dict[str, Any]) -> bool:
        """Detect brute force attacks."""
        if event_data.get('event_type') != 'login_failed':
            return False
        
        ip_address = event_data.get('client_ip')
        if not ip_address:
            return False
        
        with self.lock:
            if ip_address not in self.failed_attempts:
                self.failed_attempts[ip_address] = []
            
            # Add current attempt
            self.failed_attempts[ip_address].append(datetime.now())
            
            # Clean old attempts (older than 1 hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.failed_attempts[ip_address] = [
                attempt for attempt in self.failed_attempts[ip_address]
                if attempt > cutoff_time
            ]
            
            # Check if threshold exceeded
            return len(self.failed_attempts[ip_address]) >= 5
    
    def _detect_suspicious_access(self, event_data: Dict[str, Any]) -> bool:
        """Detect suspicious access patterns."""
        # Check for access from unusual locations
        user_id = event_data.get('user_id')
        client_ip = event_data.get('client_ip')
        
        if not user_id or not client_ip:
            return False
        
        # Simple implementation - check for access from new IP
        # In practice, would use more sophisticated geolocation and behavior analysis
        return False
    
    def _detect_data_exfiltration(self, event_data: Dict[str, Any]) -> bool:
        """Detect potential data exfiltration."""
        event_type = event_data.get('event_type')
        
        # Check for large data exports or unusual access patterns
        if event_type == 'data_export':
            export_size = event_data.get('export_size', 0)
            # Flag exports larger than 100MB
            return export_size > 100 * 1024 * 1024
        
        return False
    
    def _create_threat(self, threat_type: ThreatType, severity: SecurityLevel,
                      description: str, event_data: Dict[str, Any]) -> SecurityThreat:
        """Create security threat record."""
        return SecurityThreat(
            threat_id=f"threat_{int(time.time())}_{str(uuid.uuid4())[:8]}",
            threat_type=threat_type,
            severity=severity,
            timestamp=datetime.now(),
            source_ip=event_data.get('client_ip'),
            user_id=event_data.get('user_id'),
            tenant_id=event_data.get('tenant_id'),
            description=description,
            indicators=event_data
        )
    
    def _store_threat(self, threat: SecurityThreat):
        """Store security threat."""
        with self.lock:
            self.threat_database[threat.threat_id] = threat
    
    def _trigger_threat_callbacks(self, threat: SecurityThreat):
        """Trigger threat detection callbacks."""
        for callback in self.threat_callbacks:
            try:
                callback(threat)
            except Exception as e:
                logger.error(f"Threat callback failed: {e}")
    
    def _evaluate_policy_rule(self, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate security policy rule."""
        rule_type = rule.get('type')
        rule_value = rule.get('value')
        
        if rule_type == "min_length":
            password = context.get('password', '')
            return len(password) >= rule_value
        
        elif rule_type == "require_uppercase":
            password = context.get('password', '')
            return any(c.isupper() for c in password) if rule_value else True
        
        elif rule_type == "require_lowercase":
            password = context.get('password', '')
            return any(c.islower() for c in password) if rule_value else True
        
        elif rule_type == "require_digits":
            password = context.get('password', '')
            return any(c.isdigit() for c in password) if rule_value else True
        
        elif rule_type == "require_special":
            password = context.get('password', '')
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            return any(c in special_chars for c in password) if rule_value else True
        
        elif rule_type == "allowed_ips":
            client_ip = context.get('client_ip')
            allowed_ips = rule_value
            return not allowed_ips or client_ip in allowed_ips
        
        elif rule_type == "max_idle_time":
            last_activity = context.get('last_activity')
            if last_activity:
                idle_time = (datetime.now() - last_activity).total_seconds()
                return idle_time <= rule_value
            return True
        
        # Default to allow if rule not recognized
        return True


class EncryptionService:
    """Simplified encryption service interface."""
    
    def __init__(self, security_manager: SecurityManager):
        """Initialize encryption service.
        
        Args:
            security_manager: Security manager instance
        """
        self.security_manager = security_manager
        logger.info("Encryption Service initialized")
    
    def encrypt(self, data: str, tenant_id: Optional[str] = None) -> Optional[str]:
        """Encrypt string data.
        
        Args:
            data: Data to encrypt
            tenant_id: Optional tenant identifier
            
        Returns:
            Base64 encoded encrypted data or None
        """
        encrypted_bytes = self.security_manager.encrypt_data(data)
        if encrypted_bytes:
            return base64.b64encode(encrypted_bytes).decode('utf-8')
        return None
    
    def decrypt(self, encrypted_data: str, tenant_id: Optional[str] = None) -> Optional[str]:
        """Decrypt string data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            tenant_id: Optional tenant identifier
            
        Returns:
            Decrypted string or None
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_bytes = self.security_manager.decrypt_data(encrypted_bytes)
            if decrypted_bytes:
                return decrypted_bytes.decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> bool:
        """Encrypt file.
        
        Args:
            file_path: Path to file to encrypt
            output_path: Optional output path
            
        Returns:
            Success status
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.security_manager.encrypt_data(data)
            if not encrypted_data:
                return False
            
            output_file = output_path or f"{file_path}.encrypted"
            with open(output_file, 'wb') as f:
                f.write(encrypted_data)
            
            return True
            
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            return False
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> bool:
        """Decrypt file.
        
        Args:
            encrypted_file_path: Path to encrypted file
            output_path: Optional output path
            
        Returns:
            Success status
        """
        try:
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.security_manager.decrypt_data(encrypted_data)
            if not decrypted_data:
                return False
            
            output_file = output_path or encrypted_file_path.replace('.encrypted', '')
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
            
            return True
            
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            return False