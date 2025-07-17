"""
Tests for AuthenticationService
"""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from ..domain.entities.user import User
from ..domain.services.authentication_service import AuthenticationService

class TestAuthenticationService:
    """Test suite for AuthenticationService"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.auth_service = AuthenticationService()
        
        # Create test user
        self.user = User(
            id=uuid4(),
            email="test@example.com",
            username="testuser",
            password_hash="",  # Will be set by tests
            created_at=datetime.now(),
            is_active=True,
            is_verified=True
        )
    
    def test_hash_password_creates_valid_hash(self):
        """Test password hashing creates valid hash"""
        password = "testpassword123"
        password_hash = self.auth_service.hash_password(password)
        
        # Should contain salt and hash separated by colon
        assert ":" in password_hash
        salt, hash_part = password_hash.split(":")
        assert len(salt) == 64  # 32 bytes hex = 64 chars
        assert len(hash_part) == 64  # SHA256 hex = 64 chars
    
    def test_verify_password_with_correct_password(self):
        """Test password verification with correct password"""
        password = "testpassword123"
        password_hash = self.auth_service.hash_password(password)
        
        result = self.auth_service.verify_password(password, password_hash)
        assert result is True
    
    def test_verify_password_with_incorrect_password(self):
        """Test password verification with incorrect password"""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        password_hash = self.auth_service.hash_password(password)
        
        result = self.auth_service.verify_password(wrong_password, password_hash)
        assert result is False
    
    def test_verify_password_with_malformed_hash(self):
        """Test password verification with malformed hash"""
        password = "testpassword123"
        malformed_hash = "malformed_hash_without_colon"
        
        result = self.auth_service.verify_password(password, malformed_hash)
        assert result is False
    
    def test_validate_password_strength_valid_password(self):
        """Test password strength validation with valid password"""
        password = "SecurePass123!"
        
        is_valid, errors = self.auth_service.validate_password_strength(password)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_password_strength_too_short(self):
        """Test password strength validation with too short password"""
        password = "Short1!"
        
        is_valid, errors = self.auth_service.validate_password_strength(password)
        
        assert is_valid is False
        assert "Password must be at least 8 characters" in errors
    
    def test_validate_password_strength_missing_uppercase(self):
        """Test password strength validation missing uppercase"""
        password = "lowercase123!"
        
        is_valid, errors = self.auth_service.validate_password_strength(password)
        
        assert is_valid is False
        assert "Password must contain at least one uppercase letter" in errors
    
    def test_validate_password_strength_missing_lowercase(self):
        """Test password strength validation missing lowercase"""
        password = "UPPERCASE123!"
        
        is_valid, errors = self.auth_service.validate_password_strength(password)
        
        assert is_valid is False
        assert "Password must contain at least one lowercase letter" in errors
    
    def test_validate_password_strength_missing_digit(self):
        """Test password strength validation missing digit"""
        password = "SecurePass!"
        
        is_valid, errors = self.auth_service.validate_password_strength(password)
        
        assert is_valid is False
        assert "Password must contain at least one digit" in errors
    
    def test_validate_password_strength_missing_special_char(self):
        """Test password strength validation missing special character"""
        password = "SecurePass123"
        
        is_valid, errors = self.auth_service.validate_password_strength(password)
        
        assert is_valid is False
        assert "Password must contain at least one special character" in errors
    
    def test_authenticate_user_with_valid_credentials(self):
        """Test user authentication with valid credentials"""
        password = "testpassword123"
        self.user.password_hash = self.auth_service.hash_password(password)
        
        result = self.auth_service.authenticate_user(self.user, password)
        
        assert result is True
        assert self.user.failed_login_attempts == 0
        assert self.user.locked_until is None
    
    def test_authenticate_user_with_invalid_credentials(self):
        """Test user authentication with invalid credentials"""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        self.user.password_hash = self.auth_service.hash_password(password)
        
        result = self.auth_service.authenticate_user(self.user, wrong_password)
        
        assert result is False
        assert self.user.failed_login_attempts == 1
    
    def test_authenticate_user_locks_account_after_max_attempts(self):
        """Test user authentication locks account after max attempts"""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        self.user.password_hash = self.auth_service.hash_password(password)
        
        # Attempt login 5 times with wrong password
        for i in range(5):
            result = self.auth_service.authenticate_user(self.user, wrong_password)
            assert result is False
        
        # Account should be locked now
        assert self.user.failed_login_attempts == 5
        assert self.user.locked_until is not None
        assert self.user.locked_until > datetime.now()
    
    def test_authenticate_user_with_inactive_account(self):
        """Test user authentication with inactive account"""
        password = "testpassword123"
        self.user.password_hash = self.auth_service.hash_password(password)
        self.user.is_active = False
        
        result = self.auth_service.authenticate_user(self.user, password)
        
        assert result is False
    
    def test_authenticate_user_with_unverified_account(self):
        """Test user authentication with unverified account"""
        password = "testpassword123"
        self.user.password_hash = self.auth_service.hash_password(password)
        self.user.is_verified = False
        
        result = self.auth_service.authenticate_user(self.user, password)
        
        assert result is False
    
    def test_authenticate_user_with_locked_account(self):
        """Test user authentication with locked account"""
        password = "testpassword123"
        self.user.password_hash = self.auth_service.hash_password(password)
        self.user.locked_until = datetime.now() + timedelta(minutes=30)
        
        result = self.auth_service.authenticate_user(self.user, password)
        
        assert result is False
    
    def test_generate_reset_token_creates_valid_token(self):
        """Test password reset token generation"""
        token = self.auth_service.generate_reset_token()
        
        assert isinstance(token, str)
        assert len(token) > 0
        # URL-safe base64 tokens should not contain certain characters
        assert "/" not in token
        assert "+" not in token
        assert "=" not in token
    
    def test_is_reset_token_valid_with_fresh_token(self):
        """Test reset token validation with fresh token"""
        token = self.auth_service.generate_reset_token()
        created_at = datetime.now()
        
        is_valid = self.auth_service.is_reset_token_valid(token, created_at)
        
        assert is_valid is True
    
    def test_is_reset_token_valid_with_expired_token(self):
        """Test reset token validation with expired token"""
        token = self.auth_service.generate_reset_token()
        created_at = datetime.now() - timedelta(hours=2)  # 2 hours ago
        
        is_valid = self.auth_service.is_reset_token_valid(token, created_at)
        
        assert is_valid is False