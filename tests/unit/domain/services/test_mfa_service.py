#!/usr/bin/env python3
"""
Unit tests for MFA service functionality.
Tests TOTP, SMS, email, and backup code authentication methods.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from pynomaly.domain.services.mfa_service import MFAService
from pynomaly.application.dto.mfa_dto import (
    MFAMethodType,
    MFAMethodStatus,
    TOTPSetupResponse,
    BackupCodesResponse,
    MFAStatisticsDTO,
)


class TestMFAService:
    """Test MFA service functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_client = Mock()
        return redis_client
    
    @pytest.fixture
    def mock_email_service(self):
        """Mock email service."""
        email_service = Mock()
        email_service.send_email.return_value = True
        return email_service
    
    @pytest.fixture
    def mock_sms_service(self):
        """Mock SMS service."""
        sms_service = Mock()
        sms_service.send_sms.return_value = True
        return sms_service
    
    @pytest.fixture
    def mfa_service(self, mock_redis, mock_email_service, mock_sms_service):
        """Create MFA service instance."""
        return MFAService(
            redis_client=mock_redis,
            email_service=mock_email_service,
            sms_service=mock_sms_service
        )
    
    def test_init(self, mfa_service):
        """Test MFA service initialization."""
        assert mfa_service.totp_window == 1
        assert mfa_service.sms_code_expiry == 300
        assert mfa_service.email_code_expiry == 600
        assert mfa_service.backup_codes_count == 10
        assert mfa_service.device_remember_duration == 2592000
    
    @patch('pynomaly.domain.services.mfa_service.PYOTP_AVAILABLE', True)
    @patch('pynomaly.domain.services.mfa_service.pyotp')
    def test_generate_totp_secret(self, mock_pyotp, mfa_service, mock_redis):
        """Test TOTP secret generation."""
        mock_pyotp.random_base32.return_value = "TESTSECRET123"
        
        secret = mfa_service.generate_totp_secret("user123")
        
        assert secret == "TESTSECRET123"
        mock_pyotp.random_base32.assert_called_once()
        mock_redis.setex.assert_called_once_with(
            "mfa_totp_setup:user123",
            3600,
            "TESTSECRET123"
        )
    
    @patch('pynomaly.domain.services.mfa_service.PYOTP_AVAILABLE', False)
    def test_generate_totp_secret_without_pyotp(self, mfa_service):
        """Test TOTP secret generation when pyotp is not available."""
        with pytest.raises(RuntimeError, match="TOTP functionality requires pyotp library"):
            mfa_service.generate_totp_secret("user123")
    
    @patch('pynomaly.domain.services.mfa_service.PYOTP_AVAILABLE', True)
    @patch('pynomaly.domain.services.mfa_service.QRCODE_AVAILABLE', True)
    @patch('pynomaly.domain.services.mfa_service.pyotp')
    @patch('pynomaly.domain.services.mfa_service.qrcode')
    def test_create_totp_setup_response(self, mock_qrcode, mock_pyotp, mfa_service):
        """Test TOTP setup response creation."""
        # Mock TOTP
        mock_totp = Mock()
        mock_totp.provisioning_uri.return_value = "otpauth://totp/test"
        mock_pyotp.totp.TOTP.return_value = mock_totp
        
        # Mock QR code
        mock_qr = Mock()
        mock_img = Mock()
        mock_qr.make_image.return_value = mock_img
        mock_qrcode.QRCode.return_value = mock_qr
        
        # Mock backup codes
        with patch.object(mfa_service, 'generate_backup_codes') as mock_backup:
            mock_backup.return_value = ["123456", "789012"]
            
            response = mfa_service.create_totp_setup_response(
                user_id="user123",
                user_email="test@example.com",
                secret="TESTSECRET123"
            )
            
            assert isinstance(response, TOTPSetupResponse)
            assert response.secret == "TESTSECRET123"
            assert response.manual_entry_key == "TESTSECRET123"
            assert response.backup_codes == ["123456", "789012"]
    
    @patch('pynomaly.domain.services.mfa_service.PYOTP_AVAILABLE', True)
    @patch('pynomaly.domain.services.mfa_service.pyotp')
    def test_verify_totp_code_success(self, mock_pyotp, mfa_service, mock_redis):
        """Test successful TOTP code verification."""
        mock_redis.get.return_value = b"TESTSECRET123"
        
        mock_totp = Mock()
        mock_totp.verify.return_value = True
        mock_pyotp.TOTP.return_value = mock_totp
        
        result = mfa_service.verify_totp_code("user123", "123456")
        
        assert result is True
        mock_totp.verify.assert_called_once_with("123456", valid_window=1)
    
    @patch('pynomaly.domain.services.mfa_service.PYOTP_AVAILABLE', True)
    @patch('pynomaly.domain.services.mfa_service.pyotp')
    def test_verify_totp_code_failure(self, mock_pyotp, mfa_service, mock_redis):
        """Test failed TOTP code verification."""
        mock_redis.get.return_value = b"TESTSECRET123"
        
        mock_totp = Mock()
        mock_totp.verify.return_value = False
        mock_pyotp.TOTP.return_value = mock_totp
        
        result = mfa_service.verify_totp_code("user123", "123456")
        
        assert result is False
        mock_totp.verify.assert_called_once_with("123456", valid_window=1)
    
    def test_confirm_totp_setup_success(self, mfa_service, mock_redis):
        """Test successful TOTP setup confirmation."""
        mock_redis.get.return_value = b"TESTSECRET123"
        
        with patch.object(mfa_service, 'verify_totp_code') as mock_verify:
            mock_verify.return_value = True
            
            result = mfa_service.confirm_totp_setup("user123", "123456")
            
            assert result is True
            mock_redis.setex.assert_called_once_with(
                "mfa_totp_confirmed:user123",
                86400 * 365,
                "TESTSECRET123"
            )
            mock_redis.delete.assert_called_once_with("mfa_totp_setup:user123")
    
    def test_confirm_totp_setup_failure(self, mfa_service, mock_redis):
        """Test failed TOTP setup confirmation."""
        mock_redis.get.return_value = b"TESTSECRET123"
        
        with patch.object(mfa_service, 'verify_totp_code') as mock_verify:
            mock_verify.return_value = False
            
            result = mfa_service.confirm_totp_setup("user123", "123456")
            
            assert result is False
            mock_redis.setex.assert_not_called()
            mock_redis.delete.assert_not_called()
    
    def test_send_sms_code_success(self, mfa_service, mock_redis, mock_sms_service):
        """Test successful SMS code sending."""
        result = mfa_service.send_sms_code("user123", "+1234567890")
        
        assert result is True
        mock_sms_service.send_sms.assert_called_once()
        mock_redis.setex.assert_called_once()
        
        # Check that a 6-digit code was stored
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "mfa_sms_code:user123"
        assert call_args[0][1] == 300  # 5 minutes
        assert len(call_args[0][2]) == 6
        assert call_args[0][2].isdigit()
    
    def test_send_sms_code_failure(self, mfa_service, mock_sms_service):
        """Test failed SMS code sending."""
        mock_sms_service.send_sms.return_value = False
        
        result = mfa_service.send_sms_code("user123", "+1234567890")
        
        assert result is False
        mock_sms_service.send_sms.assert_called_once()
    
    def test_verify_sms_code_success(self, mfa_service, mock_redis):
        """Test successful SMS code verification."""
        mock_redis.get.return_value = b"123456"
        
        result = mfa_service.verify_sms_code("user123", "123456")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("mfa_sms_code:user123")
    
    def test_verify_sms_code_failure(self, mfa_service, mock_redis):
        """Test failed SMS code verification."""
        mock_redis.get.return_value = b"123456"
        
        result = mfa_service.verify_sms_code("user123", "654321")
        
        assert result is False
        mock_redis.delete.assert_not_called()
    
    def test_send_email_code_success(self, mfa_service, mock_redis, mock_email_service):
        """Test successful email code sending."""
        result = mfa_service.send_email_code("user123", "test@example.com")
        
        assert result is True
        mock_email_service.send_email.assert_called_once()
        mock_redis.setex.assert_called_once()
        
        # Check that a 6-digit code was stored
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "mfa_email_code:user123"
        assert call_args[0][1] == 600  # 10 minutes
        assert len(call_args[0][2]) == 6
        assert call_args[0][2].isdigit()
    
    def test_verify_email_code_success(self, mfa_service, mock_redis):
        """Test successful email code verification."""
        mock_redis.get.return_value = b"123456"
        
        result = mfa_service.verify_email_code("user123", "123456")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("mfa_email_code:user123")
    
    def test_verify_email_code_failure(self, mfa_service, mock_redis):
        """Test failed email code verification."""
        mock_redis.get.return_value = b"123456"
        
        result = mfa_service.verify_email_code("user123", "654321")
        
        assert result is False
        mock_redis.delete.assert_not_called()
    
    def test_generate_backup_codes(self, mfa_service, mock_redis):
        """Test backup codes generation."""
        backup_codes = mfa_service.generate_backup_codes("user123")
        
        assert len(backup_codes) == 10
        for code in backup_codes:
            assert len(code) == 8
            assert code.isdigit()
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "mfa_backup_codes:user123"
        assert call_args[0][1] == 86400 * 365  # 1 year
    
    def test_verify_backup_code_success(self, mfa_service, mock_redis):
        """Test successful backup code verification."""
        # Mock stored hashed codes
        import hashlib
        test_code = "12345678"
        hashed_code = hashlib.sha256(test_code.encode()).hexdigest()
        mock_redis.get.return_value = f"{hashed_code}|anotherhash".encode()
        
        result = mfa_service.verify_backup_code("user123", test_code)
        
        assert result is True
        mock_redis.setex.assert_called_once()  # Update with remaining codes
    
    def test_verify_backup_code_failure(self, mfa_service, mock_redis):
        """Test failed backup code verification."""
        # Mock stored hashed codes
        import hashlib
        test_code = "12345678"
        wrong_code = "87654321"
        hashed_code = hashlib.sha256(test_code.encode()).hexdigest()
        mock_redis.get.return_value = f"{hashed_code}|anotherhash".encode()
        
        result = mfa_service.verify_backup_code("user123", wrong_code)
        
        assert result is False
        mock_redis.setex.assert_not_called()
    
    def test_get_backup_codes_count(self, mfa_service, mock_redis):
        """Test getting backup codes count."""
        mock_redis.get.return_value = b"hash1|hash2|hash3"
        
        count = mfa_service.get_backup_codes_count("user123")
        
        assert count == 3
    
    def test_get_backup_codes_count_no_codes(self, mfa_service, mock_redis):
        """Test getting backup codes count when no codes exist."""
        mock_redis.get.return_value = None
        
        count = mfa_service.get_backup_codes_count("user123")
        
        assert count == 0
    
    def test_remember_device(self, mfa_service, mock_redis):
        """Test device remembering."""
        device_info = {
            "device_name": "iPhone",
            "device_type": "mobile",
            "user_agent": "Safari/1.0",
            "ip_address": "192.168.1.1"
        }
        
        device_id = mfa_service.remember_device("user123", device_info)
        
        assert device_id is not None
        assert len(device_id) == 43  # URL-safe base64 with 32 bytes
        mock_redis.setex.assert_called_once()
    
    def test_is_device_trusted_success(self, mfa_service, mock_redis):
        """Test trusted device check success."""
        device_data = "device_id:test123|user_id:user123|device_name:iPhone"
        mock_redis.get.return_value = device_data.encode()
        
        result = mfa_service.is_device_trusted("test123", "user123")
        
        assert result is True
        mock_redis.setex.assert_called_once()  # Update last used time
    
    def test_is_device_trusted_wrong_user(self, mfa_service, mock_redis):
        """Test trusted device check with wrong user."""
        device_data = "device_id:test123|user_id:other_user|device_name:iPhone"
        mock_redis.get.return_value = device_data.encode()
        
        result = mfa_service.is_device_trusted("test123", "user123")
        
        assert result is False
    
    def test_is_device_trusted_not_found(self, mfa_service, mock_redis):
        """Test trusted device check when device not found."""
        mock_redis.get.return_value = None
        
        result = mfa_service.is_device_trusted("test123", "user123")
        
        assert result is False
    
    def test_revoke_trusted_device(self, mfa_service, mock_redis):
        """Test revoking trusted device."""
        with patch.object(mfa_service, 'is_device_trusted') as mock_trusted:
            mock_trusted.return_value = True
            
            result = mfa_service.revoke_trusted_device("test123", "user123")
            
            assert result is True
            mock_redis.delete.assert_called_once_with("mfa_trusted_device:test123")
    
    def test_revoke_trusted_device_not_found(self, mfa_service, mock_redis):
        """Test revoking trusted device when not found."""
        with patch.object(mfa_service, 'is_device_trusted') as mock_trusted:
            mock_trusted.return_value = False
            
            result = mfa_service.revoke_trusted_device("test123", "user123")
            
            assert result is False
            mock_redis.delete.assert_not_called()
    
    def test_disable_mfa_method_totp(self, mfa_service, mock_redis):
        """Test disabling TOTP MFA method."""
        result = mfa_service.disable_mfa_method("user123", MFAMethodType.TOTP)
        
        assert result is True
        mock_redis.delete.assert_called_once_with("mfa_totp_confirmed:user123")
    
    def test_disable_mfa_method_backup_codes(self, mfa_service, mock_redis):
        """Test disabling backup codes MFA method."""
        result = mfa_service.disable_mfa_method("user123", MFAMethodType.BACKUP_CODES)
        
        assert result is True
        mock_redis.delete.assert_called_once_with("mfa_backup_codes:user123")
    
    def test_get_mfa_methods_with_totp(self, mfa_service, mock_redis):
        """Test getting MFA methods with TOTP enabled."""
        mock_redis.get.side_effect = [b"TESTSECRET123", b"hash1|hash2|hash3"]
        
        methods = mfa_service.get_mfa_methods("user123")
        
        assert len(methods) == 2
        
        # Check TOTP method
        totp_method = next(m for m in methods if m.method_type == MFAMethodType.TOTP)
        assert totp_method.status == MFAMethodStatus.ACTIVE
        assert totp_method.display_name == "Authenticator App"
        assert totp_method.is_primary is True
        
        # Check backup codes method
        backup_method = next(m for m in methods if m.method_type == MFAMethodType.BACKUP_CODES)
        assert backup_method.status == MFAMethodStatus.ACTIVE
        assert backup_method.display_name == "Backup Codes"
        assert backup_method.backup_codes_remaining == 3
    
    def test_get_mfa_methods_no_methods(self, mfa_service, mock_redis):
        """Test getting MFA methods when no methods are enabled."""
        mock_redis.get.return_value = None
        
        methods = mfa_service.get_mfa_methods("user123")
        
        assert len(methods) == 0
    
    def test_get_mfa_statistics(self, mfa_service):
        """Test getting MFA statistics."""
        statistics = mfa_service.get_mfa_statistics()
        
        assert isinstance(statistics, MFAStatisticsDTO)
        assert statistics.total_users == 1000
        assert statistics.mfa_enabled_users == 650
        assert statistics.mfa_adoption_rate == 65.0
        assert "totp" in statistics.method_usage
        assert "sms" in statistics.method_usage
        assert "email" in statistics.method_usage
        assert "backup_codes" in statistics.method_usage
    
    def test_get_trusted_devices(self, mfa_service):
        """Test getting trusted devices."""
        devices = mfa_service.get_trusted_devices("user123")
        
        # This is a simplified implementation that returns empty list
        assert devices == []
    
    def test_without_redis_client(self):
        """Test MFA service without Redis client."""
        service = MFAService(redis_client=None)
        
        # Test that methods handle missing Redis gracefully
        assert service.confirm_totp_setup("user123", "123456") is False
        assert service.verify_sms_code("user123", "123456") is False
        assert service.verify_email_code("user123", "123456") is False
        assert service.verify_backup_code("user123", "12345678") is False
        assert service.get_backup_codes_count("user123") == 0
        assert service.disable_mfa_method("user123", MFAMethodType.TOTP) is False
        assert service.is_device_trusted("test123", "user123") is False
        assert service.revoke_trusted_device("test123", "user123") is False
    
    def test_error_handling_in_totp_verification(self, mfa_service, mock_redis):
        """Test error handling in TOTP verification."""
        mock_redis.get.side_effect = Exception("Redis error")
        
        result = mfa_service.verify_totp_code("user123", "123456")
        
        assert result is False
    
    def test_error_handling_in_sms_sending(self, mfa_service, mock_sms_service):
        """Test error handling in SMS sending."""
        mock_sms_service.send_sms.side_effect = Exception("SMS error")
        
        result = mfa_service.send_sms_code("user123", "+1234567890")
        
        assert result is False
    
    def test_error_handling_in_email_sending(self, mfa_service, mock_email_service):
        """Test error handling in email sending."""
        mock_email_service.send_email.side_effect = Exception("Email error")
        
        result = mfa_service.send_email_code("user123", "test@example.com")
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])