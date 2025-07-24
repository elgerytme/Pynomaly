"""Security tests for cryptographic functions and data protection."""

import pytest
import hashlib
import hmac
import secrets
import base64
import os
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch

from anomaly_detection.infrastructure.security.encryption import DataEncryption
from anomaly_detection.infrastructure.security.hashing import SecureHasher
from anomaly_detection.infrastructure.security.key_management import KeyManager


@pytest.mark.security
class TestDataEncryption:
    """Test data encryption and decryption security."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_encryption = DataEncryption()
        self.test_data = "sensitive anomaly detection data"
        self.test_binary_data = b"binary test data with \x00 null bytes"
    
    def test_symmetric_encryption_decryption(self):
        """Test symmetric encryption and decryption."""
        # Test basic encryption/decryption
        encrypted_data = self.data_encryption.encrypt(self.test_data)
        decrypted_data = self.data_encryption.decrypt(encrypted_data)
        
        assert decrypted_data == self.test_data
        assert encrypted_data != self.test_data
        assert len(encrypted_data) > len(self.test_data)
    
    def test_encryption_key_rotation(self):
        """Test encryption key rotation."""
        # Encrypt with original key
        encrypted_v1 = self.data_encryption.encrypt(self.test_data, key_version=1)
        
        # Rotate key
        self.data_encryption.rotate_key()
        
        # Encrypt with new key
        encrypted_v2 = self.data_encryption.encrypt(self.test_data, key_version=2)
        
        # Should be able to decrypt both versions
        decrypted_v1 = self.data_encryption.decrypt(encrypted_v1, key_version=1)
        decrypted_v2 = self.data_encryption.decrypt(encrypted_v2, key_version=2)
        
        assert decrypted_v1 == self.test_data
        assert decrypted_v2 == self.test_data
        assert encrypted_v1 != encrypted_v2  # Different keys produce different ciphertext
    
    def test_encryption_with_iv_uniqueness(self):
        """Test that encryption produces unique ciphertext with different IVs."""
        encrypted_1 = self.data_encryption.encrypt(self.test_data)
        encrypted_2 = self.data_encryption.encrypt(self.test_data)
        
        # Same plaintext should produce different ciphertext due to unique IVs
        assert encrypted_1 != encrypted_2
        
        # But both should decrypt to the same plaintext
        assert self.data_encryption.decrypt(encrypted_1) == self.test_data
        assert self.data_encryption.decrypt(encrypted_2) == self.test_data
    
    def test_tampered_ciphertext_detection(self):
        """Test detection of tampered ciphertext."""
        encrypted_data = self.data_encryption.encrypt(self.test_data)
        
        # Tamper with the ciphertext
        tampered_data = bytearray(encrypted_data)
        tampered_data[10] ^= 0xFF  # Flip bits in the middle
        tampered_data = bytes(tampered_data)
        
        # Should fail to decrypt or raise integrity error
        with pytest.raises((ValueError, Exception)):
            self.data_encryption.decrypt(tampered_data)
    
    def test_weak_key_rejection(self):
        """Test rejection of weak encryption keys."""
        weak_keys = [
            b"",  # Empty key
            b"123",  # Too short
            b"a" * 8,  # Predictable pattern
            b"\x00" * 32,  # All zeros
            b"\xFF" * 32,  # All ones
        ]
        
        for weak_key in weak_keys:
            with pytest.raises((ValueError, Exception)):
                DataEncryption(encryption_key=weak_key)
    
    def test_large_data_encryption(self):
        """Test encryption of large data."""
        # Test with large data (1MB)
        large_data = "x" * (1024 * 1024)
        
        encrypted = self.data_encryption.encrypt(large_data)
        decrypted = self.data_encryption.decrypt(encrypted)
        
        assert decrypted == large_data
        assert len(encrypted) > len(large_data)
    
    def test_binary_data_encryption(self):
        """Test encryption of binary data."""
        encrypted = self.data_encryption.encrypt(self.test_binary_data)
        decrypted = self.data_encryption.decrypt(encrypted)
        
        assert decrypted == self.test_binary_data
    
    def test_encryption_performance(self):
        """Test encryption performance doesn't degrade."""
        test_data = "performance test data" * 1000  # ~20KB
        
        start_time = time.perf_counter()
        encrypted = self.data_encryption.encrypt(test_data)
        encryption_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        decrypted = self.data_encryption.decrypt(encrypted)
        decryption_time = time.perf_counter() - start_time
        
        # Should complete within reasonable time (adjust based on system)
        assert encryption_time < 1.0, f"Encryption too slow: {encryption_time}s"
        assert decryption_time < 1.0, f"Decryption too slow: {decryption_time}s"
        assert decrypted == test_data


@pytest.mark.security
class TestAsymmetricCryptography:
    """Test asymmetric (public-key) cryptography."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Generate RSA key pair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.test_data = b"asymmetric encryption test data"
    
    def test_rsa_encryption_decryption(self):
        """Test RSA encryption and decryption."""
        # Encrypt with public key
        encrypted = self.public_key.encrypt(
            self.test_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt with private key
        decrypted = self.private_key.decrypt(
            encrypted,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        assert decrypted == self.test_data
        assert encrypted != self.test_data
    
    def test_digital_signature_verification(self):
        """Test digital signature creation and verification."""
        message = b"message to be signed"
        
        # Sign message with private key
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Verify signature with public key
        try:
            self.public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            signature_valid = True
        except Exception:
            signature_valid = False
        
        assert signature_valid
    
    def test_signature_tampering_detection(self):
        """Test detection of tampered signatures."""
        message = b"original message"
        tampered_message = b"tampered message"
        
        # Sign original message
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Try to verify signature against tampered message
        with pytest.raises(Exception):
            self.public_key.verify(
                signature,
                tampered_message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
    
    def test_key_size_validation(self):
        """Test RSA key size validation."""
        # Test weak key sizes
        weak_key_sizes = [512, 1024]  # Considered weak
        
        for key_size in weak_key_sizes:
            with pytest.warns(UserWarning):
                # Should generate warning for weak key sizes
                weak_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size
                )
    
    def test_key_serialization_security(self):
        """Test secure key serialization."""
        # Serialize private key with password
        password = b"strong_password_123"
        
        serialized_key = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(password)
        )
        
        # Should not contain plaintext key material
        assert b"BEGIN RSA PRIVATE KEY" not in serialized_key
        assert b"BEGIN PRIVATE KEY" in serialized_key or b"BEGIN ENCRYPTED PRIVATE KEY" in serialized_key
        
        # Should be able to deserialize with correct password
        loaded_key = serialization.load_pem_private_key(
            serialized_key,
            password=password
        )
        
        # Should fail with wrong password
        with pytest.raises(Exception):
            serialization.load_pem_private_key(
                serialized_key,
                password=b"wrong_password"
            )


@pytest.mark.security
class TestSecureHashing:
    """Test secure hashing functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.secure_hasher = SecureHasher()
        self.test_password = "user_password_123"
        self.test_data = "data to be hashed"
    
    def test_password_hashing(self):
        """Test secure password hashing."""
        # Hash password
        hashed_password = self.secure_hasher.hash_password(self.test_password)
        
        # Verify password
        assert self.secure_hasher.verify_password(self.test_password, hashed_password)
        
        # Wrong password should fail
        assert not self.secure_hasher.verify_password("wrong_password", hashed_password)
    
    def test_salt_uniqueness(self):
        """Test that password hashes use unique salts."""
        hash1 = self.secure_hasher.hash_password(self.test_password)
        hash2 = self.secure_hasher.hash_password(self.test_password)
        
        # Same password should produce different hashes due to unique salts
        assert hash1 != hash2
        
        # But both should verify correctly
        assert self.secure_hasher.verify_password(self.test_password, hash1)
        assert self.secure_hasher.verify_password(self.test_password, hash2)
    
    def test_hash_timing_attack_protection(self):
        """Test protection against timing attacks in hash verification."""
        correct_password = "correct_password"
        wrong_password = "wrong_password_123"
        
        hashed_password = self.secure_hasher.hash_password(correct_password)
        
        # Measure timing for correct password
        start_time = time.perf_counter()
        self.secure_hasher.verify_password(correct_password, hashed_password)
        correct_time = time.perf_counter() - start_time
        
        # Measure timing for wrong password
        start_time = time.perf_counter()
        self.secure_hasher.verify_password(wrong_password, hashed_password)
        wrong_time = time.perf_counter() - start_time
        
        # Times should be similar (constant-time comparison)
        time_difference = abs(correct_time - wrong_time)
        assert time_difference < 0.01, "Potential timing attack vulnerability"
    
    def test_weak_password_detection(self):
        """Test detection of weak passwords."""
        weak_passwords = [
            "",
            "123",
            "password",
            "12345678",
            "qwerty",
            "abc123",
            "a" * 100,  # Repetitive
        ]
        
        for weak_password in weak_passwords:
            with pytest.raises((ValueError, Exception)):
                self.secure_hasher.hash_password(weak_password, enforce_strength=True)
    
    def test_hmac_authentication(self):
        """Test HMAC for message authentication."""
        secret_key = secrets.token_bytes(32)
        message = b"authenticated message"
        
        # Generate HMAC
        mac = hmac.new(secret_key, message, hashlib.sha256).hexdigest()
        
        # Verify HMAC
        expected_mac = hmac.new(secret_key, message, hashlib.sha256).hexdigest()
        assert hmac.compare_digest(mac, expected_mac)
        
        # Wrong key should produce different MAC
        wrong_key = secrets.token_bytes(32)
        wrong_mac = hmac.new(wrong_key, message, hashlib.sha256).hexdigest()
        assert not hmac.compare_digest(mac, wrong_mac)
        
        # Tampered message should produce different MAC
        tampered_message = b"tampered message"
        tampered_mac = hmac.new(secret_key, tampered_message, hashlib.sha256).hexdigest()
        assert not hmac.compare_digest(mac, tampered_mac)
    
    def test_hash_algorithm_strength(self):
        """Test use of strong hash algorithms."""
        data = b"test data for hashing"
        
        # Test strong algorithms
        strong_hashes = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "sha384": hashlib.sha384(data).hexdigest(),
            "sha512": hashlib.sha512(data).hexdigest(),
        }
        
        for algorithm, hash_value in strong_hashes.items():
            assert len(hash_value) > 0
            assert hash_value != data.hex()  # Should be different from input
        
        # Test that weak algorithms are avoided
        weak_algorithms = ["md5", "sha1"]
        for algorithm in weak_algorithms:
            with pytest.warns(UserWarning):
                # Should warn about weak hash algorithms
                getattr(hashlib, algorithm)(data).hexdigest()


@pytest.mark.security
class TestKeyManagement:
    """Test cryptographic key management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key_manager = KeyManager()
    
    def test_key_generation(self):
        """Test secure key generation."""
        # Generate symmetric key
        symmetric_key = self.key_manager.generate_symmetric_key(256)  # 256-bit key
        
        assert len(symmetric_key) == 32  # 256 bits = 32 bytes
        assert symmetric_key != b"\x00" * 32  # Should not be all zeros
        
        # Generate multiple keys - should be unique
        key1 = self.key_manager.generate_symmetric_key(256)
        key2 = self.key_manager.generate_symmetric_key(256)
        assert key1 != key2
    
    def test_key_derivation(self):
        """Test key derivation from passwords."""
        password = b"user_password"
        salt = os.urandom(16)
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # High iteration count
        )
        derived_key = kdf.derive(password)
        
        assert len(derived_key) == 32
        
        # Same password and salt should produce same key
        kdf2 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        derived_key2 = kdf2.derive(password)
        assert derived_key == derived_key2
        
        # Different salt should produce different key
        different_salt = os.urandom(16)
        kdf3 = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=different_salt,
            iterations=100000,
        )
        derived_key3 = kdf3.derive(password)
        assert derived_key != derived_key3
    
    def test_key_storage_security(self):
        """Test secure key storage."""
        key_id = "test_key_1"
        key_material = self.key_manager.generate_symmetric_key(256)
        
        # Store key
        self.key_manager.store_key(key_id, key_material)
        
        # Retrieve key
        retrieved_key = self.key_manager.get_key(key_id)
        assert retrieved_key == key_material
        
        # Non-existent key should raise error
        with pytest.raises((KeyError, ValueError)):
            self.key_manager.get_key("non_existent_key")
    
    def test_key_rotation(self):
        """Test key rotation functionality."""
        key_id = "rotatable_key"
        original_key = self.key_manager.generate_symmetric_key(256)
        
        # Store original key
        self.key_manager.store_key(key_id, original_key, version=1)
        
        # Rotate key
        new_key = self.key_manager.rotate_key(key_id)
        
        # Should be able to access both versions
        assert self.key_manager.get_key(key_id, version=1) == original_key
        assert self.key_manager.get_key(key_id, version=2) == new_key
        assert original_key != new_key
    
    def test_key_access_control(self):
        """Test key access control."""
        restricted_key_id = "admin_only_key"
        key_material = self.key_manager.generate_symmetric_key(256)
        
        # Store key with access control
        self.key_manager.store_key(
            restricted_key_id, 
            key_material,
            allowed_users=["admin_user"]
        )
        
        # Admin user should be able to access
        retrieved_key = self.key_manager.get_key(
            restricted_key_id, 
            requesting_user="admin_user"
        )
        assert retrieved_key == key_material
        
        # Regular user should not be able to access
        with pytest.raises((PermissionError, ValueError)):
            self.key_manager.get_key(
                restricted_key_id,
                requesting_user="regular_user"
            )
    
    def test_key_destruction(self):
        """Test secure key destruction."""
        key_id = "temporary_key"
        key_material = self.key_manager.generate_symmetric_key(256)
        
        # Store key
        self.key_manager.store_key(key_id, key_material)
        
        # Verify key exists
        assert self.key_manager.get_key(key_id) == key_material
        
        # Destroy key
        self.key_manager.destroy_key(key_id)
        
        # Key should no longer be accessible
        with pytest.raises((KeyError, ValueError)):
            self.key_manager.get_key(key_id)


@pytest.mark.security
class TestRandomnessAndEntropy:
    """Test randomness and entropy in cryptographic operations."""
    
    def test_random_number_quality(self):
        """Test quality of random number generation."""
        # Generate multiple random values
        random_values = [secrets.token_bytes(32) for _ in range(100)]
        
        # All values should be unique
        assert len(set(random_values)) == len(random_values)
        
        # No value should be all zeros or all ones
        for value in random_values[:10]:  # Check first 10
            assert value != b"\x00" * 32
            assert value != b"\xFF" * 32
    
    def test_entropy_estimation(self):
        """Test entropy estimation for random data."""
        # Generate random data
        random_data = secrets.token_bytes(1024)
        
        # Simple entropy check - count unique bytes
        unique_bytes = len(set(random_data))
        entropy_ratio = unique_bytes / 256  # Max possible unique bytes
        
        # Should have reasonable entropy (> 70% unique bytes)
        assert entropy_ratio > 0.7, f"Low entropy detected: {entropy_ratio}"
    
    def test_predictable_sequence_detection(self):
        """Test detection of predictable sequences."""
        # Test various predictable patterns
        predictable_patterns = [
            b"\x00" * 32,  # All zeros
            b"\xFF" * 32,  # All ones
            b"\x01\x02\x03\x04" * 8,  # Repeating pattern
            bytes(range(32)),  # Sequential
        ]
        
        for pattern in predictable_patterns:
            # These should be detected as low-entropy
            unique_bytes = len(set(pattern))
            entropy_ratio = unique_bytes / len(pattern)
            assert entropy_ratio < 0.5, f"Pattern not detected as predictable: {pattern.hex()}"
    
    def test_seed_independence(self):
        """Test that cryptographic operations are seed-independent."""
        # Generate values in different processes/contexts
        values_set1 = [secrets.token_bytes(16) for _ in range(10)]
        values_set2 = [secrets.token_bytes(16) for _ in range(10)]
        
        # Sets should be completely different
        intersection = set(values_set1) & set(values_set2)
        assert len(intersection) == 0, "Random values showed correlation"


if __name__ == "__main__":
    # Run specific cryptography tests
    pytest.main([
        __file__ + "::TestDataEncryption::test_symmetric_encryption_decryption",
        "-v", "-s", "--tb=short"
    ])