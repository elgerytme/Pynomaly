#!/usr/bin/env python3
"""Simple test script for enhanced settings functionality."""

import os
import sys

sys.path.insert(0, 'src')

from pynomaly.infrastructure.config.settings import (
    EnvironmentSecretsProvider,
    MultiEnvSettingsConfigDict,
    SecretsBackend,
    SecretsManager,
    Settings,
)


def test_basic_functionality():
    """Test basic functionality."""
    print("Testing basic functionality...")
    
    # Test settings creation
    settings = Settings()
    print(f"Settings created: {settings.app.name}")
    
    # Test secrets manager
    manager = SecretsManager(backends=[SecretsBackend.ENV])
    print(f"Secrets manager created with backends: {manager.backends}")
    
    # Test environment secret retrieval
    os.environ['TEST_SECRET'] = 'test_value'
    secret = manager.get_secret('TEST_SECRET')
    print(f"Secret retrieved: {secret}")
    assert secret == 'test_value'
    
    print("Basic functionality tests passed!")

def test_secrets_providers():
    """Test secrets providers."""
    print("Testing secrets providers...")
    
    # Test environment provider
    provider = EnvironmentSecretsProvider()
    assert provider.is_available() == True
    
    os.environ['PROVIDER_TEST'] = 'provider_value'
    assert provider.get_secret('PROVIDER_TEST') == 'provider_value'
    assert provider.get_secret('NON_EXISTENT') is None
    
    print("Secrets providers tests passed!")

def test_multi_env_config():
    """Test multi-environment configuration."""
    print("Testing multi-environment configuration...")
    
    config = MultiEnvSettingsConfigDict()
    # The config is a dict-like object, check it has the expected keys
    assert 'env_prefix' in config
    assert config['env_prefix'] == 'PYNOMALY_'
    assert config['case_sensitive'] == False
    
    print("Multi-env config tests passed!")

def test_enhanced_settings():
    """Test enhanced settings features."""
    print("Testing enhanced settings features...")
    
    # Test with environment variables
    os.environ['PYNOMALY_SECRET_KEY'] = 'env-secret-key'
    os.environ['PYNOMALY_DATABASE_URL'] = 'sqlite:///test.db'
    
    settings = Settings(secrets_backends=[SecretsBackend.ENV])
    
    # Test secret retrieval methods
    assert settings.get_secret_key() == 'env-secret-key'
    assert settings.get_database_url() == 'sqlite:///test.db'
    
    # Test database configuration
    db_config = settings.get_database_config()
    assert db_config['url'] == 'sqlite:///test.db'
    assert 'pool_size' in db_config
    
    print("Enhanced settings tests passed!")

def test_environment_specific_config():
    """Test environment-specific configuration."""
    print("Testing environment-specific configuration...")
    
    # Test development environment
    os.environ['PYNOMALY_ENV'] = 'development'
    os.environ['PYNOMALY_APP__DEBUG'] = 'true'
    
    settings = Settings()
    assert settings.app.environment == 'development'
    
    # Test production environment  
    os.environ['PYNOMALY_ENV'] = 'production'
    os.environ['PYNOMALY_APP__DEBUG'] = 'false'
    
    settings = Settings()
    assert settings.app.environment == 'production'
    
    print("Environment-specific config tests passed!")

def main():
    """Run all tests."""
    print("Starting enhanced settings tests...")
    
    try:
        test_basic_functionality()
        test_secrets_providers()
        test_multi_env_config()
        test_enhanced_settings()
        test_environment_specific_config()
        
        print("\nAll tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
