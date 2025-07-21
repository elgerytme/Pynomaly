#!/usr/bin/env python3
"""
Advanced Security Hardening Script for anomaly_detection Production Environment

This script implements comprehensive security hardening measures for production deployment.
"""

import json
import logging
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_hardening.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SecurityHardeningManager:
    """Comprehensive security hardening manager."""

    def __init__(self, project_root: str = "/mnt/c/Users/andre/anomaly_detection"):
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / "config" / "security"
        self.scripts_dir = self.project_root / "scripts"
        self.docker_dir = self.project_root / "docker"

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.docker_dir.mkdir(parents=True, exist_ok=True)

    def log_action(self, action: str, status: str = "INFO", details: str = "") -> None:
        """Log security hardening actions."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "action": action,
            "status": status,
            "details": details
        }

        if status == "ERROR":
            logger.error(f"{action}: {details}")
        elif status == "WARNING":
            logger.warning(f"{action}: {details}")
        else:
            logger.info(f"{action}: {details}")

    def generate_secure_keys(self) -> dict[str, str]:
        """Generate secure cryptographic keys."""
        self.log_action("Generating cryptographic keys", "INFO")

        from cryptography.fernet import Fernet

        keys = {
            "JWT_SECRET_KEY": secrets.token_urlsafe(64),
            "ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "FERNET_KEY": Fernet.generate_key().decode(),
            "HMAC_KEY": secrets.token_hex(32),
            "SYMMETRIC_KEY": secrets.token_hex(32),
            "DATABASE_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "PII_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "SENSITIVE_DATA_KEY": Fernet.generate_key().decode(),
            "FILE_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "SESSION_SECRET": secrets.token_urlsafe(32),
            "CSRF_SECRET": secrets.token_urlsafe(32)
        }

        # Save keys to secure file
        keys_file = self.config_dir / "keys.json"
        with open(keys_file, 'w') as f:
            json.dump(keys, f, indent=2)

        # Set secure permissions
        os.chmod(keys_file, 0o600)

        self.log_action("Generated and saved cryptographic keys", "SUCCESS", f"Keys saved to {keys_file}")
        return keys

    def create_security_env_file(self, keys: dict[str, str]) -> None:
        """Create secure environment file."""
        self.log_action("Creating security environment file", "INFO")

        env_content = f"""# anomaly_detection Security Environment Variables
# Generated on {datetime.now().isoformat()}
# WARNING: Keep this file secure and never commit to version control

# JWT Configuration
JWT_SECRET_KEY={keys['JWT_SECRET_KEY']}
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Encryption Keys
ENCRYPTION_KEY={keys['ENCRYPTION_KEY']}
FERNET_KEY={keys['FERNET_KEY']}
HMAC_KEY={keys['HMAC_KEY']}
SYMMETRIC_KEY={keys['SYMMETRIC_KEY']}

# Database Encryption
DATABASE_ENCRYPTION_KEY={keys['DATABASE_ENCRYPTION_KEY']}

# Field-Level Encryption
PII_ENCRYPTION_KEY={keys['PII_ENCRYPTION_KEY']}
SENSITIVE_DATA_KEY={keys['SENSITIVE_DATA_KEY']}
FILE_ENCRYPTION_KEY={keys['FILE_ENCRYPTION_KEY']}

# Session Security
SESSION_SECRET={keys['SESSION_SECRET']}
CSRF_SECRET={keys['CSRF_SECRET']}

# API Security
API_KEY_REQUIRED=true
RATE_LIMITING_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20

# CORS Configuration
CORS_ENABLED=true
CORS_ALLOWED_ORIGINS=https://app.anomaly_detection.com,https://dashboard.anomaly_detection.com

# Authentication
JWT_REQUIRED=true
TWO_FACTOR_AUTH=true
SESSION_TIMEOUT_MINUTES=30
MAX_CONCURRENT_SESSIONS=3

# SSL/TLS Configuration
SSL_REQUIRED=true
SSL_CERT_PATH=/etc/ssl/certs/anomaly_detection.crt
SSL_KEY_PATH=/etc/ssl/private/anomaly_detection.key

# Password Policy
PASSWORD_MIN_LENGTH=12
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_NUMBERS=true
PASSWORD_REQUIRE_SYMBOLS=true

# Audit and Monitoring
AUDIT_LOGGING=true
SECURITY_MONITORING=true
THREAT_DETECTION=true

# Environment
ENVIRONMENT=production
DEBUG_MODE=false
"""

        env_file = self.config_dir / ".env.security"
        with open(env_file, 'w') as f:
            f.write(env_content)

        # Set secure permissions
        os.chmod(env_file, 0o600)

        self.log_action("Created security environment file", "SUCCESS", f"File saved to {env_file}")

    def create_security_checklist(self) -> None:
        """Create security checklist."""
        self.log_action("Creating security checklist", "INFO")

        checklist = f"""# anomaly_detection Security Checklist
Generated on {datetime.now().isoformat()}

## Pre-Deployment Security Checklist

### 1. Environment Configuration
- [ ] All cryptographic keys generated and stored securely
- [ ] Environment variables configured with secure values
- [ ] Debug mode disabled in production
- [ ] Sensitive information removed from code
- [ ] Database credentials secured
- [ ] API keys and secrets properly managed

### 2. Application Security
- [ ] Authentication system implemented
- [ ] Authorization controls in place
- [ ] Input validation and sanitization
- [ ] Rate limiting configured
- [ ] Security headers configured
- [ ] CORS properly configured

### 3. Data Security
- [ ] Encryption at rest enabled
- [ ] Encryption in transit configured
- [ ] Field-level encryption for sensitive data
- [ ] Secure key management
- [ ] Data backup encryption
- [ ] GDPR compliance measures

### 4. Infrastructure Security
- [ ] Docker containers hardened
- [ ] Non-root user configuration
- [ ] Security scanning enabled
- [ ] Regular security updates scheduled

### 5. Monitoring and Logging
- [ ] Security event logging enabled
- [ ] Audit trail configuration
- [ ] Real-time monitoring alerts
- [ ] Security incident response plan

### 6. Access Control
- [ ] Multi-factor authentication enabled
- [ ] Role-based access control (RBAC)
- [ ] Secure password policies
- [ ] Session management

## Security Tools and Scripts

### Available Scripts
- advanced_security_hardening.py - Comprehensive hardening

### Configuration Files
- security_config.py - Application security settings
- .env.security - Environment variables
- keys.json - Cryptographic keys (secure storage)

## Notes
- All security measures should be tested in staging environment first
- Regular security updates and patches are critical
- Keep security documentation up to date
"""

        checklist_file = self.config_dir / "security_checklist.md"
        with open(checklist_file, 'w') as f:
            f.write(checklist)

        self.log_action("Created security checklist", "SUCCESS", f"Checklist saved to {checklist_file}")

    def run_security_hardening(self) -> None:
        """Run complete security hardening process."""
        self.log_action("Starting comprehensive security hardening", "INFO")

        try:
            # Generate secure keys
            keys = self.generate_secure_keys()

            # Create security environment file
            self.create_security_env_file(keys)

            # Create security checklist
            self.create_security_checklist()

            self.log_action("Security hardening completed successfully", "SUCCESS")

            # Print summary
            print("\n" + "="*50)
            print("🔒 Advanced Security Hardening Complete!")
            print("="*50)
            print(f"📁 Configuration files created in: {self.config_dir}")
            print(f"🛠️  Scripts created in: {self.scripts_dir}")
            print("\n📋 Files Created:")
            print("• keys.json - Cryptographic keys (secure)")
            print("• .env.security - Environment variables")
            print("• security_checklist.md - Security checklist")
            print("\n📋 Next Steps:")
            print("1. Review the security checklist")
            print("2. Configure SSL certificates")
            print("3. Set up monitoring and alerting")
            print("4. Test all security measures")
            print("5. Deploy to staging environment first")
            print("\n⚠️  Important:")
            print("- Keep .env.security file secure")
            print("- Never commit secrets to version control")
            print("- Regularly update security configurations")
            print("- Monitor security logs daily")
            print("="*50)

        except Exception as e:
            self.log_action("Security hardening failed", "ERROR", str(e))
            raise


def main():
    """Main function to run security hardening."""
    try:
        manager = SecurityHardeningManager()
        manager.run_security_hardening()
    except Exception as e:
        logger.error(f"Security hardening failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
