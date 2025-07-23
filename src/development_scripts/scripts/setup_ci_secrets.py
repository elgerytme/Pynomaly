#!/usr/bin/env python3
"""
CI/CD Secrets Setup Script

This script helps configure the required secrets for GitHub Actions workflows.
It provides guidance and validation for setting up publishing credentials.

Usage:
    python3 setup_ci_secrets.py --check
    python3 setup_ci_secrets.py --setup
    python3 setup_ci_secrets.py --validate
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional


class CISecretsManager:
    """Manages CI/CD secrets configuration and validation."""
    
    REQUIRED_SECRETS = {
        'python': {
            'PYPI_API_TOKEN': {
                'description': 'PyPI API token for production publishing',
                'url': 'https://pypi.org/manage/account/token/',
                'format': 'pypi-*',
                'required': True
            },
            'TESTPYPI_API_TOKEN': {
                'description': 'TestPyPI API token for test publishing',
                'url': 'https://test.pypi.org/manage/account/token/',
                'format': 'pypi-*',
                'required': False
            }
        },
        'typescript': {
            'NPM_TOKEN': {
                'description': 'NPM authentication token',
                'url': 'https://www.npmjs.com/settings/tokens',
                'format': 'npm_*',
                'required': True
            }
        },
        'java': {
            'OSSRH_USERNAME': {
                'description': 'Sonatype OSSRH username',
                'url': 'https://issues.sonatype.org/secure/Dashboard.jspa',
                'format': 'username',
                'required': True
            },
            'OSSRH_TOKEN': {
                'description': 'Sonatype OSSRH token',
                'url': 'https://s01.oss.sonatype.org/#profile;User%20Token',
                'format': 'token',
                'required': True
            },
            'MAVEN_GPG_PRIVATE_KEY': {
                'description': 'GPG private key for artifact signing',
                'url': 'https://central.sonatype.org/publish/requirements/gpg/',
                'format': '-----BEGIN PGP PRIVATE KEY BLOCK-----',
                'required': True
            },
            'MAVEN_GPG_PASSPHRASE': {
                'description': 'GPG key passphrase',
                'url': 'https://central.sonatype.org/publish/requirements/gpg/',
                'format': 'passphrase',
                'required': True
            }
        }
    }
    
    def __init__(self):
        self.gh_available = self._check_gh_cli()
    
    def _check_gh_cli(self) -> bool:
        """Check if GitHub CLI is available."""
        try:
            result = subprocess.run(['gh', '--version'], 
                                 capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _run_gh_command(self, args: List[str]) -> Optional[str]:
        """Run a GitHub CLI command."""
        if not self.gh_available:
            print("❌ GitHub CLI is not available. Please install gh CLI.")
            return None
        
        try:
            result = subprocess.run(['gh'] + args, 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"❌ GitHub CLI error: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Error running gh command: {e}")
            return None
    
    def check_existing_secrets(self) -> Dict[str, bool]:
        """Check which secrets are already configured."""
        print("🔍 Checking existing GitHub secrets...")
        
        output = self._run_gh_command(['secret', 'list'])
        if output is None:
            return {}
        
        existing_secrets = set()
        for line in output.split('\n'):
            if line.strip():
                secret_name = line.split()[0]
                existing_secrets.add(secret_name)
        
        results = {}
        for sdk_type, secrets in self.REQUIRED_SECRETS.items():
            for secret_name, config in secrets.items():
                results[secret_name] = secret_name in existing_secrets
        
        return results
    
    def display_secrets_status(self, existing_secrets: Dict[str, bool]):
        """Display the status of all required secrets."""
        print("\\n📋 Secrets Configuration Status\\n")
        
        for sdk_type, secrets in self.REQUIRED_SECRETS.items():
            print(f"### {sdk_type.title()} SDK")
            
            for secret_name, config in secrets.items():
                status = "✅" if existing_secrets.get(secret_name, False) else "❌"
                required = "Required" if config['required'] else "Optional"
                
                print(f"  {status} {secret_name:<25} ({required})")
                print(f"     {config['description']}")
                
                if not existing_secrets.get(secret_name, False):
                    print(f"     📖 Setup: {config['url']}")
                print()
    
    def setup_secrets_interactive(self):
        """Interactive secrets setup process."""
        print("🚀 Interactive Secrets Setup\\n")
        
        if not self.gh_available:
            print("❌ GitHub CLI is required for automatic setup.")
            print("Please install gh CLI: https://cli.github.com/")
            print("\\nAlternatively, set up secrets manually in GitHub:")
            print("https://github.com/your-repo/settings/secrets/actions")
            return
        
        existing_secrets = self.check_existing_secrets()
        
        for sdk_type, secrets in self.REQUIRED_SECRETS.items():
            print(f"\\n=== {sdk_type.title()} SDK Secrets ===\\n")
            
            for secret_name, config in secrets.items():
                if existing_secrets.get(secret_name, False):
                    print(f"✅ {secret_name} is already configured")
                    continue
                
                print(f"\\n🔧 Setting up {secret_name}")
                print(f"Description: {config['description']}")
                print(f"Get token from: {config['url']}")
                print(f"Expected format: {config['format']}")
                
                if config['required']:
                    print("⚠️  This secret is REQUIRED for publishing")
                else:
                    print("ℹ️  This secret is optional")
                
                choice = input("\\nSet this secret now? (y/N/skip): ").lower().strip()
                
                if choice == 'y':
                    secret_value = input(f"Enter value for {secret_name}: ").strip()
                    if secret_value:
                        result = self._run_gh_command(['secret', 'set', secret_name, 
                                                     '--body', secret_value])
                        if result is not None:
                            print(f"✅ {secret_name} configured successfully")
                        else:
                            print(f"❌ Failed to configure {secret_name}")
                    else:
                        print("❌ Empty value, skipping...")
                elif choice == 'skip':
                    print("⏭️  Skipping all remaining secrets")
                    return
                else:
                    print("⏭️  Skipping this secret")
    
    def validate_secrets_format(self):
        """Validate the format of configured secrets."""
        print("🔍 Validating secrets format...\\n")
        
        # Note: We can't actually read secret values, so we provide guidance
        print("ℹ️  Secret values cannot be read back from GitHub for security reasons.")
        print("Please verify your secrets match these formats:\\n")
        
        for sdk_type, secrets in self.REQUIRED_SECRETS.items():
            print(f"### {sdk_type.title()} SDK")
            
            for secret_name, config in secrets.items():
                print(f"  {secret_name}:")
                print(f"    Expected format: {config['format']}")
                
                if secret_name == 'PYPI_API_TOKEN':
                    print("    ✅ Should start with 'pypi-'")
                    print("    ✅ Should be about 90+ characters long")
                elif secret_name == 'NPM_TOKEN':
                    print("    ✅ Should start with 'npm_'")
                    print("    ✅ Should be about 36+ characters long")
                elif secret_name == 'MAVEN_GPG_PRIVATE_KEY':
                    print("    ✅ Should start with '-----BEGIN PGP PRIVATE KEY BLOCK-----'")
                    print("    ✅ Should end with '-----END PGP PRIVATE KEY BLOCK-----'")
                    print("    ✅ Should be the complete private key block")
                
                print()
    
    def generate_setup_instructions(self):
        """Generate detailed setup instructions."""
        print("📚 Detailed Setup Instructions\\n")
        
        instructions = {
            'python': self._python_instructions(),
            'typescript': self._typescript_instructions(),
            'java': self._java_instructions()
        }
        
        for sdk_type, content in instructions.items():
            print(f"=== {sdk_type.title()} SDK Setup ===\\n")
            print(content)
            print("\\n" + "="*50 + "\\n")
    
    def _python_instructions(self) -> str:
        return """
1. Create PyPI API Token:
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Choose scope: "Entire account" or specific project
   - Copy the token (starts with 'pypi-')

2. Create TestPyPI API Token (optional):
   - Go to https://test.pypi.org/manage/account/token/
   - Follow same process as above

3. Set GitHub secrets:
   ```bash
   gh secret set PYPI_API_TOKEN --body "pypi-your-token-here"
   gh secret set TESTPYPI_API_TOKEN --body "pypi-your-test-token-here"
   ```
"""
    
    def _typescript_instructions(self) -> str:
        return """
1. Create NPM Access Token:
   - Go to https://www.npmjs.com/settings/tokens
   - Click "Generate New Token"
   - Choose "Automation" type for CI/CD
   - Copy the token (starts with 'npm_')

2. Set GitHub secret:
   ```bash
   gh secret set NPM_TOKEN --body "npm_your-token-here"
   ```

3. Ensure package is scoped correctly:
   - Package name: @anomaly_detection/client
   - Scope should be configured for public publishing
"""
    
    def _java_instructions(self) -> str:
        return """
1. Create Sonatype OSSRH account:
   - Go to https://issues.sonatype.org/secure/Dashboard.jspa
   - Create JIRA account
   - Create ticket for new project hosting

2. Generate User Token:
   - Go to https://s01.oss.sonatype.org/#profile;User%20Token
   - Generate access token

3. Create GPG Key:
   ```bash
   # Generate key
   gpg --gen-key
   
   # Export private key
   gpg --armor --export-secret-keys your-email@example.com
   
   # Upload public key to keyserver
   gpg --keyserver keyserver.ubuntu.com --send-keys YOUR-KEY-ID
   ```

4. Set GitHub secrets:
   ```bash
   gh secret set OSSRH_USERNAME --body "your-username"
   gh secret set OSSRH_TOKEN --body "your-token"
   gh secret set MAVEN_GPG_PRIVATE_KEY --body "$(gpg --armor --export-secret-keys your-email@example.com)"
   gh secret set MAVEN_GPG_PASSPHRASE --body "your-gpg-passphrase"
   ```
"""


def main():
    parser = argparse.ArgumentParser(description='Setup CI/CD secrets for SDK publishing')
    parser.add_argument('--check', action='store_true', 
                       help='Check existing secrets configuration')
    parser.add_argument('--setup', action='store_true', 
                       help='Interactive secrets setup')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate secrets format')
    parser.add_argument('--instructions', action='store_true', 
                       help='Show detailed setup instructions')
    
    args = parser.parse_args()
    
    manager = CISecretsManager()
    
    if args.check:
        existing_secrets = manager.check_existing_secrets()
        manager.display_secrets_status(existing_secrets)
    
    elif args.setup:
        manager.setup_secrets_interactive()
    
    elif args.validate:
        manager.validate_secrets_format()
    
    elif args.instructions:
        manager.generate_setup_instructions()
    
    else:
        # Default: show status and offer options
        existing_secrets = manager.check_existing_secrets()
        manager.display_secrets_status(existing_secrets)
        
        print("\\n🚀 Next Steps:")
        print("  python3 setup_ci_secrets.py --setup        # Interactive setup")
        print("  python3 setup_ci_secrets.py --instructions # Detailed guides")
        print("  python3 setup_ci_secrets.py --validate     # Validate format")


if __name__ == '__main__':
    main()