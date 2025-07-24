#!/usr/bin/env python3
"""
Security Configuration Validator
Validates security scanning configuration and ensures all tools are properly configured.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml


class SecurityConfigValidator:
    """Validates security scanning configuration and tool setup."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load security configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            sys.exit(1)
    
    def validate_tool_availability(self) -> None:
        """Check if all enabled security tools are available."""
        tools = self.config.get('security_policy', {}).get('tools', {})
        
        tool_commands = {
            'bandit': ['bandit', '--version'],
            'safety': ['safety', '--version'],
            'semgrep': ['semgrep', '--version'],
            'trivy': ['trivy', '--version'],
            'hadolint': ['hadolint', '--version'],
            'detect_secrets': ['detect-secrets', '--version'],
        }
        
        for tool_name, tool_config in tools.items():
            if not tool_config.get('enabled', True):
                continue
                
            if tool_name in tool_commands:
                try:
                    subprocess.run(
                        tool_commands[tool_name],
                        capture_output=True,
                        check=True
                    )
                    print(f"✅ {tool_name} is available")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    self.errors.append(f"{tool_name} is enabled but not installed")
    
    def validate_file_references(self) -> None:
        """Validate that referenced configuration files exist."""
        repo_root = self.config_path.parent.parent
        
        # Check Bandit config
        bandit_config = self.config.get('security_policy', {}).get('tools', {}).get('bandit', {})
        if bandit_config.get('config_file'):
            config_file = repo_root / bandit_config['config_file']
            if not config_file.exists():
                self.warnings.append(f"Bandit config file not found: {config_file}")
        
        # Check detect-secrets baseline
        detect_secrets_config = self.config.get('security_policy', {}).get('tools', {}).get('detect_secrets', {})
        if detect_secrets_config.get('baseline_file'):
            baseline_file = repo_root / detect_secrets_config['baseline_file']
            if not baseline_file.exists():
                self.warnings.append(f"detect-secrets baseline not found: {baseline_file}")
    
    def validate_thresholds(self) -> None:
        """Validate risk tolerance thresholds are reasonable."""
        risk_tolerance = self.config.get('security_policy', {}).get('risk_tolerance', {})
        
        # Check that critical threshold is 0
        if risk_tolerance.get('critical', 1) > 0:
            self.warnings.append("Critical vulnerability threshold > 0 may be risky for production")
        
        # Check that thresholds are in ascending order
        thresholds = [
            risk_tolerance.get('critical', 0),
            risk_tolerance.get('high', 0),
            risk_tolerance.get('medium', 0),
            risk_tolerance.get('low', 0)
        ]
        
        if thresholds != sorted(thresholds):
            self.warnings.append("Risk tolerance thresholds should be in ascending order")
    
    def validate_license_policy(self) -> None:
        """Validate license policy configuration."""
        license_policy = self.config.get('license_policy', {})
        
        allowed = set(license_policy.get('allowed_licenses', []))
        prohibited = set(license_policy.get('prohibited_licenses', []))
        
        # Check for overlaps
        overlap = allowed.intersection(prohibited)
        if overlap:
            self.errors.append(f"Licenses cannot be both allowed and prohibited: {overlap}")
        
        # Check for common licenses
        common_licenses = {'MIT', 'Apache-2.0', 'BSD-3-Clause'}
        if not common_licenses.intersection(allowed):
            self.warnings.append("No common permissive licenses in allowed list")
    
    def validate_dast_config(self) -> None:
        """Validate DAST configuration."""
        dast_config = self.config.get('dast_config', {})
        
        target_urls = dast_config.get('target_urls', [])
        if not target_urls:
            self.warnings.append("No target URLs configured for DAST scanning")
        
        # Check for localhost in production
        for url in target_urls:
            if 'localhost' in url or '127.0.0.1' in url:
                self.warnings.append(f"Localhost URL in DAST config may not work in CI: {url}")
    
    def validate_compliance_frameworks(self) -> None:
        """Validate compliance framework configuration."""
        compliance = self.config.get('compliance', {}).get('frameworks', {})
        
        enabled_frameworks = [name for name, config in compliance.items() 
                            if config.get('enabled', False)]
        
        if not enabled_frameworks:
            self.warnings.append("No compliance frameworks enabled")
        
        # Recommend OWASP Top 10 for web applications
        if 'owasp_top10' not in enabled_frameworks:
            self.warnings.append("OWASP Top 10 framework not enabled")
    
    def validate_quality_gates(self) -> None:
        """Validate quality gate configuration."""
        quality_gates = self.config.get('quality_gates', {})
        
        block_deployment = quality_gates.get('block_deployment', {})
        
        if not block_deployment.get('critical_vulnerabilities', False):
            self.errors.append("Quality gate should block deployment on critical vulnerabilities")
        
        if not block_deployment.get('secrets_detected', False):
            self.errors.append("Quality gate should block deployment when secrets are detected")
    
    def generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on configuration."""
        recommendations = []
        
        tools = self.config.get('security_policy', {}).get('tools', {})
        
        # Check for comprehensive tool coverage
        sast_tools = ['bandit', 'semgrep', 'codeql']
        enabled_sast = [tool for tool in sast_tools if tools.get(tool, {}).get('enabled', False)]
        
        if len(enabled_sast) < 2:
            recommendations.append("Enable multiple SAST tools for better coverage")
        
        # Check for container security
        container_tools = ['trivy', 'hadolint']
        enabled_container = [tool for tool in container_tools if tools.get(tool, {}).get('enabled', False)]
        
        if len(enabled_container) < 2:
            recommendations.append("Enable comprehensive container security scanning")
        
        # Check for secret detection
        if not tools.get('detect_secrets', {}).get('enabled', False):
            recommendations.append("Enable secret detection to prevent credential leaks")
        
        return recommendations
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating security configuration...")
        print()
        
        self.validate_tool_availability()
        self.validate_file_references()
        self.validate_thresholds()
        self.validate_license_policy()
        self.validate_dast_config()
        self.validate_compliance_frameworks()
        self.validate_quality_gates()
        
        # Print results
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   • {error}")
            print()
        
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()
        
        recommendations = self.generate_recommendations()
        if recommendations:
            print("💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   • {rec}")
            print()
        
        if not self.errors and not self.warnings:
            print("✅ Security configuration is valid!")
        elif not self.errors:
            print("✅ Security configuration is valid with warnings")
        else:
            print("❌ Security configuration has errors that must be fixed")
        
        return len(self.errors) == 0


def main():
    """Main function."""
    config_path = Path(__file__).parent.parent / "security-scanning-config.yaml"
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    validator = SecurityConfigValidator(config_path)
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()