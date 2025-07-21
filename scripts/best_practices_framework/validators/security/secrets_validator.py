#!/usr/bin/env python3
"""
Secrets Detection Validator
===========================
Detects hardcoded secrets, passwords, API keys, and other sensitive information
"""

import re
import time
from pathlib import Path
from typing import List, Dict, Pattern, Tuple
import logging

from ...core.base_validator import BaseValidator, ValidationResult


class SecretsValidator(BaseValidator):
    """
    Validates that no secrets are hardcoded in the codebase.
    
    Detects:
    - Passwords and passphrases
    - API keys and tokens
    - Private keys and certificates
    - Database connection strings
    - Cloud credentials
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Common secret patterns (ordered by likelihood of false positives)
        self.secret_patterns: Dict[str, Pattern] = {
            # High confidence patterns (low false positive rate)
            'aws_access_key': re.compile(r'\b(AKIA[0-9A-Z]{16})\b', re.IGNORECASE),
            'aws_secret_key': re.compile(r'\b([0-9a-zA-Z+/]{40})\b'),
            'github_token': re.compile(r'\b(ghp_[0-9a-zA-Z]{36})\b'),
            'slack_token': re.compile(r'\b(xox[baprs]-[0-9a-zA-Z-]+)\b'),
            'stripe_key': re.compile(r'\b(sk_live_[0-9a-zA-Z]{24})\b'),
            'google_api_key': re.compile(r'\b(AIza[0-9A-Za-z_-]{35})\b'),
            'private_key_header': re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----'),
            
            # Medium confidence patterns
            'jwt_token': re.compile(r'\b(eyJ[0-9a-zA-Z+/]+=*\.eyJ[0-9a-zA-Z+/]+=*\.[0-9a-zA-Z+/]+=*)\b'),
            'generic_api_key': re.compile(r'\b(api[_-]?key|apikey)\s*[=:]\s*["\']([0-9a-zA-Z_-]{16,})["\']', re.IGNORECASE),
            'bearer_token': re.compile(r'\b(bearer\s+[0-9a-zA-Z_-]{16,})\b', re.IGNORECASE),
            
            # Lower confidence patterns (may have false positives)
            'password_assignment': re.compile(r'\b(password|passwd|pwd)\s*[=:]\s*["\']([^"\'\s]{6,})["\']', re.IGNORECASE),
            'secret_assignment': re.compile(r'\b(secret|secret_key|secretkey)\s*[=:]\s*["\']([0-9a-zA-Z_-]{8,})["\']', re.IGNORECASE),
            'token_assignment': re.compile(r'\b(token|auth_token|authtoken)\s*[=:]\s*["\']([0-9a-zA-Z_-]{12,})["\']', re.IGNORECASE),
            'connection_string': re.compile(r'(mongodb://|postgres://|mysql://|redis://)[^\s;]+', re.IGNORECASE),
            'database_url': re.compile(r'\b(database_url|db_url)\s*[=:]\s*["\']([^"\'\s]+://[^"\'\s]+)["\']', re.IGNORECASE),
        }
        
        # Common false positive patterns to exclude
        self.false_positive_patterns: List[Pattern] = [
            re.compile(r'\b(example|test|demo|placeholder|dummy|fake|mock)\b', re.IGNORECASE),
            re.compile(r'\b(your_key_here|insert_key|api_key_goes_here)\b', re.IGNORECASE),
            re.compile(r'\b(123456|password|secret|token)\b', re.IGNORECASE),  # Common dummy values
            re.compile(r'\$\{[^}]+\}'),  # Environment variable substitution
            re.compile(r'%[^%]+%'),      # Windows environment variables
            re.compile(r'<[^>]+>'),      # XML/HTML placeholders
        ]
        
        # File types to scan
        self.scannable_extensions = {
            '.py', '.js', '.ts', '.java', '.cs', '.go', '.rb', '.php', 
            '.cpp', '.c', '.h', '.hpp', '.rs', '.scala', '.kt',
            '.yml', '.yaml', '.json', '.xml', '.properties', '.env',
            '.sh', '.bash', '.ps1', '.bat', '.cmd',
            '.sql', '.md', '.txt', '.conf', '.ini', '.cfg'
        }
    
    def get_name(self) -> str:
        return "secrets_detector"
    
    def get_category(self) -> str:
        return "security"
    
    def get_description(self) -> str:
        return "Detects hardcoded secrets, passwords, and sensitive information"
    
    @classmethod
    def get_supported_file_types(cls) -> List[str]:
        return [
            '.py', '.js', '.ts', '.java', '.cs', '.go', '.rb', '.php',
            '.yml', '.yaml', '.json', '.xml', '.env', '.properties',
            '.sh', '.bash', '.sql', '.md', '.txt', '.conf'
        ]
    
    def should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed for secrets"""
        if not super().should_analyze_file(file_path):
            return False
        
        # Only scan files with supported extensions
        if file_path.suffix.lower() not in self.scannable_extensions:
            return False
        
        # Skip binary files and common non-text files
        binary_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', '.dat',
                           '.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip'}
        if file_path.suffix.lower() in binary_extensions:
            return False
        
        return True
    
    async def validate(self) -> ValidationResult:
        """Run secrets detection validation"""
        start_time = time.time()
        self.reset_violations()
        
        self.logger.info(f"🔍 Scanning for hardcoded secrets in {self.project_root}")
        
        # Get files to scan
        files_to_scan = self.get_project_files()
        scanned_files = 0
        
        for file_path in files_to_scan:
            if self.should_analyze_file(file_path):
                try:
                    await self._scan_file_for_secrets(file_path)
                    scanned_files += 1
                except Exception as e:
                    self.logger.warning(f"Could not scan file {file_path}: {e}")
        
        execution_time = time.time() - start_time
        
        self.logger.info(f"Scanned {scanned_files} files in {execution_time:.2f}s, found {len(self.violations)} potential secrets")
        
        return self.create_result(execution_time)
    
    async def _scan_file_for_secrets(self, file_path: Path) -> None:
        """Scan a single file for potential secrets"""
        try:
            # Try multiple encodings
            content = self._read_file_safely(file_path)
            if content is None:
                return
            
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # Skip empty lines and comments in some languages
                stripped_line = line.strip()
                if not stripped_line or self._is_comment_line(stripped_line, file_path.suffix):
                    continue
                
                # Check each pattern
                for secret_type, pattern in self.secret_patterns.items():
                    matches = pattern.finditer(line)
                    
                    for match in matches:
                        potential_secret = match.group(1) if match.groups() else match.group(0)
                        
                        # Skip if likely false positive
                        if self._is_likely_false_positive(potential_secret, line):
                            continue
                        
                        # Determine severity based on secret type
                        severity = self._get_severity_for_secret_type(secret_type)
                        
                        self.add_violation(
                            rule_id=f"SECRETS_{secret_type.upper()}",
                            severity=severity,
                            message=f"Potential {secret_type.replace('_', ' ')} detected: {self._mask_secret(potential_secret)}",
                            file_path=file_path,
                            line_number=line_num,
                            column_number=match.start(),
                            suggestion=self._get_remediation_suggestion(secret_type),
                            compliance_frameworks=["OWASP", "NIST", "SOC2"],
                            secret_type=secret_type,
                            confidence=self._calculate_confidence(secret_type, potential_secret, line)
                        )
        
        except Exception as e:
            self.logger.warning(f"Error scanning {file_path}: {e}")
    
    def _read_file_safely(self, file_path: Path) -> str:
        """Safely read file content with multiple encoding attempts"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                self.logger.warning(f"Could not read {file_path}: {e}")
                break
        
        return None
    
    def _is_comment_line(self, line: str, file_extension: str) -> bool:
        """Check if line is likely a comment"""
        comment_prefixes = {
            '.py': '#',
            '.js': '//',
            '.ts': '//',
            '.java': '//',
            '.cs': '//',
            '.go': '//',
            '.rs': '//',
            '.cpp': '//',
            '.c': '//',
            '.php': '//',
            '.rb': '#',
            '.sh': '#',
            '.bash': '#',
            '.yml': '#',
            '.yaml': '#',
            '.properties': '#',
            '.sql': '--',
        }
        
        prefix = comment_prefixes.get(file_extension.lower())
        if prefix and line.startswith(prefix):
            return True
        
        # Check for multi-line comment patterns
        if line.startswith('/*') or line.startswith('<!--') or '*' in line:
            return True
        
        return False
    
    def _is_likely_false_positive(self, potential_secret: str, line: str) -> bool:
        """Check if the detected secret is likely a false positive"""
        # Check against false positive patterns
        for fp_pattern in self.false_positive_patterns:
            if fp_pattern.search(line.lower()) or fp_pattern.search(potential_secret.lower()):
                return True
        
        # Check for common placeholders
        if len(potential_secret) < 8:
            return True
        
        # Check for repeated characters (like "aaaaaaa" or "1111111")
        if len(set(potential_secret)) <= 2:
            return True
        
        # Check for sequential patterns
        if self._has_sequential_pattern(potential_secret):
            return True
        
        return False
    
    def _has_sequential_pattern(self, value: str) -> bool:
        """Check if value has obvious sequential patterns"""
        # Check for alphabetical sequences
        if len(value) >= 4:
            for i in range(len(value) - 3):
                substring = value[i:i+4].lower()
                if substring in 'abcdefghijklmnopqrstuvwxyz':
                    return True
        
        # Check for numerical sequences
        if len(value) >= 3:
            for i in range(len(value) - 2):
                substring = value[i:i+3]
                if substring.isdigit() and substring in '0123456789':
                    return True
        
        return False
    
    def _get_severity_for_secret_type(self, secret_type: str) -> str:
        """Determine severity level based on secret type"""
        high_severity_types = {
            'aws_access_key', 'aws_secret_key', 'private_key_header',
            'github_token', 'stripe_key', 'connection_string'
        }
        
        medium_severity_types = {
            'google_api_key', 'slack_token', 'jwt_token',
            'generic_api_key', 'database_url'
        }
        
        if secret_type in high_severity_types:
            return 'critical'
        elif secret_type in medium_severity_types:
            return 'high'
        else:
            return 'medium'
    
    def _calculate_confidence(self, secret_type: str, secret_value: str, line: str) -> float:
        """Calculate confidence score for secret detection"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for specific patterns
        high_confidence_types = {
            'aws_access_key', 'github_token', 'slack_token',
            'private_key_header', 'stripe_key'
        }
        
        if secret_type in high_confidence_types:
            confidence += 0.4
        
        # Increase confidence based on context
        context_indicators = [
            'key', 'secret', 'token', 'password', 'credential',
            'auth', 'api', 'access', 'private'
        ]
        
        line_lower = line.lower()
        for indicator in context_indicators:
            if indicator in line_lower:
                confidence += 0.1
                break
        
        # Decrease confidence for common false positive contexts
        false_positive_indicators = [
            'example', 'test', 'demo', 'placeholder', 'template',
            'comment', 'todo', 'fixme'
        ]
        
        for indicator in false_positive_indicators:
            if indicator in line_lower:
                confidence -= 0.2
                break
        
        return max(0.0, min(1.0, confidence))
    
    def _mask_secret(self, secret: str) -> str:
        """Mask secret value for safe logging"""
        if len(secret) <= 8:
            return '*' * len(secret)
        
        # Show first 4 and last 4 characters
        return f"{secret[:4]}{'*' * (len(secret) - 8)}{secret[-4:]}"
    
    def _get_remediation_suggestion(self, secret_type: str) -> str:
        """Get remediation suggestion based on secret type"""
        suggestions = {
            'aws_access_key': "Move AWS credentials to environment variables or AWS IAM roles",
            'github_token': "Use GitHub secrets or environment variables for token storage",
            'private_key_header': "Store private keys in secure key management systems",
            'password_assignment': "Use environment variables or secure configuration management",
            'connection_string': "Store database connection strings in environment variables",
            'generic_api_key': "Move API keys to environment variables or secrets management",
            'jwt_token': "Avoid hardcoding JWT tokens; use secure token storage",
            'slack_token': "Store Slack tokens in environment variables or secure vaults"
        }
        
        return suggestions.get(secret_type, 
            "Move sensitive values to environment variables or secure configuration management")
    
    @classmethod
    def get_required_tools(cls) -> List[str]:
        """No external tools required"""
        return []