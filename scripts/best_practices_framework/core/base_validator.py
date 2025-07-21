#!/usr/bin/env python3
"""
Base Validator Class
====================
Abstract base class for all best practices validators
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import logging


@dataclass
class RuleViolation:
    """Represents a single rule violation"""
    rule_id: str
    category: str  # architecture, security, testing, devops, sre
    severity: str  # critical, high, medium, low, info
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    suggestion: Optional[str] = None
    remediation_effort: str = "medium"  # low, medium, high
    compliance_frameworks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of running a validator"""
    validator_name: str
    category: str
    violations: List[RuleViolation]
    score: float  # 0-100 compliance score
    execution_time: float  # seconds
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_compliant(self) -> bool:
        """Check if result is fully compliant (no critical/high violations)"""
        return not any(v.severity in ['critical', 'high'] for v in self.violations)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'critical')
    
    @property
    def high_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'high')
    
    @property
    def medium_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'medium')
    
    @property
    def low_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == 'low')


class BaseValidator(ABC):
    """
    Abstract base class for all best practices validators.
    
    All validators must inherit from this class and implement the validate() method.
    """
    
    def __init__(self, config: Dict[str, Any], project_root: Path):
        self.config = config
        self.project_root = project_root
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.violations: List[RuleViolation] = []
        
        # Load validator-specific configuration
        self.validator_config = config.get(self.get_category(), {}).get(self.get_name(), {})
        self.enabled = self.validator_config.get('enabled', True)
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this validator"""
        pass
    
    @abstractmethod
    def get_category(self) -> str:
        """Return the category this validator belongs to"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of what this validator checks"""
        pass
    
    @abstractmethod
    async def validate(self) -> ValidationResult:
        """
        Run validation and return results.
        
        This is the main method that subclasses must implement.
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if this validator is enabled in configuration"""
        return self.enabled
    
    def add_violation(
        self,
        rule_id: str,
        severity: str,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        line_number: Optional[int] = None,
        column_number: Optional[int] = None,
        suggestion: Optional[str] = None,
        remediation_effort: str = "medium",
        compliance_frameworks: Optional[List[str]] = None,
        **metadata
    ) -> None:
        """Add a violation to the current validation run"""
        violation = RuleViolation(
            rule_id=rule_id,
            category=self.get_category(),
            severity=severity,
            message=message,
            file_path=str(file_path) if file_path else None,
            line_number=line_number,
            column_number=column_number,
            suggestion=suggestion,
            remediation_effort=remediation_effort,
            compliance_frameworks=compliance_frameworks or [],
            metadata=metadata
        )
        self.violations.append(violation)
        
        # Log the violation
        log_level = {
            'critical': logging.CRITICAL,
            'high': logging.ERROR,
            'medium': logging.WARNING,
            'low': logging.INFO,
            'info': logging.DEBUG
        }.get(severity, logging.INFO)
        
        self.logger.log(log_level, f"{rule_id}: {message}")
    
    def calculate_score(self) -> float:
        """
        Calculate compliance score based on violations.
        
        Scoring algorithm:
        - Start with 100 points
        - Deduct points based on violation severity:
          - Critical: -20 points
          - High: -15 points
          - Medium: -10 points
          - Low: -5 points
          - Info: -1 point
        - Minimum score is 0
        """
        if not self.violations:
            return 100.0
        
        severity_penalties = {
            'critical': 20,
            'high': 15,
            'medium': 10,
            'low': 5,
            'info': 1
        }
        
        total_penalty = sum(
            severity_penalties.get(violation.severity, 0)
            for violation in self.violations
        )
        
        return max(0.0, 100.0 - total_penalty)
    
    def create_result(self, execution_time: float) -> ValidationResult:
        """Create a ValidationResult from current violations"""
        return ValidationResult(
            validator_name=self.get_name(),
            category=self.get_category(),
            violations=self.violations.copy(),
            score=self.calculate_score(),
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def get_severity_threshold(self, rule_id: str) -> str:
        """Get severity threshold for a specific rule from config"""
        rule_config = self.validator_config.get('rules', {}).get(rule_id, {})
        return rule_config.get('severity', 'medium')
    
    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a specific rule is enabled"""
        rule_config = self.validator_config.get('rules', {}).get(rule_id, {})
        return rule_config.get('enabled', True)
    
    def get_rule_config(self, rule_id: str, key: str, default: Any = None) -> Any:
        """Get configuration value for a specific rule"""
        rule_config = self.validator_config.get('rules', {}).get(rule_id, {})
        return rule_config.get(key, default)
    
    def reset_violations(self) -> None:
        """Clear all violations for a new validation run"""
        self.violations.clear()
    
    @classmethod
    def get_supported_file_types(cls) -> List[str]:
        """Return list of file extensions this validator can analyze"""
        return ['*']  # Default: all file types
    
    @classmethod
    def get_required_tools(cls) -> List[str]:
        """Return list of external tools required by this validator"""
        return []
    
    def should_analyze_file(self, file_path: Path) -> bool:
        """
        Determine if this validator should analyze the given file.
        
        Override this method to implement file filtering logic.
        """
        supported_types = self.get_supported_file_types()
        if '*' in supported_types:
            return True
        
        file_extension = file_path.suffix.lower()
        return file_extension in supported_types
    
    def get_project_files(self, file_patterns: Optional[List[str]] = None) -> List[Path]:
        """
        Get list of project files to analyze.
        
        Args:
            file_patterns: Optional list of glob patterns to filter files
        """
        if not file_patterns:
            file_patterns = ['**/*']
        
        files = []
        for pattern in file_patterns:
            files.extend(self.project_root.glob(pattern))
        
        # Filter to files only and files this validator should analyze
        return [
            f for f in files 
            if f.is_file() 
            and self.should_analyze_file(f)
            and not self._should_exclude_file(f)
        ]
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from analysis"""
        exclude_patterns = [
            '.git/*',
            'node_modules/*',
            '__pycache__/*',
            '*.pyc',
            '.venv/*',
            'venv/*',
            '.env',
            'dist/*',
            'build/*',
            'target/*',
            '.idea/*',
            '.vscode/*'
        ]
        
        # Add validator-specific exclusions
        validator_excludes = self.validator_config.get('exclude_patterns', [])
        exclude_patterns.extend(validator_excludes)
        
        for pattern in exclude_patterns:
            if file_path.match(pattern):
                return True
        
        return False