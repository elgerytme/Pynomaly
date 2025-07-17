"""
Configuration management for repository governance.
"""

import json
import toml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckerConfig:
    """Configuration for a specific checker."""
    enabled: bool = True
    severity_override: Optional[str] = None
    custom_rules: Dict[str, Any] = field(default_factory=dict)
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    max_violations: Optional[int] = None
    fail_on_violation: bool = False


@dataclass
class FixerConfig:
    """Configuration for a specific fixer."""
    enabled: bool = True
    auto_fix: bool = True
    dry_run: bool = False
    create_backup: bool = True
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    max_fixes: Optional[int] = None


@dataclass
class ReportingConfig:
    """Configuration for reporting."""
    formats: List[str] = field(default_factory=lambda: ["console"])
    output_directory: Optional[Path] = None
    include_charts: bool = True
    include_details: bool = True
    create_github_issues: bool = False
    github_labels: List[str] = field(default_factory=lambda: ["governance"])


@dataclass
class GovernanceConfig:
    """Main configuration for repository governance."""
    
    # General settings
    enabled: bool = True
    dry_run: bool = False
    fail_on_violations: bool = False
    max_violations: Optional[int] = None
    
    # File discovery
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/__pycache__/**",
        "**/node_modules/**",
        "**/.venv/**",
        "**/venv/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**",
        "**/*.pyc",
        "**/.pytest_cache/**",
        "**/.mypy_cache/**",
        "**/.ruff_cache/**"
    ])
    
    # Checker configurations
    checkers: Dict[str, CheckerConfig] = field(default_factory=dict)
    
    # Fixer configurations
    fixers: Dict[str, FixerConfig] = field(default_factory=dict)
    
    # Reporting configuration
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    
    # CI/CD integration
    ci_enabled: bool = False
    ci_fail_on_violations: bool = True
    ci_create_pr_comments: bool = True
    ci_update_status_checks: bool = True
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_cron: str = "0 2 * * *"  # Daily at 2 AM
    schedule_on_push: bool = True
    schedule_on_pr: bool = True
    
    def __post_init__(self):
        """Post-initialization to ensure proper types."""
        if isinstance(self.reporting.output_directory, str):
            self.reporting.output_directory = Path(self.reporting.output_directory)
        
        # Initialize default checker configurations
        default_checkers = [
            "TidinessChecker",
            "DomainLeakageChecker", 
            "ArchitectureChecker"
        ]
        
        for checker_name in default_checkers:
            if checker_name not in self.checkers:
                self.checkers[checker_name] = CheckerConfig()
        
        # Initialize default fixer configurations
        default_fixers = [
            "BackupFileFixer",
            "DomainLeakageFixer",
            "StructureFixer"
        ]
        
        for fixer_name in default_fixers:
            if fixer_name not in self.fixers:
                self.fixers[fixer_name] = FixerConfig()


class GovernanceConfigManager:
    """Manages loading and saving of governance configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the config manager."""
        self.config_path = config_path or Path("governance.toml")
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_config(self) -> GovernanceConfig:
        """Load configuration from file."""
        if not self.config_path.exists():
            self.logger.info("No configuration file found, using defaults")
            return GovernanceConfig()
        
        try:
            if self.config_path.suffix == ".toml":
                return self._load_toml_config()
            elif self.config_path.suffix == ".json":
                return self._load_json_config()
            else:
                raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            self.logger.info("Using default configuration")
            return GovernanceConfig()
    
    def save_config(self, config: GovernanceConfig) -> bool:
        """Save configuration to file."""
        try:
            if self.config_path.suffix == ".toml":
                return self._save_toml_config(config)
            elif self.config_path.suffix == ".json":
                return self._save_json_config(config)
            else:
                raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
        except Exception as e:
            self.logger.error(f"Failed to save config to {self.config_path}: {e}")
            return False
    
    def _load_toml_config(self) -> GovernanceConfig:
        """Load configuration from TOML file."""
        data = toml.load(self.config_path)
        return self._dict_to_config(data)
    
    def _load_json_config(self) -> GovernanceConfig:
        """Load configuration from JSON file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self._dict_to_config(data)
    
    def _save_toml_config(self, config: GovernanceConfig) -> bool:
        """Save configuration to TOML file."""
        data = self._config_to_dict(config)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            toml.dump(data, f)
        return True
    
    def _save_json_config(self, config: GovernanceConfig) -> bool:
        """Save configuration to JSON file."""
        data = self._config_to_dict(config)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    
    def _dict_to_config(self, data: Dict[str, Any]) -> GovernanceConfig:
        """Convert dictionary to GovernanceConfig."""
        # Handle checkers
        checkers = {}
        if "checkers" in data:
            for name, checker_data in data["checkers"].items():
                checkers[name] = CheckerConfig(**checker_data)
        
        # Handle fixers
        fixers = {}
        if "fixers" in data:
            for name, fixer_data in data["fixers"].items():
                fixers[name] = FixerConfig(**fixer_data)
        
        # Handle reporting
        reporting = ReportingConfig()
        if "reporting" in data:
            reporting_data = data["reporting"]
            if "output_directory" in reporting_data:
                reporting_data["output_directory"] = Path(reporting_data["output_directory"])
            reporting = ReportingConfig(**reporting_data)
        
        # Create main config
        config_data = {k: v for k, v in data.items() if k not in ["checkers", "fixers", "reporting"]}
        config_data["checkers"] = checkers
        config_data["fixers"] = fixers
        config_data["reporting"] = reporting
        
        return GovernanceConfig(**config_data)
    
    def _config_to_dict(self, config: GovernanceConfig) -> Dict[str, Any]:
        """Convert GovernanceConfig to dictionary."""
        data = {
            "enabled": config.enabled,
            "dry_run": config.dry_run,
            "fail_on_violations": config.fail_on_violations,
            "max_violations": config.max_violations,
            "include_patterns": config.include_patterns,
            "exclude_patterns": config.exclude_patterns,
            "ci_enabled": config.ci_enabled,
            "ci_fail_on_violations": config.ci_fail_on_violations,
            "ci_create_pr_comments": config.ci_create_pr_comments,
            "ci_update_status_checks": config.ci_update_status_checks,
            "schedule_enabled": config.schedule_enabled,
            "schedule_cron": config.schedule_cron,
            "schedule_on_push": config.schedule_on_push,
            "schedule_on_pr": config.schedule_on_pr,
        }
        
        # Convert checkers
        data["checkers"] = {}
        for name, checker_config in config.checkers.items():
            data["checkers"][name] = {
                "enabled": checker_config.enabled,
                "severity_override": checker_config.severity_override,
                "custom_rules": checker_config.custom_rules,
                "exclude_patterns": checker_config.exclude_patterns,
                "include_patterns": checker_config.include_patterns,
                "max_violations": checker_config.max_violations,
                "fail_on_violation": checker_config.fail_on_violation,
            }
        
        # Convert fixers
        data["fixers"] = {}
        for name, fixer_config in config.fixers.items():
            data["fixers"][name] = {
                "enabled": fixer_config.enabled,
                "auto_fix": fixer_config.auto_fix,
                "dry_run": fixer_config.dry_run,
                "create_backup": fixer_config.create_backup,
                "exclude_patterns": fixer_config.exclude_patterns,
                "include_patterns": fixer_config.include_patterns,
                "max_fixes": fixer_config.max_fixes,
            }
        
        # Convert reporting
        data["reporting"] = {
            "formats": config.reporting.formats,
            "output_directory": str(config.reporting.output_directory) if config.reporting.output_directory else None,
            "include_charts": config.reporting.include_charts,
            "include_details": config.reporting.include_details,
            "create_github_issues": config.reporting.create_github_issues,
            "github_labels": config.reporting.github_labels,
        }
        
        return data
    
    def create_default_config(self) -> GovernanceConfig:
        """Create a default configuration with recommended settings."""
        config = GovernanceConfig()
        
        # Configure checkers with sensible defaults
        config.checkers["TidinessChecker"] = CheckerConfig(
            enabled=True,
            fail_on_violation=False,
            custom_rules={
                "max_backup_files": 50,
                "check_temp_files": True,
                "check_build_artifacts": True
            }
        )
        
        config.checkers["DomainLeakageChecker"] = CheckerConfig(
            enabled=True,
            fail_on_violation=True,
            custom_rules={
                "max_monorepo_imports": 10,
                "check_circular_dependencies": True,
                "check_cross_domain_dependencies": True
            }
        )
        
        config.checkers["ArchitectureChecker"] = CheckerConfig(
            enabled=True,
            fail_on_violation=False,
            custom_rules={
                "check_layer_violations": True,
                "check_solid_principles": True,
                "check_design_patterns": True
            }
        )
        
        # Configure fixers
        config.fixers["BackupFileFixer"] = FixerConfig(
            enabled=True,
            auto_fix=True,
            dry_run=False
        )
        
        config.fixers["DomainLeakageFixer"] = FixerConfig(
            enabled=True,
            auto_fix=False,  # Domain leakage fixes require manual review
            dry_run=True
        )
        
        config.fixers["StructureFixer"] = FixerConfig(
            enabled=True,
            auto_fix=True,
            dry_run=False
        )
        
        # Configure reporting
        config.reporting = ReportingConfig(
            formats=["console", "markdown", "json"],
            output_directory=Path("reports"),
            include_charts=True,
            include_details=True,
            create_github_issues=False
        )
        
        # CI/CD settings
        config.ci_enabled = True
        config.ci_fail_on_violations = True
        config.ci_create_pr_comments = True
        
        return config
    
    def validate_config(self, config: GovernanceConfig) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate reporting formats
        valid_formats = ["console", "json", "markdown", "html", "github"]
        for fmt in config.reporting.formats:
            if fmt not in valid_formats:
                errors.append(f"Invalid reporting format: {fmt}")
        
        # Validate output directory
        if config.reporting.output_directory and not config.reporting.output_directory.parent.exists():
            errors.append(f"Output directory parent does not exist: {config.reporting.output_directory.parent}")
        
        # Validate checker configurations
        for checker_name, checker_config in config.checkers.items():
            if checker_config.severity_override and checker_config.severity_override not in ["high", "medium", "low", "info"]:
                errors.append(f"Invalid severity override for {checker_name}: {checker_config.severity_override}")
        
        # Validate schedule cron format (basic validation)
        if config.schedule_enabled and not config.schedule_cron:
            errors.append("Schedule cron expression is required when scheduling is enabled")
        
        return errors