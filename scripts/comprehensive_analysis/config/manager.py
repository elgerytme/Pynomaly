"""Configuration management for comprehensive static analysis."""

import os
import json
import toml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

from .profiles import ANALYSIS_PROFILES

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Main configuration class for comprehensive static analysis."""
    
    # Global settings
    profile: str = "balanced"
    python_version: str = "3.11"
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".pynomaly_cache"))
    
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
    ])
    
    # Analysis phases
    enable_type_checking: bool = True
    enable_security_analysis: bool = True
    enable_dead_code_detection: bool = True
    enable_performance_analysis: bool = True
    enable_documentation_checking: bool = True
    enable_reference_checking: bool = True
    
    # Reporting
    output_format: str = "console"
    show_progress: bool = True
    colored_output: bool = True
    show_context: bool = True
    max_issues_per_file: int = 50
    
    # Tool-specific configurations
    tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization to ensure Path objects and defaults."""
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manages configuration loading from multiple sources."""
    
    def __init__(self):
        self.profiles = ANALYSIS_PROFILES
    
    def load_config(self, config_path: Optional[Path] = None) -> AnalysisConfig:
        """Load configuration from multiple sources with proper precedence."""
        config = AnalysisConfig()
        
        # Apply profile settings first
        if config.profile in self.profiles:
            config = self._apply_profile(config, self.profiles[config.profile])
        
        # Load from configuration files
        config = self._load_project_config(config)
        config = self._load_user_config(config)
        
        # Apply explicit config file
        if config_path and config_path.exists():
            config = self._load_explicit_config(config, config_path)
        
        # Apply environment variables
        config = self._apply_env_vars(config)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _apply_profile(self, config: AnalysisConfig, profile: Dict[str, Any]) -> AnalysisConfig:
        """Apply profile settings to configuration."""
        for section, settings in profile.items():
            if section == "tools":
                # Merge tool configurations
                config.tool_configs.update(settings)
            else:
                # Apply direct settings
                for key, value in settings.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        return config
    
    def _load_project_config(self, config: AnalysisConfig) -> AnalysisConfig:
        """Load configuration from project files."""
        project_files = [
            Path("pyproject.toml"),
            Path(".pynomaly.toml"),
        ]
        
        for config_file in project_files:
            if config_file.exists():
                try:
                    if config_file.suffix == ".toml":
                        data = toml.load(config_file)
                        if "tool" in data and "pynomaly" in data["tool"] and "analysis" in data["tool"]["pynomaly"]:
                            analysis_config = data["tool"]["pynomaly"]["analysis"]
                            config = self._merge_config(config, analysis_config)
                            logger.info(f"Loaded configuration from {config_file}")
                            break
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_file}: {e}")
        
        return config
    
    def _load_user_config(self, config: AnalysisConfig) -> AnalysisConfig:
        """Load configuration from user config file."""
        user_config_path = Path.home() / ".pynomaly" / "config.toml"
        
        if user_config_path.exists():
            try:
                data = toml.load(user_config_path)
                if "analysis" in data:
                    config = self._merge_config(config, data["analysis"])
                    logger.info(f"Loaded user configuration from {user_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}")
        
        return config
    
    def _load_explicit_config(self, config: AnalysisConfig, config_path: Path) -> AnalysisConfig:
        """Load configuration from explicit config file."""
        try:
            if config_path.suffix == ".toml":
                data = toml.load(config_path)
            elif config_path.suffix == ".json":
                with open(config_path, 'r') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            config = self._merge_config(config, data)
            logger.info(f"Loaded explicit configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load explicit config from {config_path}: {e}")
            raise
        
        return config
    
    def _apply_env_vars(self, config: AnalysisConfig) -> AnalysisConfig:
        """Apply environment variable overrides."""
        env_mappings = {
            "PYNOMALY_ANALYSIS_PROFILE": ("profile", str),
            "PYNOMALY_ANALYSIS_PYTHON_VERSION": ("python_version", str),
            "PYNOMALY_ANALYSIS_MAX_WORKERS": ("max_workers", int),
            "PYNOMALY_ANALYSIS_ENABLE_CACHING": ("enable_caching", bool),
            "PYNOMALY_ANALYSIS_CACHE_DIR": ("cache_dir", Path),
            "PYNOMALY_ANALYSIS_OUTPUT_FORMAT": ("output_format", str),
            "PYNOMALY_ANALYSIS_SHOW_PROGRESS": ("show_progress", bool),
            "PYNOMALY_ANALYSIS_COLORED_OUTPUT": ("colored_output", bool),
        }
        
        for env_var, (config_key, config_type) in env_mappings.items():
            if env_value := os.getenv(env_var):
                try:
                    if config_type == bool:
                        value = env_value.lower() in ("true", "1", "yes", "on")
                    elif config_type == int:
                        value = int(env_value)
                    elif config_type == Path:
                        value = Path(env_value)
                    else:
                        value = env_value
                    
                    setattr(config, config_key, value)
                    logger.debug(f"Applied environment variable {env_var} = {value}")
                except ValueError as e:
                    logger.warning(f"Invalid environment variable {env_var} = {env_value}: {e}")
        
        return config
    
    def _merge_config(self, config: AnalysisConfig, new_config: Dict[str, Any]) -> AnalysisConfig:
        """Merge new configuration settings into existing config."""
        for key, value in new_config.items():
            if key == "tool_configs" or key == "tools":
                # Merge tool configurations
                if isinstance(value, dict):
                    config.tool_configs.update(value)
            elif hasattr(config, key):
                # Set direct configuration values
                if isinstance(value, str) and key.endswith("_dir"):
                    setattr(config, key, Path(value))
                else:
                    setattr(config, key, value)
        
        return config
    
    def _validate_config(self, config: AnalysisConfig) -> None:
        """Validate configuration and raise errors for invalid settings."""
        errors = []
        
        # Validate profile
        if config.profile not in ["strict", "balanced", "permissive", "custom"]:
            errors.append(f"Invalid profile: {config.profile}")
        
        # Validate Python version
        if not config.python_version or not config.python_version.startswith("3."):
            errors.append(f"Invalid Python version: {config.python_version}")
        
        # Validate max workers
        if config.max_workers <= 0:
            errors.append(f"Max workers must be positive: {config.max_workers}")
        
        # Validate output format
        valid_formats = ["console", "json", "html", "junit"]
        if config.output_format not in valid_formats:
            errors.append(f"Invalid output format: {config.output_format}. Must be one of: {valid_formats}")
        
        # Validate cache directory
        try:
            config.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create cache directory {config.cache_dir}: {e}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def get_effective_config_dict(self, config: AnalysisConfig) -> Dict[str, Any]:
        """Get the effective configuration as a dictionary."""
        return {
            "profile": config.profile,
            "python_version": config.python_version,
            "max_workers": config.max_workers,
            "enable_caching": config.enable_caching,
            "cache_dir": str(config.cache_dir),
            "include_patterns": config.include_patterns,
            "exclude_patterns": config.exclude_patterns,
            "enable_type_checking": config.enable_type_checking,
            "enable_security_analysis": config.enable_security_analysis,
            "enable_performance_analysis": config.enable_performance_analysis,
            "enable_documentation_checking": config.enable_documentation_checking,
            "enable_reference_checking": config.enable_reference_checking,
            "output_format": config.output_format,
            "show_progress": config.show_progress,
            "colored_output": config.colored_output,
            "show_context": config.show_context,
            "max_issues_per_file": config.max_issues_per_file,
            "tool_configs": config.tool_configs,
        }