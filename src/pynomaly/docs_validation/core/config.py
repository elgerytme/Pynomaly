"""Configuration management for documentation validation."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
import yaml

logger = structlog.get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for documentation validation."""

    # Core paths
    doc_paths: List[str] = field(default_factory=lambda: ["docs/", "README.md"])
    output_path: str = "docs_validation_report"

    # File patterns and exclusions
    file_patterns: List[str] = field(default_factory=lambda: ["*.md", "*.rst", "*.txt"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        ".git/", "__pycache__/", ".pytest_cache/", "node_modules/", ".venv/", "venv/"
    ])

    # Validation toggles
    check_content: bool = True
    check_structure: bool = True
    check_links: bool = True
    check_consistency: bool = True

    # Content validation rules
    required_sections: List[str] = field(default_factory=lambda: [
        "Introduction", "Installation", "Usage", "API Reference"
    ])
    max_line_length: int = 100
    require_code_blocks_language: bool = True

    # Structure validation rules
    require_readme: bool = True
    require_changelog: bool = True
    require_contributing: bool = True
    max_heading_depth: int = 6

    # Link validation rules
    check_external_links: bool = True
    check_internal_links: bool = True
    link_timeout_seconds: int = 10

    # Consistency validation rules
    enforce_heading_style: Optional[str] = "atx"  # atx (#) or setext (===)
    enforce_list_style: Optional[str] = "dash"    # dash (-) or asterisk (*)

    # Report settings
    report_formats: List[str] = field(default_factory=lambda: ["console", "json"])
    detailed_errors: bool = True

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'ValidationConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}

            logger.info(f"Loaded configuration from {config_path}")
            return cls(**config_data)

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    @classmethod
    def from_pyproject_toml(cls, pyproject_path: Union[str, Path] = "pyproject.toml") -> 'ValidationConfig':
        """Load configuration from pyproject.toml file."""
        pyproject_path = Path(pyproject_path)

        if not pyproject_path.exists():
            logger.warning(f"pyproject.toml not found: {pyproject_path}, using defaults")
            return cls()

        try:
            import tomllib

            import 'tomllib'
            import __builtins__
            import else
            import hasattr
            import if
            import tomli

            with open(pyproject_path, 'rb') as f:
                pyproject_data = tomllib.load(f) if hasattr(__builtins__, 'tomllib') else tomli.load(f)

            # Extract docs_validation configuration
            docs_config = pyproject_data.get('tool', {}).get('docs_validation', {})

            if docs_config:
                logger.info(f"Loaded docs validation config from {pyproject_path}")
                return cls(**docs_config)
            else:
                logger.info(f"No docs_validation config found in {pyproject_path}, using defaults")
                return cls()

        except ImportError:
            logger.warning("tomllib/tomli not available, cannot parse pyproject.toml")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load pyproject.toml configuration: {e}")
            raise

    @classmethod
    def from_environment(cls) -> 'ValidationConfig':
        """Load configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        if doc_paths := os.getenv('DOCS_VALIDATION_PATHS'):
            config.doc_paths = [p.strip() for p in doc_paths.split(',')]

        if output_path := os.getenv('DOCS_VALIDATION_OUTPUT'):
            config.output_path = output_path

        if check_links := os.getenv('DOCS_VALIDATION_CHECK_LINKS'):
            config.check_links = check_links.lower() in ('true', '1', 'yes')

        if max_line_length := os.getenv('DOCS_VALIDATION_MAX_LINE_LENGTH'):
            try:
                config.max_line_length = int(max_line_length)
            except ValueError:
                logger.warning(f"Invalid max line length: {max_line_length}")

        logger.info("Configuration loaded from environment variables")
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'doc_paths': self.doc_paths,
            'output_path': self.output_path,
            'file_patterns': self.file_patterns,
            'exclude_patterns': self.exclude_patterns,
            'check_content': self.check_content,
            'check_structure': self.check_structure,
            'check_links': self.check_links,
            'check_consistency': self.check_consistency,
            'required_sections': self.required_sections,
            'max_line_length': self.max_line_length,
            'require_code_blocks_language': self.require_code_blocks_language,
            'require_readme': self.require_readme,
            'require_changelog': self.require_changelog,
            'require_contributing': self.require_contributing,
            'max_heading_depth': self.max_heading_depth,
            'check_external_links': self.check_external_links,
            'check_internal_links': self.check_internal_links,
            'link_timeout_seconds': self.link_timeout_seconds,
            'enforce_heading_style': self.enforce_heading_style,
            'enforce_list_style': self.enforce_list_style,
            'report_formats': self.report_formats,
            'detailed_errors': self.detailed_errors,
        }

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=True)

        logger.info(f"Configuration saved to {config_path}")

    def validate_config(self) -> None:
        """Validate configuration values."""
        errors = []

        if not self.doc_paths:
            errors.append("doc_paths cannot be empty")

        if self.max_line_length <= 0:
            errors.append("max_line_length must be positive")

        if self.max_heading_depth <= 0 or self.max_heading_depth > 6:
            errors.append("max_heading_depth must be between 1 and 6")

        if self.link_timeout_seconds <= 0:
            errors.append("link_timeout_seconds must be positive")

        if self.enforce_heading_style and self.enforce_heading_style not in ('atx', 'setext'):
            errors.append("enforce_heading_style must be 'atx' or 'setext'")

        if self.enforce_list_style and self.enforce_list_style not in ('dash', 'asterisk'):
            errors.append("enforce_list_style must be 'dash' or 'asterisk'")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
