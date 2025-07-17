# Static Analysis Configuration System

## Configuration Overview

The comprehensive static analysis system uses a hierarchical configuration system that allows for flexible customization while maintaining sensible defaults. Configuration can be specified at multiple levels with clear precedence rules.

## Configuration Hierarchy

### Precedence Order (highest to lowest)
1. **Command-line arguments** - Direct CLI flags and options
2. **Environment variables** - `PYNOMALY_ANALYSIS_*` variables
3. **Explicit config file** - File specified with `--config` flag
4. **Project configuration** - `pyproject.toml` or `.pynomaly.toml` in project root
5. **User configuration** - `~/.pynomaly/config.toml`
6. **Default configuration** - Built-in defaults

## Configuration File Formats

### Primary Configuration (`pyproject.toml`)
```toml
[tool.pynomaly.analysis]
# Global settings
profile = "strict"  # strict, balanced, permissive, custom
python_version = "3.11"
max_workers = 4
enable_caching = true
cache_dir = ".pynomaly_cache"

# File discovery
include_patterns = ["src/**/*.py", "tests/**/*.py"]
exclude_patterns = ["**/__pycache__/**", "**/node_modules/**"]

# Analysis phases
enable_type_checking = true
enable_security_scanning = true
enable_performance_analysis = true
enable_documentation_checking = true

# Reporting
output_format = "console"  # console, json, html, junit
show_progress = true
colored_output = true
show_context = true
max_issues_per_file = 50

[tool.pynomaly.analysis.type_checking]
# Type checking configuration
strict_mode = true
require_type_annotations = true
check_untyped_defs = true
warn_unused_ignores = true
plugins = ["mypy_django", "mypy_drf"]

[tool.pynomaly.analysis.security]
# Security scanning configuration
level = "high"  # low, medium, high
confidence_threshold = 80
ignore_vulnerabilities = ["39462", "39463"]
custom_rules = ["security/custom_rules.yaml"]

[tool.pynomaly.analysis.performance]
# Performance analysis configuration
check_complexity = true
max_complexity = 10
check_memory_usage = true
detect_antipatterns = true

[tool.pynomaly.analysis.documentation]
# Documentation checking configuration
require_docstrings = true
docstring_style = "google"  # google, numpy, sphinx
min_coverage = 80
```

### Tool-Specific Configuration
```toml
[tool.pynomaly.analysis.tools.mypy]
enabled = true
config_file = "mypy.ini"
strict = true
disallow_untyped_defs = true
warn_return_any = true
show_error_codes = true

[tool.pynomaly.analysis.tools.ruff]
enabled = true
line_length = 88
select = ["E", "F", "W", "C", "N", "B", "I"]
ignore = ["E501", "W503"]
per_file_ignores = {"tests/**" = ["B011"]}

[tool.pynomaly.analysis.tools.black]
enabled = true
line_length = 88
skip_string_normalization = false
target_version = ["py311"]

[tool.pynomaly.analysis.tools.bandit]
enabled = true
confidence_level = "medium"
severity_level = "medium"
skip_rules = ["B101", "B601"]

[tool.pynomaly.analysis.tools.safety]
enabled = true
ignore_vulnerabilities = []
include_dev_dependencies = false

[tool.pynomaly.analysis.tools.vulture]
enabled = true
min_confidence = 80
whitelist_files = ["vulture_whitelist.py"]

[tool.pynomaly.analysis.tools.semgrep]
enabled = true
config = "auto"
custom_rules = ["semgrep/custom_rules.yaml"]
```

## Configuration Profiles

### Strict Profile
```python
STRICT_PROFILE = {
    "type_checking": {
        "strict_mode": True,
        "require_type_annotations": True,
        "check_untyped_defs": True,
        "disallow_any_generics": True,
        "disallow_incomplete_defs": True,
        "warn_unused_ignores": True,
        "warn_redundant_casts": True,
        "warn_unreachable": True,
    },
    "security": {
        "level": "high",
        "confidence_threshold": 90,
        "fail_on_vulnerability": True,
    },
    "performance": {
        "check_complexity": True,
        "max_complexity": 8,
        "check_memory_usage": True,
        "detect_antipatterns": True,
    },
    "documentation": {
        "require_docstrings": True,
        "min_coverage": 90,
        "check_examples": True,
    },
    "tools": {
        "mypy": {"strict": True},
        "ruff": {"select": ["ALL"], "ignore": []},
        "bandit": {"confidence_level": "high"},
    }
}
```

### Balanced Profile
```python
BALANCED_PROFILE = {
    "type_checking": {
        "strict_mode": False,
        "require_type_annotations": False,
        "check_untyped_defs": True,
        "warn_unused_ignores": False,
    },
    "security": {
        "level": "medium",
        "confidence_threshold": 70,
        "fail_on_vulnerability": False,
    },
    "performance": {
        "check_complexity": True,
        "max_complexity": 12,
        "check_memory_usage": False,
        "detect_antipatterns": True,
    },
    "documentation": {
        "require_docstrings": False,
        "min_coverage": 60,
        "check_examples": False,
    },
    "tools": {
        "mypy": {"strict": False},
        "ruff": {"select": ["E", "F", "W", "C", "N"], "ignore": ["E501"]},
        "bandit": {"confidence_level": "medium"},
    }
}
```

### Permissive Profile
```python
PERMISSIVE_PROFILE = {
    "type_checking": {
        "strict_mode": False,
        "require_type_annotations": False,
        "check_untyped_defs": False,
        "ignore_errors": True,
    },
    "security": {
        "level": "low",
        "confidence_threshold": 50,
        "fail_on_vulnerability": False,
    },
    "performance": {
        "check_complexity": False,
        "max_complexity": 20,
        "check_memory_usage": False,
        "detect_antipatterns": False,
    },
    "documentation": {
        "require_docstrings": False,
        "min_coverage": 30,
        "check_examples": False,
    },
    "tools": {
        "mypy": {"strict": False, "ignore_errors": True},
        "ruff": {"select": ["E", "F"], "ignore": ["E501", "W503"]},
        "bandit": {"confidence_level": "low"},
    }
}
```

## Configuration Classes

### Main Configuration Class
```python
@dataclass
class AnalysisConfig:
    # Global settings
    profile: str = "balanced"
    python_version: str = "3.11"
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: Path = Path(".pynomaly_cache")
    
    # File discovery
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/__pycache__/**",
        "**/node_modules/**",
        "**/.venv/**",
        "**/venv/**"
    ])
    
    # Analysis phases
    enable_type_checking: bool = True
    enable_security_scanning: bool = True
    enable_performance_analysis: bool = True
    enable_documentation_checking: bool = True
    
    # Reporting
    output_format: str = "console"
    show_progress: bool = True
    colored_output: bool = True
    show_context: bool = True
    max_issues_per_file: int = 50
    
    # Sub-configurations
    type_checking: TypeCheckingConfig = field(default_factory=TypeCheckingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    documentation: DocumentationConfig = field(default_factory=DocumentationConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
```

### Type Checking Configuration
```python
@dataclass
class TypeCheckingConfig:
    strict_mode: bool = False
    require_type_annotations: bool = False
    check_untyped_defs: bool = True
    disallow_any_generics: bool = False
    disallow_incomplete_defs: bool = False
    warn_unused_ignores: bool = False
    warn_redundant_casts: bool = False
    warn_unreachable: bool = False
    plugins: List[str] = field(default_factory=list)
    
    # MyPy specific
    mypy_config_file: Optional[Path] = None
    mypy_cache_dir: Optional[Path] = None
    
    # Pyright specific
    pyright_config_file: Optional[Path] = None
    pyright_type_checking_mode: str = "basic"  # basic, strict
```

### Security Configuration
```python
@dataclass
class SecurityConfig:
    level: str = "medium"  # low, medium, high
    confidence_threshold: int = 70
    fail_on_vulnerability: bool = False
    ignore_vulnerabilities: List[str] = field(default_factory=list)
    custom_rules: List[Path] = field(default_factory=list)
    
    # Bandit specific
    bandit_config_file: Optional[Path] = None
    bandit_skip_rules: List[str] = field(default_factory=list)
    
    # Safety specific
    safety_ignore_ids: List[str] = field(default_factory=list)
    safety_include_dev: bool = False
```

### Performance Configuration
```python
@dataclass
class PerformanceConfig:
    check_complexity: bool = True
    max_complexity: int = 10
    check_memory_usage: bool = False
    detect_antipatterns: bool = True
    profile_imports: bool = False
    
    # Complexity thresholds
    complexity_warning_threshold: int = 8
    complexity_error_threshold: int = 15
    
    # Memory thresholds
    memory_warning_threshold: int = 100  # MB
    memory_error_threshold: int = 500    # MB
```

### Documentation Configuration
```python
@dataclass
class DocumentationConfig:
    require_docstrings: bool = False
    docstring_style: str = "google"  # google, numpy, sphinx
    min_coverage: int = 60
    check_examples: bool = False
    
    # Pydocstyle specific
    pydocstyle_config_file: Optional[Path] = None
    pydocstyle_convention: str = "google"
    
    # Sphinx specific
    sphinx_config_dir: Optional[Path] = None
    check_sphinx_build: bool = False
```

## Configuration Loading

### Configuration Manager
```python
class ConfigManager:
    def __init__(self):
        self.profiles = {
            "strict": STRICT_PROFILE,
            "balanced": BALANCED_PROFILE,
            "permissive": PERMISSIVE_PROFILE,
        }
    
    def load_config(self, config_path: Optional[Path] = None) -> AnalysisConfig:
        """Load configuration from multiple sources."""
        config = AnalysisConfig()
        
        # Apply profile settings
        if config.profile in self.profiles:
            config = self._apply_profile(config, self.profiles[config.profile])
        
        # Load from configuration files
        config = self._load_project_config(config)
        config = self._load_user_config(config)
        
        if config_path:
            config = self._load_explicit_config(config, config_path)
        
        # Apply environment variables
        config = self._apply_env_vars(config)
        
        # Validate configuration
        self._validate_config(config)
        
        return config
    
    def _load_project_config(self, config: AnalysisConfig) -> AnalysisConfig:
        """Load configuration from project files."""
        project_files = [
            Path("pyproject.toml"),
            Path(".pynomaly.toml"),
            Path("setup.cfg")
        ]
        
        for config_file in project_files:
            if config_file.exists():
                return self._merge_config(config, self._parse_config_file(config_file))
        
        return config
    
    def _apply_env_vars(self, config: AnalysisConfig) -> AnalysisConfig:
        """Apply environment variable overrides."""
        env_mappings = {
            "PYNOMALY_ANALYSIS_PROFILE": "profile",
            "PYNOMALY_ANALYSIS_PYTHON_VERSION": "python_version",
            "PYNOMALY_ANALYSIS_MAX_WORKERS": "max_workers",
            "PYNOMALY_ANALYSIS_ENABLE_CACHING": "enable_caching",
            "PYNOMALY_ANALYSIS_OUTPUT_FORMAT": "output_format",
        }
        
        for env_var, config_key in env_mappings.items():
            if env_value := os.getenv(env_var):
                setattr(config, config_key, self._convert_env_value(env_value))
        
        return config
```

### Configuration Validation
```python
class ConfigValidator:
    def validate(self, config: AnalysisConfig) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate profile
        if config.profile not in ["strict", "balanced", "permissive", "custom"]:
            errors.append(f"Invalid profile: {config.profile}")
        
        # Validate Python version
        if not re.match(r"^3\.\d+$", config.python_version):
            errors.append(f"Invalid Python version: {config.python_version}")
        
        # Validate max workers
        if config.max_workers <= 0:
            errors.append(f"Max workers must be positive: {config.max_workers}")
        
        # Validate output format
        valid_formats = ["console", "json", "html", "junit"]
        if config.output_format not in valid_formats:
            errors.append(f"Invalid output format: {config.output_format}")
        
        # Validate paths
        if not config.cache_dir.parent.exists():
            errors.append(f"Cache directory parent doesn't exist: {config.cache_dir}")
        
        # Validate tool configurations
        errors.extend(self._validate_tool_configs(config.tools))
        
        return errors
    
    def _validate_tool_configs(self, tools_config: ToolsConfig) -> List[str]:
        """Validate tool-specific configurations."""
        errors = []
        
        # Validate MyPy config
        if tools_config.mypy.config_file and not tools_config.mypy.config_file.exists():
            errors.append(f"MyPy config file not found: {tools_config.mypy.config_file}")
        
        # Validate Ruff config
        if tools_config.ruff.line_length < 50 or tools_config.ruff.line_length > 200:
            errors.append(f"Ruff line length out of range: {tools_config.ruff.line_length}")
        
        return errors
```

## Environment Variable Support

### Environment Variable Mapping
```python
ENVIRONMENT_VARIABLES = {
    # Global settings
    "PYNOMALY_ANALYSIS_PROFILE": ("profile", str),
    "PYNOMALY_ANALYSIS_PYTHON_VERSION": ("python_version", str),
    "PYNOMALY_ANALYSIS_MAX_WORKERS": ("max_workers", int),
    "PYNOMALY_ANALYSIS_ENABLE_CACHING": ("enable_caching", bool),
    "PYNOMALY_ANALYSIS_CACHE_DIR": ("cache_dir", Path),
    
    # File discovery
    "PYNOMALY_ANALYSIS_INCLUDE_PATTERNS": ("include_patterns", list),
    "PYNOMALY_ANALYSIS_EXCLUDE_PATTERNS": ("exclude_patterns", list),
    
    # Analysis phases
    "PYNOMALY_ANALYSIS_ENABLE_TYPE_CHECKING": ("enable_type_checking", bool),
    "PYNOMALY_ANALYSIS_ENABLE_SECURITY": ("enable_security_scanning", bool),
    "PYNOMALY_ANALYSIS_ENABLE_PERFORMANCE": ("enable_performance_analysis", bool),
    "PYNOMALY_ANALYSIS_ENABLE_DOCS": ("enable_documentation_checking", bool),
    
    # Reporting
    "PYNOMALY_ANALYSIS_OUTPUT_FORMAT": ("output_format", str),
    "PYNOMALY_ANALYSIS_SHOW_PROGRESS": ("show_progress", bool),
    "PYNOMALY_ANALYSIS_COLORED_OUTPUT": ("colored_output", bool),
    
    # Type checking
    "PYNOMALY_ANALYSIS_STRICT_MODE": ("type_checking.strict_mode", bool),
    "PYNOMALY_ANALYSIS_REQUIRE_ANNOTATIONS": ("type_checking.require_type_annotations", bool),
    
    # Security
    "PYNOMALY_ANALYSIS_SECURITY_LEVEL": ("security.level", str),
    "PYNOMALY_ANALYSIS_CONFIDENCE_THRESHOLD": ("security.confidence_threshold", int),
    
    # Performance
    "PYNOMALY_ANALYSIS_MAX_COMPLEXITY": ("performance.max_complexity", int),
    "PYNOMALY_ANALYSIS_CHECK_MEMORY": ("performance.check_memory_usage", bool),
}
```

## Command Line Interface

### CLI Configuration Options
```python
@click.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), 
              help="Path to configuration file")
@click.option("--profile", "-p", type=click.Choice(["strict", "balanced", "permissive"]),
              help="Analysis profile to use")
@click.option("--output-format", "-f", type=click.Choice(["console", "json", "html", "junit"]),
              help="Output format")
@click.option("--max-workers", "-j", type=int, help="Maximum number of parallel workers")
@click.option("--enable-caching/--disable-caching", default=None,
              help="Enable or disable result caching")
@click.option("--cache-dir", type=click.Path(), help="Cache directory")
@click.option("--include", multiple=True, help="Include patterns")
@click.option("--exclude", multiple=True, help="Exclude patterns")
@click.option("--enable-type-checking/--disable-type-checking", default=None,
              help="Enable or disable type checking")
@click.option("--enable-security/--disable-security", default=None,
              help="Enable or disable security scanning")
@click.option("--enable-performance/--disable-performance", default=None,
              help="Enable or disable performance analysis")
@click.option("--enable-docs/--disable-docs", default=None,
              help="Enable or disable documentation checking")
@click.option("--strict", is_flag=True, help="Enable strict mode")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--fix", is_flag=True, help="Automatically fix issues where possible")
@click.option("--show-config", is_flag=True, help="Show effective configuration and exit")
def main(paths, config, profile, output_format, max_workers, enable_caching, 
         cache_dir, include, exclude, enable_type_checking, enable_security,
         enable_performance, enable_docs, strict, quiet, verbose, fix, show_config):
    """Comprehensive static analysis for Python projects."""
    
    # Load configuration
    config_manager = ConfigManager()
    analysis_config = config_manager.load_config(config)
    
    # Apply CLI overrides
    if profile:
        analysis_config.profile = profile
    if output_format:
        analysis_config.output_format = output_format
    if max_workers:
        analysis_config.max_workers = max_workers
    if enable_caching is not None:
        analysis_config.enable_caching = enable_caching
    if cache_dir:
        analysis_config.cache_dir = Path(cache_dir)
    if include:
        analysis_config.include_patterns = list(include)
    if exclude:
        analysis_config.exclude_patterns = list(exclude)
    if enable_type_checking is not None:
        analysis_config.enable_type_checking = enable_type_checking
    if enable_security is not None:
        analysis_config.enable_security_scanning = enable_security
    if enable_performance is not None:
        analysis_config.enable_performance_analysis = enable_performance
    if enable_docs is not None:
        analysis_config.enable_documentation_checking = enable_docs
    if strict:
        analysis_config.type_checking.strict_mode = True
    if quiet:
        analysis_config.show_progress = False
    if verbose:
        analysis_config.show_context = True
    
    # Show configuration if requested
    if show_config:
        print(json.dumps(asdict(analysis_config), indent=2, default=str))
        return
    
    # Run analysis
    orchestrator = AnalysisOrchestrator(analysis_config)
    result = orchestrator.run_analysis(paths or [Path.cwd()])
    
    # Generate report
    reporter = ReporterFactory.create_reporter(analysis_config.output_format)
    reporter.generate_report(result)
```

This comprehensive configuration system provides flexibility for different use cases while maintaining clear defaults and validation to ensure reliable operation.