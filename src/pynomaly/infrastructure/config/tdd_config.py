"""Test-Driven Development (TDD) configuration and settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator


@dataclass
class TestRequirement:
    """Represents a test requirement that must be satisfied before implementation."""
    
    id: str
    module_path: str
    function_name: str
    description: str
    test_specification: str
    created_at: str
    status: str = "pending"  # pending, implemented, validated
    implementation_path: Optional[str] = None
    test_file_path: Optional[str] = None
    coverage_target: float = 0.8
    tags: Set[str] = field(default_factory=set)


@dataclass
class TDDComplianceReport:
    """Report on TDD compliance status."""
    
    total_requirements: int
    pending_requirements: int
    implemented_requirements: int
    validated_requirements: int
    overall_compliance: float
    module_compliance: Dict[str, float]
    violations: List[str]
    coverage_report: Dict[str, float]
    last_updated: str


class TDDSettings(BaseModel):
    """TDD configuration settings."""
    
    # Core TDD settings
    enabled: bool = Field(default=False, description="Enable TDD enforcement")
    strict_mode: bool = Field(default=False, description="Enforce strict TDD rules")
    auto_validation: bool = Field(default=True, description="Automatically validate TDD compliance")
    
    # Coverage settings
    min_test_coverage: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum required test coverage")
    coverage_fail_under: float = Field(default=0.7, ge=0.0, le=1.0, description="Coverage threshold for failing builds")
    branch_coverage_required: bool = Field(default=True, description="Require branch coverage analysis")
    
    # File patterns
    test_file_patterns: List[str] = Field(
        default_factory=lambda: ["test_*.py", "*_test.py", "tests/*.py"],
        description="Patterns for test files"
    )
    implementation_patterns: List[str] = Field(
        default_factory=lambda: ["*.py"],
        description="Patterns for implementation files"
    )
    exemption_patterns: List[str] = Field(
        default_factory=lambda: ["__init__.py", "setup.py", "conftest.py", "migrations/*"],
        description="File patterns exempt from TDD requirements"
    )
    
    # Module enforcement
    enforce_on_modules: List[str] = Field(
        default_factory=list,
        description="Specific modules to enforce TDD on (empty means all)"
    )
    enforce_on_packages: List[str] = Field(
        default_factory=lambda: ["src/pynomaly/domain", "src/pynomaly/application"],
        description="Packages to enforce TDD on"
    )
    
    # Test naming and organization
    test_naming_convention: str = Field(
        default="test_{function_name}",
        description="Test function naming convention"
    )
    test_class_naming_convention: str = Field(
        default="Test{ClassName}",
        description="Test class naming convention"
    )
    require_test_docstrings: bool = Field(default=True, description="Require docstrings in test functions")
    
    # Git integration
    git_hooks_enabled: bool = Field(default=True, description="Enable git hook integration")
    pre_commit_validation: bool = Field(default=True, description="Validate TDD compliance on pre-commit")
    pre_push_validation: bool = Field(default=False, description="Validate TDD compliance on pre-push")
    
    # CI/CD integration
    ci_validation_enabled: bool = Field(default=True, description="Enable CI/CD TDD validation")
    fail_on_violations: bool = Field(default=False, description="Fail builds on TDD violations")
    
    # Reporting and metrics
    generate_compliance_reports: bool = Field(default=True, description="Generate TDD compliance reports")
    report_output_path: str = Field(default="./reports/tdd", description="Path for TDD reports")
    metrics_retention_days: int = Field(default=90, description="Days to retain TDD metrics")
    
    # Development workflow
    allow_implementation_first: bool = Field(default=False, description="Allow implementation before tests in development")
    grace_period_hours: int = Field(default=24, description="Grace period for TDD compliance in development")
    require_test_plan: bool = Field(default=True, description="Require test plan before implementation")
    
    # Advanced settings
    mutation_testing_enabled: bool = Field(default=False, description="Enable mutation testing for test quality")
    property_testing_enabled: bool = Field(default=False, description="Enable property-based testing requirements")
    integration_test_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Required ratio of integration tests")
    
    @field_validator("test_naming_convention", "test_class_naming_convention")
    @classmethod
    def validate_naming_convention(cls, v: str) -> str:
        """Validate naming convention contains required placeholders."""
        if "test_" not in v.lower() and "{" not in v:
            raise ValueError("Naming convention must contain 'test_' or use placeholders like {function_name}")
        return v
    
    @field_validator("enforce_on_packages")
    @classmethod
    def validate_package_paths(cls, v: List[str]) -> List[str]:
        """Validate package paths are reasonable."""
        valid_packages = []
        for package in v:
            if not package.startswith("src/") and not package.startswith("tests/"):
                raise ValueError(f"Package path '{package}' should start with 'src/' or 'tests/'")
            valid_packages.append(package)
        return valid_packages


class TDDRuleEngine:
    """Engine for evaluating TDD rules and compliance."""
    
    def __init__(self, settings: TDDSettings):
        self.settings = settings
        
    def is_file_exempt(self, file_path: Path) -> bool:
        """Check if a file is exempt from TDD requirements."""
        file_str = str(file_path)
        for pattern in self.settings.exemption_patterns:
            if Path(file_str).match(pattern):
                return True
        return False
    
    def is_test_file(self, file_path: Path) -> bool:
        """Check if a file is a test file."""
        file_str = str(file_path)
        for pattern in self.settings.test_file_patterns:
            if Path(file_str).match(pattern):
                return True
        return False
    
    def is_implementation_file(self, file_path: Path) -> bool:
        """Check if a file is an implementation file."""
        if self.is_test_file(file_path) or self.is_file_exempt(file_path):
            return False
            
        file_str = str(file_path)
        for pattern in self.settings.implementation_patterns:
            if Path(file_str).match(pattern):
                return True
        return False
    
    def should_enforce_tdd(self, file_path: Path) -> bool:
        """Determine if TDD should be enforced for a given file."""
        if not self.settings.enabled:
            return False
            
        if self.is_file_exempt(file_path):
            return False
            
        file_str = str(file_path)
        
        # Check specific modules
        if self.settings.enforce_on_modules:
            for module in self.settings.enforce_on_modules:
                if module in file_str:
                    return True
            return False
        
        # Check packages
        for package in self.settings.enforce_on_packages:
            if file_str.startswith(package):
                return True
                
        return False
    
    def get_expected_test_file(self, implementation_file: Path) -> Path:
        """Get the expected test file path for an implementation file."""
        # Convert implementation path to test path
        file_str = str(implementation_file)
        
        # Remove .py extension
        base_name = file_str.replace(".py", "")
        
        # Convert src/pynomaly/... to tests/...
        if "src/pynomaly/" in base_name:
            test_base = base_name.replace("src/pynomaly/", "tests/")
        else:
            test_base = f"tests/{base_name}"
        
        return Path(f"{test_base}_test.py")
    
    def validate_test_naming(self, test_function_name: str, implementation_function: str) -> bool:
        """Validate test function follows naming convention."""
        expected_name = self.settings.test_naming_convention.format(
            function_name=implementation_function
        )
        return test_function_name == expected_name
    
    def calculate_compliance_score(self, report: TDDComplianceReport) -> float:
        """Calculate overall TDD compliance score."""
        if report.total_requirements == 0:
            return 1.0
            
        # Weight different compliance factors
        implementation_score = report.implemented_requirements / report.total_requirements
        validation_score = report.validated_requirements / report.total_requirements
        coverage_score = sum(report.coverage_report.values()) / len(report.coverage_report) if report.coverage_report else 0
        
        # Weighted average
        compliance_score = (
            implementation_score * 0.4 +
            validation_score * 0.4 +
            coverage_score * 0.2
        )
        
        return min(compliance_score, 1.0)


@dataclass
class TDDViolation:
    """Represents a TDD rule violation."""
    
    violation_type: str  # missing_test, low_coverage, naming_violation, etc.
    file_path: str
    line_number: Optional[int]
    description: str
    severity: str  # error, warning, info
    rule_name: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False


class TDDConfigManager:
    """Manager for TDD configuration and settings."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("./tdd_config.json")
        self._settings: Optional[TDDSettings] = None
        self._rule_engine: Optional[TDDRuleEngine] = None
    
    @property
    def settings(self) -> TDDSettings:
        """Get TDD settings, loading from file if needed."""
        if self._settings is None:
            self._settings = self.load_settings()
        return self._settings
    
    @property
    def rule_engine(self) -> TDDRuleEngine:
        """Get TDD rule engine."""
        if self._rule_engine is None:
            self._rule_engine = TDDRuleEngine(self.settings)
        return self._rule_engine
    
    def load_settings(self) -> TDDSettings:
        """Load TDD settings from configuration file."""
        if self.config_path.exists():
            import json
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            return TDDSettings(**config_data)
        else:
            return TDDSettings()
    
    def save_settings(self, settings: TDDSettings) -> None:
        """Save TDD settings to configuration file."""
        import json
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(settings.model_dump(), f, indent=2)
        self._settings = settings
        self._rule_engine = None  # Reset rule engine to use new settings
    
    def enable_tdd(self, strict_mode: bool = False) -> None:
        """Enable TDD enforcement."""
        settings = self.settings.model_copy()
        settings.enabled = True
        settings.strict_mode = strict_mode
        self.save_settings(settings)
    
    def disable_tdd(self) -> None:
        """Disable TDD enforcement."""
        settings = self.settings.model_copy()
        settings.enabled = False
        self.save_settings(settings)
    
    def add_exemption(self, pattern: str) -> None:
        """Add a file pattern exemption."""
        settings = self.settings.model_copy()
        if pattern not in settings.exemption_patterns:
            settings.exemption_patterns.append(pattern)
            self.save_settings(settings)
    
    def remove_exemption(self, pattern: str) -> None:
        """Remove a file pattern exemption."""
        settings = self.settings.model_copy()
        if pattern in settings.exemption_patterns:
            settings.exemption_patterns.remove(pattern)
            self.save_settings(settings)
    
    def update_coverage_threshold(self, threshold: float) -> None:
        """Update minimum coverage threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Coverage threshold must be between 0 and 1")
        
        settings = self.settings.model_copy()
        settings.min_test_coverage = threshold
        self.save_settings(settings)
    
    def reset_to_defaults(self) -> None:
        """Reset TDD configuration to defaults."""
        default_settings = TDDSettings()
        self.save_settings(default_settings)


# Global TDD configuration manager
_tdd_config_manager: Optional[TDDConfigManager] = None


def get_tdd_config() -> TDDConfigManager:
    """Get global TDD configuration manager."""
    global _tdd_config_manager
    if _tdd_config_manager is None:
        _tdd_config_manager = TDDConfigManager()
    return _tdd_config_manager