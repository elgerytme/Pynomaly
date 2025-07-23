"""
Comprehensive tests for BoundaryAnalyzer domain service.

Tests domain boundary violation detection, analysis logic,
and cross-domain import validation functionality.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Set

from test_utilities.factories import TestDataFactory
from test_utilities.fixtures import async_test

from core.domain.services.analyzer import (
    BoundaryAnalyzer,
    Violation, 
    ViolationType,
    Severity,
    AnalysisResult
)
from core.domain.services.scanner import Import, StringReference, ScanResult
from core.domain.services.registry import DomainRegistry, Domain, BoundaryException


class TestViolation:
    """Test Violation entity functionality."""
    
    def test_violation_creation(self):
        """Test creating a violation with all fields."""
        violation = Violation(
            type=ViolationType.CROSS_DOMAIN_IMPORT,
            severity=Severity.CRITICAL,
            from_package="ai/mlops",
            to_package="finance/billing",
            from_domain="ai",
            to_domain="finance",
            file_path="/path/to/file.py",
            line_number=42,
            import_statement="from finance.billing import service",
            description="Cross-domain import violation",
            suggestion="Use API calls instead"
        )
        
        assert violation.type == ViolationType.CROSS_DOMAIN_IMPORT
        assert violation.severity == Severity.CRITICAL
        assert violation.from_package == "ai/mlops"
        assert violation.to_package == "finance/billing"
        assert violation.from_domain == "ai"
        assert violation.to_domain == "finance"
        assert violation.file_path == "/path/to/file.py"
        assert violation.line_number == 42
        assert violation.import_statement == "from finance.billing import service"
        assert violation.description == "Cross-domain import violation"
        assert violation.suggestion == "Use API calls instead"
    
    def test_violation_is_exempted_without_exception(self):
        """Test that violation without exception is not exempted."""
        violation = Violation(
            type=ViolationType.CROSS_DOMAIN_IMPORT,
            severity=Severity.CRITICAL,
            from_package="ai/mlops",
            to_package="finance/billing",
            from_domain="ai",
            to_domain="finance",
            file_path="/path/to/file.py",
            line_number=42,
            import_statement="from finance.billing import service",
            description="Cross-domain import violation"
        )
        
        assert not violation.is_exempted()
    
    def test_violation_is_exempted_with_valid_exception(self):
        """Test that violation with valid exception is exempted."""
        mock_exception = Mock(spec=BoundaryException)
        mock_exception.is_valid.return_value = True
        
        violation = Violation(
            type=ViolationType.CROSS_DOMAIN_IMPORT,
            severity=Severity.CRITICAL,
            from_package="ai/mlops",
            to_package="finance/billing",
            from_domain="ai",
            to_domain="finance",
            file_path="/path/to/file.py",
            line_number=42,
            import_statement="from finance.billing import service",
            description="Cross-domain import violation",
            exception=mock_exception
        )
        
        assert violation.is_exempted()
        mock_exception.is_valid.assert_called_once()
    
    def test_violation_is_exempted_with_invalid_exception(self):
        """Test that violation with invalid exception is not exempted."""
        mock_exception = Mock(spec=BoundaryException)
        mock_exception.is_valid.return_value = False
        
        violation = Violation(
            type=ViolationType.CROSS_DOMAIN_IMPORT,
            severity=Severity.CRITICAL,
            from_package="ai/mlops",
            to_package="finance/billing",
            from_domain="ai",
            to_domain="finance",
            file_path="/path/to/file.py",
            line_number=42,
            import_statement="from finance.billing import service",
            description="Cross-domain import violation",
            exception=mock_exception
        )
        
        assert not violation.is_exempted()
        mock_exception.is_valid.assert_called_once()


class TestAnalysisResult:
    """Test AnalysisResult functionality."""
    
    def test_analysis_result_creation(self):
        """Test creating an empty analysis result."""
        result = AnalysisResult()
        
        assert result.violations == []
        assert result.dependencies == {}
        assert result.domain_packages == {}
        assert result.statistics == {}
    
    def test_add_violation_updates_statistics(self):
        """Test that adding violations updates statistics correctly."""
        result = AnalysisResult()
        
        violation1 = self._create_test_violation(ViolationType.CROSS_DOMAIN_IMPORT, Severity.CRITICAL)
        violation2 = self._create_test_violation(ViolationType.PRIVATE_ACCESS, Severity.WARNING)
        violation3 = self._create_test_violation(ViolationType.CROSS_DOMAIN_IMPORT, Severity.CRITICAL)
        
        result.add_violation(violation1)
        result.add_violation(violation2)
        result.add_violation(violation3)
        
        assert len(result.violations) == 3
        assert result.statistics['violations_critical'] == 2
        assert result.statistics['violations_warning'] == 1
        assert result.statistics['violations_cross_domain_import'] == 2
        assert result.statistics['violations_private_access'] == 1
    
    def test_add_violation_with_exemption_updates_statistics(self):
        """Test that exempted violations are counted separately."""
        result = AnalysisResult()
        
        mock_exception = Mock(spec=BoundaryException)
        mock_exception.is_valid.return_value = True
        
        violation = self._create_test_violation(
            ViolationType.CROSS_DOMAIN_IMPORT, 
            Severity.CRITICAL,
            exception=mock_exception
        )
        
        result.add_violation(violation)
        
        assert result.statistics['violations_critical'] == 1
        assert result.statistics['violations_cross_domain_import'] == 1
        assert result.statistics['exempted_violations'] == 1
    
    def _create_test_violation(self, type_: ViolationType, severity: Severity, exception=None):
        """Helper to create test violations."""
        return Violation(
            type=type_,
            severity=severity,
            from_package="test/package",
            to_package="other/package",
            from_domain="test",
            to_domain="other",
            file_path="/test.py",
            line_number=1,
            import_statement="import other",
            description="Test violation",
            exception=exception
        )


class TestBoundaryAnalyzer:
    """Test BoundaryAnalyzer core functionality."""
    
    @pytest.fixture
    def mock_registry(self):
        """Mock domain registry for testing."""
        registry = Mock(spec=DomainRegistry)
        
        # Setup domains
        ai_domain = Mock(spec=Domain)
        ai_domain.name = "ai"
        finance_domain = Mock(spec=Domain)
        finance_domain.name = "finance"
        
        registry.domains = {
            "ai": ai_domain,
            "finance": finance_domain
        }
        
        return registry
    
    @pytest.fixture
    def analyzer(self, mock_registry):
        """Create analyzer with mock registry."""
        return BoundaryAnalyzer(mock_registry, "/test/monorepo")
    
    def test_analyzer_initialization(self, mock_registry):
        """Test analyzer initialization."""
        analyzer = BoundaryAnalyzer(mock_registry, "/test/monorepo")
        
        assert analyzer.registry == mock_registry
        assert analyzer.monorepo_root == Path("/test/monorepo")
        assert len(analyzer.ignore_patterns) > 0
        assert 'test_' in analyzer.ignore_patterns
        assert '__pycache__' in analyzer.ignore_patterns
    
    def test_should_ignore_file(self, analyzer):
        """Test file ignoring logic."""
        # Should ignore test files
        assert analyzer._should_ignore_file("tests/test_module.py")
        assert analyzer._should_ignore_file("src/test_helper.py")
        
        # Should ignore cache and build directories
        assert analyzer._should_ignore_file("src/__pycache__/module.py")
        assert analyzer._should_ignore_file("build/output.py")
        assert analyzer._should_ignore_file("dist/package.py")
        
        # Should not ignore regular files
        assert not analyzer._should_ignore_file("src/packages/ai/module.py")
        assert not analyzer._should_ignore_file("src/packages/finance/service.py")
    
    def test_get_package_from_path(self, analyzer):
        """Test extracting package from file path."""
        # Valid package paths
        assert analyzer._get_package_from_path(
            "/test/monorepo/src/packages/ai/mlops/service.py"
        ) == "ai/mlops"
        
        assert analyzer._get_package_from_path(
            "/test/monorepo/src/packages/finance/billing/models.py"
        ) == "finance/billing"
        
        # Invalid paths
        assert analyzer._get_package_from_path("/other/path/file.py") is None
        assert analyzer._get_package_from_path("/test/monorepo/file.py") is None
        assert analyzer._get_package_from_path("/test/monorepo/src/file.py") is None
    
    def test_analyze_cross_domain_import_violation(self, analyzer, mock_registry):
        """Test detecting cross-domain import violations."""
        # Setup mock registry responses
        mock_registry.get_domain_for_package.side_effect = lambda pkg: {
            "ai/mlops": "ai",
            "finance/billing": "finance"
        }.get(pkg)
        mock_registry.is_allowed_dependency.return_value = False
        mock_registry.find_applicable_exceptions.return_value = []
        
        # Create test import
        test_import = Import(
            module="finance.billing.service",
            file_path="/test/monorepo/src/packages/ai/mlops/model.py",
            line_number=10,
            is_relative=False
        )
        
        scan_result = ScanResult(imports=[test_import])
        
        with patch.object(analyzer, '_get_package_from_path') as mock_get_package:
            mock_get_package.side_effect = lambda path: {
                "/test/monorepo/src/packages/ai/mlops/model.py": "ai/mlops"
            }.get(path)
            
            with patch('core.domain.services.analyzer.extract_package_from_import') as mock_extract:
                mock_extract.return_value = "finance/billing"
                
                result = analyzer.analyze(scan_result)
        
        # Verify violation was detected
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.type == ViolationType.CROSS_DOMAIN_IMPORT
        assert violation.severity == Severity.CRITICAL
        assert violation.from_package == "ai/mlops"
        assert violation.to_package == "finance/billing"
        assert violation.from_domain == "ai"
        assert violation.to_domain == "finance"
    
    def test_analyze_with_exception_reduces_severity(self, analyzer, mock_registry):
        """Test that valid exceptions reduce violation severity."""
        # Setup mock registry with exception
        mock_exception = Mock(spec=BoundaryException)
        mock_exception.is_valid.return_value = True
        
        mock_registry.get_domain_for_package.side_effect = lambda pkg: {
            "ai/mlops": "ai",
            "finance/billing": "finance"
        }.get(pkg)
        mock_registry.is_allowed_dependency.return_value = False
        mock_registry.find_applicable_exceptions.return_value = [mock_exception]
        
        # Create test import
        test_import = Import(
            module="finance.billing.service",
            file_path="/test/monorepo/src/packages/ai/mlops/model.py",
            line_number=10,
            is_relative=False
        )
        
        scan_result = ScanResult(imports=[test_import])
        
        with patch.object(analyzer, '_get_package_from_path') as mock_get_package:
            mock_get_package.side_effect = lambda path: {
                "/test/monorepo/src/packages/ai/mlops/model.py": "ai/mlops"
            }.get(path)
            
            with patch('core.domain.services.analyzer.extract_package_from_import') as mock_extract:
                mock_extract.return_value = "finance/billing"
                
                result = analyzer.analyze(scan_result)
        
        # Verify violation has reduced severity due to exception
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.severity == Severity.WARNING  # Reduced from CRITICAL
        assert violation.exception == mock_exception
    
    def test_analyze_private_access_violation(self, analyzer, mock_registry):
        """Test detecting private access violations."""
        mock_registry.get_domain_for_package.return_value = "ai"
        
        # Create test import with private module
        test_import = Import(
            module="ai.mlops._internal.service",
            file_path="/test/monorepo/src/packages/ai/mlops/model.py",
            line_number=10,
            is_relative=False
        )
        
        scan_result = ScanResult(imports=[test_import])
        
        with patch.object(analyzer, '_get_package_from_path') as mock_get_package:
            mock_get_package.side_effect = lambda path: {
                "/test/monorepo/src/packages/ai/mlops/model.py": "ai/mlops"
            }.get(path)
            
            with patch('core.domain.services.analyzer.extract_package_from_import') as mock_extract:
                mock_extract.return_value = "ai/mlops"
                
                result = analyzer.analyze(scan_result)
        
        # Verify private access violation was detected
        private_violations = [v for v in result.violations if v.type == ViolationType.PRIVATE_ACCESS]
        assert len(private_violations) == 1
        violation = private_violations[0]
        assert violation.severity == Severity.WARNING
        assert "private module" in violation.description.lower()
    
    def test_analyze_string_reference_violation(self, analyzer, mock_registry):
        """Test detecting string reference violations."""
        # Setup mock registry responses
        mock_registry.get_domain_for_package.side_effect = lambda pkg: {
            "ai/mlops": "ai",
            "finance/billing": "finance"
        }.get(pkg)
        mock_registry.is_allowed_dependency.return_value = False
        
        # Create test string reference
        string_ref = StringReference(
            value="finance.billing.service",
            file_path="/test/monorepo/src/packages/ai/mlops/config.py",
            line_number=20
        )
        
        scan_result = ScanResult(string_references=[string_ref])
        
        with patch.object(analyzer, '_get_package_from_path') as mock_get_package:
            mock_get_package.side_effect = lambda path: {
                "/test/monorepo/src/packages/ai/mlops/config.py": "ai/mlops"
            }.get(path)
            
            with patch.object(analyzer, '_parse_module_string') as mock_parse:
                mock_parse.return_value = "finance/billing"
                
                result = analyzer.analyze(scan_result)
        
        # Verify string reference violation was detected
        string_violations = [v for v in result.violations if v.type == ViolationType.STRING_REFERENCE]
        assert len(string_violations) == 1
        violation = string_violations[0]
        assert violation.severity == Severity.WARNING
        assert "string reference" in violation.description.lower()
    
    def test_check_circular_dependencies(self, analyzer, mock_registry):
        """Test detecting circular dependencies."""
        # Setup registry to return domains
        mock_registry.get_domain_for_package.side_effect = lambda pkg: {
            "ai/mlops": "ai",
            "ai/core": "ai"
        }.get(pkg)
        
        # Create imports that form a circular dependency
        import1 = Import(
            module="ai.core.service",
            file_path="/test/monorepo/src/packages/ai/mlops/model.py",
            line_number=10,
            is_relative=False
        )
        
        import2 = Import(
            module="ai.mlops.model",
            file_path="/test/monorepo/src/packages/ai/core/service.py",
            line_number=15,
            is_relative=False
        )
        
        scan_result = ScanResult(imports=[import1, import2])
        
        with patch.object(analyzer, '_get_package_from_path') as mock_get_package:
            mock_get_package.side_effect = lambda path: {
                "/test/monorepo/src/packages/ai/mlops/model.py": "ai/mlops",
                "/test/monorepo/src/packages/ai/core/service.py": "ai/core"
            }.get(path)
            
            with patch('core.domain.services.analyzer.extract_package_from_import') as mock_extract:
                mock_extract.side_effect = ["ai/core", "ai/mlops"]
                
                result = analyzer.analyze(scan_result)
        
        # Verify circular dependency was detected
        circular_violations = [v for v in result.violations if v.type == ViolationType.CIRCULAR_DEPENDENCY]
        assert len(circular_violations) == 1
        violation = circular_violations[0]
        assert violation.severity == Severity.CRITICAL
        assert "circular" in violation.description.lower()
    
    def test_is_private_access_detection(self, analyzer):
        """Test private access detection logic."""
        # Private module access
        assert analyzer._is_private_access("module._private")
        assert analyzer._is_private_access("package.subpackage._internal.service")
        assert analyzer._is_private_access("_private_module")
        
        # Not private access
        assert not analyzer._is_private_access("module.public")
        assert not analyzer._is_private_access("package.__init__")  # __init__ is not considered private
        assert not analyzer._is_private_access("__main__")  # __main__ is not considered private
        assert not analyzer._is_private_access("module.service")
    
    def test_parse_module_string(self, analyzer, mock_registry):
        """Test parsing module strings to extract packages."""
        # Setup registry with known domains
        mock_registry.domains = {"ai": Mock(), "finance": Mock()}
        
        # Valid module strings
        assert analyzer._parse_module_string("ai.mlops.service") == "ai/mlops"
        assert analyzer._parse_module_string("finance.billing.models") == "finance/billing"
        
        # Invalid module strings
        assert analyzer._parse_module_string("invalid") is None
        assert analyzer._parse_module_string("unknown.domain.service") is None
        assert analyzer._parse_module_string("") is None
    
    def test_get_suggestion_for_cross_domain(self, analyzer):
        """Test getting suggestions for cross-domain violations."""
        # Test known suggestion patterns
        suggestion = analyzer._get_suggestion_for_cross_domain("ai", "finance")
        assert "event-driven" in suggestion.lower() or "api calls" in suggestion.lower()
        
        suggestion = analyzer._get_suggestion_for_cross_domain("finance", "ai")
        assert "interfaces" in suggestion.lower() or "contract" in suggestion.lower()
        
        # Test fallback suggestion
        suggestion = analyzer._get_suggestion_for_cross_domain("unknown", "domain")
        assert "dependency injection" in suggestion.lower() or "events" in suggestion.lower()
    
    def test_compute_statistics(self, analyzer, mock_registry):
        """Test that statistics are computed correctly."""
        # Setup registry
        ai_domain = Mock()
        ai_domain.name = "ai"
        finance_domain = Mock()
        finance_domain.name = "finance"
        
        mock_registry.domains = {"ai": ai_domain, "finance": finance_domain}
        mock_registry.get_domain_for_package.side_effect = lambda pkg: {
            "ai/mlops": "ai",
            "ai/core": "ai",
            "finance/billing": "finance"
        }.get(pkg)
        
        # Create test scan result
        imports = [
            Import("module1", "/test/monorepo/src/packages/ai/mlops/file1.py", 1, False),
            Import("module2", "/test/monorepo/src/packages/ai/core/file2.py", 2, False),
            Import("module3", "/test/monorepo/src/packages/finance/billing/file3.py", 3, False)
        ]
        
        scan_result = ScanResult(imports=imports)
        
        with patch.object(analyzer, '_get_package_from_path') as mock_get_package:
            mock_get_package.side_effect = lambda path: {
                "/test/monorepo/src/packages/ai/mlops/file1.py": "ai/mlops",
                "/test/monorepo/src/packages/ai/core/file2.py": "ai/core",
                "/test/monorepo/src/packages/finance/billing/file3.py": "finance/billing"
            }.get(path)
            
            with patch('core.domain.services.analyzer.extract_package_from_import') as mock_extract:
                mock_extract.side_effect = ["ai/mlops", "ai/core", "finance/billing"]
                
                result = analyzer.analyze(scan_result)
        
        # Verify statistics
        assert result.statistics['total_imports'] == 3
        assert result.statistics['total_files_scanned'] == 3
        assert result.statistics['total_packages'] == 3
        assert result.statistics.get('packages_in_ai', 0) >= 2  # ai/mlops and ai/core
        assert result.statistics.get('packages_in_finance', 0) >= 1  # finance/billing
    
    def test_build_dependencies_creates_dependency_graph(self, analyzer):
        """Test that dependency graph is built correctly."""
        # Create test imports
        imports = [
            Import("target.module", "/test/monorepo/src/packages/source/pkg/file.py", 1, False)
        ]
        
        scan_result = ScanResult(imports=imports)
        
        with patch.object(analyzer, '_get_package_from_path') as mock_get_package:
            mock_get_package.return_value = "source/pkg"
            
            with patch('core.domain.services.analyzer.extract_package_from_import') as mock_extract:
                mock_extract.return_value = "target/module"
                
                result = analyzer.analyze(scan_result)
        
        # Verify dependency graph
        assert "source/pkg" in result.dependencies
        assert "target/module" in result.dependencies["source/pkg"]