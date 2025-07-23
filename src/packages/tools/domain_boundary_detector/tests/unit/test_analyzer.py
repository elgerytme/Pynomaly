"""Unit tests for the analyzer module."""

import pytest
from pathlib import Path

from core.domain.services.scanner import Import, ScanResult
from core.domain.services.registry import DomainRegistry, Domain
from core.domain.services.analyzer import (
    BoundaryAnalyzer, ViolationType, Severity, Violation
)


class TestBoundaryAnalyzer:
    """Test the BoundaryAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a test registry
        self.registry = DomainRegistry()
        
        # Add test domains
        ai_domain = Domain(
            name='ai',
            packages=['ai/mlops', 'ai/ml_platform'],
            allowed_dependencies=['shared', 'infrastructure']
        )
        self.registry.add_domain(ai_domain)
        
        finance_domain = Domain(
            name='finance',
            packages=['finance/billing', 'finance/payments'],
            allowed_dependencies=['shared', 'infrastructure']
        )
        self.registry.add_domain(finance_domain)
        
        shared_domain = Domain(
            name='shared',
            packages=['shared/utils', 'shared/types'],
            allowed_dependencies=['infrastructure']
        )
        self.registry.add_domain(shared_domain)
        
        # Set up analyzer
        self.analyzer = BoundaryAnalyzer(self.registry, '/monorepo')
        
    def test_detect_cross_domain_violation(self):
        """Test detection of cross-domain import violations."""
        # Create scan result with cross-domain import
        scan_result = ScanResult()
        scan_result.imports.append(Import(
            module='src.packages.finance.billing',
            names=['Invoice'],
            file_path='/monorepo/src/packages/ai/mlops/service.py',
            line_number=10,
            import_type='from',
            is_relative=False
        ))
        
        result = self.analyzer.analyze(scan_result)
        
        # Should detect violation
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.type == ViolationType.CROSS_DOMAIN_IMPORT
        assert violation.severity == Severity.CRITICAL
        assert violation.from_domain == 'ai'
        assert violation.to_domain == 'finance'
        assert 'Invoice' in violation.import_statement
        
    def test_allow_shared_imports(self):
        """Test that imports from shared packages are allowed."""
        scan_result = ScanResult()
        scan_result.imports.append(Import(
            module='src.packages.shared.utils',
            names=['format_date'],
            file_path='/monorepo/src/packages/ai/mlops/service.py',
            line_number=5,
            import_type='from',
            is_relative=False
        ))
        
        result = self.analyzer.analyze(scan_result)
        
        # Should not have violations
        assert len(result.violations) == 0
        
    def test_detect_private_access(self):
        """Test detection of private module access."""
        scan_result = ScanResult()
        scan_result.imports.append(Import(
            module='src.packages.shared.utils._internal',
            names=['secret_function'],
            file_path='/monorepo/src/packages/ai/mlops/hack.py',
            line_number=15,
            import_type='from',
            is_relative=False
        ))
        
        result = self.analyzer.analyze(scan_result)
        
        # Should detect private access violation
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert violation.type == ViolationType.PRIVATE_ACCESS
        assert violation.severity == Severity.WARNING
        assert '_internal' in violation.import_statement
        
    def test_same_domain_allowed(self):
        """Test that imports within same domain are allowed."""
        scan_result = ScanResult()
        scan_result.imports.append(Import(
            module='src.packages.ai.ml_platform',
            names=['Pipeline'],
            file_path='/monorepo/src/packages/ai/mlops/orchestrator.py',
            line_number=8,
            import_type='from',
            is_relative=False
        ))
        
        result = self.analyzer.analyze(scan_result)
        
        # Should not have violations
        assert len(result.violations) == 0
        
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        scan_result = ScanResult()
        
        # Package A imports from Package B
        scan_result.imports.append(Import(
            module='src.packages.ai.ml_platform',
            names=['ServiceB'],
            file_path='/monorepo/src/packages/ai/mlops/service_a.py',
            line_number=1,
            import_type='from',
            is_relative=False
        ))
        
        # Package B imports from Package A (circular)
        scan_result.imports.append(Import(
            module='src.packages.ai.mlops',
            names=['ServiceA'],
            file_path='/monorepo/src/packages/ai/ml_platform/service_b.py',
            line_number=1,
            import_type='from',
            is_relative=False
        ))
        
        result = self.analyzer.analyze(scan_result)
        
        # Should detect circular dependency
        violations = [v for v in result.violations 
                     if v.type == ViolationType.CIRCULAR_DEPENDENCY]
        assert len(violations) >= 1
        assert violations[0].severity == Severity.CRITICAL
        
    def test_statistics_computation(self):
        """Test that statistics are computed correctly."""
        scan_result = ScanResult()
        
        # Add various imports
        scan_result.imports.extend([
            Import(
                module='os',
                names=[],
                file_path='/monorepo/src/packages/ai/mlops/file1.py',
                line_number=1,
                import_type='import',
                is_relative=False
            ),
            Import(
                module='src.packages.finance.billing',
                names=['Invoice'],
                file_path='/monorepo/src/packages/ai/mlops/file2.py',
                line_number=5,
                import_type='from',
                is_relative=False
            ),
            Import(
                module='src.packages.shared.utils',
                names=['helper'],
                file_path='/monorepo/src/packages/ai/mlops/file3.py',
                line_number=3,
                import_type='from',
                is_relative=False
            )
        ])
        
        result = self.analyzer.analyze(scan_result)
        
        # Check statistics
        assert result.statistics['total_imports'] == 3
        assert result.statistics['total_files_scanned'] == 3
        assert result.statistics['total_violations'] == 1  # Cross-domain import
        assert result.statistics.get('violations_critical', 0) == 1
        
    def test_ignore_test_files(self):
        """Test that test files are ignored when configured."""
        scan_result = ScanResult()
        scan_result.imports.append(Import(
            module='src.packages.finance.billing',
            names=['Invoice'],
            file_path='/monorepo/src/packages/ai/mlops/tests/test_billing.py',
            line_number=10,
            import_type='from',
            is_relative=False
        ))
        
        result = self.analyzer.analyze(scan_result)
        
        # Should not have violations for test files
        assert len(result.violations) == 0