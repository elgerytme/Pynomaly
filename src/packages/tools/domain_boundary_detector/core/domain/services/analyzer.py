"""Analyzer for detecting domain boundary violations."""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from .scanner import Import, StringReference, ScanResult, extract_package_from_import
from .registry import DomainRegistry, BoundaryException


class ViolationType(Enum):
    """Types of boundary violations."""
    CROSS_DOMAIN_IMPORT = "cross_domain_import"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    PRIVATE_ACCESS = "private_access"
    STRING_REFERENCE = "string_reference"
    TYPE_ANNOTATION = "type_annotation"
    UNDEFINED_DOMAIN = "undefined_domain"


class Severity(Enum):
    """Severity levels for violations."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Violation:
    """Represents a domain boundary violation."""
    type: ViolationType
    severity: Severity
    from_package: str
    to_package: str
    from_domain: Optional[str]
    to_domain: Optional[str]
    file_path: str
    line_number: int
    import_statement: str
    description: str
    suggestion: str = ""
    exception: Optional[BoundaryException] = None
    
    def is_exempted(self) -> bool:
        """Check if this violation has a valid exception."""
        return self.exception is not None and self.exception.is_valid()


@dataclass
class AnalysisResult:
    """Results from analyzing scan results for violations."""
    violations: List[Violation] = field(default_factory=list)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    domain_packages: Dict[str, List[str]] = field(default_factory=dict)
    statistics: Dict[str, int] = field(default_factory=dict)
    
    def add_violation(self, violation: Violation) -> None:
        """Add a violation to the results."""
        self.violations.append(violation)
        
        # Update statistics
        severity_key = f"violations_{violation.severity.value}"
        type_key = f"violations_{violation.type.value}"
        
        self.statistics[severity_key] = self.statistics.get(severity_key, 0) + 1
        self.statistics[type_key] = self.statistics.get(type_key, 0) + 1
        
        if violation.is_exempted():
            self.statistics['exempted_violations'] = self.statistics.get('exempted_violations', 0) + 1


class BoundaryAnalyzer:
    """Analyzer for detecting domain boundary violations."""
    
    def __init__(self, registry: DomainRegistry, monorepo_root: str):
        self.registry = registry
        self.monorepo_root = Path(monorepo_root)
        self.ignore_patterns = [
            'test_', 'tests/', 'examples/', 'docs/', '__pycache__',
            '.pytest_cache', 'build/', 'dist/', '.git/'
        ]
        
    def analyze(self, scan_result: ScanResult) -> AnalysisResult:
        """Analyze scan results for boundary violations."""
        result = AnalysisResult()
        
        # Build dependency graph
        self._build_dependencies(scan_result, result)
        
        # Analyze imports
        for import_obj in scan_result.imports:
            self._analyze_import(import_obj, result)
            
        # Analyze string references
        for string_ref in scan_result.string_references:
            self._analyze_string_reference(string_ref, result)
            
        # Analyze type annotations
        for type_ann in scan_result.type_annotations:
            self._analyze_type_annotation(type_ann, result)
            
        # Check for circular dependencies
        self._check_circular_dependencies(result)
        
        # Compute final statistics
        self._compute_statistics(scan_result, result)
        
        return result
        
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if a file should be ignored."""
        for pattern in self.ignore_patterns:
            if pattern in file_path:
                return True
        return False
        
    def _build_dependencies(self, scan_result: ScanResult, result: AnalysisResult) -> None:
        """Build dependency graph from imports."""
        for import_obj in scan_result.imports:
            if self._should_ignore_file(import_obj.file_path):
                continue
                
            from_package = self._get_package_from_path(import_obj.file_path)
            to_package = extract_package_from_import(
                import_obj.module, 
                import_obj.file_path,
                str(self.monorepo_root)
            )
            
            if from_package and to_package:
                if from_package not in result.dependencies:
                    result.dependencies[from_package] = set()
                result.dependencies[from_package].add(to_package)
                
    def _get_package_from_path(self, file_path: str) -> Optional[str]:
        """Extract package identifier from file path."""
        try:
            rel_path = Path(file_path).relative_to(self.monorepo_root)
            parts = rel_path.parts
            
            # Look for src/packages pattern
            if len(parts) >= 4 and parts[0] == 'src' and parts[1] == 'packages':
                # src/packages/domain/package/... -> domain/package
                return f"{parts[2]}/{parts[3]}"
                
            return None
        except:
            return None
            
    def _analyze_import(self, import_obj: Import, result: AnalysisResult) -> None:
        """Analyze a single import for violations."""
        if self._should_ignore_file(import_obj.file_path):
            return
            
        from_package = self._get_package_from_path(import_obj.file_path)
        to_package = extract_package_from_import(
            import_obj.module,
            import_obj.file_path,
            str(self.monorepo_root)
        )
        
        if not from_package or not to_package:
            return
            
        from_domain = self.registry.get_domain_for_package(from_package)
        to_domain = self.registry.get_domain_for_package(to_package)
        
        # Check for cross-domain violation
        if from_domain and to_domain and from_domain != to_domain:
            if not self.registry.is_allowed_dependency(from_package, to_package):
                # Check for exceptions
                exceptions = self.registry.find_applicable_exceptions(from_package, to_package)
                exception = exceptions[0] if exceptions else None
                
                violation = Violation(
                    type=ViolationType.CROSS_DOMAIN_IMPORT,
                    severity=Severity.CRITICAL if not exception else Severity.WARNING,
                    from_package=from_package,
                    to_package=to_package,
                    from_domain=from_domain,
                    to_domain=to_domain,
                    file_path=import_obj.file_path,
                    line_number=import_obj.line_number,
                    import_statement=str(import_obj),
                    description=f"{from_domain} domain importing from {to_domain} domain",
                    suggestion=self._get_suggestion_for_cross_domain(from_domain, to_domain),
                    exception=exception
                )
                result.add_violation(violation)
                
        # Check for private access
        if to_package and self._is_private_access(import_obj.module):
            violation = Violation(
                type=ViolationType.PRIVATE_ACCESS,
                severity=Severity.WARNING,
                from_package=from_package or "unknown",
                to_package=to_package,
                from_domain=from_domain,
                to_domain=to_domain,
                file_path=import_obj.file_path,
                line_number=import_obj.line_number,
                import_statement=str(import_obj),
                description="Accessing private module (starts with _)",
                suggestion="Use public API instead of private implementation details"
            )
            result.add_violation(violation)
            
    def _analyze_string_reference(self, string_ref: StringReference, result: AnalysisResult) -> None:
        """Analyze string references for potential violations."""
        if self._should_ignore_file(string_ref.file_path):
            return
            
        # Try to parse as module reference
        potential_package = self._parse_module_string(string_ref.value)
        if not potential_package:
            return
            
        from_package = self._get_package_from_path(string_ref.file_path)
        if not from_package:
            return
            
        from_domain = self.registry.get_domain_for_package(from_package)
        to_domain = self.registry.get_domain_for_package(potential_package)
        
        if from_domain and to_domain and from_domain != to_domain:
            if not self.registry.is_allowed_dependency(from_package, potential_package):
                violation = Violation(
                    type=ViolationType.STRING_REFERENCE,
                    severity=Severity.WARNING,
                    from_package=from_package,
                    to_package=potential_package,
                    from_domain=from_domain,
                    to_domain=to_domain,
                    file_path=string_ref.file_path,
                    line_number=string_ref.line_number,
                    import_statement=f'"{string_ref.value}"',
                    description=f"String reference to {to_domain} domain from {from_domain}",
                    suggestion="Use dependency injection or configuration instead of hardcoded references"
                )
                result.add_violation(violation)
                
    def _analyze_type_annotation(self, type_ann: StringReference, result: AnalysisResult) -> None:
        """Analyze type annotations for violations."""
        # Similar to string reference analysis but specifically for type hints
        self._analyze_string_reference(type_ann, result)
        
    def _check_circular_dependencies(self, result: AnalysisResult) -> None:
        """Check for circular dependencies between packages."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(package: str, path: List[str]) -> Optional[List[str]]:
            visited.add(package)
            rec_stack.add(package)
            path.append(package)
            
            if package in result.dependencies:
                for neighbor in result.dependencies[package]:
                    if neighbor not in visited:
                        cycle = has_cycle(neighbor, path.copy())
                        if cycle:
                            return cycle
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        return path[cycle_start:] + [neighbor]
                        
            rec_stack.remove(package)
            return None
            
        for package in result.dependencies:
            if package not in visited:
                cycle = has_cycle(package, [])
                if cycle:
                    # Create violation for circular dependency
                    violation = Violation(
                        type=ViolationType.CIRCULAR_DEPENDENCY,
                        severity=Severity.CRITICAL,
                        from_package=cycle[0],
                        to_package=cycle[1],
                        from_domain=self.registry.get_domain_for_package(cycle[0]),
                        to_domain=self.registry.get_domain_for_package(cycle[1]),
                        file_path="",
                        line_number=0,
                        import_statement=f"Circular: {' -> '.join(cycle)}",
                        description=f"Circular dependency detected: {' -> '.join(cycle)}",
                        suggestion="Refactor to remove circular dependency, possibly using interfaces or events"
                    )
                    result.add_violation(violation)
                    
    def _is_private_access(self, module: str) -> bool:
        """Check if the import accesses private modules."""
        parts = module.split('.')
        return any(part.startswith('_') and not part.startswith('__') for part in parts)
        
    def _parse_module_string(self, s: str) -> Optional[str]:
        """Try to parse a string as a module reference and extract package."""
        # Look for patterns like "ai.mlops.service" or "finance.billing"
        if '.' not in s:
            return None
            
        parts = s.split('.')
        if len(parts) >= 2:
            # Check if it matches known domain patterns
            potential_domain = parts[0]
            if potential_domain in self.registry.domains:
                return f"{parts[0]}/{parts[1]}" if len(parts) > 1 else None
                
        return None
        
    def _get_suggestion_for_cross_domain(self, from_domain: str, to_domain: str) -> str:
        """Get suggestion for fixing cross-domain violation."""
        suggestions = {
            ('ai', 'finance'): "Use event-driven communication or API calls instead of direct imports",
            ('finance', 'ai'): "Define interfaces in a shared contract package",
            ('data', 'finance'): "Use data transfer objects (DTOs) for communication",
        }
        
        key = (from_domain, to_domain)
        if key in suggestions:
            return suggestions[key]
            
        return "Consider using dependency injection, events, or well-defined interfaces"
        
    def _compute_statistics(self, scan_result: ScanResult, result: AnalysisResult) -> None:
        """Compute final statistics."""
        result.statistics['total_imports'] = len(scan_result.imports)
        result.statistics['total_files_scanned'] = len(set(imp.file_path for imp in scan_result.imports))
        result.statistics['total_violations'] = len(result.violations)
        result.statistics['total_packages'] = len(result.dependencies)
        
        # Count packages per domain
        for domain in self.registry.domains.values():
            packages_in_domain = [p for p in result.dependencies.keys() 
                                 if self.registry.get_domain_for_package(p) == domain.name]
            result.domain_packages[domain.name] = packages_in_domain
            result.statistics[f'packages_in_{domain.name}'] = len(packages_in_domain)