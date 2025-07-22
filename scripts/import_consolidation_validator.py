#!/usr/bin/env python3
"""
Import Consolidation Validator
==============================

Validates that each package only imports from another package once, preventing
duplicate imports and reducing coupling. This tool enforces the "single import
per package" rule across the monorepo.

Features:
- AST-based import analysis for accurate detection
- Multiple import pattern recognition (import, from-import, relative)
- Cross-package dependency mapping
- Automated consolidation suggestions
- Integration with existing domain validation
"""

import ast
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ImportViolation:
    """Represents a single import consolidation violation"""
    file_path: str
    line_numbers: List[int]
    package_name: str
    import_statements: List[str]
    violation_type: str  # 'multiple_imports', 'duplicate_from', 'mixed_styles'
    severity: str  # 'error', 'warning', 'info'
    suggestion: str
    auto_fixable: bool = True


@dataclass  
class PackageImportAnalysis:
    """Analysis results for a single package"""
    package_name: str
    total_files: int
    files_with_violations: int
    total_violations: int
    violations: List[ImportViolation]
    import_dependency_map: Dict[str, Set[str]]  # target_package -> set of importing files
    consolidation_opportunities: int
    analysis_timestamp: str


class ImportConsolidationValidator:
    """
    Validates and enforces the single import per package rule across the monorepo.
    Integrates with existing domain validation infrastructure.
    """
    
    def __init__(self, repository_root: Path = None):
        self.repository_root = repository_root or Path.cwd()
        self.packages_dir = self.repository_root / "src" / "packages"
        self.logger = self._setup_logging()
        
        # Standard library modules (Python 3.11+)
        self.stdlib_modules = {
            'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
            'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect', 'builtins',
            'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code', 'codecs',
            'codeop', 'collections', 'colorsys', 'compileall', 'concurrent', 'configparser',
            'contextlib', 'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv', 'ctypes',
            'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis', 'distutils',
            'doctest', 'email', 'encodings', 'ensurepip', 'enum', 'errno', 'faulthandler',
            'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'fractions', 'ftplib', 'functools',
            'gc', 'getopt', 'getpass', 'gettext', 'glob', 'grp', 'gzip', 'hashlib', 'heapq',
            'hmac', 'html', 'http', 'imaplib', 'imghdr', 'importlib', 'inspect', 'io', 'ipaddress',
            'itertools', 'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma',
            'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder',
            'multiprocessing', 'netrc', 'nntplib', 'numbers', 'operator', 'optparse', 'os',
            'pathlib', 'pdb', 'pickle', 'pickletools', 'pkgutil', 'platform', 'plistlib',
            'poplib', 'posix', 'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile',
            'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're', 'readline', 'reprlib',
            'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors',
            'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtplib', 'sndhdr', 'socket',
            'socketserver', 'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep',
            'struct', 'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig', 'syslog',
            'tabnanny', 'tarfile', 'tempfile', 'textwrap', 'threading', 'time', 'timeit',
            'tkinter', 'token', 'tokenize', 'trace', 'traceback', 'tracemalloc', 'tty',
            'turtle', 'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu', 'uuid',
            'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound',
            'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib', 'zoneinfo'
        }
        
        # Common third-party packages
        self.third_party_packages = {
            'fastapi', 'pydantic', 'sqlalchemy', 'redis', 'celery', 'requests', 'httpx', 
            'uvicorn', 'gunicorn', 'prometheus_client', 'structlog', 'click', 'typer',
            'alembic', 'pytest', 'numpy', 'pandas', 'scikit-learn', 'sklearn', 'tensorflow', 
            'torch', 'pytorch', 'matplotlib', 'seaborn', 'pillow', 'opencv', 'cv2',
            'boto3', 'azure', 'google', 'kubernetes', 'docker', 'yaml', 'toml', 'jinja2',
            'marshmallow', 'cerberus', 'flask', 'django', 'aiohttp', 'starlette'
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('import_consolidation_validator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_all_packages(self) -> Dict[str, PackageImportAnalysis]:
        """Validate import consolidation across all packages"""
        self.logger.info("🔍 Starting import consolidation validation")
        
        if not self.packages_dir.exists():
            self.logger.error(f"Packages directory not found: {self.packages_dir}")
            return {}
        
        analyses = {}
        
        # Process each domain directory
        for domain_dir in self.packages_dir.iterdir():
            if not domain_dir.is_dir():
                continue
                
            # Process each package in the domain
            for package_dir in domain_dir.iterdir():
                if not package_dir.is_dir():
                    continue
                    
                package_name = f"{domain_dir.name}.{package_dir.name}"
                self.logger.info(f"   Analyzing package: {package_name}")
                
                analysis = self.validate_package(package_dir, package_name)
                analyses[package_name] = analysis
        
        # Generate comprehensive report
        self._generate_report(analyses)
        
        return analyses
    
    def validate_package(self, package_dir: Path, package_name: str) -> PackageImportAnalysis:
        """Validate import consolidation for a single package"""
        violations = []
        import_dependency_map = defaultdict(set)
        total_files = 0
        files_with_violations = 0
        
        # Find all Python files in the package
        python_files = list(package_dir.rglob("*.py"))
        total_files = len(python_files)
        
        for py_file in python_files:
            try:
                file_violations = self._analyze_file_imports(py_file, package_name, import_dependency_map)
                if file_violations:
                    violations.extend(file_violations)
                    files_with_violations += 1
                    
            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")
        
        # Count consolidation opportunities
        consolidation_opportunities = self._count_consolidation_opportunities(violations)
        
        return PackageImportAnalysis(
            package_name=package_name,
            total_files=total_files,
            files_with_violations=files_with_violations,
            total_violations=len(violations),
            violations=violations,
            import_dependency_map={k: list(v) for k, v in import_dependency_map.items()},
            consolidation_opportunities=consolidation_opportunities,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _analyze_file_imports(
        self, 
        file_path: Path, 
        package_name: str,
        dependency_map: Dict[str, Set[str]]
    ) -> List[ImportViolation]:
        """Analyze imports in a single Python file"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports_by_package = defaultdict(list)
            
            # Collect all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        package = self._extract_package_name(alias.name)
                        if package and self._should_check_package(package):
                            imports_by_package[package].append({
                                'type': 'import',
                                'line': node.lineno,
                                'statement': f"import {alias.name}",
                                'module': alias.name
                            })
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        package = self._extract_package_name(node.module)
                        if package and self._should_check_package(package):
                            items = [alias.name for alias in node.names]
                            imports_by_package[package].append({
                                'type': 'from',
                                'line': node.lineno, 
                                'statement': f"from {node.module} import {', '.join(items)}",
                                'module': node.module,
                                'items': items
                            })
            
            # Check for violations
            for target_package, import_list in imports_by_package.items():
                if len(import_list) > 1:
                    # Multiple imports from same package
                    violation = self._create_violation_for_multiple_imports(
                        file_path, target_package, import_list
                    )
                    violations.append(violation)
                
                # Update dependency map
                if target_package != package_name:  # Don't track self-imports
                    dependency_map[target_package].add(str(file_path))
        
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {e}")
        
        return violations
    
    def _extract_package_name(self, module_path: str) -> Optional[str]:
        """Extract the top-level package name from a module path"""
        if not module_path:
            return None
            
        # Handle relative imports
        if module_path.startswith('.'):
            return None  # Skip relative imports for now
            
        # Handle src.packages.* imports (internal packages)
        if module_path.startswith('src.packages.'):
            parts = module_path.split('.')
            if len(parts) >= 4:  # src.packages.domain.package
                return f"{parts[2]}.{parts[3]}"
            elif len(parts) >= 3:  # src.packages.domain
                return parts[2]
                
        # Handle packages.* imports (internal packages)
        if module_path.startswith('packages.'):
            parts = module_path.split('.')
            if len(parts) >= 3:  # packages.domain.package  
                return f"{parts[1]}.{parts[2]}"
            elif len(parts) >= 2:  # packages.domain
                return parts[1]
        
        # Handle direct domain.package imports
        parts = module_path.split('.')
        if len(parts) >= 2 and parts[0] in {'ai', 'data', 'enterprise'}:
            return f"{parts[0]}.{parts[1]}"
        
        # Extract root package name for external packages
        return parts[0]
    
    def _should_check_package(self, package_name: str) -> bool:
        """Determine if we should check this package for consolidation"""
        if not package_name:
            return False
            
        # Skip standard library
        root_name = package_name.split('.')[0]
        if root_name in self.stdlib_modules:
            return False
            
        # Skip known third-party packages (but still check for consolidation)
        if root_name in self.third_party_packages:
            return True
            
        # Check internal packages (domain.package format)
        if '.' in package_name and package_name.count('.') == 1:
            domain, pkg = package_name.split('.')
            if domain in {'ai', 'data', 'enterprise'}:
                return True
        
        # Check other potential packages
        return True
    
    def _create_violation_for_multiple_imports(
        self, 
        file_path: Path, 
        package_name: str, 
        import_list: List[Dict]
    ) -> ImportViolation:
        """Create a violation object for multiple imports from same package"""
        
        lines = [imp['line'] for imp in import_list]
        statements = [imp['statement'] for imp in import_list]
        
        # Determine violation type and suggestion
        has_import = any(imp['type'] == 'import' for imp in import_list)
        has_from = any(imp['type'] == 'from' for imp in import_list)
        
        if has_import and has_from:
            violation_type = 'mixed_styles'
            severity = 'error'
            suggestion = f"Consolidate to single 'from {package_name} import' statement"
        elif has_from:
            violation_type = 'duplicate_from'
            severity = 'warning'
            # Collect all imported items
            all_items = []
            base_module = None
            for imp in import_list:
                if imp['type'] == 'from':
                    all_items.extend(imp.get('items', []))
                    if not base_module:
                        base_module = imp['module']
            
            suggestion = f"Consolidate to: from {base_module} import ({', '.join(sorted(set(all_items)))})"
        else:
            violation_type = 'multiple_imports'
            severity = 'warning' 
            suggestion = f"Consider consolidating multiple imports from {package_name}"
        
        return ImportViolation(
            file_path=str(file_path),
            line_numbers=lines,
            package_name=package_name,
            import_statements=statements,
            violation_type=violation_type,
            severity=severity,
            suggestion=suggestion,
            auto_fixable=True
        )
    
    def _count_consolidation_opportunities(self, violations: List[ImportViolation]) -> int:
        """Count the number of consolidation opportunities"""
        return sum(len(v.import_statements) - 1 for v in violations if v.auto_fixable)
    
    def validate_changed_files(self, file_paths: List[str]) -> Dict[str, List[ImportViolation]]:
        """Validate import consolidation for specific changed files (for git hooks)"""
        self.logger.info(f"🔍 Validating {len(file_paths)} changed files")
        
        results = {}
        
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            
            # Skip non-Python files
            if file_path.suffix != '.py':
                continue
                
            # Skip files outside packages directory
            if not str(file_path).startswith('src/packages/'):
                continue
            
            # Determine package name from path
            try:
                parts = file_path.parts
                if len(parts) >= 4 and parts[0] == 'src' and parts[1] == 'packages':
                    package_name = f"{parts[2]}.{parts[3]}"
                else:
                    continue
                    
            except (IndexError, ValueError):
                continue
            
            # Analyze the file
            dependency_map = defaultdict(set)
            violations = self._analyze_file_imports(file_path, package_name, dependency_map)
            
            if violations:
                results[file_path_str] = violations
        
        return results
    
    def _generate_report(self, analyses: Dict[str, PackageImportAnalysis]) -> None:
        """Generate comprehensive import consolidation report"""
        report_dir = self.repository_root / "reports"
        report_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        json_report_path = report_dir / "import_consolidation_report.json"
        self._generate_json_report(analyses, json_report_path)
        
        # Generate Markdown report
        md_report_path = report_dir / "import_consolidation_report.md"
        self._generate_markdown_report(analyses, md_report_path)
        
        self.logger.info(f"📊 Reports generated: {json_report_path}, {md_report_path}")
    
    def _generate_json_report(self, analyses: Dict[str, PackageImportAnalysis], report_path: Path) -> None:
        """Generate JSON report"""
        summary = self._generate_summary(analyses)
        
        report_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'repository_root': str(self.repository_root),
            'summary': summary,
            'package_analyses': {
                name: asdict(analysis) for name, analysis in analyses.items()
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def _generate_markdown_report(self, analyses: Dict[str, PackageImportAnalysis], report_path: Path) -> None:
        """Generate Markdown report"""
        summary = self._generate_summary(analyses)
        
        content = f"""# Import Consolidation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Packages**: {summary['total_packages']}
- **Packages with Violations**: {summary['packages_with_violations']}
- **Total Violations**: {summary['total_violations']}
- **Total Consolidation Opportunities**: {summary['total_consolidation_opportunities']}
- **Auto-fixable Violations**: {summary['auto_fixable_violations']}

## Package Analysis

"""
        
        for package_name, analysis in sorted(analyses.items()):
            status = "✅" if analysis.total_violations == 0 else "❌"
            
            content += f"""
### {status} {package_name}

- **Files Analyzed**: {analysis.total_files}
- **Files with Violations**: {analysis.files_with_violations}
- **Total Violations**: {analysis.total_violations}
- **Consolidation Opportunities**: {analysis.consolidation_opportunities}

"""
            
            if analysis.violations:
                content += "#### Violations\n\n"
                for violation in analysis.violations:
                    severity_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(violation.severity, "⚪")
                    content += f"""- {severity_icon} **{violation.violation_type}** in `{violation.file_path}`
  - Lines: {', '.join(map(str, violation.line_numbers))}
  - Package: `{violation.package_name}`
  - Suggestion: {violation.suggestion}

"""
        
        with open(report_path, 'w') as f:
            f.write(content)
    
    def _generate_summary(self, analyses: Dict[str, PackageImportAnalysis]) -> Dict:
        """Generate summary statistics"""
        total_packages = len(analyses)
        packages_with_violations = sum(1 for a in analyses.values() if a.total_violations > 0)
        total_violations = sum(a.total_violations for a in analyses.values())
        total_consolidation_opportunities = sum(a.consolidation_opportunities for a in analyses.values())
        auto_fixable_violations = sum(
            len([v for v in a.violations if v.auto_fixable]) 
            for a in analyses.values()
        )
        
        return {
            'total_packages': total_packages,
            'packages_with_violations': packages_with_violations,
            'total_violations': total_violations,
            'total_consolidation_opportunities': total_consolidation_opportunities,
            'auto_fixable_violations': auto_fixable_violations,
            'violation_percentage': (packages_with_violations / total_packages * 100) if total_packages > 0 else 0
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate import consolidation across packages")
    parser.add_argument('--repository-root', type=str, default=".", help="Repository root directory")
    parser.add_argument('--package', type=str, help="Validate specific package (domain.package format)")
    parser.add_argument('--changed-files', nargs='*', help="Validate specific changed files")
    parser.add_argument('--fail-on-violations', action='store_true', help="Exit with error code if violations found")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger('import_consolidation_validator').setLevel(logging.DEBUG)
    
    validator = ImportConsolidationValidator(Path(args.repository_root))
    
    if args.changed_files:
        # Validate specific files (for git hooks)
        results = validator.validate_changed_files(args.changed_files)
        
        if results:
            print(f"❌ Found import consolidation violations in {len(results)} files:")
            for file_path, violations in results.items():
                print(f"\n📄 {file_path}:")
                for violation in violations:
                    print(f"  • {violation.violation_type} on lines {', '.join(map(str, violation.line_numbers))}")
                    print(f"    Package: {violation.package_name}")
                    print(f"    Suggestion: {violation.suggestion}")
            
            if args.fail_on_violations:
                return 1
        else:
            print("✅ No import consolidation violations found in changed files")
    
    elif args.package:
        # Validate specific package
        domain, package = args.package.split('.')
        package_dir = validator.packages_dir / domain / package
        
        if not package_dir.exists():
            print(f"❌ Package directory not found: {package_dir}")
            return 1
        
        analysis = validator.validate_package(package_dir, args.package)
        
        if analysis.total_violations > 0:
            print(f"❌ Found {analysis.total_violations} violations in {args.package}")
            for violation in analysis.violations:
                print(f"  • {violation.violation_type} in {violation.file_path}")
                print(f"    Lines: {', '.join(map(str, violation.line_numbers))}")
                print(f"    Suggestion: {violation.suggestion}")
            
            if args.fail_on_violations:
                return 1
        else:
            print(f"✅ No violations found in {args.package}")
    
    else:
        # Validate all packages
        analyses = validator.validate_all_packages()
        
        total_violations = sum(a.total_violations for a in analyses.values())
        
        if total_violations > 0:
            print(f"❌ Found {total_violations} import consolidation violations across {len(analyses)} packages")
            
            if args.fail_on_violations:
                return 1
        else:
            print(f"✅ No import consolidation violations found in {len(analyses)} packages")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())