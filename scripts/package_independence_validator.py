#!/usr/bin/env python3
"""
Package Independence Validator
=============================
Validates and enforces that packages are truly self-contained and independent
"""

import os
import sys
import json
import ast
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import yaml
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class DependencyViolation:
    """Represents a dependency violation"""
    source_package: str
    target_package: str
    violation_type: str  # 'import', 'config', 'database', 'api_call'
    location: str
    severity: str  # 'error', 'warning', 'info'
    description: str
    suggestion: Optional[str] = None


@dataclass
class PackageAnalysis:
    """Results of package independence analysis"""
    package_name: str
    is_self_contained: bool
    self_containment_score: float  # 0-100
    violations: List[DependencyViolation]
    missing_components: List[str]
    external_dependencies: List[str]
    internal_dependencies: List[str]
    configuration_issues: List[str]
    test_coverage: float
    documentation_completeness: float
    deployment_readiness: float
    analysis_timestamp: str


class PackageIndependenceValidator:
    """
    Validates that packages are truly self-contained and independent,
    with no inappropriate cross-package dependencies
    """
    
    def __init__(self, repository_root: Path = None):
        self.repository_root = repository_root or Path.cwd()
        self.packages_dir = self.repository_root / "src" / "packages"
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        
        # Dependency analysis patterns
        self.import_patterns = [
            r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
            r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
        ]
        
        # Required components for self-contained packages
        self.required_components = {
            'core': ['__init__.py', 'main.py', 'config.py'],
            'tests': ['test_*.py', '*_test.py'],
            'docs': ['README.md', 'CHANGELOG.md'],
            'deployment': ['Dockerfile', 'docker-compose.yml'],
            'ci_cd': ['.github/workflows/*.yml', 'Makefile'],
            'monitoring': ['monitoring/*.yml', 'health-check.sh'],
            'configuration': ['config/*.yml', '.env*'],
        }
        
        # Allowed external dependencies (not cross-package)
        self.allowed_external_deps = {
            'standard_library',
            'third_party_packages',
            'system_packages',
            'cloud_services',
        }
        
        # Duplicate package detection patterns
        self.duplicate_patterns = [
            r'^(.+)_new$',      # package_name_new
            r'^(.+)_v2$',       # package_name_v2  
            r'^(.+)_backup$',   # package_name_backup
            r'^(.+)_old$',      # package_name_old
            r'^new_(.+)$',      # new_package_name
            r'^(.+)_copy$',     # package_name_copy
            r'^(.+)_temp$',     # package_name_temp
        ]
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger('package_independence_validator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        config_file = self.repository_root / ".github" / "PACKAGE_INDEPENDENCE_RULES.yml"
        
        default_config = {
            'enforcement_level': 'strict',  # strict, moderate, lenient
            'allowed_cross_package_deps': [],
            'ignored_packages': ['common', 'shared', 'utils'],
            'required_self_containment_score': 85.0,
            'max_external_dependencies': 50,
            'require_all_components': True,
            'auto_fix_violations': False,
            'notification_channels': [],
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(f"Could not load config file: {e}")
        
        return default_config
    
    def _get_package_layer(self, package_name: str) -> Optional[str]:
        """Determine which architectural layer a package belongs to"""
        if not self.config.get('dependency_layers'):
            return None
            
        # Normalize package name
        normalized_name = package_name.replace('_', '.')
        
        for layer_name, layer_config in self.config['dependency_layers'].items():
            if isinstance(layer_config, dict):
                packages = layer_config.get('packages', [])
            else:
                packages = layer_config
                
            if isinstance(packages, list):
                for pattern in packages:
                    # Check exact match or prefix match
                    if (normalized_name == pattern or 
                        normalized_name.startswith(pattern + '.') or
                        package_name in pattern or
                        pattern in normalized_name):
                        return layer_name
        return None
    
    def _validate_layer_dependency(self, source_package: str, target_package: str) -> Tuple[bool, str]:
        """Validate that dependency follows hierarchical layer rules"""
        if not self.config.get('hierarchical_validation', {}).get('enable_layer_validation', False):
            return True, "Layer validation disabled"
        
        source_layer = self._get_package_layer(source_package)
        target_layer = self._get_package_layer(target_package)
        
        if not source_layer or not target_layer:
            return True, f"Could not determine layers for {source_package} -> {target_package}"
        
        # Extract layer numbers for comparison
        def get_layer_number(layer: str) -> int:
            if 'layer_1' in layer: return 1
            elif 'layer_2' in layer: return 2  
            elif 'layer_3' in layer: return 3
            elif 'layer_4' in layer: return 4
            elif 'layer_5' in layer: return 5
            return 0
        
        source_num = get_layer_number(source_layer)
        target_num = get_layer_number(target_layer)
        
        # Higher numbered layers can depend on lower numbered layers
        if source_num >= target_num:
            return True, f"Valid dependency: {source_layer}({source_num}) -> {target_layer}({target_num})"
        else:
            return False, f"Invalid dependency: {source_layer}({source_num}) -> {target_layer}({target_num}) violates architecture (higher -> lower only)"
    
    def validate_all_packages(self) -> Dict[str, PackageAnalysis]:
        """Validate independence of all packages"""
        self.logger.info("🔍 Starting package independence validation")
        
        if not self.packages_dir.exists():
            self.logger.error(f"Packages directory not found: {self.packages_dir}")
            return {}
        
        package_analyses = {}
        package_dirs = [d for d in self.packages_dir.iterdir() if d.is_dir()]
        
        for package_dir in package_dirs:
            package_name = package_dir.name
            
            # Skip ignored packages
            if package_name in self.config.get('ignored_packages', []):
                self.logger.info(f"   Skipping ignored package: {package_name}")
                continue
            
            self.logger.info(f"   Analyzing package: {package_name}")
            analysis = self.validate_package_independence(package_dir)
            package_analyses[package_name] = analysis
        
        # Perform cross-package analysis
        self._analyze_cross_package_dependencies(package_analyses)
        
        # Validate for duplicate packages
        duplicate_violations = self._validate_duplicate_packages(package_dirs)
        if duplicate_violations:
            for violation in duplicate_violations:
                # Add duplicate violations to the affected package analysis
                if violation.source_package in package_analyses:
                    package_analyses[violation.source_package].violations.append(violation)
        
        # Generate report
        self._generate_independence_report(package_analyses)
        
        return package_analyses
    
    def validate_package_independence(self, package_dir: Path) -> PackageAnalysis:
        """Validate independence of a single package"""
        package_name = package_dir.name
        violations = []
        
        # Analyze imports and dependencies
        import_violations = self._analyze_imports(package_dir, package_name)
        violations.extend(import_violations)
        
        # Analyze configuration dependencies
        config_violations = self._analyze_configuration_dependencies(package_dir, package_name)
        violations.extend(config_violations)
        
        # Analyze database dependencies
        db_violations = self._analyze_database_dependencies(package_dir, package_name)
        violations.extend(db_violations)
        
        # Analyze API call dependencies
        api_violations = self._analyze_api_dependencies(package_dir, package_name)
        violations.extend(api_violations)
        
        # Check required components
        missing_components = self._check_required_components(package_dir)
        
        # Analyze external dependencies
        external_deps, internal_deps = self._analyze_dependencies_from_requirements(package_dir)
        
        # Check configuration completeness
        config_issues = self._check_configuration_completeness(package_dir)
        
        # Calculate metrics
        test_coverage = self._calculate_test_coverage(package_dir)
        docs_completeness = self._calculate_documentation_completeness(package_dir)
        deployment_readiness = self._calculate_deployment_readiness(package_dir)
        
        # Calculate self-containment score
        self_containment_score = self._calculate_self_containment_score(
            violations, missing_components, external_deps, config_issues
        )
        
        is_self_contained = (
            self_containment_score >= self.config.get('required_self_containment_score', 85.0)
            and len([v for v in violations if v.severity == 'error']) == 0
        )
        
        return PackageAnalysis(
            package_name=package_name,
            is_self_contained=is_self_contained,
            self_containment_score=self_containment_score,
            violations=violations,
            missing_components=missing_components,
            external_dependencies=external_deps,
            internal_dependencies=internal_deps,
            configuration_issues=config_issues,
            test_coverage=test_coverage,
            documentation_completeness=docs_completeness,
            deployment_readiness=deployment_readiness,
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def _analyze_imports(self, package_dir: Path, package_name: str) -> List[DependencyViolation]:
        """Analyze import statements for cross-package dependencies"""
        violations = []
        python_files = list(package_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                violations.extend(self._analyze_file_imports(py_file, content, package_name))
            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")
        
        return violations
    
    def _analyze_file_imports(self, file_path: Path, content: str, package_name: str) -> List[DependencyViolation]:
        """Analyze imports in a single file"""
        violations = []
        
        # Parse the AST to get import information
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return [DependencyViolation(
                source_package=package_name,
                target_package="",
                violation_type="syntax_error",
                location=str(file_path),
                severity="error",
                description=f"Syntax error in file: {e}"
            )]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    violation = self._check_import_dependency(
                        alias.name, package_name, str(file_path), node.lineno
                    )
                    if violation:
                        violations.append(violation)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    violation = self._check_import_dependency(
                        node.module, package_name, str(file_path), node.lineno
                    )
                    if violation:
                        violations.append(violation)
        
        return violations
    
    def _check_import_dependency(
        self, 
        import_module: str, 
        package_name: str, 
        file_path: str, 
        line_no: int
    ) -> Optional[DependencyViolation]:
        """Check if an import represents a problematic dependency"""
        
        # Skip standard library and third-party imports
        if self._is_standard_library(import_module) or self._is_third_party(import_module):
            return None
        
        # Check for cross-package dependencies
        if import_module.startswith('src.packages.') or import_module.startswith('packages.'):
            parts = import_module.split('.')
            target_package = None
            
            # Extract target package name from different import patterns
            if import_module.startswith('src.packages.') and len(parts) >= 3:
                target_package = parts[2]
            elif import_module.startswith('packages.') and len(parts) >= 2:
                target_package = parts[1]
                
            if target_package and target_package != package_name:
                # First check hierarchical layer validation
                layer_valid, layer_msg = self._validate_layer_dependency(package_name, target_package)
                
                if not layer_valid:
                    return DependencyViolation(
                        source_package=package_name,
                        target_package=target_package,
                        violation_type="layer_violation",
                        location=f"{file_path}:{line_no}",
                        severity="error", 
                        description=f"Architectural layer violation: {layer_msg}",
                        suggestion="Restructure to follow hierarchical dependency rules (higher -> lower layers only)"
                    )
                
                # Then check if this cross-package dependency is explicitly allowed
                allowed_deps = self.config.get('allowed_cross_package_deps', [])
                dep_key = f"{package_name} -> {target_package}"
                alt_dep_key = f"data.{package_name} -> ai.{target_package}"  # Handle domain prefixes
                alt_dep_key2 = f"data.{package_name} -> data.{target_package}"
                
                if not any(dep in allowed_deps for dep in [dep_key, alt_dep_key, alt_dep_key2]):
                    return DependencyViolation(
                        source_package=package_name,
                        target_package=target_package,
                        violation_type="import",
                        location=f"{file_path}:{line_no}",
                        severity="warning",  # Reduced to warning since layer validation passed
                        description=f"Cross-package import not explicitly allowed: {import_module}",
                        suggestion=f"Add '{dep_key}' to allowed_cross_package_deps if this is intentional"
                    )
        
        # Check for relative imports outside the package
        if import_module.startswith('..') and 'packages' in import_module:
            return DependencyViolation(
                source_package=package_name,
                target_package="unknown",
                violation_type="import",
                location=f"{file_path}:{line_no}",
                severity="error",
                description=f"Relative import outside package boundary: {import_module}",
                suggestion="Use absolute imports or restructure code within package"
            )
        
        return None
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if module is part of Python standard library"""
        stdlib_modules = {
            'os', 'sys', 'json', 'yaml', 'datetime', 'typing', 'pathlib', 're',
            'subprocess', 'logging', 'collections', 'itertools', 'functools',
            'asyncio', 'concurrent', 'multiprocessing', 'threading', 'queue',
            'http', 'urllib', 'socket', 'email', 'hashlib', 'hmac', 'secrets',
            'uuid', 'base64', 'struct', 'pickle', 'sqlite3', 'csv', 'xml',
            'html', 'unittest', 'pytest', 'mock', 'tempfile', 'shutil',
        }
        
        root_module = module_name.split('.')[0]
        return root_module in stdlib_modules
    
    def _is_third_party(self, module_name: str) -> bool:
        """Check if module is a third-party package"""
        common_third_party = {
            'fastapi', 'pydantic', 'sqlalchemy', 'redis', 'celery', 'requests',
            'httpx', 'uvicorn', 'gunicorn', 'prometheus_client', 'structlog',
            'click', 'typer', 'alembic', 'pytest', 'numpy', 'pandas',
            'scikit-learn', 'tensorflow', 'torch', 'matplotlib', 'seaborn',
            'pillow', 'opencv', 'boto3', 'azure', 'google', 'kubernetes',
            'docker', 'yaml', 'toml', 'jinja2', 'marshmallow', 'cerberus',
        }
        
        root_module = module_name.split('.')[0]
        return root_module in common_third_party
    
    def _analyze_configuration_dependencies(self, package_dir: Path, package_name: str) -> List[DependencyViolation]:
        """Analyze configuration files for cross-package dependencies"""
        violations = []
        
        # Check environment files
        env_files = list(package_dir.glob(".env*"))
        for env_file in env_files:
            violations.extend(self._analyze_env_file(env_file, package_name))
        
        # Check YAML configuration files
        config_files = list(package_dir.rglob("*.yml")) + list(package_dir.rglob("*.yaml"))
        for config_file in config_files:
            violations.extend(self._analyze_config_file(config_file, package_name))
        
        return violations
    
    def _analyze_env_file(self, env_file: Path, package_name: str) -> List[DependencyViolation]:
        """Analyze environment file for dependencies"""
        violations = []
        
        try:
            content = env_file.read_text()
            lines = content.split('\n')
            
            for line_no, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check for references to other packages
                    if 'packages/' in line and package_name not in line:
                        violations.append(DependencyViolation(
                            source_package=package_name,
                            target_package="unknown",
                            violation_type="config",
                            location=f"{env_file}:{line_no}",
                            severity="warning",
                            description=f"Configuration reference to other package: {line}"
                        ))
        
        except Exception as e:
            self.logger.warning(f"Could not analyze {env_file}: {e}")
        
        return violations
    
    def _analyze_config_file(self, config_file: Path, package_name: str) -> List[DependencyViolation]:
        """Analyze YAML configuration file for dependencies"""
        violations = []
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Check for package references in configuration
            if 'packages/' in content:
                lines = content.split('\n')
                for line_no, line in enumerate(lines, 1):
                    if 'packages/' in line and package_name not in line:
                        violations.append(DependencyViolation(
                            source_package=package_name,
                            target_package="unknown",
                            violation_type="config",
                            location=f"{config_file}:{line_no}",
                            severity="warning",
                            description=f"Configuration dependency: {line.strip()}"
                        ))
        
        except Exception as e:
            self.logger.warning(f"Could not analyze {config_file}: {e}")
        
        return violations
    
    def _analyze_database_dependencies(self, package_dir: Path, package_name: str) -> List[DependencyViolation]:
        """Analyze database schema and migration dependencies"""
        violations = []
        
        # Check Alembic migrations
        migrations_dir = package_dir / "alembic" / "versions"
        if migrations_dir.exists():
            for migration_file in migrations_dir.glob("*.py"):
                violations.extend(self._analyze_migration_file(migration_file, package_name))
        
        # Check SQL files
        sql_files = list(package_dir.rglob("*.sql"))
        for sql_file in sql_files:
            violations.extend(self._analyze_sql_file(sql_file, package_name))
        
        return violations
    
    def _analyze_migration_file(self, migration_file: Path, package_name: str) -> List[DependencyViolation]:
        """Analyze database migration file for cross-package dependencies"""
        violations = []
        
        try:
            content = migration_file.read_text()
            
            # Check for foreign key references to other packages
            fk_pattern = r"ForeignKey\s*\(\s*['\"]([^'\"]+)['\"]"
            matches = re.findall(fk_pattern, content)
            
            for match in matches:
                if '.' in match:
                    table_name = match.split('.')[0]
                    # If table name suggests another package
                    if package_name not in table_name and any(
                        pkg in table_name for pkg in ['user', 'product', 'order', 'payment']
                    ):
                        violations.append(DependencyViolation(
                            source_package=package_name,
                            target_package="unknown",
                            violation_type="database",
                            location=str(migration_file),
                            severity="error",
                            description=f"Cross-package foreign key reference: {match}",
                            suggestion="Use event-driven approach instead of direct DB references"
                        ))
        
        except Exception as e:
            self.logger.warning(f"Could not analyze {migration_file}: {e}")
        
        return violations
    
    def _analyze_sql_file(self, sql_file: Path, package_name: str) -> List[DependencyViolation]:
        """Analyze SQL file for cross-package dependencies"""
        violations = []
        
        try:
            content = sql_file.read_text().lower()
            
            # Check for JOIN statements with tables from other packages
            join_pattern = r"join\s+([a-zA-Z_][a-zA-Z0-9_]*)"
            matches = re.findall(join_pattern, content)
            
            for table_name in matches:
                if package_name not in table_name:
                    violations.append(DependencyViolation(
                        source_package=package_name,
                        target_package="unknown",
                        violation_type="database",
                        location=str(sql_file),
                        severity="warning",
                        description=f"Potential cross-package table reference: {table_name}"
                    ))
        
        except Exception as e:
            self.logger.warning(f"Could not analyze {sql_file}: {e}")
        
        return violations
    
    def _analyze_api_dependencies(self, package_dir: Path, package_name: str) -> List[DependencyViolation]:
        """Analyze API call dependencies"""
        violations = []
        
        python_files = list(package_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                content = py_file.read_text()
                violations.extend(self._analyze_api_calls(py_file, content, package_name))
            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")
        
        return violations
    
    def _analyze_api_calls(self, file_path: Path, content: str, package_name: str) -> List[DependencyViolation]:
        """Analyze API calls in a file"""
        violations = []
        
        # Look for HTTP client calls to internal services
        api_patterns = [
            r"requests\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
            r"httpx\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
            r"client\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
        ]
        
        for pattern in api_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for method, url in matches:
                # Check if URL suggests internal service call
                if any(indicator in url.lower() for indicator in ['localhost', '127.0.0.1', 'internal', 'service']):
                    if package_name not in url:
                        violations.append(DependencyViolation(
                            source_package=package_name,
                            target_package="unknown",
                            violation_type="api_call",
                            location=str(file_path),
                            severity="warning",
                            description=f"Internal API call: {method.upper()} {url}",
                            suggestion="Use event-driven communication or message queues"
                        ))
        
        return violations
    
    def _check_required_components(self, package_dir: Path) -> List[str]:
        """Check for missing required components"""
        missing_components = []
        
        for component_type, patterns in self.required_components.items():
            component_found = False
            
            for pattern in patterns:
                if '*' in pattern:
                    # Use glob pattern
                    matches = list(package_dir.rglob(pattern))
                    if matches:
                        component_found = True
                        break
                else:
                    # Check specific file
                    if (package_dir / pattern).exists():
                        component_found = True
                        break
            
            if not component_found and self.config.get('require_all_components', True):
                missing_components.append(component_type)
        
        return missing_components
    
    def _analyze_dependencies_from_requirements(self, package_dir: Path) -> Tuple[List[str], List[str]]:
        """Analyze dependencies from requirements files and pyproject.toml"""
        external_deps = []
        internal_deps = []
        
        # Check pyproject.toml
        pyproject_file = package_dir / "pyproject.toml"
        if pyproject_file.exists():
            try:
                import tomli
                with open(pyproject_file, 'rb') as f:
                    data = tomli.load(f)
                
                # Get dependencies from project section
                deps = data.get('project', {}).get('dependencies', [])
                for dep in deps:
                    dep_name = dep.split('>=')[0].split('==')[0].split('[')[0].strip()
                    if dep_name.startswith('src.packages.'):
                        internal_deps.append(dep_name)
                    else:
                        external_deps.append(dep_name)
                
            except Exception as e:
                self.logger.warning(f"Could not parse {pyproject_file}: {e}")
        
        # Check requirements files
        req_files = list(package_dir.glob("requirements*.txt"))
        for req_file in req_files:
            try:
                content = req_file.read_text()
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep_name = line.split('>=')[0].split('==')[0].split('[')[0].strip()
                        if dep_name.startswith('src.packages.'):
                            internal_deps.append(dep_name)
                        else:
                            external_deps.append(dep_name)
            
            except Exception as e:
                self.logger.warning(f"Could not analyze {req_file}: {e}")
        
        return external_deps, internal_deps
    
    def _check_configuration_completeness(self, package_dir: Path) -> List[str]:
        """Check configuration completeness"""
        issues = []
        
        # Check for required configuration files
        required_configs = [
            '.env.example',
            'config/app.yml',
            'config/logging.yml',
        ]
        
        for config_path in required_configs:
            if not (package_dir / config_path).exists():
                issues.append(f"Missing configuration file: {config_path}")
        
        # Check Docker configuration
        if not (package_dir / "Dockerfile").exists():
            issues.append("Missing Dockerfile for containerization")
        
        if not (package_dir / "docker-compose.yml").exists():
            issues.append("Missing docker-compose.yml for local development")
        
        # Check CI/CD configuration
        github_workflows = package_dir / ".github" / "workflows"
        if not github_workflows.exists() or not list(github_workflows.glob("*.yml")):
            issues.append("Missing CI/CD workflow configuration")
        
        return issues
    
    def _calculate_test_coverage(self, package_dir: Path) -> float:
        """Calculate test coverage percentage"""
        # This is a simplified calculation
        # In practice, you'd run pytest --cov and parse the output
        
        python_files = list((package_dir / "src").rglob("*.py")) if (package_dir / "src").exists() else []
        test_files = list((package_dir / "tests").rglob("*.py")) if (package_dir / "tests").exists() else []
        
        if not python_files:
            return 0.0
        
        # Simple heuristic: ratio of test files to source files
        coverage_ratio = len(test_files) / len(python_files)
        return min(coverage_ratio * 100, 100.0)
    
    def _calculate_documentation_completeness(self, package_dir: Path) -> float:
        """Calculate documentation completeness percentage"""
        required_docs = ['README.md', 'CHANGELOG.md', 'docs/API.md', 'docs/ARCHITECTURE.md']
        existing_docs = sum(1 for doc in required_docs if (package_dir / doc).exists())
        
        return (existing_docs / len(required_docs)) * 100
    
    def _calculate_deployment_readiness(self, package_dir: Path) -> float:
        """Calculate deployment readiness percentage"""
        required_deployment_files = [
            'Dockerfile',
            'docker-compose.yml',
            'k8s/deployment.yml',
            'Makefile',
            'scripts/health-check.sh'
        ]
        
        existing_files = sum(1 for file_path in required_deployment_files 
                           if (package_dir / file_path).exists())
        
        return (existing_files / len(required_deployment_files)) * 100
    
    def _calculate_self_containment_score(
        self,
        violations: List[DependencyViolation],
        missing_components: List[str],
        external_deps: List[str],
        config_issues: List[str]
    ) -> float:
        """Calculate overall self-containment score"""
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points for violations
        error_violations = [v for v in violations if v.severity == 'error']
        warning_violations = [v for v in violations if v.severity == 'warning']
        
        score -= len(error_violations) * 10.0  # 10 points per error
        score -= len(warning_violations) * 3.0  # 3 points per warning
        
        # Deduct points for missing components
        score -= len(missing_components) * 5.0  # 5 points per missing component
        
        # Deduct points for excessive external dependencies
        max_external_deps = self.config.get('max_external_dependencies', 50)
        if len(external_deps) > max_external_deps:
            score -= (len(external_deps) - max_external_deps) * 1.0
        
        # Deduct points for configuration issues
        score -= len(config_issues) * 2.0  # 2 points per config issue
        
        return max(score, 0.0)
    
    def _analyze_cross_package_dependencies(self, package_analyses: Dict[str, PackageAnalysis]) -> None:
        """Analyze dependencies between packages"""
        # Create dependency graph
        dependency_graph = defaultdict(set)
        
        for package_name, analysis in package_analyses.items():
            for violation in analysis.violations:
                if violation.violation_type == 'import' and violation.target_package:
                    dependency_graph[package_name].add(violation.target_package)
        
        # Detect circular dependencies
        self._detect_circular_dependencies(dependency_graph, package_analyses)
        
        # Detect tightly coupled packages
        self._detect_tight_coupling(dependency_graph, package_analyses)
    
    def _detect_circular_dependencies(
        self, 
        dependency_graph: Dict[str, Set[str]], 
        package_analyses: Dict[str, PackageAnalysis]
    ) -> None:
        """Detect circular dependencies between packages"""
        
        def has_cycle(node: str, visited: Set[str], path: Set[str]) -> List[str]:
            if node in path:
                return [node]  # Cycle detected
            
            if node in visited:
                return []
            
            visited.add(node)
            path.add(node)
            
            for neighbor in dependency_graph.get(node, set()):
                cycle = has_cycle(neighbor, visited, path)
                if cycle:
                    if cycle[0] == node:
                        return cycle  # Complete cycle found
                    else:
                        return [node] + cycle
            
            path.remove(node)
            return []
        
        visited = set()
        for package_name in dependency_graph.keys():
            if package_name not in visited:
                cycle = has_cycle(package_name, visited, set())
                if cycle:
                    # Add circular dependency violation to all packages in cycle
                    for pkg in cycle:
                        if pkg in package_analyses:
                            package_analyses[pkg].violations.append(DependencyViolation(
                                source_package=pkg,
                                target_package=" -> ".join(cycle),
                                violation_type="circular_dependency",
                                location="package_structure",
                                severity="error",
                                description=f"Circular dependency detected: {' -> '.join(cycle)}",
                                suggestion="Refactor to remove circular dependencies using events or shared interfaces"
                            ))
    
    def _detect_tight_coupling(
        self, 
        dependency_graph: Dict[str, Set[str]], 
        package_analyses: Dict[str, PackageAnalysis]
    ) -> None:
        """Detect tightly coupled packages"""
        
        for package_name, dependencies in dependency_graph.items():
            if len(dependencies) > 3:  # Threshold for too many dependencies
                if package_name in package_analyses:
                    package_analyses[package_name].violations.append(DependencyViolation(
                        source_package=package_name,
                        target_package=", ".join(dependencies),
                        violation_type="tight_coupling",
                        location="package_structure",
                        severity="warning",
                        description=f"Package has too many dependencies: {len(dependencies)}",
                        suggestion="Consider breaking down the package or using event-driven patterns"
                    ))
    
    def _validate_duplicate_packages(self, package_dirs: List[Path]) -> List[DependencyViolation]:
        """Detect and validate duplicate packages based on naming patterns"""
        violations = []
        package_names = [d.name for d in package_dirs if d.is_dir()]
        
        self.logger.info("🔍 Checking for duplicate packages...")
        
        for package_name in package_names:
            # Check if this package name matches duplicate patterns
            for pattern in self.duplicate_patterns:
                match = re.match(pattern, package_name)
                if match:
                    base_name = match.group(1)
                    
                    # Check if the base package exists
                    potential_duplicates = []
                    for other_name in package_names:
                        if other_name == base_name or (other_name != package_name and re.match(rf'^{re.escape(base_name)}(_new|_v2|_backup|_old|_copy|_temp)$', other_name)):
                            potential_duplicates.append(other_name)
                    
                    if potential_duplicates:
                        violations.append(DependencyViolation(
                            source_package=package_name,
                            target_package=", ".join(potential_duplicates),
                            violation_type="duplicate_package",
                            location=f"src/packages/{package_name}",
                            severity="error",
                            description=f"Duplicate package detected: '{package_name}' appears to be a duplicate of '{base_name}' or conflicts with: {', '.join(potential_duplicates)}",
                            suggestion=f"Consolidate with the main package '{base_name}' or choose a distinct name. Remove unused duplicate packages."
                        ))
                        self.logger.warning(f"⚠️  Duplicate package detected: {package_name} -> {potential_duplicates}")
        
        # Also check for similar names that might be duplicates
        for i, package_name in enumerate(package_names):
            for j, other_name in enumerate(package_names):
                if i >= j:
                    continue
                    
                # Check for very similar names (edit distance)
                similarity = self._calculate_name_similarity(package_name, other_name)
                if similarity > 0.8:  # High similarity threshold
                    violations.append(DependencyViolation(
                        source_package=package_name,
                        target_package=other_name,
                        violation_type="similar_package_names",
                        location=f"src/packages/{package_name}",
                        severity="warning",
                        description=f"Package names are very similar: '{package_name}' and '{other_name}' (similarity: {similarity:.1%})",
                        suggestion="Consider using more distinct package names to avoid confusion."
                    ))
        
        return violations
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two package names using Levenshtein distance"""
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(name1.lower(), name2.lower())
        max_length = max(len(name1), len(name2))
        return 1.0 - (distance / max_length) if max_length > 0 else 0.0
    
    def _generate_independence_report(self, package_analyses: Dict[str, PackageAnalysis]) -> None:
        """Generate independence validation report"""
        report_path = self.repository_root / "reports" / "package_independence_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare report data
        report_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'repository_root': str(self.repository_root),
            'configuration': self.config,
            'summary': self._generate_summary(package_analyses),
            'packages': {
                name: asdict(analysis) 
                for name, analysis in package_analyses.items()
            }
        }
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(package_analyses, report_path.with_suffix('.md'))
        
        self.logger.info(f"📊 Independence report generated: {report_path}")
    
    def _generate_summary(self, package_analyses: Dict[str, PackageAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_packages = len(package_analyses)
        self_contained_packages = sum(1 for analysis in package_analyses.values() 
                                    if analysis.is_self_contained)
        
        avg_score = sum(analysis.self_containment_score for analysis in package_analyses.values()) / total_packages if total_packages > 0 else 0
        
        total_violations = sum(len(analysis.violations) for analysis in package_analyses.values())
        error_violations = sum(
            len([v for v in analysis.violations if v.severity == 'error'])
            for analysis in package_analyses.values()
        )
        
        return {
            'total_packages': total_packages,
            'self_contained_packages': self_contained_packages,
            'self_containment_percentage': (self_contained_packages / total_packages * 100) if total_packages > 0 else 0,
            'average_self_containment_score': avg_score,
            'total_violations': total_violations,
            'error_violations': error_violations,
            'packages_with_errors': sum(1 for analysis in package_analyses.values() 
                                      if any(v.severity == 'error' for v in analysis.violations)),
        }
    
    def _generate_markdown_report(self, package_analyses: Dict[str, PackageAnalysis], report_path: Path) -> None:
        """Generate markdown report"""
        summary = self._generate_summary(package_analyses)
        
        report_content = f"""# Package Independence Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Packages**: {summary['total_packages']}
- **Self-Contained Packages**: {summary['self_contained_packages']} ({summary['self_containment_percentage']:.1f}%)
- **Average Self-Containment Score**: {summary['average_self_containment_score']:.1f}
- **Total Violations**: {summary['total_violations']}
- **Error Violations**: {summary['error_violations']}
- **Packages with Errors**: {summary['packages_with_errors']}

## Package Analysis

"""
        
        for package_name, analysis in sorted(package_analyses.items()):
            status_emoji = "✅" if analysis.is_self_contained else "❌"
            
            report_content += f"""
### {status_emoji} {package_name}

- **Self-Contained**: {'Yes' if analysis.is_self_contained else 'No'}
- **Score**: {analysis.self_containment_score:.1f}/100
- **Test Coverage**: {analysis.test_coverage:.1f}%
- **Documentation**: {analysis.documentation_completeness:.1f}%
- **Deployment Ready**: {analysis.deployment_readiness:.1f}%

"""
            
            if analysis.violations:
                report_content += "#### Violations\n\n"
                for violation in analysis.violations:
                    severity_emoji = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(violation.severity, "⚪")
                    report_content += f"- {severity_emoji} **{violation.violation_type}**: {violation.description}\n"
                    if violation.suggestion:
                        report_content += f"  - *Suggestion: {violation.suggestion}*\n"
                report_content += "\n"
            
            if analysis.missing_components:
                report_content += f"#### Missing Components\n\n"
                for component in analysis.missing_components:
                    report_content += f"- {component}\n"
                report_content += "\n"
            
            if analysis.configuration_issues:
                report_content += f"#### Configuration Issues\n\n"
                for issue in analysis.configuration_issues:
                    report_content += f"- {issue}\n"
                report_content += "\n"
        
        # Save markdown report
        report_path.write_text(report_content)
        self.logger.info(f"📄 Markdown report generated: {report_path}")
    
    def enforce_independence(self, package_analyses: Dict[str, PackageAnalysis]) -> bool:
        """Enforce package independence based on configuration"""
        enforcement_level = self.config.get('enforcement_level', 'strict')
        
        if enforcement_level == 'lenient':
            return True  # Always pass in lenient mode
        
        packages_with_errors = [
            name for name, analysis in package_analyses.items()
            if any(v.severity == 'error' for v in analysis.violations)
        ]
        
        if packages_with_errors:
            self.logger.error(f"❌ Package independence enforcement failed")
            self.logger.error(f"Packages with errors: {', '.join(packages_with_errors)}")
            
            if enforcement_level == 'strict':
                return False
            elif enforcement_level == 'moderate':
                # Allow some errors but warn
                self.logger.warning("🟡 Continuing with warnings due to moderate enforcement level")
                return True
        
        # Check self-containment scores
        required_score = self.config.get('required_self_containment_score', 85.0)
        failing_packages = [
            name for name, analysis in package_analyses.items()
            if analysis.self_containment_score < required_score
        ]
        
        if failing_packages and enforcement_level == 'strict':
            self.logger.error(f"❌ Packages below required self-containment score ({required_score}): {', '.join(failing_packages)}")
            return False
        
        self.logger.info("✅ Package independence enforcement passed")
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate package independence")
    parser.add_argument('--repository-root', type=str, default=".", 
                       help="Repository root directory")
    parser.add_argument('--package', type=str, help="Validate specific package")
    parser.add_argument('--enforce', action='store_true', 
                       help="Enforce independence (exit with error if violations)")
    parser.add_argument('--config', type=str, 
                       help="Configuration file path")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger('package_independence_validator').setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = PackageIndependenceValidator(Path(args.repository_root))
    
    if args.package:
        # Validate specific package
        package_dir = validator.packages_dir / args.package
        if not package_dir.exists():
            print(f"❌ Package directory not found: {package_dir}")
            return 1
        
        analysis = validator.validate_package_independence(package_dir)
        print(json.dumps(asdict(analysis), indent=2))
        
        if args.enforce and not analysis.is_self_contained:
            return 1
    
    else:
        # Validate all packages
        package_analyses = validator.validate_all_packages()
        
        if args.enforce:
            if not validator.enforce_independence(package_analyses):
                return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())