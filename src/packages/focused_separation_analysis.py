#!/usr/bin/env python3
"""
Focused Repository Separation Analysis Tool
Analyzes main packages for repository separation readiness.
"""

import os
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter

class FocusedPackageAnalyzer:
    """Analyzes main packages for repository separation."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.main_packages = {}
        self.dependencies = defaultdict(set)
        self.external_dependencies = defaultdict(set)
        self.internal_dependencies = defaultdict(set)
        self.package_files = defaultdict(list)
        
    def identify_main_packages(self) -> Dict[str, Path]:
        """Identify main packages (not sub-packages)."""
        packages = {}
        
        # Define main package directories
        main_dirs = [
            'ai/machine_learning',
            'ai/mlops', 
            'data/anomaly_detection',
            'data/data_observability',
            'data/data_platform',
            'formal_sciences/mathematics',
            'ops/people_ops',
            'software/core',
            'software/domain_library',
            'software/enterprise',
            'software/interfaces',
            'software/mobile',
            'software/services'
        ]
        
        for main_dir in main_dirs:
            package_path = self.base_path / main_dir
            if package_path.exists():
                package_name = main_dir.replace('/', '.')
                packages[package_name] = package_path
        
        return packages
    
    def analyze_python_file(self, file_path: Path) -> Tuple[Set[str], Set[str]]:
        """Analyze a Python file for imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = set()
            from_imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        from_imports.add(node.module)
            
            return imports, from_imports
            
        except Exception as e:
            # Skip files with syntax errors
            return set(), set()
    
    def categorize_import(self, import_name: str) -> str:
        """Categorize an import as internal, external, or standard library."""
        # Standard library modules
        stdlib_modules = {
            'os', 'sys', 'json', 'ast', 'pathlib', 'collections', 'typing',
            'dataclasses', 'datetime', 'time', 'logging', 'unittest', 'pytest',
            'functools', 'itertools', 'operator', 'math', 're', 'copy',
            'pickle', 'csv', 'sqlite3', 'http', 'urllib', 'email', 'io',
            'tempfile', 'shutil', 'subprocess', 'threading', 'multiprocessing',
            'abc', 'enum', 'contextlib', 'warnings', 'decimal', 'fractions',
            'statistics', 'random', 'hashlib', 'secrets', 'base64', 'uuid',
            'concurrent', 'asyncio', 'socket', 'ssl', 'ipaddress', 'mimetypes'
        }
        
        # Third-party common packages
        third_party_modules = {
            'numpy', 'pandas', 'scikit-learn', 'sklearn', 'tensorflow', 'torch',
            'flask', 'django', 'fastapi', 'requests', 'boto3', 'redis',
            'sqlalchemy', 'pydantic', 'click', 'typer', 'rich', 'pytest',
            'matplotlib', 'seaborn', 'plotly', 'streamlit', 'dash',
            'pyspark', 'dask', 'ray', 'celery', 'docker', 'kubernetes',
            'jinja2', 'marshmallow', 'werkzeug', 'gunicorn', 'uvicorn'
        }
        
        root_module = import_name.split('.')[0]
        
        if root_module in stdlib_modules:
            return 'stdlib'
        elif root_module in third_party_modules:
            return 'external'
        elif any(import_name.startswith(pkg.replace('.', '.')) for pkg in self.main_packages.keys()):
            return 'internal'
        elif import_name.startswith('src.packages.'):
            return 'internal'
        else:
            return 'external'
    
    def analyze_package_dependencies(self, package_name: str, package_path: Path):
        """Analyze dependencies for a specific package."""
        python_files = []
        
        # Find all Python files in package (limit depth to avoid too many files)
        for root, dirs, files in os.walk(package_path):
            # Skip certain directories
            skip_dirs = {'__pycache__', '.git', 'node_modules', '.pytest_cache', 
                        'build', 'dist', '.venv', 'venv', 'htmlcov', 'migration-backup',
                        'docs', 'examples', 'scripts', 'training', 'reporting'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            # Limit depth to avoid too many nested packages
            depth = len(Path(root).relative_to(package_path).parts)
            if depth > 4:  # Limit depth
                continue
                
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        # Limit number of files to analyze
        if len(python_files) > 100:
            python_files = python_files[:100]
            
        self.package_files[package_name] = python_files
        
        all_imports = set()
        all_from_imports = set()
        
        # Analyze each Python file
        for py_file in python_files:
            imports, from_imports = self.analyze_python_file(py_file)
            all_imports.update(imports)
            all_from_imports.update(from_imports)
        
        # Categorize all imports
        for import_name in all_imports | all_from_imports:
            category = self.categorize_import(import_name)
            
            if category == 'external':
                self.external_dependencies[package_name].add(import_name)
            elif category == 'internal':
                self.internal_dependencies[package_name].add(import_name)
        
        # Update general dependencies
        self.dependencies[package_name] = all_imports | all_from_imports
    
    def calculate_separation_score(self, package_name: str) -> Tuple[int, Dict[str, str]]:
        """Calculate separation readiness score (1-10) for a package."""
        score = 10
        blockers = {}
        
        # Check for internal dependencies
        internal_deps = self.internal_dependencies.get(package_name, set())
        if internal_deps:
            internal_count = len(internal_deps)
            if internal_count > 10:
                score -= 6
                blockers['high_internal_deps'] = f"{internal_count} internal dependencies"
            elif internal_count > 5:
                score -= 4
                blockers['medium_internal_deps'] = f"{internal_count} internal dependencies"
            else:
                score -= 2
                blockers['low_internal_deps'] = f"{internal_count} internal dependencies"
        
        # Check package structure
        package_path = self.main_packages.get(package_name)
        if package_path:
            # Check for proper clean architecture structure
            has_domain = any('domain' in str(f) for f in self.package_files.get(package_name, []))
            has_application = any('application' in str(f) for f in self.package_files.get(package_name, []))
            has_infrastructure = any('infrastructure' in str(f) for f in self.package_files.get(package_name, []))
            
            architecture_score = 0
            if has_domain:
                architecture_score += 1
            if has_application:
                architecture_score += 1
            if has_infrastructure:
                architecture_score += 1
            
            if architecture_score < 2:
                score -= 2
                blockers['incomplete_architecture'] = f"Missing clean architecture layers ({architecture_score}/3)"
            
            # Check for pyproject.toml
            if not (package_path / 'pyproject.toml').exists():
                score -= 1
                blockers['no_pyproject'] = "Missing pyproject.toml"
            
            # Check for tests
            has_tests = any('test' in str(f) for f in self.package_files.get(package_name, []))
            if not has_tests:
                score -= 1
                blockers['no_tests'] = "Missing tests"
            
            # Check for README
            readme_files = ['README.md', 'README.rst', 'README.txt']
            has_readme = any((package_path / readme).exists() for readme in readme_files)
            if not has_readme:
                score -= 1
                blockers['no_readme'] = "Missing README"
        
        # Ensure score is between 1-10
        score = max(1, min(10, score))
        
        return score, blockers
    
    def analyze_all_packages(self) -> Dict[str, Dict]:
        """Analyze all main packages for separation readiness."""
        print("🔍 Identifying main packages...")
        self.main_packages = self.identify_main_packages()
        
        print(f"📦 Found {len(self.main_packages)} main packages")
        
        results = {}
        
        for package_name, package_path in self.main_packages.items():
            print(f"🔍 Analyzing package: {package_name}")
            
            # Analyze dependencies
            self.analyze_package_dependencies(package_name, package_path)
            
            # Calculate separation score
            score, blockers = self.calculate_separation_score(package_name)
            
            # Package statistics
            file_count = len(self.package_files.get(package_name, []))
            internal_dep_count = len(self.internal_dependencies.get(package_name, set()))
            external_dep_count = len(self.external_dependencies.get(package_name, set()))
            
            results[package_name] = {
                'path': str(package_path),
                'separation_score': score,
                'blockers': blockers,
                'file_count': file_count,
                'internal_dependencies': list(self.internal_dependencies.get(package_name, set())),
                'external_dependencies': list(self.external_dependencies.get(package_name, set())),
                'internal_dep_count': internal_dep_count,
                'external_dep_count': external_dep_count,
                'has_pyproject': (package_path / 'pyproject.toml').exists(),
                'has_tests': any('test' in str(f) for f in self.package_files.get(package_name, [])),
                'has_readme': any((package_path / readme).exists() for readme in ['README.md', 'README.rst', 'README.txt'])
            }
        
        return results
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate a comprehensive separation readiness report."""
        report = []
        report.append("# Repository Separation Analysis Report")
        report.append("=" * 50)
        
        # Summary statistics
        total_packages = len(results)
        ready_packages = sum(1 for r in results.values() if r['separation_score'] >= 8)
        partially_ready = sum(1 for r in results.values() if 5 <= r['separation_score'] < 8)
        not_ready = sum(1 for r in results.values() if r['separation_score'] < 5)
        
        report.append(f"\n## Executive Summary")
        report.append(f"- **Total main packages analyzed**: {total_packages}")
        report.append(f"- **Ready for separation** (score ≥ 8): {ready_packages} ({ready_packages/total_packages*100:.1f}%)")
        report.append(f"- **Partially ready** (score 5-7): {partially_ready} ({partially_ready/total_packages*100:.1f}%)")
        report.append(f"- **Not ready** (score < 5): {not_ready} ({not_ready/total_packages*100:.1f}%)")
        
        # Package analysis
        report.append(f"\n## Package Analysis")
        
        # Sort by separation score (descending)
        sorted_packages = sorted(results.items(), key=lambda x: x[1]['separation_score'], reverse=True)
        
        for package_name, data in sorted_packages:
            score = data['separation_score']
            if score >= 8:
                status = "✅ READY FOR SEPARATION"
            elif score >= 5:
                status = "⚠️ PARTIALLY READY"
            else:
                status = "❌ NOT READY"
            
            report.append(f"\n### {package_name}")
            report.append(f"**Status**: {status} (Score: {score}/10)")
            report.append(f"- **Path**: `{data['path']}`")
            report.append(f"- **Files analyzed**: {data['file_count']}")
            report.append(f"- **Internal dependencies**: {data['internal_dep_count']}")
            report.append(f"- **External dependencies**: {data['external_dep_count']}")
            report.append(f"- **Has pyproject.toml**: {data['has_pyproject']}")
            report.append(f"- **Has tests**: {data['has_tests']}")
            report.append(f"- **Has README**: {data['has_readme']}")
            
            if data['blockers']:
                report.append("- **Blockers for separation:**")
                for blocker, description in data['blockers'].items():
                    report.append(f"  - {blocker}: {description}")
            
            # Show key internal dependencies
            if data['internal_dependencies']:
                report.append("- **Key internal dependencies:**")
                for dep in list(data['internal_dependencies'])[:5]:  # Show first 5
                    report.append(f"  - `{dep}`")
                if len(data['internal_dependencies']) > 5:
                    report.append(f"  - ... and {len(data['internal_dependencies']) - 5} more")
        
        # Dependency analysis
        report.append(f"\n## Dependency Analysis")
        
        # Internal dependency matrix
        report.append(f"\n### Internal Dependencies Matrix")
        report.append("| Package | Depends On |")
        report.append("|---------|------------|")
        
        for package_name, data in sorted_packages:
            internal_deps = data['internal_dependencies']
            if internal_deps:
                # Filter to only main packages
                main_deps = [dep for dep in internal_deps if any(dep.startswith(pkg.replace('.', '.')) for pkg in self.main_packages.keys())]
                if main_deps:
                    deps_str = ", ".join(main_deps[:3])  # Show first 3
                    if len(main_deps) > 3:
                        deps_str += f" (+{len(main_deps)-3} more)"
                    report.append(f"| {package_name} | {deps_str} |")
                else:
                    report.append(f"| {package_name} | (no main package deps) |")
            else:
                report.append(f"| {package_name} | (none) |")
        
        # Common external dependencies
        report.append(f"\n### Common External Dependencies")
        all_external_deps = []
        for data in results.values():
            all_external_deps.extend(data['external_dependencies'])
        
        common_deps = Counter(all_external_deps)
        report.append("| Dependency | Used by # packages |")
        report.append("|------------|-------------------|")
        
        for dep, count in common_deps.most_common(10):
            report.append(f"| {dep} | {count} |")
        
        # Recommendations
        report.append(f"\n## Recommendations")
        
        report.append(f"\n### Immediate Actions (Ready packages)")
        ready_packages_list = [pkg for pkg, data in results.items() if data['separation_score'] >= 8]
        if ready_packages_list:
            for pkg in ready_packages_list:
                report.append(f"- **{pkg}**: Can be moved to separate repository immediately")
                report.append(f"  - Create new repository: `{pkg.replace('.', '-')}`")
                report.append(f"  - Ensure CI/CD pipeline is configured")
                report.append(f"  - Update documentation and references")
        else:
            report.append("- No packages are fully ready for immediate separation")
        
        report.append(f"\n### Short-term Actions (Partially ready packages)")
        partial_packages = [(pkg, data) for pkg, data in results.items() if 5 <= data['separation_score'] < 8]
        for pkg, data in partial_packages:
            report.append(f"- **{pkg}** (Score: {data['separation_score']}/10):")
            for blocker, desc in data['blockers'].items():
                if blocker == 'high_internal_deps' or blocker == 'medium_internal_deps':
                    report.append(f"  - Reduce internal dependencies: {desc}")
                elif blocker == 'incomplete_architecture':
                    report.append(f"  - Implement clean architecture: {desc}")
                elif blocker == 'no_pyproject':
                    report.append(f"  - Add pyproject.toml with proper dependencies")
                elif blocker == 'no_tests':
                    report.append(f"  - Add comprehensive test suite")
                elif blocker == 'no_readme':
                    report.append(f"  - Add README with usage instructions")
        
        report.append(f"\n### Long-term Actions (Not ready packages)")
        not_ready_packages = [(pkg, data) for pkg, data in results.items() if data['separation_score'] < 5]
        for pkg, data in not_ready_packages:
            report.append(f"- **{pkg}** (Score: {data['separation_score']}/10): Requires major refactoring")
            report.append(f"  - Address {len(data['blockers'])} critical blockers")
            report.append(f"  - Consider breaking into smaller, more focused packages")
        
        # Shared infrastructure recommendations
        report.append(f"\n### Shared Infrastructure Strategy")
        report.append("For packages that will be separated:")
        report.append("- **Shared utilities**: Create shared utility libraries for common functionality")
        report.append("- **Interface definitions**: Maintain interfaces package for cross-package communication")
        report.append("- **CI/CD templates**: Create reusable CI/CD templates for consistent deployment")
        report.append("- **Documentation standards**: Establish consistent documentation standards")
        report.append("- **Dependency management**: Use dependency management tools to track versions")
        
        return "\n".join(report)

def main():
    """Main function to run the focused analysis."""
    base_path = "/mnt/c/Users/andre/Pynomaly/src/packages"
    
    if not os.path.exists(base_path):
        print(f"❌ Path does not exist: {base_path}")
        sys.exit(1)
    
    analyzer = FocusedPackageAnalyzer(base_path)
    results = analyzer.analyze_all_packages()
    
    # Generate report
    report = analyzer.generate_report(results)
    
    # Save results
    with open(f"{base_path}/focused_separation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(f"{base_path}/focused_separation_report.md", "w") as f:
        f.write(report)
    
    print(f"\n📊 Analysis complete!")
    print(f"📄 Report saved to: {base_path}/focused_separation_report.md")
    print(f"📊 Raw results saved to: {base_path}/focused_separation_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("REPOSITORY SEPARATION ANALYSIS SUMMARY")
    print("="*60)
    
    total_packages = len(results)
    ready_packages = sum(1 for r in results.values() if r['separation_score'] >= 8)
    partially_ready = sum(1 for r in results.values() if 5 <= r['separation_score'] < 8)
    not_ready = sum(1 for r in results.values() if r['separation_score'] < 5)
    
    print(f"📦 Total packages analyzed: {total_packages}")
    print(f"✅ Ready for separation (score ≥ 8): {ready_packages} ({ready_packages/total_packages*100:.1f}%)")
    print(f"⚠️  Partially ready (score 5-7): {partially_ready} ({partially_ready/total_packages*100:.1f}%)")
    print(f"❌ Not ready (score < 5): {not_ready} ({not_ready/total_packages*100:.1f}%)")
    
    # Show packages by category
    print(f"\n🏆 PACKAGES BY READINESS:")
    sorted_packages = sorted(results.items(), key=lambda x: x[1]['separation_score'], reverse=True)
    
    if ready_packages > 0:
        print(f"\n✅ READY FOR SEPARATION:")
        for pkg, data in sorted_packages:
            if data['separation_score'] >= 8:
                print(f"   • {pkg}: {data['separation_score']}/10")
    
    if partially_ready > 0:
        print(f"\n⚠️  PARTIALLY READY:")
        for pkg, data in sorted_packages:
            if 5 <= data['separation_score'] < 8:
                print(f"   • {pkg}: {data['separation_score']}/10")
    
    if not_ready > 0:
        print(f"\n❌ NOT READY:")
        for pkg, data in sorted_packages:
            if data['separation_score'] < 5:
                print(f"   • {pkg}: {data['separation_score']}/10")

if __name__ == "__main__":
    main()