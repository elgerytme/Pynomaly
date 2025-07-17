#!/usr/bin/env python3
"""
Repository Separation Analysis Tool
Analyzes package dependencies and evaluates readiness for repository separation.
"""

import os
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import re

class PackageAnalyzer:
    """Analyzes package structure and dependencies for repository separation."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.packages = {}
        self.dependencies = defaultdict(set)
        self.external_dependencies = defaultdict(set)
        self.internal_dependencies = defaultdict(set)
        self.package_files = defaultdict(list)
        
    def discover_packages(self) -> Dict[str, Path]:
        """Discover all packages in the repository."""
        packages = {}
        
        # Look for packages with pyproject.toml or __init__.py
        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)
            
            # Skip certain directories
            skip_dirs = {'__pycache__', '.git', 'node_modules', '.pytest_cache', 
                        'build', 'dist', '.venv', 'venv', 'htmlcov', 'migration-backup'}
            
            # Remove skip_dirs from dirs to prevent os.walk from entering them
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            if 'pyproject.toml' in files or '__init__.py' in files:
                # Determine package name from path
                rel_path = root_path.relative_to(self.base_path)
                package_name = str(rel_path).replace(os.sep, '.')
                
                # Clean up package name
                if package_name.startswith('.'):
                    package_name = package_name[1:]
                
                packages[package_name] = root_path
                
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
            print(f"Error analyzing {file_path}: {e}")
            return set(), set()
    
    def categorize_import(self, import_name: str, package_name: str) -> str:
        """Categorize an import as internal, external, or standard library."""
        # Standard library modules (partial list)
        stdlib_modules = {
            'os', 'sys', 'json', 'ast', 'pathlib', 'collections', 'typing',
            'dataclasses', 'datetime', 'time', 'logging', 'unittest', 'pytest',
            'functools', 'itertools', 'operator', 'math', 're', 'copy',
            'pickle', 'csv', 'sqlite3', 'http', 'urllib', 'email', 'io',
            'tempfile', 'shutil', 'subprocess', 'threading', 'multiprocessing'
        }
        
        # Split import to get root module
        root_module = import_name.split('.')[0]
        
        if root_module in stdlib_modules:
            return 'stdlib'
        elif import_name.startswith('src.packages.') or any(import_name.startswith(pkg) for pkg in self.packages.keys()):
            return 'internal'
        else:
            return 'external'
    
    def analyze_package_dependencies(self, package_name: str, package_path: Path):
        """Analyze dependencies for a specific package."""
        python_files = []
        
        # Find all Python files in package
        for root, dirs, files in os.walk(package_path):
            # Skip certain directories
            skip_dirs = {'__pycache__', '.git', 'node_modules', '.pytest_cache', 
                        'build', 'dist', '.venv', 'venv', 'htmlcov', 'migration-backup'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
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
            category = self.categorize_import(import_name, package_name)
            
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
            # Reduce score based on number of internal dependencies
            internal_count = len(internal_deps)
            if internal_count > 10:
                score -= 5
                blockers['high_internal_deps'] = f"{internal_count} internal dependencies"
            elif internal_count > 5:
                score -= 3
                blockers['medium_internal_deps'] = f"{internal_count} internal dependencies"
            else:
                score -= 1
                blockers['low_internal_deps'] = f"{internal_count} internal dependencies"
        
        # Check for circular dependencies
        circular_deps = self.find_circular_dependencies(package_name)
        if circular_deps:
            score -= 4
            blockers['circular_deps'] = f"Circular dependencies: {', '.join(circular_deps)}"
        
        # Check package structure
        package_path = self.packages.get(package_name)
        if package_path:
            # Check for proper structure (domain, application, infrastructure layers)
            has_domain = any('domain' in str(f) for f in self.package_files.get(package_name, []))
            has_application = any('application' in str(f) for f in self.package_files.get(package_name, []))
            has_infrastructure = any('infrastructure' in str(f) for f in self.package_files.get(package_name, []))
            
            if not (has_domain and has_application):
                score -= 2
                blockers['incomplete_layers'] = "Missing domain/application layers"
            
            # Check for pyproject.toml
            if not (package_path / 'pyproject.toml').exists():
                score -= 1
                blockers['no_pyproject'] = "Missing pyproject.toml"
            
            # Check for tests
            has_tests = any('test' in str(f) for f in self.package_files.get(package_name, []))
            if not has_tests:
                score -= 1
                blockers['no_tests'] = "Missing tests"
        
        # Ensure score is between 1-10
        score = max(1, min(10, score))
        
        return score, blockers
    
    def find_circular_dependencies(self, package_name: str) -> List[str]:
        """Find circular dependencies for a package."""
        visited = set()
        rec_stack = set()
        circular = []
        
        def dfs(pkg):
            if pkg in rec_stack:
                return True  # Cycle detected
            if pkg in visited:
                return False
            
            visited.add(pkg)
            rec_stack.add(pkg)
            
            # Check internal dependencies
            for dep in self.internal_dependencies.get(pkg, set()):
                # Convert import to package name
                dep_pkg = self.import_to_package(dep)
                if dep_pkg and dep_pkg in self.packages:
                    if dfs(dep_pkg):
                        circular.append(dep_pkg)
            
            rec_stack.remove(pkg)
            return False
        
        dfs(package_name)
        return circular
    
    def import_to_package(self, import_name: str) -> Optional[str]:
        """Convert import name to package name."""
        for pkg_name in self.packages.keys():
            if import_name.startswith(pkg_name.replace('.', '.')):
                return pkg_name
        return None
    
    def create_dependency_graph(self) -> Dict[str, Dict[str, List[str]]]:
        """Create a dependency graph."""
        graph = {}
        
        for package_name in self.packages.keys():
            internal_deps = []
            external_deps = []
            
            for dep in self.internal_dependencies.get(package_name, set()):
                dep_pkg = self.import_to_package(dep)
                if dep_pkg:
                    internal_deps.append(dep_pkg)
            
            external_deps = list(self.external_dependencies.get(package_name, set()))
            
            graph[package_name] = {
                'internal': internal_deps,
                'external': external_deps
            }
        
        return graph
    
    def analyze_all_packages(self) -> Dict[str, Dict]:
        """Analyze all packages for separation readiness."""
        print("🔍 Discovering packages...")
        self.packages = self.discover_packages()
        
        print(f"📦 Found {len(self.packages)} packages")
        
        results = {}
        
        for package_name, package_path in self.packages.items():
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
                'has_tests': any('test' in str(f) for f in self.package_files.get(package_name, []))
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
        
        report.append(f"\n## Summary")
        report.append(f"- Total packages: {total_packages}")
        report.append(f"- Ready for separation (score ≥ 8): {ready_packages}")
        report.append(f"- Partially ready (score 5-7): {partially_ready}")
        report.append(f"- Not ready (score < 5): {not_ready}")
        
        # Dependency graph
        graph = self.create_dependency_graph()
        report.append(f"\n## Dependency Graph")
        for pkg, deps in graph.items():
            if deps['internal']:
                report.append(f"- {pkg} → {', '.join(deps['internal'])}")
        
        # Package analysis
        report.append(f"\n## Package Analysis")
        
        # Sort by separation score (descending)
        sorted_packages = sorted(results.items(), key=lambda x: x[1]['separation_score'], reverse=True)
        
        for package_name, data in sorted_packages:
            score = data['separation_score']
            status = "✅ READY" if score >= 8 else "⚠️ PARTIAL" if score >= 5 else "❌ NOT READY"
            
            report.append(f"\n### {package_name} - {status} (Score: {score}/10)")
            report.append(f"- Files: {data['file_count']}")
            report.append(f"- Internal dependencies: {data['internal_dep_count']}")
            report.append(f"- External dependencies: {data['external_dep_count']}")
            report.append(f"- Has pyproject.toml: {data['has_pyproject']}")
            report.append(f"- Has tests: {data['has_tests']}")
            
            if data['blockers']:
                report.append("- **Blockers:**")
                for blocker, description in data['blockers'].items():
                    report.append(f"  - {blocker}: {description}")
            
            if data['internal_dependencies']:
                report.append("- **Internal dependencies:**")
                for dep in data['internal_dependencies'][:5]:  # Show first 5
                    report.append(f"  - {dep}")
                if len(data['internal_dependencies']) > 5:
                    report.append(f"  - ... and {len(data['internal_dependencies']) - 5} more")
        
        # Recommendations
        report.append(f"\n## Recommendations for Repository Separation")
        
        report.append(f"\n### High Priority (Ready packages - score ≥ 8)")
        ready_packages_list = [pkg for pkg, data in results.items() if data['separation_score'] >= 8]
        if ready_packages_list:
            for pkg in ready_packages_list:
                report.append(f"- **{pkg}**: Can be separated immediately")
        else:
            report.append("- No packages are fully ready for separation")
        
        report.append(f"\n### Medium Priority (Partially ready - score 5-7)")
        partial_packages = [pkg for pkg, data in results.items() if 5 <= data['separation_score'] < 8]
        for pkg in partial_packages:
            blockers = results[pkg]['blockers']
            report.append(f"- **{pkg}**: Address {len(blockers)} blockers")
            for blocker, desc in list(blockers.items())[:2]:  # Show top 2 blockers
                report.append(f"  - {blocker}: {desc}")
        
        report.append(f"\n### Low Priority (Not ready - score < 5)")
        not_ready_packages = [pkg for pkg, data in results.items() if data['separation_score'] < 5]
        for pkg in not_ready_packages:
            report.append(f"- **{pkg}**: Requires significant refactoring")
        
        # Shared infrastructure analysis
        report.append(f"\n## Shared Infrastructure Analysis")
        
        # Count common external dependencies
        all_external_deps = []
        for data in results.values():
            all_external_deps.extend(data['external_dependencies'])
        
        common_deps = Counter(all_external_deps)
        report.append(f"\n### Most Common External Dependencies")
        for dep, count in common_deps.most_common(10):
            report.append(f"- {dep}: used by {count} packages")
        
        # Circular dependencies
        circular_packages = [pkg for pkg, data in results.items() if 'circular_deps' in data['blockers']]
        if circular_packages:
            report.append(f"\n### Circular Dependencies (Critical)")
            for pkg in circular_packages:
                report.append(f"- {pkg}: {results[pkg]['blockers']['circular_deps']}")
        
        return "\n".join(report)

def main():
    """Main function to run the analysis."""
    base_path = "/mnt/c/Users/andre/Pynomaly/src/packages"
    
    if not os.path.exists(base_path):
        print(f"❌ Path does not exist: {base_path}")
        sys.exit(1)
    
    analyzer = PackageAnalyzer(base_path)
    results = analyzer.analyze_all_packages()
    
    # Generate report
    report = analyzer.generate_report(results)
    
    # Save results
    with open(f"{base_path}/separation_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(f"{base_path}/separation_analysis_report.md", "w") as f:
        f.write(report)
    
    print(f"\n📊 Analysis complete!")
    print(f"📄 Report saved to: {base_path}/separation_analysis_report.md")
    print(f"📊 Raw results saved to: {base_path}/separation_analysis_results.json")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    total_packages = len(results)
    ready_packages = sum(1 for r in results.values() if r['separation_score'] >= 8)
    partially_ready = sum(1 for r in results.values() if 5 <= r['separation_score'] < 8)
    not_ready = sum(1 for r in results.values() if r['separation_score'] < 5)
    
    print(f"📦 Total packages: {total_packages}")
    print(f"✅ Ready for separation (score ≥ 8): {ready_packages}")
    print(f"⚠️  Partially ready (score 5-7): {partially_ready}")
    print(f"❌ Not ready (score < 5): {not_ready}")
    
    # Show top 5 packages by score
    print(f"\n🏆 Top 5 packages by separation score:")
    sorted_packages = sorted(results.items(), key=lambda x: x[1]['separation_score'], reverse=True)
    for i, (pkg, data) in enumerate(sorted_packages[:5]):
        print(f"{i+1}. {pkg}: {data['separation_score']}/10")

if __name__ == "__main__":
    main()