#!/usr/bin/env python3
"""
Dependency Optimization and Management Framework

Implements automated dependency optimization, vulnerability scanning,
and version management across all packages.

Issue: #827 - Optimize Package Dependencies
"""

import os
import sys
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import toml
import requests
from packaging import version


@dataclass
class DependencyInfo:
    """Dependency information structure"""
    name: str
    current_version: str
    latest_version: Optional[str] = None
    pinned_version: Optional[str] = None
    vulnerabilities: List[str] = None
    size_mb: Optional[float] = None
    license: Optional[str] = None
    alternatives: List[str] = None
    required_by: List[str] = None
    
    def __post_init__(self):
        if self.vulnerabilities is None:
            self.vulnerabilities = []
        if self.alternatives is None:
            self.alternatives = []
        if self.required_by is None:
            self.required_by = []


@dataclass
class PackageDependencyStatus:
    """Package dependency status"""
    package_path: str
    dependencies: List[DependencyInfo]
    total_dependencies: int = 0
    outdated_dependencies: int = 0
    vulnerable_dependencies: int = 0
    large_dependencies: int = 0
    optimization_score: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class DependencyOptimizer:
    """Main dependency optimization framework"""
    
    def __init__(self):
        self.package_statuses: List[PackageDependencyStatus] = []
        self.vulnerability_db = self._load_vulnerability_db()
        self.alternative_packages = self._load_alternative_packages()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_vulnerability_db(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load vulnerability database"""
        # Mock vulnerability database (in real implementation, would query OSV/PyUp)
        return {
            "requests": [
                {
                    "id": "GHSA-j8r2-6x86-q33q",
                    "severity": "HIGH",
                    "title": "Unintended proxy authentication leakage",
                    "affected_versions": "<2.27.1",
                    "fixed_version": "2.27.1"
                }
            ],
            "urllib3": [
                {
                    "id": "GHSA-v845-jxx5-vc9f",
                    "severity": "MEDIUM",
                    "title": "Proxy-authorization header leakage",
                    "affected_versions": "<1.26.5",
                    "fixed_version": "1.26.5"
                }
            ],
            "pillow": [
                {
                    "id": "GHSA-8vj2-vxx3-667w",
                    "severity": "HIGH",
                    "title": "Buffer overflow in image processing",
                    "affected_versions": "<8.3.2",
                    "fixed_version": "8.3.2"
                }
            ]
        }
    
    def _load_alternative_packages(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load alternative package suggestions"""
        return {
            "requests": [
                {
                    "name": "httpx",
                    "reason": "Async support, better performance",
                    "compatibility": "high"
                },
                {
                    "name": "aiohttp",
                    "reason": "Async-first design",
                    "compatibility": "medium"
                }
            ],
            "pandas": [
                {
                    "name": "polars",
                    "reason": "Better performance, lower memory usage",
                    "compatibility": "medium"
                }
            ],
            "flask": [
                {
                    "name": "fastapi",
                    "reason": "Better performance, automatic OpenAPI",
                    "compatibility": "high"
                }
            ]
        }
    
    def find_packages(self, root_path: str = "src/packages") -> List[str]:
        """Find all packages in the repository"""
        packages = []
        root = Path(root_path)
        
        if not root.exists():
            self.logger.warning(f"Package root {root_path} does not exist")
            return packages
        
        # Find all directories with pyproject.toml
        for pyproject_file in root.rglob("pyproject.toml"):
            package_dir = pyproject_file.parent
            packages.append(str(package_dir))
        
        # Also check root directory
        if Path("pyproject.toml").exists():
            packages.append(".")
        
        return packages
    
    def analyze_package_dependencies(self, package_path: str) -> PackageDependencyStatus:
        """Analyze dependencies for a package"""
        status = PackageDependencyStatus(package_path=package_path, dependencies=[])
        package_dir = Path(package_path)
        
        # Read pyproject.toml
        pyproject_file = package_dir / "pyproject.toml"
        if not pyproject_file.exists():
            self.logger.warning(f"No pyproject.toml found in {package_path}")
            return status
        
        try:
            with open(pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)
            
            # Extract dependencies
            dependencies = pyproject_data.get("project", {}).get("dependencies", [])
            optional_deps = pyproject_data.get("project", {}).get("optional-dependencies", {})
            
            # Add optional dependencies
            for group_name, group_deps in optional_deps.items():
                dependencies.extend(group_deps)
            
            # Analyze each dependency
            for dep_spec in dependencies:
                dep_info = self._analyze_dependency(dep_spec)
                status.dependencies.append(dep_info)
            
            # Calculate statistics
            status.total_dependencies = len(status.dependencies)
            status.outdated_dependencies = sum(
                1 for dep in status.dependencies 
                if dep.latest_version and self._is_outdated(dep.current_version, dep.latest_version)
            )
            status.vulnerable_dependencies = sum(
                1 for dep in status.dependencies 
                if dep.vulnerabilities
            )
            status.large_dependencies = sum(
                1 for dep in status.dependencies 
                if dep.size_mb and dep.size_mb > 50
            )
            
            # Calculate optimization score
            status.optimization_score = self._calculate_optimization_score(status)
            
            # Generate recommendations
            status.recommendations = self._generate_recommendations(status)
            
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies in {package_path}: {e}")
        
        return status
    
    def _analyze_dependency(self, dep_spec: str) -> DependencyInfo:
        """Analyze a single dependency"""
        # Parse dependency specification
        match = re.match(r'^([a-zA-Z0-9_-]+)([>=<~!]*)(.*)$', dep_spec.strip())
        if not match:
            name = dep_spec.strip()
            current_version = "unknown"
        else:
            name, operator, version_spec = match.groups()
            current_version = version_spec or "latest"
        
        dep_info = DependencyInfo(name=name, current_version=current_version)
        
        # Check for vulnerabilities
        if name in self.vulnerability_db:
            for vuln in self.vulnerability_db[name]:
                if self._is_vulnerable(current_version, vuln["affected_versions"]):
                    dep_info.vulnerabilities.append(f"{vuln['id']}: {vuln['title']}")
        
        # Mock latest version check (in real implementation, would query PyPI)
        dep_info.latest_version = self._get_latest_version(name)
        
        # Mock size information
        dep_info.size_mb = self._get_package_size(name)
        
        # Mock license information
        dep_info.license = self._get_package_license(name)
        
        # Check for alternatives
        if name in self.alternative_packages:
            dep_info.alternatives = [alt["name"] for alt in self.alternative_packages[name]]
        
        return dep_info
    
    def _get_latest_version(self, package_name: str) -> Optional[str]:
        """Get latest version from PyPI (mock implementation)"""
        # Mock version data
        mock_versions = {
            "requests": "2.31.0",
            "urllib3": "2.0.7",
            "pydantic": "2.5.0",
            "fastapi": "0.104.1",
            "pandas": "2.1.3",
            "numpy": "1.25.2",
            "scikit-learn": "1.3.2"
        }
        return mock_versions.get(package_name, "1.0.0")
    
    def _get_package_size(self, package_name: str) -> Optional[float]:
        """Get package size (mock implementation)"""
        # Mock size data in MB
        mock_sizes = {
            "pandas": 120.5,
            "numpy": 45.2,
            "scikit-learn": 85.3,
            "tensorflow": 500.0,
            "torch": 750.0,
            "requests": 2.1,
            "pydantic": 5.4,
            "fastapi": 3.2
        }
        return mock_sizes.get(package_name, 10.0)
    
    def _get_package_license(self, package_name: str) -> Optional[str]:
        """Get package license (mock implementation)"""
        # Mock license data
        mock_licenses = {
            "requests": "Apache-2.0",
            "urllib3": "MIT",
            "pydantic": "MIT",
            "fastapi": "MIT",
            "pandas": "BSD-3-Clause",
            "numpy": "BSD-3-Clause",
            "scikit-learn": "BSD-3-Clause"
        }
        return mock_licenses.get(package_name, "MIT")
    
    def _is_vulnerable(self, current_version: str, affected_versions: str) -> bool:
        """Check if current version is vulnerable"""
        try:
            if "<" in affected_versions:
                fix_version = affected_versions.replace("<", "").strip()
                return version.parse(current_version) < version.parse(fix_version)
            elif ">" in affected_versions:
                min_version = affected_versions.replace(">", "").strip()
                return version.parse(current_version) > version.parse(min_version)
            elif "==" in affected_versions:
                exact_version = affected_versions.replace("==", "").strip()
                return version.parse(current_version) == version.parse(exact_version)
        except Exception:
            pass
        return False
    
    def _is_outdated(self, current: str, latest: str) -> bool:
        """Check if current version is outdated"""
        try:
            return version.parse(current) < version.parse(latest)
        except Exception:
            return False
    
    def _calculate_optimization_score(self, status: PackageDependencyStatus) -> float:
        """Calculate optimization score (0-100)"""
        if status.total_dependencies == 0:
            return 100.0
        
        # Base score
        score = 100.0
        
        # Deduct points for issues
        if status.vulnerable_dependencies > 0:
            score -= (status.vulnerable_dependencies / status.total_dependencies) * 40
        
        if status.outdated_dependencies > 0:
            score -= (status.outdated_dependencies / status.total_dependencies) * 20
        
        if status.large_dependencies > 0:
            score -= (status.large_dependencies / status.total_dependencies) * 15
        
        # Bonus for low dependency count
        if status.total_dependencies < 10:
            score += 10
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, status: PackageDependencyStatus) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Vulnerability recommendations
        vulnerable_deps = [dep for dep in status.dependencies if dep.vulnerabilities]
        if vulnerable_deps:
            recommendations.append(f"Update {len(vulnerable_deps)} vulnerable dependencies")
        
        # Outdated dependency recommendations
        outdated_deps = [
            dep for dep in status.dependencies 
            if dep.latest_version and self._is_outdated(dep.current_version, dep.latest_version)
        ]
        if outdated_deps:
            recommendations.append(f"Update {len(outdated_deps)} outdated dependencies")
        
        # Large dependency recommendations
        large_deps = [dep for dep in status.dependencies if dep.size_mb and dep.size_mb > 50]
        if large_deps:
            recommendations.append(f"Consider alternatives for {len(large_deps)} large dependencies")
        
        # Alternative package recommendations
        deps_with_alternatives = [dep for dep in status.dependencies if dep.alternatives]
        if deps_with_alternatives:
            recommendations.append(f"Consider alternatives for {len(deps_with_alternatives)} packages")
        
        # Pin version recommendations
        unpinned_deps = [dep for dep in status.dependencies if not dep.pinned_version]
        if len(unpinned_deps) > len(status.dependencies) * 0.5:
            recommendations.append("Consider pinning critical dependency versions")
        
        return recommendations
    
    def optimize_package_dependencies(self, package_path: str) -> bool:
        """Optimize dependencies for a package"""
        try:
            package_dir = Path(package_path)
            pyproject_file = package_dir / "pyproject.toml"
            
            if not pyproject_file.exists():
                self.logger.warning(f"No pyproject.toml found in {package_path}")
                return False
            
            # Read current configuration
            with open(pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)
            
            # Get current dependencies
            dependencies = pyproject_data.get("project", {}).get("dependencies", [])
            
            # Optimize dependencies
            optimized_deps = []
            for dep_spec in dependencies:
                optimized_dep = self._optimize_dependency(dep_spec)
                optimized_deps.append(optimized_dep)
            
            # Update pyproject.toml
            if "project" not in pyproject_data:
                pyproject_data["project"] = {}
            
            pyproject_data["project"]["dependencies"] = optimized_deps
            
            # Add dependency management section
            if "tool" not in pyproject_data:
                pyproject_data["tool"] = {}
            
            pyproject_data["tool"]["dependency-optimizer"] = {
                "last_optimized": datetime.now().isoformat(),
                "optimization_enabled": True,
                "auto_update": False,
                "vulnerability_check": True
            }
            
            # Write updated configuration
            with open(pyproject_file, 'w') as f:
                toml.dump(pyproject_data, f)
            
            self.logger.info(f"✅ Optimized dependencies for {package_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing dependencies for {package_path}: {e}")
            return False
    
    def _optimize_dependency(self, dep_spec: str) -> str:
        """Optimize a single dependency specification"""
        # Parse dependency
        match = re.match(r'^([a-zA-Z0-9_-]+)([>=<~!]*)(.*)$', dep_spec.strip())
        if not match:
            return dep_spec
        
        name, operator, version_spec = match.groups()
        
        # Check for vulnerabilities and update if needed
        if name in self.vulnerability_db:
            for vuln in self.vulnerability_db[name]:
                if self._is_vulnerable(version_spec or "0.0.0", vuln["affected_versions"]):
                    # Update to fixed version
                    fixed_version = vuln["fixed_version"]
                    return f"{name}>={fixed_version}"
        
        # Pin to specific version ranges for stability
        if not operator or operator == ">=":
            latest_version = self._get_latest_version(name)
            if latest_version:
                # Pin to major.minor version
                major_minor = ".".join(latest_version.split(".")[:2])
                return f"{name}~={major_minor}.0"
        
        return dep_spec
    
    def generate_dependency_report(self, output_file: str = "dependency_report.html"):
        """Generate comprehensive dependency report"""
        
        # Calculate summary statistics
        total_packages = len(self.package_statuses)
        total_dependencies = sum(status.total_dependencies for status in self.package_statuses)
        total_vulnerabilities = sum(status.vulnerable_dependencies for status in self.package_statuses)
        total_outdated = sum(status.outdated_dependencies for status in self.package_statuses)
        avg_optimization_score = sum(status.optimization_score for status in self.package_statuses) / total_packages if total_packages > 0 else 0
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dependency Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .summary-item {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 5px; }}
                .good {{ background: #28a745; color: white; }}
                .warning {{ background: #ffc107; color: black; }}
                .danger {{ background: #dc3545; color: white; }}
                .package-status {{ margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid; }}
                .package-status.good {{ background: #d1ecf1; border-left-color: #28a745; }}
                .package-status.warning {{ background: #fff3cd; border-left-color: #ffc107; }}
                .package-status.danger {{ background: #f8d7da; border-left-color: #dc3545; }}
                .dependency-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                .dependency-table th, .dependency-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .dependency-table th {{ background-color: #f2f2f2; }}
                .vulnerability {{ background: #f8d7da; padding: 5px; border-radius: 3px; margin: 2px 0; }}
                .recommendation {{ background: #d4edda; padding: 5px; border-radius: 3px; margin: 2px 0; }}
                .score {{ font-size: 24px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Dependency Optimization Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="summary-item">
                    <h3>{total_packages}</h3>
                    <p>Total Packages</p>
                </div>
                <div class="summary-item">
                    <h3>{total_dependencies}</h3>
                    <p>Total Dependencies</p>
                </div>
                <div class="summary-item {'danger' if total_vulnerabilities > 0 else 'good'}">
                    <h3>{total_vulnerabilities}</h3>
                    <p>Vulnerabilities</p>
                </div>
                <div class="summary-item {'warning' if total_outdated > 0 else 'good'}">
                    <h3>{total_outdated}</h3>
                    <p>Outdated</p>
                </div>
                <div class="summary-item {'good' if avg_optimization_score >= 80 else 'warning' if avg_optimization_score >= 60 else 'danger'}">
                    <h3 class="score">{avg_optimization_score:.1f}%</h3>
                    <p>Avg Optimization Score</p>
                </div>
            </div>
            
            <h2>Package Dependency Status</h2>
        """
        
        for status in self.package_statuses:
            status_class = "good" if status.optimization_score >= 80 else "warning" if status.optimization_score >= 60 else "danger"
            
            html_content += f"""
            <div class="package-status {status_class}">
                <h3>{status.package_path}</h3>
                <div class="score">Optimization Score: {status.optimization_score:.1f}%</div>
                <p><strong>Dependencies:</strong> {status.total_dependencies}</p>
                <p><strong>Vulnerabilities:</strong> {status.vulnerable_dependencies}</p>
                <p><strong>Outdated:</strong> {status.outdated_dependencies}</p>
                <p><strong>Large Dependencies:</strong> {status.large_dependencies}</p>
                
                <h4>Recommendations:</h4>
                {''.join(f'<div class="recommendation">{rec}</div>' for rec in status.recommendations)}
                
                <h4>Dependencies:</h4>
                <table class="dependency-table">
                    <thead>
                        <tr>
                            <th>Package</th>
                            <th>Current Version</th>
                            <th>Latest Version</th>
                            <th>Size (MB)</th>
                            <th>License</th>
                            <th>Issues</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for dep in status.dependencies:
                issues = []
                if dep.vulnerabilities:
                    issues.extend(dep.vulnerabilities)
                if dep.latest_version and self._is_outdated(dep.current_version, dep.latest_version):
                    issues.append("Outdated")
                
                issues_text = "<br>".join(f'<span class="vulnerability">{issue}</span>' for issue in issues) if issues else "None"
                
                html_content += f"""
                    <tr>
                        <td>{dep.name}</td>
                        <td>{dep.current_version}</td>
                        <td>{dep.latest_version or 'Unknown'}</td>
                        <td>{dep.size_mb:.1f if dep.size_mb else 'Unknown'}</td>
                        <td>{dep.license or 'Unknown'}</td>
                        <td>{issues_text}</td>
                    </tr>
                """
            
            html_content += """
                    </tbody>
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Dependency report generated: {output_file}")
    
    def run_optimization(self, packages: List[str] = None) -> bool:
        """Run complete dependency optimization process"""
        if packages is None:
            packages = self.find_packages()
        
        self.logger.info(f"Running dependency optimization on {len(packages)} packages")
        
        success = True
        
        # Analyze current dependencies
        for package_path in packages:
            status = self.analyze_package_dependencies(package_path)
            self.package_statuses.append(status)
            
            self.logger.info(f"📊 {package_path} - Score: {status.optimization_score:.1f}%")
        
        # Optimize dependencies
        for package_path in packages:
            if not self.optimize_package_dependencies(package_path):
                success = False
        
        # Re-analyze after optimization
        self.package_statuses = []
        for package_path in packages:
            status = self.analyze_package_dependencies(package_path)
            self.package_statuses.append(status)
        
        # Generate report
        self.generate_dependency_report()
        
        # Check final results
        low_score_packages = [s for s in self.package_statuses if s.optimization_score < 60]
        if low_score_packages:
            self.logger.warning(f"⚠️  {len(low_score_packages)} packages have low optimization scores")
        
        vulnerable_packages = [s for s in self.package_statuses if s.vulnerable_dependencies > 0]
        if vulnerable_packages:
            self.logger.error(f"❌ {len(vulnerable_packages)} packages have vulnerable dependencies")
            success = False
        
        if success:
            self.logger.info("✅ Dependency optimization completed successfully")
        
        return success
    
    def run_ci_dependency_check(self, packages: List[str] = None) -> bool:
        """Run dependency check in CI environment"""
        if packages is None:
            packages = self.find_packages()
        
        self.logger.info("Running CI dependency check")
        
        # Analyze all packages
        for package_path in packages:
            status = self.analyze_package_dependencies(package_path)
            self.package_statuses.append(status)
        
        # Generate report
        self.generate_dependency_report()
        
        # Check for critical issues
        critical_issues = []
        for status in self.package_statuses:
            if status.vulnerable_dependencies > 0:
                critical_issues.append(f"{status.package_path}: {status.vulnerable_dependencies} vulnerable dependencies")
        
        if critical_issues:
            self.logger.error("Critical dependency issues found:")
            for issue in critical_issues:
                self.logger.error(f"  - {issue}")
            return False
        
        self.logger.info("Dependency check passed")
        return True


def main():
    """Main entry point for dependency optimization"""
    if len(sys.argv) > 1 and sys.argv[1] == "ci":
        # Run CI dependency check
        optimizer = DependencyOptimizer()
        success = optimizer.run_ci_dependency_check()
        sys.exit(0 if success else 1)
    else:
        # Run full optimization
        optimizer = DependencyOptimizer()
        success = optimizer.run_optimization()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()