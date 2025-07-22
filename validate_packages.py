#!/usr/bin/env python3
"""
Comprehensive Package Validation Script
Validates all packages in src/packages/ against the standard template structure.
"""

import os
import json
import toml
from pathlib import Path
from typing import Dict, List, Tuple, Any

class PackageValidator:
    def __init__(self, packages_root: str):
        self.packages_root = Path(packages_root)
        self.validation_results = {}
        
    def get_all_packages(self) -> List[Path]:
        """Find all packages with pyproject.toml files."""
        packages = []
        for pyproject in self.packages_root.rglob("pyproject.toml"):
            # Skip template files
            if "template" not in str(pyproject):
                packages.append(pyproject.parent)
        return sorted(packages)
    
    def validate_package_structure(self, package_path: Path) -> Dict[str, Any]:
        """Validate a single package against template structure."""
        package_name = package_path.name
        relative_path = package_path.relative_to(self.packages_root)
        
        result = {
            'package_name': package_name,
            'path': str(relative_path),
            'issues': [],
            'warnings': [],
            'compliance_score': 0,
            'structure_check': {},
            'ddd_layers': {},
            'entry_points': {},
            'pyproject_validation': {}
        }
        
        # Check required top-level structure
        required_structure = [
            'src', 'pyproject.toml', 'BUCK', 'build', 'deploy', 
            'scripts', 'examples', 'docs', 'tests'
        ]
        
        structure_score = 0
        for item in required_structure:
            item_path = package_path / item
            exists = item_path.exists()
            result['structure_check'][item] = exists
            if exists:
                structure_score += 1
            else:
                result['issues'].append(f"Missing required structure: {item}")
        
        # Check src/{package_name}/ structure
        src_package_path = package_path / 'src' / package_name
        if src_package_path.exists():
            result['structure_check']['src_package_dir'] = True
            
            # Check DDD layers
            ddd_layers = ['application', 'domain', 'infrastructure', 'presentation']
            ddd_score = 0
            for layer in ddd_layers:
                layer_path = src_package_path / layer
                exists = layer_path.exists()
                result['ddd_layers'][layer] = exists
                if exists:
                    ddd_score += 1
                else:
                    result['warnings'].append(f"Missing DDD layer: {layer}")
            
            # Check entry points
            entry_points = ['cli.py', 'server.py', 'worker.py']
            entry_score = 0
            for entry in entry_points:
                entry_path = src_package_path / entry
                exists = entry_path.exists()
                result['entry_points'][entry] = exists
                if exists:
                    entry_score += 1
                else:
                    result['warnings'].append(f"Missing entry point: {entry}")
            
            # Calculate compliance score as percentage (max possible: 9 structure + 4 DDD + 3 entry = 16 points)
            max_score = len(required_structure) + 4 + 3  # 16 total
            actual_score = structure_score + ddd_score + entry_score
            result['compliance_score'] = (actual_score / max_score) * 100
        else:
            result['issues'].append(f"Missing src/{package_name}/ directory")
            result['structure_check']['src_package_dir'] = False
        
        # Validate pyproject.toml
        pyproject_path = package_path / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r') as f:
                    pyproject_data = toml.load(f)
                
                result['pyproject_validation'] = self.validate_pyproject(pyproject_data, package_name)
            except Exception as e:
                result['issues'].append(f"Failed to parse pyproject.toml: {str(e)}")
        
        return result
    
    def validate_pyproject(self, data: Dict, package_name: str) -> Dict[str, Any]:
        """Validate pyproject.toml structure and content."""
        validation = {
            'has_project_section': False,
            'has_correct_name': False,
            'has_fastapi': False,
            'has_uvicorn': False,
            'has_build_system': False,
            'has_scripts': False,
            'issues': [],
            'warnings': []
        }
        
        # Check [project] section
        if 'project' in data:
            validation['has_project_section'] = True
            project = data['project']
            
            # Check name
            if 'name' in project:
                expected_name = package_name.replace('_', '-')
                if project['name'] == expected_name:
                    validation['has_correct_name'] = True
                else:
                    validation['warnings'].append(f"Package name mismatch: {project['name']} vs {expected_name}")
            
            # Check dependencies
            if 'dependencies' in project:
                deps = project['dependencies']
                if any('fastapi' in dep.lower() for dep in deps):
                    validation['has_fastapi'] = True
                if any('uvicorn' in dep.lower() for dep in deps):
                    validation['has_uvicorn'] = True
        else:
            validation['issues'].append("Missing [project] section")
        
        # Check build system
        if 'build-system' in data:
            validation['has_build_system'] = True
        else:
            validation['warnings'].append("Missing [build-system] section")
        
        # Check scripts
        if 'project' in data and 'scripts' in data['project']:
            validation['has_scripts'] = True
        else:
            validation['warnings'].append("Missing [project.scripts] section")
        
        return validation
    
    def categorize_packages(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize packages by compliance level."""
        categories = {
            'fully_compliant': [],  # Score >= 90%
            'minor_issues': [],     # Score 70-89%
            'major_issues': [],     # Score < 70%
            'broken': []            # Critical issues
        }
        
        for result in results:
            score = result['compliance_score']
            has_critical = any('Missing src/' in issue for issue in result['issues'])
            
            if has_critical or score < 50:
                categories['broken'].append(result)
            elif score >= 90:
                categories['fully_compliant'].append(result)
            elif score >= 70:
                categories['minor_issues'].append(result)
            else:
                categories['major_issues'].append(result)
        
        return categories
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        packages = self.get_all_packages()
        all_results = []
        
        print(f"Validating {len(packages)} packages...")
        
        for package_path in packages:
            result = self.validate_package_structure(package_path)
            all_results.append(result)
            print(f"✓ {result['package_name']} - Score: {result['compliance_score']:.1f}%")
        
        categories = self.categorize_packages(all_results)
        
        # Generate report
        report = []
        report.append("# COMPREHENSIVE PACKAGE VALIDATION REPORT")
        report.append("=" * 50)
        report.append("")
        report.append(f"**Total packages validated:** {len(all_results)}")
        report.append(f"**Fully compliant packages:** {len(categories['fully_compliant'])}")
        report.append(f"**Packages with minor issues:** {len(categories['minor_issues'])}")
        report.append(f"**Packages with major issues:** {len(categories['major_issues'])}")
        report.append(f"**Broken packages:** {len(categories['broken'])}")
        report.append("")
        
        # Detailed breakdown by category
        for category, packages in categories.items():
            if not packages:
                continue
                
            report.append(f"## {category.replace('_', ' ').title()} ({len(packages)} packages)")
            report.append("")
            
            for pkg in packages:
                report.append(f"### {pkg['package_name']} (Score: {pkg['compliance_score']:.1f}%)")
                report.append(f"**Path:** `{pkg['path']}`")
                report.append("")
                
                # Structure compliance
                report.append("**Structure Compliance:**")
                for item, exists in pkg['structure_check'].items():
                    status = "✓" if exists else "✗"
                    report.append(f"- {status} {item}")
                report.append("")
                
                # DDD layers
                if pkg['ddd_layers']:
                    report.append("**DDD Layers:**")
                    for layer, exists in pkg['ddd_layers'].items():
                        status = "✓" if exists else "✗"
                        report.append(f"- {status} {layer}/")
                    report.append("")
                
                # Entry points
                if pkg['entry_points']:
                    report.append("**Entry Points:**")
                    for entry, exists in pkg['entry_points'].items():
                        status = "✓" if exists else "✗"
                        report.append(f"- {status} {entry}")
                    report.append("")
                
                # Issues and warnings
                if pkg['issues']:
                    report.append("**Critical Issues:**")
                    for issue in pkg['issues']:
                        report.append(f"- ⚠️ {issue}")
                    report.append("")
                
                if pkg['warnings']:
                    report.append("**Warnings:**")
                    for warning in pkg['warnings']:
                        report.append(f"- ⚡ {warning}")
                    report.append("")
                
                report.append("---")
                report.append("")
        
        # Summary recommendations
        report.append("## RECOMMENDATIONS")
        report.append("")
        
        if categories['broken']:
            report.append("### Immediate Action Required:")
            for pkg in categories['broken']:
                report.append(f"- **{pkg['package_name']}**: {', '.join(pkg['issues'])}")
            report.append("")
        
        if categories['major_issues']:
            report.append("### High Priority Fixes:")
            for pkg in categories['major_issues']:
                missing_structure = [k for k, v in pkg['structure_check'].items() if not v]
                if missing_structure:
                    report.append(f"- **{pkg['package_name']}**: Missing {', '.join(missing_structure)}")
            report.append("")
        
        # Overall statistics
        avg_score = sum(r['compliance_score'] for r in all_results) / len(all_results)
        report.append(f"**Average compliance score:** {avg_score:.1f}%")
        report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    validator = PackageValidator("/mnt/c/Users/andre/Pynomaly/src/packages")
    report = validator.generate_report()
    
    # Save report
    with open("/mnt/c/Users/andre/Pynomaly/PACKAGE_VALIDATION_REPORT.md", "w") as f:
        f.write(report)
    
    print("\n" + "=" * 50)
    print("VALIDATION COMPLETE!")
    print("=" * 50)
    print(report)