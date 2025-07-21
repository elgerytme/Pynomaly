#!/usr/bin/env python3
"""
Migration Validation Script
=========================
Validates that the migration to new DDD structure is successful.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

class MigrationValidator:
    """Validates migrated package structure and functionality"""
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.src_path = package_path / "src" / "anomaly_detection"
        
    def validate_structure(self) -> Dict[str, bool]:
        """Validate the DDD folder structure exists"""
        print("🏗️  Validating folder structure...")
        
        required_dirs = [
            "build",
            "deploy/docker", 
            "deploy/k8s",
            "deploy/monitoring",
            "docs",
            "scripts",
            "tests",
            "src/anomaly_detection",
            "src/anomaly_detection/application",
            "src/anomaly_detection/domain", 
            "src/anomaly_detection/infrastructure",
            "src/anomaly_detection/presentation",
        ]
        
        results = {}
        for dir_path in required_dirs:
            full_path = self.package_path / dir_path
            exists = full_path.exists() and full_path.is_dir()
            results[dir_path] = exists
            status = "✅" if exists else "❌"
            print(f"  {status} {dir_path}")
        
        return results
    
    def validate_ddd_layers(self) -> Dict[str, bool]:
        """Validate DDD layer structure"""
        print("\n🏛️  Validating DDD layers...")
        
        layer_subdirs = {
            "domain": ["entities", "value_objects", "services", "repositories", "exceptions"],
            "application": ["services", "use_cases", "dto"],
            "infrastructure": ["adapters", "repositories", "api", "cli", "config", "security", "monitoring"],
            "presentation": ["api", "cli", "sdk", "web"]
        }
        
        results = {}
        for layer, subdirs in layer_subdirs.items():
            layer_path = self.src_path / layer
            layer_exists = layer_path.exists()
            results[layer] = layer_exists
            
            if layer_exists:
                print(f"  ✅ {layer}/ layer")
                for subdir in subdirs:
                    subdir_path = layer_path / subdir
                    if subdir_path.exists():
                        print(f"    ✅ {subdir}/")
                    else:
                        print(f"    ⚠️  {subdir}/ (missing)")
            else:
                print(f"  ❌ {layer}/ layer (missing)")
        
        return results
    
    def validate_key_files(self) -> Dict[str, bool]:
        """Validate key files exist in correct locations"""
        print("\n📄 Validating key files...")
        
        key_files = [
            "pyproject.toml",
            "docs/README.md",
            "docs/CHANGELOG.md", 
            "build/Makefile",
            "deploy/docker/Dockerfile",
            "src/anomaly_detection/__init__.py",
        ]
        
        results = {}
        for file_path in key_files:
            full_path = self.package_path / file_path
            exists = full_path.exists() and full_path.is_file()
            results[file_path] = exists
            status = "✅" if exists else "❌"
            print(f"  {status} {file_path}")
            
        return results
    
    def validate_imports(self) -> Dict[str, bool]:
        """Validate package imports work"""
        print("\n🔗 Validating imports...")
        
        # Add package to Python path
        sys.path.insert(0, str(self.package_path / "src"))
        
        import_tests = [
            ("anomaly_detection", "Main package"),
            ("anomaly_detection.domain", "Domain layer"),
            ("anomaly_detection.application", "Application layer"),  
            ("anomaly_detection.infrastructure", "Infrastructure layer"),
            ("anomaly_detection.presentation", "Presentation layer"),
        ]
        
        results = {}
        for module_name, description in import_tests:
            try:
                __import__(module_name)
                results[module_name] = True
                print(f"  ✅ {description} imports successfully")
            except ImportError as e:
                results[module_name] = False
                print(f"  ❌ {description} import failed: {e}")
        
        return results
    
    def count_migrated_files(self) -> Dict[str, int]:
        """Count files in each layer"""
        print("\n📊 Counting migrated files...")
        
        counts = {}
        layers = ["application", "domain", "infrastructure", "presentation"]
        
        for layer in layers:
            layer_path = self.src_path / layer
            if layer_path.exists():
                py_files = list(layer_path.rglob("*.py"))
                counts[layer] = len(py_files)
                print(f"  📁 {layer}: {len(py_files)} Python files")
            else:
                counts[layer] = 0
                print(f"  📁 {layer}: 0 files (layer missing)")
        
        # Count other important directories
        for dir_name in ["deploy", "docs", "scripts"]:
            dir_path = self.package_path / dir_name
            if dir_path.exists():
                all_files = list(dir_path.rglob("*"))
                file_count = len([f for f in all_files if f.is_file()])
                counts[dir_name] = file_count
                print(f"  📁 {dir_name}: {file_count} files")
                
        return counts
    
    def validate_architecture_rules(self) -> List[str]:
        """Validate DDD architecture rules"""
        print("\n🏛️  Validating architecture rules...")
        
        violations = []
        
        # Check domain layer doesn't import from other layers
        domain_path = self.src_path / "domain"
        if domain_path.exists():
            for py_file in domain_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for imports from other layers
                    forbidden_imports = [
                        "from .application",
                        "from .infrastructure", 
                        "from .presentation",
                        "from anomaly_detection.application",
                        "from anomaly_detection.infrastructure",
                        "from anomaly_detection.presentation"
                    ]
                    
                    for forbidden in forbidden_imports:
                        if forbidden in content:
                            violations.append(f"Domain layer violation: {py_file.name} imports {forbidden}")
                            
                except Exception as e:
                    violations.append(f"Error checking {py_file}: {e}")
        
        if not violations:
            print("  ✅ Domain layer architecture rules validated")
        else:
            print("  ❌ Architecture violations found:")
            for violation in violations[:5]:  # Show first 5
                print(f"    • {violation}")
                
        return violations
    
    def generate_validation_report(self, structure_results: Dict[str, bool], 
                                 layer_results: Dict[str, bool],
                                 file_results: Dict[str, bool],
                                 import_results: Dict[str, bool],
                                 file_counts: Dict[str, int],
                                 violations: List[str]) -> str:
        """Generate comprehensive validation report"""
        
        structure_score = sum(structure_results.values()) / len(structure_results)
        layer_score = sum(layer_results.values()) / len(layer_results)
        file_score = sum(file_results.values()) / len(file_results) 
        import_score = sum(import_results.values()) / len(import_results)
        
        overall_score = (structure_score + layer_score + file_score + import_score) / 4
        
        report = f"""
Migration Validation Report
==========================

Package: {self.package_path.name}
Validation Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Score: {overall_score:.1%}

### Structure Validation: {structure_score:.1%}
- Required directories: {sum(structure_results.values())}/{len(structure_results)}

### DDD Layer Validation: {layer_score:.1%}  
- Core layers present: {sum(layer_results.values())}/{len(layer_results)}

### Key Files Validation: {file_score:.1%}
- Essential files: {sum(file_results.values())}/{len(file_results)}

### Import Validation: {import_score:.1%}
- Successful imports: {sum(import_results.values())}/{len(import_results)}

### File Distribution:
- Application layer: {file_counts.get('application', 0)} files
- Domain layer: {file_counts.get('domain', 0)} files  
- Infrastructure layer: {file_counts.get('infrastructure', 0)} files
- Presentation layer: {file_counts.get('presentation', 0)} files
- Deploy configs: {file_counts.get('deploy', 0)} files
- Documentation: {file_counts.get('docs', 0)} files

### Architecture Violations: {len(violations)}
{chr(10).join(f"- {v}" for v in violations[:3]) if violations else "None found"}

### Migration Status:
{'✅ SUCCESSFUL MIGRATION' if overall_score >= 0.8 else '⚠️ MIGRATION ISSUES DETECTED' if overall_score >= 0.6 else '❌ MIGRATION FAILED'}

### Recommendations:
{self._generate_recommendations(structure_results, layer_results, file_results, import_results, violations)}
"""
        
        return report
    
    def _generate_recommendations(self, structure_results, layer_results, file_results, import_results, violations) -> str:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Structure issues
        missing_dirs = [k for k, v in structure_results.items() if not v]
        if missing_dirs:
            recommendations.append(f"Create missing directories: {', '.join(missing_dirs[:3])}")
        
        # Layer issues  
        missing_layers = [k for k, v in layer_results.items() if not v]
        if missing_layers:
            recommendations.append(f"Set up missing DDD layers: {', '.join(missing_layers)}")
            
        # File issues
        missing_files = [k for k, v in file_results.items() if not v]
        if missing_files:
            recommendations.append(f"Create missing files: {', '.join(missing_files[:2])}")
            
        # Import issues
        failed_imports = [k for k, v in import_results.items() if not v]
        if failed_imports:
            recommendations.append("Fix import issues - check __init__.py files and dependencies")
            
        # Architecture violations
        if violations:
            recommendations.append("Fix architecture violations - ensure domain layer independence")
            
        return "\n".join(f"• {rec}" for rec in recommendations) or "• Migration appears successful - no immediate issues detected"

def main():
    parser = argparse.ArgumentParser(description="Validate package migration")
    parser.add_argument('--package-path', required=True, help='Path to migrated package')
    parser.add_argument('--report', help='Output path for validation report')
    
    args = parser.parse_args()
    
    package_path = Path(args.package_path)
    if not package_path.exists():
        print(f"❌ Package path not found: {package_path}")
        return 1
    
    print(f"🔍 Validating migration: {package_path.name}")
    
    validator = MigrationValidator(package_path)
    
    # Run all validations
    structure_results = validator.validate_structure()
    layer_results = validator.validate_ddd_layers() 
    file_results = validator.validate_key_files()
    import_results = validator.validate_imports()
    file_counts = validator.count_migrated_files()
    violations = validator.validate_architecture_rules()
    
    # Generate report
    report = validator.generate_validation_report(
        structure_results, layer_results, file_results, 
        import_results, file_counts, violations
    )
    
    print(report)
    
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\n📄 Validation report saved to: {args.report}")
    
    # Determine exit code
    overall_score = (
        sum(structure_results.values()) / len(structure_results) +
        sum(layer_results.values()) / len(layer_results) + 
        sum(file_results.values()) / len(file_results) +
        sum(import_results.values()) / len(import_results)
    ) / 4
    
    if overall_score >= 0.8:
        print("\n🎉 Migration validation PASSED!")
        return 0
    else:
        print(f"\n⚠️ Migration validation completed with issues (score: {overall_score:.1%})")
        return 1

if __name__ == "__main__":
    exit(main())