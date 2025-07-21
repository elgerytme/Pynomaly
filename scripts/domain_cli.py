#!/usr/bin/env python3
"""
Domain CLI Tool

A comprehensive command-line interface for domain-bounded development.
Provides intelligent domain management, validation, and developer assistance.
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import shutil

# Import domain analysis tools
sys.path.append(str(Path(__file__).parent))
from domain_boundary_validator import DomainBoundaryValidator, NewDomainDetection
from create_domain_package import IntelligentDomainAnalyzer, DomainSuggestion

@dataclass
class DomainStatus:
    """Represents the status of a domain package"""
    name: str
    path: str
    exists: bool
    violations: int
    entities: List[str]
    services: List[str]
    tests: int
    coverage: float
    last_modified: str

class DomainCLI:
    """Main CLI tool for domain management"""
    
    def __init__(self):
        self.validator = DomainBoundaryValidator()
        self.analyzer = IntelligentDomainAnalyzer()
        self.root_path = Path(".")
        self.packages_path = Path("src/packages")
    
    def status(self, package_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of domain packages"""
        if package_name:
            return self._get_single_domain_status(package_name)
        else:
            return self._get_all_domains_status()
    
    def _get_single_domain_status(self, package_name: str) -> Dict[str, Any]:
        """Get detailed status of a single domain package"""
        package_path = self.packages_path / package_name
        
        if not package_path.exists():
            return {
                "name": package_name,
                "exists": False,
                "error": f"Package '{package_name}' not found at {package_path}"
            }
        
        # Count entities and services
        entities_dir = package_path / "core" / "domain" / "entities"
        services_dir = package_path / "core" / "domain" / "services"
        tests_dir = package_path / "tests"
        
        entities = []
        services = []
        if entities_dir.exists():
            entities = [f.stem for f in entities_dir.glob("*.py") if f.name != "__init__.py"]
        if services_dir.exists():
            services = [f.stem for f in services_dir.glob("*.py") if f.name != "__init__.py"]
        
        # Count tests
        test_count = 0
        if tests_dir.exists():
            test_count = len(list(tests_dir.rglob("test_*.py")))
        
        # Check for violations
        violations = self.validator.validate_package(str(package_path))
        
        # Get last modified time
        last_modified = datetime.fromtimestamp(package_path.stat().st_mtime).isoformat()
        
        return {
            "name": package_name,
            "path": str(package_path),
            "exists": True,
            "violations": len(violations),
            "entities": entities,
            "services": services,
            "tests": test_count,
            "coverage": self._get_coverage(package_path),
            "last_modified": last_modified,
            "detailed_violations": [
                {
                    "file": v.file_path,
                    "line": v.line_number,
                    "term": v.prohibited_term,
                    "severity": v.severity
                } for v in violations[:10]  # Limit to first 10
            ]
        }
    
    def _get_all_domains_status(self) -> Dict[str, Any]:
        """Get status of all domain packages"""
        domains = {}
        
        if not self.packages_path.exists():
            return {"error": f"Packages directory not found: {self.packages_path}"}
        
        for package_dir in self.packages_path.iterdir():
            if package_dir.is_dir() and not package_dir.name.startswith('.'):
                # Handle nested structure (e.g., ai/machine_learning)
                if any(sub.is_dir() and not sub.name.startswith('.') for sub in package_dir.iterdir()):
                    for subdir in package_dir.iterdir():
                        if subdir.is_dir() and not subdir.name.startswith('.'):
                            domain_name = f"{package_dir.name}/{subdir.name}"
                            domains[domain_name] = self._get_domain_summary(subdir)
                else:
                    domains[package_dir.name] = self._get_domain_summary(package_dir)
        
        return {
            "total_domains": len(domains),
            "domains": domains,
            "summary": self._generate_summary(domains)
        }
    
    def _get_domain_summary(self, package_path: Path) -> Dict[str, Any]:
        """Get summary information for a domain package"""
        entities_count = 0
        services_count = 0
        tests_count = 0
        
        entities_dir = package_path / "core" / "domain" / "entities"
        services_dir = package_path / "core" / "domain" / "services"
        tests_dir = package_path / "tests"
        
        if entities_dir.exists():
            entities_count = len([f for f in entities_dir.glob("*.py") if f.name != "__init__.py"])
        if services_dir.exists():
            services_count = len([f for f in services_dir.glob("*.py") if f.name != "__init__.py"])
        if tests_dir.exists():
            tests_count = len(list(tests_dir.rglob("test_*.py")))
        
        violations = self.validator.validate_package(str(package_path))
        
        return {
            "entities": entities_count,
            "services": services_count,
            "tests": tests_count,
            "violations": len(violations),
            "coverage": self._get_coverage(package_path),
            "healthy": len(violations) == 0 and tests_count > 0
        }
    
    def _get_coverage(self, package_path: Path) -> float:
        """Get test coverage for a package"""
        # Placeholder - would integrate with actual coverage tools
        return 0.0
    
    def _generate_summary(self, domains: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_entities = sum(d.get('entities', 0) for d in domains.values())
        total_services = sum(d.get('services', 0) for d in domains.values())
        total_tests = sum(d.get('tests', 0) for d in domains.values())
        total_violations = sum(d.get('violations', 0) for d in domains.values())
        healthy_domains = sum(1 for d in domains.values() if d.get('healthy', False))
        
        return {
            "total_entities": total_entities,
            "total_services": total_services,
            "total_tests": total_tests,
            "total_violations": total_violations,
            "healthy_domains": healthy_domains,
            "health_percentage": (healthy_domains / len(domains)) * 100 if domains else 0
        }
    
    def validate(self, package_name: Optional[str] = None, fix_violations: bool = False) -> Dict[str, Any]:
        """Validate domain boundaries"""
        print("🔍 Running domain boundary validation...")
        
        if package_name:
            # Validate single package
            package_path = self.packages_path / package_name
            if not package_path.exists():
                return {"error": f"Package '{package_name}' not found"}
            
            violations = self.validator.validate_package(str(package_path))
            results = {package_name: violations} if violations else {}
        else:
            # Validate all packages
            results = self.validator.validate_all_packages()
        
        # Detect new domains
        self.validator.detect_new_domains()
        
        # Generate report
        report = self.validator.generate_report(results)
        
        # Auto-fix violations if requested
        if fix_violations and results:
            print("🔧 Attempting to fix violations...")
            self._auto_fix_violations(results)
        
        return {
            "violations": results,
            "new_domains": [
                {
                    "name": nd.domain_name,
                    "confidence": nd.confidence,
                    "concepts": nd.concepts
                } for nd in self.validator.new_domains
            ],
            "report_file": "domain_violations_report.json",
            "summary": {
                "total_violations": sum(len(v) for v in results.values()),
                "packages_with_violations": len(results),
                "new_domains_detected": len(self.validator.new_domains)
            }
        }
    
    def _auto_fix_violations(self, results: Dict[str, List]) -> None:
        """Attempt to automatically fix domain violations"""
        print("🔧 Auto-fixing domain violations...")
        
        for package_name, violations in results.items():
            print(f"  Fixing {len(violations)} violations in {package_name}...")
            
            # Group violations by file
            violations_by_file = {}
            for violation in violations:
                file_path = violation.file_path
                if file_path not in violations_by_file:
                    violations_by_file[file_path] = []
                violations_by_file[file_path].append(violation)
            
            # Apply fixes to each file
            for file_path, file_violations in violations_by_file.items():
                self._fix_file_violations(file_path, file_violations)
    
    def _fix_file_violations(self, file_path: str, violations: List) -> None:
        """Fix violations in a single file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            modified = False
            lines = content.split('\n')
            
            for violation in violations:
                line_idx = violation.line_number - 1
                if 0 <= line_idx < len(lines):
                    # Simple fixes: comment out problematic terms
                    if violation.prohibited_term in lines[line_idx]:
                        lines[line_idx] = f"# TODO: Fix domain violation - {violation.prohibited_term} - {lines[line_idx]}"
                        modified = True
            
            if modified:
                with open(file_path, 'w') as f:
                    f.write('\n'.join(lines))
                print(f"    ✅ Fixed violations in {file_path}")
        
        except Exception as e:
            print(f"    ❌ Could not fix violations in {file_path}: {e}")
    
    def suggest(self, interactive: bool = False) -> List[DomainSuggestion]:
        """Suggest new domain packages based on code analysis"""
        print("🔍 Analyzing codebase for domain suggestions...")
        suggestions = self.analyzer.analyze_existing_code()
        
        if not suggestions:
            print("ℹ️  No domain patterns detected in current codebase")
            return []
        
        print(f"\n🎯 Found {len(suggestions)} domain suggestions:")
        
        for i, suggestion in enumerate(suggestions, 1):
            confidence_emoji = "🟢" if suggestion.confidence > 0.8 else "🟡" if suggestion.confidence > 0.6 else "🔴"
            print(f"  {i}. {confidence_emoji} {suggestion.name} (confidence: {suggestion.confidence:.2f})")
            print(f"     Concepts: {', '.join(suggestion.concepts[:3])}")
            print(f"     Files: {len(suggestion.files)}")
            print(f"     {suggestion.reasoning}")
            print()
        
        if interactive:
            self._interactive_package_creation(suggestions)
        
        return suggestions
    
    def _interactive_package_creation(self, suggestions: List[DomainSuggestion]) -> None:
        """Interactive package creation from suggestions"""
        print("🎯 Create packages interactively:")
        print("Enter numbers to create packages (e.g., '1,3,5' or 'all' or 'none'):")
        
        try:
            choice = input("Your choice: ").strip().lower()
            
            if choice == 'none':
                return
            elif choice == 'all':
                indices = range(len(suggestions))
            else:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
            
            for idx in indices:
                if 0 <= idx < len(suggestions):
                    suggestion = suggestions[idx]
                    print(f"\n🚀 Creating package: {suggestion.name}")
                    
                    # Use the existing package creation script
                    result = subprocess.run([
                        "python", "scripts/create_domain_package.py",
                        suggestion.name, "--intelligent"
                    ], capture_output=False)
                    
                    if result.returncode == 0:
                        print(f"✅ Successfully created {suggestion.name}")
                    else:
                        print(f"❌ Failed to create {suggestion.name}")
        
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled")
    
    def create(self, package_name: str, intelligent: bool = True, skip_samples: bool = False) -> Dict[str, Any]:
        """Create a new domain package"""
        print(f"🚀 Creating domain package: {package_name}")
        
        # Use the existing package creation script
        cmd = ["python", "scripts/create_domain_package.py", package_name]
        if intelligent:
            cmd.append("--intelligent")
        if skip_samples:
            cmd.append("--skip-samples")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
            "package_name": package_name
        }
    
    def clean(self, package_name: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up domain packages (remove empty directories, fix structure)"""
        print("🧹 Cleaning domain packages...")
        
        cleaned = []
        errors = []
        
        if package_name:
            packages_to_clean = [self.packages_path / package_name]
        else:
            packages_to_clean = [p for p in self.packages_path.iterdir() if p.is_dir()]
        
        for package_path in packages_to_clean:
            try:
                result = self._clean_package(package_path, dry_run)
                cleaned.append(result)
            except Exception as e:
                errors.append(f"Error cleaning {package_path.name}: {e}")
        
        return {
            "cleaned": cleaned,
            "errors": errors,
            "dry_run": dry_run
        }
    
    def _clean_package(self, package_path: Path, dry_run: bool) -> Dict[str, Any]:
        """Clean a single package"""
        actions = []
        
        # Remove empty directories
        for root, dirs, files in os.walk(package_path, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                if not any(dir_path.iterdir()):
                    action = f"Remove empty directory: {dir_path.relative_to(package_path)}"
                    actions.append(action)
                    if not dry_run:
                        dir_path.rmdir()
        
        # Ensure required directories exist
        required_dirs = [
            "core/domain/entities",
            "core/domain/services",
            "core/application/services",
            "tests/unit"
        ]
        
        for required_dir in required_dirs:
            dir_path = package_path / required_dir
            if not dir_path.exists():
                action = f"Create required directory: {required_dir}"
                actions.append(action)
                if not dry_run:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    (dir_path / "__init__.py").touch()
        
        return {
            "package": package_path.name,
            "actions": actions
        }
    
    def monitor(self, continuous: bool = False, interval: int = 60) -> None:
        """Monitor domain compliance continuously"""
        print("📊 Starting domain compliance monitoring...")
        
        if not continuous:
            # Single run monitoring
            self._run_monitoring_cycle()
        else:
            # Continuous monitoring
            import time
            try:
                while True:
                    print(f"\n⏰ Running monitoring cycle at {datetime.now().isoformat()}")
                    self._run_monitoring_cycle()
                    print(f"😴 Sleeping for {interval} seconds...")
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped")
    
    def _run_monitoring_cycle(self) -> None:
        """Run a single monitoring cycle"""
        # Check domain boundaries
        results = self.validator.validate_all_packages()
        self.validator.detect_new_domains()
        
        total_violations = sum(len(v) for v in results.values())
        new_domains_count = len(self.validator.new_domains)
        
        print(f"📈 Violations: {total_violations}")
        print(f"📈 New domains detected: {new_domains_count}")
        
        # Generate alerts for high-violation packages
        for package_name, violations in results.items():
            if len(violations) > 10:
                print(f"🚨 ALERT: {package_name} has {len(violations)} violations")
        
        # Alert for high-confidence new domains
        for new_domain in self.validator.new_domains:
            if new_domain.confidence > 0.8:
                print(f"🆕 ALERT: High-confidence new domain '{new_domain.domain_name}' detected")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Domain CLI - Comprehensive domain management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Show all domain status
  %(prog)s status user_management    # Show specific domain status
  %(prog)s validate                  # Validate all domains
  %(prog)s validate --fix           # Validate and auto-fix violations
  %(prog)s suggest --interactive     # Suggest and create domains interactively
  %(prog)s create user_management    # Create new domain package
  %(prog)s clean --dry-run          # Preview cleanup actions
  %(prog)s monitor --continuous     # Start continuous monitoring
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show domain package status')
    status_parser.add_argument('package', nargs='?', help='Specific package name')
    status_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate domain boundaries')
    validate_parser.add_argument('package', nargs='?', help='Specific package to validate')
    validate_parser.add_argument('--fix', action='store_true', help='Auto-fix violations')
    validate_parser.add_argument('--report', default='domain_violations_report.json', help='Report output file')
    
    # Suggest command
    suggest_parser = subparsers.add_parser('suggest', help='Suggest new domain packages')
    suggest_parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    suggest_parser.add_argument('--save', help='Save suggestions to file')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new domain package')
    create_parser.add_argument('name', help='Package name')
    create_parser.add_argument('--no-intelligent', action='store_true', help='Disable intelligent suggestions')
    create_parser.add_argument('--skip-samples', action='store_true', help='Skip sample files')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean domain packages')
    clean_parser.add_argument('package', nargs='?', help='Specific package to clean')
    clean_parser.add_argument('--dry-run', action='store_true', default=True, help='Preview actions only')
    clean_parser.add_argument('--execute', action='store_true', help='Actually perform cleaning')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor domain compliance')
    monitor_parser.add_argument('--continuous', action='store_true', help='Continuous monitoring')
    monitor_parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = DomainCLI()
    
    try:
        if args.command == 'status':
            result = cli.status(args.package)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                elif 'domains' in result:
                    # Multiple domains
                    print(f"📦 Found {result['total_domains']} domain packages")
                    print(f"📊 Health: {result['summary']['healthy_domains']}/{result['total_domains']} healthy ({result['summary']['health_percentage']:.1f}%)")
                    print(f"📈 Total: {result['summary']['total_entities']} entities, {result['summary']['total_services']} services, {result['summary']['total_tests']} tests")
                    
                    if result['summary']['total_violations'] > 0:
                        print(f"🚨 {result['summary']['total_violations']} violations found")
                    
                    print("\n📋 Domain Status:")
                    for name, domain in result['domains'].items():
                        status_emoji = "✅" if domain['healthy'] else "❌"
                        print(f"  {status_emoji} {name}: {domain['entities']} entities, {domain['services']} services, {domain['tests']} tests")
                        if domain['violations'] > 0:
                            print(f"    🚨 {domain['violations']} violations")
                else:
                    # Single domain
                    print(f"📦 Domain: {result['name']}")
                    print(f"📁 Path: {result['path']}")
                    print(f"📊 Entities: {len(result['entities'])}")
                    print(f"🔧 Services: {len(result['services'])}")
                    print(f"🧪 Tests: {result['tests']}")
                    
                    if result['violations'] > 0:
                        print(f"🚨 Violations: {result['violations']}")
                        print("📋 Recent violations:")
                        for v in result['detailed_violations'][:5]:
                            print(f"  • {v['term']} at {Path(v['file']).name}:{v['line']}")
                    else:
                        print("✅ No violations found")
        
        elif args.command == 'validate':
            result = cli.validate(args.package, args.fix)
            
            print(f"📊 Validation Summary:")
            print(f"  Violations: {result['summary']['total_violations']}")
            print(f"  Packages with violations: {result['summary']['packages_with_violations']}")
            print(f"  New domains detected: {result['summary']['new_domains_detected']}")
            
            if result['violations']:
                print(f"\n📋 Violations by package:")
                for package, violations in result['violations'].items():
                    print(f"  🚨 {package}: {len(violations)} violations")
            
            if result['new_domains']:
                print(f"\n🆕 New domains detected:")
                for domain in result['new_domains']:
                    confidence_emoji = "🟢" if domain['confidence'] > 0.8 else "🟡"
                    print(f"  {confidence_emoji} {domain['name']} (confidence: {domain['confidence']:.2f})")
            
            print(f"\n📄 Detailed report saved to: {result['report_file']}")
        
        elif args.command == 'suggest':
            suggestions = cli.suggest(args.interactive)
            
            if args.save and suggestions:
                with open(args.save, 'w') as f:
                    json.dump([
                        {
                            'name': s.name,
                            'confidence': s.confidence,
                            'concepts': s.concepts,
                            'files': s.files,
                            'reasoning': s.reasoning
                        } for s in suggestions
                    ], f, indent=2)
                print(f"\n💾 Suggestions saved to {args.save}")
        
        elif args.command == 'create':
            result = cli.create(
                args.name,
                intelligent=not args.no_intelligent,
                skip_samples=args.skip_samples
            )
            
            if result['success']:
                print(f"✅ Successfully created domain package: {args.name}")
            else:
                print(f"❌ Failed to create domain package: {args.name}")
                if result['error']:
                    print(f"Error: {result['error']}")
        
        elif args.command == 'clean':
            dry_run = not args.execute
            result = cli.clean(args.package, dry_run)
            
            if dry_run:
                print("🔍 Cleanup preview (use --execute to perform actions):")
            else:
                print("🧹 Cleanup completed:")
            
            for cleaned_package in result['cleaned']:
                print(f"\n📦 {cleaned_package['package']}:")
                for action in cleaned_package['actions']:
                    print(f"  • {action}")
            
            if result['errors']:
                print(f"\n❌ Errors:")
                for error in result['errors']:
                    print(f"  • {error}")
        
        elif args.command == 'monitor':
            cli.monitor(args.continuous, args.interval)
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()