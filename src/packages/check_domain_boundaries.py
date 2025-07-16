#!/usr/bin/env python3
"""Script to check domain boundaries and generate report."""

import json
from infrastructure.quality.domain_boundary_validator import DomainBoundaryValidator


def main():
    """Run domain boundary validation."""
    validator = DomainBoundaryValidator("/mnt/c/Users/andre/Pynomaly/src/packages")
    
    print("🔍 Checking domain boundaries...")
    report = validator.generate_dependency_report()
    
    print(f"\n📊 DOMAIN BOUNDARY REPORT")
    print("=" * 50)
    print(f"Total packages: {report['total_packages']}")
    print(f"Packages with violations: {report['packages_with_violations']}")
    print(f"Total violations: {report['summary']['total_violations']}")
    
    if report['violations']:
        print(f"\n❌ VIOLATIONS FOUND:")
        for package, violations in report['violations'].items():
            print(f"\n📦 {package} ({len(violations)} violations):")
            for violation in violations[:3]:  # Show first 3
                print(f"  - {violation}")
            if len(violations) > 3:
                print(f"  ... and {len(violations) - 3} more")
    
    if report['clean_packages']:
        print(f"\n✅ CLEAN PACKAGES:")
        for package in report['clean_packages']:
            print(f"  - {package}")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    recommendations = validator.generate_fix_recommendations()
    for rec in recommendations[:10]:  # Show first 10
        print(rec)
    
    # Save detailed report
    with open('/mnt/c/Users/andre/Pynomaly/src/packages/domain_boundary_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📝 Detailed report saved to domain_boundary_report.json")


if __name__ == "__main__":
    main()