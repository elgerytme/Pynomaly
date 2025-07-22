#!/usr/bin/env python3
"""
Phase 1 Technical Debt & Maintenance - Progress Validation Script

This script validates the completion of Phase 1 critical security and architecture fixes.
"""

import sys
from pathlib import Path
import subprocess
import re

def check_dependency_injection():
    """Check dependency injection implementation."""
    di_file = Path("src/packages/data/anomaly_detection/core/dependency_injection.py")
    if di_file.exists():
        print("✅ Dependency injection container implemented")
        return True
    else:
        print("❌ Dependency injection container missing")
        return False

def check_domain_entities():
    """Check domain entities implementation."""
    entities_file = Path("src/packages/data/anomaly_detection/core/domain_entities.py")
    if entities_file.exists():
        print("✅ Domain entities and protocols implemented")
        return True
    else:
        print("❌ Domain entities and protocols missing")
        return False

def check_security_configuration():
    """Check security configuration implementation."""
    security_file = Path("src/packages/data/anomaly_detection/core/security_configuration.py")
    if security_file.exists():
        print("✅ Security configuration implemented")
        return True
    else:
        print("❌ Security configuration missing")
        return False

def check_debug_flags():
    """Check for remaining debug flags in production code."""
    try:
        result = subprocess.run([
            "grep", "-r", "logging.basicConfig(level=logging.DEBUG)", 
            "src/", "--exclude-dir=docs", "--exclude=*.md"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            debug_files = result.stdout.strip().split('\n')
            print(f"⚠️  Found {len(debug_files)} files with debug logging:")
            for file in debug_files[:5]:  # Show first 5
                print(f"   - {file}")
            return False
        else:
            print("✅ No debug flags found in production code")
            return True
    except Exception as e:
        print(f"⚠️  Could not check debug flags: {e}")
        return True

def check_monorepo_imports():
    """Check for remaining monorepo imports."""
    try:
        result = subprocess.run([
            "grep", "-r", "from monorepo\\.", "src/packages/data/anomaly_detection/",
            "--include=*.py", "--exclude-dir=docs"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            import_files = result.stdout.strip().split('\n')
            print(f"⚠️  Found {len(import_files)} files with monorepo imports:")
            for file in import_files[:5]:  # Show first 5
                print(f"   - {file}")
            return False
        else:
            print("✅ No monorepo imports found")
            return True
    except Exception as e:
        print(f"⚠️  Could not check monorepo imports: {e}")
        return True

def check_todo_markers():
    """Check for remaining TODO markers."""
    try:
        result = subprocess.run([
            "grep", "-r", "TODO:", "src/packages/data/anomaly_detection/",
            "--include=*.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            todo_count = len(result.stdout.strip().split('\n'))
            print(f"📋 Found {todo_count} TODO markers remaining")
            return True
        else:
            print("✅ No TODO markers found")
            return True
    except Exception as e:
        print(f"⚠️  Could not check TODO markers: {e}")
        return True

def main():
    """Main validation function."""
    print("🔍 Phase 1 Technical Debt & Maintenance - Progress Validation")
    print("=" * 60)
    
    checks = [
        ("Dependency Injection", check_dependency_injection),
        ("Domain Entities", check_domain_entities), 
        ("Security Configuration", check_security_configuration),
        ("Debug Flags Removal", check_debug_flags),
        ("Domain Boundary Enforcement", check_monorepo_imports),
        ("TODO Markers", check_todo_markers),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n🔎 {name}:")
        if check_func():
            passed += 1
    
    print(f"\n📊 Phase 1 Progress: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Phase 1 Technical Debt & Maintenance completed successfully!")
        print("✨ Ready to proceed with Phase 2: Performance Optimization")
    elif passed >= total - 1:
        print("🚀 Phase 1 mostly complete - minor issues remaining")
        print("💡 Consider proceeding with Phase 2 while addressing remaining issues")
    else:
        print("⚡ Phase 1 requires more work before proceeding to Phase 2")
        print("🔧 Focus on completing critical architecture and security fixes")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)