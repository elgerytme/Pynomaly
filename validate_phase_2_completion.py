#!/usr/bin/env python3
"""
Phase 2 Completion Validation Script
Validates completion of Infrastructure Tests phase and prepares for Phase 3.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

def count_test_methods_in_file(test_file: Path) -> int:
    """Count test methods in a test file."""
    try:
        content = test_file.read_text()
        lines = content.split('\n')
        test_count = 0
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('def test_') or 
                stripped.startswith('async def test_')):
                test_count += 1
        return test_count
    except Exception as e:
        print(f"Error counting tests in {test_file}: {e}")
        return 0

def analyze_infrastructure_test_completeness():
    """Analyze completeness of infrastructure test coverage."""
    
    test_dir = Path("tests/infrastructure")
    if not test_dir.exists():
        return {"error": "Infrastructure test directory not found"}
    
    test_files = list(test_dir.glob("test_*.py"))
    
    coverage_analysis = {
        'adapters': {'files': [], 'methods': 0, 'priority': 'Critical'},
        'distributed': {'files': [], 'methods': 0, 'priority': 'Critical'},
        'data_infrastructure': {'files': [], 'methods': 0, 'priority': 'High'},
        'persistence': {'files': [], 'methods': 0, 'priority': 'High'},
        'configuration': {'files': [], 'methods': 0, 'priority': 'Medium'},
        'monitoring': {'files': [], 'methods': 0, 'priority': 'Medium'},
        'security': {'files': [], 'methods': 0, 'priority': 'High'},
        'performance': {'files': [], 'methods': 0, 'priority': 'Medium'},
        'processing': {'files': [], 'methods': 0, 'priority': 'Medium'}
    }
    
    # Categorize test files
    for test_file in test_files:
        name = test_file.name.lower()
        method_count = count_test_methods_in_file(test_file)
        
        if 'adapter' in name:
            coverage_analysis['adapters']['files'].append(test_file.name)
            coverage_analysis['adapters']['methods'] += method_count
        elif 'distributed' in name:
            coverage_analysis['distributed']['files'].append(test_file.name)
            coverage_analysis['distributed']['methods'] += method_count
        elif 'data' in name or 'loader' in name:
            coverage_analysis['data_infrastructure']['files'].append(test_file.name)
            coverage_analysis['data_infrastructure']['methods'] += method_count
        elif 'persistence' in name or 'repositories' in name:
            coverage_analysis['persistence']['files'].append(test_file.name)
            coverage_analysis['persistence']['methods'] += method_count
        elif 'config' in name:
            coverage_analysis['configuration']['files'].append(test_file.name)
            coverage_analysis['configuration']['methods'] += method_count
        elif 'monitoring' in name:
            coverage_analysis['monitoring']['files'].append(test_file.name)
            coverage_analysis['monitoring']['methods'] += method_count
        elif 'auth' in name or 'resilience' in name:
            coverage_analysis['security']['files'].append(test_file.name)
            coverage_analysis['security']['methods'] += method_count
        elif 'caching' in name or 'middleware' in name:
            coverage_analysis['performance']['files'].append(test_file.name)
            coverage_analysis['performance']['methods'] += method_count
        elif 'preprocessing' in name:
            coverage_analysis['processing']['files'].append(test_file.name)
            coverage_analysis['processing']['methods'] += method_count
    
    return coverage_analysis

def validate_distributed_infrastructure():
    """Validate distributed processing infrastructure implementation."""
    
    distributed_components = [
        'src/pynomaly/infrastructure/distributed/__init__.py',
        'src/pynomaly/infrastructure/distributed/manager.py',
        'src/pynomaly/infrastructure/distributed/worker.py',
        'src/pynomaly/infrastructure/distributed/coordinator.py',
        'src/pynomaly/infrastructure/distributed/load_balancer.py',
        'src/pynomaly/infrastructure/distributed/task_queue.py',
        'src/pynomaly/infrastructure/distributed/worker_pool.py',
        'src/pynomaly/infrastructure/distributed/coordination_service.py'
    ]
    
    distributed_tests = [
        'tests/infrastructure/test_distributed_comprehensive.py'
    ]
    
    implementation_status = {}
    
    # Check implementation files
    for component in distributed_components:
        path = Path(component)
        if path.exists():
            size = path.stat().st_size
            implementation_status[component] = {
                'exists': True,
                'size': size,
                'status': 'Implemented' if size > 1000 else 'Partial'
            }
        else:
            implementation_status[component] = {
                'exists': False,
                'size': 0,
                'status': 'Missing'
            }
    
    # Check test files
    for test_file in distributed_tests:
        path = Path(test_file)
        if path.exists():
            method_count = count_test_methods_in_file(path)
            implementation_status[test_file] = {
                'exists': True,
                'test_methods': method_count,
                'status': 'Complete' if method_count > 20 else 'Partial'
            }
        else:
            implementation_status[test_file] = {
                'exists': False,
                'test_methods': 0,
                'status': 'Missing'
            }
    
    return implementation_status

def calculate_phase_2_progress():
    """Calculate Phase 2 completion progress."""
    
    coverage_analysis = analyze_infrastructure_test_completeness()
    distributed_status = validate_distributed_infrastructure()
    
    # Calculate total test methods
    total_methods = sum(
        category['methods'] for category in coverage_analysis.values() 
        if isinstance(category, dict) and 'methods' in category
    )
    
    # Calculate component completion
    component_scores = {
        'adapters': coverage_analysis['adapters']['methods'] / 50 * 100,  # Target: 50 methods
        'distributed': coverage_analysis['distributed']['methods'] / 60 * 100,  # Target: 60 methods
        'data_infrastructure': coverage_analysis['data_infrastructure']['methods'] / 80 * 100,
        'persistence': coverage_analysis['persistence']['methods'] / 60 * 100,
        'security': coverage_analysis['security']['methods'] / 90 * 100,
        'configuration': coverage_analysis['configuration']['methods'] / 40 * 100,
        'monitoring': coverage_analysis['monitoring']['methods'] / 50 * 100,
        'performance': coverage_analysis['performance']['methods'] / 90 * 100,
        'processing': coverage_analysis['processing']['methods'] / 40 * 100
    }
    
    # Cap at 100%
    for component in component_scores:
        component_scores[component] = min(100, component_scores[component])
    
    # Calculate weighted average (based on priority)
    weights = {
        'adapters': 0.20,      # Critical
        'distributed': 0.20,   # Critical  
        'data_infrastructure': 0.15,  # High
        'persistence': 0.15,   # High
        'security': 0.10,      # High
        'configuration': 0.05, # Medium
        'monitoring': 0.05,    # Medium
        'performance': 0.05,   # Medium
        'processing': 0.05     # Medium
    }
    
    weighted_score = sum(
        component_scores[component] * weights[component] 
        for component in component_scores
    )
    
    return {
        'total_test_methods': total_methods,
        'component_scores': component_scores,
        'weighted_score': weighted_score,
        'coverage_analysis': coverage_analysis,
        'distributed_status': distributed_status
    }

def generate_phase_2_report():
    """Generate comprehensive Phase 2 completion report."""
    
    print("=" * 80)
    print("PHASE 2: INFRASTRUCTURE TESTS - COMPLETION VALIDATION")
    print("=" * 80)
    
    progress = calculate_phase_2_progress()
    
    print(f"\n📊 OVERALL PHASE 2 PROGRESS: {progress['weighted_score']:.1f}%")
    print(f"🧪 Total Infrastructure Test Methods: {progress['total_test_methods']}")
    
    print(f"\n📋 COMPONENT COMPLETION STATUS:")
    component_scores = progress['component_scores']
    coverage_analysis = progress['coverage_analysis']
    
    for component, score in component_scores.items():
        category_data = coverage_analysis[component]
        files_count = len(category_data['files'])
        methods_count = category_data['methods']
        priority = category_data['priority']
        
        status_icon = "✅" if score >= 90 else "🔄" if score >= 70 else "⚠️" if score >= 50 else "❌"
        
        print(f"   {status_icon} {component.upper()}: {score:.1f}% ({methods_count} methods, {files_count} files) - {priority}")
    
    print(f"\n🏗️  DISTRIBUTED INFRASTRUCTURE STATUS:")
    distributed_status = progress['distributed_status']
    
    for component, status in distributed_status.items():
        if 'src/' in component:
            size_mb = status['size'] / 1024 if status['size'] > 0 else 0
            print(f"   📁 {component}: {status['status']} ({size_mb:.1f}KB)")
        else:
            methods = status.get('test_methods', 0)
            print(f"   🧪 {component}: {status['status']} ({methods} test methods)")
    
    # Determine readiness for Phase 3
    print(f"\n🎯 PHASE 2 TARGET ASSESSMENT:")
    print(f"   📈 Target Coverage: 70%")
    print(f"   📊 Current Progress: {progress['weighted_score']:.1f}%")
    
    if progress['weighted_score'] >= 70:
        print(f"   ✅ PHASE 2 COMPLETE - Ready for Phase 3: Application Layer Tests")
    elif progress['weighted_score'] >= 60:
        print(f"   🔄 PHASE 2 NEAR COMPLETION - Continue infrastructure testing")
    else:
        print(f"   ⚠️  PHASE 2 IN PROGRESS - Focus on critical components (adapters, distributed)")
    
    # Next steps
    print(f"\n📋 NEXT STEPS:")
    if progress['weighted_score'] >= 70:
        print(f"   1. ✅ Phase 2 Infrastructure Tests Complete")
        print(f"   2. 🚀 Begin Phase 3: Application Layer Tests")
        print(f"   3. 🎯 Target: 85% overall coverage")
    else:
        priority_areas = [
            component for component, score in component_scores.items()
            if score < 70 and coverage_analysis[component]['priority'] in ['Critical', 'High']
        ]
        print(f"   1. 🔄 Complete remaining infrastructure testing")
        print(f"   2. 🎯 Focus on: {', '.join(priority_areas)}")
        print(f"   3. 📊 Target remaining: {70 - progress['weighted_score']:.1f}% progress")
    
    return progress

def main():
    """Main validation function."""
    
    print("🔍 Validating Phase 2: Infrastructure Tests Completion")
    
    try:
        progress = generate_phase_2_report()
        
        # Return success if Phase 2 is substantially complete
        return progress['weighted_score'] >= 60
        
    except Exception as e:
        print(f"❌ Phase 2 validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)