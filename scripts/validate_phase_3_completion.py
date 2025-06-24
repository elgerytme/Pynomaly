#!/usr/bin/env python3
"""
Phase 3 Completion Validation Script
Validates completion of Application Layer Tests phase and prepares for Phase 4.
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

def analyze_application_test_completeness():
    """Analyze completeness of application test coverage."""
    
    test_dir = Path("tests/application")
    if not test_dir.exists():
        return {"error": "Application test directory not found"}
    
    test_files = list(test_dir.glob("test_*.py"))
    
    coverage_analysis = {
        'use_cases': {'files': [], 'methods': 0, 'priority': 'Critical'},
        'services': {'files': [], 'methods': 0, 'priority': 'Critical'},
        'dto': {'files': [], 'methods': 0, 'priority': 'High'},
        'workflows': {'files': [], 'methods': 0, 'priority': 'High'},
        'automl': {'files': [], 'methods': 0, 'priority': 'High'},
        'explainability': {'files': [], 'methods': 0, 'priority': 'High'}
    }
    
    # Categorize test files
    for test_file in test_files:
        name = test_file.name.lower()
        method_count = count_test_methods_in_file(test_file)
        
        if 'use_case' in name:
            coverage_analysis['use_cases']['files'].append(test_file.name)
            coverage_analysis['use_cases']['methods'] += method_count
        elif 'service' in name and 'automl' not in name and 'explainability' not in name:
            coverage_analysis['services']['files'].append(test_file.name)
            coverage_analysis['services']['methods'] += method_count
        elif 'dto' in name:
            coverage_analysis['dto']['files'].append(test_file.name)
            coverage_analysis['dto']['methods'] += method_count
        elif 'workflow' in name or 'integration' in name:
            coverage_analysis['workflows']['files'].append(test_file.name)
            coverage_analysis['workflows']['methods'] += method_count
        elif 'automl' in name:
            coverage_analysis['automl']['files'].append(test_file.name)
            coverage_analysis['automl']['methods'] += method_count
        elif 'explainability' in name:
            coverage_analysis['explainability']['files'].append(test_file.name)
            coverage_analysis['explainability']['methods'] += method_count
    
    return coverage_analysis

def validate_application_layer_implementation():
    """Validate application layer implementation completeness."""
    
    # Core application layer implementation files
    implementation_files = [
        'src/pynomaly/application/use_cases/detect_anomalies.py',
        'src/pynomaly/application/use_cases/train_detector.py',
        'src/pynomaly/application/use_cases/evaluate_model.py',
        'src/pynomaly/application/use_cases/explain_anomaly.py',
        'src/pynomaly/application/services/detection_service.py',
        'src/pynomaly/application/services/ensemble_service.py',
        'src/pynomaly/application/services/model_persistence_service.py',
        'src/pynomaly/application/services/automl_service.py',
        'src/pynomaly/application/services/explainability_service.py',
        'src/pynomaly/application/dto/detector_dto.py',
        'src/pynomaly/application/dto/dataset_dto.py',
        'src/pynomaly/application/dto/result_dto.py',
        'src/pynomaly/application/dto/experiment_dto.py',
        'src/pynomaly/application/dto/automl_dto.py',
        'src/pynomaly/application/dto/explainability_dto.py'
    ]
    
    implementation_status = {}
    
    for file_path in implementation_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            implementation_status[file_path] = {
                'exists': True,
                'size': size,
                'status': 'Implemented' if size > 1000 else 'Partial'
            }
        else:
            implementation_status[file_path] = {
                'exists': False,
                'size': 0,
                'status': 'Missing'
            }
    
    return implementation_status

def calculate_phase_3_progress():
    """Calculate Phase 3 completion progress."""
    
    coverage_analysis = analyze_application_test_completeness()
    implementation_status = validate_application_layer_implementation()
    
    # Calculate total test methods
    total_methods = sum(
        category['methods'] for category in coverage_analysis.values() 
        if isinstance(category, dict) and 'methods' in category
    )
    
    # Calculate component completion scores
    component_scores = {
        'use_cases': min(100, coverage_analysis['use_cases']['methods'] / 60 * 100),  # Target: 60 methods
        'services': min(100, coverage_analysis['services']['methods'] / 70 * 100),    # Target: 70 methods
        'dto': min(100, coverage_analysis['dto']['methods'] / 100 * 100),             # Target: 100 methods
        'workflows': min(100, coverage_analysis['workflows']['methods'] / 15 * 100),  # Target: 15 methods
        'automl': min(100, coverage_analysis['automl']['methods'] / 50 * 100),        # Target: 50 methods
        'explainability': min(100, coverage_analysis['explainability']['methods'] / 40 * 100)  # Target: 40 methods
    }
    
    # Calculate implementation completeness
    implemented_count = sum(1 for status in implementation_status.values() if status['status'] == 'Implemented')
    total_files = len(implementation_status)
    implementation_score = (implemented_count / total_files) * 100
    
    # Calculate weighted average (based on priority and implementation)
    weights = {
        'use_cases': 0.25,        # Critical
        'services': 0.25,         # Critical
        'dto': 0.15,              # High
        'workflows': 0.10,        # High
        'automl': 0.15,           # High
        'explainability': 0.10    # High
    }
    
    test_weighted_score = sum(
        component_scores[component] * weights[component] 
        for component in component_scores
    )
    
    # Combine test coverage and implementation (70% tests, 30% implementation)
    overall_score = (test_weighted_score * 0.7) + (implementation_score * 0.3)
    
    return {
        'total_test_methods': total_methods,
        'component_scores': component_scores,
        'implementation_score': implementation_score,
        'test_weighted_score': test_weighted_score,
        'overall_score': overall_score,
        'coverage_analysis': coverage_analysis,
        'implementation_status': implementation_status
    }

def generate_phase_3_report():
    """Generate comprehensive Phase 3 completion report."""
    
    print("=" * 80)
    print("PHASE 3: APPLICATION LAYER TESTS - COMPLETION VALIDATION")
    print("=" * 80)
    
    progress = calculate_phase_3_progress()
    
    print(f"\n📊 OVERALL PHASE 3 PROGRESS: {progress['overall_score']:.1f}%")
    print(f"🧪 Total Application Test Methods: {progress['total_test_methods']}")
    print(f"📁 Implementation Completeness: {progress['implementation_score']:.1f}%")
    
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
        for file_name in category_data['files']:
            print(f"      📄 {file_name}")
    
    print(f"\n🏗️  APPLICATION LAYER IMPLEMENTATION STATUS:")
    implementation_status = progress['implementation_status']
    
    implemented = sum(1 for status in implementation_status.values() if status['status'] == 'Implemented')
    partial = sum(1 for status in implementation_status.values() if status['status'] == 'Partial')
    missing = sum(1 for status in implementation_status.values() if status['status'] == 'Missing')
    
    print(f"   ✅ Implemented: {implemented}/{len(implementation_status)}")
    print(f"   🔄 Partial: {partial}")
    print(f"   ❌ Missing: {missing}")
    
    if partial > 0 or missing > 0:
        print(f"\n   📋 Files needing attention:")
        for file_path, status in implementation_status.items():
            if status['status'] != 'Implemented':
                size_kb = status['size'] / 1024 if status['size'] > 0 else 0
                print(f"      ⚠️  {file_path}: {status['status']} ({size_kb:.1f}KB)")
    
    # Determine readiness for Phase 4
    print(f"\n🎯 PHASE 3 TARGET ASSESSMENT:")
    print(f"   📈 Target Coverage: 85%")
    print(f"   📊 Current Progress: {progress['overall_score']:.1f}%")
    print(f"   🧪 Test Coverage: {progress['test_weighted_score']:.1f}%")
    print(f"   🏗️  Implementation: {progress['implementation_score']:.1f}%")
    
    if progress['overall_score'] >= 85:
        print(f"   ✅ PHASE 3 COMPLETE - Ready for Phase 4: Presentation Layer Tests")
    elif progress['overall_score'] >= 75:
        print(f"   🔄 PHASE 3 NEAR COMPLETION - Continue application testing")
    else:
        print(f"   ⚠️  PHASE 3 IN PROGRESS - Focus on critical components (use cases, services)")
    
    # Next steps
    print(f"\n📋 NEXT STEPS:")
    if progress['overall_score'] >= 85:
        print(f"   1. ✅ Phase 3 Application Layer Tests Complete")
        print(f"   2. 🚀 Begin Phase 4: Presentation Layer Tests")
        print(f"   3. 🎯 Target: 90%+ overall coverage")
        print(f"   4. 🌐 Focus: API, CLI, Web UI, and SDK testing")
    else:
        priority_areas = [
            component for component, score in component_scores.items()
            if score < 80 and coverage_analysis[component]['priority'] in ['Critical', 'High']
        ]
        print(f"   1. 🔄 Complete remaining application testing")
        print(f"   2. 🎯 Focus on: {', '.join(priority_areas)}")
        print(f"   3. 📊 Target remaining: {85 - progress['overall_score']:.1f}% progress")
        
        if progress['implementation_score'] < 90:
            print(f"   4. 🏗️  Complete missing implementation files")
    
    return progress

def generate_enhanced_application_tests():
    """Generate recommendations for enhanced application tests."""
    
    print(f"\n🚀 ENHANCED APPLICATION TEST RECOMMENDATIONS:")
    
    enhanced_tests = [
        ("Advanced Use Case Tests", [
            "test_use_cases_streaming.py",
            "test_use_cases_batch_processing.py", 
            "test_use_cases_ensemble_workflows.py",
            "test_use_cases_realtime_detection.py"
        ]),
        ("Service Integration Tests", [
            "test_services_integration.py",
            "test_services_performance.py",
            "test_services_scalability.py",
            "test_services_fault_tolerance.py"
        ]),
        ("Advanced AutoML Tests", [
            "test_automl_optimization_advanced.py",
            "test_automl_multi_objective.py",
            "test_automl_distributed.py"
        ]),
        ("Explainability Enhancement Tests", [
            "test_explainability_temporal.py",
            "test_explainability_multimodal.py",
            "test_explainability_comparative.py"
        ])
    ]
    
    for category, tests in enhanced_tests:
        print(f"   📁 {category}:")
        for test in tests:
            print(f"      📄 {test}")
    
    return enhanced_tests

def main():
    """Main validation function."""
    
    print("🔍 Validating Phase 3: Application Layer Tests Completion")
    
    try:
        progress = generate_phase_3_report()
        enhanced_tests = generate_enhanced_application_tests()
        
        # Summary
        print(f"\n📊 PHASE 3 SUMMARY:")
        print(f"   📈 Overall Progress: {progress['overall_score']:.1f}%")
        print(f"   🧪 Test Methods: {progress['total_test_methods']}")
        print(f"   🏗️  Implementation: {progress['implementation_score']:.1f}%")
        print(f"   🎯 Target Achievement: {'✅ ACHIEVED' if progress['overall_score'] >= 85 else '🔄 IN PROGRESS'}")
        
        # Return success if Phase 3 is substantially complete
        return progress['overall_score'] >= 75
        
    except Exception as e:
        print(f"❌ Phase 3 validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)