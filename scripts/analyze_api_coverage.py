#!/usr/bin/env python3
"""
API Test Coverage Analysis Script

This script analyzes the actual test files and updates the API test coverage matrix.
It compares the endpoints defined in create_app() with the existing test files.
"""

import ast
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def extract_endpoints_from_create_app(app_file_path: str) -> Dict[str, List[Dict]]:
    """Extract all endpoints from the create_app() function."""
    endpoints = defaultdict(list)
    
    with open(app_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find app.include_router calls
    router_pattern = r'app\.include_router\(([^,]+)\.router(?:,\s*prefix="([^"]*)")?(?:,\s*tags=\[([^\]]*)\])?'
    router_matches = re.findall(router_pattern, content)
    
    for router_name, prefix, tags in router_matches:
        # Clean up the router name
        router_name = router_name.strip()
        prefix = prefix.strip() if prefix else ""
        tags = tags.strip(' "[]') if tags else ""
        
        endpoints[router_name].append({
            'prefix': prefix,
            'tags': tags,
            'router_name': router_name
        })
    
    return endpoints


def extract_endpoints_from_router_files(src_dir: str) -> Dict[str, List[Dict]]:
    """Extract endpoints from individual router files."""
    endpoints = {}
    
    # Find all router files
    router_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py') and ('endpoints' in root or 'routers' in root):
                router_files.append(os.path.join(root, file))
    
    for router_file in router_files:
        try:
            with open(router_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract router endpoints using regex
            endpoint_pattern = r'@router\.(get|post|put|patch|delete)\(["\']([^"\']+)["\']'
            endpoint_matches = re.findall(endpoint_pattern, content)
            
            router_name = os.path.basename(router_file).replace('.py', '')
            endpoints[router_name] = []
            
            for method, path in endpoint_matches:
                endpoints[router_name].append({
                    'method': method.upper(),
                    'path': path,
                    'file': router_file
                })
                
        except Exception as e:
            print(f"Error processing {router_file}: {e}")
    
    return endpoints


def extract_test_coverage(test_dir: str) -> Dict[str, Set[str]]:
    """Extract test coverage from test files."""
    coverage = defaultdict(set)
    
    # Find all test files
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract test function names
            test_pattern = r'def (test_[^(]+)\('
            test_matches = re.findall(test_pattern, content)
            
            # Extract HTTP calls from test content
            http_pattern = r'client\.(get|post|put|patch|delete)\(["\']([^"\']+)["\']'
            http_matches = re.findall(http_pattern, content)
            
            file_name = os.path.basename(test_file)
            coverage[file_name].update(test_matches)
            
            # Map HTTP calls to endpoints
            for method, path in http_matches:
                endpoint_key = f"{method.upper()} {path}"
                coverage[file_name].add(endpoint_key)
                
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
    
    return coverage


def analyze_coverage_status(endpoints: Dict, test_coverage: Dict) -> Dict[str, str]:
    """Analyze coverage status for each endpoint."""
    status = {}
    
    # Create a mapping of endpoint paths to test coverage
    tested_endpoints = set()
    for test_file, tests in test_coverage.items():
        for test in tests:
            if any(method in test for method in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']):
                tested_endpoints.add(test)
    
    # Check coverage for each endpoint
    for router_name, router_endpoints in endpoints.items():
        for endpoint in router_endpoints:
            method = endpoint.get('method', 'GET')
            path = endpoint.get('path', '')
            
            endpoint_key = f"{method} {path}"
            
            # Determine coverage status
            if endpoint_key in tested_endpoints:
                status[endpoint_key] = 'C'  # Covered
            elif any(path in test_key for test_key in tested_endpoints):
                status[endpoint_key] = 'P'  # Partial
            else:
                status[endpoint_key] = 'M'  # Missing
    
    return status


def generate_coverage_report(endpoints: Dict, test_coverage: Dict, output_file: str):
    """Generate a comprehensive coverage report."""
    coverage_status = analyze_coverage_status(endpoints, test_coverage)
    
    # Count coverage
    total = len(coverage_status)
    covered = sum(1 for s in coverage_status.values() if s == 'C')
    partial = sum(1 for s in coverage_status.values() if s == 'P')
    missing = sum(1 for s in coverage_status.values() if s == 'M')
    
    # Generate markdown report
    report = f"""# API Test Coverage Analysis Report

Generated on: {os.popen('date').read().strip()}

## Summary
- **Total Endpoints:** {total}
- **Covered (C):** {covered} ({covered/total*100:.1f}%)
- **Partial (P):** {partial} ({partial/total*100:.1f}%)
- **Missing (M):** {missing} ({missing/total*100:.1f}%)

## Detailed Coverage

"""
    
    for router_name, router_endpoints in endpoints.items():
        report += f"### {router_name}\n\n"
        report += "| Endpoint | Method | Coverage | Notes |\n"
        report += "|----------|--------|----------|-------|\n"
        
        for endpoint in router_endpoints:
            method = endpoint.get('method', 'GET')
            path = endpoint.get('path', '')
            endpoint_key = f"{method} {path}"
            status = coverage_status.get(endpoint_key, 'M')
            
            report += f"| `{path}` | {method} | {status} | |\n"
        
        report += "\n"
    
    # Test file analysis
    report += "## Test File Analysis\n\n"
    for test_file, tests in test_coverage.items():
        report += f"### {test_file}\n"
        report += f"- Test functions: {len([t for t in tests if t.startswith('test_')])}\n"
        report += f"- HTTP calls: {len([t for t in tests if any(m in t for m in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])])}\n\n"
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Coverage report generated: {output_file}")
    print(f"Total endpoints: {total}, Covered: {covered}, Partial: {partial}, Missing: {missing}")


def main():
    """Main function to run the coverage analysis."""
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    app_file = project_root / "src" / "pynomaly" / "presentation" / "api" / "app.py"
    src_dir = project_root / "src" / "pynomaly" / "presentation" / "api"
    test_dir = project_root / "tests" / "presentation" / "api"
    output_file = project_root / "docs" / "api_coverage_analysis.md"
    
    print("Starting API coverage analysis...")
    
    # Extract endpoints from router files
    print("Extracting endpoints from router files...")
    endpoints = extract_endpoints_from_router_files(str(src_dir))
    
    # Extract test coverage
    print("Analyzing test coverage...")
    test_coverage = extract_test_coverage(str(test_dir))
    
    # Generate report
    print("Generating coverage report...")
    generate_coverage_report(endpoints, test_coverage, str(output_file))
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
