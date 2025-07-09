#!/usr/bin/env python3
"""
Generate trend analysis for test results and coverage.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_test_results(file_path: str) -> dict:
    """Parse JUnit XML test results."""
    if not os.path.exists(file_path):
        print(f"Test results file not found: {file_path}")
        return {}
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract test statistics
    total_tests = int(root.get('tests', 0))
    failures = int(root.get('failures', 0))
    errors = int(root.get('errors', 0))
    skipped = int(root.get('skipped', 0))
    time = float(root.get('time', 0))
    
    passed = total_tests - failures - errors - skipped
    
    return {
        'total_tests': total_tests,
        'passed': passed,
        'failures': failures,
        'errors': errors,
        'skipped': skipped,
        'pass_rate': (passed / total_tests * 100) if total_tests > 0 else 0,
        'execution_time': time,
        'timestamp': datetime.now().isoformat()
    }


def parse_coverage_results(file_path: str) -> dict:
    """Parse coverage XML results."""
    if not os.path.exists(file_path):
        print(f"Coverage file not found: {file_path}")
        return {}
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract coverage statistics
    line_rate = float(root.get('line-rate', 0)) * 100
    branch_rate = float(root.get('branch-rate', 0)) * 100
    lines_covered = int(root.get('lines-covered', 0))
    lines_valid = int(root.get('lines-valid', 0))
    branches_covered = int(root.get('branches-covered', 0))
    branches_valid = int(root.get('branches-valid', 0))
    
    return {
        'line_coverage': line_rate,
        'branch_coverage': branch_rate,
        'lines_covered': lines_covered,
        'lines_valid': lines_valid,
        'branches_covered': branches_covered,
        'branches_valid': branches_valid,
        'timestamp': datetime.now().isoformat()
    }


def load_historical_data(file_path: str) -> list:
    """Load historical trend data."""
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_trend_data(data: list, file_path: str):
    """Save trend data to file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"Error saving trend data: {e}")


def calculate_trends(historical_data: list) -> dict:
    """Calculate trend metrics."""
    if len(historical_data) < 2:
        return {
            'test_trend': 'stable',
            'coverage_trend': 'stable',
            'recent_changes': []
        }
    
    recent = historical_data[-1]
    previous = historical_data[-2]
    
    # Calculate test trends
    test_change = recent['test_data']['pass_rate'] - previous['test_data']['pass_rate']
    if test_change > 5:
        test_trend = 'improving'
    elif test_change < -5:
        test_trend = 'declining'
    else:
        test_trend = 'stable'
    
    # Calculate coverage trends
    coverage_change = recent['coverage_data']['line_coverage'] - previous['coverage_data']['line_coverage']
    if coverage_change > 2:
        coverage_trend = 'improving'
    elif coverage_change < -2:
        coverage_trend = 'declining'
    else:
        coverage_trend = 'stable'
    
    # Identify recent changes
    recent_changes = []
    if test_change != 0:
        recent_changes.append(f"Test pass rate changed by {test_change:.1f}%")
    if coverage_change != 0:
        recent_changes.append(f"Coverage changed by {coverage_change:.1f}%")
    
    return {
        'test_trend': test_trend,
        'coverage_trend': coverage_trend,
        'recent_changes': recent_changes,
        'test_change': test_change,
        'coverage_change': coverage_change
    }


def generate_quality_score(test_data: dict, coverage_data: dict) -> dict:
    """Generate overall quality score."""
    # Weighted scoring
    test_weight = 0.4
    coverage_weight = 0.6
    
    test_score = test_data.get('pass_rate', 0)
    coverage_score = coverage_data.get('line_coverage', 0)
    
    overall_score = (test_score * test_weight) + (coverage_score * coverage_weight)
    
    # Determine quality level
    if overall_score >= 95:
        level = 'excellent'
    elif overall_score >= 90:
        level = 'good'
    elif overall_score >= 80:
        level = 'fair'
    else:
        level = 'needs_improvement'
    
    return {
        'overall_score': overall_score,
        'quality_level': level,
        'test_contribution': test_score * test_weight,
        'coverage_contribution': coverage_score * coverage_weight
    }


def main():
    parser = argparse.ArgumentParser(description='Generate trend analysis for test results and coverage')
    parser.add_argument('--test-results', required=True, help='Path to JUnit XML test results file')
    parser.add_argument('--coverage-results', required=True, help='Path to coverage XML file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--historical-data', default='trend-history.json', help='Historical trend data file')
    
    args = parser.parse_args()
    
    # Parse current results
    test_data = parse_test_results(args.test_results)
    coverage_data = parse_coverage_results(args.coverage_results)
    
    if not test_data or not coverage_data:
        print("Error: Could not parse test or coverage results")
        return 1
    
    # Load historical data
    historical_data = load_historical_data(args.historical_data)
    
    # Create current entry
    current_entry = {
        'timestamp': datetime.now().isoformat(),
        'test_data': test_data,
        'coverage_data': coverage_data,
        'quality_score': generate_quality_score(test_data, coverage_data)
    }
    
    # Add to historical data
    historical_data.append(current_entry)
    
    # Keep only last 30 entries
    historical_data = historical_data[-30:]
    
    # Calculate trends
    trends = calculate_trends(historical_data)
    
    # Generate final analysis
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'current_results': current_entry,
        'trends': trends,
        'historical_summary': {
            'entries_count': len(historical_data),
            'date_range': {
                'start': historical_data[0]['timestamp'] if historical_data else None,
                'end': historical_data[-1]['timestamp'] if historical_data else None
            }
        },
        'recommendations': []
    }
    
    # Add recommendations
    if trends['test_trend'] == 'declining':
        analysis['recommendations'].append("Test pass rate is declining. Review recent changes and fix failing tests.")
    
    if trends['coverage_trend'] == 'declining':
        analysis['recommendations'].append("Code coverage is declining. Add tests for new code and improve existing coverage.")
    
    if current_entry['quality_score']['overall_score'] < 90:
        analysis['recommendations'].append("Overall quality score is below 90%. Focus on improving both test coverage and pass rates.")
    
    # Save updated historical data
    save_trend_data(historical_data, args.historical_data)
    
    # Save analysis
    try:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"✅ Trend analysis saved to {args.output}")
    except IOError as e:
        print(f"❌ Error saving analysis: {e}")
        return 1
    
    # Print summary
    print("\n📊 Trend Analysis Summary")
    print("=" * 30)
    print(f"Test Pass Rate: {test_data['pass_rate']:.1f}% ({trends['test_trend']})")
    print(f"Line Coverage: {coverage_data['line_coverage']:.1f}% ({trends['coverage_trend']})")
    print(f"Quality Score: {current_entry['quality_score']['overall_score']:.1f}% ({current_entry['quality_score']['quality_level']})")
    
    if trends['recent_changes']:
        print("\nRecent Changes:")
        for change in trends['recent_changes']:
            print(f"  • {change}")
    
    if analysis['recommendations']:
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
