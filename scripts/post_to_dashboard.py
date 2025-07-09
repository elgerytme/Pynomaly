#!/usr/bin/env python3
"""
Post test results and trends to project dashboard.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import requests


def load_json_file(file_path: str) -> dict:
    """Load JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def post_to_webhook(webhook_url: str, data: dict) -> bool:
    """Post data to webhook."""
    try:
        response = requests.post(webhook_url, json=data, timeout=30)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error posting to webhook: {e}")
        return False


def generate_dashboard_data(test_results: dict, coverage_results: dict, trend_analysis: dict) -> dict:
    """Generate dashboard data payload."""
    return {
        'timestamp': datetime.now().isoformat(),
        'project': 'pynomaly',
        'test_summary': {
            'total_files': 424,  # Known test file count
            'pass_rate': test_results.get('pass_rate', 0),
            'total_tests': test_results.get('total_tests', 0),
            'failed_tests': test_results.get('failures', 0) + test_results.get('errors', 0),
            'execution_time': test_results.get('execution_time', 0)
        },
        'coverage_summary': {
            'line_coverage': coverage_results.get('line_coverage', 0),
            'branch_coverage': coverage_results.get('branch_coverage', 0),
            'lines_covered': coverage_results.get('lines_covered', 0),
            'lines_valid': coverage_results.get('lines_valid', 0)
        },
        'quality_metrics': {
            'overall_score': trend_analysis.get('current_results', {}).get('quality_score', {}).get('overall_score', 0),
            'quality_level': trend_analysis.get('current_results', {}).get('quality_score', {}).get('quality_level', 'unknown'),
            'test_trend': trend_analysis.get('trends', {}).get('test_trend', 'stable'),
            'coverage_trend': trend_analysis.get('trends', {}).get('coverage_trend', 'stable')
        },
        'recommendations': trend_analysis.get('recommendations', []),
        'artifacts': {
            'test_report_url': f"https://github.com/pynomaly/pynomaly/actions/runs/{os.getenv('GITHUB_RUN_ID', 'unknown')}",
            'coverage_report_url': f"https://github.com/pynomaly/pynomaly/actions/runs/{os.getenv('GITHUB_RUN_ID', 'unknown')}",
            'trend_analysis_url': f"https://github.com/pynomaly/pynomaly/actions/runs/{os.getenv('GITHUB_RUN_ID', 'unknown')}"
        }
    }


def generate_slack_message(dashboard_data: dict) -> dict:
    """Generate Slack message payload."""
    test_summary = dashboard_data['test_summary']
    coverage_summary = dashboard_data['coverage_summary']
    quality_metrics = dashboard_data['quality_metrics']
    
    # Determine color based on quality
    if quality_metrics['overall_score'] >= 95:
        color = 'good'
        status_emoji = '🟢'
    elif quality_metrics['overall_score'] >= 90:
        color = 'warning'
        status_emoji = '🟡'
    else:
        color = 'danger'
        status_emoji = '🔴'
    
    # Build message
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_emoji} Pynomaly Nightly Test Results"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Overall Quality Score:* {quality_metrics['overall_score']:.1f}% ({quality_metrics['quality_level']})\n*Test Pass Rate:* {test_summary['pass_rate']:.1f}% ({quality_metrics['test_trend']})\n*Line Coverage:* {coverage_summary['line_coverage']:.1f}% ({quality_metrics['coverage_trend']})"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Total Tests:* {test_summary['total_tests']:,}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Failed Tests:* {test_summary['failed_tests']}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Execution Time:* {test_summary['execution_time']:.1f}s"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Lines Covered:* {coverage_summary['lines_covered']:,}/{coverage_summary['lines_valid']:,}"
                }
            ]
        }
    ]
    
    # Add recommendations if any
    if dashboard_data['recommendations']:
        recommendations_text = "\\n".join([f"• {rec}" for rec in dashboard_data['recommendations']])
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Recommendations:*\\n{recommendations_text}"
            }
        })
    
    # Add action buttons
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Test Report"
                },
                "url": dashboard_data['artifacts']['test_report_url']
            },
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "View Coverage Report"
                },
                "url": dashboard_data['artifacts']['coverage_report_url']
            }
        ]
    })
    
    return {
        "attachments": [
            {
                "color": color,
                "blocks": blocks
            }
        ]
    }


def save_dashboard_data(data: dict, file_path: str):
    """Save dashboard data to file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Dashboard data saved to {file_path}")
    except IOError as e:
        print(f"❌ Error saving dashboard data: {e}")


def main():
    parser = argparse.ArgumentParser(description='Post test results to project dashboard')
    parser.add_argument('--test-results', required=True, help='Path to test results JSON')
    parser.add_argument('--coverage-results', required=True, help='Path to coverage results JSON')
    parser.add_argument('--trend-analysis', required=True, help='Path to trend analysis JSON')
    parser.add_argument('--comprehensive-report', help='Path to comprehensive HTML report')
    parser.add_argument('--dashboard-webhook', help='Dashboard webhook URL')
    parser.add_argument('--slack-webhook', help='Slack webhook URL')
    parser.add_argument('--output', default='dashboard-data.json', help='Output dashboard data file')
    
    args = parser.parse_args()
    
    # Load input data
    print("📊 Loading test results and analysis...")
    
    # For test results, we need to parse the XML first
    import xml.etree.ElementTree as ET
    
    test_data = {}
    if os.path.exists(args.test_results):
        try:
            tree = ET.parse(args.test_results)
            root = tree.getroot()
            
            total_tests = int(root.get('tests', 0))
            failures = int(root.get('failures', 0))
            errors = int(root.get('errors', 0))
            time = float(root.get('time', 0))
            passed = total_tests - failures - errors
            
            test_data = {
                'total_tests': total_tests,
                'failures': failures,
                'errors': errors,
                'pass_rate': (passed / total_tests * 100) if total_tests > 0 else 0,
                'execution_time': time
            }
        except Exception as e:
            print(f"Error parsing test results: {e}")
    
    # For coverage results, parse XML
    coverage_data = {}
    if os.path.exists(args.coverage_results):
        try:
            tree = ET.parse(args.coverage_results)
            root = tree.getroot()
            
            coverage_data = {
                'line_coverage': float(root.get('line-rate', 0)) * 100,
                'branch_coverage': float(root.get('branch-rate', 0)) * 100,
                'lines_covered': int(root.get('lines-covered', 0)),
                'lines_valid': int(root.get('lines-valid', 0))
            }
        except Exception as e:
            print(f"Error parsing coverage results: {e}")
    
    # Load trend analysis
    trend_analysis = load_json_file(args.trend_analysis)
    
    if not test_data or not coverage_data:
        print("❌ Error: Could not load required data files")
        return 1
    
    # Generate dashboard data
    dashboard_data = generate_dashboard_data(test_data, coverage_data, trend_analysis)
    
    # Save dashboard data
    save_dashboard_data(dashboard_data, args.output)
    
    # Post to dashboard webhook if provided
    if args.dashboard_webhook:
        print("📡 Posting to dashboard webhook...")
        success = post_to_webhook(args.dashboard_webhook, dashboard_data)
        if success:
            print("✅ Successfully posted to dashboard webhook")
        else:
            print("❌ Failed to post to dashboard webhook")
    
    # Post to Slack if webhook provided
    if args.slack_webhook:
        print("📡 Posting to Slack...")
        slack_message = generate_slack_message(dashboard_data)
        success = post_to_webhook(args.slack_webhook, slack_message)
        if success:
            print("✅ Successfully posted to Slack")
        else:
            print("❌ Failed to post to Slack")
    
    # Print summary
    print("\\n📊 Dashboard Summary")
    print("=" * 30)
    print(f"Quality Score: {dashboard_data['quality_metrics']['overall_score']:.1f}%")
    print(f"Test Pass Rate: {dashboard_data['test_summary']['pass_rate']:.1f}%")
    print(f"Line Coverage: {dashboard_data['coverage_summary']['line_coverage']:.1f}%")
    print(f"Total Tests: {dashboard_data['test_summary']['total_tests']:,}")
    print(f"Failed Tests: {dashboard_data['test_summary']['failed_tests']}")
    
    if dashboard_data['recommendations']:
        print("\\nRecommendations:")
        for rec in dashboard_data['recommendations']:
            print(f"  • {rec}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
