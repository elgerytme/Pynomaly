#!/usr/bin/env python3
"""Run comprehensive complexity monitoring for CI/CD automation.

This script performs automated complexity analysis and trend tracking
for continuous integration and deployment pipelines.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Suppress warnings for cleaner CI output
warnings.filterwarnings('ignore')


def run_complexity_analysis(
    source_path: Path,
    output_format: str = "json",
    output_file: Optional[Path] = None,
    trend_analysis: bool = False,
    baseline_file: Optional[Path] = None
) -> Dict:
    """Run comprehensive complexity analysis.
    
    Args:
        source_path: Path to source code directory
        output_format: Output format (json, yaml, text)
        output_file: Output file path
        trend_analysis: Whether to perform trend analysis
        baseline_file: Baseline file for comparison
        
    Returns:
        Analysis results dictionary
    """
    try:
        from pynomaly.infrastructure.monitoring.complexity_monitor import ComplexityMonitor
        
        print(f"🔍 Starting complexity analysis for: {source_path}")
        
        # Initialize complexity monitor
        monitor = ComplexityMonitor(project_root=source_path)
        
        # Run comprehensive analysis
        metrics = monitor.measure_all()
        
        # Load baseline for comparison if provided
        baseline_metrics = None
        if baseline_file and baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    baseline_metrics = baseline_data.get('metrics', {})
                print(f"📊 Loaded baseline from: {baseline_file}")
            except Exception as e:
                print(f"⚠️ Could not load baseline: {e}")
        
        # Calculate trends and changes
        trends = {}
        if baseline_metrics:
            trends = calculate_complexity_trends(metrics, baseline_metrics)
        
        # Prepare comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_path': str(source_path),
            'analysis_type': 'comprehensive',
            'metrics': metrics.to_dict(),
            'baseline_comparison': trends if baseline_metrics else None,
            'quality_assessment': assess_quality_status(metrics, baseline_metrics),
            'recommendations': generate_recommendations(metrics, trends),
            'ci_metadata': {
                'git_ref': get_git_ref(),
                'git_commit': get_git_commit(),
                'run_id': datetime.now().strftime('%Y%m%d_%H%M%S')
            }
        }
        
        # Add detailed analysis for CI
        report['detailed_analysis'] = {
            'complexity_distribution': analyze_complexity_distribution(source_path),
            'hotspots': identify_complexity_hotspots(source_path),
            'technical_debt': estimate_technical_debt(metrics),
            'maintainability_score': calculate_maintainability_score(metrics)
        }
        
        # Save report
        if output_file:
            save_report(report, output_file, output_format)
            print(f"📄 Report saved to: {output_file}")
        
        # Print summary for CI logs
        print_ci_summary(report)
        
        return report
        
    except Exception as e:
        print(f"❌ Complexity analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def calculate_complexity_trends(current_metrics, baseline_metrics) -> Dict:
    """Calculate complexity trends compared to baseline."""
    trends = {}
    
    # Compare key metrics
    comparisons = [
        ('total_files', 'Total Files'),
        ('total_lines', 'Total Lines'),
        ('python_files', 'Python Files'),
        ('cyclomatic_complexity', 'Cyclomatic Complexity'),
        ('cognitive_complexity', 'Cognitive Complexity'),
        ('maintainability_index', 'Maintainability Index'),
        ('total_dependencies', 'Dependencies'),
        ('startup_time', 'Startup Time'),
        ('memory_usage', 'Memory Usage')
    ]
    
    current = current_metrics.to_dict() if hasattr(current_metrics, 'to_dict') else current_metrics
    
    for metric_key, metric_name in comparisons:
        current_value = current.get(metric_key, 0)
        baseline_value = baseline_metrics.get(metric_key, 0)
        
        if baseline_value > 0:
            change_percent = ((current_value - baseline_value) / baseline_value) * 100
            change_abs = current_value - baseline_value
            
            trends[metric_key] = {
                'name': metric_name,
                'current': current_value,
                'baseline': baseline_value,
                'change_percent': round(change_percent, 2),
                'change_absolute': change_abs,
                'trend': 'increasing' if change_percent > 0 else 'decreasing' if change_percent < 0 else 'stable',
                'severity': assess_change_severity(metric_key, change_percent)
            }
    
    return trends


def assess_change_severity(metric_key: str, change_percent: float) -> str:
    """Assess the severity of metric changes."""
    # Define thresholds for different metrics
    thresholds = {
        'cyclomatic_complexity': {'warning': 10, 'critical': 25},
        'cognitive_complexity': {'warning': 15, 'critical': 30},
        'total_lines': {'warning': 20, 'critical': 50},
        'total_dependencies': {'warning': 10, 'critical': 25},
        'startup_time': {'warning': 20, 'critical': 50},
        'memory_usage': {'warning': 15, 'critical': 30}
    }
    
    # Maintainability index decreasing is bad
    if metric_key == 'maintainability_index':
        change_percent = -change_percent
    
    threshold = thresholds.get(metric_key, {'warning': 15, 'critical': 30})
    
    if abs(change_percent) >= threshold['critical']:
        return 'critical'
    elif abs(change_percent) >= threshold['warning']:
        return 'warning'
    else:
        return 'info'


def assess_quality_status(metrics, baseline_metrics) -> Dict:
    """Assess overall quality status."""
    current = metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
    
    # Define quality thresholds
    status = {
        'overall': 'good',
        'issues': [],
        'warnings': [],
        'critical_issues': []
    }
    
    # Check complexity thresholds
    if current.get('cyclomatic_complexity', 0) > 10:
        status['critical_issues'].append(f"High cyclomatic complexity: {current.get('cyclomatic_complexity', 0):.1f}")
        status['overall'] = 'critical'
    
    if current.get('cognitive_complexity', 0) > 15:
        status['warnings'].append(f"High cognitive complexity: {current.get('cognitive_complexity', 0):.1f}")
        if status['overall'] == 'good':
            status['overall'] = 'warning'
    
    # Check maintainability
    if current.get('maintainability_index', 100) < 60:
        status['critical_issues'].append(f"Low maintainability index: {current.get('maintainability_index', 0):.1f}")
        status['overall'] = 'critical'
    
    # Check file count growth
    if current.get('total_files', 0) > 1000:
        status['warnings'].append(f"Large file count: {current.get('total_files', 0)} files")
        if status['overall'] == 'good':
            status['overall'] = 'warning'
    
    # Check dependency count
    if current.get('total_dependencies', 0) > 100:
        status['warnings'].append(f"High dependency count: {current.get('total_dependencies', 0)}")
        if status['overall'] == 'good':
            status['overall'] = 'warning'
    
    return status


def generate_recommendations(metrics, trends) -> List[str]:
    """Generate actionable recommendations."""
    recommendations = []
    current = metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
    
    # Complexity recommendations
    if current.get('cyclomatic_complexity', 0) > 8:
        recommendations.append("Consider refactoring complex functions to reduce cyclomatic complexity")
    
    if current.get('cognitive_complexity', 0) > 12:
        recommendations.append("Simplify code logic to reduce cognitive load")
    
    # Size recommendations
    if current.get('total_files', 0) > 500:
        recommendations.append("Consider modularizing the codebase to manage file count")
    
    # Dependency recommendations
    if current.get('total_dependencies', 0) > 50:
        recommendations.append("Review dependencies and remove unused packages")
    
    # Performance recommendations
    if current.get('startup_time', 0) > 2.0:
        recommendations.append("Optimize import structure to improve startup time")
    
    # Trend-based recommendations
    if trends:
        for metric_key, trend in trends.items():
            if trend['severity'] == 'critical' and trend['trend'] == 'increasing':
                recommendations.append(f"Critical increase in {trend['name']} detected - immediate attention required")
            elif trend['severity'] == 'warning' and trend['trend'] == 'increasing':
                recommendations.append(f"Monitor {trend['name']} trend - consider preventive action")
    
    return recommendations


def analyze_complexity_distribution(source_path: Path) -> Dict:
    """Analyze complexity distribution across files."""
    # Simplified complexity distribution analysis
    distribution = {
        'low_complexity': 0,    # < 5
        'medium_complexity': 0, # 5-10
        'high_complexity': 0,   # 10-15
        'very_high_complexity': 0, # > 15
        'total_analyzed': 0
    }
    
    try:
        python_files = list(source_path.rglob('*.py'))
        distribution['total_analyzed'] = len(python_files)
        
        # For simplicity, use file size as a proxy for complexity
        for file_path in python_files[:50]:  # Limit for performance
            try:
                lines = len(file_path.read_text().splitlines())
                complexity = lines / 50  # Rough complexity estimate
                
                if complexity < 5:
                    distribution['low_complexity'] += 1
                elif complexity < 10:
                    distribution['medium_complexity'] += 1
                elif complexity < 15:
                    distribution['high_complexity'] += 1
                else:
                    distribution['very_high_complexity'] += 1
            except:
                continue
                
    except Exception as e:
        print(f"⚠️ Could not analyze complexity distribution: {e}")
    
    return distribution


def identify_complexity_hotspots(source_path: Path) -> List[Dict]:
    """Identify complexity hotspots."""
    hotspots = []
    
    try:
        python_files = list(source_path.rglob('*.py'))
        
        # Analyze top files by size (proxy for complexity)
        file_sizes = []
        for file_path in python_files[:100]:  # Limit for performance
            try:
                size = file_path.stat().st_size
                lines = len(file_path.read_text().splitlines())
                file_sizes.append({
                    'path': str(file_path.relative_to(source_path)),
                    'size': size,
                    'lines': lines,
                    'complexity_estimate': lines / 50
                })
            except:
                continue
        
        # Sort by complexity estimate
        file_sizes.sort(key=lambda x: x['complexity_estimate'], reverse=True)
        
        # Take top 5 as hotspots
        for file_info in file_sizes[:5]:
            if file_info['complexity_estimate'] > 5:
                hotspots.append({
                    'file': file_info['path'],
                    'lines': file_info['lines'],
                    'estimated_complexity': round(file_info['complexity_estimate'], 1),
                    'recommendation': 'Consider refactoring to reduce complexity'
                })
    
    except Exception as e:
        print(f"⚠️ Could not identify hotspots: {e}")
    
    return hotspots


def estimate_technical_debt(metrics) -> Dict:
    """Estimate technical debt based on metrics."""
    current = metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
    
    debt_score = 0
    debt_factors = []
    
    # Factor in high complexity
    complexity = current.get('cyclomatic_complexity', 0)
    if complexity > 10:
        debt_score += (complexity - 10) * 10
        debt_factors.append(f"High cyclomatic complexity (+{(complexity - 10) * 10:.0f})")
    
    # Factor in low maintainability
    maintainability = current.get('maintainability_index', 100)
    if maintainability < 70:
        debt_score += (70 - maintainability) * 2
        debt_factors.append(f"Low maintainability index (+{(70 - maintainability) * 2:.0f})")
    
    # Factor in large codebase
    total_lines = current.get('total_lines', 0)
    if total_lines > 50000:
        debt_score += (total_lines - 50000) / 1000
        debt_factors.append(f"Large codebase (+{(total_lines - 50000) / 1000:.0f})")
    
    # Assess debt level
    if debt_score < 50:
        debt_level = 'low'
    elif debt_score < 150:
        debt_level = 'medium'
    elif debt_score < 300:
        debt_level = 'high'
    else:
        debt_level = 'critical'
    
    return {
        'score': round(debt_score, 1),
        'level': debt_level,
        'factors': debt_factors,
        'estimated_days': round(debt_score / 10, 1)  # Rough estimate
    }


def calculate_maintainability_score(metrics) -> Dict:
    """Calculate overall maintainability score."""
    current = metrics.to_dict() if hasattr(metrics, 'to_dict') else current
    
    # Weight factors for maintainability
    factors = {
        'complexity': 0.3,
        'size': 0.2,
        'documentation': 0.2,
        'dependencies': 0.15,
        'architecture': 0.15
    }
    
    scores = {}
    
    # Complexity score (inverted - lower complexity is better)
    complexity = current.get('cyclomatic_complexity', 5)
    scores['complexity'] = max(0, 100 - (complexity * 5))
    
    # Size score (inverted - smaller is better, up to a point)
    total_lines = current.get('total_lines', 1000)
    optimal_size = 25000  # Optimal project size
    if total_lines < optimal_size:
        scores['size'] = 100
    else:
        scores['size'] = max(0, 100 - ((total_lines - optimal_size) / 1000))
    
    # Documentation score
    doc_coverage = current.get('docstring_coverage', 70)
    scores['documentation'] = min(100, doc_coverage)
    
    # Dependencies score (fewer is better)
    deps = current.get('total_dependencies', 10)
    scores['dependencies'] = max(0, 100 - (deps * 2))
    
    # Architecture score (based on maintainability index)
    arch_score = current.get('maintainability_index', 70)
    scores['architecture'] = min(100, arch_score)
    
    # Calculate weighted average
    weighted_score = sum(scores[factor] * weight for factor, weight in factors.items())
    
    # Determine grade
    if weighted_score >= 90:
        grade = 'A'
    elif weighted_score >= 80:
        grade = 'B'
    elif weighted_score >= 70:
        grade = 'C'
    elif weighted_score >= 60:
        grade = 'D'
    else:
        grade = 'F'
    
    return {
        'overall_score': round(weighted_score, 1),
        'grade': grade,
        'factor_scores': {k: round(v, 1) for k, v in scores.items()},
        'factor_weights': factors
    }


def save_report(report: Dict, output_file: Path, output_format: str):
    """Save report to file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'json':
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    elif output_format == 'yaml':
        import yaml
        with open(output_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
    else:
        # Text format
        with open(output_file, 'w') as f:
            f.write(format_report_text(report))


def format_report_text(report: Dict) -> str:
    """Format report as readable text."""
    lines = []
    lines.append("# Complexity Monitoring Report")
    lines.append(f"Generated: {report['timestamp']}")
    lines.append(f"Project: {report['project_path']}")
    lines.append("")
    
    # Metrics summary
    metrics = report['metrics']
    lines.append("## Metrics Summary")
    lines.append(f"Total Files: {metrics.get('total_files', 0)}")
    lines.append(f"Total Lines: {metrics.get('total_lines', 0)}")
    lines.append(f"Python Files: {metrics.get('python_files', 0)}")
    lines.append(f"Cyclomatic Complexity: {metrics.get('cyclomatic_complexity', 0):.1f}")
    lines.append(f"Dependencies: {metrics.get('total_dependencies', 0)}")
    lines.append("")
    
    # Quality assessment
    quality = report['quality_assessment']
    lines.append(f"## Quality Status: {quality['overall'].upper()}")
    if quality['critical_issues']:
        lines.append("### Critical Issues:")
        for issue in quality['critical_issues']:
            lines.append(f"- {issue}")
    if quality['warnings']:
        lines.append("### Warnings:")
        for warning in quality['warnings']:
            lines.append(f"- {warning}")
    lines.append("")
    
    # Recommendations
    if report['recommendations']:
        lines.append("## Recommendations")
        for rec in report['recommendations']:
            lines.append(f"- {rec}")
    
    return "\n".join(lines)


def print_ci_summary(report: Dict):
    """Print summary for CI logs."""
    print("\n" + "="*60)
    print("📊 COMPLEXITY ANALYSIS SUMMARY")
    print("="*60)
    
    metrics = report['metrics']
    quality = report['quality_assessment']
    maintainability = report['detailed_analysis']['maintainability_score']
    
    print(f"📁 Files: {metrics.get('total_files', 0)} ({metrics.get('python_files', 0)} Python)")
    print(f"📏 Lines: {metrics.get('total_lines', 0):,}")
    print(f"🔄 Complexity: {metrics.get('cyclomatic_complexity', 0):.1f}")
    print(f"📦 Dependencies: {metrics.get('total_dependencies', 0)}")
    print(f"🎯 Maintainability: {maintainability['grade']} ({maintainability['overall_score']:.1f}/100)")
    print(f"🚦 Quality Status: {quality['overall'].upper()}")
    
    if quality['critical_issues']:
        print(f"🚨 Critical Issues: {len(quality['critical_issues'])}")
        for issue in quality['critical_issues']:
            print(f"   - {issue}")
    
    if quality['warnings']:
        print(f"⚠️ Warnings: {len(quality['warnings'])}")
        for warning in quality['warnings'][:3]:  # Limit to first 3
            print(f"   - {warning}")
    
    # Baseline comparison
    if report['baseline_comparison']:
        print("\n📈 CHANGES FROM BASELINE:")
        trends = report['baseline_comparison']
        for metric_key, trend in trends.items():
            if trend['severity'] in ['warning', 'critical']:
                icon = '🚨' if trend['severity'] == 'critical' else '⚠️'
                print(f"   {icon} {trend['name']}: {trend['change_percent']:+.1f}%")
    
    print("="*60)


def get_git_ref() -> str:
    """Get current git reference."""
    import subprocess
    try:
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except:
        return 'unknown'


def get_git_commit() -> str:
    """Get current git commit hash."""
    import subprocess
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
    except:
        return 'unknown'


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run complexity monitoring for CI/CD')
    parser.add_argument('--source-path', '-s', type=Path, default=Path('src'),
                       help='Source code path to analyze')
    parser.add_argument('--output-format', '-f', choices=['json', 'yaml', 'text'],
                       default='json', help='Output format')
    parser.add_argument('--output-file', '-o', type=Path,
                       help='Output file path')
    parser.add_argument('--trend-analysis', '-t', action='store_true',
                       help='Perform trend analysis')
    parser.add_argument('--baseline-file', '-b', type=Path,
                       help='Baseline file for comparison')
    
    args = parser.parse_args()
    
    # Run analysis
    try:
        report = run_complexity_analysis(
            source_path=args.source_path,
            output_format=args.output_format,
            output_file=args.output_file,
            trend_analysis=args.trend_analysis,
            baseline_file=args.baseline_file
        )
        
        # Exit with appropriate code
        quality_status = report['quality_assessment']['overall']
        if quality_status == 'critical':
            print("❌ Critical complexity issues detected")
            sys.exit(1)
        elif quality_status == 'warning':
            print("⚠️ Complexity warnings detected")
            sys.exit(0)  # Don't fail build on warnings
        else:
            print("✅ Complexity analysis passed")
            sys.exit(0)
            
    except Exception as e:
        print(f"💥 Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()