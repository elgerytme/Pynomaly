#!/usr/bin/env python3
"""Comprehensive performance regression analysis across all test suites."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ComprehensivePerformanceAnalyzer:
    """Comprehensive performance regression analysis."""
    
    def __init__(self, input_dir: Path, output_file: Path):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "input_directory": str(input_dir),
            "suites_analyzed": [],
            "overall_summary": {},
            "detailed_analysis": {},
            "cross_suite_patterns": {},
            "recommendations": []
        }
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_all_benchmark_data(self) -> Dict[str, Dict]:
        """Load all benchmark data from input directory."""
        benchmark_data = {}
        
        # Find all benchmark files
        benchmark_files = list(self.input_dir.glob("**/benchmark-*.json"))
        
        for benchmark_file in benchmark_files:
            try:
                with open(benchmark_file, 'r') as f:
                    data = json.load(f)
                
                # Extract suite name
                suite_name = benchmark_file.stem.replace("benchmark-", "")
                benchmark_data[suite_name] = data
                
                print(f"📊 Loaded {suite_name} benchmark data")
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️  Warning: Could not load {benchmark_file}: {e}")
        
        return benchmark_data
    
    def load_all_regression_data(self) -> Dict[str, Dict]:
        """Load all regression analysis data."""
        regression_data = {}
        
        # Find all regression files
        regression_files = list(self.input_dir.glob("**/regression-report-*.json"))
        
        for regression_file in regression_files:
            try:
                with open(regression_file, 'r') as f:
                    data = json.load(f)
                
                # Extract suite name
                suite_name = regression_file.stem.replace("regression-report-", "")
                regression_data[suite_name] = data
                
                print(f"📈 Loaded {suite_name} regression data")
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️  Warning: Could not load {regression_file}: {e}")
        
        return regression_data
    
    def analyze_cross_suite_patterns(self, 
                                   benchmark_data: Dict[str, Dict],
                                   regression_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze patterns across different test suites."""
        patterns = {
            "performance_distribution": {},
            "regression_patterns": {},
            "suite_correlations": {},
            "hotspots": []
        }
        
        # Analyze performance distribution across suites
        suite_metrics = {}
        for suite_name, data in benchmark_data.items():
            benchmarks = data.get('benchmarks', [])
            if benchmarks:
                mean_times = [b.get('stats', {}).get('mean', 0) for b in benchmarks]
                suite_metrics[suite_name] = {
                    "avg_time": statistics.mean(mean_times),
                    "median_time": statistics.median(mean_times),
                    "max_time": max(mean_times),
                    "min_time": min(mean_times),
                    "std_dev": statistics.stdev(mean_times) if len(mean_times) > 1 else 0,
                    "test_count": len(benchmarks)
                }
        
        patterns["performance_distribution"] = suite_metrics
        
        # Analyze regression patterns
        regression_summary = {}
        all_regressions = []
        all_improvements = []
        
        for suite_name, data in regression_data.items():
            regressions = data.get('regressions', [])
            improvements = data.get('improvements', [])
            
            regression_summary[suite_name] = {
                "regression_count": len(regressions),
                "improvement_count": len(improvements),
                "regression_severity": self._calculate_regression_severity(regressions)
            }
            
            all_regressions.extend(regressions)
            all_improvements.extend(improvements)
        
        patterns["regression_patterns"] = regression_summary
        
        # Identify performance hotspots
        hotspots = []
        for suite_name, metrics in suite_metrics.items():
            if metrics["avg_time"] > 1.0:  # More than 1 second average
                hotspots.append({
                    "suite": suite_name,
                    "type": "slow_suite",
                    "avg_time": metrics["avg_time"],
                    "severity": "high" if metrics["avg_time"] > 5.0 else "medium"
                })
        
        # Add regression hotspots
        for suite_name, reg_data in regression_summary.items():
            if reg_data["regression_count"] > 3:  # More than 3 regressions
                hotspots.append({
                    "suite": suite_name,
                    "type": "regression_hotspot",
                    "regression_count": reg_data["regression_count"],
                    "severity": reg_data["regression_severity"]
                })
        
        patterns["hotspots"] = hotspots
        
        return patterns
    
    def _calculate_regression_severity(self, regressions: List[Dict]) -> str:
        """Calculate overall regression severity for a suite."""
        if not regressions:
            return "none"
        
        severity_scores = []
        for regression in regressions:
            change_percent = abs(regression.get('change_percent', 0))
            if change_percent > 50:
                severity_scores.append(3)  # Critical
            elif change_percent > 20:
                severity_scores.append(2)  # High
            elif change_percent > 10:
                severity_scores.append(1)  # Medium
            else:
                severity_scores.append(0)  # Low
        
        avg_severity = statistics.mean(severity_scores)
        
        if avg_severity >= 2.5:
            return "critical"
        elif avg_severity >= 1.5:
            return "high"
        elif avg_severity >= 0.5:
            return "medium"
        else:
            return "low"
    
    def generate_performance_plots(self, 
                                 benchmark_data: Dict[str, Dict],
                                 output_dir: Path) -> List[str]:
        """Generate performance visualization plots."""
        output_dir.mkdir(exist_ok=True)
        generated_plots = []
        
        # 1. Performance distribution across suites
        suite_names = []
        avg_times = []
        test_counts = []
        
        for suite_name, data in benchmark_data.items():
            benchmarks = data.get('benchmarks', [])
            if benchmarks:
                mean_times = [b.get('stats', {}).get('mean', 0) for b in benchmarks]
                suite_names.append(suite_name)
                avg_times.append(statistics.mean(mean_times))
                test_counts.append(len(benchmarks))
        
        if suite_names:
            # Performance by suite bar chart
            plt.figure(figsize=(12, 6))
            bars = plt.bar(suite_names, avg_times, color='skyblue', alpha=0.7)
            plt.title('Average Performance by Test Suite')
            plt.xlabel('Test Suite')
            plt.ylabel('Average Execution Time (seconds)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, avg_time in zip(bars, avg_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{avg_time:.3f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = output_dir / 'performance_by_suite.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
        
        # 2. Test count vs performance correlation
        if len(suite_names) > 1:
            plt.figure(figsize=(10, 6))
            plt.scatter(test_counts, avg_times, alpha=0.7, s=100)
            
            # Add labels for each point
            for i, suite in enumerate(suite_names):
                plt.annotate(suite, (test_counts[i], avg_times[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            plt.title('Test Count vs Average Performance')
            plt.xlabel('Number of Tests')
            plt.ylabel('Average Execution Time (seconds)')
            plt.grid(True, alpha=0.3)
            
            plot_path = output_dir / 'test_count_vs_performance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append(str(plot_path))
        
        # 3. Performance distribution heatmap
        if len(suite_names) > 1:
            # Create performance matrix
            performance_matrix = []
            for suite_name, data in benchmark_data.items():
                benchmarks = data.get('benchmarks', [])
                if benchmarks:
                    mean_times = [b.get('stats', {}).get('mean', 0) for b in benchmarks]
                    # Normalize to percentiles for heatmap
                    performance_matrix.append([
                        statistics.quantile(mean_times, q) for q in [0.25, 0.5, 0.75, 0.9, 0.95]
                    ])
            
            if performance_matrix:
                df = pd.DataFrame(performance_matrix, 
                                index=suite_names,
                                columns=['P25', 'P50', 'P75', 'P90', 'P95'])
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd')
                plt.title('Performance Distribution Heatmap (seconds)')
                plt.xlabel('Percentiles')
                plt.ylabel('Test Suite')
                
                plot_path = output_dir / 'performance_heatmap.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_plots.append(str(plot_path))
        
        return generated_plots
    
    def generate_comprehensive_recommendations(self, 
                                             patterns: Dict[str, Any],
                                             benchmark_data: Dict[str, Dict],
                                             regression_data: Dict[str, Dict]) -> List[str]:
        """Generate comprehensive recommendations based on analysis."""
        recommendations = []
        
        # Performance distribution recommendations
        perf_dist = patterns.get("performance_distribution", {})
        if perf_dist:
            # Find slowest suite
            slowest_suite = max(perf_dist.items(), key=lambda x: x[1]["avg_time"])
            if slowest_suite[1]["avg_time"] > 2.0:
                recommendations.append(
                    f"🐌 **Critical Performance Issue**: {slowest_suite[0]} suite averages "
                    f"{slowest_suite[1]['avg_time']:.2f}s per test. This requires immediate optimization."
                )
            
            # Find most variable suite
            most_variable = max(perf_dist.items(), key=lambda x: x[1]["std_dev"])
            if most_variable[1]["std_dev"] > 0.5:
                recommendations.append(
                    f"📊 **Performance Inconsistency**: {most_variable[0]} suite shows high "
                    f"variability (σ={most_variable[1]['std_dev']:.3f}). Review test determinism."
                )
        
        # Regression pattern recommendations
        reg_patterns = patterns.get("regression_patterns", {})
        total_regressions = sum(data["regression_count"] for data in reg_patterns.values())
        total_improvements = sum(data["improvement_count"] for data in reg_patterns.values())
        
        if total_regressions > total_improvements:
            recommendations.append(
                f"📉 **Regression Trend**: {total_regressions} regressions vs "
                f"{total_improvements} improvements. Focus on optimization efforts."
            )
        elif total_improvements > total_regressions * 2:
            recommendations.append(
                f"📈 **Positive Trend**: {total_improvements} improvements vs "
                f"{total_regressions} regressions. Excellent optimization work!"
            )
        
        # Hotspot recommendations
        hotspots = patterns.get("hotspots", [])
        critical_hotspots = [h for h in hotspots if h.get("severity") == "critical"]
        
        if critical_hotspots:
            recommendations.append(
                f"🚨 **Critical Hotspots**: {len(critical_hotspots)} critical performance "
                f"issues identified: {', '.join(h['suite'] for h in critical_hotspots)}"
            )
        
        # Suite-specific recommendations
        for suite_name, metrics in perf_dist.items():
            if metrics["test_count"] > 50 and metrics["avg_time"] > 0.5:
                recommendations.append(
                    f"🔍 **{suite_name} Suite**: {metrics['test_count']} tests averaging "
                    f"{metrics['avg_time']:.3f}s. Consider test parallelization or optimization."
                )
        
        # Cross-suite recommendations
        if len(perf_dist) > 1:
            fastest_suite = min(perf_dist.items(), key=lambda x: x[1]["avg_time"])
            slowest_suite = max(perf_dist.items(), key=lambda x: x[1]["avg_time"])
            
            speed_ratio = slowest_suite[1]["avg_time"] / fastest_suite[1]["avg_time"]
            if speed_ratio > 10:
                recommendations.append(
                    f"⚖️ **Performance Imbalance**: {slowest_suite[0]} is {speed_ratio:.1f}x "
                    f"slower than {fastest_suite[0]}. Consider architecture review."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "✅ **Excellent Performance**: All metrics are within acceptable ranges. "
                "Continue current practices and monitor for trends."
            )
        
        return recommendations
    
    def run_comprehensive_analysis(self, generate_plots: bool = True) -> Dict[str, Any]:
        """Run comprehensive performance analysis."""
        print("🔍 Running comprehensive performance analysis...")
        
        # Load all data
        benchmark_data = self.load_all_benchmark_data()
        regression_data = self.load_all_regression_data()
        
        if not benchmark_data and not regression_data:
            print("❌ No performance data found to analyze")
            return {}
        
        # Analyze cross-suite patterns
        patterns = self.analyze_cross_suite_patterns(benchmark_data, regression_data)
        
        # Generate plots if requested
        generated_plots = []
        if generate_plots:
            plots_dir = self.output_file.parent / "performance-plots"
            generated_plots = self.generate_performance_plots(benchmark_data, plots_dir)
            print(f"📊 Generated {len(generated_plots)} performance plots")
        
        # Generate recommendations
        recommendations = self.generate_comprehensive_recommendations(
            patterns, benchmark_data, regression_data
        )
        
        # Compile final results
        self.analysis_results.update({
            "suites_analyzed": list(benchmark_data.keys()),
            "overall_summary": {
                "total_suites": len(benchmark_data),
                "total_tests": sum(len(data.get('benchmarks', [])) for data in benchmark_data.values()),
                "total_regressions": sum(len(data.get('regressions', [])) for data in regression_data.values()),
                "total_improvements": sum(len(data.get('improvements', [])) for data in regression_data.values()),
                "critical_issues": len([h for h in patterns.get("hotspots", []) if h.get("severity") == "critical"])
            },
            "detailed_analysis": {
                "benchmark_data": benchmark_data,
                "regression_data": regression_data
            },
            "cross_suite_patterns": patterns,
            "recommendations": recommendations,
            "generated_plots": generated_plots
        })
        
        return self.analysis_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to output file."""
        # Remove detailed data to reduce file size
        output_data = results.copy()
        output_data["detailed_analysis"] = {
            "note": "Detailed data excluded from output for size optimization",
            "suites_processed": len(results.get("suites_analyzed", []))
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Analysis results saved to {self.output_file}")


def main():
    """Main entry point for comprehensive performance analysis."""
    parser = argparse.ArgumentParser(description="Comprehensive performance analysis")
    parser.add_argument("--input-dir", type=Path, required=True,
                       help="Directory containing benchmark and regression results")
    parser.add_argument("--output", type=Path, default="comprehensive-regression-report.json",
                       help="Output file for analysis results")
    parser.add_argument("--generate-plots", action="store_true",
                       help="Generate performance visualization plots")
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1
    
    print(f"🔍 Starting comprehensive analysis of {args.input_dir}")
    
    # Create analyzer
    analyzer = ComprehensivePerformanceAnalyzer(args.input_dir, args.output)
    
    # Run analysis
    results = analyzer.run_comprehensive_analysis(generate_plots=args.generate_plots)
    
    if not results:
        print("❌ Analysis failed - no results generated")
        return 1
    
    # Save results
    analyzer.save_results(results)
    
    # Print summary
    summary = results.get("overall_summary", {})
    print("\n📊 Analysis Summary:")
    print(f"- Suites analyzed: {summary.get('total_suites', 0)}")
    print(f"- Total tests: {summary.get('total_tests', 0)}")
    print(f"- Regressions: {summary.get('total_regressions', 0)}")
    print(f"- Improvements: {summary.get('total_improvements', 0)}")
    print(f"- Critical issues: {summary.get('critical_issues', 0)}")
    
    recommendations = results.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
            print(f"{i}. {rec}")
    
    print(f"\n✅ Comprehensive analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())