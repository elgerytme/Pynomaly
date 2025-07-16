#!/usr/bin/env python3
"""Memory analysis script for performance testing."""

import gc
import os
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil


class MemoryAnalyzer:
    """Advanced memory analysis for performance testing."""
    
    def __init__(self, output_dir: str = "memory_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Memory thresholds (in MB)
        self.thresholds = {
            "warning": 100,
            "critical": 200,
            "max_allowed": 500
        }
        
        # Track memory snapshots
        self.snapshots: List[Tuple[str, tracemalloc.Snapshot]] = []
        self.process = psutil.Process()
        
        # Initialize tracemalloc
        tracemalloc.start(25)  # Track 25 frames
        
        self.report_data = {
            "timestamp": datetime.now().isoformat(),
            "process_info": {
                "pid": os.getpid(),
                "python_version": sys.version,
                "platform": sys.platform
            },
            "analysis_results": {},
            "memory_snapshots": [],
            "recommendations": []
        }
    
    def take_snapshot(self, label: str) -> None:
        """Take a memory snapshot with label."""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))
        
        # Also track process memory
        memory_info = self.process.memory_info()
        self.report_data["memory_snapshots"].append({
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent()
        })
    
    def analyze_memory_growth(self) -> Dict:
        """Analyze memory growth between snapshots."""
        if len(self.snapshots) < 2:
            return {"error": "Need at least 2 snapshots for growth analysis"}
        
        growth_analysis = {}
        
        for i in range(1, len(self.snapshots)):
            prev_label, prev_snapshot = self.snapshots[i-1]
            curr_label, curr_snapshot = self.snapshots[i]
            
            # Compare snapshots
            top_stats = curr_snapshot.compare_to(prev_snapshot, 'lineno')
            
            # Get top memory consumers
            top_consumers = []
            for stat in top_stats[:10]:  # Top 10
                top_consumers.append({
                    "file": stat.traceback.format()[-1],
                    "size_mb": stat.size / 1024 / 1024,
                    "size_diff_mb": stat.size_diff / 1024 / 1024,
                    "count": stat.count,
                    "count_diff": stat.count_diff
                })
            
            growth_analysis[f"{prev_label}_to_{curr_label}"] = {
                "top_consumers": top_consumers,
                "total_size_mb": sum(stat.size for stat in top_stats) / 1024 / 1024,
                "total_diff_mb": sum(stat.size_diff for stat in top_stats) / 1024 / 1024
            }
        
        return growth_analysis
    
    def analyze_memory_leaks(self) -> List[Dict]:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 3:
            return []
        
        # Look for consistently growing memory areas
        potential_leaks = []
        
        # Analyze growth patterns
        for i in range(2, len(self.snapshots)):
            prev_snapshot = self.snapshots[i-2][1]
            curr_snapshot = self.snapshots[i][1]
            
            # Compare memory usage
            top_stats = curr_snapshot.compare_to(prev_snapshot, 'lineno')
            
            # Look for consistently growing allocations
            for stat in top_stats[:20]:
                if stat.size_diff > 1024 * 1024:  # > 1MB growth
                    potential_leaks.append({
                        "file": stat.traceback.format()[-1],
                        "size_growth_mb": stat.size_diff / 1024 / 1024,
                        "count_growth": stat.count_diff,
                        "current_size_mb": stat.size / 1024 / 1024,
                        "traceback": stat.traceback.format()
                    })
        
        return potential_leaks
    
    def analyze_garbage_collection(self) -> Dict:
        """Analyze garbage collection statistics."""
        gc_stats = {}
        
        # Get GC counts
        gc_counts = gc.get_count()
        gc_stats["generation_counts"] = {
            "gen0": gc_counts[0],
            "gen1": gc_counts[1],
            "gen2": gc_counts[2]
        }
        
        # Get GC stats
        gc_stats["gc_stats"] = gc.get_stats()
        
        # Get GC threshold
        gc_stats["gc_threshold"] = gc.get_threshold()
        
        # Check for unreachable objects
        unreachable = gc.collect()
        gc_stats["unreachable_objects"] = unreachable
        
        return gc_stats
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on analysis."""
        recommendations = []
        
        # Check current memory usage
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        if current_memory > self.thresholds["critical"]:
            recommendations.append(
                f"CRITICAL: Memory usage ({current_memory:.1f}MB) exceeds critical threshold "
                f"({self.thresholds['critical']}MB). Consider optimizing memory usage."
            )
        elif current_memory > self.thresholds["warning"]:
            recommendations.append(
                f"WARNING: Memory usage ({current_memory:.1f}MB) exceeds warning threshold "
                f"({self.thresholds['warning']}MB). Monitor for memory leaks."
            )
        
        # Check for rapid growth
        snapshots = self.report_data["memory_snapshots"]
        if len(snapshots) >= 2:
            growth_rate = snapshots[-1]["rss_mb"] - snapshots[0]["rss_mb"]
            if growth_rate > 50:  # More than 50MB growth
                recommendations.append(
                    f"High memory growth detected: {growth_rate:.1f}MB increase. "
                    "Review memory allocation patterns."
                )
        
        # Check garbage collection
        gc_stats = self.analyze_garbage_collection()
        if gc_stats["unreachable_objects"] > 1000:
            recommendations.append(
                f"High number of unreachable objects ({gc_stats['unreachable_objects']}). "
                "Consider explicit cleanup or review object lifecycle."
            )
        
        return recommendations
    
    def generate_html_report(self) -> str:
        """Generate an HTML report of memory analysis."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Memory Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .warning {{ color: #ff6600; }}
                .critical {{ color: #ff0000; }}
                .success {{ color: #00aa00; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .code {{ background-color: #f4f4f4; padding: 10px; border-radius: 3px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Memory Analysis Report</h1>
                <p>Generated: {self.report_data['timestamp']}</p>
                <p>Process ID: {self.report_data['process_info']['pid']}</p>
            </div>
        """
        
        # Memory snapshots section
        html_content += """
            <div class="section">
                <h2>Memory Snapshots</h2>
                <table>
                    <tr>
                        <th>Label</th>
                        <th>RSS (MB)</th>
                        <th>VMS (MB)</th>
                        <th>Memory %</th>
                    </tr>
        """
        
        for snapshot in self.report_data["memory_snapshots"]:
            html_content += f"""
                    <tr>
                        <td>{snapshot['label']}</td>
                        <td>{snapshot['rss_mb']:.1f}</td>
                        <td>{snapshot['vms_mb']:.1f}</td>
                        <td>{snapshot['percent']:.1f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # Recommendations section
        html_content += """
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for recommendation in self.generate_recommendations():
            css_class = "critical" if "CRITICAL" in recommendation else "warning" if "WARNING" in recommendation else "success"
            html_content += f'<li class="{css_class}">{recommendation}</li>'
        
        html_content += """
                </ul>
            </div>
        """
        
        # Memory growth analysis
        growth_analysis = self.analyze_memory_growth()
        if "error" not in growth_analysis:
            html_content += """
                <div class="section">
                    <h2>Memory Growth Analysis</h2>
            """
            
            for period, data in growth_analysis.items():
                html_content += f"""
                    <h3>{period}</h3>
                    <p>Total Size: {data['total_size_mb']:.1f}MB</p>
                    <p>Total Diff: {data['total_diff_mb']:.1f}MB</p>
                    <h4>Top Memory Consumers:</h4>
                    <table>
                        <tr>
                            <th>File</th>
                            <th>Size (MB)</th>
                            <th>Size Diff (MB)</th>
                            <th>Count</th>
                            <th>Count Diff</th>
                        </tr>
                """
                
                for consumer in data['top_consumers']:
                    html_content += f"""
                        <tr>
                            <td>{consumer['file']}</td>
                            <td>{consumer['size_mb']:.2f}</td>
                            <td>{consumer['size_diff_mb']:.2f}</td>
                            <td>{consumer['count']}</td>
                            <td>{consumer['count_diff']}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                """
            
            html_content += """
                </div>
            """
        
        # Potential leaks section
        leaks = self.analyze_memory_leaks()
        if leaks:
            html_content += """
                <div class="section">
                    <h2>Potential Memory Leaks</h2>
                    <table>
                        <tr>
                            <th>File</th>
                            <th>Growth (MB)</th>
                            <th>Count Growth</th>
                            <th>Current Size (MB)</th>
                        </tr>
            """
            
            for leak in leaks:
                html_content += f"""
                    <tr>
                        <td>{leak['file']}</td>
                        <td>{leak['size_growth_mb']:.2f}</td>
                        <td>{leak['count_growth']}</td>
                        <td>{leak['current_size_mb']:.2f}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def run_analysis(self) -> None:
        """Run comprehensive memory analysis."""
        print("🔍 Starting memory analysis...")
        
        # Initial snapshot
        self.take_snapshot("initial")
        
        # Simulate some work (in real scenario, this would be actual test execution)
        print("📊 Running memory-intensive operations...")
        
        # Allocate some memory to demonstrate analysis
        test_data = []
        for i in range(10):
            test_data.append([j for j in range(1000)])
            if i % 3 == 0:
                self.take_snapshot(f"iteration_{i}")
            time.sleep(0.1)
        
        # Final snapshot
        self.take_snapshot("final")
        
        # Run analysis
        print("📈 Analyzing memory usage...")
        growth_analysis = self.analyze_memory_growth()
        leaks = self.analyze_memory_leaks()
        gc_stats = self.analyze_garbage_collection()
        recommendations = self.generate_recommendations()
        
        # Store results
        self.report_data["analysis_results"] = {
            "memory_growth": growth_analysis,
            "potential_leaks": leaks,
            "gc_stats": gc_stats
        }
        self.report_data["recommendations"] = recommendations
        
        # Generate HTML report
        html_report = self.generate_html_report()
        html_path = self.output_dir / f"memory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        html_path.write_text(html_report)
        
        print(f"✅ Memory analysis complete! Report saved to: {html_path}")
        
        # Print summary
        print("\n📋 Memory Analysis Summary:")
        print(f"- Total snapshots: {len(self.snapshots)}")
        print(f"- Current memory usage: {self.process.memory_info().rss / 1024 / 1024:.1f}MB")
        print(f"- Potential leaks found: {len(leaks)}")
        print(f"- Recommendations: {len(recommendations)}")
        
        if recommendations:
            print("\n⚠️  Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")


def main():
    """Main entry point for memory analysis."""
    analyzer = MemoryAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()