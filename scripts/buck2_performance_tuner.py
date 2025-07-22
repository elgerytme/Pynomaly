#!/usr/bin/env python3
"""
Buck2 Performance Tuning and Optimization Script
Analyzes and optimizes Buck2 build performance for the Pynomaly monorepo
"""

import argparse
import json
import os
import platform
import psutil
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class Buck2PerformanceTuner:
    """Buck2 performance analysis and optimization tool"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.system_info = self._get_system_info()
        self.benchmark_results = {}
        
    def _get_system_info(self) -> Dict:
        """Get system information for optimization recommendations"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3)),
            'disk_free_gb': round(shutil.disk_usage('.').free / (1024**3)),
            'platform': platform.system(),
            'python_version': platform.python_version(),
        }
    
    def log(self, message: str) -> None:
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def run_benchmark(self, target: str, iterations: int = 3) -> Dict:
        """Run build benchmark for a target"""
        self.log(f"Benchmarking target: {target}")
        
        results = {
            'target': target,
            'iterations': [],
            'avg_time': 0,
            'min_time': float('inf'),
            'max_time': 0,
        }
        
        for i in range(iterations):
            self.log(f"Iteration {i+1}/{iterations}")
            
            # Clean build
            subprocess.run(['buck2', 'clean'], capture_output=True)
            
            # Time the build
            start_time = time.time()
            result = subprocess.run(
                ['buck2', 'build', target],
                capture_output=True, text=True
            )
            end_time = time.time()
            
            build_time = end_time - start_time
            results['iterations'].append({
                'time': build_time,
                'success': result.returncode == 0,
                'output_size': len(result.stdout) + len(result.stderr)
            })
            
            results['min_time'] = min(results['min_time'], build_time)
            results['max_time'] = max(results['max_time'], build_time)
        
        successful_times = [
            r['time'] for r in results['iterations'] if r['success']
        ]
        
        if successful_times:
            results['avg_time'] = sum(successful_times) / len(successful_times)
        
        return results
    
    def analyze_cache_performance(self) -> Dict:
        """Analyze cache hit rates and performance"""
        self.log("Analyzing cache performance...")
        
        cache_stats = {
            'local_cache_size': 0,
            'cache_hit_rate': 0,
            'cache_efficiency': 'unknown'
        }
        
        try:
            # Check local cache size
            buck_cache = Path('.buck-cache')
            if buck_cache.exists():
                cache_stats['local_cache_size'] = sum(
                    f.stat().st_size for f in buck_cache.rglob('*') if f.is_file()
                ) / (1024**2)  # MB
            
            # Run build with cache stats
            result = subprocess.run(
                ['buck2', 'build', '//:pynomaly', '--show-output'],
                capture_output=True, text=True
            )
            
            # Parse cache statistics from output
            if 'cache hit rate' in result.stderr.lower():
                import re
                hit_rate_match = re.search(r'cache hit rate: ([\d.]+)%', result.stderr)
                if hit_rate_match:
                    cache_stats['cache_hit_rate'] = float(hit_rate_match.group(1))
            
        except Exception as e:
            self.log(f"Cache analysis error: {e}")
        
        # Determine cache efficiency
        if cache_stats['cache_hit_rate'] > 80:
            cache_stats['cache_efficiency'] = 'excellent'
        elif cache_stats['cache_hit_rate'] > 60:
            cache_stats['cache_efficiency'] = 'good'  
        elif cache_stats['cache_hit_rate'] > 30:
            cache_stats['cache_efficiency'] = 'moderate'
        else:
            cache_stats['cache_efficiency'] = 'poor'
        
        return cache_stats
    
    def generate_optimized_config(self) -> Dict:
        """Generate optimized .buckconfig based on system specs"""
        self.log("Generating optimized configuration...")
        
        config = {
            'build': {
                'threads': self.system_info['cpu_count'],
            },
            'cache': {
                'disk_cache_size_mb': min(
                    self.system_info['disk_free_gb'] * 100,  # 10% of free disk
                    20480  # Max 20GB
                ),
            },
            'download': {
                'max_downloads': min(self.system_info['cpu_count'] * 2, 16),
            },
            'python': {
                'enable_bytecode_cache': True,
                'cache_bytecode_files': True,
                'precompile_modules': True,
            }
        }
        
        # Memory-based optimizations
        if self.system_info['memory_gb'] >= 16:
            config['action_cache'] = {
                'max_entries': 200000,
                'cleanup_threshold': 0.9,
            }
        elif self.system_info['memory_gb'] >= 8:
            config['action_cache'] = {
                'max_entries': 100000,
                'cleanup_threshold': 0.8,
            }
        else:
            config['action_cache'] = {
                'max_entries': 50000,
                'cleanup_threshold': 0.7,
            }
        
        return config
    
    def apply_optimizations(self, config: Dict) -> None:
        """Apply optimizations to .buckconfig"""
        self.log("Applying optimizations to .buckconfig...")
        
        buckconfig_path = Path('.buckconfig')
        if not buckconfig_path.exists():
            self.log("❌ .buckconfig not found")
            return
        
        # Read existing config
        with open(buckconfig_path, 'r') as f:
            lines = f.readlines()
        
        # Generate optimization section
        optimization_section = "\n# Performance optimizations (auto-generated)\n"
        
        for section, settings in config.items():
            optimization_section += f"[{section}]\n"
            for key, value in settings.items():
                optimization_section += f"  {key} = {value}\n"
            optimization_section += "\n"
        
        # Append optimizations
        with open(buckconfig_path, 'a') as f:
            f.write(optimization_section)
        
        self.log("✅ Optimizations applied to .buckconfig")
    
    def benchmark_suite(self) -> Dict:
        """Run comprehensive benchmark suite"""
        self.log("Running comprehensive benchmark suite...")
        
        targets = [
            '//:ai-anomaly-detection',
            '//:data-engineering', 
            '//:enterprise-auth',
            '//:pynomaly'
        ]
        
        suite_results = {
            'system_info': self.system_info,
            'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'target_results': [],
            'cache_analysis': self.analyze_cache_performance(),
        }
        
        for target in targets:
            try:
                result = self.run_benchmark(target, iterations=2)
                suite_results['target_results'].append(result)
            except Exception as e:
                self.log(f"Benchmark failed for {target}: {e}")
                suite_results['target_results'].append({
                    'target': target,
                    'error': str(e)
                })
        
        return suite_results
    
    def generate_report(self, results: Dict) -> str:
        """Generate performance tuning report"""
        report = f"""# Buck2 Performance Tuning Report
Generated: {results['benchmark_timestamp']}

## System Information
- CPU Cores: {results['system_info']['cpu_count']}
- Memory: {results['system_info']['memory_gb']} GB  
- Free Disk: {results['system_info']['disk_free_gb']} GB
- Platform: {results['system_info']['platform']}

## Cache Performance Analysis
- Local Cache Size: {results['cache_analysis']['local_cache_size']:.1f} MB
- Cache Hit Rate: {results['cache_analysis']['cache_hit_rate']:.1f}%
- Cache Efficiency: {results['cache_analysis']['cache_efficiency']}

## Build Performance Benchmarks
"""
        
        for target_result in results['target_results']:
            if 'error' in target_result:
                report += f"- {target_result['target']}: ❌ FAILED ({target_result['error']})\n"
            else:
                report += f"""- {target_result['target']}:
  - Average: {target_result['avg_time']:.2f}s
  - Range: {target_result['min_time']:.2f}s - {target_result['max_time']:.2f}s
  - Success Rate: {len([r for r in target_result['iterations'] if r['success']])}/{len(target_result['iterations'])}
"""
        
        # Generate recommendations
        report += "\n## Optimization Recommendations\n"
        
        if results['cache_analysis']['cache_hit_rate'] < 60:
            report += "- 🔧 **Improve caching**: Consider enabling remote cache or increasing cache size\n"
        
        if results['system_info']['memory_gb'] >= 16:
            report += "- 🚀 **High memory system**: Consider increasing action cache limits\n"
        
        if results['system_info']['cpu_count'] >= 8:
            report += "- ⚡ **Multi-core optimization**: Parallel builds should provide significant speedup\n"
        
        report += f"""
## Recommended .buckconfig Settings
Based on your system configuration:

```ini
[build]
threads = {self.system_info['cpu_count']}

[cache] 
disk_cache_size_mb = {min(self.system_info['disk_free_gb'] * 100, 20480)}

[download]
max_downloads = {min(self.system_info['cpu_count'] * 2, 16)}

[action_cache]
max_entries = {200000 if self.system_info['memory_gb'] >= 16 else 100000}
cleanup_threshold = 0.8
```

## Next Steps
1. Apply recommended .buckconfig settings
2. Enable remote caching if working in a team
3. Consider using Buck2 daemon for faster subsequent builds  
4. Monitor cache hit rates and adjust cache sizes as needed
"""
        
        return report

def main():
    parser = argparse.ArgumentParser(
        description="Buck2 Performance Tuning and Optimization"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true", 
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply performance optimizations to .buckconfig"
    )
    parser.add_argument(
        "--report",
        help="Generate performance report to file"
    )
    parser.add_argument(
        "--target",
        default="//:pynomaly",
        help="Target to benchmark (default: //:pynomaly)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    tuner = Buck2PerformanceTuner(verbose=args.verbose)
    
    print("🔧 Buck2 Performance Tuner")
    print(f"System: {tuner.system_info['cpu_count']} cores, {tuner.system_info['memory_gb']} GB RAM")
    
    if args.benchmark:
        print("\n🚀 Running performance benchmarks...")
        results = tuner.benchmark_suite()
        
        if args.report:
            report = tuner.generate_report(results)
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"📊 Report saved to {args.report}")
        
        # Print summary
        successful_targets = [r for r in results['target_results'] if 'avg_time' in r]
        if successful_targets:
            avg_build_time = sum(r['avg_time'] for r in successful_targets) / len(successful_targets)
            print(f"📈 Average build time: {avg_build_time:.2f}s")
        
        print(f"💾 Cache efficiency: {results['cache_analysis']['cache_efficiency']}")
        
    if args.optimize:
        print("\n⚡ Generating optimized configuration...")
        config = tuner.generate_optimized_config()
        tuner.apply_optimizations(config)
        print("✅ Optimizations applied! Restart Buck2 daemon for changes to take effect.")
    
    if not args.benchmark and not args.optimize:
        print("\n💡 Use --benchmark to test performance or --optimize to apply optimizations")
        print("Example: python buck2_performance_tuner.py --benchmark --optimize --report performance_report.md")

if __name__ == "__main__":
    main()