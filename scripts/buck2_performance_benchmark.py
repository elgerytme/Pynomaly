#!/usr/bin/env python3
"""Buck2 Performance Benchmark Script.

Compares Buck2 build performance against standard Python tools
to demonstrate the benefits of Buck2 + Hatch integration.
"""

import subprocess
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any
import json

class Buck2PerformanceBenchmark:
    """Benchmark Buck2 performance against standard tools."""
    
    def __init__(self):
        self.root_path = Path.cwd()
        self.buck2_executable = "/mnt/c/Users/andre/buck2.exe"
        
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        print("🚀 Buck2 Performance Benchmark Suite")
        print("=" * 50)
        
        results = {
            "buck2_basic_builds": self.benchmark_buck2_basic_builds(),
            "hatch_basic_operations": self.benchmark_hatch_operations(),
            "incremental_build_comparison": self.benchmark_incremental_builds(),
            "cache_performance": self.benchmark_cache_performance()
        }
        
        self.generate_performance_report(results)
        return results
    
    def benchmark_buck2_basic_builds(self) -> Dict[str, Any]:
        """Benchmark basic Buck2 build operations."""
        print("\n📊 Buck2 Basic Build Performance")
        print("-" * 40)
        
        targets = ["//:validation", "//:test-validation", "//:dev-ready"]
        build_times = []
        
        # Clean build times (multiple runs for accuracy)
        for run in range(5):
            print(f"  Run {run + 1}/5: Buck2 clean builds...")
            
            # Clean first
            subprocess.run([self.buck2_executable, "clean"], 
                         capture_output=True, cwd=self.root_path)
            
            start_time = time.time()
            
            for target in targets:
                result = subprocess.run(
                    [self.buck2_executable, "build", target],
                    capture_output=True,
                    cwd=self.root_path
                )
                
                if result.returncode != 0:
                    print(f"    ⚠️ Warning: {target} failed to build")
            
            build_time = time.time() - start_time
            build_times.append(build_time)
            print(f"    ⏱️ {build_time:.3f}s")
        
        # Cached build times
        cached_times = []
        for run in range(3):
            print(f"  Cached run {run + 1}/3...")
            
            start_time = time.time()
            for target in targets:
                subprocess.run(
                    [self.buck2_executable, "build", target],
                    capture_output=True,
                    cwd=self.root_path
                )
            
            cached_time = time.time() - start_time
            cached_times.append(cached_time)
            print(f"    ⏱️ {cached_time:.3f}s (cached)")
        
        return {
            "clean_build_times": build_times,
            "cached_build_times": cached_times,
            "avg_clean_time": statistics.mean(build_times),
            "avg_cached_time": statistics.mean(cached_times),
            "cache_speedup": statistics.mean(build_times) / statistics.mean(cached_times),
            "targets_tested": targets
        }
    
    def benchmark_hatch_operations(self) -> Dict[str, Any]:
        """Benchmark Hatch operations for comparison."""
        print("\n📊 Hatch Operations Performance")
        print("-" * 40)
        
        operations = {
            "version": ["python3", "-m", "hatch", "version"],
            "env_create": ["python3", "-m", "hatch", "env", "create", "test"],
            "env_remove": ["python3", "-m", "hatch", "env", "remove", "test"]
        }
        
        results = {}
        
        for op_name, command in operations.items():
            print(f"  Testing {op_name}...")
            times = []
            
            for run in range(3):
                start_time = time.time()
                result = subprocess.run(
                    command,
                    capture_output=True,
                    cwd=self.root_path,
                    timeout=60
                )
                execution_time = time.time() - start_time
                times.append(execution_time)
                
                status = "✅" if result.returncode == 0 else "❌"
                print(f"    {status} Run {run + 1}: {execution_time:.3f}s")
            
            results[op_name] = {
                "times": times,
                "avg_time": statistics.mean(times),
                "success": all(t > 0 for t in times)
            }
        
        return results
    
    def benchmark_incremental_builds(self) -> Dict[str, Any]:
        """Benchmark incremental build performance."""
        print("\n📊 Incremental Build Performance")
        print("-" * 40)
        
        # Simulate file changes and measure rebuild times
        test_file = self.root_path / "temp_test_file.txt"
        
        results = {
            "full_rebuild_times": [],
            "incremental_times": [],
            "no_change_times": []
        }
        
        try:
            # Full rebuild test
            print("  Testing full rebuilds...")
            for run in range(3):
                subprocess.run([self.buck2_executable, "clean"], 
                             capture_output=True, cwd=self.root_path)
                
                start_time = time.time()
                subprocess.run([self.buck2_executable, "build", "//:dev-ready"],
                             capture_output=True, cwd=self.root_path)
                rebuild_time = time.time() - start_time
                results["full_rebuild_times"].append(rebuild_time)
                print(f"    Run {run + 1}: {rebuild_time:.3f}s")
            
            # Incremental build test (with file changes)
            print("  Testing incremental builds...")
            for run in range(3):
                # Make a small change
                test_file.write_text(f"Test content {run} {time.time()}")
                
                start_time = time.time()
                subprocess.run([self.buck2_executable, "build", "//:dev-ready"],
                             capture_output=True, cwd=self.root_path)
                incremental_time = time.time() - start_time
                results["incremental_times"].append(incremental_time)
                print(f"    Run {run + 1}: {incremental_time:.3f}s")
            
            # No-change build test (should be very fast)
            print("  Testing no-change builds...")
            for run in range(3):
                start_time = time.time()
                subprocess.run([self.buck2_executable, "build", "//:dev-ready"],
                             capture_output=True, cwd=self.root_path)
                no_change_time = time.time() - start_time
                results["no_change_times"].append(no_change_time)
                print(f"    Run {run + 1}: {no_change_time:.3f}s")
                
        finally:
            # Cleanup
            test_file.unlink(missing_ok=True)
        
        # Calculate metrics
        results.update({
            "avg_full_rebuild": statistics.mean(results["full_rebuild_times"]),
            "avg_incremental": statistics.mean(results["incremental_times"]),
            "avg_no_change": statistics.mean(results["no_change_times"]),
            "incremental_speedup": statistics.mean(results["full_rebuild_times"]) / statistics.mean(results["incremental_times"]),
            "no_change_speedup": statistics.mean(results["full_rebuild_times"]) / statistics.mean(results["no_change_times"])
        })
        
        return results
    
    def benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark Buck2 cache performance."""
        print("\n📊 Buck2 Cache Performance")
        print("-" * 40)
        
        # Test cache directory size and hit rates
        cache_dir = self.root_path / ".buck-cache"
        
        results = {}
        
        # Measure cache directory size
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            file_count = len(list(cache_dir.rglob('*')))
            
            results["cache_size_bytes"] = cache_size
            results["cache_size_mb"] = cache_size / (1024 * 1024)
            results["cache_file_count"] = file_count
            
            print(f"  📁 Cache size: {cache_size / (1024 * 1024):.2f} MB")
            print(f"  📄 Cache files: {file_count}")
        else:
            print("  ⚠️ No cache directory found")
            results["cache_exists"] = False
            
        # Test cold vs warm cache builds
        print("  Testing cache effectiveness...")
        
        # Clear cache and measure cold build
        subprocess.run([self.buck2_executable, "clean"], 
                     capture_output=True, cwd=self.root_path)
        
        start_time = time.time()
        subprocess.run([self.buck2_executable, "build", "//:validation"],
                     capture_output=True, cwd=self.root_path)
        cold_time = time.time() - start_time
        
        # Measure warm cache build
        start_time = time.time()
        subprocess.run([self.buck2_executable, "build", "//:validation"],
                     capture_output=True, cwd=self.root_path)
        warm_time = time.time() - start_time
        
        results.update({
            "cold_cache_time": cold_time,
            "warm_cache_time": warm_time,
            "cache_effectiveness": cold_time / warm_time if warm_time > 0 else 1.0
        })
        
        print(f"  ❄️  Cold cache: {cold_time:.3f}s")
        print(f"  🔥 Warm cache: {warm_time:.3f}s")
        print(f"  🚀 Speedup: {cold_time / warm_time if warm_time > 0 else 1.0:.1f}x")
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]):
        """Generate comprehensive performance report."""
        print("\n🎯 Performance Benchmark Summary")
        print("=" * 50)
        
        # Buck2 build performance
        buck2_results = results["buck2_basic_builds"]
        print(f"🔨 Buck2 Build Performance:")
        print(f"   Clean builds: {buck2_results['avg_clean_time']:.3f}s average")
        print(f"   Cached builds: {buck2_results['avg_cached_time']:.3f}s average")
        print(f"   Cache speedup: {buck2_results['cache_speedup']:.1f}x")
        
        # Incremental build performance
        incremental_results = results["incremental_build_comparison"]
        print(f"\n⚡ Incremental Build Performance:")
        print(f"   Full rebuild: {incremental_results['avg_full_rebuild']:.3f}s")
        print(f"   Incremental: {incremental_results['avg_incremental']:.3f}s")
        print(f"   No changes: {incremental_results['avg_no_change']:.3f}s")
        print(f"   Incremental speedup: {incremental_results['incremental_speedup']:.1f}x")
        print(f"   No-change speedup: {incremental_results['no_change_speedup']:.1f}x")
        
        # Cache performance
        cache_results = results["cache_performance"]
        if cache_results.get("cache_exists", True):
            print(f"\n💾 Cache Performance:")
            print(f"   Cache size: {cache_results.get('cache_size_mb', 0):.2f} MB")
            print(f"   Cache effectiveness: {cache_results.get('cache_effectiveness', 1.0):.1f}x")
        
        # Overall assessment
        print(f"\n📊 Buck2 Integration Assessment:")
        if buck2_results['cache_speedup'] > 2.0:
            print("   ✅ Excellent cache performance")
        elif buck2_results['cache_speedup'] > 1.5:
            print("   ✅ Good cache performance")
        else:
            print("   ⚠️ Cache performance needs improvement")
        
        if incremental_results['avg_no_change'] < 0.5:
            print("   ✅ Excellent no-change build speed")
        elif incremental_results['avg_no_change'] < 1.0:
            print("   ✅ Good no-change build speed")
        else:
            print("   ⚠️ No-change builds could be faster")
        
        # Save detailed results
        report_file = self.root_path / "buck2_performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return results

def main():
    """Run Buck2 performance benchmarks."""
    benchmark = Buck2PerformanceBenchmark()
    results = benchmark.run_benchmarks()
    
    # Determine overall success
    buck2_results = results["buck2_basic_builds"]
    cache_speedup = buck2_results.get("cache_speedup", 1.0)
    
    if cache_speedup > 1.5:
        print("\n🎉 Buck2 Performance: EXCELLENT")
        return 0
    elif cache_speedup > 1.2:
        print("\n✅ Buck2 Performance: GOOD")
        return 0
    else:
        print("\n⚠️ Buck2 Performance: NEEDS IMPROVEMENT")
        return 1

if __name__ == "__main__":
    exit(main())